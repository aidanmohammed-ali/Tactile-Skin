from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .aruco import ArucoPaperCalibrator, PaperCalibration, load_calibration
from .camera import CameraFrame, ThreadedCamera, parse_camera_source
from .geometry import BLOCK_SIDE_CM, BLOCK_SIDE_TAXEL, SENSOR_COORD_FRAME, SheetConfig
from .labels import (
    CapturedSample,
    ManualAnnotation,
    annotation_from_label,
    build_manual_tactile_pose_label,
    build_no_block_tactile_pose_label,
    create_manual_annotation_at_center,
    create_manual_annotation_from_taxel_pose,
    label_tactile_values,
    tactile_values_for_training,
    validate_manual_sample,
)
from .overlay import UiStatus, draw_overlay, draw_tactile_port_selector, source_input_rect
from .sheet import add_geometry_args, build_config_from_args
from .tactile import TactileSnapshot, ThreadedTactileReader, available_tactile_ports
from .writer import DatasetStore, list_recent_runs


TaskKind = Literal["calibrate"]
NO_KEY = 255
DEFAULT_SOURCE_STORE = Path("block_position_prediction/data_collection_manual/assets/camera_source.txt")
DEFAULT_CALIBRATION = Path("block_position_prediction/data_collection_manual/assets/paper_calibration.json")
DEFAULT_OUTPUT = Path("block_position_prediction/data_collection_manual/runs")
NUDGE_STEP_TAXEL = 0.1
ROTATE_STEP_RAD = math.radians(1.0)


@dataclass(frozen=True)
class VisionTask:
    kind: TaskKind
    frame: CameraFrame
    save_calibration: bool = False


@dataclass(frozen=True)
class VisionSnapshot:
    calibration: PaperCalibration | None
    status: str
    busy: bool


class SourceEditor:
    def __init__(self, text: str) -> None:
        self.text = str(text)
        self.editing = False
        self._before_edit = self.text

    def begin(self) -> None:
        if self.editing:
            return
        self._before_edit = self.text
        self.editing = True

    def cancel(self) -> None:
        self.text = self._before_edit
        self.editing = False

    def finish(self) -> str:
        self.text = self.text.strip()
        self.editing = False
        return self.text

    def handle_key(self, key: int) -> str | None:
        if not self.editing or key == NO_KEY:
            return None
        if key in (10, 13):
            return "commit"
        if key == 27:
            self.cancel()
            return "cancel"
        if key in (8, 127):
            self.text = self.text[:-1]
            return None
        if 32 <= key <= 126:
            self.text += chr(key)
        return None


class VisionWorker:
    """Single background worker for ArUco calibration."""

    def __init__(self, sheet_config: SheetConfig, calibration_path: str | Path) -> None:
        self.sheet_config = sheet_config
        self.calibration_path = Path(calibration_path)
        self.calibrator = ArucoPaperCalibrator(sheet_config)
        self._queue: queue.Queue[VisionTask | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._calibration: PaperCalibration | None = None
        self._status = "idle"
        self._busy = False
        self._load_existing_calibration()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="manual_collection_calibration", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
        self._thread = None

    def request_calibration(self, frame: CameraFrame, save: bool = True) -> None:
        self._replace_task(VisionTask("calibrate", frame, save_calibration=save))

    def request_auto_calibration(self, frame: CameraFrame) -> bool:
        return self._enqueue_if_idle(VisionTask("calibrate", frame, save_calibration=False))

    def snapshot(self) -> VisionSnapshot:
        with self._lock:
            return VisionSnapshot(calibration=self._calibration, status=self._status, busy=self._busy)

    def _load_existing_calibration(self) -> None:
        if not self.calibration_path.exists():
            return
        try:
            calibration = load_calibration(self.calibration_path)
            with self._lock:
                self._calibration = calibration
                self._status = f"loaded {self.calibration_path}"
        except Exception as exc:
            with self._lock:
                self._status = f"could not load calibration: {exc}"

    def _replace_task(self, task: VisionTask) -> None:
        while True:
            try:
                self._queue.put_nowait(task)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return

    def _enqueue_if_idle(self, task: VisionTask) -> bool:
        with self._lock:
            if self._busy:
                return False
        if not self._queue.empty():
            return False
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            task = self._queue.get()
            if task is None:
                break
            with self._lock:
                self._busy = True
                self._status = task.kind
            try:
                self._run_calibration(task.frame, save=task.save_calibration)
            finally:
                with self._lock:
                    self._busy = False
                self._queue.task_done()

    def _run_calibration(self, frame: CameraFrame, save: bool) -> None:
        started = time.perf_counter()
        try:
            calibration = self.calibrator.calibrate(frame.image)
            if save:
                calibration.save_json(self.calibration_path)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            median = calibration.quality.median_paper_error_mm
            median_text = "nan" if median is None else f"{median:.3f}mm"
            prefix = "calibrated saved" if save else "auto calibrated"
            status = f"{prefix} markers={len(calibration.marker_ids)} median={median_text} {elapsed_ms:.0f}ms"
            with self._lock:
                self._calibration = calibration
                self._status = status
        except Exception as exc:
            with self._lock:
                self._status = f"calibration failed: {exc}"


class DataCollectionApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sheet_config = build_config_from_args(args)
        self.source_store = Path(args.source_store)
        initial_source = load_camera_source(args.source, self.source_store)
        self.source_editor = SourceEditor(initial_source)
        self.connected_source_text = initial_source
        self.source_input_rect = (0, 0, 0, 0)
        self.view_image_width = 0
        self.tactile_ports = tuple(available_tactile_ports())
        self.tactile_dropdown_open = False
        self.tactile_port_rects: list[tuple[str, tuple[int, int, int, int]]] = []
        self.camera = ThreadedCamera(
            parse_camera_source(initial_source),
            frame_width=args.frame_width if args.frame_width > 0 else None,
            frame_height=args.frame_height if args.frame_height > 0 else None,
            low_latency=not args.no_low_latency,
        )
        self._camera_connect_lock = threading.Lock()
        self._camera_connect_thread: threading.Thread | None = None
        self._camera_connecting = False
        self.worker = VisionWorker(self.sheet_config, args.calibration)
        self.tactile = ThreadedTactileReader(args.tactile_port, args.tactile_baud)
        self.dataset: DatasetStore | None = None
        self.window_name = args.window
        self.auto_calibration_enabled = not args.no_auto_calibrate
        self.last_auto_calibrate_at = 0.0
        self.mode: Literal["live", "draft", "saved"] = "live"
        self.draft: CapturedSample | None = None
        self.draft_annotation: ManualAnnotation | None = None
        self.saved_index: int | None = None
        self.saved_annotation_override: ManualAnnotation | None = None
        self._saved_image_cache: tuple[int, Any] | None = None
        self._dragging_annotation = False
        self.delete_button_rect = (0, 0, 0, 0)
        self.user_status = ""

    def run(self) -> None:
        import cv2

        self.dataset = self._select_dataset(cv2)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        self.worker.start()
        self.tactile.start()
        self._start_camera_async(self.connected_source_text, reconnect=False)
        try:
            while True:
                frame = self.camera.latest_frame()
                snapshot = self.worker.snapshot()
                if frame is not None:
                    self._maybe_submit_auto_calibration(frame, snapshot)
                display_frame, calibration, annotation, tactile_snapshot, tactile_values = self._current_view(frame, snapshot)
                display = draw_overlay(
                    display_frame,
                    self.sheet_config,
                    calibration,
                    annotation,
                    self._ui_status(snapshot, annotation),
                    tactile=tactile_snapshot,
                    tactile_values=tactile_values,
                )
                self.view_image_width = display_frame.shape[1]
                self._draw_tactile_selector(display, self.view_image_width, self.tactile.port)
                self._draw_delete_button(display)
                self.source_input_rect = source_input_rect(self.view_image_width)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_source_key(key):
                    continue
                if key == 27:
                    if self.mode == "live":
                        break
                    self._return_to_live("returned to live")
                    continue
                if key in (ord("u"), ord("U")):
                    self.source_editor.begin()
                elif key in (ord("c"), ord("C")) and frame is not None:
                    self.worker.request_calibration(frame, save=True)
                    self.user_status = "calibration requested"
                elif key in (ord("z"), ord("Z")):
                    self._go_previous()
                elif key in (ord("x"), ord("X")):
                    self._go_next()
                elif key in (ord("r"), ord("R")):
                    self._return_to_live("returned to live")
                elif key in (ord("w"), ord("W")):
                    self._nudge_active_annotation(0.0, -NUDGE_STEP_TAXEL)
                elif key in (ord("a"), ord("A")):
                    self._nudge_active_annotation(-NUDGE_STEP_TAXEL, 0.0)
                elif key in (ord("s"), ord("S")):
                    self._nudge_active_annotation(0.0, NUDGE_STEP_TAXEL)
                elif key in (ord("d"), ord("D")):
                    self._nudge_active_annotation(NUDGE_STEP_TAXEL, 0.0)
                elif key in (ord("q"), ord("Q")):
                    self._rotate_active_annotation(-ROTATE_STEP_RAD)
                elif key in (ord("e"), ord("E")):
                    self._rotate_active_annotation(ROTATE_STEP_RAD)
                elif key == ord(" "):
                    self._capture_draft(frame, snapshot)
                elif key in (ord("b"), ord("B")):
                    self._save_no_block(frame, snapshot)
                elif key in (ord("f"), ord("F")):
                    self._save_active()
                elif key in (8, 127):
                    self._delete_saved_sample()
                elif key in (ord("t"), ord("T")):
                    self.tactile.tare()
                    self.user_status = "tactile tared"
        finally:
            self.tactile.stop()
            self.worker.stop()
            self.camera.stop()
            self._join_camera_connect_thread()
            cv2.destroyWindow(self.window_name)

    def _select_dataset(self, cv2: Any) -> DatasetStore:
        if self.args.dataset:
            return DatasetStore.resume(self.args.dataset)
        if self.args.new_run:
            return DatasetStore.create_new(self.args.output)
        return choose_dataset_interactively(self.args.output, self.window_name + " - Dataset")

    def _current_view(
        self,
        frame: CameraFrame | None,
        snapshot: VisionSnapshot,
    ) -> tuple[Any, PaperCalibration | None, ManualAnnotation | None, TactileSnapshot | None, Any | None]:
        import numpy as np

        if self.mode == "draft" and self.draft is not None:
            return self.draft.image, self.draft.calibration, self.draft_annotation, self.draft.tactile, None
        if self.mode == "saved" and self.saved_index is not None and self.dataset is not None and self.dataset.labels:
            label = self.dataset.labels[self.saved_index]
            image = self._saved_image(self.saved_index)
            annotation = self.saved_annotation_override or annotation_from_label(label)
            return image, self._saved_label_calibration(label), annotation, None, label_tactile_values(label)
        if frame is not None:
            return frame.image, snapshot.calibration, None, self.tactile.snapshot(), None
        blank = np.zeros((360, 960, 3), dtype=np.uint8)
        return blank, snapshot.calibration, None, self.tactile.snapshot(), None

    def _ui_status(self, snapshot: VisionSnapshot, annotation: ManualAnnotation | None) -> UiStatus:
        dataset = "" if self.dataset is None else str(self.dataset.session_dir)
        return UiStatus(
            camera=self.camera.status.message,
            source_text=self.source_editor.text,
            source_editing=self.source_editor.editing,
            calibration=snapshot.status,
            annotation=self._annotation_status(annotation),
            writer=self._writer_status(),
            mode=self.mode,
            auto_calibration=self.auto_calibration_enabled,
            dataset=dataset,
            sample_position=self._sample_position_text(),
            tactile_ports=self.tactile_ports,
            tactile_dropdown_open=self.tactile_dropdown_open,
        )

    def _sample_position_text(self) -> str:
        total = 0 if self.dataset is None else len(self.dataset.labels)
        if self.mode == "live":
            return f"{total} saved + LIVE"
        if self.mode == "draft":
            return "new draft"
        if self.saved_index is None:
            return f"0/{total}"
        return f"{self.saved_index + 1}/{total}"

    def _handle_source_key(self, key: int) -> bool:
        action = self.source_editor.handle_key(key)
        if action is None:
            return self.source_editor.editing and key != NO_KEY
        if action == "commit":
            self._connect_source(self.source_editor.finish())
        return True

    def _connect_source(self, source_text: str) -> bool:
        if not source_text:
            self.user_status = "source skipped: empty"
            return False
        self.source_editor.text = source_text
        self._start_camera_async(source_text, reconnect=True)
        return True

    def _start_camera_async(self, source_text: str, reconnect: bool) -> bool:
        with self._camera_connect_lock:
            if self._camera_connecting:
                self.user_status = "camera connect already in progress"
                return False
            self._camera_connecting = True
        self.user_status = f"connecting camera: {source_text}"
        self._camera_connect_thread = threading.Thread(
            target=self._camera_connect_worker,
            args=(source_text, reconnect),
            name="manual_collection_camera_connect",
            daemon=True,
        )
        self._camera_connect_thread.start()
        return True

    def _camera_connect_worker(self, source_text: str, reconnect: bool) -> None:
        try:
            if reconnect:
                self.camera.reconnect(
                    parse_camera_source(source_text),
                    frame_width=self.args.frame_width if self.args.frame_width > 0 else None,
                    frame_height=self.args.frame_height if self.args.frame_height > 0 else None,
                    low_latency=not self.args.no_low_latency,
                )
            else:
                self.camera.start()
        except Exception as exc:
            self.user_status = f"source failed: {exc}"
        else:
            self.connected_source_text = source_text
            self.source_editor.text = source_text
            save_camera_source(source_text, self.source_store)
            self.user_status = f"source connected and saved: {source_text}"
        finally:
            with self._camera_connect_lock:
                self._camera_connecting = False

    def _join_camera_connect_thread(self) -> None:
        thread = self._camera_connect_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: Any) -> None:
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._handle_delete_button_click(x, y):
                return
            if self._handle_tactile_port_click(x, y):
                return
            left, top, right, bottom = self.source_input_rect
            if left <= x <= right and top <= y <= bottom:
                self.source_editor.begin()
                return
            if self.mode in ("draft", "saved") and x < self.view_image_width:
                if self._active_calibration() is None:
                    self.user_status = "edit disabled: saved sample has no calibration snapshot"
                    return
                self._dragging_annotation = True
                self._set_active_annotation_center((float(x), float(y)))
                return
        if self.mode not in ("draft", "saved") or not self._dragging_annotation:
            return
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self._set_active_annotation_center((float(x), float(y)))
        elif event == cv2.EVENT_LBUTTONUP:
            self._set_active_annotation_center((float(x), float(y)))
            self._dragging_annotation = False

    def _set_active_annotation_center(self, center_px: tuple[float, float]) -> None:
        calibration = self._active_calibration()
        if calibration is None:
            self.user_status = "annotation skipped: not calibrated"
            return
        current = self._active_annotation()
        yaw = 0.0 if current is None else current.yaw_mod90_rad
        try:
            annotation = create_manual_annotation_at_center(self.sheet_config, calibration, center_px, yaw)
        except Exception as exc:
            self.user_status = f"annotation failed: {exc}"
            return
        self._store_active_annotation(annotation, "annotation position updated")

    def _nudge_active_annotation(self, dx_taxel: float, dy_taxel: float) -> None:
        current = self._active_annotation()
        calibration = self._active_calibration()
        if current is None:
            if self.mode in ("draft", "saved"):
                self.user_status = "nudge skipped: no annotation"
            return
        if calibration is None:
            if self.mode in ("draft", "saved"):
                self.user_status = "edit disabled: saved sample has no calibration snapshot"
            return
        center = (current.center_taxel[0] + dx_taxel, current.center_taxel[1] + dy_taxel)
        annotation = create_manual_annotation_from_taxel_pose(
            self.sheet_config,
            calibration,
            center,
            current.yaw_mod90_rad,
        )
        self._store_active_annotation(annotation, "annotation nudged")

    def _rotate_active_annotation(self, delta_rad: float) -> None:
        current = self._active_annotation()
        calibration = self._active_calibration()
        if current is None:
            if self.mode in ("draft", "saved"):
                self.user_status = "rotation skipped: no annotation"
            return
        if calibration is None:
            if self.mode in ("draft", "saved"):
                self.user_status = "edit disabled: saved sample has no calibration snapshot"
            return
        annotation = create_manual_annotation_from_taxel_pose(
            self.sheet_config,
            calibration,
            current.center_taxel,
            current.yaw_mod90_rad + delta_rad,
        )
        self._store_active_annotation(annotation, "annotation rotated")

    def _store_active_annotation(self, annotation: ManualAnnotation, status: str) -> None:
        if self.mode == "draft":
            self.draft_annotation = annotation
        elif self.mode == "saved":
            self.saved_annotation_override = annotation
        self.user_status = status

    def _active_annotation(self) -> ManualAnnotation | None:
        if self.mode == "draft":
            return self.draft_annotation
        if self.mode == "saved":
            if self.saved_annotation_override is not None:
                return self.saved_annotation_override
            if self.dataset is not None and self.saved_index is not None and 0 <= self.saved_index < len(self.dataset.labels):
                return annotation_from_label(self.dataset.labels[self.saved_index])
        return None

    def _active_calibration(self) -> PaperCalibration | None:
        if self.mode == "draft" and self.draft is not None:
            return self.draft.calibration
        if self.mode == "saved":
            if self.dataset is not None and self.saved_index is not None and 0 <= self.saved_index < len(self.dataset.labels):
                return self._saved_label_calibration(self.dataset.labels[self.saved_index])
            return None
        return self.worker.snapshot().calibration

    def _draw_tactile_selector(self, display: Any, x_offset: int, selected_port: str) -> None:
        self.tactile_port_rects = draw_tactile_port_selector(
            display,
            x_offset=x_offset,
            selected_port=selected_port,
            ports=self.tactile_ports,
            open_dropdown=self.tactile_dropdown_open,
        )

    def _draw_delete_button(self, display: Any) -> None:
        import cv2

        if self.mode != "saved":
            self.delete_button_rect = (0, 0, 0, 0)
            return
        right = max(120, self.view_image_width - 16)
        left = max(16, right - 112)
        top = 94
        bottom = 124
        self.delete_button_rect = (left, top, right, bottom)
        cv2.rectangle(display, (left, top), (right, bottom), (30, 30, 70), -1)
        cv2.rectangle(display, (left, top), (right, bottom), (80, 80, 255), 1)
        cv2.putText(display, "Delete", (left + 16, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    def _handle_delete_button_click(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.delete_button_rect
        if right <= left or bottom <= top:
            return False
        if left <= x <= right and top <= y <= bottom:
            self._delete_saved_sample()
            return True
        return False

    def _handle_tactile_port_click(self, x: int, y: int) -> bool:
        for port, rect in self.tactile_port_rects:
            left, top, right, bottom = rect
            if not (left <= x <= right and top <= y <= bottom):
                continue
            if port == "__toggle__":
                self.tactile_ports = tuple(available_tactile_ports())
                if self.tactile.port not in self.tactile_ports:
                    self.tactile_ports = (self.tactile.port,) + self.tactile_ports
                self.tactile_dropdown_open = not self.tactile_dropdown_open
                return True
            self.tactile_dropdown_open = False
            self._connect_tactile_port(port)
            return True
        if self.tactile_dropdown_open:
            self.tactile_dropdown_open = False
        return False

    def _connect_tactile_port(self, port: str) -> None:
        previous_port = self.tactile.port
        self.tactile.reconnect(port, self.args.tactile_baud)
        snapshot = self.tactile.snapshot()
        if snapshot.error:
            self.user_status = f"tactile port {port} unavailable: {snapshot.error}"
        else:
            self.user_status = f"tactile port connected: {port}"
        if port not in self.tactile_ports:
            self.tactile_ports = (port,) + self.tactile_ports
        if previous_port and previous_port not in self.tactile_ports:
            self.tactile_ports = self.tactile_ports + (previous_port,)

    def _maybe_submit_auto_calibration(self, frame: CameraFrame, snapshot: VisionSnapshot) -> bool:
        if not self.auto_calibration_enabled or snapshot.busy:
            return False
        now = time.time()
        interval = 1.0 / max(0.1, float(self.args.auto_calibrate_rate_hz))
        if now - self.last_auto_calibrate_at < interval:
            return False
        if self.worker.request_auto_calibration(frame):
            self.last_auto_calibrate_at = now
            return True
        return False

    def _capture_draft(self, frame: CameraFrame | None, snapshot: VisionSnapshot) -> None:
        if frame is None:
            self.user_status = "capture skipped: no camera frame"
            return
        if snapshot.calibration is None:
            self.user_status = "capture skipped: not calibrated"
            return
        self.draft = CapturedSample(
            image=frame.image.copy(),
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            tactile=self.tactile.snapshot(),
            calibration=snapshot.calibration,
        )
        self.draft_annotation = None
        self.saved_index = None
        self.saved_annotation_override = None
        self.mode = "draft"
        self.user_status = "captured draft; click or drag to set block center"

    def _save_active(self) -> None:
        if self.dataset is None:
            self.user_status = "save needs a dataset"
            return
        if self.mode == "draft":
            self._save_draft()
        elif self.mode == "saved":
            self._overwrite_saved()

    def _save_no_block(self, frame: CameraFrame | None, snapshot: VisionSnapshot) -> None:
        if self.dataset is None:
            self.user_status = "blank save needs a dataset"
            return
        if self.mode != "live":
            self.user_status = "blank save is available in live mode"
            return
        tactile_snapshot = self.tactile.snapshot()
        values = tactile_values_for_training(tactile_snapshot)
        if values is None:
            self.user_status = "blank save needs tactile top5 input"
            return
        if frame is None:
            import numpy as np

            image = np.zeros((360, 960, 3), dtype=np.uint8)
            frame_id = 0
            timestamp = time.time()
        else:
            image = frame.image.copy()
            frame_id = frame.frame_id
            timestamp = frame.timestamp
        label = build_no_block_tactile_pose_label(
            self.sheet_config,
            snapshot.calibration,
            tactile_snapshot,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        self.dataset.write_metadata(
            build_session_metadata(
                self.sheet_config,
                snapshot.calibration,
                self.args,
                self.connected_source_text,
                tactile_snapshot,
            )
        )
        index = self.dataset.save_new(image, label)
        self._saved_image_cache = None
        self._return_to_live(f"saved no-block sample {index + 1}/{len(self.dataset.labels)}")

    def _save_draft(self) -> None:
        if self.dataset is None or self.draft is None:
            self.user_status = "save needs a draft"
            return
        error = validate_manual_sample(self.sheet_config, self.draft.calibration, self.draft_annotation, self.draft.tactile)
        if error:
            self.user_status = f"save needs {error}"
            return
        assert self.draft.calibration is not None
        assert self.draft_annotation is not None
        label = build_manual_tactile_pose_label(
            self.sheet_config,
            self.draft.calibration,
            self.draft_annotation,
            self.draft.tactile,
            frame_id=self.draft.frame_id,
            timestamp=self.draft.timestamp,
        )
        self.dataset.write_metadata(
            build_session_metadata(
                self.sheet_config,
                self.draft.calibration,
                self.args,
                self.connected_source_text,
                self.draft.tactile,
            )
        )
        index = self.dataset.save_new(self.draft.image, label)
        self._saved_image_cache = None
        self._return_to_live(f"saved sample {index + 1}/{len(self.dataset.labels)}")

    def _overwrite_saved(self) -> None:
        if self.dataset is None or self.saved_index is None:
            self.user_status = "save needs a saved sample"
            return
        old_label = self.dataset.labels[self.saved_index]
        annotation = self.saved_annotation_override or annotation_from_label(old_label)
        calibration = self._active_calibration()
        values = label_tactile_values(old_label)
        error = validate_manual_sample(self.sheet_config, calibration, annotation, tactile=None, tactile_values=values)
        if error:
            self.user_status = f"save needs {error}"
            return
        assert calibration is not None
        assert annotation is not None
        label = build_manual_tactile_pose_label(
            self.sheet_config,
            calibration,
            annotation,
            tactile_values=values,
            frame_id=int(old_label.get("frame_id", 0)),
            timestamp=float(old_label.get("timestamp", time.time())),
            sample_id=str(old_label.get("sample_id") or ""),
        )
        self.dataset.update_label(self.saved_index, label)
        self.saved_annotation_override = annotation
        self.user_status = f"updated sample {self.saved_index + 1}/{len(self.dataset.labels)}"

    def _delete_saved_sample(self) -> None:
        if self.mode != "saved" or self.dataset is None or self.saved_index is None:
            return
        deleted_index = self.saved_index
        deleted = self.dataset.delete_sample(deleted_index, delete_image=False)
        self._saved_image_cache = None
        self.saved_annotation_override = None
        total = len(self.dataset.labels)
        if total == 0:
            self._return_to_live(f"deleted sample {deleted.get('sample_id') or deleted_index + 1}")
            return
        self.saved_index = min(deleted_index, total - 1)
        self.mode = "saved"
        self.user_status = f"deleted sample {deleted.get('sample_id') or deleted_index + 1}; showing {self.saved_index + 1}/{total}"

    def _go_previous(self) -> None:
        if self.dataset is None or not self.dataset.labels:
            self.user_status = "no saved samples"
            return
        if self.mode == "live":
            self.saved_index = len(self.dataset.labels) - 1
            self.saved_annotation_override = None
            self.mode = "saved"
            self.user_status = "showing latest saved sample"
            return
        if self.mode == "saved" and self.saved_index is not None and self.saved_index > 0:
            self.saved_index -= 1
            self.saved_annotation_override = None
            self.user_status = f"showing sample {self.saved_index + 1}"
        elif self.mode == "draft":
            self.user_status = "finish or Esc the draft before browsing"
        else:
            self.user_status = "already at first sample"

    def _go_next(self) -> None:
        if self.mode == "saved" and self.dataset is not None and self.saved_index is not None:
            if self.saved_index < len(self.dataset.labels) - 1:
                self.saved_index += 1
                self.saved_annotation_override = None
                self.user_status = f"showing sample {self.saved_index + 1}"
            else:
                self._return_to_live("returned to live")
        elif self.mode == "draft":
            self.user_status = "finish or Esc the draft before browsing"

    def _return_to_live(self, status: str) -> None:
        self.mode = "live"
        self.draft = None
        self.draft_annotation = None
        self.saved_index = None
        self.saved_annotation_override = None
        self._dragging_annotation = False
        self.user_status = status

    def _saved_image(self, index: int) -> Any:
        if self.dataset is None:
            raise RuntimeError("dataset is not selected")
        if self._saved_image_cache is not None and self._saved_image_cache[0] == index:
            return self._saved_image_cache[1].copy()
        image = self.dataset.load_image(index)
        self._saved_image_cache = (index, image.copy())
        return image

    def _dataset_calibration(self) -> PaperCalibration | None:
        if self.dataset is None:
            return None
        data = self.dataset.metadata.get("calibration")
        if not data:
            return None
        try:
            return PaperCalibration.from_dict(data)
        except Exception:
            return None

    def _saved_label_calibration(self, label: dict[str, Any]) -> PaperCalibration | None:
        data = label.get("calibration")
        if not data:
            return None
        try:
            return PaperCalibration.from_dict(data)
        except Exception:
            return None

    def _annotation_status(self, annotation: ManualAnnotation | None) -> str:
        if self.mode == "saved" and self.saved_index is not None and self.dataset is not None:
            label = self.dataset.labels[self.saved_index]
            target = label.get("target") or {}
            if target.get("object_present") is False:
                return "no-block sample"
            if self._saved_label_calibration(label) is None:
                return "view only: sample has no calibration snapshot"
        if annotation is None and self.mode != "live":
            return "click or drag to set center"
        if self.mode == "live":
            return "live preview"
        return "ready"

    def _writer_status(self) -> str:
        parts = []
        if self.user_status:
            parts.append(self.user_status)
        if self.dataset is not None and self.dataset.last_saved:
            parts.append(f"saved {self.dataset.last_saved}")
        if self.dataset is not None and self.dataset.last_error:
            parts.append(f"writer error: {self.dataset.last_error}")
        return " | ".join(parts)


def choose_dataset_interactively(root: str | Path, window_name: str) -> DatasetStore:
    import cv2
    import numpy as np

    root = Path(root)
    selected = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            runs = list_recent_runs(root)
            selected = max(0, min(selected, max(0, len(runs) - 1)))
            canvas = np.zeros((420, 900, 3), dtype=np.uint8)
            canvas[:] = (18, 18, 18)
            cv2.putText(canvas, "Select Dataset", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Enter continue | N new dataset | Up/Down select | Q quit", (32, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
            if not runs:
                cv2.putText(canvas, "No existing runs. Press Enter or N to create one.", (32, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
            for index, run_dir in enumerate(runs[:10]):
                y = 136 + index * 28
                color = (0, 230, 255) if index == selected else (235, 235, 235)
                prefix = "> " if index == selected else "  "
                cv2.putText(canvas, prefix + str(run_dir), (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                raise SystemExit("dataset selection cancelled")
            if key in (ord("n"), ord("N")):
                return DatasetStore.create_new(root)
            if key in (10, 13):
                return DatasetStore.resume(runs[selected]) if runs else DatasetStore.create_new(root)
            if key == 82:
                selected = max(0, selected - 1)
            elif key == 84:
                selected = min(max(0, len(runs) - 1), selected + 1)
    finally:
        cv2.destroyWindow(window_name)


def build_session_metadata(
    config: SheetConfig,
    calibration: PaperCalibration | None,
    args: argparse.Namespace,
    camera_source: str,
    tactile: TactileSnapshot | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "manual_collection_v1",
        "created_at": time.time(),
        "app_mode": "manual_aruco",
        "sensor_coordinate_frame": SENSOR_COORD_FRAME,
        "sensor_geometry": config.to_dict(),
        "block": {
            "side_cm": BLOCK_SIDE_CM,
            "side_taxel": BLOCK_SIDE_TAXEL,
        },
        "calibration": None if calibration is None else calibration.to_dict(),
        "camera": {
            "source": str(camera_source),
            "frame_width": int(args.frame_width),
            "frame_height": int(args.frame_height),
        },
        "tactile": {
            "port": None if tactile is None else tactile.port,
            "baud": int(args.tactile_baud),
            "hardware_online": False if tactile is None else tactile.hardware_online,
        },
    }


def sensor_payload_from_tactile(tactile: TactileSnapshot | None) -> dict[str, Any]:
    if tactile is None:
        return {
            "available": False,
            "timestamp": None,
            "port": None,
            "hardware_online": False,
            "status": "not connected",
            "error": "no tactile snapshot",
            "rows": 8,
            "cols": 16,
            "tared": False,
            "top5_normalized": None,
            "top5_raw_average": None,
            "recent_raw_frames": [],
        }
    return tactile.to_sensor_payload()


def load_camera_source(default_source: str, source_store: str | Path = DEFAULT_SOURCE_STORE) -> str:
    path = Path(source_store)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return str(default_source)
    if not text:
        return str(default_source)
    return text


def save_camera_source(source: str, source_store: str | Path = DEFAULT_SOURCE_STORE) -> Path:
    path = Path(source_store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(source).strip() + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually label block poses from webcam + tactile sensor data.")
    parser.add_argument("--source", default="0", help="cv2.VideoCapture source: camera index, RTSP, HTTP, etc.")
    parser.add_argument(
        "--source-store",
        default=str(DEFAULT_SOURCE_STORE),
        help="Path used to remember the last successfully connected camera source.",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION),
        help="Calibration JSON path to load/save.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Dataset output root.")
    parser.add_argument("--dataset", default=None, help="Existing run directory or labels.jsonl to continue.")
    parser.add_argument("--new-run", action="store_true", help="Create a new run without showing the dataset chooser.")
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--no-low-latency", action="store_true")
    parser.add_argument("--window", default="Tactile Manual Data Collection")
    parser.add_argument("--tactile-port", default="SIMULATOR", help="Serial port for tactile sensor, or SIMULATOR.")
    parser.add_argument("--tactile-baud", type=int, default=115200, help="Tactile serial baud rate.")
    parser.add_argument("--auto-calibrate-rate-hz", type=float, default=5.0)
    parser.add_argument("--no-auto-calibrate", action="store_true", help="Disable automatic in-memory ArUco calibration.")
    add_geometry_args(parser)
    return parser


def main() -> None:
    DataCollectionApp(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
