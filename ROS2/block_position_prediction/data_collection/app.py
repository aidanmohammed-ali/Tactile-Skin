from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .aruco import ArucoPaperCalibrator, PaperCalibration, load_calibration
from .camera import CameraFrame, ThreadedCamera, parse_camera_source
from .detector import BlockDetector, DetectionResult, default_weights_path, parse_device
from .geometry import BLOCK_SIDE_CM, BLOCK_SIDE_TAXEL, SENSOR_COORD_FRAME, SheetConfig
from .labels import SCHEMA_VERSION, build_tactile_pose_label
from .overlay import UiStatus, draw_overlay, draw_tactile_port_selector, source_input_rect
from .sheet import add_geometry_args, build_config_from_args
from .tactile import TactileSnapshot, ThreadedTactileReader, available_tactile_ports
from .writer import DatasetWriter


TaskKind = Literal["detect", "calibrate"]
NO_KEY = 255
DEFAULT_SOURCE_STORE = Path("block_position_prediction/data_collection/assets/camera_source.txt")


@dataclass(frozen=True)
class VisionTask:
    kind: TaskKind
    frame: CameraFrame
    save_calibration: bool = False


@dataclass(frozen=True)
class VisionSnapshot:
    calibration: PaperCalibration | None
    detection: DetectionResult | None
    detection_frame: CameraFrame | None
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
    """Single background worker for expensive calibration and YOLO inference."""

    def __init__(
        self,
        sheet_config: SheetConfig,
        calibration_path: str | Path,
        detector: BlockDetector,
    ) -> None:
        self.sheet_config = sheet_config
        self.calibration_path = Path(calibration_path)
        self.detector = detector
        self.calibrator = ArucoPaperCalibrator(sheet_config)
        self._queue: queue.Queue[VisionTask | None] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._calibration: PaperCalibration | None = None
        self._detection: DetectionResult | None = None
        self._detection_frame: CameraFrame | None = None
        self._status = "idle"
        self._busy = False
        self._load_existing_calibration()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="data_collection_vision", daemon=True)
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

    def request_detection(self, frame: CameraFrame) -> None:
        self._replace_task(VisionTask("detect", frame))

    def request_calibration(self, frame: CameraFrame, save: bool = True) -> None:
        self._replace_task(VisionTask("calibrate", frame, save_calibration=save))

    def request_auto_calibration(self, frame: CameraFrame) -> bool:
        return self._enqueue_if_idle(VisionTask("calibrate", frame, save_calibration=False))

    def snapshot(self) -> VisionSnapshot:
        with self._lock:
            return VisionSnapshot(
                calibration=self._calibration,
                detection=self._detection,
                detection_frame=None if self._detection_frame is None else self._detection_frame.copy(),
                status=self._status,
                busy=self._busy,
            )

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
                if task.kind == "calibrate":
                    self._run_calibration(task.frame, save=task.save_calibration)
                else:
                    self._run_detection(task.frame)
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

    def _run_detection(self, frame: CameraFrame) -> None:
        result = self.detector.detect(frame.image, frame_id=frame.frame_id, timestamp=frame.timestamp)
        with self._lock:
            self._detection = result
            self._detection_frame = frame.copy()
            if result.error:
                self._status = f"detection failed: {result.error}"
            elif result.best is None:
                self._status = f"no block detected conf>={self.detector.confidence:.2f}"
            else:
                self._status = f"detected {len(result.detections)} block(s)"


class DataCollectionApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sheet_config = build_config_from_args(args)
        self.source_store = Path(args.source_store)
        initial_source = load_camera_source(args.source, self.source_store)
        self.source_editor = SourceEditor(initial_source)
        self.connected_source_text = initial_source
        self.source_input_rect = (0, 0, 0, 0)
        self.tactile_ports = tuple(available_tactile_ports())
        self.tactile_dropdown_open = False
        self.tactile_port_rects: list[tuple[str, tuple[int, int, int, int]]] = []
        self.camera = ThreadedCamera(
            parse_camera_source(initial_source),
            frame_width=args.frame_width if args.frame_width > 0 else None,
            frame_height=args.frame_height if args.frame_height > 0 else None,
            low_latency=not args.no_low_latency,
        )
        self.detector = BlockDetector(
            weights_path=args.weights,
            confidence=args.confidence,
            image_size=args.image_size if args.image_size > 0 else None,
            anchor_mode=args.anchor_mode,
            device=parse_device(args.device),
        )
        self.worker = VisionWorker(self.sheet_config, args.calibration, self.detector)
        self.tactile = ThreadedTactileReader(args.tactile_port, args.tactile_baud)
        self.writer = DatasetWriter(args.output)
        self.window_name = args.window
        self.paused = False
        self.auto_calibration_enabled = not args.no_auto_calibrate
        self.last_submit_at = 0.0
        self.last_auto_calibrate_at = 0.0
        self.last_block_detection: DetectionResult | None = None
        self.user_status = ""

    def run(self) -> None:
        import cv2

        try:
            self.camera.start()
            save_camera_source(self.connected_source_text, self.source_store)
        except Exception as exc:
            self.user_status = f"camera open failed: {exc}"
        self.worker.start()
        self.tactile.start()
        self.writer.start()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                frame = self.camera.latest_frame()
                snapshot = self.worker.snapshot()
                if frame is None:
                    self._show_blank(cv2, snapshot)
                    key = cv2.waitKey(30) & 0xFF
                else:
                    auto_submitted = self._maybe_submit_auto_calibration(frame, snapshot)
                    if not auto_submitted:
                        self._maybe_submit_detection(frame, snapshot)
                    display_detection = self._display_detection(snapshot.detection)
                    tactile_snapshot = self.tactile.snapshot()
                    display = draw_overlay(
                        frame.image,
                        self.sheet_config,
                        snapshot.calibration,
                        display_detection,
                        UiStatus(
                            camera=self.camera.status.message,
                            source_text=self.source_editor.text,
                            source_editing=self.source_editor.editing,
                            calibration=snapshot.status,
                            detection=snapshot.status,
                            writer=self._writer_status(),
                            paused=self.paused,
                            auto_calibration=self.auto_calibration_enabled,
                            tactile_ports=self.tactile_ports,
                            tactile_dropdown_open=self.tactile_dropdown_open,
                            retained_block=self._using_retained_detection(snapshot.detection, display_detection),
                        ),
                        tactile=tactile_snapshot,
                    )
                    self._draw_tactile_selector(display, frame.image.shape[1], tactile_snapshot.port)
                    self.source_input_rect = source_input_rect(frame.image.shape[1])
                    cv2.imshow(self.window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                if self._handle_source_key(key):
                    continue
                if key in (ord("q"), 27):
                    break
                if key == ord("u"):
                    self.source_editor.begin()
                if key == ord("d"):
                    self.paused = not self.paused
                if key == ord("a"):
                    self._toggle_auto_calibration()
                if key in (ord("l"), ord("L")):
                    self._clear_last_block_detection()
                if key == ord("c") and frame is not None:
                    self.worker.request_calibration(frame, save=True)
                    self.user_status = "calibration requested"
                if key == ord("s"):
                    self._capture_sample()
        finally:
            self.writer.stop()
            self.tactile.stop()
            self.worker.stop()
            self.camera.stop()
            cv2.destroyWindow(self.window_name)

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
        previous_source = self.camera.source
        previous_text = self.connected_source_text
        previous_opened = self.camera.status.opened
        try:
            self.camera.reconnect(
                parse_camera_source(source_text),
                frame_width=self.args.frame_width if self.args.frame_width > 0 else None,
                frame_height=self.args.frame_height if self.args.frame_height > 0 else None,
                low_latency=not self.args.no_low_latency,
            )
        except Exception as exc:
            self.user_status = f"source failed: {exc}"
            if previous_opened:
                try:
                    self.camera.reconnect(
                        previous_source,
                        frame_width=self.args.frame_width if self.args.frame_width > 0 else None,
                        frame_height=self.args.frame_height if self.args.frame_height > 0 else None,
                        low_latency=not self.args.no_low_latency,
                    )
                    self.source_editor.text = previous_text
                except Exception as restore_exc:
                    self.user_status = f"source failed: {exc}; restore failed: {restore_exc}"
            return False
        self.connected_source_text = source_text
        self.source_editor.text = source_text
        save_camera_source(source_text, self.source_store)
        self.last_submit_at = 0.0
        self.user_status = f"source connected and saved: {source_text}"
        return True

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        import cv2

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._handle_tactile_port_click(x, y):
            return
        left, top, right, bottom = self.source_input_rect
        if left <= x <= right and top <= y <= bottom:
            self.source_editor.begin()

    def _draw_tactile_selector(self, display: Any, x_offset: int, selected_port: str) -> None:
        self.tactile_port_rects = draw_tactile_port_selector(
            display,
            x_offset=x_offset,
            selected_port=selected_port,
            ports=self.tactile_ports,
            open_dropdown=self.tactile_dropdown_open,
        )

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

    def _maybe_submit_detection(self, frame: CameraFrame, snapshot: VisionSnapshot) -> None:
        if self.paused or snapshot.busy:
            return
        now = time.time()
        interval = 1.0 / max(0.1, float(self.args.detect_rate_hz))
        if now - self.last_submit_at < interval:
            return
        self.last_submit_at = now
        self.worker.request_detection(frame)
        current = snapshot.detection
        if current is not None and current.best is not None:
            self.last_block_detection = current

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

    def _toggle_auto_calibration(self) -> None:
        self.auto_calibration_enabled = not self.auto_calibration_enabled
        if self.auto_calibration_enabled:
            self.user_status = "auto calibration enabled"
            return
        snapshot = self.worker.snapshot()
        if snapshot.calibration is None:
            self.user_status = "auto calibration disabled; no calibration to save"
            return
        try:
            snapshot.calibration.save_json(self.worker.calibration_path)
            self.user_status = f"auto calibration disabled; saved {self.worker.calibration_path}"
        except Exception as exc:
            self.user_status = f"auto calibration disabled; save failed: {exc}"

    def _capture_sample(self) -> None:
        snapshot = self.worker.snapshot()
        if snapshot.calibration is None:
            self.user_status = "capture skipped: not calibrated"
            return
        detection = self._display_detection(snapshot.detection)
        if detection is None or detection.best is None or snapshot.detection_frame is None:
            self.user_status = "capture skipped: no valid detection"
            return
        tactile_snapshot = self.tactile.snapshot()
        label = build_sample_label(
            self.sheet_config,
            snapshot.calibration,
            detection,
            tactile_snapshot,
            retained_block=detection is self.last_block_detection and (snapshot.detection is None or snapshot.detection.best is None),
        )
        self.writer.write_metadata(
            build_session_metadata(
                self.sheet_config,
                snapshot.calibration,
                self.args,
                self.connected_source_text,
                tactile_snapshot,
            )
        )
        if self.writer.enqueue(snapshot.detection_frame.image, label):
            self.user_status = f"capture queued frame={detection.frame_id}"
        else:
            self.user_status = self.writer.last_error or "capture queue failed"

    def _display_detection(self, detection: DetectionResult | None) -> DetectionResult | None:
        if detection is not None and detection.best is not None:
            self.last_block_detection = detection
            return detection
        return self.last_block_detection

    def _using_retained_detection(self, latest: DetectionResult | None, display: DetectionResult | None) -> bool:
        return display is not None and (latest is None or latest.best is None) and display is self.last_block_detection

    def _clear_last_block_detection(self) -> None:
        self.last_block_detection = None
        self.user_status = "last block position cleared"

    def _writer_status(self) -> str:
        parts = []
        if self.user_status:
            parts.append(self.user_status)
        if self.writer.last_saved:
            parts.append(f"saved {self.writer.last_saved}")
        if self.writer.last_error:
            parts.append(f"writer error: {self.writer.last_error}")
        return " | ".join(parts)

    def _show_blank(self, cv2: Any, snapshot: VisionSnapshot) -> None:
        import numpy as np

        blank = np.zeros((360, 960, 3), dtype=np.uint8)
        display = draw_overlay(
            blank,
            self.sheet_config,
            snapshot.calibration,
            self._display_detection(snapshot.detection),
            UiStatus(
                camera=self.camera.status.message,
                source_text=self.source_editor.text,
                source_editing=self.source_editor.editing,
                calibration=snapshot.status,
                detection=snapshot.status,
                writer=self._writer_status(),
                paused=self.paused,
                auto_calibration=self.auto_calibration_enabled,
                tactile_ports=self.tactile_ports,
                tactile_dropdown_open=self.tactile_dropdown_open,
                retained_block=self._using_retained_detection(snapshot.detection, self.last_block_detection),
            ),
            tactile=self.tactile.snapshot(),
        )
        self._draw_tactile_selector(display, blank.shape[1], self.tactile.port)
        self.source_input_rect = source_input_rect(blank.shape[1])
        cv2.putText(
            display,
            f"Waiting for camera: {self.camera.status.message} | {snapshot.status}",
            (24, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, display)


def build_sample_label(
    config: SheetConfig,
    calibration: PaperCalibration,
    detection: DetectionResult,
    tactile: TactileSnapshot | None = None,
    retained_block: bool = False,
) -> dict[str, Any]:
    return build_tactile_pose_label(config, calibration, detection, tactile, retained_block=retained_block)


def build_session_metadata(
    config: SheetConfig,
    calibration: PaperCalibration,
    args: argparse.Namespace,
    camera_source: str,
    tactile: TactileSnapshot | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.time(),
        "sensor_coordinate_frame": SENSOR_COORD_FRAME,
        "sensor_geometry": config.to_dict(),
        "block": {
            "side_cm": BLOCK_SIDE_CM,
            "side_taxel": BLOCK_SIDE_TAXEL,
        },
        "calibration": calibration.to_dict(),
        "camera": {
            "source": str(camera_source),
            "frame_width": int(args.frame_width),
            "frame_height": int(args.frame_height),
        },
        "vision": {
            "weights": str(args.weights),
            "confidence": float(args.confidence),
            "image_size": int(args.image_size),
            "anchor_mode": str(args.anchor_mode),
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
    parser = argparse.ArgumentParser(description="Collect block position labels from webcam + ArUco calibration.")
    parser.add_argument("--source", default="0", help="cv2.VideoCapture source: camera index, RTSP, HTTP, etc.")
    parser.add_argument(
        "--source-store",
        default=str(DEFAULT_SOURCE_STORE),
        help="Path used to remember the last successfully connected camera source.",
    )
    parser.add_argument("--weights", default=str(default_weights_path()), help="YOLO weights path.")
    parser.add_argument(
        "--calibration",
        default="block_position_prediction/data_collection/assets/paper_calibration.json",
        help="Calibration JSON path to load/save.",
    )
    parser.add_argument("--output", default="block_position_prediction/data_collection/runs", help="Dataset output root.")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--anchor-mode", choices=("center", "bottom-center"), default="center")
    parser.add_argument("--detect-rate-hz", type=float, default=5.0)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--no-low-latency", action="store_true")
    parser.add_argument("--window", default="Tactile Data Collection")
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
