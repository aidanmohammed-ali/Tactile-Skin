from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_aruco_src import (  # noqa: E402
    ArucoBoardCalibrator,
    ArucoBoardConfig,
    BoardCalibration,
    HoleRefineConfig,
    draw_board_overlay,
    load_calibration,
    pixel_to_board,
)


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    center: tuple[float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class _Button:
    name: str
    label: str
    rect: tuple[int, int, int, int]

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


class RealtimeBlockDetector:
    """Live camera view with calibrated board grid and YOLO block position."""

    def __init__(
        self,
        source: str | int,
        weights_path: str | Path,
        calibration_path: str | Path,
        config: ArucoBoardConfig,
        confidence: float = 0.1,
        image_size: int | None = None,
        anchor: str = "center",
        device: str | int | None = None,
        frame_width: int | None = 1920,
        frame_height: int | None = 1080,
        window_name: str = "Block Position",
    ) -> None:
        self.source = source
        self.weights_path = Path(weights_path)
        self.calibration_path = Path(calibration_path)
        self.config = config
        self.confidence = float(confidence)
        self.image_size = image_size
        self.anchor = anchor
        self.device = device if device is not None else default_device()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.window_name = window_name
        self.calibrator = ArucoBoardCalibrator(config)
        self.calibration: BoardCalibration | None = None
        self.status = "Loading..."
        self.status_color = (235, 235, 235)
        self.buttons: list[_Button] = []
        self._calibrate_requested = False
        self.model = self._load_model()

    def run(self) -> None:
        self._load_existing_calibration()
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.frame_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video source: {self.source}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("Waiting for video frame...", error=True)
                    time.sleep(0.05)
                    continue

                if self._calibrate_requested:
                    self._calibrate_requested = False
                    self._calibrate_from_frame(frame)

                detection = self._detect_best_block(frame)
                display = self._render_frame(frame, detection)
                cv2.imshow(self.window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    self._calibrate_from_frame(frame)
                if key == ord("s"):
                    self._save_snapshot(display)
        finally:
            capture.release()
            cv2.destroyWindow(self.window_name)

    def _load_model(self) -> Any:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLOv8 inference. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        return YOLO(str(self.weights_path))

    def _load_existing_calibration(self) -> None:
        if not self.calibration_path.exists():
            self._set_status("No calibration loaded; press C to calibrate", error=True)
            return
        try:
            self.calibration = load_calibration(self.calibration_path)
            self._set_status(f"Loaded calibration: {self.calibration_path}")
        except Exception as exc:
            self._set_status(f"Could not load calibration: {exc}", error=True)

    def _calibrate_from_frame(self, frame: Any) -> None:
        started = time.perf_counter()
        try:
            calibration = self.calibrator.calibrate(frame)
            calibration.save_json(self.calibration_path)
            self.calibration = calibration
            quality = calibration.quality
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            median = quality.median_grid_error
            median_text = "n/a" if median is None else f"{median:.4f}"
            self._set_status(
                "Calibrated "
                f"markers={len(calibration.marker_ids)} "
                f"holes={quality.hole_point_count} "
                f"median_grid={median_text} "
                f"({elapsed_ms:.0f} ms)"
            )
        except Exception as exc:
            self._set_status(f"Calibration failed: {exc}", error=True)

    def _detect_best_block(self, frame: Any) -> Detection | None:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size
        results = self.model.predict(source=frame, **kwargs)
        if not results:
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        names = getattr(result, "names", {}) or {}
        best: Detection | None = None
        for box in boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            center = self._position_anchor(x1, y1, x2, y2)
            class_name = str(names.get(class_id, f"class_{class_id}"))
            detection = Detection(
                xyxy=(x1, y1, x2, y2),
                center=center,
                confidence=conf,
                class_id=class_id,
                class_name=class_name,
            )
            if best is None or detection.confidence > best.confidence:
                best = detection
        return best

    def _position_anchor(self, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
        if self.anchor == "bottom-center":
            return ((x1 + x2) / 2.0, float(y2))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _render_frame(self, frame: Any, detection: Detection | None) -> Any:
        if self.calibration is None:
            display = frame.copy()
        else:
            display = draw_board_overlay(frame, self.calibration)

        board_position: tuple[float, float] | None = None
        if detection is not None:
            board_position = self._draw_detection(display, detection)

        self._draw_toolbar(display, detection, board_position)
        return display

    def _draw_detection(self, display: Any, detection: Detection) -> tuple[float, float] | None:
        x1, y1, x2, y2 = detection.xyxy
        u, v = detection.center
        cv2.rectangle(display, (x1, y1), (x2, y2), (40, 170, 255), 2)
        cv2.drawMarker(
            display,
            (int(round(u)), int(round(v))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )

        board_position = None
        if self.calibration is not None:
            board_position = pixel_to_board(u, v, self.calibration)
            col, row = board_position
            label = (
                f"{detection.class_name} {detection.confidence:.2f} "
                f"px=({u:.0f},{v:.0f}) grid=({col:.2f},{row:.2f})"
            )
        else:
            label = (
                f"{detection.class_name} {detection.confidence:.2f} "
                f"px=({u:.0f},{v:.0f})"
            )
        self._draw_label(display, label, x1, max(82, y1 - 10))
        return board_position

    def _draw_toolbar(
        self,
        frame: Any,
        detection: Detection | None,
        board_position: tuple[float, float] | None,
    ) -> None:
        toolbar = frame.copy()
        cv2.rectangle(toolbar, (0, 0), (frame.shape[1], 76), (20, 20, 20), -1)
        cv2.addWeighted(toolbar, 0.72, frame, 0.28, 0, frame)
        self.buttons = self._toolbar_buttons(frame.shape[1])
        for button in self.buttons:
            self._draw_button(frame, button)

        status = self.status
        if detection is None:
            position_text = "Block: not detected"
        elif board_position is None:
            u, v = detection.center
            position_text = f"Block pixel: {u:.0f}, {v:.0f}"
        else:
            col, row = board_position
            position_text = f"Block grid: col {col:.2f}, row {row:.2f}"

        cv2.putText(
            frame,
            position_text,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status,
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.status_color,
            2,
            cv2.LINE_AA,
        )

    def _toolbar_buttons(self, frame_width: int) -> list[_Button]:
        right = max(150, frame_width - 16)
        return [
            _Button("calibrate", "Recalibrate", (right - 156, 18, right, 58)),
        ]

    def _draw_button(self, frame: Any, button: _Button) -> None:
        cv2.rectangle(frame, button.rect[:2], button.rect[2:], (45, 120, 45), -1)
        cv2.rectangle(frame, button.rect[:2], button.rect[2:], (230, 230, 230), 1)
        cv2.putText(
            frame,
            button.label,
            (button.rect[0] + 14, button.rect[1] + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y) and button.name == "calibrate":
                self._calibrate_requested = True
                self._set_status("Calibrating...")
                return

    def _draw_label(self, frame: Any, text: str, x: int, y: int) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(4, min(x, frame.shape[1] - width - 8))
        y = max(height + 4, min(y, frame.shape[0] - baseline - 4))
        cv2.rectangle(
            frame,
            (x - 4, y - height - 6),
            (x + width + 4, y + baseline + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _save_snapshot(self, frame: Any) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = self.weights_path.parent / f"block_position_{timestamp}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            self._set_status("Snapshot failed: could not encode frame", error=True)
            return
        encoded.tofile(str(output))
        self._set_status(f"Saved snapshot: {output}")

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status = message
        self.status_color = (80, 80, 255) if error else (235, 235, 235)


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def parse_device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def default_device() -> str | int:
    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 block detection with calibrated board-grid coordinates."
    )
    parser.add_argument("--source", default="0", help="cv2.VideoCapture source: camera index, RTSP, HTTP, etc.")
    parser.add_argument("--weights", default="ObjectDetection/block2.pt", help="YOLOv8 weights path.")
    parser.add_argument(
        "--calibration",
        default="cv_aruco_src/board_calibration.json",
        help="Board calibration JSON path to load/save.",
    )
    parser.add_argument("--confidence", type=float, default=0.1, help="Minimum YOLO confidence.")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example 0 for CUDA GPU or cpu. Defaults to CUDA 0 when available.",
    )
    parser.add_argument("--frame-width", type=int, default=1920, help="Requested camera frame width.")
    parser.add_argument("--frame-height", type=int, default=1080, help="Requested camera frame height.")
    parser.add_argument(
        "--anchor",
        choices=("center", "bottom-center"),
        default="center",
        help="Bounding-box point to convert to board coordinates.",
    )
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name.")
    parser.add_argument("--hole-pitch-mm", type=float, default=25.0, help="Physical hole pitch in millimeters.")
    parser.add_argument("--marker-size-grid", type=float, default=1.6, help="Marker side length in hole spacings.")
    parser.add_argument(
        "--marker-margin-grid",
        type=float,
        default=-1.40,
        help="Marker margin in hole spacings; negative means markers are inside the board.",
    )
    parser.add_argument("--no-hole-refine", action="store_true", help="Use ArUco corners only when recalibrating.")
    parser.add_argument("--window", default="Block Position", help="OpenCV window name.")
    args = parser.parse_args()

    config = ArucoBoardConfig(
        hole_pitch_mm=args.hole_pitch_mm,
        aruco_dictionary=args.dictionary,
        marker_size_grid=args.marker_size_grid,
        marker_margin_grid=args.marker_margin_grid,
        refine_holes=HoleRefineConfig(enabled=not args.no_hole_refine),
    )
    app = RealtimeBlockDetector(
        source=parse_source(args.source),
        weights_path=args.weights,
        calibration_path=args.calibration,
        config=config,
        confidence=args.confidence,
        image_size=args.imgsz,
        anchor=args.anchor,
        device=None if args.device is None else parse_device(args.device),
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        window_name=args.window,
    )
    app.run()


if __name__ == "__main__":
    main()
