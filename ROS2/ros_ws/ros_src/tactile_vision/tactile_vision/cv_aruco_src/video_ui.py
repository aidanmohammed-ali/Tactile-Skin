from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .calibration import ArucoBoardCalibrator
from .config import ArucoBoardConfig, HoleRefineConfig
from .overlay import draw_board_overlay
from .transform import BoardCalibration, load_calibration


@dataclass
class _Button:
    name: str
    label: str
    rect: tuple[int, int, int, int]

    def contains(self, x: int, y: int) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


class CalibrationVideoApp:
    """OpenCV HighGUI app for fixed-camera calibration and live overlay."""

    def __init__(
        self,
        source: str | int,
        calibration_path: str | Path,
        config: ArucoBoardConfig | None = None,
        window_name: str = "ArUco Board Calibration",
    ) -> None:
        self.source = source
        self.calibration_path = Path(calibration_path)
        self.window_name = window_name
        self.calibrator = ArucoBoardCalibrator(config)
        self.calibration: BoardCalibration | None = None
        self.status = "Press Calibrate or C"
        self.status_color = (235, 235, 235)
        self.buttons: list[_Button] = []
        self._calibrate_requested = False
        self._quit_requested = False

    def run(self) -> None:
        self._load_existing_calibration()
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"could not open video source: {self.source}")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        last_frame = None
        try:
            while not self._quit_requested:
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._set_status("Waiting for video frame...", error=True)
                    time.sleep(0.05)
                    continue
                last_frame = frame
                if self._calibrate_requested:
                    self._calibrate_requested = False
                    self._calibrate_from_frame(frame)
                display = self._render_frame(frame)
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c") and last_frame is not None:
                    self._calibrate_from_frame(last_frame)
        finally:
            capture.release()
            cv2.destroyWindow(self.window_name)

    def _load_existing_calibration(self) -> None:
        if not self.calibration_path.exists():
            return
        try:
            self.calibration = load_calibration(self.calibration_path)
            self._set_status(f"Loaded {self.calibration_path}")
        except Exception as exc:
            self._set_status(f"Could not load calibration: {exc}", error=True)

    def _calibrate_from_frame(self, frame) -> None:
        started = time.perf_counter()
        try:
            calibration = self.calibrator.calibrate(frame)
            calibration.save_json(self.calibration_path)
            self.calibration = calibration
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            quality = calibration.quality
            self._set_status(
                "Calibrated "
                f"markers={len(calibration.marker_ids)} "
                f"holes={quality.hole_point_count} "
                f"median_grid={quality.median_grid_error:.4f} "
                f"({elapsed_ms:.0f} ms)"
            )
        except Exception as exc:
            self._set_status(f"Calibration failed: {exc}", error=True)

    def _render_frame(self, frame):
        if self.calibration is not None:
            display = draw_board_overlay(frame, self.calibration)
        else:
            display = frame.copy()
        self._draw_toolbar(display)
        return display

    def _draw_toolbar(self, frame) -> None:
        self.buttons = [
            _Button("calibrate", "Calibrate", (16, 14, 162, 58)),
            _Button("quit", "Quit", (176, 14, 268, 58)),
        ]
        toolbar = frame.copy()
        cv2.rectangle(toolbar, (0, 0), (frame.shape[1], 76), (20, 20, 20), -1)
        cv2.addWeighted(toolbar, 0.72, frame, 0.28, 0, frame)
        for button in self.buttons:
            color = (45, 120, 45) if button.name == "calibrate" else (80, 80, 80)
            cv2.rectangle(frame, button.rect[:2], button.rect[2:], color, -1)
            cv2.rectangle(frame, button.rect[:2], button.rect[2:], (230, 230, 230), 1)
            cv2.putText(
                frame,
                button.label,
                (button.rect[0] + 16, button.rect[1] + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            self.status,
            (292, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            self.status_color,
            2,
            cv2.LINE_AA,
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if not button.contains(x, y):
                continue
            if button.name == "calibrate":
                self._calibrate_requested = True
                self._set_status("Calibrating...")
            elif button.name == "quit":
                self._quit_requested = True
            return

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status = message
        self.status_color = (80, 80, 255) if error else (235, 235, 235)


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Live fixed-camera ArUco board calibration UI.")
    parser.add_argument("--source", default="0", help="cv2.VideoCapture source: camera index, RTSP, HTTP, etc.")
    parser.add_argument(
        "--calibration",
        default="cv_aruco_src/board_calibration.json",
        help="Calibration JSON path to load/save.",
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
    parser.add_argument("--no-hole-refine", action="store_true", help="Use ArUco corners only.")
    parser.add_argument("--window", default="ArUco Board Calibration", help="OpenCV window name.")
    args = parser.parse_args()

    config = ArucoBoardConfig(
        hole_pitch_mm=args.hole_pitch_mm,
        aruco_dictionary=args.dictionary,
        marker_size_grid=args.marker_size_grid,
        marker_margin_grid=args.marker_margin_grid,
        refine_holes=HoleRefineConfig(enabled=not args.no_hole_refine),
    )
    app = CalibrationVideoApp(
        source=parse_source(args.source),
        calibration_path=args.calibration,
        config=config,
        window_name=args.window,
    )
    app.run()


if __name__ == "__main__":
    main()
