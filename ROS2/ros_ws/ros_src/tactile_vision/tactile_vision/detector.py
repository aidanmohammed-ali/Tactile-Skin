from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .cv_aruco_src import (
    ArucoBoardCalibrator,
    ArucoBoardConfig,
    BoardCalibration,
    HoleRefineConfig,
    draw_board_overlay,
    load_calibration,
    pixel_to_board,
)


@dataclass(frozen=True)
class BlockDetection:
    xyxy: tuple[int, int, int, int]
    center: tuple[float, float]
    confidence: float
    class_id: int
    class_name: str


def default_weights_path() -> str:
    return str(Path(__file__).with_name("block3.pt"))


def default_calibration_path() -> str:
    return str(Path(__file__).parent / "cv_aruco_src" / "board_calibration.json")


class BlockDetector:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        confidence: float = 0.1,
        image_size: int | None = None,
        anchor: str = "center",
        device: str | int | None = None,
        board_config: ArucoBoardConfig | None = None,
    ) -> None:
        self.weights_path = Path(weights_path or default_weights_path())
        self.calibration_path = Path(calibration_path or default_calibration_path())
        self.confidence = float(confidence)
        self.image_size = image_size
        self.anchor = anchor
        self.device = device
        self.board_config = board_config or ArucoBoardConfig()
        self.calibrator = ArucoBoardCalibrator(self.board_config)
        self.calibration: BoardCalibration | None = None
        self._model: Any | None = None
        self.load_existing_calibration()

    def load_existing_calibration(self) -> None:
        if not self.calibration_path.exists():
            self.calibration = None
            return
        self.calibration = load_calibration(self.calibration_path)

    def calibrate_from_frame(self, frame: Any, save: bool = True) -> BoardCalibration:
        calibration = self.calibrator.calibrate(frame)
        if save:
            calibration.save_json(self.calibration_path)
        self.calibration = calibration
        return calibration

    def detect_best_block(self, frame: Any) -> BlockDetection | None:
        detections = self.detect_blocks(frame)
        return detections[0] if detections else None

    def detect_blocks(self, frame: Any) -> list[BlockDetection]:
        model = self._ensure_model()
        kwargs: dict[str, Any] = {"conf": self.confidence, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        if self.image_size:
            kwargs["imgsz"] = self.image_size
        results = model.predict(source=frame, **kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = getattr(result, "names", {}) or {}
        detections: list[BlockDetection] = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue
            class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            detections.append(
                BlockDetection(
                    xyxy=(x1, y1, x2, y2),
                    center=self._position_anchor(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=str(names.get(class_id, f"class_{class_id}")),
                )
            )
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

    def board_position(self, detection: BlockDetection) -> tuple[float, float] | None:
        if self.calibration is None:
            return None
        return pixel_to_board(detection.center[0], detection.center[1], self.calibration)

    def render_frame(self, frame: Any, detections: BlockDetection | Sequence[BlockDetection] | None) -> Any:
        import cv2

        if self.calibration is None:
            display = frame.copy()
        else:
            display = draw_board_overlay(frame, self.calibration)
        detection_list = self._detection_list(detections)
        if not detection_list:
            self._draw_label(display, f"Block: not detected conf>={self.confidence:.2f}", 16, 32)
            return display
        for index, detection in enumerate(detection_list, start=1):
            x1, y1, x2, y2 = detection.xyxy
            u, v = detection.center
            color = (40, 170, 255) if index == 1 else (70, 220, 90)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.drawMarker(
                display,
                (int(round(u)), int(round(v))),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=22,
                thickness=2,
            )
            label = self._detection_label(index, detection)
            self._draw_label(display, label, x1, max(32, y1 - 10))
        return display

    @staticmethod
    def _detection_list(detections: BlockDetection | Sequence[BlockDetection] | None) -> list[BlockDetection]:
        if detections is None:
            return []
        if isinstance(detections, BlockDetection):
            return [detections]
        return list(detections)

    def _detection_label(self, index: int, detection: BlockDetection) -> str:
        u, v = detection.center
        board = self.board_position(detection)
        prefix = f"#{index} {detection.class_name} {detection.confidence:.2f}"
        if board is None:
            return f"{prefix} px=({u:.0f},{v:.0f})"
        col, row = board
        return f"{prefix} px=({u:.0f},{v:.0f}) grid=({col:.2f},{row:.2f})"

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLO inference") from exc
        self._model = YOLO(str(self.weights_path))
        return self._model

    def _position_anchor(self, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
        if self.anchor == "bottom-center":
            return ((x1 + x2) / 2.0, float(y2))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _draw_label(frame: Any, text: str, x: int, y: int) -> None:
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 2
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(4, min(int(x), frame.shape[1] - width - 8))
        y = max(height + 4, min(int(y), frame.shape[0] - baseline - 4))
        cv2.rectangle(
            frame,
            (x - 4, y - height - 6),
            (x + width + 4, y + baseline + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def make_board_config(
    hole_pitch_mm: float,
    dictionary: str,
    marker_size_grid: float,
    marker_margin_grid: float,
    refine_holes: bool,
) -> ArucoBoardConfig:
    return ArucoBoardConfig(
        hole_pitch_mm=float(hole_pitch_mm),
        aruco_dictionary=str(dictionary),
        marker_size_grid=float(marker_size_grid),
        marker_margin_grid=float(marker_margin_grid),
        refine_holes=HoleRefineConfig(enabled=bool(refine_holes)),
    )


def parse_device(value: str) -> str | int | None:
    text = str(value).strip()
    if not text or text == "auto":
        return None
    return int(text) if text.isdigit() else text
