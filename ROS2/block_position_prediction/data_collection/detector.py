from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


Point2 = tuple[float, float]


@dataclass(frozen=True)
class BlockDetection:
    xyxy: tuple[int, int, int, int]
    center_px: Point2
    confidence: float
    class_id: int
    class_name: str
    anchor_px: Point2
    anchor_mode: str = "center"
    angle_deg: float | None = None
    polygon_px: tuple[Point2, ...] | None = None
    contact_point_px: Point2 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": list(self.xyxy),
            "center_px": list(self.center_px),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "anchor_px": list(self.anchor_px),
            "anchor_mode": self.anchor_mode,
            "angle_deg": self.angle_deg,
            "polygon_px": None if self.polygon_px is None else [list(point) for point in self.polygon_px],
            "contact_point_px": None if self.contact_point_px is None else list(self.contact_point_px),
        }


@dataclass(frozen=True)
class DetectionResult:
    frame_id: int
    timestamp: float
    detections: tuple[BlockDetection, ...]
    elapsed_ms: float
    error: str | None = None

    @property
    def best(self) -> BlockDetection | None:
        return self.detections[0] if self.detections else None


def default_weights_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "block3.pt"


class BlockDetector:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float = 0.1,
        image_size: int | None = 640,
        anchor_mode: str = "center",
        device: str | int | None = None,
        model: Any | None = None,
    ) -> None:
        self.weights_path = Path(weights_path or default_weights_path())
        self.confidence = float(confidence)
        self.image_size = image_size
        self.anchor_mode = anchor_mode
        self.device = device
        self._model = model

    def detect(self, frame: Any, frame_id: int = 0, timestamp: float | None = None) -> DetectionResult:
        started = time.perf_counter()
        try:
            detections = self.detect_blocks(frame)
            error = None
        except Exception as exc:
            detections = []
            error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionResult(
            frame_id=int(frame_id),
            timestamp=time.time() if timestamp is None else float(timestamp),
            detections=tuple(detections),
            elapsed_ms=elapsed_ms,
            error=error,
        )

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
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            anchor = self._position_anchor(x1, y1, x2, y2)
            detections.append(
                BlockDetection(
                    xyxy=(x1, y1, x2, y2),
                    center_px=center,
                    confidence=conf,
                    class_id=class_id,
                    class_name=str(names.get(class_id, f"class_{class_id}")),
                    anchor_px=anchor,
                    anchor_mode=self.anchor_mode,
                    contact_point_px=anchor if self.anchor_mode != "center" else None,
                )
            )
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

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

    def _position_anchor(self, x1: int, y1: int, x2: int, y2: int) -> Point2:
        if self.anchor_mode == "bottom-center":
            return ((x1 + x2) / 2.0, float(y2))
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def parse_device(value: str) -> str | int | None:
    text = str(value).strip()
    if not text or text == "auto":
        return None
    return int(text) if text.isdigit() else text


def best_detection(detections: Sequence[BlockDetection]) -> BlockDetection | None:
    return detections[0] if detections else None

