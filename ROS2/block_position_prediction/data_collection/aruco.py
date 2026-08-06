from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .geometry import Point2, SheetConfig, label_from_paper_position


Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class CalibrationQuality:
    point_count: int = 0
    inlier_count: int = 0
    mean_paper_error_mm: float | None = None
    median_paper_error_mm: float | None = None
    max_paper_error_mm: float | None = None
    mean_pixel_error: float | None = None
    median_pixel_error: float | None = None
    max_pixel_error: float | None = None


@dataclass(frozen=True)
class PaperCalibration:
    image_to_paper: Matrix3
    paper_to_image: Matrix3
    marker_ids: tuple[int, ...]
    dictionary: str
    quality: CalibrationQuality = field(default_factory=CalibrationQuality)
    sheet: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_to_paper", _as_matrix3(self.image_to_paper))
        object.__setattr__(self, "paper_to_image", _as_matrix3(self.paper_to_image))
        object.__setattr__(self, "marker_ids", tuple(int(value) for value in self.marker_ids))
        object.__setattr__(self, "sheet", dict(self.sheet))

    def image_to_paper_mm(self, u: float, v: float) -> Point2:
        return apply_homography(self.image_to_paper, float(u), float(v))

    def paper_to_image_px(self, x_mm: float, y_mm: float) -> Point2:
        return apply_homography(self.paper_to_image, float(x_mm), float(y_mm))

    def position_label(self, config: SheetConfig, u: float, v: float):
        paper_x, paper_y = self.image_to_paper_mm(u, v)
        return label_from_paper_position(config, paper_x, paper_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_to_paper": [list(row) for row in self.image_to_paper],
            "paper_to_image": [list(row) for row in self.paper_to_image],
            "marker_ids": list(self.marker_ids),
            "dictionary": self.dictionary,
            "quality": asdict(self.quality),
            "sheet": dict(self.sheet),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperCalibration":
        quality_data = data.get("quality") or {}
        quality = quality_data if isinstance(quality_data, CalibrationQuality) else CalibrationQuality(**quality_data)
        return cls(
            image_to_paper=data["image_to_paper"],
            paper_to_image=data["paper_to_image"],
            marker_ids=tuple(data.get("marker_ids") or ()),
            dictionary=str(data.get("dictionary", "DICT_4X4_50")),
            quality=quality,
            sheet=data.get("sheet") or {},
            created_at=float(data.get("created_at", time.time())),
        )

    def save_json(self, path: str | Path) -> None:
        save_calibration(path, self)


class ArucoPaperCalibrator:
    """One-shot ArUco calibration from image pixels to A4 paper millimeters."""

    def __init__(self, sheet_config: SheetConfig | None = None) -> None:
        self.sheet_config = sheet_config or SheetConfig()
        self._cv2 = None
        self._np = None
        self._detector = None

    def calibrate(self, image: str | Path | Any) -> PaperCalibration:
        cv2, np = self._require_cv()
        frame = _load_image(cv2, np, image)
        gray = frame if getattr(frame, "ndim", None) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_points, paper_points, marker_ids = self._detect_correspondences(gray)
        if len(image_points) < 4:
            raise RuntimeError("not enough configured ArUco marker corners were detected")
        h_image_to_paper, inlier_mask = _find_homography(cv2, np, image_points, paper_points, threshold=3.0)
        h_paper_to_image = np.linalg.inv(h_image_to_paper)
        quality = _quality_stats(np, h_image_to_paper, h_paper_to_image, image_points, paper_points, inlier_mask)
        return PaperCalibration(
            image_to_paper=_matrix_to_tuple(h_image_to_paper),
            paper_to_image=_matrix_to_tuple(h_paper_to_image),
            marker_ids=tuple(marker_ids),
            dictionary=self.sheet_config.marker_layout.dictionary,
            quality=quality,
            sheet=self.sheet_config.to_dict(),
        )

    def detect_marker_ids(self, image: str | Path | Any) -> tuple[int, ...]:
        cv2, np = self._require_cv()
        frame = _load_image(cv2, np, image)
        gray = frame if getattr(frame, "ndim", None) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, ids, _ = self._detect_markers(gray)
        if ids is None:
            return ()
        return tuple(int(value) for value in ids.flatten().tolist())

    def _detect_correspondences(self, gray: Any) -> tuple[list[Point2], list[Point2], list[int]]:
        cv2, _ = self._require_cv()
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            return [], [], []
        specs = {spec.marker_id: spec for spec in self.sheet_config.marker_specs()}
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        image_points: list[Point2] = []
        paper_points: list[Point2] = []
        marker_ids: list[int] = []
        for raw_id, raw_corners in zip(ids.flatten().tolist(), corners):
            marker_id = int(raw_id)
            spec = specs.get(marker_id)
            if spec is None:
                continue
            marker_corners = raw_corners.reshape(4, 2).astype("float32")
            try:
                refined = marker_corners.reshape(-1, 1, 2)
                cv2.cornerSubPix(gray, refined, (5, 5), (-1, -1), criteria)
                marker_corners = refined.reshape(4, 2)
            except cv2.error:
                pass
            for image_corner, paper_corner in zip(marker_corners, spec.paper_corners_mm):
                image_points.append((float(image_corner[0]), float(image_corner[1])))
                paper_points.append((float(paper_corner[0]), float(paper_corner[1])))
            marker_ids.append(marker_id)
        return image_points, paper_points, marker_ids

    def _detect_markers(self, gray: Any) -> tuple[Any, Any, Any]:
        self._ensure_detector()
        return self._detector.detectMarkers(gray)

    def _ensure_detector(self) -> None:
        cv2, _ = self._require_cv()
        if self._detector is not None:
            return
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is missing; install opencv-contrib-python")
        dictionary_id = getattr(cv2.aruco, self.sheet_config.marker_layout.dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"unknown ArUco dictionary: {self.sheet_config.marker_layout.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def _require_cv(self) -> tuple[Any, Any]:
        if self._cv2 is None or self._np is None:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("numpy and opencv-contrib-python are required for calibration") from exc
            self._cv2 = cv2
            self._np = np
        return self._cv2, self._np


def load_calibration(path: str | Path) -> PaperCalibration:
    with Path(path).open("r", encoding="utf-8") as file:
        return PaperCalibration.from_dict(json.load(file))


def save_calibration(path: str | Path, calibration: PaperCalibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(calibration.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")


def apply_homography(matrix: Matrix3, x: float, y: float) -> Point2:
    denom = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denom) < 1e-12:
        raise ZeroDivisionError("homogeneous point has zero scale")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denom,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denom,
    )


def _find_homography(cv2: Any, np: Any, image_points: Sequence[Point2], paper_points: Sequence[Point2], threshold: float):
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(paper_points, dtype=np.float64).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(threshold))
    if h is None:
        raise RuntimeError("could not estimate homography")
    return h, mask


def _quality_stats(
    np: Any,
    h_image_to_paper: Any,
    h_paper_to_image: Any,
    image_points: Sequence[Point2],
    paper_points: Sequence[Point2],
    inlier_mask: Any,
) -> CalibrationQuality:
    image_array = np.asarray(image_points, dtype=np.float64)
    paper_array = np.asarray(paper_points, dtype=np.float64)
    predicted_paper = _apply_homography_np(np, h_image_to_paper, image_array)
    predicted_image = _apply_homography_np(np, h_paper_to_image, paper_array)
    paper_errors = np.linalg.norm(predicted_paper - paper_array, axis=1)
    pixel_errors = np.linalg.norm(predicted_image - image_array, axis=1)
    if inlier_mask is not None:
        mask = inlier_mask.reshape(-1).astype(bool)
        paper_errors = paper_errors[mask]
        pixel_errors = pixel_errors[mask]
        inlier_count = int(mask.sum())
    else:
        inlier_count = len(image_points)

    def stat(values: Any, fn: Any) -> float | None:
        if len(values) == 0:
            return None
        return float(fn(values))

    return CalibrationQuality(
        point_count=len(image_points),
        inlier_count=inlier_count,
        mean_paper_error_mm=stat(paper_errors, np.mean),
        median_paper_error_mm=stat(paper_errors, np.median),
        max_paper_error_mm=stat(paper_errors, np.max),
        mean_pixel_error=stat(pixel_errors, np.mean),
        median_pixel_error=stat(pixel_errors, np.median),
        max_pixel_error=stat(pixel_errors, np.max),
    )


def _apply_homography_np(np: Any, matrix: Any, points: Sequence[Point2] | Any) -> Any:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return transformed[:, :2] / transformed[:, 2:3]


def _load_image(cv2: Any, np: Any, image: str | Path | Any) -> Any:
    if isinstance(image, (str, Path)):
        raw = np.fromfile(str(image), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not read image: {image}")
        return frame
    return image


def _matrix_to_tuple(matrix: Any) -> Matrix3:
    rows = tuple(tuple(float(value) for value in row) for row in matrix.tolist())
    return rows  # type: ignore[return-value]


def _as_matrix3(matrix: Sequence[Sequence[float]]) -> Matrix3:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("homography must be a 3x3 matrix")
    return rows  # type: ignore[return-value]

