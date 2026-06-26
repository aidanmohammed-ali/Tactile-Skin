from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import ArucoBoardConfig, BoardGeometry, HoleRefineConfig, MarkerSpec, Point2
from .transform import BoardCalibration, CalibrationQuality


@dataclass(frozen=True)
class HoleMatch:
    image_point: Point2
    board_point: Point2
    grid_error: float


class ArucoBoardCalibrator:
    """Reusable ArUco detector plus one-shot board calibrator."""

    def __init__(self, config: ArucoBoardConfig | None = None) -> None:
        self.config = config or ArucoBoardConfig()
        self._cv2 = None
        self._np = None
        self._detector = None
        self._dictionary = None

    def calibrate(self, image: str | Path | Any) -> BoardCalibration:
        cv2, np = self._require_cv()
        frame = _load_image(cv2, np, image)
        gray = _to_gray(cv2, frame)
        image_points, board_points, marker_ids = self._detect_marker_correspondences(gray)
        if len(image_points) < 4:
            raise RuntimeError(
                "not enough configured ArUco marker corners were detected; "
                "check marker IDs, dictionary, focus, glare, and placement"
            )

        h_image_to_board, _ = _find_homography(
            cv2,
            np,
            image_points,
            board_points,
            self.config.ransac_threshold_grid,
        )
        hole_points: list[Point2] = []
        hole_board_points: list[Point2] = []
        if self.config.refine_holes.enabled:
            matches = _detect_hole_matches(
                cv2,
                np,
                gray,
                h_image_to_board,
                self.config.geometry,
                self.config.refine_holes,
            )
            if len(matches) >= self.config.refine_holes.min_holes:
                hole_points = [match.image_point for match in matches]
                hole_board_points = [match.board_point for match in matches]

        all_image_points = list(image_points) + hole_points
        all_board_points = list(board_points) + hole_board_points
        h_image_to_board, inlier_mask = _find_homography(
            cv2,
            np,
            all_image_points,
            all_board_points,
            self.config.ransac_threshold_grid,
        )
        h_board_to_image = np.linalg.inv(h_image_to_board)
        quality = _quality_stats(
            np,
            h_image_to_board,
            h_board_to_image,
            all_image_points,
            all_board_points,
            inlier_mask,
            marker_point_count=len(board_points),
            hole_point_count=len(hole_board_points),
        )
        return BoardCalibration(
            image_to_board=_matrix_to_tuple(h_image_to_board),
            board_to_image=_matrix_to_tuple(h_board_to_image),
            rows=self.config.geometry.rows,
            cols=self.config.geometry.cols,
            aruco_dictionary=self.config.aruco_dictionary,
            marker_ids=tuple(marker_ids),
            quality=quality,
            metadata={
                "method": "aruco+hole_refine" if hole_board_points else "aruco",
                "hole_pitch_mm": self.config.hole_pitch_mm,
                "configured_marker_size_grid": self.config.marker_size_grid,
                "configured_marker_size_mm": self.config.marker_size_mm,
                "configured_marker_margin_grid": self.config.marker_margin_grid,
                "configured_marker_margin_mm": self.config.marker_margin_mm,
                "configured_marker_inner_offset_mm": self.config.marker_inner_offset_mm,
                "detected_marker_ids": marker_ids,
                "refined_holes": len(hole_board_points),
            },
        )

    def detect_marker_ids(self, image: str | Path | Any) -> tuple[int, ...]:
        cv2, np = self._require_cv()
        frame = _load_image(cv2, np, image)
        gray = _to_gray(cv2, frame)
        _, ids, _ = self._detect_markers(gray)
        if ids is None:
            return ()
        return tuple(int(value) for value in ids.flatten().tolist())

    def _detect_marker_correspondences(self, gray: Any) -> tuple[list[Point2], list[Point2], list[int]]:
        cv2, _ = self._require_cv()
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            return [], [], []

        by_id = {spec.marker_id: spec for spec in self.config.resolved_marker_specs()}
        image_points: list[Point2] = []
        board_points: list[Point2] = []
        detected_marker_ids: list[int] = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        for raw_id, raw_corners in zip(ids.flatten().tolist(), corners):
            marker_id = int(raw_id)
            marker_spec = by_id.get(marker_id)
            if marker_spec is None:
                continue
            marker_corners = raw_corners.reshape(4, 2).astype("float32")
            try:
                refined = marker_corners.reshape(-1, 1, 2)
                cv2.cornerSubPix(gray, refined, (5, 5), (-1, -1), criteria)
                marker_corners = refined.reshape(4, 2)
            except cv2.error:
                pass
            for image_corner, board_corner in zip(marker_corners, marker_spec.board_corners):
                image_points.append((float(image_corner[0]), float(image_corner[1])))
                board_points.append((float(board_corner[0]), float(board_corner[1])))
            detected_marker_ids.append(marker_id)
        return image_points, board_points, detected_marker_ids

    def _detect_markers(self, gray: Any) -> tuple[Any, Any, Any]:
        self._ensure_detector()
        return self._detector.detectMarkers(gray)

    def _ensure_detector(self) -> None:
        cv2, _ = self._require_cv()
        if self._detector is not None:
            return
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is missing; install opencv-contrib-python")
        dictionary_id = getattr(cv2.aruco, self.config.aruco_dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"unknown ArUco dictionary: {self.config.aruco_dictionary}")
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, parameters)

    def _require_cv(self) -> tuple[Any, Any]:
        if self._cv2 is None or self._np is None:
            try:
                import cv2
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    "cv_aruco_src requires numpy and opencv-contrib-python"
                ) from exc
            self._cv2 = cv2
            self._np = np
        return self._cv2, self._np


def calibrate_image(image: str | Path | Any, config: ArucoBoardConfig | None = None) -> BoardCalibration:
    return ArucoBoardCalibrator(config).calibrate(image)


def _detect_hole_matches(
    cv2: Any,
    np: Any,
    gray: Any,
    h_image_to_board: Any,
    geometry: BoardGeometry,
    config: HoleRefineConfig,
) -> list[HoleMatch]:
    candidates = _detect_bright_square_candidates(cv2, np, gray, config)
    best_by_cell: dict[tuple[int, int], HoleMatch] = {}
    for point in candidates:
        board_estimate = _apply_homography_np(np, h_image_to_board, [point])[0]
        board_x = float(board_estimate[0])
        board_y = float(board_estimate[1])
        if not (
            -config.board_margin_grid <= board_x <= geometry.max_col + config.board_margin_grid
            and -config.board_margin_grid <= board_y <= geometry.max_row + config.board_margin_grid
        ):
            continue
        col = int(round(board_x))
        row = int(round(board_y))
        if not (0 <= col <= geometry.max_col and 0 <= row <= geometry.max_row):
            continue
        grid_error = math.hypot(board_x - col, board_y - row)
        if grid_error > config.assignment_max_grid_error:
            continue
        match = HoleMatch(
            image_point=(float(point[0]), float(point[1])),
            board_point=(float(col), float(row)),
            grid_error=grid_error,
        )
        key = (row, col)
        previous = best_by_cell.get(key)
        if previous is None or match.grid_error < previous.grid_error:
            best_by_cell[key] = match
    return sorted(best_by_cell.values(), key=lambda item: (item.board_point[1], item.board_point[0]))


def _detect_bright_square_candidates(cv2: Any, np: Any, gray: Any, config: HoleRefineConfig) -> list[Point2]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_c,
    )
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(adaptive, otsu)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = gray.shape[0] * gray.shape[1] * config.max_area_fraction
    max_area = min(max_area, config.max_area_px)
    candidates: list[Point2] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area_px or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect_ratio = w / float(h)
        if not (config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < config.min_fill_ratio:
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        candidates.append((float(cx), float(cy)))
    return candidates


def _find_homography(cv2: Any, np: Any, image_points: Sequence[Point2], board_points: Sequence[Point2], threshold: float) -> tuple[Any, Any]:
    if len(image_points) < 4:
        raise RuntimeError("at least four point correspondences are required")
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(board_points, dtype=np.float64).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(threshold))
    if h is None:
        raise RuntimeError("could not estimate homography")
    return h, mask


def _quality_stats(
    np: Any,
    h_image_to_board: Any,
    h_board_to_image: Any,
    image_points: Sequence[Point2],
    board_points: Sequence[Point2],
    inlier_mask: Any,
    marker_point_count: int,
    hole_point_count: int,
) -> CalibrationQuality:
    image_array = np.asarray(image_points, dtype=np.float64)
    board_array = np.asarray(board_points, dtype=np.float64)
    predicted_board = _apply_homography_np(np, h_image_to_board, image_array)
    predicted_image = _apply_homography_np(np, h_board_to_image, board_array)
    grid_errors = np.linalg.norm(predicted_board - board_array, axis=1)
    pixel_errors = np.linalg.norm(predicted_image - image_array, axis=1)
    if inlier_mask is not None:
        mask = inlier_mask.reshape(-1).astype(bool)
        grid_errors = grid_errors[mask]
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
        marker_point_count=marker_point_count,
        hole_point_count=hole_point_count,
        mean_grid_error=stat(grid_errors, np.mean),
        median_grid_error=stat(grid_errors, np.median),
        max_grid_error=stat(grid_errors, np.max),
        mean_pixel_error=stat(pixel_errors, np.mean),
        median_pixel_error=stat(pixel_errors, np.median),
        max_pixel_error=stat(pixel_errors, np.max),
    )


def _apply_homography_np(np: Any, matrix: Any, points: Sequence[Point2] | Any) -> Any:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return transformed[:, :2] / transformed[:, 2:3]


def _matrix_to_tuple(matrix: Any) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rows = tuple(tuple(float(value) for value in row) for row in matrix.tolist())
    return rows  # type: ignore[return-value]


def _load_image(cv2: Any, np: Any, image: str | Path | Any) -> Any:
    if isinstance(image, (str, Path)):
        raw = np.fromfile(str(image), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not read image: {image}")
        return frame
    return image


def _to_gray(cv2: Any, frame: Any) -> Any:
    if getattr(frame, "ndim", None) == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
