from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .aruco import PaperCalibration
from .detector import BlockDetection, DetectionResult
from .geometry import (
    BLOCK_SIDE_TAXEL,
    SENSOR_COORD_FRAME,
    Point2,
    SheetConfig,
    fixed_square_footprint,
    normalize_yaw_mod90,
    yaw_mod90_vector,
)
from .preview_render import PreviewPose, TactilePreview
from .tactile import NUM_TAXELS, TactileSnapshot, canonicalize_tactile_values


SCHEMA_VERSION = "tactile_pose_v1"
LABEL_SOURCE = "vision_yolo_aruco"


def build_tactile_pose_label(
    config: SheetConfig,
    calibration: PaperCalibration,
    detection: DetectionResult,
    tactile: TactileSnapshot | None = None,
    retained_block: bool = False,
    sample_id: str = "",
) -> dict[str, Any]:
    best = detection.best
    if best is None:
        raise ValueError("detection result has no best detection")

    position = calibration.position_label(config, best.anchor_px[0], best.anchor_px[1])
    position_taxel = position.array_col_row
    values = tactile_values_for_training(tactile)
    pose = pose_target_from_detection(config, calibration, best, position_taxel)
    median_error = calibration.quality.median_paper_error_mm

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "timestamp": detection.timestamp,
        "frame_id": detection.frame_id,
        "input": {
            "frame": SENSOR_COORD_FRAME,
            "rows": config.sensor.rows,
            "cols": config.sensor.cols,
            "values": None if values is None else values.astype(float).tolist(),
        },
        "target": {
            "frame": SENSOR_COORD_FRAME,
            "position_taxel": list(position_taxel),
            "position_normalized": list(config.sensor.taxel_center_to_normalized(*position_taxel)),
            "position_cm_from_taxel0": list(config.sensor.taxel_center_to_cm_from_taxel0(*position_taxel)),
            "pose": pose,
        },
        "quality": {
            "label_source": LABEL_SOURCE,
            "confidence": best.confidence,
            "retained_block_position": bool(retained_block),
            "tactile_available": values is not None,
            "calibration_median_error_mm": median_error,
        },
    }


def tactile_values_for_training(tactile: TactileSnapshot | None) -> np.ndarray | None:
    if tactile is None or tactile.top5_normalized is None:
        return None
    return canonicalize_tactile_values(tactile.top5_normalized)


def pose_target_from_detection(
    config: SheetConfig,
    calibration: PaperCalibration,
    detection: BlockDetection,
    center_taxel: Point2,
) -> dict[str, Any]:
    polygon = detection_polygon_taxel(config, calibration, detection)
    if polygon is None or len(polygon) < 2:
        return unavailable_pose()

    yaw = _estimate_yaw_from_polygon(polygon)
    yaw_mod90 = normalize_yaw_mod90(yaw)
    footprint = fixed_square_footprint(center_taxel, yaw_mod90, BLOCK_SIDE_TAXEL)
    return {
        "available": True,
        "source": pose_source_from_detection(detection),
        "yaw_mod90_rad": yaw_mod90,
        "yaw_mod90_vector": list(yaw_mod90_vector(yaw_mod90)),
        "footprint_side_taxel": BLOCK_SIDE_TAXEL,
        "footprint_corners_taxel": [list(point) for point in footprint],
        "fully_inside_sensor": config.sensor.footprint_fully_inside_sensor(footprint),
    }


def unavailable_pose() -> dict[str, Any]:
    return {
        "available": False,
        "source": None,
        "yaw_mod90_rad": None,
        "yaw_mod90_vector": None,
        "footprint_side_taxel": BLOCK_SIDE_TAXEL,
        "footprint_corners_taxel": None,
        "fully_inside_sensor": None,
    }


def pose_source_from_detection(detection: BlockDetection) -> str:
    if detection.polygon_px is not None and len(detection.polygon_px) >= 3:
        return "polygon_homography"
    if detection.angle_deg is not None:
        return "angle_homography"
    return "bbox_homography"


def detection_polygon_taxel(
    config: SheetConfig,
    calibration: PaperCalibration,
    detection: BlockDetection,
) -> tuple[Point2, ...] | None:
    image_polygon = detection_image_polygon_px(detection)
    if len(image_polygon) < 3:
        return None
    taxel_points = []
    for u, v in image_polygon:
        paper_x, paper_y = calibration.image_to_paper_mm(float(u), float(v))
        sensor_x, sensor_y = config.paper_to_sensor_mm(paper_x, paper_y)
        taxel_points.append(config.sensor.sensor_mm_to_taxel_center(sensor_x, sensor_y))
    return tuple(taxel_points)


def detection_image_polygon_px(detection: BlockDetection) -> tuple[Point2, ...]:
    if detection.polygon_px is not None and len(detection.polygon_px) >= 3:
        return tuple((float(x), float(y)) for x, y in detection.polygon_px)
    x1, y1, x2, y2 = detection.xyxy
    if detection.angle_deg is None:
        return ((float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2)))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max(0.0, (x2 - x1) / 2.0)
    half_h = max(0.0, (y2 - y1) / 2.0)
    cos_t = math.cos(math.radians(float(detection.angle_deg)))
    sin_t = math.sin(math.radians(float(detection.angle_deg)))
    points = []
    for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        points.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    return tuple(points)


def live_preview_from_detection(
    config: SheetConfig,
    calibration: PaperCalibration | None,
    detection: DetectionResult | None,
    tactile: TactileSnapshot | None,
    title: str = "10-frame top5 avg",
) -> TactilePreview:
    values = tactile_values_for_training(tactile)
    if calibration is None or detection is None or detection.best is None:
        return TactilePreview(values=values, position_taxel=None, title=title)
    best = detection.best
    position = calibration.position_label(config, best.anchor_px[0], best.anchor_px[1])
    pose = preview_pose_from_target(pose_target_from_detection(config, calibration, best, position.array_col_row))
    return TactilePreview(values=values, position_taxel=position.array_col_row, pose=pose, title=title)


def preview_from_label(label: Mapping[str, Any], config: SheetConfig | None = None) -> TactilePreview:
    config = config or SheetConfig()
    if label.get("schema_version") == SCHEMA_VERSION:
        return _preview_from_current_label(label)
    return _preview_from_legacy_label(label, config)


def preview_pose_from_target(pose: Mapping[str, Any] | None) -> PreviewPose:
    if not pose:
        return PreviewPose()
    corners = pose.get("footprint_corners_taxel")
    return PreviewPose(
        available=bool(pose.get("available")),
        source=None if pose.get("source") is None else str(pose["source"]),
        yaw_mod90_rad=None if pose.get("yaw_mod90_rad") is None else float(pose["yaw_mod90_rad"]),
        footprint_side_taxel=float(pose.get("footprint_side_taxel", BLOCK_SIDE_TAXEL)),
        footprint_corners_taxel=None if corners is None else tuple((float(x), float(y)) for x, y in corners),
        fully_inside_sensor=pose.get("fully_inside_sensor"),
    )


def _preview_from_current_label(label: Mapping[str, Any]) -> TactilePreview:
    input_data = label.get("input") or {}
    target = label.get("target") or {}
    values = _values_array(input_data.get("values"))
    position = _point_or_none(target.get("position_taxel"))
    pose = preview_pose_from_target(target.get("pose") or {})
    title = f"{label.get('sample_id') or 'sample'} tactile_pose_v1"
    return TactilePreview(values=values, position_taxel=position, pose=pose, title=title)


def _preview_from_legacy_label(label: Mapping[str, Any], config: SheetConfig) -> TactilePreview:
    sensor_data = label.get("sensor_data") or {}
    values = _canonical_values_or_none(sensor_data.get("top5_normalized"))
    position_data = label.get("position") or {}
    position = _point_or_none(position_data.get("array_col_row"))
    pose = PreviewPose()

    detection_data = label.get("detection") or {}
    if position is not None and (detection_data.get("angle_deg") is not None or detection_data.get("polygon_px") is not None):
        try:
            calibration = PaperCalibration.from_dict(label["calibration"])
            detection = _block_detection_from_legacy(detection_data)
            pose = preview_pose_from_target(pose_target_from_detection(config, calibration, detection, position))
        except Exception:
            pose = PreviewPose()

    title = f"{label.get('image_path') or 'legacy sample'} legacy"
    return TactilePreview(values=values, position_taxel=position, pose=pose, title=title)


def _block_detection_from_legacy(data: Mapping[str, Any]) -> BlockDetection:
    bbox = tuple(int(value) for value in data.get("bbox_xyxy", (0, 0, 0, 0)))
    center = tuple(float(value) for value in data.get("center_px", (0.0, 0.0)))
    anchor = tuple(float(value) for value in data.get("anchor_px", center))
    polygon = data.get("polygon_px")
    return BlockDetection(
        xyxy=bbox,  # type: ignore[arg-type]
        center_px=center,  # type: ignore[arg-type]
        confidence=float(data.get("confidence", 0.0)),
        class_id=int(data.get("class_id", 0)),
        class_name=str(data.get("class_name", "block")),
        anchor_px=anchor,  # type: ignore[arg-type]
        anchor_mode=str(data.get("anchor_mode", "center")),
        angle_deg=None if data.get("angle_deg") is None else float(data["angle_deg"]),
        polygon_px=None if polygon is None else tuple((float(x), float(y)) for x, y in polygon),
        contact_point_px=None,
    )


def _values_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.size != NUM_TAXELS:
        return None
    return array.reshape(NUM_TAXELS)


def _canonical_values_or_none(values: Any) -> np.ndarray | None:
    try:
        return canonicalize_tactile_values(values)
    except ValueError:
        return None


def _point_or_none(value: Any) -> Point2 | None:
    if value is None or len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _estimate_yaw_from_polygon(polygon_taxel: tuple[Point2, ...]) -> float:
    x1, y1 = polygon_taxel[0]
    x2, y2 = polygon_taxel[1]
    return math.atan2(y2 - y1, x2 - x1)
