from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .aruco import PaperCalibration
from .geometry import (
    BLOCK_SIDE_CM,
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
LABEL_SOURCE = "manual_aruco"
NO_BLOCK_LABEL_SOURCE = "manual_no_block"
INTERACTION_MODE = "center_drag"
BLOCK_SIDE_MM = BLOCK_SIDE_CM * 10.0


@dataclass(frozen=True)
class ManualAnnotation:
    center_px: Point2
    direction_px: Point2
    center_sensor_mm: Point2
    center_taxel: Point2
    yaw_rad: float
    yaw_mod90_rad: float
    footprint_corners_taxel: tuple[Point2, Point2, Point2, Point2]
    footprint_corners_px: tuple[Point2, Point2, Point2, Point2]
    fully_inside_sensor: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source": LABEL_SOURCE,
            "interaction": INTERACTION_MODE,
            "center_px": list(self.center_px),
            "direction_px": list(self.direction_px),
            "center_sensor_mm": list(self.center_sensor_mm),
            "center_taxel": list(self.center_taxel),
            "yaw_rad": self.yaw_rad,
            "yaw_mod90_rad": self.yaw_mod90_rad,
            "block_side_mm": BLOCK_SIDE_MM,
            "footprint_corners_taxel": [list(point) for point in self.footprint_corners_taxel],
            "footprint_corners_px": [list(point) for point in self.footprint_corners_px],
            "fully_inside_sensor": self.fully_inside_sensor,
            "updated_at": time.time(),
        }


@dataclass(frozen=True)
class CapturedSample:
    image: Any
    frame_id: int
    timestamp: float
    tactile: TactileSnapshot | None
    calibration: PaperCalibration | None


def create_manual_annotation(
    config: SheetConfig,
    calibration: PaperCalibration,
    center_px: Point2,
    direction_px: Point2 | None = None,
) -> ManualAnnotation:
    center_px = _point(center_px)
    direction_px = _point(direction_px or (center_px[0] + 40.0, center_px[1]))
    center_sensor_mm = image_px_to_sensor_mm(config, calibration, center_px)
    direction_sensor_mm = image_px_to_sensor_mm(config, calibration, direction_px)
    dx = direction_sensor_mm[0] - center_sensor_mm[0]
    dy = direction_sensor_mm[1] - center_sensor_mm[1]
    yaw_rad = 0.0 if abs(dx) + abs(dy) < 1e-9 else math.atan2(dy, dx)
    yaw_mod90_rad = normalize_yaw_mod90(yaw_rad)
    center_taxel = config.sensor.sensor_mm_to_taxel_center(*center_sensor_mm)
    footprint = fixed_square_footprint(center_taxel, yaw_mod90_rad, BLOCK_SIDE_TAXEL)
    footprint_px = taxel_footprint_to_image_px(config, calibration, footprint)
    return ManualAnnotation(
        center_px=center_px,
        direction_px=direction_px,
        center_sensor_mm=center_sensor_mm,
        center_taxel=center_taxel,
        yaw_rad=float(yaw_rad),
        yaw_mod90_rad=float(yaw_mod90_rad),
        footprint_corners_taxel=footprint,
        footprint_corners_px=footprint_px,
        fully_inside_sensor=config.sensor.footprint_fully_inside_sensor(footprint),
    )


def create_manual_annotation_at_center(
    config: SheetConfig,
    calibration: PaperCalibration,
    center_px: Point2,
    yaw_mod90_rad: float = 0.0,
) -> ManualAnnotation:
    center_sensor_mm = image_px_to_sensor_mm(config, calibration, center_px)
    center_taxel = config.sensor.sensor_mm_to_taxel_center(*center_sensor_mm)
    return create_manual_annotation_from_taxel_pose(config, calibration, center_taxel, yaw_mod90_rad)


def create_manual_annotation_from_taxel_pose(
    config: SheetConfig,
    calibration: PaperCalibration,
    center_taxel: Point2,
    yaw_mod90_rad: float = 0.0,
) -> ManualAnnotation:
    center_taxel = _point(center_taxel)
    yaw_mod90_rad = normalize_yaw_mod90(float(yaw_mod90_rad))
    center_sensor_mm = config.sensor.taxel_center_to_sensor_mm(*center_taxel)
    center_paper_mm = config.sensor_to_paper_mm(*center_sensor_mm)
    center_px = calibration.paper_to_image_px(*center_paper_mm)
    direction_sensor_mm = (
        center_sensor_mm[0] + math.cos(yaw_mod90_rad) * BLOCK_SIDE_MM * 0.5,
        center_sensor_mm[1] + math.sin(yaw_mod90_rad) * BLOCK_SIDE_MM * 0.5,
    )
    direction_paper_mm = config.sensor_to_paper_mm(*direction_sensor_mm)
    direction_px = calibration.paper_to_image_px(*direction_paper_mm)
    footprint = fixed_square_footprint(center_taxel, yaw_mod90_rad, BLOCK_SIDE_TAXEL)
    footprint_px = taxel_footprint_to_image_px(config, calibration, footprint)
    return ManualAnnotation(
        center_px=center_px,
        direction_px=direction_px,
        center_sensor_mm=center_sensor_mm,
        center_taxel=center_taxel,
        yaw_rad=float(yaw_mod90_rad),
        yaw_mod90_rad=float(yaw_mod90_rad),
        footprint_corners_taxel=footprint,
        footprint_corners_px=footprint_px,
        fully_inside_sensor=config.sensor.footprint_fully_inside_sensor(footprint),
    )


def image_px_to_sensor_mm(config: SheetConfig, calibration: PaperCalibration, point_px: Point2) -> Point2:
    paper_x, paper_y = calibration.image_to_paper_mm(float(point_px[0]), float(point_px[1]))
    return config.paper_to_sensor_mm(paper_x, paper_y)


def taxel_footprint_to_image_px(
    config: SheetConfig,
    calibration: PaperCalibration,
    footprint_taxel: Sequence[Point2],
) -> tuple[Point2, Point2, Point2, Point2]:
    corners: list[Point2] = []
    for x_taxel, y_taxel in footprint_taxel:
        sensor_mm = config.sensor.taxel_center_to_sensor_mm(x_taxel, y_taxel)
        paper_mm = config.sensor_to_paper_mm(*sensor_mm)
        corners.append(calibration.paper_to_image_px(*paper_mm))
    if len(corners) != 4:
        raise ValueError("manual block footprint must have exactly four corners")
    return tuple(corners)  # type: ignore[return-value]


def build_manual_tactile_pose_label(
    config: SheetConfig,
    calibration: PaperCalibration,
    annotation: ManualAnnotation,
    tactile: TactileSnapshot | None = None,
    tactile_values: Sequence[float] | np.ndarray | None = None,
    frame_id: int = 0,
    timestamp: float | None = None,
    sample_id: str = "",
) -> dict[str, Any]:
    values = tactile_values_for_training(tactile, tactile_values=tactile_values)
    position_taxel = annotation.center_taxel
    pose = pose_target_from_annotation(config, annotation)
    median_error = calibration.quality.median_paper_error_mm
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "frame_id": int(frame_id),
        "input": {
            "frame": SENSOR_COORD_FRAME,
            "rows": config.sensor.rows,
            "cols": config.sensor.cols,
            "values": None if values is None else values.astype(float).tolist(),
        },
        "target": {
            "frame": SENSOR_COORD_FRAME,
            "object_present": True,
            "position_taxel": list(position_taxel),
            "position_normalized": list(config.sensor.taxel_center_to_normalized(*position_taxel)),
            "position_cm_from_taxel0": list(config.sensor.taxel_center_to_cm_from_taxel0(*position_taxel)),
            "pose": pose,
        },
        "quality": {
            "label_source": LABEL_SOURCE,
            "confidence": None,
            "tactile_available": values is not None,
            "calibration_median_error_mm": median_error,
            "annotation_fully_inside_sensor": annotation.fully_inside_sensor,
        },
        "calibration": calibration.to_dict(),
        "annotation": annotation.to_metadata(),
    }


def build_no_block_tactile_pose_label(
    config: SheetConfig,
    calibration: PaperCalibration | None = None,
    tactile: TactileSnapshot | None = None,
    tactile_values: Sequence[float] | np.ndarray | None = None,
    frame_id: int = 0,
    timestamp: float | None = None,
    sample_id: str = "",
) -> dict[str, Any]:
    values = tactile_values_for_training(tactile, tactile_values=tactile_values)
    median_error = None if calibration is None else calibration.quality.median_paper_error_mm
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "frame_id": int(frame_id),
        "input": {
            "frame": SENSOR_COORD_FRAME,
            "rows": config.sensor.rows,
            "cols": config.sensor.cols,
            "values": None if values is None else values.astype(float).tolist(),
        },
        "target": {
            "frame": SENSOR_COORD_FRAME,
            "object_present": False,
            "position_taxel": None,
            "position_normalized": None,
            "position_cm_from_taxel0": None,
            "pose": {
                "available": False,
                "source": NO_BLOCK_LABEL_SOURCE,
                "yaw_mod90_rad": None,
                "yaw_mod90_vector": None,
                "footprint_side_taxel": BLOCK_SIDE_TAXEL,
                "footprint_corners_taxel": None,
                "fully_inside_sensor": False,
            },
        },
        "quality": {
            "label_source": NO_BLOCK_LABEL_SOURCE,
            "confidence": None,
            "tactile_available": values is not None,
            "calibration_median_error_mm": median_error,
            "annotation_fully_inside_sensor": False,
        },
        "calibration": None if calibration is None else calibration.to_dict(),
        "annotation": None,
    }


def tactile_values_for_training(
    tactile: TactileSnapshot | None,
    tactile_values: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    if tactile_values is not None:
        array = np.asarray(tactile_values, dtype=np.float32)
        if array.size != NUM_TAXELS:
            raise ValueError(f"expected {NUM_TAXELS} tactile values, got {array.size}")
        return array.reshape(NUM_TAXELS).astype(np.float32)
    if tactile is None or tactile.top5_normalized is None:
        return None
    return canonicalize_tactile_values(tactile.top5_normalized)


def pose_target_from_annotation(config: SheetConfig, annotation: ManualAnnotation) -> dict[str, Any]:
    return {
        "available": True,
        "source": LABEL_SOURCE,
        "yaw_mod90_rad": annotation.yaw_mod90_rad,
        "yaw_mod90_vector": list(yaw_mod90_vector(annotation.yaw_mod90_rad)),
        "footprint_side_taxel": BLOCK_SIDE_TAXEL,
        "footprint_corners_taxel": [list(point) for point in annotation.footprint_corners_taxel],
        "fully_inside_sensor": config.sensor.footprint_fully_inside_sensor(annotation.footprint_corners_taxel),
    }


def validate_manual_sample(
    config: SheetConfig,
    calibration: PaperCalibration | None,
    annotation: ManualAnnotation | None,
    tactile: TactileSnapshot | None,
    tactile_values: Sequence[float] | np.ndarray | None = None,
) -> str | None:
    if calibration is None:
        return "not calibrated"
    if annotation is None:
        return "no manual annotation"
    return None


def preview_from_annotation(
    annotation: ManualAnnotation | None,
    tactile: TactileSnapshot | None = None,
    tactile_values: Sequence[float] | np.ndarray | None = None,
    title: str = "10-frame top5 avg",
) -> TactilePreview:
    values = tactile_values_for_training(tactile, tactile_values=tactile_values)
    if annotation is None:
        return TactilePreview(values=values, position_taxel=None, title=title)
    pose = PreviewPose(
        available=True,
        source=LABEL_SOURCE,
        yaw_mod90_rad=annotation.yaw_mod90_rad,
        footprint_side_taxel=BLOCK_SIDE_TAXEL,
        footprint_corners_taxel=annotation.footprint_corners_taxel,
        fully_inside_sensor=annotation.fully_inside_sensor,
    )
    return TactilePreview(values=values, position_taxel=annotation.center_taxel, pose=pose, title=title)


def preview_from_label(label: Mapping[str, Any], config: SheetConfig | None = None) -> TactilePreview:
    config = config or SheetConfig()
    if label.get("schema_version") == SCHEMA_VERSION:
        return _preview_from_current_label(label)
    return _preview_from_legacy_label(label, config)


def annotation_from_label(label: Mapping[str, Any]) -> ManualAnnotation | None:
    target = label.get("target") or {}
    if target.get("object_present") is False:
        return None
    pose = target.get("pose") or {}
    annotation = label.get("annotation") or {}
    center_taxel = _point_or_none(target.get("position_taxel") or annotation.get("center_taxel"))
    center_sensor_mm = _point_or_none(annotation.get("center_sensor_mm"))
    center_px = _point_or_none(annotation.get("center_px"))
    direction_px = _point_or_none(annotation.get("direction_px"))
    footprint_taxel = _points4_or_none(pose.get("footprint_corners_taxel") or annotation.get("footprint_corners_taxel"))
    footprint_px = _points4_or_none(annotation.get("footprint_corners_px"))
    if (
        center_taxel is None
        or center_sensor_mm is None
        or center_px is None
        or direction_px is None
        or footprint_taxel is None
        or footprint_px is None
    ):
        return None
    yaw = float(annotation.get("yaw_rad", pose.get("yaw_mod90_rad") or 0.0))
    yaw_mod90 = float(pose.get("yaw_mod90_rad", annotation.get("yaw_mod90_rad", 0.0)))
    return ManualAnnotation(
        center_px=center_px,
        direction_px=direction_px,
        center_sensor_mm=center_sensor_mm,
        center_taxel=center_taxel,
        yaw_rad=yaw,
        yaw_mod90_rad=yaw_mod90,
        footprint_corners_taxel=footprint_taxel,
        footprint_corners_px=footprint_px,
        fully_inside_sensor=bool(pose.get("fully_inside_sensor", annotation.get("fully_inside_sensor", False))),
    )


def label_tactile_values(label: Mapping[str, Any]) -> np.ndarray | None:
    input_data = label.get("input") or {}
    return _values_array(input_data.get("values"))


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
    kind = "no-block" if target.get("object_present") is False else "tactile_pose_v1"
    title = f"{label.get('sample_id') or 'sample'} {kind}"
    return TactilePreview(values=values, position_taxel=position, pose=pose, title=title)


def _preview_from_legacy_label(label: Mapping[str, Any], _config: SheetConfig) -> TactilePreview:
    sensor_data = label.get("sensor_data") or {}
    values = _canonical_values_or_none(sensor_data.get("top5_normalized"))
    position_data = label.get("position") or {}
    position = _point_or_none(position_data.get("array_col_row"))
    title = f"{label.get('image_path') or 'legacy sample'} legacy"
    return TactilePreview(values=values, position_taxel=position, pose=PreviewPose(), title=title)


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


def _point(value: Point2 | Sequence[float]) -> Point2:
    if value is None or len(value) < 2:  # type: ignore[arg-type]
        raise ValueError("point must have at least two coordinates")
    return (float(value[0]), float(value[1]))


def _point_or_none(value: Any) -> Point2 | None:
    if value is None or len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _points4_or_none(value: Any) -> tuple[Point2, Point2, Point2, Point2] | None:
    if value is None or len(value) != 4:
        return None
    return tuple((float(point[0]), float(point[1])) for point in value)  # type: ignore[return-value]
