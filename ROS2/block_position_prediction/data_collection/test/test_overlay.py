import math

import numpy as np

from block_position_prediction.data_collection.aruco import PaperCalibration
from block_position_prediction.data_collection.detector import BlockDetection, DetectionResult
from block_position_prediction.data_collection.geometry import SheetConfig
from block_position_prediction.data_collection.labels import (
    build_tactile_pose_label,
    detection_polygon_taxel,
    live_preview_from_detection,
    preview_from_label,
)
from block_position_prediction.data_collection.preview_render import draw_tactile_preview


def test_draw_tactile_preview_adds_fixed_bbox_homography_footprint_pixels():
    config = SheetConfig()
    calibration = _identity_calibration()
    top_left = config.sensor_to_paper_mm(5.5, 5.5)
    bottom_right = config.sensor_to_paper_mm(13.5, 13.5)
    center = config.sensor_to_paper_mm(9.5, 9.5)
    detection = _detection(
        xyxy=(
            int(round(top_left[0])),
            int(round(top_left[1])),
            int(round(bottom_right[0])),
            int(round(bottom_right[1])),
        ),
        anchor_px=center,
    )

    preview = live_preview_from_detection(config, calibration, detection, tactile=None)
    heatmap = draw_tactile_preview(preview, config=config, width=320)

    assert int(heatmap[:, :, 2].max()) == 255
    assert int(heatmap[:, :, 0].max()) == 255
    assert preview.pose.available is True
    assert preview.pose.source == "bbox_homography"
    assert preview.position_taxel is not None


def test_detection_sensor_polygon_preserves_block_angle():
    config = SheetConfig()
    calibration = _identity_calibration()
    top_left = config.sensor_to_paper_mm(20.0, 10.0)
    bottom_right = config.sensor_to_paper_mm(40.0, 22.0)
    center = config.sensor_to_paper_mm(30.0, 16.0)
    detection = _detection(
        xyxy=(
            int(round(top_left[0])),
            int(round(top_left[1])),
            int(round(bottom_right[0])),
            int(round(bottom_right[1])),
        ),
        anchor_px=center,
        angle_deg=30.0,
    )

    assert detection.best is not None
    polygon = detection_polygon_taxel(config, calibration, detection.best)

    assert polygon is not None
    top_edge = (polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1])
    side_edge = (polygon[3][0] - polygon[0][0], polygon[3][1] - polygon[0][1])
    assert abs(top_edge[1]) > 1.0
    assert abs(side_edge[0]) > 1.0


def test_detection_sensor_polygon_preserves_calibration_rotation():
    config = SheetConfig()
    angle = math.radians(30.0)
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)
    sensor_origin_x, sensor_origin_y = config.sensor_origin_mm
    tx = sensor_origin_x + 24.0
    ty = sensor_origin_y + 10.0
    image_to_paper = ((cos_t, -sin_t, tx), (sin_t, cos_t, ty), (0.0, 0.0, 1.0))
    calibration = PaperCalibration(
        image_to_paper=image_to_paper,
        paper_to_image=_invert_matrix(image_to_paper),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
    detection = _detection(xyxy=(0, 0, 16, 10), anchor_px=(8.0, 5.0))

    assert detection.best is not None
    polygon = detection_polygon_taxel(config, calibration, detection.best)

    assert polygon is not None
    top_edge = (polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1])
    side_edge = (polygon[3][0] - polygon[0][0], polygon[3][1] - polygon[0][1])
    assert abs(top_edge[1]) > 1.0
    assert abs(side_edge[0]) > 1.0


def test_current_label_and_preview_label_have_same_target_rendering():
    config = SheetConfig()
    calibration = PaperCalibration(
        image_to_paper=((1, 0, config.sensor_origin_mm[0]), (0, 1, config.sensor_origin_mm[1]), (0, 0, 1)),
        paper_to_image=((1, 0, -config.sensor_origin_mm[0]), (0, 1, -config.sensor_origin_mm[1]), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
    detection = _detection(xyxy=(20, 10, 40, 22), anchor_px=(30.0, 16.0), angle_deg=20.0)

    live_preview = live_preview_from_detection(config, calibration, detection, tactile=None)
    label_preview = preview_from_label(build_tactile_pose_label(config, calibration, detection), config=config)

    assert live_preview.position_taxel == label_preview.position_taxel
    assert live_preview.pose.footprint_corners_taxel == label_preview.pose.footprint_corners_taxel


def _identity_calibration() -> PaperCalibration:
    return PaperCalibration(
        image_to_paper=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        paper_to_image=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )


def _detection(
    xyxy: tuple[int, int, int, int],
    anchor_px: tuple[float, float],
    angle_deg: float | None = None,
    polygon_px: tuple[tuple[float, float], ...] | None = None,
) -> DetectionResult:
    x1, y1, x2, y2 = xyxy
    return DetectionResult(
        frame_id=1,
        timestamp=0.0,
        detections=(
            BlockDetection(
                xyxy=xyxy,
                center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                confidence=0.9,
                class_id=0,
                class_name="block",
                anchor_px=anchor_px,
                angle_deg=angle_deg,
                polygon_px=polygon_px,
            ),
        ),
        elapsed_ms=1.0,
    )


def _invert_matrix(matrix):
    return tuple(tuple(float(value) for value in row) for row in np.linalg.inv(np.asarray(matrix, dtype=np.float64)))
