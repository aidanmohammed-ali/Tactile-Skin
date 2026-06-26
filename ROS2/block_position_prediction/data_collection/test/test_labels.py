import numpy as np

from block_position_prediction.data_collection.aruco import PaperCalibration
from block_position_prediction.data_collection.detector import BlockDetection, DetectionResult
from block_position_prediction.data_collection.geometry import SheetConfig
from block_position_prediction.data_collection.labels import build_tactile_pose_label, preview_from_label
from block_position_prediction.data_collection.preview_dataset import draw_dataset_item
from block_position_prediction.data_collection.tactile import NUM_TAXELS, TactileSnapshot


def test_pose_label_uses_fixed_square_for_angle_detection():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    detection = DetectionResult(
        frame_id=1,
        timestamp=2.0,
        detections=(
            BlockDetection(
                xyxy=(20, 10, 40, 22),
                center_px=(30.0, 16.0),
                confidence=0.8,
                class_id=0,
                class_name="block",
                anchor_px=(30.0, 16.0),
                angle_deg=25.0,
            ),
        ),
        elapsed_ms=1.0,
    )

    label = build_tactile_pose_label(config, calibration, detection)
    pose = label["target"]["pose"]

    assert pose["available"] is True
    assert pose["footprint_side_taxel"] == 6.0
    assert len(pose["footprint_corners_taxel"]) == 4
    assert len(pose["yaw_mod90_vector"]) == 2


def test_bbox_only_pose_uses_homography_angle_to_sensor_frame():
    config = SheetConfig()
    angle = np.deg2rad(30.0)
    ox, oy = config.sensor_origin_mm
    calibration = PaperCalibration(
        image_to_paper=((float(np.cos(angle)), float(-np.sin(angle)), ox + 20.0), (float(np.sin(angle)), float(np.cos(angle)), oy + 10.0), (0, 0, 1)),
        paper_to_image=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
    detection = DetectionResult(
        frame_id=1,
        timestamp=2.0,
        detections=(
            BlockDetection(
                xyxy=(0, 0, 16, 8),
                center_px=(8.0, 4.0),
                confidence=0.8,
                class_id=0,
                class_name="block",
                anchor_px=(8.0, 4.0),
            ),
        ),
        elapsed_ms=1.0,
    )

    label = build_tactile_pose_label(config, calibration, detection)
    pose = label["target"]["pose"]

    assert pose["available"] is True
    assert pose["source"] == "bbox_homography"
    assert 0.1 < pose["yaw_mod90_rad"] < 1.4


def test_preview_from_legacy_label_canonicalizes_tactile_values():
    values = np.arange(NUM_TAXELS, dtype=np.float32) / float(NUM_TAXELS)
    legacy = {
        "image_path": "images/000001.jpg",
        "sensor_data": {"top5_normalized": values.tolist()},
        "position": {"array_col_row": [3.0, 4.0]},
        "detection": {"confidence": 0.7},
    }

    preview = preview_from_label(legacy)

    assert preview.values is not None
    assert float(preview.values[0]) == float(values[15])
    assert preview.position_taxel == (3.0, 4.0)
    assert preview.pose.available is False


def test_preview_from_current_label_preserves_pose_target():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    tactile = TactileSnapshot(
        timestamp=1.0,
        port="SIMULATOR",
        hardware_online=False,
        status="simulation",
        error=None,
        tared=False,
        processed=None,
        top5_normalized=np.zeros(NUM_TAXELS, dtype=np.float32),
        top5_raw_average=np.zeros(NUM_TAXELS, dtype=np.float32),
        recent_raw_frames=(),
    )
    detection = DetectionResult(
        frame_id=1,
        timestamp=2.0,
        detections=(
            BlockDetection(
                xyxy=(20, 10, 40, 22),
                center_px=(30.0, 16.0),
                confidence=0.8,
                class_id=0,
                class_name="block",
                anchor_px=(30.0, 16.0),
                angle_deg=10.0,
            ),
        ),
        elapsed_ms=1.0,
    )

    preview = preview_from_label(build_tactile_pose_label(config, calibration, detection, tactile, sample_id="x"))

    assert preview.values is not None
    assert preview.position_taxel == (6.5, 3.0)
    assert preview.pose.available is True


def test_draw_dataset_item_renders_current_label_without_window():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    detection = DetectionResult(
        frame_id=1,
        timestamp=2.0,
        detections=(
            BlockDetection(
                xyxy=(20, 10, 40, 22),
                center_px=(30.0, 16.0),
                confidence=0.8,
                class_id=0,
                class_name="block",
                anchor_px=(30.0, 16.0),
            ),
        ),
        elapsed_ms=1.0,
    )

    image = draw_dataset_item(build_tactile_pose_label(config, calibration, detection), index=0, total=1, config=config, width=160)

    assert image.shape[1] == 160
    assert image.shape[0] > 120


def _calibration_at_sensor_origin(config: SheetConfig) -> PaperCalibration:
    ox, oy = config.sensor_origin_mm
    return PaperCalibration(
        image_to_paper=((1, 0, ox), (0, 1, oy), (0, 0, 1)),
        paper_to_image=((1, 0, -ox), (0, 1, -oy), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
