import numpy as np

from block_position_prediction.data_collection_manual.aruco import PaperCalibration
from block_position_prediction.data_collection_manual.geometry import BLOCK_SIDE_TAXEL, SheetConfig
from block_position_prediction.data_collection_manual.labels import (
    LABEL_SOURCE,
    NO_BLOCK_LABEL_SOURCE,
    annotation_from_label,
    build_manual_tactile_pose_label,
    build_no_block_tactile_pose_label,
    create_manual_annotation,
    create_manual_annotation_from_taxel_pose,
    preview_from_label,
    validate_manual_sample,
)
from block_position_prediction.data_collection_manual.preview_dataset import draw_dataset_item
from block_position_prediction.data_collection_manual.tactile import NUM_TAXELS, TactileSnapshot


def test_manual_annotation_maps_center_and_direction_to_sensor_frame():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)

    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 20.0), direction_px=(40.0, 30.0))

    assert annotation.center_sensor_mm == (30.0, 20.0)
    assert annotation.center_taxel == (6.5, 4.0)
    assert 0.7 < annotation.yaw_mod90_rad < 0.9
    assert annotation.fully_inside_sensor
    assert len(annotation.footprint_corners_taxel) == 4
    assert len(annotation.footprint_corners_px) == 4


def test_manual_label_keeps_tactile_pose_schema_and_source():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    top5 = np.arange(NUM_TAXELS, dtype=np.float32) / float(NUM_TAXELS)
    tactile = _tactile(top5)

    label = build_manual_tactile_pose_label(config, calibration, annotation, tactile, frame_id=3, timestamp=123.0)

    assert label["schema_version"] == "tactile_pose_v1"
    assert label["frame_id"] == 3
    assert label["input"]["frame"] == "taxel_center_v1"
    assert len(label["input"]["values"]) == NUM_TAXELS
    assert label["input"]["values"][0] == float(top5[15])
    assert label["target"]["position_taxel"] == [6.5, 3.0]
    assert label["target"]["object_present"] is True
    assert label["target"]["pose"]["available"] is True
    assert label["target"]["pose"]["source"] == LABEL_SOURCE
    assert label["target"]["pose"]["footprint_side_taxel"] == BLOCK_SIDE_TAXEL
    assert label["quality"]["label_source"] == LABEL_SOURCE
    assert label["quality"]["confidence"] is None
    assert label["annotation"]["block_side_mm"] == 24.0
    assert label["calibration"]["image_to_paper"] == [[1, 0, 114.5], [0, 1, 85.0], [0, 0, 1]]


def test_no_block_label_has_no_pose_target():
    config = SheetConfig()
    top5 = np.arange(NUM_TAXELS, dtype=np.float32) / float(NUM_TAXELS)
    label = build_no_block_tactile_pose_label(config, None, _tactile(top5), frame_id=4, timestamp=456.0)

    preview = preview_from_label(label)

    assert label["target"]["object_present"] is False
    assert label["target"]["position_taxel"] is None
    assert label["target"]["pose"]["available"] is False
    assert label["quality"]["label_source"] == NO_BLOCK_LABEL_SOURCE
    assert label["calibration"] is None
    assert annotation_from_label(label) is None
    assert preview.position_taxel is None


def test_annotation_round_trips_from_manual_label():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    label = build_manual_tactile_pose_label(config, calibration, annotation, _tactile(np.zeros(NUM_TAXELS)))

    restored = annotation_from_label(label)

    assert restored is not None
    assert restored.center_px == annotation.center_px
    assert restored.direction_px == annotation.direction_px
    assert restored.footprint_corners_taxel == annotation.footprint_corners_taxel


def test_manual_annotation_can_be_rebuilt_from_taxel_pose_for_keyboard_tuning():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)

    annotation = create_manual_annotation_from_taxel_pose(config, calibration, center_taxel=(6.5, 3.0), yaw_mod90_rad=0.25)

    assert annotation.center_taxel == (6.5, 3.0)
    assert abs(annotation.yaw_mod90_rad - 0.25) < 1e-8
    assert annotation.direction_px != annotation.center_px


def test_preview_and_dataset_render_use_manual_target():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    label = build_manual_tactile_pose_label(config, calibration, annotation, _tactile(np.zeros(NUM_TAXELS)), sample_id="x")

    preview = preview_from_label(label)
    image = draw_dataset_item(label, index=0, total=1, config=config, width=160)

    assert preview.position_taxel == (6.5, 3.0)
    assert preview.pose.source == LABEL_SOURCE
    assert image.shape[1] == 160
    assert image.shape[0] > 120


def test_save_validation_only_rejects_missing_label_prerequisites():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    tactile = _tactile(np.zeros(NUM_TAXELS))
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    outside = create_manual_annotation(config, calibration, center_px=(4.0, 4.0), direction_px=(14.0, 4.0))

    assert validate_manual_sample(config, None, annotation, tactile) == "not calibrated"
    assert validate_manual_sample(config, calibration, None, tactile) == "no manual annotation"
    assert validate_manual_sample(config, calibration, annotation, None) is None
    assert validate_manual_sample(config, calibration, outside, tactile) is None
    assert validate_manual_sample(config, calibration, annotation, tactile) is None


def _calibration_at_sensor_origin(config: SheetConfig) -> PaperCalibration:
    ox, oy = config.sensor_origin_mm
    return PaperCalibration(
        image_to_paper=((1, 0, ox), (0, 1, oy), (0, 0, 1)),
        paper_to_image=((1, 0, -ox), (0, 1, -oy), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )


def _tactile(values):
    return TactileSnapshot(
        timestamp=1.0,
        port="SIMULATOR",
        hardware_online=False,
        status="simulation",
        error=None,
        tared=False,
        processed=None,
        top5_normalized=np.asarray(values, dtype=np.float32),
        top5_raw_average=np.zeros(NUM_TAXELS, dtype=np.float32),
        recent_raw_frames=(),
    )
