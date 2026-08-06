import numpy as np

from block_position_prediction.data_collection_manual.aruco import PaperCalibration
from block_position_prediction.data_collection_manual.geometry import SheetConfig
from block_position_prediction.data_collection_manual.labels import (
    create_manual_annotation,
    preview_from_annotation,
)
from block_position_prediction.data_collection_manual.overlay import UiStatus, draw_overlay
from block_position_prediction.data_collection_manual.preview_render import draw_tactile_preview
from block_position_prediction.data_collection_manual.tactile import NUM_TAXELS


def test_draw_tactile_preview_adds_manual_footprint_pixels():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))

    preview = preview_from_annotation(annotation, tactile_values=np.zeros(NUM_TAXELS, dtype=np.float32))
    heatmap = draw_tactile_preview(preview, config=config, width=320)

    assert int(heatmap[:, :, 2].max()) == 255
    assert int(heatmap[:, :, 0].max()) == 255
    assert preview.pose.available is True
    assert preview.pose.source == "manual_aruco"
    assert preview.position_taxel == (6.5, 3.0)


def test_draw_overlay_renders_manual_annotation_and_side_panel():
    config = SheetConfig()
    calibration = _calibration_at_sensor_origin(config)
    annotation = create_manual_annotation(config, calibration, center_px=(30.0, 16.0), direction_px=(40.0, 16.0))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    image = draw_overlay(
        frame,
        config,
        calibration,
        annotation,
        UiStatus(mode="draft", dataset="test", sample_position="new draft"),
        tactile_values=np.zeros(NUM_TAXELS, dtype=np.float32),
    )

    assert image.shape[0] == 240
    assert image.shape[1] == 680
    assert int(image[:, :, 2].max()) > 0


def _calibration_at_sensor_origin(config: SheetConfig) -> PaperCalibration:
    ox, oy = config.sensor_origin_mm
    return PaperCalibration(
        image_to_paper=((1, 0, ox), (0, 1, oy), (0, 0, 1)),
        paper_to_image=((1, 0, -ox), (0, 1, -oy), (0, 0, 1)),
        marker_ids=(0, 1, 2, 3),
        dictionary="DICT_4X4_50",
    )
