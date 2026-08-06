import numpy as np

from block_position_prediction.data_collection.aruco import ArucoPaperCalibrator
from block_position_prediction.data_collection.geometry import SheetConfig


class SyntheticCalibrator(ArucoPaperCalibrator):
    def _detect_correspondences(self, gray):
        image_points = []
        paper_points = []
        marker_ids = []
        for spec in self.sheet_config.marker_specs():
            marker_ids.append(spec.marker_id)
            for x_mm, y_mm in spec.paper_corners_mm:
                paper_points.append((x_mm, y_mm))
                image_points.append((10.0 + x_mm * 3.0, 20.0 + y_mm * 3.0))
        return image_points, paper_points, marker_ids


def test_synthetic_marker_homography_maps_pixels_to_paper_and_sensor():
    config = SheetConfig()
    calibration = SyntheticCalibrator(config).calibrate(np.zeros((720, 960), dtype=np.uint8))

    paper = calibration.image_to_paper_mm(10.0 + 118.5 * 3.0, 20.0 + 89.0 * 3.0)
    assert abs(paper[0] - 118.5) < 1e-6
    assert abs(paper[1] - 89.0) < 1e-6

    label = calibration.position_label(config, 10.0 + 118.5 * 3.0, 20.0 + 89.0 * 3.0)
    assert abs(label.sensor_mm[0] - 4.0) < 1e-6
    assert abs(label.sensor_mm[1] - 4.0) < 1e-6
    assert abs(label.array_col_row[0]) < 1e-6
    assert abs(label.array_col_row[1]) < 1e-6
    assert len(calibration.marker_ids) == 4
