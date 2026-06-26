from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_sensor.tactile_visualization import (  # noqa: E402
    draw_tactile_heatmap,
    draw_tactile_prediction,
    normalize_tactile_frame,
)


def test_normalize_tactile_frame_auto_scales_raw_and_flips_columns():
    frame = np.zeros((8, 16), dtype=np.float32)
    frame[0, 0] = 65535.0

    normalized = normalize_tactile_frame(frame)

    assert float(normalized[0, 15]) == 1.0
    assert float(normalized[0, 0]) == 0.0


def test_draw_tactile_heatmap_marks_high_value_red():
    frame = np.zeros((8, 16), dtype=np.float32)
    frame[0, 0] = 1.0

    heatmap = draw_tactile_heatmap(
        frame,
        width=160,
        value_max=1.0,
        flip_x=False,
    )

    assert heatmap.shape == (124, 160, 3)
    assert tuple(int(value) for value in heatmap[45, 1]) == (0, 0, 255)


def test_draw_tactile_prediction_adds_result_and_center_marker():
    frame = np.zeros((8, 16), dtype=np.float32)

    image = draw_tactile_prediction(
        frame,
        detected=True,
        position_taxel=(0.0, 0.0),
        angle_deg=12.5,
        confidence=0.9,
        fully_inside_sensor=True,
        width=160,
    )

    assert image.shape == (188, 160, 3)
    assert tuple(int(value) for value in image[49, 5]) == (0, 0, 255)
    assert np.any(image[124:] != 20)
