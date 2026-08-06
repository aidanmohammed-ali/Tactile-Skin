import numpy as np
import pytest

from tactile_sensor.tactile_filtering import (
    TACTILE_COLS,
    TACTILE_ROWS,
    TACTILE_TAXELS,
    mean_tactile_frames,
    median_filter_3x3,
    subtract_tare_baseline,
)


def test_median_filter_3x3_removes_isolated_spike():
    values = np.zeros(TACTILE_TAXELS, dtype=np.float32)
    values[3 * TACTILE_COLS + 7] = 1.0

    filtered = median_filter_3x3(values)

    assert filtered.shape == (TACTILE_TAXELS,)
    assert np.count_nonzero(filtered) == 0


def test_median_filter_3x3_preserves_uniform_frame():
    values = np.full(TACTILE_TAXELS, 0.4, dtype=np.float32)

    filtered = median_filter_3x3(values)

    np.testing.assert_allclose(filtered, values)


def test_median_filter_3x3_uses_replicated_edges():
    values = np.zeros((TACTILE_ROWS, TACTILE_COLS), dtype=np.float32)
    values[0, 0] = 1.0
    values[0, 1] = 1.0

    filtered = median_filter_3x3(values)

    assert float(filtered[0]) == 1.0


def test_median_filter_3x3_rejects_wrong_frame_size():
    with pytest.raises(ValueError, match="expected 128 tactile values"):
        median_filter_3x3(np.zeros(127, dtype=np.float32))


def test_mean_tactile_frames_averages_already_filtered_frames():
    frames = np.stack(
        [
            np.full(TACTILE_TAXELS, value, dtype=np.float32)
            for value in range(5)
        ]
    )

    averaged = mean_tactile_frames(frames)

    np.testing.assert_allclose(averaged, 2.0)


def test_subtract_tare_baseline_removes_each_taxel_offset():
    baseline = np.linspace(0.0, 0.5, TACTILE_TAXELS, dtype=np.float32)
    values = baseline + 0.2

    centered = subtract_tare_baseline(values, baseline)

    np.testing.assert_allclose(centered, 0.2, atol=1e-6)


def test_subtract_tare_baseline_clamps_negative_pressure():
    baseline = np.full(TACTILE_TAXELS, 0.4, dtype=np.float32)
    values = np.full(TACTILE_TAXELS, 0.1, dtype=np.float32)

    centered = subtract_tare_baseline(values, baseline)

    assert np.count_nonzero(centered) == 0


def test_subtract_tare_baseline_supports_raw_value_scale():
    baseline = np.full(TACTILE_TAXELS, 0.25, dtype=np.float32)
    values = np.full(TACTILE_TAXELS, 32767.5, dtype=np.float32)

    centered = subtract_tare_baseline(values, baseline, scale=65535.0)

    np.testing.assert_allclose(centered, 16383.75)
