"""Spatial filters for tactile sensor frames."""

from __future__ import annotations

import numpy as np


TACTILE_ROWS = 8
TACTILE_COLS = 16
TACTILE_TAXELS = TACTILE_ROWS * TACTILE_COLS


def median_filter_3x3(values) -> np.ndarray:
    """Apply a 3x3 median filter while preserving tactile frame shape."""

    array = np.asarray(values, dtype=np.float32)
    if array.size != TACTILE_TAXELS:
        raise ValueError(
            f"expected {TACTILE_TAXELS} tactile values, got {array.size}"
        )

    grid = array.reshape(TACTILE_ROWS, TACTILE_COLS)
    padded = np.pad(grid, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.median(windows, axis=(-2, -1)).astype(np.float32).reshape(
        TACTILE_TAXELS
    )


def mean_tactile_frames(frames) -> np.ndarray:
    """Average tactile frames after their per-frame spatial filtering."""

    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != TACTILE_TAXELS:
        raise ValueError(
            "expected tactile frames with shape "
            f"(count, {TACTILE_TAXELS}), got {array.shape}"
        )
    if array.shape[0] < 1:
        raise ValueError("expected at least one tactile frame")
    return np.mean(array, axis=0).astype(np.float32)


def subtract_tare_baseline(values, baseline, scale: float = 1.0) -> np.ndarray:
    """Subtract a per-taxel tare baseline and clamp pressure below zero."""

    array = np.asarray(values, dtype=np.float32)
    baseline_array = np.asarray(baseline, dtype=np.float32)
    if array.size != TACTILE_TAXELS:
        raise ValueError(
            f"expected {TACTILE_TAXELS} tactile values, got {array.size}"
        )
    if baseline_array.size != TACTILE_TAXELS:
        raise ValueError(
            "expected "
            f"{TACTILE_TAXELS} tare values, got {baseline_array.size}"
        )
    return np.maximum(
        array.reshape(TACTILE_TAXELS)
        - baseline_array.reshape(TACTILE_TAXELS) * float(scale),
        0.0,
    ).astype(np.float32)
