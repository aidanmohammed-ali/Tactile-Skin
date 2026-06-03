"""Fixed-camera ArUco calibration and fast pixel-to-board mapping."""

from .calibration import ArucoBoardCalibrator, calibrate_image
from .config import (
    ArucoBoardConfig,
    BoardGeometry,
    HoleRefineConfig,
    MarkerSpec,
    default_marker_specs,
    marker_specs_from_centers,
)
from .overlay import draw_board_overlay, save_board_overlay
from .transform import BoardCalibration, board_to_pixel, load_calibration, pixel_to_board, save_calibration

__all__ = [
    "ArucoBoardCalibrator",
    "ArucoBoardConfig",
    "BoardCalibration",
    "BoardGeometry",
    "HoleRefineConfig",
    "MarkerSpec",
    "board_to_pixel",
    "calibrate_image",
    "default_marker_specs",
    "draw_board_overlay",
    "load_calibration",
    "marker_specs_from_centers",
    "pixel_to_board",
    "save_board_overlay",
    "save_calibration",
]
