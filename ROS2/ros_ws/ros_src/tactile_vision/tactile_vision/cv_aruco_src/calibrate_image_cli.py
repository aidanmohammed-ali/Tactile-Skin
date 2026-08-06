from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_image
from .config import ArucoBoardConfig, HoleRefineConfig
from .overlay import save_board_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a still image with ArUco markers.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output calibration JSON path.")
    parser.add_argument("--overlay", help="Optional output overlay image path.")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--hole-pitch-mm", type=float, default=25.0)
    parser.add_argument("--marker-size-grid", type=float, default=1.6)
    parser.add_argument("--marker-margin-grid", type=float, default=-1.40)
    parser.add_argument("--no-hole-refine", action="store_true")
    args = parser.parse_args()

    config = ArucoBoardConfig(
        hole_pitch_mm=args.hole_pitch_mm,
        aruco_dictionary=args.dictionary,
        marker_size_grid=args.marker_size_grid,
        marker_margin_grid=args.marker_margin_grid,
        refine_holes=HoleRefineConfig(enabled=not args.no_hole_refine),
    )
    calibration = calibrate_image(args.image, config)
    calibration.save_json(args.output)
    print(args.output)
    print(f"method: {calibration.metadata.get('method')}")
    print(f"marker_ids: {calibration.marker_ids}")
    print(f"refined_holes: {calibration.quality.hole_point_count}")
    print(f"median_grid_error: {calibration.quality.median_grid_error}")
    print(f"median_pixel_error: {calibration.quality.median_pixel_error}")
    if args.overlay:
        print(f"overlay: {save_board_overlay(args.image, Path(args.output), args.overlay)}")


if __name__ == "__main__":
    main()
