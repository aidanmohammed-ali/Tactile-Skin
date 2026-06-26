from __future__ import annotations

import argparse
import json
from pathlib import Path

from .geometry import MarkerLayout, SensorArrayConfig, SheetConfig


def generate_calibration_sheet(
    output_path: str | Path,
    config: SheetConfig | None = None,
    dpi: int = 300,
    metadata_path: str | Path | None = None,
) -> Path:
    import cv2
    import numpy as np

    config = config or SheetConfig()
    output_path = Path(output_path)
    px_per_mm = float(dpi) / 25.4
    width_px = int(round(config.paper_width_mm * px_per_mm))
    height_px = int(round(config.paper_height_mm * px_per_mm))
    sheet = np.full((height_px, width_px), 255, dtype=np.uint8)

    dictionary_id = getattr(cv2.aruco, config.marker_layout.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {config.marker_layout.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    def pt(point_mm: tuple[float, float]) -> tuple[int, int]:
        return (int(round(point_mm[0] * px_per_mm)), int(round(point_mm[1] * px_per_mm)))

    x1, y1, x2, y2 = config.sensor_rect_mm
    cv2.rectangle(sheet, pt((x1, y1)), pt((x2, y2)), 0, 3)
    cv2.putText(
        sheet,
        "SENSOR 68x40 mm",
        pt((x1 + 2.0, y1 - 3.0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        2,
        cv2.LINE_AA,
    )
    for row in range(config.sensor.rows):
        for col in range(config.sensor.cols):
            center = config.sensor.taxel_center_mm(col, row)
            cv2.circle(sheet, pt(config.sensor_to_paper_mm(*center)), max(2, int(round(0.5 * px_per_mm))), 0, 1)

    marker_px = int(round(config.marker_layout.marker_size_mm * px_per_mm))
    quiet_px = int(round(config.marker_layout.quiet_zone_mm * px_per_mm))
    for spec in config.marker_specs():
        marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, int(spec.marker_id), marker_px, marker, 1)
        top_left = spec.paper_corners_mm[0]
        x = int(round(top_left[0] * px_per_mm))
        y = int(round(top_left[1] * px_per_mm))
        block_x1 = x - quiet_px
        block_y1 = y - quiet_px
        block_x2 = x + marker_px + quiet_px
        block_y2 = y + marker_px + quiet_px
        cv2.rectangle(sheet, (block_x1, block_y1), (block_x2, block_y2), 0, 2)
        sheet[y : y + marker_px, x : x + marker_px] = marker
        cv2.putText(
            sheet,
            f"ID {spec.marker_id}",
            (block_x1, block_y2 + int(round(5 * px_per_mm))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            0,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        sheet,
        f"{config.marker_layout.dictionary} | print at {dpi} DPI / 100% scale",
        pt((12.0, config.paper_height_mm - 8.0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        0,
        2,
        cv2.LINE_AA,
    )

    _save_sheet(output_path, sheet, dpi)

    metadata = Path(metadata_path) if metadata_path else output_path.with_suffix(".json")
    with metadata.open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def _save_sheet(output_path: Path, sheet, dpi: int) -> None:
    suffix = output_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".pdf":
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to write PDF calibration sheets") from exc
        Image.fromarray(sheet).convert("RGB").save(output_path, "PDF", resolution=float(dpi))
        return

    import cv2

    ok, encoded = cv2.imencode(suffix or ".png", sheet)
    if not ok:
        raise RuntimeError(f"could not encode calibration sheet: {output_path}")
    encoded.tofile(str(output_path))


def build_config_from_args(args: argparse.Namespace) -> SheetConfig:
    sensor = SensorArrayConfig(
        width_mm=args.sensor_width_mm,
        height_mm=args.sensor_height_mm,
        cols=args.sensor_cols,
        rows=args.sensor_rows,
        pitch_mm=args.sensor_pitch_mm,
        left_margin_mm=args.sensor_left_margin_mm,
        top_margin_mm=args.sensor_top_margin_mm,
    )
    marker_layout = MarkerLayout(
        marker_size_mm=args.marker_mm,
        quiet_zone_mm=args.quiet_mm,
        gap_from_sensor_mm=args.marker_gap_mm,
        ids=tuple(args.marker_ids),
        dictionary=args.dictionary,
    )
    return SheetConfig(
        paper_width_mm=args.paper_width_mm,
        paper_height_mm=args.paper_height_mm,
        sensor=sensor,
        marker_layout=marker_layout,
    )


def add_geometry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-width-mm", type=float, default=297.0)
    parser.add_argument("--paper-height-mm", type=float, default=210.0)
    parser.add_argument("--sensor-width-mm", type=float, default=68.0)
    parser.add_argument("--sensor-height-mm", type=float, default=40.0)
    parser.add_argument("--sensor-cols", type=int, default=16)
    parser.add_argument("--sensor-rows", type=int, default=8)
    parser.add_argument("--sensor-pitch-mm", type=float, default=4.0)
    parser.add_argument("--sensor-left-margin-mm", type=float, default=4.0)
    parser.add_argument("--sensor-top-margin-mm", type=float, default=4.0)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-ids", type=int, nargs=4, default=(0, 1, 2, 3))
    parser.add_argument("--marker-mm", type=float, default=20.0)
    parser.add_argument("--quiet-mm", type=float, default=4.0)
    parser.add_argument("--marker-gap-mm", type=float, default=10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable A4 ArUco sensor calibration sheet.")
    parser.add_argument(
        "--output",
        default="block_position_prediction/data_collection/assets/a4_sensor_calibration.pdf",
        help="Output path; use .pdf for printable A4 PDF or an image suffix such as .png.",
    )
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    add_geometry_args(parser)
    args = parser.parse_args()
    print(generate_calibration_sheet(args.output, build_config_from_args(args), dpi=args.dpi, metadata_path=args.metadata))


if __name__ == "__main__":
    main()
