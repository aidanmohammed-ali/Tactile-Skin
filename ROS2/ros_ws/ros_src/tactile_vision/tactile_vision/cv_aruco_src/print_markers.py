from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_marker_sheet(
    output_path: str | Path,
    dictionary_name: str = "DICT_4X4_50",
    marker_ids: tuple[int, ...] = (0, 1, 2, 3),
    marker_size_mm: float = 40.0,
    quiet_zone_mm: float = 8.0,
    dpi: int = 300,
) -> Path:
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    output_path = Path(output_path)
    px_per_mm = dpi / 25.4
    a4_w = int(round(210 * px_per_mm))
    a4_h = int(round(297 * px_per_mm))
    marker_px = int(round(marker_size_mm * px_per_mm))
    quiet_px = int(round(quiet_zone_mm * px_per_mm))
    block_w = marker_px + quiet_px * 2
    block_h = marker_px + quiet_px * 2 + int(round(12 * px_per_mm))
    sheet = np.full((a4_h, a4_w), 255, dtype=np.uint8)
    positions = [
        (int(round(20 * px_per_mm)), int(round(25 * px_per_mm))),
        (int(round(115 * px_per_mm)), int(round(25 * px_per_mm))),
        (int(round(115 * px_per_mm)), int(round(150 * px_per_mm))),
        (int(round(20 * px_per_mm)), int(round(150 * px_per_mm))),
    ]
    names = ("top-left", "top-right", "bottom-right", "bottom-left")
    for marker_id, position, name in zip(marker_ids, positions, names):
        marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, int(marker_id), marker_px, marker, 1)
        x, y = position
        cv2.rectangle(sheet, (x, y), (x + block_w, y + block_h), 0, 2)
        marker_x = x + quiet_px
        marker_y = y + quiet_px
        sheet[marker_y : marker_y + marker_px, marker_x : marker_x + marker_px] = marker
        cv2.putText(
            sheet,
            f"ID {marker_id}  {name}",
            (x + quiet_px, y + block_h - int(round(4 * px_per_mm))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            0,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        sheet,
        f"{dictionary_name} | marker {marker_size_mm:g} mm | print at {dpi} DPI / 100% scale",
        (int(round(20 * px_per_mm)), int(round(285 * px_per_mm))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        0,
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix or ".png", sheet)
    if not ok:
        raise RuntimeError(f"could not encode marker sheet: {output_path}")
    encoded.tofile(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable ArUco marker sheet.")
    parser.add_argument("--output", default="cv_aruco_src/aruco_4x4_50_ids_0_3_a4.png")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-mm", type=float, default=40.0)
    parser.add_argument("--quiet-mm", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    print(
        generate_marker_sheet(
            args.output,
            dictionary_name=args.dictionary,
            marker_size_mm=args.marker_mm,
            quiet_zone_mm=args.quiet_mm,
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()
