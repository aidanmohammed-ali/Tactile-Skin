from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .geometry import SheetConfig
from .labels import preview_from_label
from .preview_render import draw_tactile_preview


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    labels_path = _labels_path(path)
    labels: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                labels.append(json.loads(text))
    return labels


def draw_dataset_item(
    label: Mapping[str, Any],
    index: int,
    total: int,
    config: SheetConfig | None = None,
    width: int = 480,
) -> Any:
    import cv2
    import numpy as np

    config = config or SheetConfig()
    preview = preview_from_label(label, config=config)
    heatmap = draw_tactile_preview(preview, config=config, width=width)
    footer_h = 112
    canvas = np.zeros((heatmap.shape[0] + footer_h, heatmap.shape[1], 3), dtype=np.uint8)
    canvas[: heatmap.shape[0], : heatmap.shape[1]] = heatmap
    canvas[heatmap.shape[0] :] = (18, 18, 18)

    sample_id = label.get("sample_id") or label.get("image_path") or "sample"
    target = label.get("target") or {}
    quality = label.get("quality") or {}
    pose = target.get("pose") or {}
    position = target.get("position_taxel") or (label.get("position") or {}).get("array_col_row")
    pos_text = "pos=unavailable" if position is None else f"pos=({float(position[0]):.2f},{float(position[1]):.2f}) taxel"
    pose_text = f"pose={pose.get('source') or 'available'}" if pose.get("available") else "pose=unavailable"
    source_text = f"source={quality.get('label_source') or pose.get('source') or 'unknown'}"
    inside_text = ""
    if pose.get("fully_inside_sensor") is not None:
        inside_text = "inside" if pose.get("fully_inside_sensor") else "outside"

    lines = [
        f"{index + 1}/{total}  {sample_id}  {label.get('schema_version', 'legacy')}",
        f"{pos_text}  {pose_text}  {source_text}  {inside_text}",
        "N/Right/Space next | P/Left previous | Q/Esc quit",
    ]
    y = heatmap.shape[0] + 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        y += 30
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview tactile data collection labels in sequence.")
    parser.add_argument("run_dir", help="Run directory or labels.jsonl path.")
    parser.add_argument("--width", type=int, default=480, help="Preview heatmap width in pixels.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based sample index to show first.")
    parser.add_argument("--window", default="Dataset Preview")
    return parser


def main() -> None:
    import cv2

    args = build_parser().parse_args()
    labels = load_labels(args.run_dir)
    if not labels:
        raise SystemExit(f"no labels found in {_labels_path(args.run_dir)}")

    index = max(0, min(int(args.start), len(labels) - 1))
    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    try:
        while True:
            cv2.imshow(args.window, draw_dataset_item(labels[index], index, len(labels), width=args.width))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord(" "), 83):
                index = min(len(labels) - 1, index + 1)
            elif key in (ord("p"), 81):
                index = max(0, index - 1)
    finally:
        cv2.destroyWindow(args.window)


def _labels_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / "labels.jsonl"
    return candidate


if __name__ == "__main__":
    main()
