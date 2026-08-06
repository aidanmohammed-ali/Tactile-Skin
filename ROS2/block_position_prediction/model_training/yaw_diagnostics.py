"""Diagnose yaw errors for a trained tactile pose checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from block_position_prediction.data_collection_manual.geometry import (
    BLOCK_SIDE_TAXEL,
    SheetConfig,
    fixed_square_footprint,
)
from block_position_prediction.data_collection_manual.preview_render import (
    PreviewPose,
    TactilePreview,
    draw_tactile_preview,
)

from .dataset import AugmentConfig, PITCH_MM, TactilePoseDataset, sample_key, tactile_map_channels
from .metrics import yaw_error_rad, yaw_from_vector_np
from .model import TactilePoseNet, load_tactile_pose_state
from .train import dataset_from_checkpoint_subset, resolve_device


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config") or {}
    data_root = args.data_root or config.get("data_root") or "block_position_prediction/data_set"
    samples = dataset_from_checkpoint_subset(checkpoint, data_root, args.subset)
    samples = [sample for sample in samples if sample.object_present]
    if not samples:
        raise SystemExit(f"No present-object samples found for subset {args.subset!r} in {data_root}")

    rows = predict_rows(checkpoint, samples, args)
    rows.sort(key=lambda row: row["yaw_error_deg"], reverse=True)
    write_worst_csv(output_dir / "worst_yaw_samples.csv", rows)
    write_json(output_dir / "yaw_diagnostics.json", summarize(rows))
    write_worst_heatmaps(output_dir / "worst_yaw_heatmaps.png", rows[: max(1, args.top_k)])

    print(f"Wrote {output_dir / 'yaw_diagnostics.json'}")
    print(f"Wrote {output_dir / 'worst_yaw_samples.csv'}")
    print(f"Wrote {output_dir / 'worst_yaw_heatmaps.png'}")


def predict_rows(checkpoint: dict[str, Any], samples: list, args: argparse.Namespace) -> list[dict[str, Any]]:
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    dataset = TactilePoseDataset(
        samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=AugmentConfig(enabled=False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = resolve_device(args.device)
    model = TactilePoseNet().to(device)
    load_tactile_pose_state(model, checkpoint["model_state"])
    model.eval()

    rows: list[dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            maps = batch["maps"].to(device)
            physics = batch["physics"].to(device)
            output = model(maps, physics)
            pred_position = output["position"].detach().cpu().numpy()
            pred_yaw_vector = output["yaw_vector"].detach().cpu().numpy()
            pred_yaw = yaw_from_vector_np(pred_yaw_vector)
            target_position = batch["position"].numpy()
            target_yaw = batch["yaw"].numpy()
            yaw_error = yaw_error_rad(pred_yaw, target_yaw)
            for i in range(pred_position.shape[0]):
                sample = samples[offset + i]
                pos_error_taxel = float(np.linalg.norm(pred_position[i] - target_position[i]))
                force_sum = float(sample.values.sum())
                target_yaw_deg = float(np.degrees(target_yaw[i]))
                row = {
                    "sample_id": sample.sample_id,
                    "sample_key": sample_key(sample),
                    "image_path": sample.image_path,
                    "force_sum": force_sum,
                    "inside": bool(sample.fully_inside_sensor),
                    "position_region": position_region(target_position[i]),
                    "yaw_label_bin": yaw_label_bin(target_yaw_deg),
                    "target_position_taxel": target_position[i].astype(float).tolist(),
                    "predicted_position_taxel": pred_position[i].astype(float).tolist(),
                    "position_error_taxel": pos_error_taxel,
                    "position_error_mm": pos_error_taxel * PITCH_MM,
                    "target_yaw_deg": target_yaw_deg,
                    "predicted_yaw_deg": float(np.degrees(pred_yaw[i])),
                    "yaw_error_deg": float(np.degrees(yaw_error[i])),
                    "_values": sample.values.astype(np.float32),
                }
                rows.append(row)
            offset += pred_position.shape[0]
    add_force_bins(rows)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "overall": error_summary(rows),
        "by_inside": group_summary(rows, lambda row: "inside" if row["inside"] else "edge_or_outside"),
        "by_position_region": group_summary(rows, lambda row: row["position_region"]),
        "by_force_bin": group_summary(rows, lambda row: row["force_bin"]),
        "by_yaw_label_bin": group_summary(rows, lambda row: row["yaw_label_bin"]),
        "top_sample_keys": [row["sample_key"] for row in rows[:20]],
    }


def error_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = np.asarray([row["yaw_error_deg"] for row in rows], dtype=np.float32)
    pos_errors = np.asarray([row["position_error_taxel"] for row in rows], dtype=np.float32)
    if errors.size == 0:
        return {"count": 0.0}
    return {
        "count": float(errors.size),
        "yaw_mae_deg": float(errors.mean()),
        "yaw_p50_deg": float(np.percentile(errors, 50)),
        "yaw_p90_deg": float(np.percentile(errors, 90)),
        "yaw_max_deg": float(errors.max()),
        "position_mae_taxel": float(pos_errors.mean()),
    }


def group_summary(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row)), []).append(row)
    return {key: error_summary(value) for key, value in sorted(groups.items())}


def position_region(position: np.ndarray) -> str:
    x = float(position[0])
    y = float(position[1])
    if x < 0.0 or x > 15.0 or y < 0.0 or y > 7.0:
        return "center_outside_sensor"
    if x < 1.0 or x > 14.0 or y < 1.0 or y > 6.0:
        return "near_edge"
    return "inner"


def yaw_label_bin(yaw_deg: float) -> str:
    start = int(np.floor(float(yaw_deg) / 15.0) * 15)
    start = max(0, min(75, start))
    return f"{start:02d}_{start + 15:02d}deg"


def add_force_bins(rows: list[dict[str, Any]]) -> None:
    values = np.asarray([row["force_sum"] for row in rows], dtype=np.float32)
    if values.size == 0:
        return
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    for row in rows:
        force = row["force_sum"]
        if force <= q1:
            row["force_bin"] = "q1_low"
        elif force <= q2:
            row["force_bin"] = "q2"
        elif force <= q3:
            row["force_bin"] = "q3"
        else:
            row["force_bin"] = "q4_high"


def write_worst_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0].keys()) if public_rows else [])
        writer.writeheader()
        writer.writerows(public_rows)


def write_worst_heatmaps(path: Path, rows: list[dict[str, Any]]) -> None:
    import cv2

    if not rows:
        return
    sheet = SheetConfig()
    sample_images: list[np.ndarray] = []
    for rank, row in enumerate(rows, start=1):
        target_pos = tuple(row["target_position_taxel"])
        pred_pos = tuple(row["predicted_position_taxel"])
        target_yaw = float(np.radians(row["target_yaw_deg"]))
        pred_yaw = float(np.radians(row["predicted_yaw_deg"]))
        target = draw_tactile_preview(
            TactilePreview(
                values=row["_values"],
                position_taxel=target_pos,
                pose=PreviewPose(
                    available=True,
                    source="label",
                    yaw_mod90_rad=target_yaw,
                    footprint_side_taxel=BLOCK_SIDE_TAXEL,
                    footprint_corners_taxel=fixed_square_footprint(target_pos, target_yaw, BLOCK_SIDE_TAXEL),
                    fully_inside_sensor=row["inside"],
                ),
                title=f"#{rank} label yaw {row['target_yaw_deg']:.1f}",
            ),
            config=sheet,
            width=320,
        )
        predicted = draw_tactile_preview(
            TactilePreview(
                values=row["_values"],
                position_taxel=pred_pos,
                pose=PreviewPose(
                    available=True,
                    source="model",
                    yaw_mod90_rad=pred_yaw,
                    footprint_side_taxel=BLOCK_SIDE_TAXEL,
                    footprint_corners_taxel=fixed_square_footprint(pred_pos, pred_yaw, BLOCK_SIDE_TAXEL),
                    fully_inside_sensor=sheet.sensor.footprint_fully_inside_sensor(
                        fixed_square_footprint(pred_pos, pred_yaw, BLOCK_SIDE_TAXEL)
                    ),
                ),
                title=f"pred yaw {row['predicted_yaw_deg']:.1f}, err {row['yaw_error_deg']:.1f}",
            ),
            config=sheet,
            width=320,
        )
        combined = np.hstack([target, predicted])
        cv2.putText(
            combined,
            row["sample_id"],
            (12, combined.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        sample_images.append(combined)
    cv2.imwrite(str(path), np.vstack(sample_images))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--subset", choices=["val", "train", "all"], default="val")
    parser.add_argument("--output-dir", default="block_position_prediction/model_training/runs/yaw_diagnostics")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
