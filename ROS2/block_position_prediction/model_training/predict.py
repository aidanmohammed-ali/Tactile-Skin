"""Run tactile pose inference for a labels.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import AugmentConfig, TactilePoseDataset, load_samples, sample_key
from .metrics import yaw_error_rad, yaw_from_vector_np
from .model import TactilePoseNet, load_tactile_pose_state
from .train import resolve_device


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    samples = load_samples(args.labels_jsonl)
    if not samples:
        raise SystemExit(f"No valid samples found in {args.labels_jsonl}")
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    dataset = TactilePoseDataset(
        samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=AugmentConfig(enabled=False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = resolve_device(args.device)
    model = TactilePoseNet().to(device)
    load_tactile_pose_state(model, checkpoint["model_state"])
    model.eval()

    rows: list[dict] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            maps = batch["maps"].to(device)
            physics = batch["physics"].to(device)
            output = model(maps, physics)
            pred_pos = output["position"].detach().cpu().numpy()
            pred_yaw_vec = output["yaw_vector"].detach().cpu().numpy()
            pred_yaw = yaw_from_vector_np(pred_yaw_vec)
            if getattr(model, "presence_available", True) and "presence_logit" in output:
                confidence = torch.sigmoid(output["presence_logit"]).detach().cpu().numpy()
            else:
                confidence = np.full(pred_pos.shape[0], np.nan, dtype=np.float32)
            target_pos = batch["position"].numpy()
            target_yaw = batch["yaw"].numpy()
            target_present = batch["object_present"].numpy().astype(bool)
            yaw_err = yaw_error_rad(pred_yaw, target_yaw)
            for i in range(pred_pos.shape[0]):
                sample = samples[offset + i]
                pos_error = pred_pos[i] - target_pos[i]
                position_error = None if not target_present[i] else pos_error.astype(float).tolist()
                position_distance = None if not target_present[i] else float(np.linalg.norm(pos_error))
                yaw_error = None if not target_present[i] else float(yaw_err[i])
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "sample_key": sample_key(sample),
                        "image_path": sample.image_path,
                        "target_object_present": bool(target_present[i]),
                        "predicted_object_confidence": None if np.isnan(confidence[i]) else float(confidence[i]),
                        "predicted_position_taxel": pred_pos[i].astype(float).tolist(),
                        "target_position_taxel": None if not target_present[i] else target_pos[i].astype(float).tolist(),
                        "position_error_taxel": position_error,
                        "position_error_distance_taxel": position_distance,
                        "predicted_yaw_mod90_rad": float(pred_yaw[i]),
                        "target_yaw_mod90_rad": None if not target_present[i] else float(target_yaw[i]),
                        "yaw_error_rad": yaw_error,
                    }
                )
            offset += pred_pos.shape[0]

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    main()

