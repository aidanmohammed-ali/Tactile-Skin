"""Evaluate a trained tactile pose checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import AugmentConfig, TactilePoseDataset
from .metrics import format_metrics
from .model import TactilePoseNet, load_tactile_pose_state
from .train import dataset_from_checkpoint_subset, evaluate_model, resolve_device


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config") or {}
    data_root = args.data_root or config.get("data_root") or "block_position_prediction/data_set"
    samples = dataset_from_checkpoint_subset(checkpoint, data_root, args.subset)
    if not samples:
        raise SystemExit(f"No samples found for subset {args.subset!r} in {data_root}")
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    dataset = TactilePoseDataset(
        samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=AugmentConfig(enabled=False),
    )
    device = resolve_device(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = TactilePoseNet().to(device)
    load_tactile_pose_state(model, checkpoint["model_state"])
    report = evaluate_model(
        model,
        loader,
        device,
        yaw_weight=float(config.get("yaw_weight", 0.05)),
        presence_weight=float(config.get("presence_weight", 1.0)),
    )
    print(format_metrics(report["metrics"]))
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            json.dump(report["metrics"], handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--subset", choices=["val", "train", "all"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()

