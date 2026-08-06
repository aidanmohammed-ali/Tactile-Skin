"""Train the tactile pose model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .baselines import baseline_metrics, fit_ridge_position_baseline, present_samples, targets
from .dataset import (
    AugmentConfig,
    TactilePoseDataset,
    compute_feature_normalization,
    filter_by_keys,
    load_samples,
    sample_key,
    split_samples,
)
from .losses import pose_loss
from .metrics import format_metrics, pose_metrics, position_metrics, presence_metrics
from .model import TactilePoseNet


def main() -> None:
    args = parse_args()
    train(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="block_position_prediction/data_set")
    parser.add_argument("--output-dir", default="block_position_prediction/model_training/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--yaw-weight", type=float, default=0.05)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--force-scale-min", type=float, default=0.85)
    parser.add_argument("--force-scale-max", type=float, default=1.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--dropout-prob", type=float, default=0.02)
    parser.add_argument("--spatial-shift", type=int, default=0)
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    samples = load_samples(args.data_root)
    if not samples:
        raise SystemExit(f"No valid samples found under {args.data_root}")
    present_count = sum(1 for sample in samples if sample.object_present)
    absent_count = len(samples) - present_count
    train_samples, val_samples = split_samples(
        samples,
        strategy=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    feature_mean, feature_std = compute_feature_normalization(train_samples)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    device = resolve_device(args.device)
    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "sample_count": len(samples),
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "present_count": present_count,
            "absent_count": absent_count,
        }
    )
    write_json(run_dir / "config.json", config)

    print(
        f"Loaded {len(samples)} samples: train={len(train_samples)} val={len(val_samples)} "
        f"present={present_count} absent={absent_count}"
    )
    print(f"Using device: {device}")
    print("CoP baseline val:", format_metrics(baseline_metrics(val_samples)))
    ridge_val_metrics: dict[str, float] = {}
    train_present = present_samples(train_samples)
    val_present = present_samples(val_samples)
    if train_present and val_present:
        ridge = fit_ridge_position_baseline(train_present, alpha=args.ridge_alpha)
        ridge_val_pred = ridge.predict(val_present)
        ridge_val_metrics = position_metrics(ridge_val_pred, targets(val_present))
        print("Ridge baseline val:", format_metrics(ridge_val_metrics))
    else:
        print("Ridge baseline val: skipped (no present samples)")

    augment = AugmentConfig(
        enabled=not args.no_augment,
        force_scale_min=args.force_scale_min,
        force_scale_max=args.force_scale_max,
        noise_std=args.noise_std,
        dropout_prob=args.dropout_prob,
        spatial_shift=args.spatial_shift,
    )
    train_dataset = TactilePoseDataset(train_samples, feature_mean=feature_mean, feature_std=feature_std, augment=augment)
    val_dataset = TactilePoseDataset(
        val_samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=AugmentConfig(enabled=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TactilePoseNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_metric = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            yaw_weight=args.yaw_weight,
            presence_weight=args.presence_weight,
            use_amp=use_amp,
        )
        scheduler.step()
        val_report = evaluate_model(
            model,
            val_loader,
            device,
            yaw_weight=args.yaw_weight,
            presence_weight=args.presence_weight,
        )
        val_metric = best_selection_metric(val_report["metrics"])
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_report["metrics"].items()}}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or val_metric < best_metric:
            print(
                f"epoch {epoch:04d} train_loss={train_loss:.5f} "
                f"val={format_metrics(val_report['metrics'])}"
            )
        if val_metric < best_metric:
            best_metric = val_metric
            best_epoch = epoch
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                args=args,
                feature_mean=feature_mean,
                feature_std=feature_std,
                train_samples=train_samples,
                val_samples=val_samples,
                epoch=epoch,
                best_metric=best_metric,
            )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    last_report = evaluate_model(
        model,
        val_loader,
        device,
        yaw_weight=args.yaw_weight,
        presence_weight=args.presence_weight,
    )
    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        args=args,
        feature_mean=feature_mean,
        feature_std=feature_std,
        train_samples=train_samples,
        val_samples=val_samples,
        epoch=history[-1]["epoch"] if history else 0,
        best_metric=best_metric,
    )
    report = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_distance_mae_taxel": best_metric if "distance_mae_taxel" in last_report["metrics"] else None,
        "last_val_metrics": last_report["metrics"],
        "cop_val_metrics": baseline_metrics(val_samples),
        "ridge_val_metrics": ridge_val_metrics,
        "history": history,
    }
    write_json(run_dir / "metrics.json", report)
    print(f"Best checkpoint: {run_dir / 'best.pt'}")
    print(f"Best val distance_mae_taxel={best_metric:.4f} at epoch {best_epoch}")
    return run_dir


def train_one_epoch(
    model: TactilePoseNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    yaw_weight: float,
    presence_weight: float,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        maps = batch["maps"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        position = batch["position"].to(device, non_blocking=True)
        yaw_vector = batch["yaw_vector"].to(device, non_blocking=True)
        object_present = batch["object_present"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(maps, physics)
            presence_logit = output.get("presence_logit") if getattr(model, "presence_available", True) else None
            losses = pose_loss(
                output["position"],
                output["yaw_vector"],
                position,
                yaw_vector,
                presence_logit=presence_logit,
                object_present_target=object_present,
                yaw_weight=yaw_weight,
                presence_weight=presence_weight,
            )
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        count = maps.shape[0]
        total_loss += float(losses["total"].detach().cpu()) * count
        total_count += count
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate_model(
    model: TactilePoseNet,
    loader: DataLoader,
    device: torch.device,
    *,
    yaw_weight: float,
    presence_weight: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    pred_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    pred_yaws: list[np.ndarray] = []
    target_yaws: list[np.ndarray] = []
    inside_masks: list[np.ndarray] = []
    object_present_masks: list[np.ndarray] = []
    presence_logits: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        maps = batch["maps"].to(device, non_blocking=True)
        physics = batch["physics"].to(device, non_blocking=True)
        position = batch["position"].to(device, non_blocking=True)
        yaw_vector = batch["yaw_vector"].to(device, non_blocking=True)
        object_present = batch["object_present"].to(device, non_blocking=True)
        output = model(maps, physics)
        presence_logit = output.get("presence_logit") if getattr(model, "presence_available", True) else None
        loss = pose_loss(
            output["position"],
            output["yaw_vector"],
            position,
            yaw_vector,
            presence_logit=presence_logit,
            object_present_target=object_present,
            yaw_weight=yaw_weight,
            presence_weight=presence_weight,
        )["total"]
        losses.append(float(loss.detach().cpu()))
        pred_positions.append(output["position"].detach().cpu().numpy())
        target_positions.append(position.detach().cpu().numpy())
        pred_yaws.append(output["yaw_vector"].detach().cpu().numpy())
        target_yaws.append(batch["yaw"].detach().cpu().numpy())
        inside_masks.append(batch["inside"].detach().cpu().numpy())
        object_present_masks.append(batch["object_present"].detach().cpu().numpy())
        if getattr(model, "presence_available", True) and "presence_logit" in output:
            presence_logits.append(output["presence_logit"].detach().cpu().numpy())
    pred_position = np.concatenate(pred_positions, axis=0)
    target_position = np.concatenate(target_positions, axis=0)
    pred_yaw = np.concatenate(pred_yaws, axis=0)
    target_yaw = np.concatenate(target_yaws, axis=0)
    inside = np.concatenate(inside_masks, axis=0)
    object_present = np.concatenate(object_present_masks, axis=0)
    metrics = pose_metrics(pred_position, target_position, pred_yaw, target_yaw, inside, object_present)
    if presence_logits:
        metrics.update(presence_metrics(np.concatenate(presence_logits, axis=0), object_present))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return {"metrics": metrics}


def save_checkpoint(
    path: Path,
    *,
    model: TactilePoseNet,
    args: argparse.Namespace,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    train_samples: list,
    val_samples: list,
    epoch: int,
    best_metric: float,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "model_name": "TactilePoseNet",
        "model_version": "presence_v1",
        "config": vars(args).copy(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "train_keys": [sample_key(sample) for sample in train_samples],
        "val_keys": [sample_key(sample) for sample in val_samples],
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    torch.save(checkpoint, path)


def best_selection_metric(metrics: dict[str, float]) -> float:
    if "distance_mae_taxel" in metrics:
        return float(metrics["distance_mae_taxel"])
    if "presence_accuracy" in metrics:
        return 1.0 - float(metrics["presence_accuracy"])
    return float("inf")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def make_run_dir(output_dir: str | Path, run_name: str | None) -> Path:
    root = Path(output_dir)
    name = run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / name
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def dataset_from_checkpoint_subset(
    checkpoint: dict[str, Any],
    data_root: str | Path,
    subset: str,
) -> list:
    samples = load_samples(data_root)
    if subset == "all":
        return samples
    key_name = "val_keys" if subset == "val" else "train_keys"
    keys = checkpoint.get(key_name) or []
    selected = filter_by_keys(samples, keys)
    return selected or samples


if __name__ == "__main__":
    main()
