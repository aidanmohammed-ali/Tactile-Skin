"""Evaluation metrics for tactile pose models."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .dataset import PITCH_MM
from .losses import YAW_PERIOD_RAD


def position_metrics(
    pred_position: np.ndarray,
    target_position: np.ndarray,
    *,
    prefix: str = "",
    pitch_mm: float = PITCH_MM,
) -> dict[str, float]:
    pred = np.asarray(pred_position, dtype=np.float32)
    target = np.asarray(target_position, dtype=np.float32)
    if pred.size == 0:
        return {}
    diff = pred - target
    abs_diff = np.abs(diff)
    distance = np.linalg.norm(diff, axis=1)
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}axis_mae_taxel": float(abs_diff.mean()),
        f"{key}x_mae_taxel": float(abs_diff[:, 0].mean()),
        f"{key}y_mae_taxel": float(abs_diff[:, 1].mean()),
        f"{key}distance_mae_taxel": float(distance.mean()),
        f"{key}distance_rmse_taxel": float(np.sqrt(np.mean(distance * distance))),
        f"{key}distance_p50_taxel": float(np.percentile(distance, 50)),
        f"{key}distance_p90_taxel": float(np.percentile(distance, 90)),
        f"{key}distance_mae_mm": float(distance.mean() * pitch_mm),
        f"{key}distance_p90_mm": float(np.percentile(distance, 90) * pitch_mm),
    }


def yaw_from_vector_np(yaw_vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(yaw_vector, dtype=np.float32)
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    angle = np.mod(angle, 2.0 * math.pi)
    return angle / 4.0


def yaw_error_rad(pred_yaw: np.ndarray, target_yaw: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_yaw, dtype=np.float32)
    target = np.asarray(target_yaw, dtype=np.float32)
    diff = np.mod(pred - target + YAW_PERIOD_RAD / 2.0, YAW_PERIOD_RAD)
    return np.abs(diff - YAW_PERIOD_RAD / 2.0)


def pose_metrics(
    pred_position: np.ndarray,
    target_position: np.ndarray,
    pred_yaw_vector: np.ndarray | None,
    target_yaw: np.ndarray | None,
    inside_mask: np.ndarray | None = None,
    object_present_mask: np.ndarray | None = None,
    *,
    pitch_mm: float = PITCH_MM,
) -> dict[str, float]:
    if object_present_mask is None:
        present = np.ones(len(target_position), dtype=bool)
    else:
        present = np.asarray(object_present_mask, dtype=bool)
    metrics: dict[str, float] = {
        "present_count": float(present.sum()),
        "absent_count": float((~present).sum()),
    }
    if present.any():
        metrics.update(position_metrics(pred_position[present], target_position[present], pitch_mm=pitch_mm))
    if pred_yaw_vector is not None and target_yaw is not None and present.any():
        pred_yaw = yaw_from_vector_np(pred_yaw_vector[present])
        yaw_error = yaw_error_rad(pred_yaw, np.asarray(target_yaw)[present])
        metrics.update(
            {
                "yaw_mae_rad": float(yaw_error.mean()),
                "yaw_mae_deg": float(np.degrees(yaw_error.mean())),
                "yaw_p90_deg": float(np.degrees(np.percentile(yaw_error, 90))),
            }
        )
    if inside_mask is not None and len(inside_mask):
        inside = np.asarray(inside_mask, dtype=bool)
        inside_present = inside & present
        edge_present = (~inside) & present
        if inside_present.any():
            metrics.update(position_metrics(pred_position[inside_present], target_position[inside_present], prefix="inside", pitch_mm=pitch_mm))
        if edge_present.any():
            metrics.update(position_metrics(pred_position[edge_present], target_position[edge_present], prefix="edge", pitch_mm=pitch_mm))
    return metrics


def presence_metrics(
    presence_logit: np.ndarray,
    object_present_target: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    logits = np.asarray(presence_logit, dtype=np.float32).reshape(-1)
    target = np.asarray(object_present_target, dtype=bool).reshape(-1)
    if logits.size == 0:
        return {}
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = prob >= float(threshold)
    tp = int((pred & target).sum())
    tn = int((~pred & ~target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "presence_accuracy": float((tp + tn) / max(1, len(target))),
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "presence_false_positive": float(fp),
        "presence_false_negative": float(fn),
        "presence_prob_mean": float(prob.mean()),
        "presence_prob_present_mean": float(prob[target].mean()) if target.any() else 0.0,
        "presence_prob_absent_mean": float(prob[~target].mean()) if (~target).any() else 0.0,
    }


def format_metrics(metrics: Mapping[str, float]) -> str:
    preferred = [
        "distance_mae_taxel",
        "distance_p90_taxel",
        "distance_mae_mm",
        "x_mae_taxel",
        "y_mae_taxel",
        "yaw_mae_deg",
        "presence_accuracy",
        "presence_precision",
        "presence_recall",
        "inside_distance_mae_taxel",
        "edge_distance_mae_taxel",
    ]
    parts: list[str] = []
    for key in preferred:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    for key in sorted(metrics):
        if key not in preferred:
            parts.append(f"{key}={metrics[key]:.4f}")
    return ", ".join(parts)

