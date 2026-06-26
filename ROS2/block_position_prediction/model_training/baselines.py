"""Simple baselines for tactile position prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset import COLS, ROWS, TactilePoseSample, physics_features
from .metrics import position_metrics


def cop_predict(samples: Sequence[TactilePoseSample]) -> np.ndarray:
    predictions: list[list[float]] = []
    xs = np.arange(COLS, dtype=np.float32)[None, :]
    ys = np.arange(ROWS, dtype=np.float32)[:, None]
    for sample in samples:
        pressure = sample.values.reshape(ROWS, COLS)
        total = float(pressure.sum())
        if total > 1e-6:
            pred_x = float((pressure * xs).sum() / total)
            pred_y = float((pressure * ys).sum() / total)
        else:
            pred_x = (COLS - 1) / 2.0
            pred_y = (ROWS - 1) / 2.0
        predictions.append([pred_x, pred_y])
    return np.asarray(predictions, dtype=np.float32)


def targets(samples: Sequence[TactilePoseSample]) -> np.ndarray:
    return np.stack([sample.position_taxel for sample in samples], axis=0).astype(np.float32)


def present_samples(samples: Sequence[TactilePoseSample]) -> list[TactilePoseSample]:
    return [sample for sample in samples if sample.object_present]


def baseline_metrics(samples: Sequence[TactilePoseSample]) -> dict[str, float]:
    present = present_samples(samples)
    if not present:
        return {}
    return position_metrics(cop_predict(present), targets(present))


@dataclass
class RidgePositionBaseline:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray

    def predict(self, samples: Sequence[TactilePoseSample]) -> np.ndarray:
        features = ridge_features(samples)
        features = (features - self.mean) / self.std
        design = np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float32)], axis=1)
        return (design @ self.weights).astype(np.float32)


def ridge_features(samples: Sequence[TactilePoseSample]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in samples:
        rows.append(np.concatenate([sample.values.astype(np.float32), physics_features(sample.values)], axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


def fit_ridge_position_baseline(
    samples: Sequence[TactilePoseSample],
    *,
    alpha: float = 1.0,
) -> RidgePositionBaseline:
    samples = present_samples(samples)
    if not samples:
        raise ValueError("Cannot fit ridge baseline without present samples.")
    features = ridge_features(samples)
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-6).astype(np.float32)
    normalized = (features - mean) / std
    design = np.concatenate([normalized, np.ones((normalized.shape[0], 1), dtype=np.float32)], axis=1)
    target = targets(samples)
    eye = np.eye(design.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + float(alpha) * eye, design.T @ target)
    return RidgePositionBaseline(mean=mean, std=std, weights=weights.astype(np.float32))

