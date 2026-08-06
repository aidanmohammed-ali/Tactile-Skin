"""Dataset loading and feature extraction for tactile pose labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ROWS = 8
COLS = 16
NUM_TAXELS = ROWS * COLS
PITCH_MM = 4.0
ACTIVE_THRESHOLD = 0.05
EPS = 1e-6


@dataclass(frozen=True)
class TactilePoseSample:
    values: np.ndarray
    position_taxel: np.ndarray
    yaw_mod90_rad: float
    object_present: bool
    fully_inside_sensor: bool
    sample_id: str
    dataset_dir: Path
    label_path: Path
    image_path: str | None
    quality: dict


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    force_scale_min: float = 0.85
    force_scale_max: float = 1.15
    noise_std: float = 0.01
    dropout_prob: float = 0.02
    spatial_shift: int = 0


def sample_key(sample: TactilePoseSample) -> str:
    return f"{sample.label_path}:{sample.sample_id}"


def yaw_to_vector_np(yaw_mod90_rad: float) -> np.ndarray:
    return np.asarray(
        [math.cos(4.0 * float(yaw_mod90_rad)), math.sin(4.0 * float(yaw_mod90_rad))],
        dtype=np.float32,
    )


def vector_to_yaw_np(yaw_vector: np.ndarray) -> float:
    angle = math.atan2(float(yaw_vector[1]), float(yaw_vector[0]))
    return (angle % (2.0 * math.pi)) / 4.0


def resolve_label_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if (root / "labels.jsonl").is_file():
        return [root / "labels.jsonl"]
    return sorted(root.glob("*/labels.jsonl"))


def load_samples(path: str | Path) -> list[TactilePoseSample]:
    samples: list[TactilePoseSample] = []
    for label_path in resolve_label_paths(path):
        samples.extend(_load_label_file(label_path))
    return samples


def _load_label_file(label_path: Path) -> list[TactilePoseSample]:
    rows: list[TactilePoseSample] = []
    dataset_dir = label_path.parent
    with label_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                label = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample = _sample_from_label(label, label_path, dataset_dir, line_no)
            if sample is not None:
                rows.append(sample)
    return rows


def _sample_from_label(
    label: dict,
    label_path: Path,
    dataset_dir: Path,
    line_no: int,
) -> TactilePoseSample | None:
    values = (label.get("input") or {}).get("values")
    target = label.get("target") or {}
    pose = target.get("pose") or {}
    position = target.get("position_taxel")
    yaw = pose.get("yaw_mod90_rad")
    object_present = bool(target.get("object_present", True))
    if not isinstance(values, list) or len(values) != NUM_TAXELS:
        return None
    try:
        values_array = np.asarray(values, dtype=np.float32)
        if object_present:
            if not isinstance(position, list) or len(position) < 2:
                return None
            if yaw is None:
                return None
            position_array = np.asarray(position[:2], dtype=np.float32)
            yaw_float = float(yaw)
        else:
            position_array = np.zeros(2, dtype=np.float32)
            yaw_float = 0.0
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values_array)):
        return None
    if not np.all(np.isfinite(position_array)) or not math.isfinite(yaw_float):
        return None
    sample_id = str(label.get("sample_id") or f"line_{line_no:06d}")
    return TactilePoseSample(
        values=values_array,
        position_taxel=position_array,
        yaw_mod90_rad=yaw_float,
        object_present=object_present,
        fully_inside_sensor=bool(pose.get("fully_inside_sensor", False)),
        sample_id=sample_id,
        dataset_dir=dataset_dir,
        label_path=label_path,
        image_path=label.get("image_path") or (label.get("image") or {}).get("path"),
        quality=dict(label.get("quality") or {}),
    )


def tactile_map_channels(values: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    pressure = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    total = float(pressure.sum())
    normalized = pressure / total if total > EPS else np.zeros_like(pressure)
    x_coord = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :].repeat(rows, axis=0)
    y_coord = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None].repeat(cols, axis=1)
    return np.stack([pressure, normalized, x_coord, y_coord], axis=0).astype(np.float32)


def physics_features(values: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    pressure = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    total = float(pressure.sum())
    max_value = float(pressure.max()) if pressure.size else 0.0
    mean = float(pressure.mean()) if pressure.size else 0.0
    std = float(pressure.std()) if pressure.size else 0.0
    active_fraction = float((pressure > ACTIVE_THRESHOLD).mean()) if pressure.size else 0.0

    xs = np.arange(cols, dtype=np.float32)[None, :]
    ys = np.arange(rows, dtype=np.float32)[:, None]
    if total > EPS:
        cop_x = float((pressure * xs).sum() / total)
        cop_y = float((pressure * ys).sum() / total)
        dx = xs - cop_x
        dy = ys - cop_y
        var_x = float((pressure * dx * dx).sum() / total)
        var_y = float((pressure * dy * dy).sum() / total)
        cov_xy = float((pressure * dx * dy).sum() / total)
    else:
        cop_x = (cols - 1) / 2.0
        cop_y = (rows - 1) / 2.0
        var_x = 0.0
        var_y = 0.0
        cov_xy = 0.0

    return np.asarray(
        [total, max_value, mean, std, active_fraction, cop_x, cop_y, var_x, var_y, cov_xy],
        dtype=np.float32,
    )


def compute_feature_normalization(samples: Sequence[TactilePoseSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot compute normalization without samples.")
    features = np.stack([physics_features(sample.values) for sample in samples], axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def split_samples(
    samples: Sequence[TactilePoseSample],
    *,
    strategy: str = "grouped",
    val_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[list[TactilePoseSample], list[TactilePoseSample]]:
    if not samples:
        raise ValueError("No samples to split.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    samples_list = list(samples)
    if len(samples_list) == 1:
        return samples_list, samples_list
    if strategy == "random":
        indices = list(range(len(samples_list)))
        rng.shuffle(indices)
        val_count = max(1, min(len(indices) - 1, round(len(indices) * val_fraction)))
        val_indices = set(indices[:val_count])
        train = [sample for i, sample in enumerate(samples_list) if i not in val_indices]
        val = [sample for i, sample in enumerate(samples_list) if i in val_indices]
        return train, val
    if strategy != "grouped":
        raise ValueError(f"Unknown split strategy: {strategy}")

    groups: dict[object, list[TactilePoseSample]] = {}
    for index, sample in enumerate(samples_list):
        key = nearest_taxel_bin(sample.position_taxel) if sample.object_present else ("no_block", index)
        groups.setdefault(key, []).append(sample)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    target_val = max(1, round(len(samples_list) * val_fraction))
    val_groups: set[object] = set()
    val_count = 0
    for key, group in group_items:
        if len(val_groups) > 0 and val_count >= target_val:
            break
        val_groups.add(key)
        val_count += len(group)
    def key_for(index: int, sample: TactilePoseSample) -> object:
        return nearest_taxel_bin(sample.position_taxel) if sample.object_present else ("no_block", index)

    val = [sample for i, sample in enumerate(samples_list) if key_for(i, sample) in val_groups]
    train = [sample for i, sample in enumerate(samples_list) if key_for(i, sample) not in val_groups]
    if not train:
        return split_samples(samples_list, strategy="random", val_fraction=val_fraction, seed=seed)
    return train, val


def nearest_taxel_bin(position_taxel: np.ndarray) -> tuple[int, int]:
    x = int(round(float(position_taxel[0])))
    y = int(round(float(position_taxel[1])))
    return (min(COLS - 1, max(0, x)), min(ROWS - 1, max(0, y)))


def filter_by_keys(samples: Sequence[TactilePoseSample], keys: Iterable[str]) -> list[TactilePoseSample]:
    key_set = set(keys)
    return [sample for sample in samples if sample_key(sample) in key_set]


class TactilePoseDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[TactilePoseSample],
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        augment: AugmentConfig | None = None,
    ) -> None:
        self.samples = list(samples)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.augment = augment or AugmentConfig(enabled=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        sample = self.samples[index]
        values = sample.values.reshape(ROWS, COLS).astype(np.float32, copy=True)
        position = sample.position_taxel.astype(np.float32, copy=True)
        if self.augment.enabled:
            values, position = self._augment(values, position, shift_position=sample.object_present)
        flat_values = values.reshape(-1)
        features = physics_features(flat_values)
        features = (features - self.feature_mean) / self.feature_std
        return {
            "maps": torch.from_numpy(tactile_map_channels(flat_values)),
            "physics": torch.from_numpy(features.astype(np.float32)),
            "position": torch.from_numpy(position.astype(np.float32)),
            "yaw_vector": torch.from_numpy(yaw_to_vector_np(sample.yaw_mod90_rad)),
            "yaw": torch.tensor(float(sample.yaw_mod90_rad), dtype=torch.float32),
            "object_present": torch.tensor(float(sample.object_present), dtype=torch.float32),
            "inside": torch.tensor(bool(sample.fully_inside_sensor), dtype=torch.bool),
            "sample_key": sample_key(sample),
        }

    def _augment(self, values: np.ndarray, position: np.ndarray, *, shift_position: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.augment.force_scale_max > 0.0:
            scale = random.uniform(self.augment.force_scale_min, self.augment.force_scale_max)
            values = values * np.float32(scale)
        if self.augment.noise_std > 0.0:
            values = values + np.random.normal(0.0, self.augment.noise_std, size=values.shape).astype(np.float32)
        if self.augment.dropout_prob > 0.0:
            keep = np.random.random(size=values.shape) >= self.augment.dropout_prob
            values = values * keep.astype(np.float32)
        values = np.clip(values, 0.0, None)
        max_shift = int(self.augment.spatial_shift)
        if max_shift > 0:
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            if dx or dy:
                values = shift_map_zero_fill(values, dx=dx, dy=dy)
                if shift_position:
                    position = position + np.asarray([dx, dy], dtype=np.float32)
        return values.astype(np.float32), position.astype(np.float32)


def shift_map_zero_fill(values: np.ndarray, *, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    rows, cols = values.shape
    src_x0 = max(0, -dx)
    src_x1 = min(cols, cols - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(cols, cols + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(rows, rows - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(rows, rows + dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return shifted

