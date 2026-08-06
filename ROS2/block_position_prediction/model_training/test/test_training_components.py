import json
import math

import numpy as np
import torch

from block_position_prediction.model_training.dataset import (
    AugmentConfig,
    TactilePoseDataset,
    compute_feature_normalization,
    load_samples,
    split_samples,
    yaw_to_vector_np,
    vector_to_yaw_np,
)
from block_position_prediction.model_training.losses import pose_loss
from block_position_prediction.model_training.model import TactilePoseNet


def test_load_samples_and_model_forward(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    label_path = run_dir / "labels.jsonl"
    rows = [
        _label("000001", x=4.0, y=2.0, yaw=0.1),
        _label("000002", x=10.0, y=5.0, yaw=0.7),
    ]
    label_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    samples = load_samples(tmp_path)

    assert len(samples) == 2
    feature_mean, feature_std = compute_feature_normalization(samples)
    dataset = TactilePoseDataset(
        samples,
        feature_mean=feature_mean,
        feature_std=feature_std,
        augment=AugmentConfig(enabled=False),
    )
    batch = [dataset[0], dataset[1]]
    maps = torch.stack([item["maps"] for item in batch])
    physics = torch.stack([item["physics"] for item in batch])
    output = TactilePoseNet()(maps, physics)

    assert maps.shape == (2, 4, 8, 16)
    assert physics.shape == (2, 10)
    assert torch.all(torch.stack([item["object_present"] for item in batch]) == 1.0)
    assert output["position"].shape == (2, 2)
    assert output["yaw_vector"].shape == (2, 2)
    assert output["presence_logit"].shape == (2,)


def test_no_block_label_loads_and_masks_pose_loss(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    values = np.zeros(128, dtype=float)
    values[0] = 0.01
    no_block = {
        "sample_id": "blank",
        "schema_version": "tactile_pose_v1",
        "input": {"rows": 8, "cols": 16, "values": values.tolist()},
        "target": {
            "object_present": False,
            "position_taxel": None,
            "pose": {"available": False, "yaw_mod90_rad": None},
        },
        "quality": {"label_source": "manual_no_block"},
    }
    (run_dir / "labels.jsonl").write_text(json.dumps(no_block), encoding="utf-8")

    samples = load_samples(tmp_path)
    feature_mean, feature_std = compute_feature_normalization(samples)
    dataset = TactilePoseDataset(samples, feature_mean=feature_mean, feature_std=feature_std, augment=AugmentConfig(enabled=False))
    item = dataset[0]
    losses = pose_loss(
        torch.tensor([[100.0, -100.0]]),
        torch.tensor([[0.0, 0.0]]),
        item["position"][None],
        item["yaw_vector"][None],
        presence_logit=torch.tensor([-20.0]),
        object_present_target=item["object_present"][None],
    )

    assert len(samples) == 1
    assert samples[0].object_present is False
    assert item["object_present"].item() == 0.0
    assert losses["position"].item() == 0.0
    assert losses["yaw"].item() == 0.0
    assert losses["presence"].item() < 1e-6


def test_yaw_vector_roundtrip():
    for yaw in np.linspace(0.0, math.pi / 2.0, 9, endpoint=False):
        decoded = vector_to_yaw_np(yaw_to_vector_np(float(yaw)))
        assert abs(decoded - yaw) < 1e-6


def test_grouped_split_keeps_nearest_taxel_bins_together(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    labels = [
        _label("a", x=1.1, y=1.0, yaw=0.1),
        _label("b", x=1.2, y=1.1, yaw=0.2),
        _label("c", x=10.0, y=4.0, yaw=0.3),
        _label("d", x=10.2, y=4.1, yaw=0.4),
        _label("e", x=14.0, y=6.0, yaw=0.5),
    ]
    (run_dir / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in labels), encoding="utf-8")
    samples = load_samples(tmp_path)

    train, val = split_samples(samples, strategy="grouped", val_fraction=0.4, seed=1)

    train_ids = {sample.sample_id for sample in train}
    val_ids = {sample.sample_id for sample in val}
    assert train_ids.isdisjoint(val_ids)
    assert {"a", "b"}.issubset(train_ids) or {"a", "b"}.issubset(val_ids)
    assert {"c", "d"}.issubset(train_ids) or {"c", "d"}.issubset(val_ids)


def _label(sample_id, *, x, y, yaw):
    values = np.zeros(128, dtype=float)
    values[2 * 16 + 4] = 1.0
    values[2 * 16 + 5] = 0.7
    return {
        "sample_id": sample_id,
        "schema_version": "tactile_pose_v1",
        "input": {"rows": 8, "cols": 16, "values": values.tolist()},
        "target": {
            "position_taxel": [x, y],
            "pose": {
                "yaw_mod90_rad": yaw,
                "fully_inside_sensor": True,
            },
        },
        "quality": {"label_source": "manual_aruco"},
    }
