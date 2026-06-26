# GPT Context: Tactile Pose Model Training

This document explains the `block_position_prediction/model_training/` package
so the training work can be moved to a new workspace with minimal context loss.

## Purpose

The package trains a tactile-only model that predicts the pose of a fixed
24 mm square block from one tactile sensor frame.

Primary objective:

- Predict block center position accurately in `taxel_center_v1` coordinates.

Secondary objective:

- Predict square-symmetric yaw as `yaw_mod90_rad`, where yaw is periodic every
  90 degrees.

The current model is intentionally small. It should run in real time on a normal
computer, train on CUDA when available, and remain usable while the dataset is
still small.

## Expected Dataset Format

The training scripts read:

```text
block_position_prediction/data_set/*/labels.jsonl
```

Each JSONL row must contain:

```json
{
  "input": {
    "rows": 8,
    "cols": 16,
    "values": [128 tactile values]
  },
  "target": {
    "position_taxel": [x, y],
    "pose": {
      "yaw_mod90_rad": 0.0,
      "fully_inside_sensor": true
    }
  }
}
```

The loader accepts older labels as long as these fields exist. Per-sample
calibration is not used as a model input. It is only label-quality metadata.

The current local dataset used during implementation was:

```text
block_position_prediction/data_set/20260611_155745/labels.jsonl
```

At implementation time it contained 234 valid samples:

- 234 samples with 128 tactile values
- 234 samples with position labels
- 234 samples with yaw labels
- 61 samples where `fully_inside_sensor` was true

This is enough to validate the pipeline, but not enough to judge final model
accuracy.

## Files Added

```text
block_position_prediction/model_training/
  __init__.py
  baselines.py
  dataset.py
  evaluate.py
  losses.py
  metrics.py
  model.py
  predict.py
  train.py
  README.md
  GPT.md
  test/test_training_components.py
```

`.gitignore` was also updated to ignore:

```text
/block_position_prediction/model_training/runs/
```

Training artifacts should not be committed.

## Model Architecture

The main model is `TactilePoseNet` in `model.py`.

It is a hybrid model:

1. A compact CNN branch processes tactile maps.
2. A physics-feature MLP branch processes hand-crafted tactile statistics.
3. A fusion head predicts position and yaw.

### CNN Input

Each tactile frame becomes a 4-channel tensor:

```text
4 x 8 x 16
```

Channels:

1. Raw pressure map
2. Force-normalized pressure map
3. Fixed x-coordinate map
4. Fixed y-coordinate map

The coordinate channels help the CNN learn absolute position and edge effects.

### Physics Features

`dataset.py::physics_features()` computes 10 features:

1. `force_sum`
2. `max`
3. `mean`
4. `std`
5. `active_fraction`
6. `cop_x`
7. `cop_y`
8. `var_x`
9. `var_y`
10. `cov_xy`

These features are normalized using mean/std computed from the training split.

### CNN Backbone

The implemented backbone is:

```text
ConvGNAct 4 -> 24, 3x3
ResidualBlock 24
ConvGNAct 24 -> 32, stride=(1,2)
ResidualBlock 32
ConvGNAct 32 -> 48, stride=2
ResidualBlock 48
flatten: 48 x 4 x 4 = 768
```

GroupNorm is used instead of BatchNorm because the dataset is small and batch
sizes may be unstable.

### Fusion Head

```text
physics MLP: 10 -> 32
fusion MLP: 768 + 32 -> 128 -> 64
position head: 64 -> 2
yaw head: 64 -> 2
```

Position output:

```text
[x_taxel, y_taxel]
```

Yaw output:

```text
[cos(4 * yaw_mod90_rad), sin(4 * yaw_mod90_rad)]
```

The factor of 4 handles the 90-degree periodicity of a square block.

## Loss

Implemented in `losses.py`.

The total loss is:

```text
SmoothL1(position_pred, position_target)
+ yaw_weight * MSE(normalized_yaw_vector_pred, yaw_vector_target)
```

Default:

```text
yaw_weight = 0.05
```

Position is the primary target. Yaw is only an auxiliary task.

## Data Splitting

Implemented in `dataset.py::split_samples()`.

Default split:

```text
--split grouped
```

Grouped split assigns samples to validation by nearest taxel bin. This reduces
leakage from nearly duplicated neighboring samples.

Fast debug split:

```text
--split random
```

Random split is only for sanity checks. It may overestimate real performance.

## Data Augmentation

Default training augmentation:

- Force scaling
- Gaussian noise
- Taxel dropout

Optional augmentation:

- Spatial shift with synchronized position-label shift

Spatial shift is disabled by default because the current dataset is small and
the real sensor may have asymmetric taxel response.

No default flips or rotations are used. Do not add them unless validation proves
they help.

## Baselines

Implemented in `baselines.py`.

Two baselines are printed during training:

1. Center-of-pressure baseline
2. Ridge regression baseline

The Hybrid CNN should beat the center-of-pressure baseline. With very small
data, ridge regression may temporarily outperform the CNN until enough samples
are collected or the CNN is trained longer.

## Metrics

Implemented in `metrics.py`.

Main metrics:

- `distance_mae_taxel`
- `distance_rmse_taxel`
- `distance_p50_taxel`
- `distance_p90_taxel`
- `distance_mae_mm`
- `x_mae_taxel`
- `y_mae_taxel`
- `yaw_mae_deg`

Metrics are also split by:

- `inside_*`: samples whose footprint is fully inside the sensor
- `edge_*`: edge or partial-contact samples

The best checkpoint is selected by validation `distance_mae_taxel`.

## Main Commands

Train on auto device:

```bash
python3 -m block_position_prediction.model_training.train \
  --data-root block_position_prediction/data_set \
  --device auto
```

Train on CUDA:

```bash
python3 -m block_position_prediction.model_training.train \
  --data-root block_position_prediction/data_set \
  --device cuda
```

Quick CPU smoke test:

```bash
python3 -m block_position_prediction.model_training.train \
  --data-root block_position_prediction/data_set \
  --device cpu \
  --epochs 1 \
  --batch-size 32 \
  --run-name smoke_test \
  --patience 0
```

Evaluate:

```bash
python3 -m block_position_prediction.model_training.evaluate \
  --checkpoint block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/best.pt \
  --subset val
```

Predict all rows in one labels file:

```bash
python3 -m block_position_prediction.model_training.predict \
  --checkpoint block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/best.pt \
  --labels-jsonl block_position_prediction/data_set/20260611_155745/labels.jsonl \
  --output predictions.jsonl
```

Run tests:

```bash
python3 -m pytest block_position_prediction/model_training/test
```

## Checkpoint Contents

Checkpoints are saved as `.pt` files.

Each checkpoint contains:

- `model_state`
- `model_name`
- `config`
- `feature_mean`
- `feature_std`
- `train_keys`
- `val_keys`
- `epoch`
- `best_metric`

This is enough to reload the model in another workspace if the package code and
dataset paths are preserved or remapped.

## Migration Checklist

To move this training setup to a new workspace:

1. Copy `block_position_prediction/model_training/`.
2. Copy or recreate `block_position_prediction/data_set/`.
3. Ensure `block_position_prediction/__init__.py` exists.
4. Install Python dependencies:

   ```bash
   python3 -c "import numpy, torch, pytest"
   ```

5. Run:

   ```bash
   python3 -m pytest block_position_prediction/model_training/test
   ```

6. Run a 1-epoch smoke test.
7. Train normally with `--device cuda` on the CUDA machine.

## Known Limitations

- The current model uses a single tactile frame only.
- It does not use temporal history.
- It does not use camera images.
- It does not use calibration as input.
- Current data volume is small and spatially imbalanced.
- Current edge/partial-contact samples are common, so always inspect inside vs
  edge metrics separately.

## Recommended Next Steps

Collect more data before making conclusions about final accuracy.

Suggested data targets:

- Minimum pipeline validation: 500 to 1000 samples
- First useful model: 1500 to 3000 samples
- More stable sub-taxel accuracy: 5000+ samples

Collect data across:

- All sensor regions
- Center, edges, and corners
- Multiple yaw angles
- Different contact pressures
- Multiple sessions and calibrations

The target metric for the first serious model is:

```text
validation position error < 0.5 taxel
```

This is approximately 2 mm because one taxel pitch is 4 mm.

