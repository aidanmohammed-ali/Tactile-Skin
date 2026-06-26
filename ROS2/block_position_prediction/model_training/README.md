# Tactile Pose Model Training

This package trains a small tactile-only model that predicts square-block
position and modulo-90 yaw from one 8x16 tactile frame.

The current model is intentionally compact: a 4-channel tactile CNN plus a
small physical-feature branch. Position is the primary target. Yaw is trained as
a low-weight auxiliary task.

## Data

The loader scans:

```text
block_position_prediction/data_set/*/labels.jsonl
```

Each valid block sample must contain:

- `input.values`: 128 tactile values in 8x16 order
- `target.object_present`: optional; missing means `true` for older labels
- `target.position_taxel`: `[x, y]`
- `target.pose.yaw_mod90_rad`

No-block samples use `target.object_present=false`, `target.position_taxel=null`,
and `target.pose.available=false`. Position and yaw losses are masked out for
these samples, while the confidence head still trains on them.

Per-sample calibration is not used as a model input. It remains useful as label
quality metadata.

## Train

From the repository root:

```bash
python3 -m block_position_prediction.model_training.train \
  --data-root block_position_prediction/data_set \
  --device auto
```

Use CUDA explicitly on a CUDA machine:

```bash
python3 -m block_position_prediction.model_training.train \
  --data-root block_position_prediction/data_set \
  --device cuda
```

Training runs are written under:

```text
block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/
```

Each run contains:

- `config.json`
- `metrics.json`
- `best.pt`
- `last.pt`

## Evaluate

```bash
python3 -m block_position_prediction.model_training.evaluate \
  --checkpoint block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/best.pt \
  --subset val
```

Subsets:

- `val`: checkpoint validation split
- `train`: checkpoint training split
- `all`: every valid sample under the data root

## Predict

```bash
python3 -m block_position_prediction.model_training.predict \
  --checkpoint block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/best.pt \
  --labels-jsonl block_position_prediction/data_set/20260611_155745/labels.jsonl \
  --output predictions.jsonl
```

If `--output` is omitted, predictions are printed to stdout.

## Yaw Diagnostics

```bash
python3 -m block_position_prediction.model_training.yaw_diagnostics \
  --checkpoint block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/best.pt \
  --subset val \
  --output-dir block_position_prediction/model_training/runs/YYYYMMDD_HHMMSS/yaw_diagnostics
```

This writes worst-yaw sample rows, binned statistics, and label-vs-prediction
heatmaps for inspecting edge cases and ambiguous contact patterns.

## Model

CNN input channels:

- raw pressure map
- force-normalized pressure map
- fixed x-coordinate map
- fixed y-coordinate map

Physics features:

- force sum, max, mean, std, active fraction
- center of pressure x/y
- pressure variance x/y and covariance

Outputs:

- `presence_logit`: object-present confidence logit
- `position_taxel`: two continuous values
- `yaw_vector`: `[cos(4*yaw), sin(4*yaw)]`

The loss is:

```text
BCEWithLogits(presence) + present_mask * (SmoothL1(position) + 0.05 * MSE(normalized_yaw_vector))
```

Best checkpoint selection uses validation position distance MAE in taxels.

## Notes

The current dataset is small. It is enough to validate the pipeline and check
that the model beats the center-of-pressure baseline, but final accuracy should
be judged after collecting substantially more spatially balanced data.
