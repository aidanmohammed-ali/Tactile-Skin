# Realtime Tactile Pose Prediction

This UI reads tactile sensor frames using the same `ThreadedTactileReader`
pipeline as `data_collection_manual`, feeds the 10-frame top-5 normalized
sensor values into the trained model, and draws the predicted block footprint
on the tactile heatmap.

Run with the bundled best checkpoint:

```bash
python -m block_position_prediction.prediction.app
```

Use real hardware:

```bash
python -m block_position_prediction.prediction.app --tactile-port /dev/ttyACM0
```

On Windows, use the COM port directly:

```bash
python -m block_position_prediction.prediction.app --tactile-port COM10
```

New confidence-aware checkpoints output `sigmoid(presence_logit)`. When
confidence is below `--confidence-threshold` the UI does not draw a block. Older
checkpoints without a presence head still run, using `--legacy-force-threshold`
as a simple empty-sensor fallback until a no-block model is trained.

Keyboard:

- `Q` / `Esc`: quit
- `P`: cycle tactile port
- `R`: reset tactile processor
- `T`: tare tactile processor

The default checkpoint is:

```text
block_position_prediction/model/tactile_pose_best.pt
```
