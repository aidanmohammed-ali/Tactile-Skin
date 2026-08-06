# Deployed Tactile Pose Model

`tactile_pose_best.pt` is the current best checkpoint copied from:

```text
block_position_prediction/model_training/runs/presence_full_shift1_lr5e4_p1_yaw005_20260612/best.pt
```

It was trained on the 300-sample dataset with no-block confidence labels using:

```text
--spatial-shift 1 --lr 0.0005 --yaw-weight 0.05 --presence-weight 1.0
```

Validation metrics:

- position MAE: 0.4805 taxel / 1.9221 mm
- position P90: 0.7691 taxel / 3.0763 mm
- yaw MAE: 13.7911 deg
- presence accuracy/precision/recall: 1.0 / 1.0 / 1.0
- no-block validation errors: 0 false positive, 0 false negative
