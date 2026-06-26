# ObjectDetection Handoff Notes

Last updated: 2026-05-21

This folder contains the YOLOv8 block detector used together with the calibrated
board grid from `cv_aruco_src`.

## Runtime Script

Run from the repository root:

```powershell
python ObjectDetection/realtime_block_detector.py --source 0
```

For the HTTP camera stream used by `real_time_detection.py`:

```powershell
python ObjectDetection/realtime_block_detector.py --source "http://192.168.108.213:3588/video"
```

Defaults:

- YOLO weights: `ObjectDetection/block2.pt`
- YOLO confidence threshold: `0.1`
- Calibration JSON: `cv_aruco_src/board_calibration.json`
- ArUco dictionary: `DICT_4X4_50`
- Board coordinates: continuous grid units, `col` and `row`

The script draws:

- the saved calibration grid over the live camera frame
- the highest-confidence detected block bounding box
- the selected block anchor point
- pixel coordinates and calibrated board coordinates

Keyboard controls:

- `C`: recalibrate from the current frame and save the JSON
- `S`: save an annotated snapshot in `ObjectDetection`
- `Q` or `Esc`: quit

On-screen controls:

- `Recalibrate`: recalibrate from the current frame and save the JSON

Use `--anchor bottom-center` if the desired position is the block's contact
point on the board rather than the bounding-box center.

## Dependency

YOLOv8 inference uses `ultralytics`, now listed in the repository
`requirements.txt`.
