# Tactile Block Data Collection

Non-ROS data collection UI for labeling block positions on the tactile sensor
from webcam video. The tool uses an A4 ArUco calibration sheet, YOLO block
detection, and saves image + label samples for future tactile-sensor training.
It can also read tactile frames from the serial sensor, or use the built-in
simulator when hardware is not connected.

The training label coordinate frame is `taxel_center_v1`: the center of the
top-left taxel is `(0, 0)`, `x` increases to the right, `y` increases downward,
and one unit is one 4 mm taxel pitch. The 2.4 cm block is represented as a
fixed 6-taxel square footprint when pose is available.

## Requirements

Run from the project root:

```bash
cd <path_to_project_directory>/ROS2
```

Python packages used by the tool:

```bash
python3 -c "import cv2, ultralytics, PIL, numpy"
```

If any import fails, install the missing package in your environment. OpenCV
must include `cv2.aruco`, so use `opencv-contrib-python` if needed.
Real tactile hardware also requires `pyserial`; simulator mode does not.

## Generate The A4 Calibration Sheet

The PDF version is recommended for printing:

```bash
python3 -m block_position_prediction.data_collection.sheet \
  --output block_position_prediction/data_collection/assets/a4_sensor_calibration.pdf
```

Print the PDF at **100% scale** on landscape A4. Do not use fit-to-page.
Place the sensor inside the printed reserved rectangle.

## Start The UI

Use the last successfully connected camera source if one has been saved:

```bash
python3 -m block_position_prediction.data_collection.app
```

By default, tactile input runs in `SIMULATOR` mode. To use hardware:

```bash
python3 -m block_position_prediction.data_collection.app \
  --tactile-port /dev/ttyACM0 \
  --tactile-baud 115200
```

Or start with a specific webcam index or URL:

```bash
python3 -m block_position_prediction.data_collection.app --source 0
```

```bash
python3 -m block_position_prediction.data_collection.app \
  --source "http://CAMERA_IP/video"
```

Network camera sources such as HTTP/RTSP are captured in an isolated helper
process. If OpenCV/FFmpeg crashes or exits after a disconnect, the UI should
stay alive and allow reconnecting.

The UI remembers the last successfully connected source in:

```text
block_position_prediction/data_collection/assets/camera_source.txt
```

## UI Controls

- `U`: edit webcam source
- Click the `Source` input box: edit webcam source
- Click the right-side `Port` box: choose tactile serial port or `SIMULATOR`
- `Enter`: connect to the typed source
- `Esc`: cancel source editing, or quit when not editing
- `A`: toggle automatic ArUco calibration
- `C`: calibrate using the current frame
- `S`: save one sample from the latest valid detection and tactile top-5 data
- `L`: clear the retained last block position
- `D`: pause/resume automatic detection
- `Q`: quit

## Typical Workflow

1. Print `assets/a4_sensor_calibration.pdf` at 100% scale.
2. Put the tactile sensor inside the printed rectangle.
3. Start the UI.
4. Set or confirm the webcam source.
5. Make sure all four ArUco markers are visible.
6. Leave automatic calibration on, or press `C` to calibrate manually.
7. Place a block on the sensor.
8. Press `S` to save each labeled sample.

Automatic calibration is on by default and tries to update in memory at up to
5 Hz. It does not write the calibration file continuously. Press `A` to turn it
off; when turning it off, the latest successful calibration is saved.

If YOLO briefly loses the block, the UI keeps showing the last valid block
position on both the camera view and tactile heatmap. Press `L` to clear that
retained position.

## Output

Samples are written under:

```text
block_position_prediction/data_collection/runs/YYYYMMDD_HHMMSS/
```

Each session contains:

```text
images/000001.jpg
labels.jsonl
metadata.json
```

Each JSONL row uses the compact `tactile_pose_v1` schema. The main training
fields are:

- `input.values`: 128 canonical tactile values in physical sensor order
- `target.position_taxel`: block position in `taxel_center_v1`
- `target.position_normalized`: position scaled by `[15, 7]`
- `target.position_cm_from_taxel0`: physical position from the top-left taxel
- `target.pose`: yaw and fixed 6-taxel footprint; current `block3.pt` uses `bbox_homography`
- `quality`: confidence, retained-position flag, tactile availability, calibration error

`target.pose.source` is `bbox_homography` for the current bbox-only model. This
projects the image bbox through ArUco calibration, so it accounts for the angle
between webcam image axes and the sensor coordinate frame. Future OBB/keypoint
models can use `angle_homography` or `polygon_homography` for stronger block
orientation labels.

The original webcam image is still saved for auditing, and `image_path` is
added to each JSONL row. Session-level calibration and geometry are stored once
in `metadata.json`.

## Preview A Dataset

Open a saved run and step through samples:

```bash
python3 -m block_position_prediction.data_collection.preview_dataset \
  block_position_prediction/data_collection/runs/YYYYMMDD_HHMMSS
```

Controls:

- `N`, `Right`, or `Space`: next sample
- `P` or `Left`: previous sample
- `Q` or `Esc`: quit

The preview uses the same tactile heatmap and fixed-footprint renderer as the
right side of the live UI, so online and offline visualization stay consistent.

## Useful Options

```bash
python3 -m block_position_prediction.data_collection.app \
  --source 0 \
  --tactile-port SIMULATOR \
  --confidence 0.1 \
  --detect-rate-hz 5 \
  --auto-calibrate-rate-hz 5 \
  --frame-width 1280 \
  --frame-height 720
```

Use `--help` to see all geometry and camera options:

```bash
python3 -m block_position_prediction.data_collection.app --help
```
