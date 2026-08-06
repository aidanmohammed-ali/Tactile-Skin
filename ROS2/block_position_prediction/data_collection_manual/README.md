# Tactile Block Manual Data Collection

Manual, non-ROS data collection UI for labeling a 2.4 cm square block on the
tactile sensor from webcam video. The tool shows live camera and tactile
previews, freezes one synchronized camera/tactile snapshot on demand, lets you
set the block center by mouse and fine-tune position/rotation from the keyboard,
and saves labels that remain compatible with the `tactile_pose_v1` training
schema.

The training label coordinate frame is `taxel_center_v1`: the center of the
top-left taxel is `(0, 0)`, `x` increases to the right, `y` increases downward,
and one unit is one 4 mm taxel pitch. The block side is fixed at 2.4 cm, so the
saved footprint is always a 6-taxel square.

## Requirements

Run from the project root:

```bash
cd <path_to_project_directory>/ROS2
```

Python packages used by the tool:

```bash
python3 -c "import cv2, PIL, numpy"
```

OpenCV must include `cv2.aruco`, so use `opencv-contrib-python` if needed. Real
tactile hardware also requires `pyserial`; simulator mode does not.

## Generate The A4 Calibration Sheet

The PDF version is recommended for printing:

```bash
python3 -m block_position_prediction.data_collection_manual.sheet \
  --output block_position_prediction/data_collection_manual/assets/a4_sensor_calibration.pdf
```

Print the PDF at 100% scale on landscape A4. Do not use fit-to-page. Place the
sensor inside the printed reserved rectangle.

## Start The UI

Use the last successfully connected camera source:

```bash
python3 -m block_position_prediction.data_collection_manual.app
```

By default, tactile input runs in `SIMULATOR` mode. To use hardware:

```bash
python3 -m block_position_prediction.data_collection_manual.app \
  --tactile-port /dev/ttyACM0 \
  --tactile-baud 115200
```

Start with a specific webcam index or URL:

```bash
python3 -m block_position_prediction.data_collection_manual.app --source 0
```

```bash
python3 -m block_position_prediction.data_collection_manual.app \
  --source "http://CAMERA_IP/video"
```

Network camera sources such as HTTP/RTSP are captured in an isolated helper
process. If OpenCV/FFmpeg crashes or exits after a disconnect, the UI should
stay alive and allow reconnecting.

The UI remembers the last successfully connected source in:

```text
block_position_prediction/data_collection_manual/assets/camera_source.txt
```

## Dataset Selection

On startup, the UI shows recent runs under:

```text
block_position_prediction/data_collection_manual/runs/
```

Controls on the selection screen:

- `Up` / `Down`: choose an existing run
- `Enter`: continue the selected run, or create one if none exist
- `N`: create a new run
- `Q` / `Esc`: quit

After a dataset is selected, the main UI opens immediately. The camera source is
connected in the background, so a missing webcam or offline network stream does
not block the UI. Use the `Source` input box to type a new source and press
`Enter` to reconnect.

You can skip the chooser:

```bash
python3 -m block_position_prediction.data_collection_manual.app --new-run
```

```bash
python3 -m block_position_prediction.data_collection_manual.app \
  --dataset block_position_prediction/data_collection_manual/runs/YYYYMMDD_HHMMSS
```

## Live UI Controls

- `Space`: freeze the latest camera frame, tactile snapshot, and calibration
- `B`: save a blank/no-block tactile sample from the live sensor stream
- Click or drag left mouse button on a frozen image: set block center only
- `W` / `A` / `S` / `D`: fine-tune the block up, left, down, right
- `Q` / `E`: fine-tune block rotation
- `F`: save the current draft, or overwrite the current saved sample label
- `Z`: previous saved sample; from live, opens the latest saved sample
- `X`: next saved sample; from the newest saved sample, returns to live
- Click the `Delete` button, or press `Backspace`/`Delete`: remove the
  currently viewed saved sample from `labels.jsonl`
- `R`: return to the live view immediately
- `C`: calibrate from the latest live camera frame and save calibration
- `U`: edit webcam source
- Click the right-side `Port` box: choose tactile serial port or `SIMULATOR`;
  missing hardware does not block the UI
- `T`: tare tactile processing
- `Esc`: leave draft/saved view and return to live; quits from live

Automatic ArUco calibration is enabled by default and updates the in-memory
homography at a low rate. A frozen draft keeps the calibration snapshot from
the capture moment, so later automatic calibration updates do not change that
draft's coordinates.

## Manual Labeling Workflow

1. Start or select a dataset.
2. Confirm live camera, tactile preview, and ArUco calibration are working.
3. Press `Space` to freeze the current camera image and tactile data.
4. Click or drag on the block center.
5. Use `W/A/S/D` and `Q/E` to fine-tune position and rotation.
6. Check the fixed 2.4 cm square on the camera image and the right-side tactile
   coordinate preview.
7. Press `F` to save.

Saving only requires a calibration and an annotation. Missing tactile data or a
footprint outside the sensor is recorded in `quality`, but no longer blocks
saving.

For no-block/confidence training, leave the sensor empty and press `B` from the
live view. Blank samples save `target.object_present=false`, with no manual
center or yaw label. This uses the same canonical `10-frame top5_normalized`
tactile input as normal block samples.

## Output

Each run contains:

```text
images/000001.jpg
labels.jsonl
metadata.json
```

Each JSONL row uses the `tactile_pose_v1` schema. Main fields:

- `input.values`: 128 canonical tactile values in physical sensor order
- `target.object_present`: `true` for block samples, `false` for blank samples
- `target.position_taxel`: manual block center in `taxel_center_v1`
- `target.position_normalized`: position scaled by `[15, 7]`
- `target.position_cm_from_taxel0`: physical position from the top-left taxel
- `target.pose`: fixed 6-taxel square footprint and `yaw_mod90_rad`
- `quality.label_source`: `manual_aruco`
- `annotation`: manual UI metadata, including image center, direction point,
  24 mm block side, and image/taxel footprint corners

Saved samples can be revisited with `Z` / `X`. Editing a saved sample and
pressing `F` overwrites only that label row; the original image and
`sample_id` remain unchanged.

New labels store a per-sample `calibration` snapshot. Older datasets that do
not have per-sample calibration can still be previewed and deleted, but their
annotations are view-only and cannot be edited safely.

## Preview A Dataset

Open a saved run and step through samples:

```bash
python3 -m block_position_prediction.data_collection_manual.preview_dataset \
  block_position_prediction/data_collection_manual/runs/YYYYMMDD_HHMMSS
```

Controls:

- `N`, `Right`, or `Space`: next sample
- `P` or `Left`: previous sample
- `Q` or `Esc`: quit

The preview uses the same tactile heatmap and fixed-footprint renderer as the
right side of the live UI.

## Useful Options

```bash
python3 -m block_position_prediction.data_collection_manual.app \
  --source 0 \
  --tactile-port SIMULATOR \
  --auto-calibrate-rate-hz 5 \
  --frame-width 1280 \
  --frame-height 720 \
  --new-run
```

Use `--help` to see all geometry and camera options:

```bash
python3 -m block_position_prediction.data_collection_manual.app --help
```
