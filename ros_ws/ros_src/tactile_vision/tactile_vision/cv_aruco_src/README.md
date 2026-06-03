# Fixed-camera ArUco board calibration

This package is the production-oriented version of the board mapper:

1. Detect ArUco markers once.
2. Estimate image-to-board homography.
3. Optionally refine that homography with visible board holes.
4. Save JSON.
5. During runtime, load JSON and use only the 3x3 homography.

Runtime mapping is intentionally cheap: no marker detection and no hole
detection are needed unless the camera or board moves.

## Marker Choice

Use OpenCV `DICT_4X4_50`, marker IDs:

- `0`: top-left
- `1`: top-right
- `2`: bottom-right
- `3`: bottom-left

`DICT_4X4_50` is a good fit here because only four IDs are needed and the 4x4
payload has large cells, which is easier to detect at oblique angles and in
ordinary lab lighting. Print with a white quiet zone around each marker.

Recommended physical size:

- Marker black square side: `1.6 x hole_pitch`.
- Your hole pitch is `25 mm`, so print each marker as `40 mm x 40 mm`.
- White quiet zone: at least `8 mm`, or about `0.2 x marker side`.
- Place markers inside the grid corners. The tuned default is
  `marker_margin_grid = -1.40`, which means each 40 mm black square spans about
  `-0.2..1.4` grid units at a corner and covers the corner `2 x 2` hole block.
- Keep markers flat, matte, unwrinkled, and not under the robot's usual
  occlusion path.

The default code assumes marker side `1.6` grid units and margin `-1.40` grid
units. A negative margin means the marker is inside the hole grid. If your
physical marker placement differs, use `MarkerSpec` or
`marker_specs_from_centers()` and pass a custom `ArucoBoardConfig`.

## Print Markers

Generate an A4 300-DPI sheet:

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.print_markers `
  --output "cv_aruco_src\aruco_4x4_50_ids_0_3_a4.png" `
  --marker-mm 40 `
  --quiet-mm 8
```

Print at 100% scale. Do not let the printer or PDF viewer fit-to-page.

## Still-image Calibration

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.calibrate_image_cli `
  --image "C:\Users\30390\Desktop\calibration.jpg" `
  --output "cv_aruco_src\board_calibration.json" `
  --overlay "cv_aruco_src\calibration_overlay.jpg"
```

## Live Video UI

For a webcam:

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.video_ui `
  --source 0 `
  --calibration "cv_aruco_src\board_calibration.json"
```

For a network stream:

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.video_ui `
  --source "rtsp://user:password@192.168.1.10:554/stream1" `
  --calibration "cv_aruco_src\board_calibration.json"
```

Click **Calibrate** or press `C`. The JSON is saved and the grid is overlaid on
the live stream. Press `Q` or `Esc` to exit.

## API

```python
from cv_aruco_src import (
    ArucoBoardCalibrator,
    ArucoBoardConfig,
    load_calibration,
    pixel_to_board,
)

config = ArucoBoardConfig(
    hole_pitch_mm=25.0,
    aruco_dictionary="DICT_4X4_50",
    marker_size_grid=1.6,
    marker_margin_grid=-1.40,
)

calibrator = ArucoBoardCalibrator(config)
calibration = calibrator.calibrate(frame)  # NumPy image or image path
calibration.save_json("cv_aruco_src/board_calibration.json")

calibration = load_calibration("cv_aruco_src/board_calibration.json")
col, row = pixel_to_board(2016, 1512, calibration)
```

Coordinate convention: `(0, 0)` is the top-left hole; `(16, 11)` is the
bottom-right hole.
