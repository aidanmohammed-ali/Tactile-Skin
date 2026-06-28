# Tactile ROS 2 Refactor

Refactored ROS 2 implementation for camera calibration, block detection, arm control, and pick-place.

## First Build

`<path_to_ros_workspace>` means the absolute path to this repository's ROS 2
workspace directory. In the current repository layout, that is:
`<path_to_project_directory>/ROS2/ros_ws`.

Run once after cloning or after interface/package changes:

```bash
cd <path_to_ros_workspace>
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros_src
source install/setup.bash
```

## Every New Terminal

Use these commands when opening a new terminal:

```bash
cd <path_to_ros_workspace>
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=sim
```

For the real Dynamixel arm:

```bash
cd <path_to_ros_workspace>
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=hardware
```

If ROS says `Package 'tactile_bringup' not found`, the terminal has not sourced the workspace:

```bash
cd <path_to_ros_workspace>
source install/setup.bash
```

Optional shortcut for future terminals:

```bash
echo 'source <path_to_ros_workspace>/install/setup.bash' >> ~/.bashrc
```

Then open a new terminal and launch directly from the workspace:

```bash
cd <path_to_ros_workspace>
ros2 launch tactile_bringup system.launch.py mode:=sim
```

## Package Layout

- `tactile_interfaces`: shared messages and synchronous service definitions.
- `tactile_vision`: camera capture, ArUco calibration, YOLO block detection, annotated image publishing.
- `tactile_arm`: hardware and simulated arm controllers with the same ROS API.
- `tactile_sensor`: tactile serial reader, pose prediction service, frame publishing, and heatmap display.
- `tactile_task`: task script runner; scripts in `tactile_task/tasks` call `commands.py`.
- `tactile_ui`: unified Tkinter operator UI.
- `tactile_bringup`: launch files and YAML configuration.

## Tactile Pose Node

`tactile_sensor/tactile_pose_node` reads the `16x8` tactile sensor, runs the
tactile pose model on request, optionally publishes tactile frames, and
directly displays an OpenCV heatmap window.

The reader runs continuously, but model inference only runs when
`/tactile_sensor/predict_pose` is called. Enabling frame streaming does not
continuously run the model.

For every new raw frame, the model path first calculates the historical
top-five average over the reader's sliding window and normalizes it. A spatial
`3x3` median filter is then applied to that result. Five consecutive filtered
results are averaged and converted into physical sensor order before
inference. The `raw_top5_average` and `top5_normalized` stream modes also apply
the spatial median filter before publishing or heatmap display.

With the heatmap window focused, press `T` to capture the current filtered
value of every taxel as a tare baseline. The baseline is subtracted and
clamped at zero before the final five-frame average. Press `T` again to disable
tare. The heatmap title shows the current `TARE ON/OFF` state.

### Parameters

Configure the node under `tactile_pose_node.ros__parameters` in
`tactile_bringup/config/system.yaml`:

| Parameter | Default | Description |
| --- | --- | --- |
| `checkpoint` | bundled `tactile_pose_best.pt` | Model checkpoint path. A relative filename is also searched under `block_position_prediction/model`. |
| `device` | `"auto"` | PyTorch device: `auto`, `cpu`, or `cuda`. |
| `tactile_port` | `"SIMULATOR"` | Tactile serial port, such as `/dev/ttyACM0`, or `SIMULATOR`. |
| `tactile_baud` | `115200` | Serial baud rate. |
| `confidence_threshold` | `0.5` | Minimum model presence confidence for `detected=true`. |
| `legacy_force_threshold` | `1.0` | Empty-sensor fallback for old checkpoints without a presence head. |
| `service_name` | `"/tactile_sensor/predict_pose"` | Pose prediction service name. |
| `stream_frames` | `false` | Publish once for every tactile frame received; there is no fixed ROS publish timer. |
| `stream_topic` | `"/tactile_sensor/frame"` | `Float32MultiArray` frame topic. |
| `stream_frame_kind` | `"raw_latest"` | Published/displayed data: `raw_latest`, `raw_top5_average`, `top5_normalized`, or `processed`. Both top-5 modes include the `3x3` median filter. |
| `display_heatmap` | `true` | Open the heatmap window from `tactile_pose_node`. |
| `heatmap_window` | `"Tactile Sensor Heatmap"` | OpenCV window title. |
| `heatmap_width` | `640` | Requested heatmap width in pixels. |
| `heatmap_value_max` | `0.0` | Color scale maximum. `0.0` automatically selects `65535` for raw data or `1.0` for normalized data. |
| `heatmap_flip_x` | `true` | Flip hardware column order for physical left-to-right display. |
| `heatmap_rate_hz` | `30.0` | Maximum heatmap refresh rate. |

Current hardware configuration:

```yaml
tactile_pose_node:
  ros__parameters:
    device: "auto"
    tactile_port: "/dev/ttyACM0"
    tactile_baud: 115200
    confidence_threshold: 0.5
    legacy_force_threshold: 1.0
    service_name: "/tactile_sensor/predict_pose"
    stream_frames: true
    stream_topic: "/tactile_sensor/frame"
    stream_frame_kind: "top5_normalized"
    display_heatmap: true
    heatmap_window: "Tactile Sensor Heatmap"
    heatmap_width: 640
    heatmap_value_max: 0.0
    heatmap_flip_x: true
    heatmap_rate_hz: 30.0
```

Use `stream_frame_kind: "top5_normalized"` to display the historical top-5
average with the spatial median filter. Use `"raw_latest"` to inspect each
latest raw hardware frame.

Streaming is event-driven: hardware input publishes once for each complete
serial frame received. In `SIMULATOR` mode, frames are generated at the
reader's simulation rate, which defaults to approximately `60Hz`.
`heatmap_rate_hz` only limits GUI redraws and does not limit the frame topic.

### Build and Run

The node depends on `pyserial`, PyTorch, NumPy, and OpenCV. Install the Python
packages required by the tactile reader and model if they are not already
available:

```bash
python3 -m pip install pyserial torch numpy opencv-python
```

Build after adding `tactile_sensor` or changing `GetTactilePose.srv`:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros_src
source install/setup.bash
```

Run only the tactile node with the YAML configuration:

```bash
ros2 run tactile_sensor tactile_pose_node \
  --ros-args \
  --params-file ros_src/tactile_bringup/config/system.yaml
```

Override individual parameters from the command line:

```bash
ros2 run tactile_sensor tactile_pose_node \
  --ros-args \
  -p tactile_port:=/dev/ttyACM0 \
  -p stream_frames:=true \
  -p stream_frame_kind:=top5_normalized \
  -p display_heatmap:=true
```

The heatmap requires a graphical session. Set `display_heatmap:=false` when
running headless or over SSH without X/Wayland forwarding.

### Call the Pose Model

Call the prediction service with an empty request:

```bash
ros2 service call /tactile_sensor/predict_pose \
  tactile_interfaces/srv/GetTactilePose "{}"
```

The response contains:

- `success`: sensor data and model inference were available
- `detected`: confidence passed the configured threshold
- `x`, `y`: continuous block center in taxel coordinates
- `angle_rad`, `angle_deg`: block yaw modulo 90 degrees
- `confidence`: model presence confidence
- `fully_inside_sensor`: predicted block footprint is inside the sensor
- `message`: success, no-block, or error status

For the `16x8` sensor, `x` runs along 16 columns and `y` along 8 rows. One
taxel unit is one sensor pitch. With a 4 mm pitch:

```text
x offset from center = (x - 7.5) * 4 mm
y offset from center = (y - 3.5) * 4 mm
```

Inspect the streamed frame topic:

```bash
ros2 topic echo /tactile_sensor/frame
ros2 topic hz /tactile_sensor/frame
```

Task scripts call the same service through
`robot.detect_tactile_pose(timeout=3.0)`. The service name passed to
`task_node` must match:

```yaml
task_node:
  ros__parameters:
    tactile_pose_service: "/tactile_sensor/predict_pose"
```

### Launch File Integration

The tactile node is already included in
`tactile_bringup/launch/system.launch.py`:

```python
Node(
    package="tactile_sensor",
    executable="tactile_pose_node",
    name="tactile_pose_node",
    output="screen",
    parameters=[config_path],
),
```

The project launch file already includes this node before `task_node`, so the
normal launch command starts the tactile reader, model service, frame topic,
and heatmap:

```bash
ros2 launch tactile_bringup system.launch.py mode:=hardware
```

To run without the Tkinter operator UI while keeping the tactile heatmap:

```bash
ros2 launch tactile_bringup system.launch.py \
  mode:=hardware \
  start_ui:=false
```

The bringup package declares the corresponding runtime dependency:

```xml
<depend>tactile_sensor</depend>
```

## Tactile Pick Place

`tactile_task/tasks/tactile_pick_place.py` extends the original
`pick_place.py` flow with tactile correction after grasping:

1. Run vision detection and convert the detected grid position into robot
   `pick_x` and `pick_y`.
2. Calculate a distance-dependent `pick_z`.
3. Approach, open the gripper, descend, close the gripper, and lift.
4. Wait for `tactile_pose_wait_sec`, then call
   `robot.detect_tactile_pose()`.
5. Convert the predicted taxel position into a metric offset.
6. Use the tactile Y offset to move the requested place point radially toward
   or away from the robot origin.
7. Place and retreat using the corrected `place_x` and `place_y`.

### Pick Height Correction

The current pick height calculation is:

```text
distance = sqrt(pick_x² + pick_y²)
pick_z_corrected = pick_z + (distance - 0.1) * 0.1
```

For example:

- at `distance = 0.10 m`, no height correction is applied
- at `distance = 0.15 m`, `pick_z` is raised by `0.005 m`
- at `distance = 0.20 m`, `pick_z` is raised by `0.010 m`

The current implementation does not clamp the correction. Positions closer
than `0.10 m` lower the resulting pick height. The configuration keys
`pick_z_raise_start_distance_m`, `pick_z_raise_end_distance_m`, and
`pick_z_max_raise_m` are present, but `_pick_height()` currently uses the
hard-coded formula above rather than those values.

### Tactile Place Correction

The model reports the block center in continuous taxel coordinates. The
configured sensor center is `(7.5, 3.5)` and the current taxel pitch is
`0.004 m`:

```text
offset_x = (prediction.x - 7.5) * 0.004
offset_y = (prediction.y - 3.5) * 0.004
```

The task currently uses only `offset_y` for placement correction. For an
original place point `(x, y)`, it calculates:

```text
radius = sqrt(x² + y²)
corrected_x = x + offset_y * x / radius
corrected_y = y + offset_y * y / radius
```

This preserves the direction of the place point from the robot origin while
changing its distance from the origin. If the requested place point is the
origin, the offset is applied directly to its Y coordinate.

The calculated tactile `offset_x`, `offset_y`, and corrected place position
are written to the task log before the placement move.

### Reachable Angle Adjustment

All task moves use `Robot.move()`. Before publishing an arm command, it checks
the target pose with the same inverse kinematics implementation as
`tactile_arm`. If the requested angle is unreachable, it tests:

```text
angle, angle + 0.01, angle + 0.02, ...
```

The search stops at `-1.0 rad`. For example, a requested angle of `-1.57`
tries `-1.57`, `-1.56`, `-1.55`, and so on. The first reachable angle is sent
to the arm. If no angle is reachable by `-1.0 rad`, the task fails without
sending that move.

### Main Configuration

Important defaults in `tactile_pick_place.py` include:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `pick_z` | `0.02` | Base pick height before distance correction. |
| `carry_z` | `0.12` | Height used when approaching and transferring. |
| `retreat_z` | `0.08` | Height immediately after grasping. |
| `pick_angle_rad` | `-1.57` | Requested gripper angle; IK may adjust it toward `-1.0`. |
| `tactile_pose_wait_sec` | `2.0` | Delay after lifting before tactile inference. |
| `tactile_pose_timeout_sec` | `3.0` | Tactile pose service timeout. |
| `tactile_center_x` | `7.5` | Sensor center column. |
| `tactile_center_y` | `3.5` | Sensor center row. |
| `tactile_pitch_m` | `0.004` | Physical distance represented by one taxel. |
| `gripper_open_position` | `1400` | Dynamixel open command used by this task. |
| `gripper_close_position` | `2300` | Dynamixel close command used by this task. |

Run the task by publishing its module name:

```bash
ros2 topic pub --once /task/run tactile_interfaces/msg/TaskCommand \
  "{id: tactile-pick-1, task: tactile_pick_place, args_json: '{}'}"
```

Task arguments can override top-level configuration and the nested place
pose. For example:

```bash
ros2 topic pub --once /task/run tactile_interfaces/msg/TaskCommand \
  "{id: tactile-pick-2, task: tactile_pick_place, args_json: \
'{\"pick_z\": 0.025, \"place\": {\"x\": 0.12, \"y\": 0.10, \"z\": 0.04, \
\"angle_rad\": -1.57}}'}"
```

## ROS API

Most runtime commands are topics with explicit command ids:

- `/arm/command` (`tactile_interfaces/ArmCommand`): move, gripper, or stop command
- `/arm/state` (`tactile_interfaces/ArmState`): connection, busy state, current pose, command completion, last error
- `/task/run` (`tactile_interfaces/TaskCommand`): run a named task script with JSON args
- `/task/cancel` (`std_msgs/Empty`): request task cancellation and arm stop
- `/task/state` (`tactile_interfaces/TaskState`): task stage, result, and log message
- `/vision/detect_trigger` (`std_msgs/Empty`): run one YOLO detection
- `/vision/confidence_threshold` (`std_msgs/Float32`): set minimum detection confidence; all detections above it are drawn
- `/vision/calibrate_trigger` (`std_msgs/Bool`): calibrate current frame; `true` saves calibration
- `/vision/camera_source` (`std_msgs/String`): update network camera URL/source
- `/vision/block_detection` (`tactile_interfaces/BlockDetection`): latest detection result
- `/vision/annotated_image` (`sensor_msgs/Image`): full annotated video frame, published with low-latency video QoS
- `/vision/status` (`std_msgs/String`): camera/calibration/detection status
- `/tactile_sensor/frame` (`std_msgs/Float32MultiArray`): optional row-major `8x16` tactile frame selected by `stream_frame_kind`

When `/vision/camera_source` reconnects successfully, the new source is saved as the next default. By default it is stored at:

```text
~/.config/tactile_ros/camera_source.txt
```

Set `camera_source_store` in `system.yaml` or `TACTILE_CAMERA_SOURCE_FILE` to use a different file.

## Low-Latency Video

The video preview is configured to avoid building a queue of old frames. `vision_node` and `operator_ui_node` use `image_qos_depth: 1` and `image_qos_reliability: "best_effort"` for `/vision/annotated_image`, while control topics keep their existing reliable behavior.

The default capture resolution and publish rate remain `1280x720` at `10Hz` so detection and calibration quality are not reduced by default. If the UI still shows delayed frames, check the camera app/device first: prefer a low-latency MJPEG or RTSP endpoint, 720p at 10-15fps, and disable camera-side buffering, high-quality-priority, or cloud relay modes when available.

Only these services remain because they need synchronous return values:

- `/arm/list_ports` (`tactile_interfaces/ListArmPorts`)
- `/arm/set_connection` (`tactile_interfaces/SetArmConnection`)
- `/tactile_sensor/predict_pose` (`tactile_interfaces/GetTactilePose`)

## UI Features

The UI can:

- scan `/dev/ttyUSB*`, `/dev/ttyACM*`, and `/dev/serial/by-id/*`
- connect/disconnect the arm port
- set the network camera URL
- set the YOLO confidence threshold for visible detections
- calibrate and run one-shot `Detect Now` block detection
- display the full video frame without cropping
- show long errors and topic status messages in an in-window log panel
- manually move the arm
- open/close the gripper
- emergency stop
- run the `pick_place` task with a manually entered place pose

## Dynamixel SDK

The old `src/Tactile_GroupProject/dynamixel_sdk` folder is a local/vendor copy of the ROBOTIS SDK. This refactor does not copy it into `ros_src`.

Install the Python SDK for hardware mode:

```bash
pip install dynamixel-sdk
```

If it is missing, `/arm/set_connection` returns a clear error.

## Dynamixel Port Diagnostics

Run only one arm controller at a time. Do not launch the old `src/Tactile_GroupProject` control stack together with this refactor, because both can try to own the same `/dev/ttyUSB*` port.

Before hardware Pick Place, a healthy node list should contain one arm node only:

```bash
ros2 node list
```

If Pick Place reports `Port is in use!` or the port remains busy after Disconnect, check which process owns the serial device:

```bash
lsof /dev/ttyUSB0
```

The new hardware node serializes Dynamixel SDK reads and writes internally, so repeated `Port is in use!` errors usually mean an old `control_node`, a second `hardware_arm_node`, or another process still has the port.

## Checks

```bash
colcon build --symlink-install --base-paths ros_src
python3 -m pytest -q \
  ros_src/tactile_sensor/test \
  ros_src/tactile_task/test \
  ros_src/tactile_ui/test \
  ros_src/tactile_arm/test \
  ros_src/tactile_vision/test
```

## Connecting a Dynamixel USB Serial Adapter in WSL2

WSL2 does not automatically take control of Windows USB devices. Dynamixel USB-to-serial adapters normally need to be attached to WSL through `usbipd-win`.

Install it from an elevated PowerShell window:

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

After connecting the U2D2 or other USB serial adapter, list the devices in PowerShell:

```powershell
usbipd list
```

Find the required device's `BUSID`. The initial bind requires administrator privileges:

```powershell
usbipd bind --busid <BUSID>
```

Attach the device whenever it needs to be used from WSL:

```powershell
usbipd attach --wsl --busid <BUSID>
```

Confirm the device from Ubuntu:

```bash
lsusb
ls /dev/ttyUSB* /dev/ttyACM*
```

Grant the current user serial-port access:

```bash
sudo usermod -aG dialout $USER
```

Close and reopen Ubuntu afterwards. For temporary debugging, you may instead run:

```bash
sudo chmod a+rw /dev/ttyUSB0
```
