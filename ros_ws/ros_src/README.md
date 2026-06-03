# Tactile ROS 2 Refactor

Refactored ROS 2 implementation for camera calibration, block detection, arm control, and pick-place.

## First Build

Run once after cloning or after interface/package changes:

```bash
cd /home/peterchen/Documents/tactile_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros_src
source install/setup.bash
```

## Every New Terminal

Use these commands when opening a new terminal:

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=sim
```

For the real Dynamixel arm:

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=hardware
```

If ROS says `Package 'tactile_bringup' not found`, the terminal has not sourced the workspace:

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
```

Optional shortcut for future terminals:

```bash
echo 'source /home/peterchen/Documents/tactile_ws/install/setup.bash' >> ~/.bashrc
```

Then open a new terminal and launch directly from the workspace:

```bash
cd /home/peterchen/Documents/tactile_ws
ros2 launch tactile_bringup system.launch.py mode:=sim
```

## Package Layout

- `tactile_interfaces`: shared messages and the two connection services.
- `tactile_vision`: camera capture, ArUco calibration, YOLO block detection, annotated image publishing.
- `tactile_arm`: hardware and simulated arm controllers with the same ROS API.
- `tactile_task`: topic-based pick-place coordinator.
- `tactile_ui`: unified Tkinter operator UI.
- `tactile_bringup`: launch files and YAML configuration.

## ROS API

Most runtime commands are topics, matching the simpler legacy control style:

- `/arm/cartesian_goal` (`tactile_interfaces/ArmMove`): move end effector to pose over `duration_sec`
- `/arm/gripper_position` (`std_msgs/Int32`): set absolute Dynamixel gripper position
- `/arm/emergency_stop` (`std_msgs/Empty`): stop current motion and disable torque
- `/arm/state` (`tactile_interfaces/ArmState`): connection, busy state, current pose, last error
- `/task/pick_place_goal` (`tactile_interfaces/ArmPose`): run pick-place with this final place pose
- `/task/status` (`std_msgs/String`): task stage and result log
- `/vision/detect_trigger` (`std_msgs/Empty`): run one YOLO detection
- `/vision/confidence_threshold` (`std_msgs/Float32`): set minimum detection confidence; all detections above it are drawn
- `/vision/calibrate_trigger` (`std_msgs/Bool`): calibrate current frame; `true` saves calibration
- `/vision/camera_source` (`std_msgs/String`): update network camera URL/source
- `/vision/block_detection` (`tactile_interfaces/BlockDetection`): latest detection result
- `/vision/annotated_image` (`sensor_msgs/Image`): full annotated video frame
- `/vision/status` (`std_msgs/String`): camera/calibration/detection status

When `/vision/camera_source` reconnects successfully, the new source is saved as the next default. By default it is stored at:

```text
~/.config/tactile_ros/camera_source.txt
```

Set `camera_source_store` in `system.yaml` or `TACTILE_CAMERA_SOURCE_FILE` to use a different file.

Only these services remain because they need synchronous return values:

- `/arm/list_ports` (`tactile_interfaces/ListArmPorts`)
- `/arm/set_connection` (`tactile_interfaces/SetArmConnection`)

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
- run pick-place with a manually entered place pose

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
python3 -m pytest -q ros_src/tactile_ui/test ros_src/tactile_arm/test ros_src/tactile_vision/test
```

## 在 WSL2 中连接 Dynamixel USB 串口

WSL2 默认不会自动接管 Windows USB 设备。Dynamixel USB 转串口通常需要通过 `usbipd-win` 挂载到 WSL。

在 PowerShell 管理员窗口安装：

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

插入 U2D2/USB 串口后，在 PowerShell 查看设备：

```powershell
usbipd list
```

找到对应设备的 `BUSID`，首次共享需要管理员权限：

```powershell
usbipd bind --busid <BUSID>
```

每次要给 WSL 使用时执行：

```powershell
usbipd attach --wsl --busid <BUSID>
```

在 Ubuntu 中确认设备：

```bash
lsusb
ls /dev/ttyUSB* /dev/ttyACM*
```

给当前用户串口权限：

```bash
sudo usermod -aG dialout $USER
```

然后退出并重新打开 Ubuntu。临时调试也可以：

```bash
sudo chmod a+rw /dev/ttyUSB0
```
