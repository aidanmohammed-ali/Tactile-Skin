# AI Handoff Notes: ROS2 Tactile Block Pick Refactor

本文档面向下一个接手本项目的 AI 或工程师，记录本轮重构做了什么、为什么这么做、哪些地方已经验证、哪些地方还需要在真实硬件上继续确认。

## 1. 重构目标

原项目主要代码在 `src/Tactile_GroupProject/` 下：

- `robot_vision`: OpenCV/ArUco/YOLO 方块识别与标定。
- `dynamixel_python`: 机械臂控制、Tkinter 控制界面、夹爪服务。
- `dynamixel_sdk`: ROBOTIS Dynamixel SDK 的本地 vendor/ROS2 包。
- `my_interfaces`: 旧的 msg/srv 接口。

用户希望把“用 CV 识别方块并操控机械臂移动方块”的代码重构到 `ros_src/`，继续使用 ROS2，并完成：

- 文件结构合理、便于维护。
- 合并“标定/识别界面”和“机械臂控制界面”。
- 服务边界更简洁，运行期控制尽可能使用 topic。
- 早期版本使用 ROS2 Action，当前版本已简化为 topic 触发 Pick-Place 流程。
- UI 增加机械臂端口检测/设置。
- UI 增加网络摄像头地址输入。
- 视频流必须完整显示在窗口中，不裁切。

本轮实现采用“新代码全部放入 `ros_src/`，旧 `src/Tactile_GroupProject/` 不改动、不保证兼容旧入口”的策略。

## 2. Dynamixel SDK 取舍

结论：

- `src/Tactile_GroupProject/dynamixel_sdk` 是 ROBOTIS Dynamixel SDK 的本地 ROS2/vendor 包。
- `src/Tactile_GroupProject/dynamixel_python` 不是 SDK，是项目自写的 ROS2 机械臂控制/UI 包。
- 新 `ros_src/` 没有复制旧 `dynamixel_sdk`。
- 新硬件节点只在运行时 import Python SDK：

```bash
pip install dynamixel-sdk
```

如果没安装，硬件节点不会在 import 阶段崩掉，而是在 `/arm/set_connection` 连接真实串口时返回清晰错误：

```text
Python package 'dynamixel-sdk' is required. Install it with: pip install dynamixel-sdk
```

## 3. 新包结构

### `tactile_interfaces`

ROS2 接口包，集中定义少量 msg/srv。当前版本已移除 Action，运行期控制尽量走 topic。

关键文件：

- `msg/ArmPose.msg`: 末端笛卡尔位姿，字段为 `x y z angle_rad`。
- `msg/ArmMove.msg`: 末端移动命令，字段为 `target_pose duration_sec`。
- `msg/ArmState.msg`: 机械臂连接、忙碌状态、当前端口、当前位姿、关节角、夹爪位置、最后错误。
- `msg/BlockDetection.msg`: 方块检测结果，包含 pixel 坐标、grid 坐标、confidence、class_name、message。
- `srv/SetArmConnection.srv`: UI 设置机械臂端口并连接/断开。
- `srv/ListArmPorts.srv`: 扫描可用端口。

### `tactile_vision`

视觉包，负责相机、YOLO、ArUco 标定、坐标转换、annotated image 发布。

关键文件：

- `tactile_vision/camera.py`
  - `parse_camera_source()` 把 `"0"` 转成整数相机索引，URL 保持字符串。
  - `ThreadedCamera` 用后台线程持续读取最新帧，只保留最新帧，避免视频延迟堆积。
- `tactile_vision/detector.py`
  - `BlockDetector` 封装 YOLO 检测、ArUco 标定、board/grid 坐标换算、画框渲染。
  - 迁移并复用旧 `cv_aruco_src` 核心算法。
- `tactile_vision/image_messages.py`
  - `bgr_to_image_msg()` 把 OpenCV BGR frame 转成 `sensor_msgs/Image`。
- `tactile_vision/vision_node.py`
  - ROS2 节点 `vision_node`。
  - 发布 `/vision/annotated_image`。
  - 发布 `/vision/block_detection`。
  - 发布 `/vision/status`。
  - 订阅 `/vision/detect_trigger`。
  - 订阅 `/vision/confidence_threshold`，动态设置 YOLO confidence 阈值。
  - 订阅 `/vision/calibrate_trigger`。
  - 订阅 `/vision/camera_source`。
  - 检测时绘制所有 confidence 高于阈值的目标框；`/vision/block_detection` 仍发布最高 confidence 目标供 Pick-Place 使用。
  - `/vision/camera_source` 重连成功后，会把新 source 保存为下次启动默认值。
  - 不再使用 OpenCV 弹窗，视频统一交给 Tkinter UI 显示。

注意：`/vision/camera_source` 如果新地址打开失败，会尝试恢复旧 source，避免现场调试时把已有画面弄丢。
默认地址保存在 `~/.config/tactile_ros/camera_source.txt`；也可以通过 `camera_source_store` 参数或 `TACTILE_CAMERA_SOURCE_FILE` 环境变量改路径。UI 启动时也读取同一个保存值作为输入框默认地址。

### `tactile_arm`

机械臂包，负责硬件和仿真控制。硬件与仿真暴露同一套 ROS API。

关键文件：

- `tactile_arm/ports.py`
  - 扫描 `/dev/serial/by-id/*`、`/dev/ttyUSB*`、`/dev/ttyACM*`。
  - 排序时优先 USB，再 ACM，再 by-id。
- `tactile_arm/kinematics.py`
  - `ArmPose`
  - `ArmKinematics`
  - IK/FK
  - DXL position 与 radian 转换
  - 平滑插值工具
- `tactile_arm/dynamixel_driver.py`
  - 真实 Dynamixel 串口驱动。
  - 运行时 import `dynamixel_sdk`。
  - 负责连接、断开、初始化电机、设置关节角、读关节角、设置夹爪、disable torque。
- `tactile_arm/arm_node_base.py`
  - 硬件与仿真的公共 ROS2 节点逻辑。
  - 提供 `/arm/list_ports`、`/arm/set_connection`。
  - 订阅 `/arm/cartesian_goal`、`/arm/gripper_position`、`/arm/emergency_stop`。
  - 发布 `/arm/state`。
- `tactile_arm/hardware_arm_node.py`
  - 真实机械臂节点。
  - 默认不自动连接串口，交给 UI 选择端口后调用 `/arm/set_connection`。
- `tactile_arm/sim_arm_node.py`
  - 仿真机械臂节点。
  - 不依赖硬件，端口列表返回 `sim`。

### `tactile_task`

任务编排包，使用 topic 触发完整 Pick-Place 流程。

关键文件：

- `tactile_task/pick_place_node.py`
  - 订阅 `/task/pick_place_goal`。
  - 发布 `/vision/detect_trigger` 触发识别，并等待 `/vision/block_detection`。
  - 将 grid 坐标转换为 robot 坐标。
  - 发布 `/arm/cartesian_goal` 移动机械臂。
  - 发布 `/arm/gripper_position` 开合夹爪。
  - 发布 `/task/status` 供 UI log 显示阶段与结果。

当前默认 Pick-Place 流程：

1. Detect block.
2. Move above pick pose.
3. Open gripper.
4. Descend to pick height.
5. Close gripper.
6. Lift.
7. Move above place pose.
8. Descend to place pose.
9. Open gripper.
10. Retreat.

配置参数在 `tactile_bringup/config/system.yaml`：

- `grid_center_x`
- `grid_center_y`
- `grid_pitch_m`
- `pick_z`
- `carry_z`
- `retreat_z`
- `pick_angle_rad`
- `approach_duration_sec`
- `descend_duration_sec`
- `lift_duration_sec`
- `transfer_duration_sec`
- `retreat_duration_sec`

### `tactile_ui`

统一 Tkinter 操作界面。

关键文件：

- `tactile_ui/image_display.py`
  - `letterbox_size()` 计算完整显示视频的等比缩放尺寸。
  - `image_msg_to_bgr()` 将 ROS Image 转成 OpenCV frame。
  - `bgr_to_letterboxed_photo()` 将 frame 缩放进固定画布，使用 letterbox/pillarbox，不裁切。
- `tactile_ui/operator_ui_node.py`
  - 统一 UI。
  - 订阅 `/vision/annotated_image` 和 `/arm/state`。
  - 端口刷新、连接、断开。
  - 摄像头 URL 输入和应用。
  - Confidence 阈值输入和应用。
  - 标定、检测、夹爪、急停。
  - 通过 `/arm/cartesian_goal` topic 手动移动。
  - 手动输入放置 pose 后发布 `/task/pick_place_goal`。
  - `Detect Now` 是手动触发一次 YOLO 检测；默认不再持续 10Hz 跑 YOLO。
  - 右侧 log 区显示 topic status、服务返回和长错误，顶部 status 只放短摘要。

视频显示问题的处理点在 `image_display.py`：

- 目标是完整显示整帧。
- 使用 `scale = min(canvas_w / src_w, canvas_h / src_h)`。
- 多余区域用深色背景填充。
- 不做中心裁切。
- UI 渲染做了节流：只有新帧/窗口尺寸变化且满足最小渲染间隔时才重新编码 Tkinter 图像。

### `tactile_bringup`

启动与配置包。

关键文件：

- `launch/system.launch.py`
  - `mode:=sim` 启动 `sim_arm_node`。
  - `mode:=hardware` 启动 `hardware_arm_node`。
  - 默认 `start_ui:=true`。
- `config/system.yaml`
  - 视觉参数。
  - 硬件机械臂参数。
  - 仿真机械臂参数。
  - Pick-Place 任务参数。

## 4. ROS API 总览

Topics:

- `/vision/annotated_image`: `sensor_msgs/Image`
- `/vision/block_detection`: `tactile_interfaces/msg/BlockDetection`
- `/vision/status`: `std_msgs/String`
- `/vision/detect_trigger`: `std_msgs/Empty`
- `/vision/confidence_threshold`: `std_msgs/Float32`
- `/vision/calibrate_trigger`: `std_msgs/Bool`
- `/vision/camera_source`: `std_msgs/String`
- `/arm/state`: `tactile_interfaces/msg/ArmState`
- `/arm/cartesian_goal`: `tactile_interfaces/msg/ArmMove`
- `/arm/gripper_position`: `std_msgs/Int32`
- `/arm/emergency_stop`: `std_msgs/Empty`
- `/task/pick_place_goal`: `tactile_interfaces/msg/ArmPose`
- `/task/status`: `std_msgs/String`

Services:

- `/arm/list_ports`: `ListArmPorts`
- `/arm/set_connection`: `SetArmConnection`

## 5. 构建与运行

第一次构建：

```bash
cd /home/peterchen/Documents/tactile_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros_src
source install/setup.bash
```

每次新终端最短运行：

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=sim
```

真实机械臂：

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
ros2 launch tactile_bringup system.launch.py mode:=hardware
```

如果出现：

```text
Package 'tactile_bringup' not found, searching: ['/opt/ros/humble']
```

说明当前终端没有 source 工作区 overlay。执行：

```bash
cd /home/peterchen/Documents/tactile_ws
source install/setup.bash
```

## 6. 已验证内容

已执行：

```bash
colcon build --symlink-install --base-paths ros_src
python3 -m pytest -q ros_src/tactile_ui/test ros_src/tactile_arm/test ros_src/tactile_vision/test
```

结果：

- 6 个包 build 通过。
- 9 个轻量测试通过。

测试覆盖：

- 串口排序和推荐端口。
- 摄像头 source 解析。
- 视频完整显示的 letterbox 尺寸计算。
- IK/FK 可达点往返。
- IK 不可达点报错。

也做过：

```bash
ros2 pkg executables tactile_arm
ros2 pkg executables tactile_vision
ros2 pkg executables tactile_task
ros2 pkg executables tactile_ui
ros2 interface list | grep tactile_interfaces
```

节点实例化 smoke test 在沙箱中需要设置：

```bash
export ROS_LOG_DIR=/tmp
```

沙箱里 DDS/UDP 会打印权限警告，这是沙箱网络限制，不代表代码必然错误。

## 7. 尚未真实验收的内容

这些需要在用户本机的真实 ROS2/硬件环境继续验证：

- 网络摄像头真实 URL 是否能稳定打开。
- YOLO `block3.pt` 在当前机器依赖下是否可推理。
- ArUco 标定质量是否满足机械臂抓取。
- `/arm/set_connection` 对真实 `/dev/ttyUSB*` 权限是否正常。
- Dynamixel 电机 ID、方向、offset、夹爪开合位置是否与硬件一致。
- Pick-Place 中 grid-to-robot 坐标映射是否需要现场微调。

## 8. 后续建议

优先顺序：

1. 在 sim 模式启动 UI，确认界面和视频完整显示。
2. 只启动 vision，测试摄像头地址、标定、检测。
3. 启动 hardware 模式，先只测试端口连接、夹爪开合、急停。
4. 测试单次 `/arm/cartesian_goal` 小幅移动。
5. 在低速和安全高度下测试 `/task/pick_place_goal`。
6. 根据实测修改 `tactile_bringup/config/system.yaml` 中的坐标映射、高度、速度。

如果后续需要把 UI 做得更稳，建议：

- 增加 named place poses 下拉框。
- 增加 topic 级 cancel/stop-pick-place 按钮。
- 在 UI 中显示 `/vision/camera_source` 失败后恢复旧 source 的状态。
- 把 Pick-Place 每一步配置化，而不是写死在 task node 中。
