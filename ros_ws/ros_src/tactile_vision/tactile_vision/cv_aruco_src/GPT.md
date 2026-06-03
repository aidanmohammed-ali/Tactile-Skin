# cv_aruco_src Project Handoff Notes

Last updated: 2026-05-21

This file is the migration / handoff note for future GPT/Codex sessions. Keep it
up to date whenever `cv_aruco_src` calibration logic, physical marker layout,
CLI options, saved calibration files, or test expectations change.

## 维护约定

后续任何模型继续维护这个项目时，请优先阅读并更新本文件。尤其是以下变化必须同步记录：

- ArUco 字典、marker ID、marker 尺寸或 marker 物理位置改变。
- 孔距、板子行列数、坐标系定义改变。
- 标定流程、孔检测/refine 策略、RANSAC 阈值改变。
- CLI/UI 命令、默认参数、JSON schema 改变。
- 新增或删除关键文件、测试、依赖。
- 重新调试出新的标定结果或误差统计。

## 项目目标

目标是把任意图像像素坐标 `(u, v)` 快速转换为黑色塑料孔板上的连续板坐标 `(col, row)`。

当前推荐方案：

1. 固定相机、固定板。
2. 一次性检测 4 个 ArUco marker。
3. 使用 marker 的已知板坐标估计 homography。
4. 在标定阶段检测可见孔中心，用孔点 refine homography。
5. 保存 `board_calibration.json`。
6. 运行时只加载 JSON，用 3x3 homography 做坐标变换，不再检测 marker 或孔。

这样运行时非常快，核心开销只是一次矩阵投影。

## 坐标系

板坐标使用连续 grid units，不直接输出毫米：

- 原点 `(0, 0)`：左上角孔中心。
- `x = col`：列坐标，范围约 `0..16`。
- `y = row`：行坐标，范围约 `0..11`。
- 板子尺寸：`12 x 17` 个孔，即 `rows=12`, `cols=17`。
- 孔距：`25 mm`。
- 右下角孔中心是 `(16, 11)`。

如果需要毫米坐标：

```python
x_mm = col * 25.0
y_mm = row * 25.0
```

## 当前物理参数

当前已按用户的新图 `C:\Users\30390\Desktop\test1.jpg` 调试稳定：

- ArUco dictionary: `DICT_4X4_50`
- Marker IDs:
  - `0`: top-left
  - `1`: top-right
  - `2`: bottom-right
  - `3`: bottom-left
- Marker 黑色方块边长：`40 mm`
- 孔距：`25 mm`
- `marker_size_grid = 1.6`
- `marker_margin_grid = -1.40`

`marker_size_grid = 1.6` 的含义是：

```text
40 mm / 25 mm = 1.6 grid units
```

`marker_margin_grid = -1.40` 的含义是 marker 更靠内。以左上角 marker 为例，
黑色方块大约覆盖：

```text
x: -0.2 .. 1.4
y: -0.2 .. 1.4
```

也就是会遮挡角落 `2 x 2` 的孔点。这正是当前 `test1.jpg` 里的摆放方式。

四个角的默认 marker 黑色方块板坐标大约为：

```text
top-left:     (-0.2, -0.2) -> (1.4, 1.4)
top-right:    (14.6, -0.2) -> (16.2, 1.4)
bottom-right: (14.6, 9.6)  -> (16.2, 11.2)
bottom-left:  (-0.2, 9.6)  -> (1.4, 11.2)
```

注意：如果 marker 实际贴得更靠外或更靠内，必须调整 `marker_margin_grid`，
否则初始 homography 会偏，孔 refine 也可能不稳定。

## 为什么使用 DICT_4X4_50

当前只需要 4 个 ID，因此 `DICT_4X4_50` 足够。4x4 marker 单元较大，在普通实验室光照、
轻微透视、相机分辨率有限时更容易稳定识别。marker 需要保留白色 quiet zone。

当前不建议随意换字典。若必须换，请同步修改：

- `ArucoBoardConfig.aruco_dictionary`
- `calibrate_image_cli.py --dictionary` 默认值
- `video_ui.py --dictionary` 默认值
- `print_markers.py --dictionary` 默认值
- README 和本文件
- 已保存 JSON 中的 `aruco_dictionary`

## 打印建议

打印 marker：

- 黑色方块边长：`40 mm`
- 白色 quiet zone：至少 `8 mm`
- 打印比例：`100%`，不要 fit-to-page
- 尽量使用哑光纸，贴平，避免皱褶和反光

生成打印图：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.print_markers `
  --output "cv_aruco_src\aruco_4x4_50_ids_0_3_a4_40mm.png" `
  --marker-mm 40 `
  --quiet-mm 8
```

## 当前代码结构

`cv_aruco_src` 是当前生产方向代码。早期 `cv_src` 是通用/原型版本，仍可参考，
但后续建议优先维护 `cv_aruco_src`。

关键文件：

- `config.py`
  - `BoardGeometry`: 孔板行列数，默认 `rows=12`, `cols=17`。
  - `HoleRefineConfig`: 孔检测/refine 参数。
  - `ArucoBoardConfig`: 主配置，包含孔距、字典、marker 尺寸、marker 位置。
  - `default_marker_specs()`: 根据尺寸和 margin 生成 4 个角 marker 的板坐标。
  - `marker_specs_from_centers()`: 如果以后实测 marker 中心，可用中心点生成 specs。

- `calibration.py`
  - `ArucoBoardCalibrator`: 标定主类。
  - `calibrate_image()`: 便捷函数。
  - 流程：读图 -> 灰度 -> 检测 ArUco -> 初始 homography -> 检测孔候选 -> 匹配 lattice -> RANSAC/LS refine。
  - 只在标定阶段检测孔，运行时不检测孔。

- `transform.py`
  - `BoardCalibration`: 标定结果 dataclass。
  - `pixel_to_board()`: 像素坐标到板坐标。
  - `board_to_pixel()`: 板坐标到像素坐标。
  - `load_calibration()` / `save_calibration()`。

- `overlay.py`
  - `draw_board_overlay()`: 在图像上画孔网格、边界、文字。
  - `save_board_overlay()`。

- `calibrate_image_cli.py`
  - 静态图片标定 CLI。

- `video_ui.py`
  - OpenCV HighGUI UI。
  - 用 `cv2.VideoCapture` 打开本地摄像头、HTTP 或 RTSP 网络流。
  - 点击 `Calibrate` 或按 `C` 标定。
  - 标定后保存 JSON，并把 overlay 叠加到视频流。

- `print_markers.py`
  - 生成可打印 marker sheet。

- `README.md`
  - 用户使用说明。

- `GPT.md`
  - 本交接文件。后续维护请更新。

## 公开 API

包入口 `cv_aruco_src/__init__.py` 当前导出：

```python
from cv_aruco_src import (
    ArucoBoardCalibrator,
    ArucoBoardConfig,
    BoardCalibration,
    BoardGeometry,
    HoleRefineConfig,
    MarkerSpec,
    board_to_pixel,
    calibrate_image,
    default_marker_specs,
    draw_board_overlay,
    load_calibration,
    marker_specs_from_centers,
    pixel_to_board,
    save_board_overlay,
    save_calibration,
)
```

典型 API 用法：

```python
from cv_aruco_src import ArucoBoardCalibrator, ArucoBoardConfig, load_calibration, pixel_to_board

config = ArucoBoardConfig(
    hole_pitch_mm=25.0,
    aruco_dictionary="DICT_4X4_50",
    marker_size_grid=1.6,
    marker_margin_grid=-1.40,
)

calibration = ArucoBoardCalibrator(config).calibrate("C:/Users/30390/Desktop/test1.jpg")
calibration.save_json("cv_aruco_src/board_calibration.json")

calibration = load_calibration("cv_aruco_src/board_calibration.json")
col, row = pixel_to_board(2016, 1512, calibration)
```

## CLI 用法

静态图标定：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.calibrate_image_cli `
  --image "C:\Users\30390\Desktop\test1.jpg" `
  --output "cv_aruco_src\board_calibration.json" `
  --overlay "cv_aruco_src\board_calibration_overlay.jpg"
```

默认参数已经是当前稳定参数：

```text
--dictionary DICT_4X4_50
--hole-pitch-mm 25.0
--marker-size-grid 1.6
--marker-margin-grid -1.40
```

如果只想用 ArUco，不用孔 refine：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.calibrate_image_cli `
  --image "C:\Users\30390\Desktop\test1.jpg" `
  --output "cv_aruco_src\board_calibration_no_refine.json" `
  --no-hole-refine
```

视频 UI，本地摄像头：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.video_ui `
  --source 0 `
  --calibration "cv_aruco_src\board_calibration.json"
```

视频 UI，网络流：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.video_ui `
  --source "rtsp://user:password@192.168.1.10:554/stream1" `
  --calibration "cv_aruco_src\board_calibration.json"
```

UI 操作：

- 点击 `Calibrate` 或按 `C`: 对当前帧标定并保存 JSON。
- 按 `Q` 或 `Esc`: 退出。
- 如果已有 calibration JSON，UI 启动时会加载并叠加网格。

## 依赖和环境

当前 `requirements.txt`：

```text
numpy>=1.26
opencv-contrib-python>=4.8
```

需要 `opencv-contrib-python`，因为 OpenCV ArUco 模块在 contrib 包里。

当前虚拟环境位于：

```text
.venv
```

常用测试命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m compileall cv_aruco_src tests
```

最近一次验证结果：

```text
Ran 11 tests in 0.955s
OK
```

## 当前标定结果

基于图片：

```text
C:\Users\30390\Desktop\test1.jpg
```

命令：

```powershell
.\.venv\Scripts\python.exe -m cv_aruco_src.calibrate_image_cli `
  --image "C:\Users\30390\Desktop\test1.jpg" `
  --output "cv_aruco_src\board_calibration.json" `
  --overlay "cv_aruco_src\board_calibration_overlay.jpg"
```

输出统计：

```text
method: aruco+hole_refine
marker_ids: (0, 2, 1, 3)
refined_holes: 130
median_grid_error: 0.01436098153329839
median_pixel_error: 2.679465750212613
```

生成文件：

- `cv_aruco_src/board_calibration.json`
- `cv_aruco_src/board_calibration_overlay.jpg`
- `cv_aruco_src/test1_board_calibration_default.json`
- `cv_aruco_src/test1_board_overlay_default.jpg`
- `cv_aruco_src/test1_board_calibration_inner_40mm.json`
- `cv_aruco_src/test1_board_overlay_inner_40mm.jpg`

其中 `board_calibration.json` 是默认运行时会加载的标定结果。

## 参数调试记录

在 `test1.jpg` 上扫描过 `marker_margin_grid`。固定条件：

```text
hole_pitch_mm = 25.0
marker_size_grid = 1.6
dictionary = DICT_4X4_50
hole_refine = enabled
```

关键结果：

```text
margin -0.50: holes 34,  inliers 21,  median_grid 0.1066
margin -0.70: holes 25,  inliers 24,  median_grid 0.0871
margin -0.90: holes 41,  inliers 38,  median_grid 0.0294
margin -1.00: holes 61,  inliers 63,  median_grid 0.0458
margin -1.10: holes 108, inliers 103, median_grid 0.0129
margin -1.20: holes 130, inliers 135, median_grid 0.0259
margin -1.25: holes 130, inliers 137, median_grid 0.0215
margin -1.30: holes 130, inliers 140, median_grid 0.0214
margin -1.35: holes 131, inliers 142, median_grid 0.0168
margin -1.40: holes 130, inliers 143, median_grid 0.0144
margin -1.50: worsened, median_grid 0.0266
```

更细扫描中 `-1.40` 是当前最好的稳定折中：

```text
margin -1.38: holes 130, inliers 143, median_grid 0.01583
margin -1.40: holes 130, inliers 143, median_grid 0.01436, mean_grid 0.02210, median_px 2.6795
margin -1.42: holes 128, inliers 143, median_grid 0.01479
```

因此默认值设为：

```python
marker_margin_grid = -1.40
```

## JSON 内容说明

`BoardCalibration` JSON 主要包含：

- `aruco_dictionary`
- `rows`, `cols`
- `homography_image_to_board`
- `homography_board_to_image`
- `marker_ids`
- `quality`
- `metadata`

`quality` 里保存 reprojection / refinement 误差统计。

`metadata` 里记录本次标定使用的关键参数，例如：

- `hole_pitch_mm`
- `marker_size_mm`
- `marker_inner_offset_mm`
- `configured_marker_size_grid`
- `configured_marker_margin_grid`
- `detected_marker_ids`
- `used_hole_points`

后续迁移时，如果只有 JSON，不需要重新标定就可以做运行时坐标转换。

## 运行时逻辑

运行时不应该每帧检测孔，也不应该每帧检测 marker。固定相机/固定板情况下：

```python
from cv_aruco_src import load_calibration, pixel_to_board

calibration = load_calibration("cv_aruco_src/board_calibration.json")
col, row = pixel_to_board(u, v, calibration)
```

只有以下情况需要重新标定：

- 相机移动。
- 板子移动。
- 焦距、分辨率、裁剪方式改变。
- marker 被重新贴过。
- 使用新相机或新镜头。

## 孔 refine 策略

孔 refine 当前只用于标定阶段。大致流程：

1. 根据 ArUco marker 建立初始 image-to-board homography。
2. 对灰度图做 adaptive threshold，检测亮色方形孔候选。
3. 用面积、长宽比、fill ratio 过滤候选。
4. 把候选点通过初始 homography 投影到 board 坐标。
5. 舍入到最近 lattice 点，检查误差是否小于 `assignment_max_grid_error`。
6. 忽略被遮挡的孔；只使用可见孔。
7. 使用 marker corners + visible hole centers 重新估计 homography。
8. 保存误差统计。

当前默认 `HoleRefineConfig`：

```text
min_area_px = 25.0
max_area_px = 1500.0
max_area_fraction = 0.002
adaptive_block_size = 31
adaptive_c = -5.0
min_aspect_ratio = 0.45
max_aspect_ratio = 2.2
min_fill_ratio = 0.18
assignment_max_grid_error = 0.35
board_margin_grid = 0.55
min_holes = 12
```

这些参数是为了适配黑色板上白/亮色孔、普通手机/摄像头俯视图、机器人随机遮挡。

## 已知注意事项

- 角落 marker 会遮挡角上 2x2 孔，这是预期行为。
- `marker_margin_grid` 是最敏感参数之一。物理位置变了必须重新调。
- 如果 overlay 整体平移或缩放错，优先检查 marker 物理位置和 `marker_margin_grid`。
- 如果 overlay 在中心准但边缘歪，可能是镜头畸变或 marker/孔板不平整；后续可加入 camera matrix/distortion undistort。
- 如果只看到 2 个 marker，homography 不可靠；建议标定时确保 4 个 marker 都清晰可见。
- `board_calibration.json` 与相机位姿绑定。不要把一台相机的 JSON 用到另一台相机。
- OpenCV 版本差异可能影响 ArUco API；代码里已经兼容新旧 detector API，但升级 OpenCV 后仍建议跑测试。

## 后续建议

短期：

- 每次标定后检查 overlay 图，确认所有可见孔上网格点贴合。
- 在视频 UI 中增加保存当前帧截图功能，便于调试失败案例。
- 增加 CLI 输出 `mean_grid_error`, `max_grid_error`, `used_hole_points`，方便快速判断质量。

中期：

- 支持 camera intrinsic calibration 和 undistort，提高边缘精度。
- 加入多帧标定：采集几帧稳定结果，取鲁棒平均 homography。
- 在 UI 中显示 marker 检测数量、孔 refine 数量、median error。

长期：

- 如果机器人遮挡经常覆盖 marker，可考虑把 marker 放到永远不会被遮挡的板外刚性区域。
- 如果需要毫米级物理定位，建议增加相机内参标定和实际孔距误差校正。
