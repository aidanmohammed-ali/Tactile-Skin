from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .aruco import PaperCalibration
from .geometry import SheetConfig
from .labels import BLOCK_SIDE_MM, ManualAnnotation, preview_from_annotation
from .preview_render import draw_tactile_preview
from .tactile import TactileSnapshot


@dataclass(frozen=True)
class UiStatus:
    camera: str = ""
    source_text: str = ""
    source_editing: bool = False
    calibration: str = "not calibrated"
    annotation: str = "no annotation"
    writer: str = ""
    mode: str = "live"
    auto_calibration: bool = True
    dataset: str = ""
    sample_position: str = ""
    tactile_ports: tuple[str, ...] = ("SIMULATOR",)
    tactile_dropdown_open: bool = False


def draw_overlay(
    frame: Any,
    config: SheetConfig,
    calibration: PaperCalibration | None,
    annotation: ManualAnnotation | None,
    status: UiStatus,
    tactile: TactileSnapshot | None = None,
    tactile_values: Sequence[float] | None = None,
) -> Any:
    import cv2
    import numpy as np

    display = frame.copy()
    if calibration is not None:
        _draw_sensor_overlay(display, config, calibration)
    if annotation is not None:
        _draw_manual_annotation(display, config, annotation)
    _draw_status_panel(display, calibration, annotation, status)
    side = _draw_tactile_side_panel(tactile, display.shape[0], config, annotation, status, tactile_values=tactile_values)
    if side.shape[0] != display.shape[0]:
        side = cv2.resize(side, (side.shape[1], display.shape[0]), interpolation=cv2.INTER_AREA)
    return np.hstack([display, side])


def _draw_sensor_overlay(frame: Any, config: SheetConfig, calibration: PaperCalibration) -> None:
    import cv2

    x1, y1, x2, y2 = config.sensor_rect_mm
    corners = [
        _rounded(calibration.paper_to_image_px(x1, y1)),
        _rounded(calibration.paper_to_image_px(x2, y1)),
        _rounded(calibration.paper_to_image_px(x2, y2)),
        _rounded(calibration.paper_to_image_px(x1, y2)),
    ]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        cv2.line(frame, start, end, (50, 220, 80), 2, cv2.LINE_AA)
    for row in range(config.sensor.rows):
        for col in range(config.sensor.cols):
            sensor_point = config.sensor.taxel_center_mm(col, row)
            paper_point = config.sensor_to_paper_mm(*sensor_point)
            center = _rounded(calibration.paper_to_image_px(*paper_point))
            cv2.circle(frame, center, 3, (0, 210, 255), -1, cv2.LINE_AA)


def _draw_manual_annotation(frame: Any, _config: SheetConfig, annotation: ManualAnnotation) -> None:
    import cv2
    import numpy as np

    pts = np.asarray([_rounded(point) for point in annotation.footprint_corners_px], dtype=np.int32).reshape((-1, 1, 2))
    color = (0, 80, 255) if annotation.fully_inside_sensor else (0, 0, 255)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    center = _rounded(annotation.center_px)
    direction = _rounded(annotation.direction_px)
    cv2.line(frame, center, direction, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(frame, center, direction, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(frame, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3)
    cv2.drawMarker(frame, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=1)

    col, row = annotation.center_taxel
    yaw_deg = annotation.yaw_mod90_rad * 180.0 / 3.141592653589793
    label = f"manual {BLOCK_SIDE_MM:.0f}mm pos=({col:.2f},{row:.2f}) yaw={yaw_deg:.1f}deg"
    if not annotation.fully_inside_sensor:
        label += " outside"
    _draw_label(frame, label, center[0] + 12, center[1] - 12)


def _draw_status_panel(
    frame: Any,
    calibration: PaperCalibration | None,
    annotation: ManualAnnotation | None,
    status: UiStatus,
) -> None:
    import cv2

    overlay = frame.copy()
    panel_h = 132
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.putText(
        frame,
        "Space cap | B blank | Mouse pos | WASD move | Q/E rot | Z/X browse | F save | Delete old | R live | C cal",
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )
    _draw_source_input(frame, status)
    mode_text = status.mode.upper()
    if status.sample_position:
        mode_text = f"{mode_text} {status.sample_position}"
    lines = [
        f"Mode: {mode_text} | Dataset: {_clip_plain(status.dataset, 48)}",
        f"Camera: {status.camera} | Calibration: {_calibration_text(calibration, status.calibration)} | AutoCal: {'on' if status.auto_calibration else 'off'}",
        f"Annotation: {_annotation_text(annotation, status.annotation)}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(frame, text, (16, 88 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 2, cv2.LINE_AA)
    if status.writer:
        _draw_label(frame, status.writer, 16, panel_h + 28)


def source_input_rect(frame_width: int) -> tuple[int, int, int, int]:
    return (92, 36, max(220, min(int(frame_width) - 16, 980)), 66)


def _draw_source_input(frame: Any, status: UiStatus) -> None:
    import cv2

    left, top, right, bottom = source_input_rect(frame.shape[1])
    border = (0, 220, 255) if status.source_editing else (220, 220, 220)
    fill = (36, 36, 36) if status.source_editing else (28, 28, 28)
    cv2.putText(frame, "Source:", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), fill, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), border, 1)
    text = status.source_text + ("_" if status.source_editing and int(time.time() * 2) % 2 == 0 else "")
    clipped = _clip_text_to_width(text, right - left - 16, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.putText(frame, clipped, (left + 8, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_tactile_side_panel(
    tactile: TactileSnapshot | None,
    target_height: int,
    config: SheetConfig,
    annotation: ManualAnnotation | None,
    status: UiStatus,
    tactile_values: Sequence[float] | None = None,
) -> Any:
    import cv2
    import numpy as np

    panel_w = 360
    panel = np.zeros((max(260, int(target_height)), panel_w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    cv2.putText(panel, "Tactile", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 230, 255), 2, cv2.LINE_AA)
    if tactile is None:
        source = "SAVED"
        ready = tactile_values is not None
        detail = "saved label values" if ready else "no saved tactile values"
        error = None
    else:
        source = f"{'LIVE' if tactile.hardware_online else 'SIM'} {tactile.port}"
        if tactile.error:
            source = f"ERR {tactile.port}"
        ready = tactile.available
        detail = f"{tactile.status} | top5 {'ready' if ready else 'waiting'}"
        error = tactile.error
    cv2.putText(panel, source, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        detail,
        (16, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 220, 180) if ready else (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    if error:
        cv2.putText(panel, _clip_plain(error, 42), (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (90, 90, 255), 1, cv2.LINE_AA)

    title = "saved tactile" if tactile is None else "10-frame top5 avg"
    preview = preview_from_annotation(annotation, tactile, tactile_values=tactile_values, title=title)
    heatmap = draw_tactile_preview(preview, config=config, width=320)
    x = (panel_w - heatmap.shape[1]) // 2
    y = 124
    bottom = min(panel.shape[0], y + heatmap.shape[0])
    visible_h = max(0, bottom - y)
    if visible_h > 0:
        panel[y:bottom, x : x + heatmap.shape[1]] = heatmap[:visible_h]
    return panel


def draw_tactile_port_selector(
    image: Any,
    x_offset: int,
    selected_port: str,
    ports: tuple[str, ...],
    open_dropdown: bool,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    import cv2

    option_rects: list[tuple[str, tuple[int, int, int, int]]] = []
    left = x_offset + 16
    top = 94
    right = x_offset + 344
    bottom = top + 30
    cv2.putText(image, "Port", (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(image, (left, top), (right, bottom), (34, 34, 34), -1)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 220, 255) if open_dropdown else (220, 220, 220), 1)
    text = _clip_plain(str(selected_port), 38)
    cv2.putText(image, text, (left + 8, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "v", (right - 18, bottom - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    option_rects.append(("__toggle__", (left, top, right, bottom)))
    if not open_dropdown:
        return option_rects
    option_top = bottom + 4
    for index, port in enumerate(ports[:8]):
        row_top = option_top + index * 28
        row_bottom = row_top + 26
        fill = (48, 48, 48) if port == selected_port else (28, 28, 28)
        cv2.rectangle(image, (left, row_top), (right, row_bottom), fill, -1)
        cv2.rectangle(image, (left, row_top), (right, row_bottom), (85, 85, 85), 1)
        cv2.putText(
            image,
            _clip_plain(port, 42),
            (left + 8, row_bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        option_rects.append((port, (left, row_top, right, row_bottom)))
    return option_rects


def _calibration_text(calibration: PaperCalibration | None, fallback: str) -> str:
    if calibration is None:
        return fallback
    median = calibration.quality.median_paper_error_mm
    median_text = "nan" if median is None else f"{median:.3f}mm"
    return f"markers={len(calibration.marker_ids)} median={median_text}"


def _annotation_text(annotation: ManualAnnotation | None, fallback: str) -> str:
    if annotation is None:
        return fallback
    col, row = annotation.center_taxel
    inside = "inside" if annotation.fully_inside_sensor else "outside"
    yaw = annotation.yaw_mod90_rad * 180.0 / 3.141592653589793
    return f"manual pos=({col:.2f},{row:.2f}) yaw={yaw:.1f}deg {inside}"


def _draw_label(frame: Any, text: str, x: int, y: int) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(4, min(int(x), frame.shape[1] - width - 8))
    y = max(height + 4, min(int(y), frame.shape[0] - baseline - 4))
    cv2.rectangle(frame, (x - 4, y - height - 6), (x + width + 4, y + baseline + 4), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _clip_text_to_width(text: str, max_width: int, font: int, scale: float, thickness: int) -> str:
    import cv2

    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    trimmed = str(text)
    while trimmed and cv2.getTextSize(ellipsis + trimmed, font, scale, thickness)[0][0] > max_width:
        trimmed = trimmed[1:]
    return ellipsis + trimmed


def _clip_plain(text: str | None, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _rounded(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
