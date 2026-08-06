from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .geometry import BLOCK_SIDE_TAXEL, Point2, SensorArrayConfig, SheetConfig, fixed_square_footprint
from .tactile import draw_tactile_heatmap


@dataclass(frozen=True)
class PreviewPose:
    available: bool = False
    source: str | None = None
    yaw_mod90_rad: float | None = None
    footprint_side_taxel: float = BLOCK_SIDE_TAXEL
    footprint_corners_taxel: tuple[Point2, ...] | None = None
    fully_inside_sensor: bool | None = None


@dataclass(frozen=True)
class TactilePreview:
    values: np.ndarray | None
    position_taxel: Point2 | None
    pose: PreviewPose = PreviewPose()
    title: str = "Tactile"


def draw_tactile_preview(
    preview: TactilePreview,
    config: SheetConfig | None = None,
    width: int = 320,
) -> Any:
    config = config or SheetConfig()
    heatmap = draw_tactile_heatmap(preview.values, width=width, title=preview.title, flip_x=False)
    _draw_target(heatmap, config.sensor, preview)
    return heatmap


def taxel_to_heatmap_px(
    sensor: SensorArrayConfig,
    x_taxel: float,
    y_taxel: float,
    heatmap_width: int,
    heatmap_height: int,
    header_h: int = 44,
) -> tuple[int, int]:
    cell_w = float(heatmap_width) / float(sensor.cols)
    cell_h = float(heatmap_height - header_h) / float(sensor.rows)
    x = int(round((float(x_taxel) + 0.5) * cell_w))
    y = int(round(header_h + (float(y_taxel) + 0.5) * cell_h))
    return (_clip_coord(x), _clip_coord(y))


def _draw_target(heatmap: Any, sensor: SensorArrayConfig, preview: TactilePreview) -> None:
    import cv2

    if preview.position_taxel is None:
        return

    corners = preview.pose.footprint_corners_taxel
    dashed = False
    if corners is None:
        corners = fixed_square_footprint(
            preview.position_taxel,
            yaw_rad=0.0,
            side_taxel=preview.pose.footprint_side_taxel,
        )
        dashed = True

    points = [
        taxel_to_heatmap_px(sensor, x, y, heatmap.shape[1], heatmap.shape[0])
        for x, y in corners
    ]
    if dashed:
        _draw_dashed_polygon(heatmap, points, (255, 255, 255), thickness=3)
        _draw_dashed_polygon(heatmap, points, (0, 220, 255), thickness=1)
    else:
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(heatmap, [pts], isClosed=True, color=(255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(heatmap, [pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    center = taxel_to_heatmap_px(sensor, preview.position_taxel[0], preview.position_taxel[1], heatmap.shape[1], heatmap.shape[0])
    cv2.drawMarker(heatmap, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=3)
    cv2.drawMarker(heatmap, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)


def _draw_dashed_polygon(image: Any, points: Sequence[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
    for start, end in zip(points, tuple(points[1:]) + tuple(points[:1])):
        _draw_dashed_line(image, start, end, color, thickness)


def _draw_dashed_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_px: float = 8.0,
    gap_px: float = 6.0,
) -> None:
    import cv2

    x1, y1 = start
    x2, y2 = end
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = max(1.0, float((dx * dx + dy * dy) ** 0.5))
    step = dash_px + gap_px
    current = 0.0
    while current < length:
        dash_end = min(length, current + dash_px)
        p1 = (int(round(x1 + dx * current / length)), int(round(y1 + dy * current / length)))
        p2 = (int(round(x1 + dx * dash_end / length)), int(round(y1 + dy * dash_end / length)))
        cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
        current += step


def _clip_coord(value: int) -> int:
    return max(-10000, min(10000, int(value)))
