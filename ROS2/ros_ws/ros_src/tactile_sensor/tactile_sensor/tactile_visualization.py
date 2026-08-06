from __future__ import annotations

from typing import Any

import numpy as np


TACTILE_ROWS = 8
TACTILE_COLS = 16


def normalize_tactile_frame(
    frame: Any,
    value_max: float = 0.0,
    flip_x: bool = True,
) -> np.ndarray:
    grid = np.asarray(frame, dtype=np.float32).reshape(
        TACTILE_ROWS,
        TACTILE_COLS,
    )
    scale = float(value_max)
    if scale <= 0.0:
        scale = 65535.0 if float(np.max(grid, initial=0.0)) > 1.0 else 1.0
    normalized = np.clip(grid / scale, 0.0, 1.0)
    if flip_x:
        normalized = np.fliplr(normalized)
    return normalized.astype(np.float32, copy=True)


def draw_tactile_heatmap(
    frame: Any,
    width: int = 640,
    value_max: float = 0.0,
    flip_x: bool = True,
    title: str = "Live tactile input",
) -> np.ndarray:
    import cv2

    grid = normalize_tactile_frame(
        frame,
        value_max=value_max,
        flip_x=flip_x,
    )
    cell = max(10, int(width) // TACTILE_COLS)
    header_height = 44
    image = np.zeros(
        (header_height + TACTILE_ROWS * cell, TACTILE_COLS * cell, 3),
        dtype=np.uint8,
    )
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], header_height),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    cell_gap = max(1, cell // 16)
    for row in range(TACTILE_ROWS):
        for col in range(TACTILE_COLS):
            intensity = int(round(float(grid[row, col]) * 255.0))
            color = (0, 255 - intensity, intensity)
            x0 = col * cell
            y0 = header_height + row * cell
            cv2.rectangle(
                image,
                (x0, y0),
                (x0 + cell - cell_gap, y0 + cell - cell_gap),
                color,
                -1,
            )
    return image


def draw_tactile_prediction(
    frame: Any,
    *,
    detected: bool,
    position_taxel: tuple[float, float],
    angle_deg: float,
    confidence: float,
    fully_inside_sensor: bool,
    footprint_corners_taxel: tuple[tuple[float, float], ...] = (),
    width: int = 640,
    value_max: float = 1.0,
    flip_x: bool = False,
) -> np.ndarray:
    import cv2

    status = "DETECTED" if detected else "NO BLOCK"
    image = draw_tactile_heatmap(
        frame,
        width=width,
        value_max=value_max,
        flip_x=flip_x,
        title=f"Tactile prediction | {status} | confidence={confidence:.3f}",
    )
    if detected:
        points = [
            _taxel_to_heatmap_px(
                x,
                y,
                image.shape[1],
                image.shape[0],
                flip_x=flip_x,
            )
            for x, y in footprint_corners_taxel
        ]
        if points:
            polygon = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image,
                [polygon],
                isClosed=True,
                color=(255, 255, 255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                image,
                [polygon],
                isClosed=True,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        center = _taxel_to_heatmap_px(
            position_taxel[0],
            position_taxel[1],
            image.shape[1],
            image.shape[0],
            flip_x=flip_x,
        )
        cv2.drawMarker(
            image,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=4,
        )
        cv2.drawMarker(
            image,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )

    footer_height = 64
    output = cv2.copyMakeBorder(
        image,
        0,
        footer_height,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    if detected:
        x, y = position_taxel
        location = "inside" if fully_inside_sensor else "edge/outside"
        result_text = (
            f"x={x:.2f}, y={y:.2f}, angle={angle_deg:.1f} deg, "
            f"footprint={location}"
        )
    else:
        result_text = "No block detected"
    cv2.putText(
        output,
        result_text,
        (10, image.shape[0] + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return output


def _taxel_to_heatmap_px(
    x_taxel: float,
    y_taxel: float,
    image_width: int,
    image_height: int,
    *,
    flip_x: bool,
) -> tuple[int, int]:
    visible_x = (
        float(TACTILE_COLS - 1) - float(x_taxel)
        if flip_x
        else float(x_taxel)
    )
    cell_width = float(image_width) / float(TACTILE_COLS)
    cell_height = float(image_height - 44) / float(TACTILE_ROWS)
    x = int(round((visible_x + 0.5) * cell_width))
    y = int(round(44 + (float(y_taxel) + 0.5) * cell_height))
    return x, y
