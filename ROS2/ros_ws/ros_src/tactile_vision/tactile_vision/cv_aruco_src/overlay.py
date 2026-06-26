from __future__ import annotations

from pathlib import Path
from typing import Any

from .transform import BoardCalibration, board_to_pixel, load_calibration


def draw_board_overlay(
    image: Any,
    calibration: BoardCalibration,
    alpha: float = 0.78,
    circle_radius: int = 8,
) -> Any:
    import cv2

    frame = image.copy()
    overlay = frame.copy()
    line_color = (70, 180, 70)
    circle_color = (0, 255, 0)
    label_color = (0, 255, 255)
    for row in range(calibration.rows):
        points = [_rounded(board_to_pixel(col, row, calibration)) for col in range(calibration.cols)]
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, start, end, line_color, 1)
    for col in range(calibration.cols):
        points = [_rounded(board_to_pixel(col, row, calibration)) for row in range(calibration.rows)]
        for start, end in zip(points, points[1:]):
            cv2.line(overlay, start, end, line_color, 1)
    for row in range(calibration.rows):
        for col in range(calibration.cols):
            center = _rounded(board_to_pixel(col, row, calibration))
            cv2.circle(overlay, center, circle_radius, circle_color, 2)
            if row in (0, calibration.rows - 1) and col in (0, calibration.cols - 1):
                cv2.putText(
                    overlay,
                    f"{col},{row}",
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    label_color,
                    2,
                    cv2.LINE_AA,
                )
    alpha = max(0.0, min(1.0, float(alpha)))
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def save_board_overlay(
    image_path: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.78,
) -> Path:
    import cv2
    import numpy as np

    image_path = Path(image_path)
    output_path = Path(output_path)
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {image_path}")
    calibration = load_calibration(calibration_path)
    overlay = draw_board_overlay(image, calibration, alpha=alpha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", overlay)
    if not ok:
        raise RuntimeError(f"could not encode output image: {output_path}")
    encoded.tofile(str(output_path))
    return output_path


def _rounded(point: tuple[float, float]) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
