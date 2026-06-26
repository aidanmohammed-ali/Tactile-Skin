from __future__ import annotations

import base64
import tkinter as tk
from typing import Any


def letterbox_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return 0, 0, 0, 0
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    width = max(1, int(round(src_w * scale)))
    height = max(1, int(round(src_h * scale)))
    left = (dst_w - width) // 2
    top = (dst_h - height) // 2
    return width, height, left, top


def should_render_frame(
    frame_seq: int,
    rendered_frame_seq: int,
    canvas_size: tuple[int, int],
    rendered_canvas_size: tuple[int, int],
    video_dirty: bool,
    now: float,
    last_render_time: float,
    min_interval_sec: float,
) -> bool:
    if video_dirty:
        return True
    if frame_seq == rendered_frame_seq and canvas_size == rendered_canvas_size:
        return False
    return now - last_render_time >= min_interval_sec


def image_msg_to_bgr(msg: Any) -> Any:
    import cv2
    import numpy as np

    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
    if msg.encoding == "bgr8":
        return frame.copy()
    if msg.encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def bgr_to_letterboxed_photo(frame: Any, canvas_w: int, canvas_h: int, background=(18, 18, 18)) -> tk.PhotoImage:
    png_bytes = bgr_to_letterboxed_png(frame, canvas_w, canvas_h, background)
    data = base64.b64encode(png_bytes)
    return tk.PhotoImage(data=data, format="png")


def bgr_to_letterboxed_png(frame: Any, canvas_w: int, canvas_h: int, background=(18, 18, 18)) -> bytes:
    import cv2
    import numpy as np

    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    frame_h, frame_w = frame.shape[:2]
    width, height, left, top = letterbox_size(frame_w, frame_h, canvas_w, canvas_h)
    output = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    output[:, :] = background
    if width and height:
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        output[top : top + height, left : left + width] = resized
    # OpenCV encoders expect BGR input and write a standard RGB PNG.
    # Converting to RGB before imencode would swap red and blue in Tkinter.
    ok, encoded = cv2.imencode(".png", output)
    if not ok:
        raise RuntimeError("could not encode frame for Tkinter")
    return encoded.tobytes()
