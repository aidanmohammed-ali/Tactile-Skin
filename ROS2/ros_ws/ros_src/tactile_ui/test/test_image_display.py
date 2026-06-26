from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_ui.image_display import bgr_to_letterboxed_png, letterbox_size, should_render_frame


def test_letterbox_wide_source_in_square_canvas():
    width, height, left, top = letterbox_size(1920, 1080, 640, 640)

    assert (width, height) == (640, 360)
    assert left == 0
    assert top == 140


def test_letterbox_tall_source_in_wide_canvas():
    width, height, left, top = letterbox_size(720, 1280, 800, 400)

    assert (width, height) == (225, 400)
    assert left == 287
    assert top == 0


def test_letterbox_rejects_empty_dimensions():
    assert letterbox_size(0, 1080, 640, 640) == (0, 0, 0, 0)


def test_should_render_frame_when_dirty():
    assert should_render_frame(1, 1, (640, 480), (640, 480), True, 1.0, 1.0, 0.1)


def test_should_not_render_duplicate_frame_same_canvas():
    assert not should_render_frame(1, 1, (640, 480), (640, 480), False, 2.0, 1.0, 0.1)


def test_should_throttle_new_frame_until_interval_passes():
    assert not should_render_frame(2, 1, (640, 480), (640, 480), False, 1.05, 1.0, 0.1)
    assert should_render_frame(2, 1, (640, 480), (640, 480), False, 1.11, 1.0, 0.1)


def test_bgr_png_encoding_preserves_red_channel():
    import cv2
    import numpy as np

    red_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    red_bgr[:, :] = (0, 0, 255)
    png = bgr_to_letterboxed_png(red_bgr, 2, 2)
    decoded = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert tuple(int(v) for v in decoded[0, 0]) == (0, 0, 255)
