from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_vision.camera import parse_camera_source


def test_parse_camera_source_keeps_urls():
    assert parse_camera_source("http://192.168.0.2:3588/video") == "http://192.168.0.2:3588/video"


def test_parse_camera_source_converts_integer_text():
    assert parse_camera_source("0") == 0
