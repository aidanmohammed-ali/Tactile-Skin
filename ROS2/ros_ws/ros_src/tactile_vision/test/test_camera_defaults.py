from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_vision.camera_defaults import camera_source_store_path, load_camera_source, save_camera_source


def test_camera_source_defaults_to_fallback_when_missing(tmp_path):
    path = tmp_path / "camera_source.txt"

    assert load_camera_source("http://fallback/video", str(path)) == "http://fallback/video"


def test_camera_source_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "camera_source.txt"

    saved_path = save_camera_source("http://192.168.1.2:8080/video", str(path))

    assert saved_path == path
    assert load_camera_source("http://fallback/video", str(path)) == "http://192.168.1.2:8080/video"


def test_camera_source_store_path_honors_env(tmp_path, monkeypatch):
    path = tmp_path / "from_env.txt"
    monkeypatch.setenv("TACTILE_CAMERA_SOURCE_FILE", str(path))

    assert camera_source_store_path() == path
