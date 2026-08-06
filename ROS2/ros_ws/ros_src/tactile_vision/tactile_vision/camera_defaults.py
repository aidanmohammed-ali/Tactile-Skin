from __future__ import annotations

import os
from pathlib import Path


DEFAULT_CAMERA_SOURCE = "http://192.168.213.190:3588/video"


def camera_source_store_path(path: str | None = None) -> Path:
    text = (path or os.environ.get("TACTILE_CAMERA_SOURCE_FILE") or "").strip()
    if text:
        return Path(text).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return config_home / "tactile_ros" / "camera_source.txt"


def load_camera_source(fallback: str = DEFAULT_CAMERA_SOURCE, path: str | None = None) -> str:
    try:
        text = camera_source_store_path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return str(fallback)
    return text or str(fallback)


def save_camera_source(source: str, path: str | None = None) -> Path:
    text = str(source).strip()
    if not text:
        raise ValueError("camera source is empty")
    target = camera_source_store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
