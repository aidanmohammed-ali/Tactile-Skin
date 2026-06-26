from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from os import environ
from typing import Any


def parse_camera_source(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


@dataclass(frozen=True)
class CameraStatus:
    opened: bool
    source: str
    message: str = ""


class ThreadedCamera:
    """Keeps only the newest frame from an OpenCV VideoCapture source."""

    def __init__(
        self,
        source: str | int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        capture_low_latency: bool = True,
    ) -> None:
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture_low_latency = capture_low_latency
        self._cv2 = None
        self._capture = None
        self._lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = CameraStatus(False, str(source), "not opened")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._open_capture()
        self._thread = threading.Thread(target=self._reader_loop, name="tactile_camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.status = CameraStatus(False, str(self.source), "closed")

    def reconnect(
        self,
        source: str | int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        capture_low_latency: bool | None = None,
    ) -> None:
        self.stop()
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        if capture_low_latency is not None:
            self.capture_low_latency = capture_low_latency
        with self._lock:
            self._latest_frame = None
        self.start()

    def latest_frame(self) -> Any | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _open_capture(self) -> None:
        cv2 = self._require_cv2()
        capture = self._make_capture(cv2)
        self._set_capture_buffer(cv2, capture)
        if self.frame_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.frame_width))
        if self.frame_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.frame_height))
        if not capture.isOpened():
            self.status = CameraStatus(False, str(self.source), f"could not open source: {self.source}")
            capture.release()
            raise RuntimeError(self.status.message)
        self._capture = capture
        self.status = CameraStatus(True, str(self.source), "opened")

    def _make_capture(self, cv2: Any) -> Any:
        if self.capture_low_latency and _is_url_source(self.source) and hasattr(cv2, "CAP_FFMPEG"):
            try:
                capture = _open_with_ffmpeg_options(cv2, self.source)
                if capture.isOpened():
                    return capture
                capture.release()
            except Exception:
                pass
        return cv2.VideoCapture(self.source)

    def _set_capture_buffer(self, cv2: Any, capture: Any) -> None:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            if self._capture is None:
                time.sleep(0.05)
                continue
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.status = CameraStatus(False, str(self.source), "waiting for frame")
                time.sleep(0.03)
                continue
            with self._lock:
                self._latest_frame = frame
            self.status = CameraStatus(True, str(self.source), "streaming")

    def _require_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("opencv-contrib-python is required for camera capture") from exc
            self._cv2 = cv2
        return self._cv2


def _is_url_source(source: str | int) -> bool:
    if isinstance(source, int):
        return False
    text = str(source).strip().lower()
    return text.startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))


def _open_with_ffmpeg_options(cv2: Any, source: str | int) -> Any:
    previous = environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
    )
    try:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    finally:
        if previous is None:
            environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous
