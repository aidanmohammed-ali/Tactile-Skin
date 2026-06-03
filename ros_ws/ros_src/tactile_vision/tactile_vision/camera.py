from __future__ import annotations

import threading
import time
from dataclasses import dataclass
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
    ) -> None:
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
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
    ) -> None:
        self.stop()
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        with self._lock:
            self._latest_frame = None
        self.start()

    def latest_frame(self) -> Any | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _open_capture(self) -> None:
        cv2 = self._require_cv2()
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
