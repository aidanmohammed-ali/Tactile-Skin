from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
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
class CameraFrame:
    frame_id: int
    image: Any
    timestamp: float

    def copy(self) -> "CameraFrame":
        return CameraFrame(self.frame_id, self.image.copy(), self.timestamp)


@dataclass(frozen=True)
class CameraStatus:
    opened: bool
    source: str
    message: str = ""


class ThreadedCamera:
    """Low-latency OpenCV capture that keeps only the newest frame."""

    def __init__(
        self,
        source: str | int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        low_latency: bool = True,
    ) -> None:
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.low_latency = low_latency
        self.status = CameraStatus(False, str(source), "not opened")
        self._cv2 = None
        self._capture = None
        self._process: Any | None = None
        self._process_stop: Any | None = None
        self._frame_queue: Any | None = None
        self._status_queue: Any | None = None
        self._lock = threading.Lock()
        self._latest_frame: CameraFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ids = itertools.count(1)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if _should_use_process_capture(self.source):
            self._open_process_capture()
        else:
            self._open_capture()
        self._thread = threading.Thread(target=self._reader_loop, name="data_collection_camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._stop_process_capture()
        self.status = CameraStatus(False, str(self.source), "closed")

    def reconnect(
        self,
        source: str | int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        low_latency: bool | None = None,
    ) -> None:
        self.stop()
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        if low_latency is not None:
            self.low_latency = low_latency
        self._ids = itertools.count(1)
        with self._lock:
            self._latest_frame = None
        self.start()

    def latest_frame(self) -> CameraFrame | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _open_capture(self) -> None:
        cv2 = self._require_cv2()
        capture = self._make_capture(cv2)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
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

    def _open_process_capture(self) -> None:
        context = mp.get_context("spawn")
        self._frame_queue = context.Queue(maxsize=1)
        self._status_queue = context.Queue(maxsize=8)
        self._process_stop = context.Event()
        self._process = context.Process(
            target=_capture_process_loop,
            args=(
                self.source,
                self.frame_width,
                self.frame_height,
                self.low_latency,
                self._frame_queue,
                self._status_queue,
                self._process_stop,
            ),
            name="data_collection_camera_capture",
            daemon=True,
        )
        self._process.start()
        self.status = CameraStatus(False, str(self.source), "capture process starting")
        self._wait_for_process_open_status(timeout_s=0.75)

    def _make_capture(self, cv2: Any) -> Any:
        if self.low_latency and _is_url_source(self.source) and hasattr(cv2, "CAP_FFMPEG"):
            try:
                capture = _open_with_ffmpeg_options(cv2, self.source)
                if capture.isOpened():
                    return capture
                capture.release()
            except Exception:
                pass
        return cv2.VideoCapture(self.source)

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            if self._process is not None:
                self._read_from_process()
                time.sleep(0.005)
                continue
            if self._capture is None:
                time.sleep(0.05)
                continue
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.status = CameraStatus(False, str(self.source), "waiting for frame")
                time.sleep(0.03)
                continue
            camera_frame = CameraFrame(next(self._ids), frame, time.time())
            with self._lock:
                self._latest_frame = camera_frame
            self.status = CameraStatus(True, str(self.source), "streaming")

    def _read_from_process(self) -> None:
        self._drain_process_status()
        self._drain_process_frames()
        if self._process is not None and not self._process.is_alive():
            exitcode = self._process.exitcode
            self.status = CameraStatus(False, str(self.source), f"capture process exited code={exitcode}")

    def _wait_for_process_open_status(self, timeout_s: float) -> None:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self._status_queue is not None:
                try:
                    opened, message = self._status_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.status = CameraStatus(bool(opened), str(self.source), str(message))
                    if opened:
                        return
                    if self._process is not None and not self._process.is_alive():
                        self._stop_process_capture()
                        raise RuntimeError(str(message))
            if self._process is not None and not self._process.is_alive():
                exitcode = self._process.exitcode
                message = f"capture process exited code={exitcode}"
                self.status = CameraStatus(False, str(self.source), message)
                self._stop_process_capture()
                raise RuntimeError(message)
            time.sleep(0.02)

    def _drain_process_status(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                opened, message = self._status_queue.get_nowait()
            except queue.Empty:
                return
            self.status = CameraStatus(bool(opened), str(self.source), str(message))

    def _drain_process_frames(self) -> None:
        if self._frame_queue is None:
            return
        latest: CameraFrame | None = None
        while True:
            try:
                frame_id, timestamp, frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            latest = CameraFrame(int(frame_id), frame, float(timestamp))
        if latest is not None:
            with self._lock:
                self._latest_frame = latest
            self.status = CameraStatus(True, str(self.source), "streaming")

    def _stop_process_capture(self) -> None:
        if self._process_stop is not None:
            self._process_stop.set()
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None
        self._process_stop = None
        self._frame_queue = None
        self._status_queue = None

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
    return str(source).strip().lower().startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))


def _should_use_process_capture(source: str | int) -> bool:
    return _is_url_source(source)


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


def _capture_process_loop(
    source: str | int,
    frame_width: int | None,
    frame_height: int | None,
    low_latency: bool,
    frame_queue: Any,
    status_queue: Any,
    stop_event: Any,
) -> None:
    try:
        import cv2
    except ImportError:
        _put_latest(status_queue, (False, "opencv-contrib-python is required for camera capture"))
        return

    capture = _make_child_capture(cv2, source, low_latency)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if frame_width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(frame_width))
    if frame_height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(frame_height))
    if not capture.isOpened():
        _put_latest(status_queue, (False, f"could not open source: {source}"))
        capture.release()
        return

    _put_latest(status_queue, (True, "opened"))
    ids = itertools.count(1)
    failures = 0
    last_status_at = 0.0
    try:
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                now = time.time()
                if now - last_status_at > 0.5:
                    _put_latest(status_queue, (False, "waiting for frame"))
                    last_status_at = now
                if failures > 90:
                    _put_latest(status_queue, (False, "stream ended or disconnected"))
                    return
                time.sleep(0.03)
                continue
            failures = 0
            _put_latest(frame_queue, (next(ids), time.time(), frame))
            now = time.time()
            if now - last_status_at > 1.0:
                _put_latest(status_queue, (True, "streaming"))
                last_status_at = now
    finally:
        capture.release()


def _make_child_capture(cv2: Any, source: str | int, low_latency: bool) -> Any:
    if low_latency and _is_url_source(source) and hasattr(cv2, "CAP_FFMPEG"):
        try:
            capture = _open_with_ffmpeg_options(cv2, source)
            if capture.isOpened():
                return capture
            capture.release()
        except Exception:
            pass
    return cv2.VideoCapture(source)


def _put_latest(target_queue: Any, item: Any) -> None:
    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        pass
