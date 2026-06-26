from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


ROWS = 8
COLS = 16
NUM_TAXELS = ROWS * COLS
FRAME_BYTES = NUM_TAXELS * 2
CAPTURE_FRAMES = 10


class Processor:
    """Copied tactile processing chain from tactile_src/python_visualiser.py."""

    STATE_UNCALIBRATED = 0
    STATE_READY = 4

    FILTER_ALPHA = 0.05
    SPATIAL_CENTER_WEIGHT = 0.80

    def __init__(self, num_taxels: int = NUM_TAXELS) -> None:
        self.num_taxels = num_taxels
        self.noise_threshold = 0.2
        self.history = np.zeros((5, num_taxels), dtype=np.float32)
        self.smoothed_data = np.zeros(num_taxels, dtype=np.float32)
        self.spatial_temp = np.zeros(num_taxels, dtype=np.float32)
        self.curve_a = np.zeros(num_taxels, dtype=np.float32)
        self.curve_b = np.ones(num_taxels, dtype=np.float32)
        self.curve_c = np.zeros(num_taxels, dtype=np.float32)
        self.state = self.STATE_UNCALIBRATED

    def reset_calibration(self) -> None:
        self.state = self.STATE_UNCALIBRATED
        self.curve_a.fill(0.0)
        self.curve_b.fill(1.0)
        self.curve_c.fill(0.0)

    def tare(self) -> None:
        self.curve_c[:] = self.spatial_temp
        denominator = 65535.0 - self.curve_c
        denominator = np.maximum(denominator, 1.0)
        self.curve_b[:] = 1.0 / denominator
        self.curve_a.fill(0.0)
        self.state = self.STATE_READY

    def process_frame(self, raw_frame: Any) -> np.ndarray:
        self._filter_frame_interval(raw_frame)

        if self.state == self.STATE_READY:
            zeroed = self.spatial_temp - self.curve_c
            out = zeroed * self.curve_b
            out[out < self.noise_threshold] = 0.0
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        return np.clip(self.spatial_temp / 65535.0, 0.0, 1.0).astype(np.float32)

    def _filter_frame_interval(self, raw_frame: Any) -> None:
        self.history[4] = self.history[3]
        self.history[3] = self.history[2]
        self.history[2] = self.history[1]
        self.history[1] = self.history[0]
        self.history[0] = raw_frame.astype(np.float32)

        median_values = np.median(self.history, axis=0)
        self.smoothed_data = self.FILTER_ALPHA * median_values + (1.0 - self.FILTER_ALPHA) * self.smoothed_data

        grid = self.smoothed_data.reshape(ROWS, COLS)
        output = np.empty_like(grid)
        neighbour_weight = (1.0 - self.SPATIAL_CENTER_WEIGHT) / 4.0

        for row in range(ROWS):
            for col in range(COLS):
                current = grid[row, col]
                spatial_sum = current * self.SPATIAL_CENTER_WEIGHT
                missing_weight = 0.0

                if col > 0:
                    spatial_sum += grid[row, col - 1] * neighbour_weight
                else:
                    missing_weight += neighbour_weight

                if col < COLS - 1:
                    spatial_sum += grid[row, col + 1] * neighbour_weight
                else:
                    missing_weight += neighbour_weight

                if row > 0:
                    spatial_sum += grid[row - 1, col] * neighbour_weight
                else:
                    missing_weight += neighbour_weight

                if row < ROWS - 1:
                    spatial_sum += grid[row + 1, col] * neighbour_weight
                else:
                    missing_weight += neighbour_weight

                if missing_weight > 0.0:
                    spatial_sum += current * missing_weight

                output[row, col] = spatial_sum

        self.spatial_temp[:] = output.reshape(NUM_TAXELS)


class SerialFrameReader:
    def __init__(self, port: str, baudrate: int = 115200) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=0)
        self.buffer = bytearray()

    def close(self) -> None:
        self.serial.close()

    def read_frame(self) -> np.ndarray | None:
        waiting = self.serial.in_waiting
        if waiting > 0:
            self.buffer.extend(self.serial.read(waiting))

        if len(self.buffer) < FRAME_BYTES:
            return None

        frame_bytes = self.buffer[:FRAME_BYTES]
        del self.buffer[:FRAME_BYTES]
        return np.frombuffer(frame_bytes, dtype="<u2").copy()


@dataclass(frozen=True)
class TactileSnapshot:
    timestamp: float
    port: str
    hardware_online: bool
    status: str
    error: str | None
    tared: bool
    processed: np.ndarray | None
    top5_normalized: np.ndarray | None
    top5_raw_average: np.ndarray | None
    recent_raw_frames: tuple[np.ndarray, ...]

    @property
    def available(self) -> bool:
        return self.top5_normalized is not None and self.top5_raw_average is not None

    def to_sensor_payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "timestamp": self.timestamp,
            "port": self.port,
            "hardware_online": self.hardware_online,
            "status": self.status,
            "error": self.error,
            "rows": ROWS,
            "cols": COLS,
            "tared": self.tared,
            "top5_normalized": None if self.top5_normalized is None else self.top5_normalized.astype(float).tolist(),
            "top5_raw_average": None if self.top5_raw_average is None else self.top5_raw_average.astype(float).tolist(),
            "recent_raw_frames": [frame.astype(int).tolist() for frame in self.recent_raw_frames],
        }


class ThreadedTactileReader:
    """Background tactile reader that exposes latest visualisation and T-mode data."""

    def __init__(self, port: str = "SIMULATOR", baudrate: int = 115200, target_hz: float = 60.0) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.target_hz = float(target_hz)
        self.processor = Processor(NUM_TAXELS)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader: SerialFrameReader | None = None
        self._hardware_requested = self.port.upper() != "SIMULATOR"
        self._hardware_online = False
        self._status = "not started"
        self._error: str | None = None
        self._timestamp = 0.0
        self._processed: np.ndarray | None = None
        self._recent_raw_frames: list[np.ndarray] = []

    @property
    def hardware_requested(self) -> bool:
        return self._hardware_requested

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if self._hardware_requested:
            try:
                self._reader = SerialFrameReader(self.port, self.baudrate)
                self._hardware_online = True
                self._status = f"live {self.port}"
                self._error = None
            except Exception as exc:
                self._reader = None
                self._hardware_online = False
                self._status = "hardware unavailable"
                self._error = str(exc)
        else:
            self._hardware_online = False
            self._status = "simulation"
            self._error = None
        self._thread = threading.Thread(target=self._loop, name="data_collection_tactile", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._reader is not None:
            self._reader.close()
        self._reader = None

    def reconnect(self, port: str, baudrate: int | None = None) -> None:
        self.stop()
        self.port = str(port)
        if baudrate is not None:
            self.baudrate = int(baudrate)
        self.processor = Processor(NUM_TAXELS)
        self._hardware_requested = self.port.upper() != "SIMULATOR"
        self._hardware_online = False
        self._status = "not started"
        self._error = None
        self._timestamp = 0.0
        with self._lock:
            self._processed = None
            self._recent_raw_frames = []
        self.start()

    def snapshot(self) -> TactileSnapshot:
        with self._lock:
            recent = tuple(frame.copy() for frame in self._recent_raw_frames)
            processed = None if self._processed is None else self._processed.copy()
            top5_normalized, top5_raw_average = top5_average_over_recent_frames(recent)
            return TactileSnapshot(
                timestamp=self._timestamp,
                port=self.port,
                hardware_online=self._hardware_online,
                status=self._status,
                error=self._error,
                tared=self.processor.state == Processor.STATE_READY,
                processed=processed,
                top5_normalized=top5_normalized,
                top5_raw_average=top5_raw_average,
                recent_raw_frames=recent,
            )

    def tare(self) -> None:
        with self._lock:
            self.processor.tare()

    def reset_calibration(self) -> None:
        with self._lock:
            self.processor.reset_calibration()

    def _loop(self) -> None:
        simulation_start = time.perf_counter()
        interval = 1.0 / max(1.0, self.target_hz)
        while not self._stop.is_set():
            started = time.perf_counter()
            raw_frame = self._read_raw_frame(started - simulation_start)
            if raw_frame is not None and raw_frame.size == NUM_TAXELS:
                processed = self.processor.process_frame(raw_frame)
                with self._lock:
                    self._timestamp = time.time()
                    self._processed = processed
                    self._recent_raw_frames.append(raw_frame.copy())
                    if len(self._recent_raw_frames) > CAPTURE_FRAMES:
                        del self._recent_raw_frames[0]
            elapsed = time.perf_counter() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _read_raw_frame(self, simulation_time: float) -> np.ndarray | None:
        if self._reader is not None:
            frame = self._reader.read_frame()
            if frame is not None:
                self._status = f"live {self.port}"
            return frame
        if self._hardware_requested:
            return None
        return make_simulated_frame(simulation_time)


def make_simulated_frame(simulation_time: float) -> np.ndarray:
    frame = np.zeros(NUM_TAXELS, dtype=np.uint16)
    target_c = 7.5 + math.sin(simulation_time * 1.2) * 5.0
    target_r = 3.5 + math.cos(simulation_time * 0.8) * 2.5

    for row in range(ROWS):
        for col in range(COLS):
            dr = row - target_r
            dc = col - target_c
            distance_squared = dr * dr + dc * dc
            intensity_curve = math.exp(-distance_squared / 3.5)
            frame[row * COLS + col] = int(intensity_curve * 65535.0)

    return frame


def top5_average_over_recent_frames(
    recent_frames: tuple[np.ndarray, ...] | list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not recent_frames:
        return None, None
    data = np.asarray(recent_frames, dtype=np.float32)
    top_count = min(5, data.shape[0])
    top_values = np.sort(data, axis=0)[-top_count:]
    raw_average = np.mean(top_values, axis=0).astype(np.float32)
    normalized = np.clip(raw_average / 65535.0, 0.0, 1.0).astype(np.float32)
    return normalized, raw_average


def canonicalize_tactile_values(values: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    """Convert hardware row-major values into physical sensor order.

    The UI previously flipped heatmaps at draw time. New training labels store
    this physical order directly, so col=0 means the leftmost visible taxel.
    """

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.size != NUM_TAXELS:
        raise ValueError(f"expected {NUM_TAXELS} tactile values, got {array.size}")
    return np.fliplr(array.reshape(ROWS, COLS)).reshape(NUM_TAXELS).astype(np.float32)


def available_tactile_ports() -> list[str]:
    ports = ["SIMULATOR"]
    if list_ports is None:
        return ports
    for port in list_ports.comports():
        device = str(getattr(port, "device", "")).strip()
        if device and _is_tty_acm_port(device) and device not in ports:
            ports.append(device)
    return ports


def _is_tty_acm_port(device: str) -> bool:
    return Path(str(device)).name.startswith("ttyACM")


def draw_tactile_heatmap(
    values: np.ndarray | None,
    width: int = 320,
    cell_gap: int = 2,
    title: str = "Tactile top5",
    flip_x: bool = False,
) -> np.ndarray:
    import cv2

    cell = max(10, int(width // COLS))
    header_h = 44
    height = header_h + ROWS * cell
    image = np.zeros((height, COLS * cell, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (image.shape[1], header_h), (20, 20, 20), -1)
    cv2.putText(image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
    if values is None:
        cv2.putText(image, "no data", (10, header_h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        return image

    grid = values.reshape(ROWS, COLS)
    if flip_x:
        grid = np.fliplr(grid)
    for row in range(ROWS):
        for col in range(COLS):
            value = float(grid[row, col])
            intensity = int(np.clip(value * 255.0, 0, 255))
            color = (0, 255 - intensity, intensity)
            x0 = col * cell
            y0 = header_h + row * cell
            x1 = x0 + cell - cell_gap
            y1 = y0 + cell - cell_gap
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
    return image
