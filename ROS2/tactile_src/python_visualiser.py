"""
OpenCV-based tactile skin visualiser.

This mirrors the C++ Raylib visualiser processing chain:
raw uint16[128] frame -> 5-sample median -> EMA low-pass -> 2D spatial smoothing
-> optional tare baseline removal -> 0..1 display values.
"""

import argparse
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


ROWS = 8
COLS = 16
NUM_TAXELS = ROWS * COLS
FRAME_BYTES = NUM_TAXELS * 2
CAPTURE_FRAMES = 10
CAPTURE_SAMPLES_PER_FRAME = NUM_TAXELS

CELL_SIZE = 75
BAR_HEIGHT = 60
WINDOW_WIDTH = COLS * CELL_SIZE
WINDOW_HEIGHT = ROWS * CELL_SIZE
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"


class Processor:
    STATE_UNCALIBRATED = 0
    STATE_READY = 4

    FILTER_ALPHA = 0.05
    SPATIAL_CENTER_WEIGHT = 0.80

    def __init__(self, num_taxels=NUM_TAXELS):
        self.num_taxels = num_taxels
        self.noise_threshold = 0.2
        self.history = np.zeros((5, num_taxels), dtype=np.float32)
        self.smoothed_data = np.zeros(num_taxels, dtype=np.float32)
        self.spatial_temp = np.zeros(num_taxels, dtype=np.float32)
        self.curve_a = np.zeros(num_taxels, dtype=np.float32)
        self.curve_b = np.ones(num_taxels, dtype=np.float32)
        self.curve_c = np.zeros(num_taxels, dtype=np.float32)
        self.state = self.STATE_UNCALIBRATED

    def reset_calibration(self):
        self.state = self.STATE_UNCALIBRATED
        self.curve_a.fill(0.0)
        self.curve_b.fill(1.0)
        self.curve_c.fill(0.0)

    def tare(self):
        self.curve_c[:] = self.spatial_temp
        denominator = 65535.0 - self.curve_c
        denominator = np.maximum(denominator, 1.0)
        self.curve_b[:] = 1.0 / denominator
        self.curve_a.fill(0.0)
        self.state = self.STATE_READY

    def process_frame(self, raw_frame):
        self._filter_frame_interval(raw_frame)

        if self.state == self.STATE_READY:
            zeroed = self.spatial_temp - self.curve_c
            out = zeroed * self.curve_b
            out[out < self.noise_threshold] = 0.0
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        return np.clip(self.spatial_temp / 65535.0, 0.0, 1.0).astype(np.float32)

    def _filter_frame_interval(self, raw_frame):
        self.history[4] = self.history[3]
        self.history[3] = self.history[2]
        self.history[2] = self.history[1]
        self.history[1] = self.history[0]
        self.history[0] = raw_frame.astype(np.float32)

        median_values = np.median(self.history, axis=0)
        self.smoothed_data = (
            self.FILTER_ALPHA * median_values
            + (1.0 - self.FILTER_ALPHA) * self.smoothed_data
        )

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
    def __init__(self, port, baudrate=115200):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=0)
        self.buffer = bytearray()

    def close(self):
        self.serial.close()

    def read_frame(self):
        waiting = self.serial.in_waiting
        if waiting > 0:
            self.buffer.extend(self.serial.read(waiting))

        if len(self.buffer) < FRAME_BYTES:
            return None

        frame_bytes = self.buffer[:FRAME_BYTES]
        del self.buffer[:FRAME_BYTES]
        return np.frombuffer(frame_bytes, dtype="<u2").copy()


def make_simulated_frame(simulation_time):
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


def draw_visualisation(processed_frame, hardware_online, tared, raw_mode, top5_average_mode, port_name):
    image = np.zeros((WINDOW_HEIGHT + (2 * BAR_HEIGHT), WINDOW_WIDTH, 3), dtype=np.uint8)
    values = processed_frame.reshape(ROWS, COLS)

    for row in range(ROWS):
        for col in range(COLS):
            value = float(values[row, col])
            intensity = int(np.clip(value * 255.0, 0, 255))
            color = (0, 255 - intensity, intensity)

            x0 = col * CELL_SIZE
            y0 = BAR_HEIGHT + row * CELL_SIZE
            x1 = x0 + CELL_SIZE - 2
            y1 = y0 + CELL_SIZE - 2
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)

            text = f"{value:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tx = x0 + (CELL_SIZE - text_size[0]) // 2
            ty = y0 + (CELL_SIZE + text_size[1]) // 2
            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, BAR_HEIGHT), (0, 0, 0), thickness=-1)
    top_label = "10-FRAME TOP-5 AVERAGE" if top5_average_mode else ("RAW PASSTHROUGH" if raw_mode else ("TARED" if tared else "SYSTEM OPERATIONAL"))
    top_color = (0, 255, 0) if tared else (0, 165, 255)
    cv2.putText(image, top_label, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, top_color, 2, cv2.LINE_AA)
    cv2.putText(image, "C: tare   R: reset   F: raw/filter   T: top5 avg   P: plot   M: snap   Q/Esc: quit", (WINDOW_WIDTH - 900, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    bottom_y = WINDOW_HEIGHT + BAR_HEIGHT
    cv2.rectangle(image, (0, bottom_y), (WINDOW_WIDTH, bottom_y + BAR_HEIGHT), (0, 0, 0), thickness=-1)
    status_color = (0, 180, 0) if hardware_online else (0, 0, 220)
    status_text = f"HARDWARE LIVE: {port_name}" if hardware_online else "SIMULATION MODE"
    cv2.rectangle(image, (0, bottom_y), (12, bottom_y + BAR_HEIGHT), status_color, thickness=-1)
    cv2.putText(image, status_text, (25, bottom_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def get_capture_samples(raw_frame):
    return raw_frame.copy()


def top5_average_over_recent_frames(recent_frames):
    data = np.asarray(recent_frames, dtype=np.float32)
    top_count = min(5, data.shape[0])
    top_values = np.sort(data, axis=0)[-top_count:]
    averaged_values = np.mean(top_values, axis=0)
    return np.clip(averaged_values / 65535.0, 0.0, 1.0).astype(np.float32)


def append_recent_raw_frame(recent_frames, raw_frame):
    recent_frames.append(get_capture_samples(raw_frame))
    if len(recent_frames) > CAPTURE_FRAMES:
        del recent_frames[0]


def draw_capture_plot(captured_frames):
    data = np.asarray(captured_frames, dtype=np.float32)
    flattened = data.reshape(-1)

    width = 1200
    height = 520
    margin_left = 70
    margin_right = 30
    margin_top = 55
    margin_bottom = 65
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = np.full((height, width, 3), 24, dtype=np.uint8)
    plot_origin = (margin_left, margin_top + plot_height)
    plot_end = (margin_left + plot_width, margin_top)

    cv2.rectangle(image, (margin_left, margin_top), plot_origin, (80, 80, 80), 1)
    cv2.putText(image, "10 frames x 128 sequential raw samples", (margin_left, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)

    min_val = float(np.min(flattened))
    max_val = float(np.max(flattened))
    span = max(max_val - min_val, 1.0)

    for frame_idx in range(1, CAPTURE_FRAMES):
        x = margin_left + int((frame_idx * CAPTURE_SAMPLES_PER_FRAME) * plot_width / (flattened.size - 1))
        cv2.line(image, (x, margin_top), (x, margin_top + plot_height), (45, 45, 45), 1)

    points = []
    for idx, value in enumerate(flattened):
        x = margin_left + int(idx * plot_width / (flattened.size - 1))
        normalized = (float(value) - min_val) / span
        y = margin_top + plot_height - int(normalized * plot_height)
        points.append((x, y))

    for p0, p1 in zip(points, points[1:]):
        cv2.line(image, p0, p1, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(image, f"min: {min_val:.0f}", (12, plot_origin[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(image, f"max: {max_val:.0f}", (12, plot_end[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(image, "sample index: 10 consecutive frames, 128 sequential raw values per frame",
                (margin_left, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)

    return image


def draw_top5_average_capture_heatmap(captured_frames):
    data = np.asarray(captured_frames, dtype=np.uint16)
    top_values = np.sort(data, axis=0)[-5:]
    averaged_values = np.mean(top_values, axis=0)
    values = averaged_values.reshape(ROWS, COLS)

    image = np.zeros((WINDOW_HEIGHT + BAR_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, BAR_HEIGHT), (0, 0, 0), thickness=-1)

    min_val = int(np.min(averaged_values))
    max_val = int(np.max(averaged_values))
    span = max(max_val - min_val, 1)

    title = f"10-frame raw top-5 average    min: {min_val}    max: {max_val}"
    cv2.putText(image, title, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    for row in range(ROWS):
        for col in range(COLS):
            raw_value = int(round(float(values[row, col])))
            normalized = (raw_value - min_val) / span
            intensity = int(np.clip(normalized * 255.0, 0, 255))
            color = (0, 255 - intensity, intensity)

            x0 = col * CELL_SIZE
            y0 = BAR_HEIGHT + row * CELL_SIZE
            x1 = x0 + CELL_SIZE - 2
            y1 = y0 + CELL_SIZE - 2
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)

            text = str(raw_value)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = x0 + (CELL_SIZE - text_size[0]) // 2
            ty = y0 + (CELL_SIZE + text_size[1]) // 2
            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def save_capture_outputs(plot_image, captured_frames):
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = CAPTURE_DIR / f"raw_10x128_capture_{timestamp}.png"
    data_path = CAPTURE_DIR / f"raw_10x128_capture_{timestamp}.csv"

    cv2.imwrite(str(image_path), plot_image)
    np.savetxt(
        data_path,
        np.asarray(captured_frames, dtype=np.uint16),
        fmt="%u",
        delimiter=",",
        header="Each row is one frame; columns are the 128 sequential raw samples.",
        comments="",
    )

    return image_path, data_path


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCV tactile skin visualiser")
    parser.add_argument("--port", default="SIMULATOR",
                        help="Serial port such as COM7, /dev/ttyACM0, or SIMULATOR")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = Processor(NUM_TAXELS)
    current_frame = np.zeros(NUM_TAXELS, dtype=np.uint16)
    reader = None
    hardware_online = args.port.upper() != "SIMULATOR"
    raw_mode = False
    capture_active = False
    max_capture_active = False
    top5_average_mode = False
    capture_frames = []
    max_capture_frames = []
    recent_raw_frames = []

    if hardware_online:
        try:
            reader = SerialFrameReader(args.port, args.baud)
            print(f"[SERIAL LINK] Opened {args.port} at {args.baud} baud.")
        except Exception as exc:
            print(f"[ERROR] Could not open {args.port}: {exc}")
            print("[SERIAL LINK] Falling back to Simulation Mode.")
            hardware_online = False

    cv2.namedWindow("Tactile Skin OpenCV Visualiser", cv2.WINDOW_AUTOSIZE)
    simulation_start = time.perf_counter()
    last_frame_time = simulation_start

    try:
        while True:
            new_frame_available = False

            if hardware_online and reader is not None:
                frame = reader.read_frame()
                if frame is not None and frame.size == NUM_TAXELS:
                    current_frame = frame
                    new_frame_available = True
            else:
                now = time.perf_counter()
                current_frame = make_simulated_frame(now - simulation_start)
                new_frame_available = True

            if capture_active and new_frame_available:
                capture_frames.append(get_capture_samples(current_frame))
                if len(capture_frames) >= CAPTURE_FRAMES:
                    capture_active = False
                    plot_image = draw_capture_plot(capture_frames)
                    image_path, data_path = save_capture_outputs(plot_image, capture_frames)
                    cv2.imshow("Raw 10x128 Capture Plot", plot_image)
                    print(f"[CAPTURE] Captured 10 frames x 128 samples. Saved: {image_path}")
                    print(f"[CAPTURE] Raw data saved: {data_path}")

            if max_capture_active and new_frame_available:
                max_capture_frames.append(get_capture_samples(current_frame))
                if len(max_capture_frames) >= CAPTURE_FRAMES:
                    max_capture_active = False
                    max_image = draw_top5_average_capture_heatmap(max_capture_frames)
                    cv2.imshow("Raw 10-frame Top-5 Average", max_image)
                    print("[MAX] Captured 10 raw frames and displayed per-taxel top-5 averages.")

            if new_frame_available:
                append_recent_raw_frame(recent_raw_frames, current_frame)

            if top5_average_mode:
                processed = top5_average_over_recent_frames(recent_raw_frames)
            elif raw_mode:
                processed = np.clip(current_frame.astype(np.float32) / 65535.0, 0.0, 1.0)
            else:
                processed = processor.process_frame(current_frame)

            image = draw_visualisation(
                processed,
                hardware_online,
                processor.state == Processor.STATE_READY,
                raw_mode,
                top5_average_mode,
                args.port,
            )
            cv2.imshow("Tactile Skin OpenCV Visualiser", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                processor.tare()
                print("[TARE] Baseline captured.")
            if key in (ord("r"), ord("R")):
                processor.reset_calibration()
                print("[RESET] Returning to raw passthrough.")
            if key in (ord("f"), ord("F")):
                raw_mode = not raw_mode
                if raw_mode:
                    top5_average_mode = False
                mode_name = "raw passthrough" if raw_mode else "filtered processing"
                print(f"[MODE] Showing {mode_name}.")
            if key in (ord("p"), ord("P")):
                capture_active = True
                capture_frames = []
                print("[CAPTURE] Capturing the next 10 frames x 128 raw samples...")
            if key in (ord("m"), ord("M")):
                max_capture_active = True
                max_capture_frames = []
                print("[MAX] Capturing the next 10 raw frames for per-taxel top-5 averages...")
            if key in (ord("t"), ord("T")):
                top5_average_mode = not top5_average_mode
                if top5_average_mode:
                    raw_mode = False
                mode_name = "10-frame top-5 average" if top5_average_mode else "filtered processing"
                print(f"[MODE] Showing {mode_name}.")

            elapsed = time.perf_counter() - last_frame_time
            if elapsed < (1.0 / 60.0):
                time.sleep((1.0 / 60.0) - elapsed)
            last_frame_time = time.perf_counter()
    finally:
        if reader is not None:
            reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
