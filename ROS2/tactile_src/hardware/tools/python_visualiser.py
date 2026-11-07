"""
OpenCV-based tactile skin visualiser.

Processing is intentionally limited to optional tare baseline removal.
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None


ROWS = 8
COLS = 8
NUM_TAXELS = ROWS * COLS
TRANSPORT_COLS = 16
TRANSPORT_TAXELS = ROWS * TRANSPORT_COLS
MAGIC_HEADER = b"\xDE\xAD\xBE\xEF"
HEADER_BYTES = len(MAGIC_HEADER)
FRAME_BYTES = TRANSPORT_TAXELS * 2
PACKET_BYTES = HEADER_BYTES + FRAME_BYTES

CELL_SIZE = 75
BAR_HEIGHT = 60
WINDOW_WIDTH = COLS * CELL_SIZE
WINDOW_HEIGHT = ROWS * CELL_SIZE


class Processor:
    def __init__(self, num_taxels=NUM_TAXELS):
        self.num_taxels = num_taxels
        self.baseline = np.zeros(num_taxels, dtype=np.float32)
        self.tared = False

    def reset_calibration(self):
        self.baseline.fill(0.0)
        self.tared = False

    def tare(self, raw_frame):
        self.baseline[:] = raw_frame.astype(np.float32)
        self.tared = True

    def process_frame(self, raw_frame):
        values = raw_frame.astype(np.float32)
        if self.tared:
            values = values - self.baseline
        return np.clip(values / 65535.0, 0.0, 1.0).astype(np.float32)


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

        header_index = self.buffer.find(MAGIC_HEADER)
        if header_index < 0:
            keep_bytes = max(HEADER_BYTES - 1, 0)
            if len(self.buffer) > keep_bytes:
                del self.buffer[:-keep_bytes]
            return None

        if header_index > 0:
            del self.buffer[:header_index]

        if len(self.buffer) < PACKET_BYTES:
            return None

        frame_start = HEADER_BYTES
        frame_end = frame_start + FRAME_BYTES
        frame_bytes = self.buffer[frame_start:frame_end]
        del self.buffer[:PACKET_BYTES]
        return np.frombuffer(frame_bytes, dtype="<u2").copy()


def extract_cdc1_frame(transport_frame):
    """Extract CDC1's eight samples from each 16-sample firmware row."""
    if transport_frame.size != TRANSPORT_TAXELS:
        raise ValueError(f"Expected {TRANSPORT_TAXELS} transport samples, got {transport_frame.size}")

    return transport_frame.reshape(ROWS, TRANSPORT_COLS)[:, :COLS].reshape(NUM_TAXELS).copy()


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


def draw_visualisation(processed_frame, hardware_online, tared, show_raw_values, port_name):
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

            if show_raw_values:
                text = str(int(round(value * 65535.0)))
                text_scale = 0.45
            else:
                text = f"{value:.2f}"
                text_scale = 0.55

            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
            tx = x0 + (CELL_SIZE - text_size[0]) // 2
            ty = y0 + (CELL_SIZE + text_size[1]) // 2
            cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(image, (0, 0), (WINDOW_WIDTH, BAR_HEIGHT), (0, 0, 0), thickness=-1)
    value_label = "0-65535" if show_raw_values else "0-1"
    top_label = f"{'TARED' if tared else 'RAW'} | {value_label}"
    top_color = (0, 255, 0) if tared else (0, 165, 255)
    cv2.putText(image, top_label, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, top_color, 2, cv2.LINE_AA)
    cv2.putText(image, "C: tare   R: reset   V: value scale   Q/Esc: quit", (WINDOW_WIDTH - 610, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    bottom_y = WINDOW_HEIGHT + BAR_HEIGHT
    cv2.rectangle(image, (0, bottom_y), (WINDOW_WIDTH, bottom_y + BAR_HEIGHT), (0, 0, 0), thickness=-1)
    status_color = (0, 180, 0) if hardware_online else (0, 0, 220)
    status_text = f"HARDWARE LIVE: {port_name}" if hardware_online else "SIMULATION MODE"
    cv2.rectangle(image, (0, bottom_y), (12, bottom_y + BAR_HEIGHT), status_color, thickness=-1)
    cv2.putText(image, status_text, (25, bottom_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image


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
    show_raw_values = False

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
                if frame is not None and frame.size == TRANSPORT_TAXELS:
                    current_frame = extract_cdc1_frame(frame)
                    new_frame_available = True
            else:
                now = time.perf_counter()
                current_frame = make_simulated_frame(now - simulation_start)
                new_frame_available = True

            processed = processor.process_frame(current_frame)

            image = draw_visualisation(
                processed,
                hardware_online,
                processor.tared,
                show_raw_values,
                args.port,
            )
            cv2.imshow("Tactile Skin OpenCV Visualiser", image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                processor.tare(current_frame)
                print("[TARE] Baseline captured.")
            if key in (ord("r"), ord("R")):
                processor.reset_calibration()
                print("[RESET] Returning to raw passthrough.")
            if key in (ord("v"), ord("V")):
                show_raw_values = not show_raw_values
                mode_name = "0-65535 raw values" if show_raw_values else "0-1 normalized values"
                print(f"[DISPLAY] Showing {mode_name}.")

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
