"""Capture five raw tactile-skin USB CDC frames to a CSV file.

The current firmware transmits each frame as 128 little-endian uint16 values
(8 rows x 16 columns, 256 bytes) without a packet header.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import struct
import sys
import time

try:
    import serial
except ImportError:
    serial = None


ROWS = 8
COLS = 16
SAMPLES_PER_FRAME = ROWS * COLS
FRAME_BYTES = SAMPLES_PER_FRAME * 2
FRAMES_TO_CAPTURE = 5
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "captures"


def read_exact(port, byte_count, timeout_seconds):
    """Read exactly byte_count bytes, failing if the frame is incomplete."""
    data = bytearray()
    deadline = time.monotonic() + timeout_seconds

    while len(data) < byte_count:
        remaining = byte_count - len(data)
        chunk = port.read(remaining)
        if chunk:
            data.extend(chunk)
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s: received {len(data)}/{byte_count} bytes"
            )

    return bytes(data)


def default_output_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"raw_5x128_capture_{timestamp}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture five 8x16 raw tactile-skin frames to CSV."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM7")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: captures/raw_5x128_capture_<timestamp>.csv)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Maximum seconds to wait for each frame (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if serial is None:
        raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"r{row}_c{col}" for row in range(ROWS) for col in range(COLS)]

    with serial.Serial(args.port, args.baud, timeout=0.1) as port:
        # Begin from the next device transmission after the port is opened.
        port.reset_input_buffer()
        print(f"[SERIAL] Opened {args.port} at {args.baud} baud.")
        print(f"[CAPTURE] Reading {FRAMES_TO_CAPTURE} raw frames...")

        frames = []
        for frame_index in range(FRAMES_TO_CAPTURE):
            frame_bytes = read_exact(port, FRAME_BYTES, args.timeout)
            frames.append(struct.unpack(f"<{SAMPLES_PER_FRAME}H", frame_bytes))
            print(f"[CAPTURE] Received frame {frame_index + 1}/{FRAMES_TO_CAPTURE}.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(frames)

    print(f"[SAVED] {output_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
