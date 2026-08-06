from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "http://192.168.108.213:3588/video"
DEFAULT_SAVE_DIR = REPO_ROOT / "ModelTraining" / "dataset_block" / "images"


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def save_image(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", frame)
    if not ok:
        raise RuntimeError(f"could not encode image as {path.suffix or '.jpg'}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture images for block model training.")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="cv2.VideoCapture source: camera index, HTTP stream, RTSP stream, etc.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SAVE_DIR),
        help="Directory where captured training images are saved.",
    )
    parser.add_argument("--ext", default=".jpg", choices=(".jpg", ".png"), help="Image file extension.")
    parser.add_argument("--frame-width", type=int, default=1920, help="Requested camera frame width.")
    parser.add_argument("--frame-height", type=int, default=1080, help="Requested camera frame height.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(parse_source(args.source))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera source: {args.source}")

    print(f"Camera opened: {args.source}")
    print(f"Saving images to: {save_dir}")
    print("S or Space = save image | Esc or Q = quit")

    image_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Failed to grab frame")
                break

            display = frame.copy()
            cv2.putText(
                display,
                f"Images saved: {image_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Capture Training Images", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("s"), ord(" ")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output = save_dir / f"img_{timestamp}{args.ext}"
                save_image(output, frame)
                image_count += 1
                print(f"Saved: {output}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print(f"Done. Saved {image_count} images.")


if __name__ == "__main__":
    main()
