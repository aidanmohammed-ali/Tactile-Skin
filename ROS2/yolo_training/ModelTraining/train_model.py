#!/usr/bin/env python3
"""
End-to-end YOLOv8 training script from Labelme annotations.

What it does:
1) Reads Labelme .json files (rectangles or polygons) + corresponding images
2) Converts annotations to YOLO detection labels (.txt)
3) Splits into train/val/test
4) Writes a data.yaml for Ultralytics YOLOv8
5) Trains YOLOv8 (and optionally runs validation)

Assumptions (default):
- Your raw images are in: ModelTraining/dataset_block/images
- Your Labelme JSON files are in: ModelTraining/dataset_block/labels
- JSON filenames match image stem: img001.json <-> img001.jpg

Install:
  pip install -r requirements.txt

Run:
  python ModelTraining/train_model.py --clean
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINING_DIR = REPO_ROOT / "ModelTraining"

# -----------------------------
# CONFIG (edit these)
# -----------------------------
RAW_IMAGES_DIR = MODEL_TRAINING_DIR / "dataset_block" / "images"
RAW_ANN_DIR = MODEL_TRAINING_DIR / "dataset_block" / "labels"

OUT_DIR = MODEL_TRAINING_DIR / "yolo_dataset_block"  # output dataset root
SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}  # must sum to 1.0

# YOLO training config
MODEL = "yolov8n.pt"
IMGSZ = 640
EPOCHS = 40
BATCH = 16
PROJECT = MODEL_TRAINING_DIR / "runs" / "detect"
RUN_NAME = "block"
EXPORT_WEIGHTS = REPO_ROOT / "ObjectDetection" / "block2.pt"

# Reproducibility
SEED = 42

# If True: include images with no labels; creates empty .txt files
INCLUDE_EMPTY_IMAGES = True

# If True: convert non-rectangle shapes to bbox (polygon->bbox). Recommended.
CONVERT_NON_RECT_TO_BBOX = True

# -----------------------------
# Helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_image_for_json(json_path: Path, images_dir: Path) -> Path:
    """Find corresponding image by stem; prefer imagePath if present."""
    with open(json_path, "r") as f:
        data = json.load(f)

    image_path = data.get("imagePath")
    if image_path:
        candidate = images_dir / Path(image_path).name
        if candidate.exists():
            return candidate

    stem = json_path.stem
    candidates = []
    for ext in IMG_EXTS:
        c = images_dir / f"{stem}{ext}"
        if c.exists():
            candidates.append(c)

    if not candidates:
        raise FileNotFoundError(f"No image found for {json_path.name} (stem={stem}) in {images_dir}")

    # If multiple, pick first
    return candidates[0]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def shape_to_bbox(shape: dict) -> Tuple[float, float, float, float]:
    """
    Returns bbox corners (x1,y1,x2,y2) from a labelme shape.
    - rectangle: 2 points
    - polygon/others: bbox from all points (if CONVERT_NON_RECT_TO_BBOX)
    """
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")

    if st == "rectangle" and len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return x1, y1, x2, y2

    if not CONVERT_NON_RECT_TO_BBOX:
        raise ValueError(f"Non-rectangle shape encountered: {st}")

    if len(pts) < 2:
        raise ValueError(f"Not enough points to form bbox for shape_type={st}")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def convert_labelme_json_to_yolo(
    json_path: Path,
    image_path: Path,
    name_to_id: Dict[str, int],
) -> List[str]:
    """Return YOLO label lines for one image."""
    w, h = Image.open(image_path).size

    with open(json_path, "r") as f:
        data = json.load(f)

    lines: List[str] = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label is None:
            continue
        if label not in name_to_id:
            continue

        x1, y1, x2, y2 = shape_to_bbox(shape)

        # Normalize to YOLO (cx, cy, bw, bh)
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        cx, cy, bw, bh = map(clamp01, (cx, cy, bw, bh))
        cls_id = name_to_id[label]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def list_class_names(json_files: List[Path]) -> List[str]:
    names = set()
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if "label" in shape and shape["label"] is not None:
                names.add(shape["label"])
    return sorted(names)


def make_splits(items: List[Tuple[Path, Path]], split: Dict[str, float]) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    items: list of (image_path, json_path)
    """
    assert abs(sum(split.values()) - 1.0) < 1e-6, "SPLIT must sum to 1.0"

    random.shuffle(items)
    n = len(items)
    n_train = int(n * split["train"])
    n_val = int(n * split["val"])
    # remainder goes to test
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def write_data_yaml(out_dir: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def copy_and_write_labels(
    split_name: str,
    pairs: List[Tuple[Path, Path]],
    out_dir: Path,
    name_to_id: Dict[str, int],
) -> None:
    img_out = out_dir / "images" / split_name
    lab_out = out_dir / "labels" / split_name
    ensure_dir(img_out)
    ensure_dir(lab_out)

    for img_path, json_path in pairs:
        # Copy image
        dst_img = img_out / img_path.name
        shutil.copy2(img_path, dst_img)

        # Convert labels
        yolo_lines = convert_labelme_json_to_yolo(json_path, img_path, name_to_id)

        dst_txt = lab_out / f"{img_path.stem}.txt"
        if yolo_lines or INCLUDE_EMPTY_IMAGES:
            with open(dst_txt, "w") as f:
                f.write("\n".join(yolo_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 block detector from Labelme JSON files.")
    parser.add_argument("--images-dir", default=str(RAW_IMAGES_DIR), help="Raw training image directory.")
    parser.add_argument("--annotations-dir", default=str(RAW_ANN_DIR), help="Labelme JSON annotation directory.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Converted YOLO dataset output directory.")
    parser.add_argument("--model", default=MODEL, help="Base YOLO model or checkpoint.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=BATCH, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--project", default=str(PROJECT), help="Ultralytics project directory.")
    parser.add_argument("--name", default=RUN_NAME, help="Ultralytics run name.")
    parser.add_argument(
        "--export-weights",
        default=str(EXPORT_WEIGHTS),
        help="Copy trained best.pt here after training. Use an empty string to disable.",
    )
    parser.add_argument("--no-train", action="store_true", help="Only convert Labelme data to YOLO format.")
    parser.add_argument("--no-empty-images", action="store_true", help="Skip images that have no YOLO labels.")
    parser.add_argument("--clean", action="store_true", help="Delete the converted YOLO dataset before rebuilding it.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    raw_images_dir = Path(args.images_dir)
    raw_ann_dir = Path(args.annotations_dir)
    out_dir = Path(args.out_dir)

    # Gather all JSON files
    json_files = sorted(raw_ann_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No Labelme JSON files found in {raw_ann_dir}")

    # Map JSON -> image
    pairs: List[Tuple[Path, Path]] = []
    for jf in json_files:
        img = find_image_for_json(jf, raw_images_dir)
        pairs.append((img, jf))

    # Build class list from JSON labels
    class_names = list_class_names(json_files)
    if not class_names:
        raise SystemExit("No labels found in JSON files (shapes[].label).")

    name_to_id = {n: i for i, n in enumerate(class_names)}
    print("Classes:", class_names)

    # Create output dirs
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    for s in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / s)
        ensure_dir(out_dir / "labels" / s)

    # Split
    splits = make_splits(pairs, SPLIT)
    for k, v in splits.items():
        print(f"{k}: {len(v)} images")

    # Convert + copy
    global INCLUDE_EMPTY_IMAGES
    INCLUDE_EMPTY_IMAGES = not args.no_empty_images
    for split_name, split_pairs in splits.items():
        copy_and_write_labels(split_name, split_pairs, out_dir, name_to_id)

    # Write data.yaml
    yaml_path = write_data_yaml(out_dir, class_names)
    print("Wrote:", yaml_path)

    if args.no_train:
        print("Skipped training because --no-train was passed.")
        return

    # Train YOLOv8
    import torch
    from ultralytics import YOLO

    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({'GPU' if torch.cuda.is_available() else 'CPU'})")
    
    model = YOLO(args.model)
    print("\nStarting training...\n")
    results = model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        device=device,
        project=str(Path(args.project)),
        name=args.name,
    )

    # Optional: validate using best weights produced
    # You can also do this later with `yolo detect val ...`
    print("\nTraining finished.")
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    print(f"Best model: {best_weights}")
    if args.export_weights:
        export_path = Path(args.export_weights)
        if best_weights.exists():
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, export_path)
            print(f"Copied best model to: {export_path}")
        else:
            print(f"Could not find best model to copy: {best_weights}")


if __name__ == "__main__":
    main()
