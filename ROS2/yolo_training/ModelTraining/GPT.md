# ModelTraining Handoff Notes

Last updated: 2026-05-21

This folder contains the capture and YOLOv8 training scripts for the block
detector used by `ObjectDetection/realtime_block_detector.py`.

## Capture Images

From the repository root:

```powershell
python ModelTraining/capture_image.py
```

From inside `ModelTraining`:

```powershell
python capture_image.py
```

Defaults:

- Camera source: `http://192.168.108.213:3588/video`
- Save directory: `ModelTraining/dataset_block/images`
- Requested resolution: `1920x1080`

Controls:

- `S` or Space: save image
- `Q` or Esc: quit

## Train Model

Label the captured images with Labelme and save JSON files in:

```text
ModelTraining/dataset_block/labels
```

Then train:

```powershell
python ModelTraining/train_model.py --clean
```

The script converts Labelme JSON files to a YOLO dataset in
`ModelTraining/yolo_dataset_block`, trains YOLOv8, and copies the best trained
weights to:

```text
ObjectDetection/block2.pt
```

That path is the default weight path used by the live detector.

Useful options:

```powershell
python ModelTraining/train_model.py --no-train --clean
python ModelTraining/train_model.py --epochs 80 --batch 8
python ModelTraining/train_model.py --export-weights ""
```
