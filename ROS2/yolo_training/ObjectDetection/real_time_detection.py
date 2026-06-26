#!/usr/bin/env python3
"""
Real-time object detection with homography transformation.

What it does:
1) Captures frames from camera (default: camera 0)
2) Detects objects using trained YOLOv8 model
3) Undistorts detected pixel coordinates
4) Applies homography to convert pixel coords to table/real-world coords
5) Displays results with both pixel and real-world coordinates

Requirements:
- Trained YOLO weights: runs/detect/train/weights/best.pt
- Homography calibration: homography_table.npz
- Camera intrinsics: camera_intrinsics.npz

Controls:
- Press 'q' to quit
- Press 'h' to toggle homography transform display

Install:
  pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import Tuple, List

# Config
WEIGHTS_PATH = Path("model_weights/block2.pt")
HOMOGRAPHY_FILE = Path("homography_table.npz")
INTRINSICS_FILE = Path("camera_intrinsics.npz")
CAMERA_ID = 0
CONF_THRESHOLD = 0.1
url = "http://192.168.108.213:3588/video"

# Device selection
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("="*70)
print("REAL-TIME OBJECT DETECTION WITH HOMOGRAPHY")
print("="*70)

# =====================================================
# Load calibration data
# =====================================================
print("\nLoading calibration data...")

# Load homography and camera intrinsics
try:
    cal = np.load(HOMOGRAPHY_FILE)
    H_inv = cal["H_inv"]
    Knew = cal["newK"]
    print(f"✓ Homography loaded from {HOMOGRAPHY_FILE}")
except FileNotFoundError:
    print(f"ERROR: {HOMOGRAPHY_FILE} not found!")
    exit(1)

try:
    intr = np.load(INTRINSICS_FILE)
    K = intr["K"]
    dist = intr["dist"]
    print(f"✓ Camera intrinsics loaded from {INTRINSICS_FILE}")
except FileNotFoundError:
    print(f"ERROR: {INTRINSICS_FILE} not found!")
    exit(1)

# =====================================================
# Coordinate transformation functions
# =====================================================

def undistort_point(u: float, v: float) -> Tuple[float, float]:
    """Convert pixel coordinates from distorted to undistorted space."""
    pts = np.array([[[u, v]]], np.float32)
    pts_ud = cv2.undistortPoints(pts, K, dist, P=Knew)
    return pts_ud[0, 0]


def pixel_to_table(u: float, v: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to table/real-world coordinates.
    
    Args:
        u, v: pixel coordinates (x, y)
    
    Returns:
        x, y: table coordinates (real-world position)
    """
    u2, v2 = undistort_point(u, v)
    p = np.array([[[u2, v2]]], np.float32)
    w = cv2.perspectiveTransform(p, H_inv)
    return w[0, 0]


def draw_detection(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_name: str,
    conf: float,
    real_x: float,
    real_y: float,
    show_real_coords: bool = True
) -> None:
    """Draw bounding box and labels on image."""
    
    # Draw bounding box (green)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Calculate center for homography point
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Draw center point (blue circle)
    cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
    
    # Draw class label with confidence (top of box)
    label_pixel = f"{class_name} ({conf:.2f})"
    cv2.putText(
        img, label_pixel, (x1, y1 - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    
    # Draw real-world coordinates (if enabled)
    if show_real_coords:
        label_real = f"Table: ({real_x:.1f}, {real_y:.1f})"
        cv2.putText(
            img, label_real, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )


# =====================================================
# Load YOLO model
# =====================================================
print("\nLoading YOLO model...")
if not WEIGHTS_PATH.exists():
    print(f"ERROR: Weights not found at {WEIGHTS_PATH}")
    print("Please run train_model.py first")
    exit(1)

model = YOLO(str(WEIGHTS_PATH))
print(f"✓ Model loaded: {WEIGHTS_PATH}")
print(f"  Classes: {model.names}")

# =====================================================
# Open camera
# =====================================================
print(f"\nOpening camera {CAMERA_ID}...")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print(f"ERROR: Cannot open camera {CAMERA_ID}")
    exit(1)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print(f"✓ Camera opened")
print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# =====================================================
# Main detection loop
# =====================================================
print("\nStarting detection loop...")
print("Controls:")
print("  'q' - Quit")
print("  'h' - Toggle real-world coordinates display")
print("  'c' - Toggle confidence display")
print("\n" + "="*70 + "\n")

frame_count = 0
show_real_coords = True
show_confidence = True


try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Failed to read frame")
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False
        )
        
        result = results[0]
        detection_count = 0
        
        # Process detections
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                detection_count += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = result.names[cls]
                
                # Calculate center point
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                # Apply homography transform
                real_x, real_y = pixel_to_table(cx, cy)
                
                # Draw on frame
                draw_detection(
                    frame, x1, y1, x2, y2,
                    class_name, conf,
                    real_x, real_y,
                    show_real_coords=show_real_coords
                )
        
        # Display statistics
        info_text = f"Frame: {frame_count} | Detections: {detection_count} | GPU: {'Yes' if torch.cuda.is_available() else 'No'}"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Display mode indicators
        mode_text = f"Real-coords: {'ON' if show_real_coords else 'OFF'} | Conf: {'ON' if show_confidence else 'OFF'}"
        cv2.putText(
            frame, mode_text, (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1
        )
        
        # Display instructions
        cv2.putText(
            frame, "Press 'h' to toggle coords, 'q' to quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )
        
        # Show frame
        disp = cv2.resize(frame, (1280, 720))
        cv2.imshow("Real-Time Object Detection", disp)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        
        elif key == ord('h'):
            # Toggle real-world coordinates
            show_real_coords = not show_real_coords
            status = "ON" if show_real_coords else "OFF"
            print(f"Real-world coordinates: {status}")
        
        elif key == ord('c'):
            # Toggle confidence display
            show_confidence = not show_confidence
            status = "ON" if show_confidence else "OFF"
            print(f"Confidence display: {status}")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Camera closed. Processed {frame_count} frames.")
