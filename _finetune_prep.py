"""
Fine-tuning data generator for Plant Grid.

Extracts labelled plant crops from the test video using the current YOLO
detector's own confident detections as pseudo-labels.  The output is a
YOLO-format dataset ready for `yolo train`.

Strategy:
  1. Run YOLOv8m on every Nth frame of the video.
  2. Keep detections that:
     - Fall inside a lane gate (are real row positions, not background).
     - Have confidence >= a threshold (to get clean labels).
     - Meet minimum size requirements.
  3. Save the frames as images and the boxes as YOLO label files.
  4. Split into train/val (80/20).
  5. Write data.yaml for Ultralytics training.

Usage:
    python _finetune_prep.py
"""

import cv2, random, shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────
VIDEO_PATH       = "test_video--big.mp4"
MODEL_WEIGHTS    = "yolov8m.pt"
TARGET_CLASS     = 58          # COCO potted plant (proxy)
CONF_THRESHOLD   = 0.20       # higher than pipeline — we want clean labels
IOU_THRESHOLD    = 0.45
MIN_BOX_AREA     = 400
FRAME_STRIDE     = 2          # every Nth frame (2 = every other frame)
VAL_FRACTION     = 0.20       # 20% validation split
EDGE_MARGIN      = 40
OUTPUT_DIR       = Path("finetune_data")

# ─── GEOMETRY (must match _run_pipeline.py Fix A) ────────────────
def build_lane_gates(w, h):
    yt, yb = int(h * 0.02), int(h * 0.98)
    L = np.array([[0, yt], [int(w * 0.45), yt],
                  [int(w * 0.48), yb], [0, yb]], np.int32)
    R = np.array([[int(w * 0.55), yt], [w - 1, yt],
                  [w - 1, yb], [int(w * 0.52), yb]], np.int32)
    return {"left": L, "right": R}

def pt_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def get_lane(anchor, gates):
    for name in ("left", "right"):
        if pt_in_poly(anchor, gates[name]):
            return name
    return None


def main():
    print("=" * 60)
    print("FINE-TUNING DATA GENERATOR")
    print("=" * 60)

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True)

    model = YOLO(MODEL_WEIGHTS)
    cap   = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gates = build_lane_gates(W, H)

    print(f"Video: {VIDEO_PATH}  ({total} frames, {W}x{H})")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Stride: every {FRAME_STRIDE} frames")
    print(f"Conf threshold: {CONF_THRESHOLD}")
    print()

    # ─── Pass 1: collect all qualifying frames + boxes ────────
    frame_data = []  # list of (frame_idx, image, boxes_normalized)
    fidx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1
        if fidx % FRAME_STRIDE != 0:
            continue

        res = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                            classes=[TARGET_CLASS], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        boxes_norm = []
        for bx, cf in zip(res.boxes.xyxy.cpu().numpy(),
                          res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, bx)
            area = (x2 - x1) * (y2 - y1)
            edge = (x1 < EDGE_MARGIN) or (x2 > W - EDGE_MARGIN)
            if area < (200 if edge else MIN_BOX_AREA):
                continue
            ax = int((x1 + x2) / 2)
            ay = int(y2)
            if get_lane((ax, ay), gates) is None:
                continue

            # YOLO format: class cx cy w h (all normalized 0-1)
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            boxes_norm.append((0, cx, cy, bw, bh))  # class 0 = plant

        if boxes_norm:
            frame_data.append((fidx, frame.copy(), boxes_norm))

        if fidx % 30 == 0:
            print(f"  scanned {fidx}/{total} frames, {len(frame_data)} qualifying frames")

    cap.release()
    print(f"\nTotal qualifying frames: {len(frame_data)}")
    print(f"Total box annotations: {sum(len(fd[2]) for fd in frame_data)}")

    if len(frame_data) == 0:
        print("ERROR: No qualifying frames found. Check detector output.")
        return

    # ─── Pass 2: split and write ──────────────────────────────
    random.seed(42)
    random.shuffle(frame_data)
    n_val = max(1, int(len(frame_data) * VAL_FRACTION))
    val_set   = frame_data[:n_val]
    train_set = frame_data[n_val:]

    print(f"Train: {len(train_set)} frames, Val: {len(val_set)} frames")

    for split, subset in [("train", train_set), ("val", val_set)]:
        for fidx, img, boxes in subset:
            stem = f"frame_{fidx:05d}"
            img_path = OUTPUT_DIR / "images" / split / f"{stem}.jpg"
            lbl_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), img)
            with open(lbl_path, "w") as f:
                for cls, cx, cy, bw, bh in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # ─── Write data.yaml ──────────────────────────────────────
    data_yaml = OUTPUT_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        # Keep this repo-portable; avoid machine-specific absolute paths.
        f.write("path: finetune_data\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names:\n")
        f.write(f"  0: plant\n")

    print(f"\nDataset written to: {OUTPUT_DIR}/")
    print(f"data.yaml: {data_yaml}")
    print()
    print("─" * 60)
    print("To fine-tune, run:")
    print(f"  yolo train model=yolov8m.pt data={data_yaml} epochs=30 imgsz=640 batch=4 name=plant_finetune")
    print("─" * 60)


if __name__ == "__main__":
    main()
