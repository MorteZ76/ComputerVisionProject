# scripts/view_yolo_vs_annotations.py
# One window: rescaled GT annotations. One window: YOLO detections.
# Keys: p=pause, r=resume, o=next frame, i=prev frame, q=quit.

import os
import cv2
import time
import numpy as np
from collections import defaultdict

# =========================
# HYPERPARAMETERS (EDIT)
# =========================
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANN_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# If you know the original annotation image size, set it to rescale correctly.
# Else it will infer scale from max(x,y) in the file.
ANNOT_ORIG_SIZE = None  # e.g., (1920, 1080) or None

# YOLO model weights (your trained checkpoint)
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s\weights\best.pt"

# YOLO inference
YOLO_CONF = 0.25
YOLO_IOU  = 0.70
YOLO_IMGSZ = 960
ALLOW_CLASSES = None  # e.g., {0,1} if you trained only Pedestrian/Biker; None = all

# Display
LEFT_WIN  = "GT (rescaled)"
RIGHT_WIN = "YOLO detections"
WINDOW_W, WINDOW_H = 1280, 720
DRAW_FPS = True
THICK = 2
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FS    = 0.5
# =========================

# ---------- helpers ----------
def parse_annotations_txt(path):
    """
    Each line: track_id xmin ymin xmax ymax frame lost occluded generated "label"
    Returns: dict[frame] -> list of dict(id, bbox(np.array xyxy), label, lost)
    """
    per_frame = defaultdict(list)
    max_x = 1.0
    max_y = 1.0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            cols = ln.split()
            if len(cols) < 10:
                continue
            tid = int(cols[0])
            xmin = float(cols[1]); ymin = float(cols[2]); xmax = float(cols[3]); ymax = float(cols[4])
            fr = int(cols[5]); lost = int(cols[6]); # occ = int(cols[7]); gen = int(cols[8])
            label_raw = " ".join(cols[9:])
            label = label_raw.strip().strip('"')
            max_x = max(max_x, xmin, xmax)
            max_y = max(max_y, ymin, ymax)
            per_frame[fr].append(dict(id=tid, bbox=np.array([xmin,ymin,xmax,ymax], dtype=float),
                                      label=label, lost=lost))
    return per_frame, max_x, max_y

def color_for_label(name):
    table = {
        "Pedestrian": (50,200,50),
        "Biker": (60,140,255),
        "Skater": (180,160,50),
        "Cart": (200,50,200),
        "Car": (60,60,220),
        "Bus": (0,120,200),
    }
    return table.get(name, (180,180,180))

def draw_box(img, xyxy, color, txt=None, thick=2):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)
    if txt:
        cv2.putText(img, txt, (x1, max(12,y1-4)), FONT, FS, color, 2, cv2.LINE_AA)

# ---------- YOLO ----------
def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def yolo_detect(model, frame_bgr):
    res = model.predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    for b in res.boxes:
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        if ALLOW_CLASSES is not None and cls_id not in ALLOW_CLASSES:
            continue
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        sc = float(b.conf[0].item()) if b.conf is not None else 0.0
        out.append((np.array([x1,y1,x2,y2], dtype=float), cls_id, sc))
    return out, getattr(res, "names", None)

# ---------- main ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ann_by_frame, max_ax, max_ay = parse_annotations_txt(ANN_PATH)
    if ANNOT_ORIG_SIZE is not None:
        sx = W / float(ANNOT_ORIG_SIZE[0])
        sy = H / float(ANNOT_ORIG_SIZE[1])
    else:
        sx = W / float(max_ax if max_ax > 0 else 1.0)
        sy = H / float(max_ay if max_ay > 0 else 1.0)

    yolo = load_yolo(YOLO_WEIGHTS)

    cv2.namedWindow(LEFT_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RIGHT_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LEFT_WIN, WINDOW_W, WINDOW_H)
    cv2.resizeWindow(RIGHT_WIN, WINDOW_W, WINDOW_H)

    paused = False
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            # hold current
            ok, frame = cap.read()
            if not ok:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        t0 = time.time()

        # Left: annotations
        left = frame.copy()
        for item in ann_by_frame.get(frame_idx, []):
            if item["lost"] == 1:
                continue
            x1,y1,x2,y2 = item["bbox"]
            x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy
            draw_box(left, (x1,y1,x2,y2), color_for_label(item["label"]), f'{item["label"]}#{item["id"]}', THICK)

        # Right: YOLO
        right = frame.copy()
        dets, names = yolo_detect(yolo, frame)
        for (b, cls, sc) in dets:
            name = names.get(cls, f"cls{cls}") if isinstance(names, dict) else f"cls{cls}"
            col = color_for_label(name)
            draw_box(right, b, col, f"{name} {sc:.2f}", THICK)

        if DRAW_FPS:
            fps = 1.0 / max(1e-6, time.time() - t0)
            cv2.putText(left,  f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12,24), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(right, f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12,24), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

        if (left.shape[1], left.shape[0]) != (WINDOW_W, WINDOW_H):
            left  = cv2.resize(left,  (WINDOW_W, WINDOW_H))
        if (right.shape[1], right.shape[0]) != (WINDOW_W, WINDOW_H):
            right = cv2.resize(right, (WINDOW_W, WINDOW_H))

        cv2.imshow(LEFT_WIN, left)
        cv2.imshow(RIGHT_WIN, right)

        k = cv2.waitKey(1 if not paused else 0) & 0xFF
        if k in (ord('q'), 27): break
        elif k == ord('p'):     paused = True
        elif k == ord('r'):     paused = False
        elif paused and k == ord('o'):
            frame_idx = min(total-1, frame_idx+1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif paused and k == ord('i'):
            frame_idx = max(0, frame_idx-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
