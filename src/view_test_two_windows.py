# scripts/view_test_two_windows.py
# Left: ground-truth annotations (rescaled) with class colors and track IDs.
# Right: YOLOv8 detections from your trained weights.
# Keys: o=next, i=prev, l=+10, k=-10, a=autoplay toggle, q/ESC=quit.

import os, re, cv2, time
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
# Test image folders (all frames extracted)
IMG_DIRS = {
    "video0": r"C:\Users\morte\ComputerVisionProject\Made dataset\images\test\little_video0",
    "video3": r"C:\Users\morte\ComputerVisionProject\Made dataset\images\test\little_video3",
}
# Master annotation files (Stanford format, need rescale)
ANN_PATHS = {
    "video0": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video0\annotations.txt",
    "video3": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video3\annotations.txt",
}
# If you know the original annotation image size, put (W,H). Else None to infer from max coords.
ANNOT_ORIG_SIZE = None  # e.g., (1920,1080) or None to auto-infer per video

# YOLO model
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"
YOLO_CONF = 0.25
YOLO_IOU  = 0.7
YOLO_IMGSZ = 960
# If you trained only Pedestrian=0, Biker=1, you can restrict to these:
ALLOW_CLASSES = {0, 1}  # set to None to allow all classes from your model

# Visualization
LEFT_WIN  = "GT (rescaled)"
RIGHT_WIN = "YOLO detections"
WINDOW_SCALE = 1.0  # >1 to enlarge
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FS    = 0.6
THICK = 2
AUTOPLAY = False
AUTOPLAY_DELAY_MS = 1  # reduce if too slow

# Class color map for GT labels
COLOR_MAP = {
    "Pedestrian": (50, 200, 50),
    "Biker": (60, 140, 255),
    "Skater": (180, 160, 50),
    "Cart": (200, 50, 200),
    "Car": (60, 60, 220),
    "Bus": (0, 120, 200),
    "Truck": (0, 80, 170),
    "Other": (180, 180, 180),
}
# =========================

def list_frames(img_dir):
    xs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    xs.sort()
    return xs

def extract_frame_idx(path):
    m = re.search(r'(\d+)(?=\.[a-zA-Z]+$)', os.path.basename(path))
    return int(m.group(1)) if m else None

def parse_annotations(ann_path, out_W, out_H, orig_size=None):
    """
    Returns:
      per_frame: dict[int -> list of {id, bbox=np.array([x1,y1,x2,y2]), label}]
      scale_x, scale_y used
    """
    raw = {}
    max_x, max_y = 1.0, 1.0
    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().split()
            if len(s) < 10: continue
            try:
                tid  = int(s[0])
                x1   = float(s[1]); y1 = float(s[2])
                x2   = float(s[3]); y2 = float(s[4])
                fr   = int(s[5])
                lost = int(s[6])
            except ValueError:
                continue
            label_raw = " ".join(s[9:]).strip().strip('"')
            label = label_raw if label_raw else "Other"
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
            raw.setdefault(fr, []).append(dict(id=tid, bbox=np.array([x1,y1,x2,y2], float),
                                               label=label, lost=lost))

    if orig_size is not None:
        oW, oH = orig_size
        sx = out_W / float(max(1.0, oW))
        sy = out_H / float(max(1.0, oH))
    else:
        sx = out_W / float(max(1.0, max_x))
        sy = out_H / float(max(1.0, max_y))

    per_frame = {}
    for fr, items in raw.items():
        lst=[]
        for r in items:
            x1,y1,x2,y2 = r["bbox"] * np.array([sx,sy,sx,sy], float)
            # clamp and skip degenerate
            x1 = max(0.0, min(out_W-1.0, x1)); y1 = max(0.0, min(out_H-1.0, y1))
            x2 = max(0.0, min(out_W-1.0, x2)); y2 = max(0.0, min(out_H-1.0, y2))
            if x2 <= x1 or y2 <= y1: continue
            lst.append(dict(id=r["id"], bbox=np.array([x1,y1,x2,y2], float),
                            label=r["label"], lost=r["lost"]))
        if lst: per_frame[fr]=lst
    return per_frame, sx, sy

def color_for(label):
    return COLOR_MAP.get(label, COLOR_MAP["Other"])

def draw_gt(disp, items):
    for r in items:
        if r.get("lost",0)==1:  # hide lost by default
            continue
        x1,y1,x2,y2 = r["bbox"].astype(int)
        lab = r["label"]
        col = color_for(lab)
        cv2.rectangle(disp, (x1,y1), (x2,y2), col, THICK, cv2.LINE_AA)
        cv2.putText(disp, f'{lab}#{r["id"]}', (x1, max(12,y1-6)), FONT, FS, col, 2, cv2.LINE_AA)

def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def yolo_detect(model, frame):
    res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    out=[]
    names = res.names if hasattr(res, "names") else {}
    if res.boxes is None:
        return out
    H, W = frame.shape[:2]
    for b in res.boxes:
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        if ALLOW_CLASSES is not None and cls_id not in ALLOW_CLASSES:
            continue
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        # clip and skip degenerate
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1: continue
        score = float(b.conf[0].item()) if b.conf is not None else 0.0
        name = names.get(cls_id, f"cls{cls_id}") if isinstance(names, dict) else f"cls{cls_id}"
        out.append(dict(box=(int(x1),int(y1),int(x2),int(y2)), name=name, score=score))
    return out

def main():
    # prepare per-video frame lists and per-video GT maps with scales
    seqs = {}
    for key in ("video0", "video3"):
        img_dir = IMG_DIRS[key]
        frames = list_frames(img_dir)
        if not frames:
            print(f"[error] no frames in {img_dir}")
            return
        # image size from first frame
        sample = cv2.imread(frames[0])
        if sample is None:
            print(f"[error] cannot read first image: {frames[0]}")
            return
        H, W = sample.shape[:2]
        ann_map, sx, sy = parse_annotations(ANN_PATHS[key], W, H, orig_size=ANNOT_ORIG_SIZE)
        seqs[key] = dict(frames=frames, W=W, H=H, ann=ann_map, sx=sx, sy=sy)

    # YOLO
    yolo = load_yolo(YOLO_WEIGHTS)

    # windows
    cv2.namedWindow(LEFT_WIN,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(RIGHT_WIN, cv2.WINDOW_NORMAL)

    # iterate both sequences back-to-back
    entries = []
    for key in ("video0","video3"):
        for p in seqs[key]["frames"]:
            entries.append((key, p))
    idx = 0
    N = len(entries)

    global AUTOPLAY
    while True:
        key, img_path = entries[idx]
        info = seqs[key]
        frame = cv2.imread(img_path)
        if frame is None:
            # skip unreadable
            idx = (idx + 1) % N
            continue

        dispL = frame.copy()
        dispR = frame.copy()

        # LEFT: GT
        fidx = extract_frame_idx(img_path)
        gt_items = info["ann"].get(fidx, [])
        draw_gt(dispL, gt_items)

        # RIGHT: YOLO
        t0 = time.time()
        dets = yolo_detect(yolo, frame)
        for d in dets:
            x1,y1,x2,y2 = d["box"]
            col = (0,255,0)
            cv2.rectangle(dispR, (x1,y1), (x2,y2), col, THICK, cv2.LINE_AA)
            cv2.putText(dispR, f'{d["name"]} {d["score"]:.2f}', (x1, max(12,y1-6)),
                        FONT, FS, col, 2, cv2.LINE_AA)
        fps = 1.0 / max(1e-6, time.time()-t0)

        # overlays
        head = f'{key}  frame={fidx if fidx is not None else "?"}  ({idx+1}/{N})'
        cv2.putText(dispL, head, (12,24), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(dispR, head + f"  det={len(dets)}  FPS~{fps:.1f}", (12,24),
                    FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if WINDOW_SCALE != 1.0:
            dispL = cv2.resize(dispL, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE, interpolation=cv2.INTER_LINEAR)
            dispR = cv2.resize(dispR, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(LEFT_WIN,  dispL)
        cv2.imshow(RIGHT_WIN, dispR)

        if AUTOPLAY:
            k = cv2.waitKey(AUTOPLAY_DELAY_MS) & 0xFF
            idx = (idx + 1) % N
        else:
            k = cv2.waitKey(0) & 0xFF

        if k in (ord('q'), 27):  # quit
            break
        elif k == ord('o'):       # next
            idx = min(N-1, idx + 1)
        elif k == ord('i'):       # prev
            idx = max(0, idx - 1)
        elif k == ord('l'):       # +10
            idx = min(N-1, idx + 10)
        elif k == ord('k'):       # -10
            idx = max(0, idx - 10)
        elif k == ord('a'):       # toggle autoplay
            AUTOPLAY = not AUTOPLAY

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
