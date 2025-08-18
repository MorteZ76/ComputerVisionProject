import cv2
import os
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# =========================
# ====== HYPERPARAMS ======
# =========================

# --- Paths ---
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
# ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"

# --- Classes and colors (must match your training order) ---
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Biker",
    2: "Skater",
    3: "Cart",
    4: "Car",
    5: "Bus",
}
LABEL_TO_ID = {
    "Pedestrian": 0,
    "Biker": 1,
    "Skater": 2,
    "Cart": 3,
    "Car": 4,
    "Bus": 5,
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
}

# --- Detection ---
DET_CONF_THRES = 0.45
DET_IOU_NMS = 0.6

# --- Evaluation ---
IOU_MATCH_THRES = 0.5      # detection-GT match threshold
COMPUTE_CENTER_MSE = True  # MSE over matched center pairs
RESULTS_CSV = "video0_det_vs_gt.csv"

# --- GT usage and playback ---
SKIP_GT_OCCLUDED = True
PAUSE_ON_START = False

# --- Drawing ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
TRAJ_MAX_LEN = 2000  # for optional GT trajectory lines

# =========================
# ======  IMPORTS   =======
# =========================
try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics not found. Install: pip install ultralytics")
    raise

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    print("scipy not found. Install: pip install scipy")
    raise

# =========================
# ======  HELPERS    ======
# =========================

def iou_xyxy(a, b):
    """IoU for [x1,y1,x2,y2]. a: (N,4) b:(M,4) -> (N,M)"""
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    x11, y11, x12, y12 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    x21, y21, x22, y2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
    inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0, np.minimum(y12, y2) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    area_a = np.maximum(0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

def xyxy_to_cxcy(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def parse_sdd_annotations(path):
    """
    Returns:
      frames[frame_idx] -> list of dicts:
        { 'id': int, 'bbox':[x1,y1,x2,y2], 'lost':0/1, 'occluded':0/1, 'label':str }
      (W_ref, H_ref) inferred canvas size
    """
    frames = defaultdict(list)
    max_x = 0
    max_y = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            tid = int(parts[0])
            x1 = float(parts[1]); y1 = float(parts[2])
            x2 = float(parts[3]); y2 = float(parts[4])
            frame = int(parts[5])
            lost = int(parts[6]); occl = int(parts[7])
            label_raw = " ".join(parts[9:])
            label = label_raw.strip().strip('"')
            frames[frame].append({
                "id": tid,
                "bbox": [x1, y1, x2, y2],
                "lost": lost,
                "occluded": occl,
                "label": label
            })
            max_x = max(max_x, x2); max_y = max(max_y, y2)
    return frames, (int(math.ceil(max_x)), int(math.ceil(max_y)))

def scale_bbox(b, sx, sy):
    x1, y1, x2, y2 = b
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

def draw_box(frame, bbox, color, text=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(frame, text, (x1, max(0, y1 - 6)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def match_dets_to_gt(gt_boxes, gt_cls, det_boxes, det_cls, iou_th=0.5):
    """
    Class-aware Hungarian on cost = 1 - IoU.
    Returns: matches [(gi, di, iou)], unmatched_gt_idx, unmatched_det_idx
    """
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(gt_boxes))), list(range(len(det_boxes)))

    A = np.array(gt_boxes, dtype=np.float32)
    B = np.array(det_boxes, dtype=np.float32)
    I = iou_xyxy(A, B)

    # Block class-mismatched pairs by setting very low IoU
    for gi in range(len(gt_cls)):
        for di in range(len(det_cls)):
            if gt_cls[gi] != det_cls[di]:
                I[gi, di] = -1.0

    cost = 1.0 - I
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_g = set()
    matched_d = set()
    for gi, di in zip(row_ind, col_ind):
        iou_val = I[gi, di]
        if iou_val >= iou_th:
            matches.append((gi, di, float(iou_val)))
            matched_g.add(gi)
            matched_d.add(di)

    unmatched_g = [i for i in range(len(gt_boxes)) if i not in matched_g]
    unmatched_d = [i for i in range(len(det_boxes)) if i not in matched_d]
    return matches, unmatched_g, unmatched_d

# =========================
# ======  MAIN LOOP  ======
# =========================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
    sx = W / float(W_ref if W_ref > 0 else W)
    sy = H / float(H_ref if H_ref > 0 else H)

    model = YOLO(YOLO_WEIGHTS)

    # Accumulators
    totals = {
        "tp": 0, "fp": 0, "fn": 0,
        "per_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    }
    mse_values = []             # per-frame center MSE over matched pairs
    rows = []                   # CSV rows: frame, gt_id, gt_cls, det_cls, det_conf, iou, gt_cx, gt_cy, det_cx, det_cy
    gt_paths = defaultdict(deque)  # optional GT trajectories

    frame_idx = 0
    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i): return max(0, min(total_frames - 1, i))
    def _seek_to(idx):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        did_seek = True

    if PAUSE_ON_START:
        print("Paused. Press 'r' to resume, 'p' to pause. 'o' next, 'i' prev, 'l' +100, 'k' -100.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_gt = frame.copy()
        vis_det = frame.copy()

        # ---------- Ground Truth ----------
        gt_boxes = []
        gt_cls = []
        gt_ids = []
        gt_centers = []

        if frame_idx in frames_gt:
            for ann in frames_gt[frame_idx]:
                if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1):
                    continue
                bb = scale_bbox(ann["bbox"], sx, sy)
                lbl = ann["label"]
                cid = LABEL_TO_ID.get(lbl, 0)
                gt_boxes.append(bb)
                gt_cls.append(cid)
                gt_ids.append(ann["id"])
                cx, cy = xyxy_to_cxcy(bb)
                gt_centers.append((cx, cy))

                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN:
                    gt_paths[ann["id"]].popleft()

                draw_box(vis_gt, bb, CLASS_COLORS.get(cid, (0, 255, 0)), f"{CLASS_NAMES.get(cid, cid)} GT:{ann['id']}")

                pts = list(gt_paths[ann["id"]])
                for i in range(1, len(pts)):
                    cv2.line(vis_gt, pts[i-1], pts[i], CLASS_COLORS.get(cid, (255, 255, 255)), 2)

        # ---------- YOLO Detections ----------
        yres = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
        det_boxes = []
        det_cls = []
        det_conf = []

        if len(yres) > 0:
            r = yres[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(xyxy, conf, cls):
                    det_boxes.append(b.astype(np.float32))
                    det_cls.append(int(c))
                    det_conf.append(float(s))
                    draw_box(vis_det, b, CLASS_COLORS.get(int(c), (200, 200, 200)),
                             f"{CLASS_NAMES.get(int(c), c)} {s:.2f}")

        # ---------- Matching and Metrics ----------
        matches, un_g, un_d = match_dets_to_gt(gt_boxes, gt_cls, det_boxes, det_cls, IOU_MATCH_THRES)

        tp = len(matches)
        fp = len(un_d)
        fn = len(un_g)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        # Per-class updates
        for gi, di, _ in matches:
            c = gt_cls[gi]
            totals["per_class"][c]["tp"] += 1
        for gi in un_g:
            c = gt_cls[gi]
            totals["per_class"][c]["fn"] += 1
        for di in un_d:
            c = det_cls[di]
            totals["per_class"][c]["fp"] += 1

        # Optional center MSE on matched pairs
        if COMPUTE_CENTER_MSE and tp > 0:
            gC = np.array([gt_centers[gi] for gi, _, _ in matches], dtype=np.float32)
            dC = np.array([xyxy_to_cxcy(det_boxes[di]) for _, di, _ in matches], dtype=np.float32)
            diff = gC - dC
            mse_values.append(float(np.mean(np.sum(diff * diff, axis=1))))

        # Log CSV rows
        for gi, di, iou_val in matches:
            gc = gt_centers[gi] if gi < len(gt_centers) else (np.nan, np.nan)
            dc = xyxy_to_cxcy(det_boxes[di]) if di < len(det_boxes) else (np.nan, np.nan)
            rows.append([
                frame_idx,
                gt_ids[gi] if gi < len(gt_ids) else -1,
                gt_cls[gi] if gi < len(gt_cls) else -1,
                det_cls[di] if di < len(det_cls) else -1,
                det_conf[di] if di < len(det_conf) else np.nan,
                iou_val,
                float(gc[0]), float(gc[1]),
                float(dc[0]), float(dc[1]),
            ])

        # On-screen per-frame stats
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        y0 = 20
        for text in [
            f"Frame {frame_idx}/{total_frames-1}",
            f"TP:{tp} FP:{fp} FN:{fn}",
            f"Prec:{prec:.3f} Rec:{rec:.3f} F1:{f1:.3f}",
            f"MSE:{(mse_values[-1] if (COMPUTE_CENTER_MSE and len(mse_values)>0) else float('nan')):.2f}"
        ]:
            cv2.putText(vis_det, text, (10, y0), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_det, text, (10, y0), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 22

        cv2.namedWindow("GT (rescaled annotations + trajectories)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detections (no tracking) + per-frame metrics", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GT (rescaled annotations + trajectories)", 860, 900)
        cv2.resizeWindow("Detections (no tracking) + per-frame metrics", 860, 900)

        # Show
        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        cv2.imshow("Detections (no tracking) + per-frame metrics", vis_det)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            pass
        if key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False

        if paused:
            if key == ord('o'):
                _seek_to(frame_idx + 1)
            elif key == ord('i'):
                _seek_to(frame_idx - 1)
            elif key == ord('l'):
                _seek_to(frame_idx + 100)
            elif key == ord('k'):
                _seek_to(frame_idx - 100)

        if not paused and not did_seek:
            frame_idx += 1
        did_seek = False

    cap.release()
    cv2.destroyAllWindows()

    # ===== Summary =====
    TP = totals["tp"]; FP = totals["fp"]; FN = totals["fn"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Overall TP:{TP} FP:{FP} FN:{FN}")
    print(f"Overall Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f}")
    if COMPUTE_CENTER_MSE and len(mse_values) > 0:
        print(f"Overall center MSE (matched only): {np.mean(mse_values):.3f}")

    # Save matches CSV
    if rows:
        df = pd.DataFrame(rows, columns=[
            "frame", "gt_id", "gt_cls", "det_cls", "det_conf", "iou",
            "gt_cx", "gt_cy", "det_cx", "det_cy"
        ])
        df.to_csv(RESULTS_CSV, index=False)
        print(f"Per-match results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()
