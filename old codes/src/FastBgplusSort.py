# -*- coding: utf-8 -*-
"""
SDD MOT Viewer: Left = GT annotations (rescaled). Right = tracker overlay.
Supports two trackers: A) BackgroundSub+Kalman (SORT-like), B) YOLOv8 + BYTETrack-like.
Keys: P=pause, R=resume, O=next, I=prev, Q=quit.
"""

# =========================
# HYPERPARAMETERS (EDIT ME)
# =========================

# ---- Paths ----
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\little\video0\video.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\little\video0\annotations.txt"

# If you know the original image size used to annotate (from SDD meta), set here for rescaling
# e.g., (1920,1080). If None, first frame size is assumed and no scale is applied.
# take the original size from the first frame of the video 
ANNOT_ORIG_SIZE = None  # (width, height) or None

# ---- Global viewer ----
WINDOW_SCALE = 1.0          # multiply final display size (increase for bigger windows)
LEFT_WIN_NAME = "GT (rescaled)"
RIGHT_WIN_NAME = "Tracker (A=BGSub+KF, B=YOLO+BYTETrack)"
START_FRAME = 0
DRAW_FPS = True
SHOW_ID_TRAILS = True
TRAIL_LEN = 32

# ---- Class colors (GT + detector) ----
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

# ==================================
# BASELINE A: BackgroundSub + KF/SORT
# ==================================
# MOG2
MOG2_HISTORY = 500
MOG2_VARTHRESH = 16
MOG2_DETECT_SHADOWS = True
SHADOW_VALUE = 127          # OpenCV MOG2 default
SHADOW_SUPPRESS = True      # remove shadow-like pixels
SHADOW_A_CH_DIFF = 10       # LAB a/b channel abs diff threshold for shadow removal
MOG2_LR = -1                # -1 for auto

# Morphology
MORPH_OPEN = 3              # kernel size for opening (0 disables)
MORPH_CLOSE = 5             # kernel size for closing (0 disables)

# Contour filtering
MIN_CONTOUR_AREA = 120      # in pixels, after scaling to video
MAX_ASPECT_RATIO = 3.0      # w/h or h/w allowed
MIN_BOX_W = 6
MIN_BOX_H = 6

# ROI mask polygon (image coords). None disables.
ROI_POLY = None  # e.g., [(100,200),(1700,200),(1700,900),(100,900)]

# SORT-like tracker
MAX_AGE = 30
MIN_HITS = 3
IOU_MATCH_THRESH = 0.2

# Purple legend box to visualize MIN_CONTOUR_AREA size
SHOW_MIN_AREA_BOX = True
MIN_BOX_LEGEND_POS = (12, 12)  # top-left corner

# ======================================
# BASELINE B: YOLOv8 + BYTETrack-like TbD
# ======================================
USE_YOLO = False             # requires `pip install ultralytics`
YOLO_MODEL = "yolov8s.pt"   # any Ultralytics v8 model
YOLO_IMG_SIZE = 960
YOLO_CONF = 0.30            # detector conf threshold before BYTETrack stages
YOLO_NMS_IOU = 0.70
DETECT_CLASSES = {0, 1, 2, 3, 5, 7}  # COCO: 0=person,1=bicycle,2=car,3=motorcycle,5=bus,7=truck

# BYTETrack thresholds
BYTE_HIGH_THRESH = 0.50
BYTE_LOW_THRESH = 0.10
BYTE_IOU_MATCH = 0.20
BYTE_MAX_AGE = 30
BYTE_MIN_HITS = 3

# =========================
# END HYPERPARAMETERS
# =========================

import os
import cv2
import math
import time
import numpy as np

# Optional: Ultralytics YOLO
try:
    if USE_YOLO:
        from ultralytics import YOLO
    YOLO_OK = USE_YOLO
except Exception:
    YOLO_OK = False

# -------------------------
# Utilities
# -------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter + 1e-6
    return inter / union

def xywh_to_xyxy(x, y, w, h):
    return np.array([x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0], dtype=float)

def clip_box(b, W, H):
    b[0] = max(0, min(W-1, b[0]))
    b[1] = max(0, min(H-1, b[1]))
    b[2] = max(0, min(W-1, b[2]))
    b[3] = max(0, min(H-1, b[3]))
    return b

def color_for_label(lbl):
    return COLOR_MAP.get(lbl, COLOR_MAP["Other"])

# -------------------------
# Annotation loader (rescale)
# -------------------------
def parse_annotations(path, out_W, out_H, orig_size=None):
    """
    Returns dict: frame -> list of dicts {id, bbox(xyxy), label}
    Rescales xy coords if orig_size given.
    """
    scale_x, scale_y = 1.0, 1.0
    if orig_size is not None:
        ox, oy = orig_size
        scale_x = out_W / float(ox)
        scale_y = out_H / float(oy)

    per_frame = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.strip().split()
            if len(cols) < 10:
                continue
            tid = int(cols[0])
            xmin = float(cols[1]) * scale_x
            ymin = float(cols[2]) * scale_y
            xmax = float(cols[3]) * scale_x
            ymax = float(cols[4]) * scale_y
            frame = int(cols[5])
            # lost = int(cols[6]); occluded = int(cols[7]); generated = int(cols[8])
            label_raw = " ".join(cols[9:])
            label = label_raw.strip().strip('"')

            if frame not in per_frame:
                per_frame[frame] = []
            per_frame[frame].append({
                "id": tid,
                "bbox": np.array([xmin, ymin, xmax, ymax], dtype=float),
                "label": label
            })
    return per_frame

# -------------------------
# Simple Kalman track (SORT-like)
# -------------------------
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # state: [cx, cy, s, r, vx, vy, vs]
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / h

        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.statePre = np.array([[cx],[cy],[s],[r],[0],[0],[0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 0
        self.no_losses = 0
        self.history = []

    def predict(self):
        self.kf.predict()
        self.history.append(self.get_state())
        if len(self.history) > TRAIL_LEN:
            self.history.pop(0)
        return self.history[-1]

    def update(self, bbox):
        x1,y1,x2,y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w*h
        r = w/max(1.0, h)
        z = np.array([[cx],[cy],[s],[r]], dtype=np.float32)
        self.kf.correct(z)
        self.hits += 1
        self.no_losses = 0

    def get_state(self):
        cx, cy, s, r = self.kf.statePost[:4, 0]
        w = math.sqrt(max(1.0, s*r))
        h = max(1.0, s / max(1.0, w))
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return np.array([x1,y1,x2,y2], dtype=float)

# -------------------------
# SORT-like manager
# -------------------------
class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_thresh=0.2):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.trackers = []

    def update(self, detections):
        # Predict
        for t in self.trackers:
            t.predict()
            t.no_losses += 1

        # Match by IoU
        N = len(self.trackers)
        M = len(detections)
        if N == 0 or M == 0:
            matches = []
            unmatched_trk = list(range(N))
            unmatched_det = list(range(M))
        else:
            iou_mat = np.zeros((N, M), dtype=np.float32)
            for i, t in enumerate(self.trackers):
                tb = t.get_state()
                for j, d in enumerate(detections):
                    iou_mat[i, j] = iou_xyxy(tb, d)
            matched_idx = []
            # Greedy
            used_t, used_d = set(), set()
            for _ in range(min(N, M)):
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i, j] < self.iou_thresh or i in used_t or j in used_d:
                    iou_mat[i, j] = -1
                    continue
                matched_idx.append((i, j))
                used_t.add(i); used_d.add(j)
                iou_mat[i, :] = -1; iou_mat[:, j] = -1
            unmatched_trk = [i for i in range(N) if i not in {m[0] for m in matched_idx}]
            unmatched_det = [j for j in range(M) if j not in {m[1] for m in matched_idx}]
            matches = matched_idx

        # Update matched
        for i, j in matches:
            self.trackers[i].update(detections[j])

        # Create new for unmatched detections
        for j in unmatched_det:
            self.trackers.append(KalmanBoxTracker(detections[j]))

        # Remove dead
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        # Return current boxes with ids
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.no_losses == 0:
                results.append((t.get_state(), t.id, t.history))
        return results

# -------------------------
# BYTETrack-like association
# -------------------------
class ByteTrackLike:
    def __init__(self, max_age=30, min_hits=3, iou_match=0.2):
        self.trk = Sort(max_age=max_age, min_hits=min_hits, iou_thresh=iou_match)

    def update(self, dets_high, dets_low):
        # First associate with high confidence
        out1 = self.trk.update(dets_high)
        # For unmatched tracks, try low-conf boxes indirectly by one more update pass
        # Gather current tracker boxes to find unmatched; here we do a second SORT pass with low-conf as detections
        out2 = self.trk.update(dets_low) if len(dets_low) > 0 else out1
        # Return final state
        # After second pass we fetch again
        final = self.trk.update([])  # no new dets, just to get current set
        return final

# -------------------------
# BG-sub detection
# -------------------------
def build_roi_mask(shape, poly):
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    else:
        mask[:, :] = 255
    return mask

def remove_shadows(frame_bgr, fgmask):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # Kill pixels near shadow gray by checking a/b small variation
    ab_diff = cv2.absdiff(A, B)
    kill = (ab_diff < SHADOW_A_CH_DIFF) & (fgmask == SHADOW_VALUE)
    fgmask2 = fgmask.copy()
    fgmask2[kill] = 0
    # Binariyze 0/255
    fgmask2[fgmask2 != 0] = 255
    return fgmask2

def detect_bgsub_boxes(frame, bg, roi_mask):
    fg = bg.apply(frame, learningRate=MOG2_LR)
    if MOG2_DETECT_SHADOWS and SHADOW_SUPPRESS:
        fg = remove_shadows(frame, fg)
    if roi_mask is not None:
        fg = cv2.bitwise_and(fg, roi_mask)

    # Morph
    if MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    if MORPH_CLOSE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue
        ar = max(w / float(h), h / float(w))
        if ar > MAX_ASPECT_RATIO:
            continue
        boxes.append(np.array([x, y, x + w, y + h], dtype=float))
    return boxes, fg

# -------------------------
# YOLO detector wrapper
# -------------------------
class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def infer(self, frame):
        # returns list of (xyxy, cls_id, score)
        res = self.model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, iou=YOLO_NMS_IOU, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            if cls_id not in DETECT_CLASSES:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            score = float(b.conf[0].item())
            out.append((np.array([x1, y1, x2, y2], dtype=float), cls_id, score))
        return out

# COCO id to coarse SDD-ish labels
COCO_ID_TO_LABEL = {
    0: "Pedestrian",
    1: "Biker",
    2: "Car",
    3: "Biker",   # motorcycle -> treat as Biker
    5: "Bus",
    7: "Truck",
}

# -------------------------
# Main
# -------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ANNOT_ORIG_SIZE = (W, H) 
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Annotations
    ann = parse_annotations(ANNOT_PATH, W, H, orig_size=ANNOT_ORIG_SIZE)

    # BGSub
    bg = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VARTHRESH,
                                            detectShadows=MOG2_DETECT_SHADOWS)
    roi_mask = build_roi_mask((H, W), ROI_POLY)
    trkA = Sort(MAX_AGE, MIN_HITS, IOU_MATCH_THRESH)

    # YOLO + BYTETrack-like
    if YOLO_OK:
        yolo = YoloDetector(YOLO_MODEL)
        trkB = ByteTrackLike(BYTE_MAX_AGE, BYTE_MIN_HITS, BYTE_IOU_MATCH)
    else:
        yolo = None
        trkB = None

    # playback
    frame_idx = START_FRAME
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, START_FRAME))
    paused = False

    # UI windows
    cv2.namedWindow(LEFT_WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RIGHT_WIN_NAME, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_idx = cur_idx
        else:
            # stay at current frame; to support prev/next we reposition cap when keys pressed
            ok, frame = cap.read()
            if not ok:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        dispL = frame.copy()
        dispR = frame.copy()
        t0 = time.time()

        # ---------- LEFT: draw GT ----------
        g = ann.get(frame_idx, [])
        # draw per class color and show track id
        for r in g:
            x1,y1,x2,y2 = r["bbox"].astype(int)
            lbl = r["label"]
            color = color_for_label(lbl)
            cv2.rectangle(dispL, (x1,y1), (x2,y2), color, 2)
            cv2.putText(dispL, f'{lbl} #{r["id"]}', (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ---------- RIGHT: run both trackers and overlay ----------
        # A) BGSub detections
        boxesA, fgmask = detect_bgsub_boxes(frame, bg, roi_mask)

        # Visualize min-area purple box
        if SHOW_MIN_AREA_BOX:
            s = int(math.sqrt(MIN_CONTOUR_AREA))
            x0, y0 = MIN_BOX_LEGEND_POS
            cv2.rectangle(dispR, (x0, y0), (x0 + s, y0 + s), (200, 0, 200), 2)
            cv2.putText(dispR, f"MinArea={MIN_CONTOUR_AREA}", (x0, y0 + s + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1, cv2.LINE_AA)

        # Track A
        tracksA = trkA.update(boxesA)
        for (box, tid, hist) in tracksA:
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(dispR, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(dispR, f"A#{tid}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            if SHOW_ID_TRAILS and len(hist) >= 2:
                pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in hist]
                for i in range(1, len(pts)):
                    cv2.line(dispR, pts[i-1], pts[i], (0,200,200), 2)

        # B) YOLO + BYTETrack-like (if available)
        if YOLO_OK:
            dets = yolo.infer(frame)
            # split high and low confidence
            high = [d[0] for d in dets if d[2] >= BYTE_HIGH_THRESH]
            low  = [d[0] for d in dets if (BYTE_LOW_THRESH <= d[2] < BYTE_HIGH_THRESH)]
            outB = trkB.update(high, low)
            # draw with class color if available from current detections by nearest IoU
            for (box, tid, hist) in outB:
                x1,y1,x2,y2 = box.astype(int)
                # guess label by the best IoU det
                lbl = "Pedestrian"
                best_iou, best_lbl = 0.0, lbl
                for b, cls_id, sc in dets:
                    i = iou_xyxy(box, b)
                    if i > best_iou:
                        best_iou = i
                        best_lbl = COCO_ID_TO_LABEL.get(cls_id, "Other")
                lbl = best_lbl
                clr = color_for_label(lbl)
                cv2.rectangle(dispR, (x1,y1), (x2,y2), clr, 2)
                cv2.putText(dispR, f"B#{tid}:{lbl}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1, cv2.LINE_AA)
                if SHOW_ID_TRAILS and len(hist) >= 2:
                    pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in hist]
                    for i in range(1, len(pts)):
                        cv2.line(dispR, pts[i-1], pts[i], clr, 2)

        # FPS overlay
        if DRAW_FPS:
            dt = max(1e-6, time.time() - t0)
            fps = 1.0 / dt
            cv2.putText(dispL, f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(dispR, f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # Resize display
        if WINDOW_SCALE != 1.0:
            dispL = cv2.resize(dispL, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE, interpolation=cv2.INTER_LINEAR)
            dispR = cv2.resize(dispR, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(LEFT_WIN_NAME, dispL)
        cv2.imshow(RIGHT_WIN_NAME, dispR)

        # key handler
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False
        elif key == ord('o'):  # next frame
            paused = True
            frame_idx = min(total - 1, frame_idx + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord('i'):  # prev frame
            paused = True
            frame_idx = max(0, frame_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
