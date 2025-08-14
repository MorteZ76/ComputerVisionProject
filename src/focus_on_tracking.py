import cv2

"""
Human Motion Analysis: SDD video0/video3
Two synchronized windows:
  1) Ground-truth (rescaled) + GT trajectories
  2) YOLOv8 detections + ByteTrack-like tracking (Hungarian) + trajectories

Author: you
"""

# =========================
# ====== HYPERPARAMS ======
# =========================

# --- Paths ---
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# To switch to video3, set:
# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
# ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# --- Classes and colors (6 classes as in SDD) ---
# Edit mapping to match your training label order
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Biker",
    2: "Skater",
    3: "Cart",
    4: "Car",
    5: "Bus",
}

# Map SDD label strings to your training class indices (normalize to lowercase)
LABEL_TO_ID = {
    "Pedestrian": 0,
    "Biker": 1,      # or "biker" in your file
    "Skater": 2,     # or "skater"
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

# --- Detection / NMS ---
# DET_CONF_THRES = 0.30
DET_CONF_THRES = 0.50

DET_IOU_NMS = 0.70

# --- ByteTrack-like association ---
BYTE_HIGH_THRES = 0.50   # high-score set
BYTE_LOW_THRES = 0.10    # low-score set
IOU_GATE = 0.20          # minimum IoU to consider a match
MAX_AGE = 30             # frames to keep "alive" without updates
MIN_HITS = 3             # warm-up before rendering id (optional usage)

# --- Drawing / Trajectories ---
MISS_FRAMES_TO_DROP_PATH = 10  # delete trajectory if not seen for 10 frames
TRAJ_MAX_LEN = 2000            # cap stored points per track to avoid memory growth
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2

# --- Metrics / Output ---
TRAJ_CSV = "video3_trajectories.csv"  # video_id,track_id,frame,x,y
COMPUTE_MSE = True

# --- Playback ---
SKIP_GT_OCCLUDED = False  # set True to skip occluded==1 or lost==1 GT boxes
PAUSE_ON_START = False    # press any key to start

# =========================
# ======  IMPORTS   =======
# =========================
import os
import cv2
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not found. Install: pip install ultralytics")
    raise

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    print("scipy not found. Install: pip install scipy")
    raise

# Kalman Filter (optional)
HAS_FILTERPY = True
try:
    from filterpy.kalman import KalmanFilter
except Exception:
    HAS_FILTERPY = False


# =========================
# ======  HELPERS    ======
# =========================

def iou_xyxy(a, b):
    """IoU for [x1,y1,x2,y2] boxes. a: (N,4) b:(M,4) -> (N,M)"""
    N = a.shape[0]
    M = b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    x11, y11, x12, y12 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    x21, y21, x22, y22 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]

    inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    area_a = np.maximum(0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=np.float32)


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# =========================
# ======  TRACKER    ======
# =========================

def make_kf(initial_cxcywh):
    """
    8D state: [cx, cy, w, h, vx, vy, vw, vh]
    """
    if HAS_FILTERPY:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0

        # State transition
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i + 4] = dt

        # Measurement function: observe [cx, cy, w, h]
        kf.H = np.zeros((4, 8), dtype=np.float32)
        kf.H[0, 0] = 1
        kf.H[1, 1] = 1
        kf.H[2, 2] = 1
        kf.H[3, 3] = 1

        # Covariances
        kf.P *= 10.0
        kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        q = 1.0
        kf.Q = np.eye(8, dtype=np.float32) * q

        kf.x[:4, 0] = initial_cxcywh.reshape(4)
        return kf
    else:
        # Lightweight stub with predict/update storing last box directly
        class DummyKF:
            def __init__(self, init_state):
                self.state = init_state.copy()

            def predict(self):
                # No motion model. Keeps last state.
                return self.state

            def update(self, z):
                self.state = z.copy()

            @property
            def x(self):
                return np.concatenate([self.state, np.zeros(4, dtype=np.float32)])[:, None]

        return DummyKF(initial_cxcywh.astype(np.float32))


class Track:
    _next_id = 1

    def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1

        self.cls = int(cls_id)
        self.conf = float(conf)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_idx

        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
        self.kf = make_kf(cxcywh)
        # Ensure state initialized even if Dummy
        if HAS_FILTERPY:
            self.kf.predict()

        self.history = deque(maxlen=TRAJ_MAX_LEN)
        cx, cy, w, h = cxcywh
        self.history.append((int(cx), int(cy)))

    def predict(self):
        if HAS_FILTERPY:
            self.kf.predict()
        # else: state remains
        pred_xyxy = cxcywh_to_xyxy(self.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state)
        return pred_xyxy

    def update(self, bbox_xyxy, cls_id, conf, frame_idx):
        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
        if HAS_FILTERPY:
            self.kf.update(cxcywh)
        else:
            self.kf.update(cxcywh)
        self.cls = int(cls_id)
        self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        cx, cy, w, h = (self.kf.x[:4, 0] if HAS_FILTERPY else self.kf.state)
        self.history.append((int(cx), int(cy)))

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1


class ByteTrackLike:
    def __init__(self, iou_gate=0.2, max_age=30, min_hits=3):
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []

    def _match(self, tracks, dets):
        """
        Hungarian on cost = 1 - IoU. Reject pairs below iou_gate.
        """
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.array([t.predict() for t in tracks], dtype=np.float32)
        det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        iou = iou_xyxy(track_boxes, det_boxes)
        cost = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)
        matches, unmatched_t, unmatched_d = [], [], []

        for r, t in enumerate(tracks):
            if r not in row_ind:
                unmatched_t.append(r)
        for c, _ in enumerate(dets):
            if c not in col_ind:
                unmatched_d.append(c)

        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_gate:
                matches.append((r, c))
            else:
                unmatched_t.append(r)
                unmatched_d.append(c)

        return matches, unmatched_t, unmatched_d

    def update(self, detections, frame_idx):
        """
        detections: list of dicts with keys: bbox, cls, conf
        Two-stage association per ByteTrack: high-score then low-score.
        """
        # Split high and low
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        # Predict done inside _match via track.predict() usage

        # Stage 1: match high-score dets
        matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
        for ti, di in matches:
            t = self.tracks[ti]
            d = high[di]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

        # Unmatched tracks from stage 1 try stage 2 with low-score dets
        remain_tracks = [self.tracks[i] for i in unmatched_t]
        m2, unmatched_t2, unmatched_low = self._match(remain_tracks, low)
        for local_ti, di in m2:
            t = remain_tracks[local_ti]
            d = low[di]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

        # Create new tracks for unmatched high-score detections only (ByteTrack policy)
        new_high_idx = [i for i in unmatched_high]
        for di in new_high_idx:
            d = high[di]
            self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        # Mark unmatched original tracks as missed
        matched_global_t_idx = {ti for ti, _ in matches}
        # add those matched in stage 2 (convert local indices)
        matched_stage2_global = {unmatched_t[i] for i, _ in m2}
        for idx, t in enumerate(self.tracks):
            if idx not in matched_global_t_idx and idx not in matched_stage2_global:
                t.mark_missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self.tracks


# =========================
# ======  GT PARSER  ======
# =========================

def parse_sdd_annotations(path):
    """
    Returns dict:
      frames[frame_idx] -> list of dicts:
         { 'id': int, 'bbox': [x1,y1,x2,y2], 'lost':0/1, 'occluded':0/1, 'label':str }
    Also returns inferred reference (W_ref, H_ref) used by the annotations file.
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
                # Some files may have more columns, but we need at least 10
                continue
            tid = int(parts[0])
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            frame = int(parts[5])
            lost = int(parts[6])
            occl = int(parts[7])
            # parts[8] = generated (unused here)
            # parts[9] = label in quotes
            label_raw = " ".join(parts[9:])
            label = label_raw.strip().strip('"')
            # print (label) 
            frames[frame].append({
                "id": tid,
                "bbox": [x1, y1, x2, y2],
                "lost": lost,
                "occluded": occl,
                "label": label
            })
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
    # Heuristic "reference size" from annotations’ max coords
    W_ref = int(math.ceil(max_x))
    H_ref = int(math.ceil(max_y))
    return frames, (W_ref, H_ref)


def scale_bbox(bbox, scale_x, scale_y):
    x1, y1, x2, y2 = bbox
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


# =========================
# ======  METRICS    ======
# =========================

def mse_per_frame(gt_centers, pred_centers):
    """
    gt_centers: list of (x,y) for GT detections in a frame
    pred_centers: list of (x,y) for predictions in a frame
    Hungarian on squared distance. Returns mean squared error for that frame or None if no matches.
    """
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return None
    G = np.array(gt_centers, dtype=np.float32)
    P = np.array(pred_centers, dtype=np.float32)
    # cost = squared euclidean distance
    diff = G[:, None, :] - P[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    r, c = linear_sum_assignment(cost)
    if len(r) == 0:
        return None
    matched_costs = cost[r, c]
    return float(np.mean(matched_costs))


# =========================
# ======  DRAWING    ======
# =========================

def draw_box_id(frame, bbox, cls, tid, conf=None, color=(0,255,0), label_map=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
    text = f"{cls_name} ID:{tid}"
    if conf is not None:
        text += f" {conf:.2f}"
    cv2.putText(frame, text, (x1, max(0, y1 - 5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def draw_traj(frame, pts, color=(255,255,255)):
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 2)


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

    # Load GT
    frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
    scale_x = W / float(W_ref if W_ref > 0 else W)
    scale_y = H / float(H_ref if H_ref > 0 else H)

    # YOLO
    model = YOLO(YOLO_WEIGHTS)

    # Tracker
    tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

    # Trajectory buffers for drawing removal after misses
    # For tracker: store per track.id -> deque of points and last_seen frame
    track_paths = {}   # id -> deque[(x,y)]
    track_last_seen = {}  # id -> frame_idx
    track_cls = {}     # id -> class id

    # For GT: per object id
    gt_paths = defaultdict(deque)  # id -> deque[(x,y)]

    # For metrics
    traj_rows = []  # video_id,track_id,frame,x,y
    mse_values = []

    frame_idx = 0

    # --- Added: playback state + seek helpers ---
    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i: int) -> int:
        return max(0, min(total_frames - 1, i))

    def _reset_tracking_state():
        nonlocal tracker, track_paths, track_last_seen, track_cls
        tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
        track_paths = {}
        track_last_seen = {}
        track_cls = {}

    def _seek_to(target_idx: int):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(target_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _reset_tracking_state()
        did_seek = True
    # --- End Added ---

    if PAUSE_ON_START:
        print("Paused. Press 'r' to resume, 'p' to pause again.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_gt = frame.copy()
        vis_det = frame.copy()

        # -----------------------
        # Ground Truth window
        # -----------------------
        gt_boxes = []
        gt_centers = []
        if frame_idx in frames_gt:
            for ann in frames_gt[frame_idx]:
                if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1):
                    continue
                bb = scale_bbox(ann["bbox"], scale_x, scale_y)
                cx, cy, _, _ = xyxy_to_cxcywh(np.array(bb, dtype=np.float32))
                gt_boxes.append(bb)
                gt_centers.append((cx, cy))
                # push to GT path
                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN:
                    gt_paths[ann["id"]].popleft()
                lbl = ann["label"]
                gt_cls_id = LABEL_TO_ID.get(lbl, 0)
                color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))  # GT label might not map to your training. Default class color 0.
                draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

        # -----------------------
        # Detection + Tracking
        # -----------------------
        # YOLO inference
        yolo_res = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
        detections = []
        if len(yolo_res) > 0:
            r = yolo_res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(xyxy, conf, cls):
                    detections.append({
                        "bbox": b.astype(np.float32),
                        "conf": float(s),
                        "cls": int(c),
                    })

        # Update tracker
        tracks = tracker.update(detections, frame_idx)

        # Prepare arrays for per-frame predicted centers for MSE
        pred_centers = []

        # Draw tracks
        for t in tracks:
            # Only render if past warm-up (optional)
            if t.hits >= MIN_HITS or t.time_since_update == 0:
                box = t.predict()  # current KF state box
                cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.kf.state)
                pred_centers.append((float(cx), float(cy)))

                # Update trajectory buffers
                if t.id not in track_paths:
                    track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
                track_paths[t.id].append((int(cx), int(cy)))
                track_last_seen[t.id] = frame_idx
                track_cls[t.id] = t.cls

                color = CLASS_COLORS.get(t.cls, (200, 200, 200))
                draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_det, list(track_paths[t.id]), color=color)

                # Save trajectory row
                traj_rows.append(["video0_or_3", t.id, frame_idx, float(cx), float(cy)])

        # Remove stale trajectories if not seen for MISS_FRAMES_TO_DROP_PATH
        stale_ids = []
        for tid, last_seen in track_last_seen.items():
            if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH:
                stale_ids.append(tid)
        for tid in stale_ids:
            track_paths.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_cls.pop(tid, None)

        # MSE for this frame
        if COMPUTE_MSE:
            frame_mse = mse_per_frame(gt_centers, pred_centers)
            if frame_mse is not None:
                mse_values.append(frame_mse)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Show two windows
        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        cv2.imshow("Detections + ByteTrack + Hungarian + trajectories", vis_det)

        # key handling: 0ms wait when paused, 1ms when playing
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            break

        # ignore space explicitly
        if key == ord(' '):
            pass

        # pause / resume
        if key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False

        # step and jumps ONLY when paused
        if paused:
            if key == ord('o'):      # next frame
                _seek_to(frame_idx + 1)
            elif key == ord('i'):    # previous frame
                _seek_to(frame_idx - 1)
            elif key == ord('l'):    # +100
                _seek_to(frame_idx + 100)
            elif key == ord('k'):    # -100
                _seek_to(frame_idx - 100)

        # advance on play
        if not paused and not did_seek:
            frame_idx += 1
        did_seek = False

    cap.release()
    cv2.destroyAllWindows()

    # Save trajectories CSV
    if traj_rows:
        pd.DataFrame(traj_rows, columns=["video_id", "track_id", "frame", "x", "y"]).to_csv(TRAJ_CSV, index=False)
        print(f"Trajectories saved to {TRAJ_CSV}")

    # Print final MSE
    if COMPUTE_MSE and len(mse_values) > 0:
        print(f"Overall MSE: {np.mean(mse_values):.3f}")


if __name__ == "__main__":
    # Minimal guard for missing OpenCV GUI support
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()
