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
# Path to your trained YOLOv8 weights and the target video + its SDD annotation file.
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
# ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# To switch to video0, set:
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"

# --- Classes and colors (6 classes as in SDD) ---
# Mapping between numeric class ids and human-readable names (must match your training class order).
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Biker",
    2: "Skater",
    3: "Cart",
    4: "Car",
    5: "Bus",
}

# Map SDD label strings (from annotations.txt) to your training class indices.
LABEL_TO_ID = {
    "Pedestrian": 0,
    "Biker": 1,      # or "biker" in your file
    "Skater": 2,     # or "skater"
    "Cart": 3,
    "Car": 4,
    "Bus": 5,
}

# BGR colors for drawing boxes per class on the frames.
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
}

# --- Detection / NMS ---
# YOLO confidence threshold and NMS IoU used at inference time.
# Lowering DET_CONF_THRES yields more detections; raising removes low-confidence ones.
# DET_CONF_THRES = 0.30

# # checked 8 18
# DET_CONF_THRES = 0.65
# DET_IOU_NMS = 0.7
DET_CONF_THRES = 0.45
DET_IOU_NMS = 0.6

# --- ByteTrack-like association ---
# Parameters for the tracker association logic.
# BYTE_HIGH_THRES = 0.50   # high-score set
BYTE_HIGH_THRES = 0.68 # high-score set
# BYTE_LOW_THRES = 0.10    # low-score set
BYTE_LOW_THRES = 0.58 # low-score set
# IOU_GATE = 0.20        # minimum IoU to consider a match
IOU_GATE = 0.0        # minimum IoU to consider a match
# MAX_AGE = 30             # frames to keep "alive" without updates
MAX_AGE = 30             # frames to keep "alive" without updates
MIN_HITS = 3             # warm-up before rendering id (optional usage)
BORDER_MARGIN = 5  # pixels from edge to consider 'exit'
# #checked 8 18 
# BYTE_HIGH_THRES = 0.85 # high-score set
# BYTE_LOW_THRES = 0.70 # low-score set
# IOU_GATE = 0.1         # minimum IoU to consider a match
# MAX_AGE = 30             # frames to keep "alive" without updates
# MIN_HITS = 3             # warm-up before rendering id (optional usage)
# BORDER_MARGIN = 15  # pixels from edge to consider 'exit'

# --- Drawing / Trajectories ---
# Limits for trajectory memory and simple path cleanup.
MISS_FRAMES_TO_DROP_PATH = 5  # delete trajectory if not seen for 10 frames
TRAJ_MAX_LEN = 2000            # cap stored points per track to avoid memory growth
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2

# --- Metrics / Output ---
# If True, compute per-frame MSE between GT centers and tracked centers.
# TRAJ_CSV = "video3_trajectoriesNEW.csv"  # video_id,track_id,frame,x,y
TRAJ_CSV = "video0_trajectoriesNEW.csv"  # video_id,track_id,frame,x,y

COMPUTE_MSE = True

# --- Playback ---
# Controls for visual playback and whether to skip occluded/lost GT boxes.
SKIP_GT_OCCLUDED = True  # set True to skip occluded==1 or lost==1 GT boxes
PAUSE_ON_START = False    # press any key to start

# ---- Anti-jerk gates (small changes only) ----
# Heuristics to reject implausible size/speed jumps when updating tracks.
MAX_SPEED_DELTA = 2.5     # max change in speed (px/frame) between consecutive updates
SIZE_CHANGE_MAX = 2.0     # max area growth ratio allowed (unless high conf)
SIZE_CHANGE_MIN = 0.5     # max area shrink ratio allowed (unless high conf)
HIGH_CONF_RELAX = 0.55    # if conf >= this, allow larger changes


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

# Ultralytics YOLOv8 for detection.
try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics not found. Install: pip install ultralytics")
    raise

# Hungarian algorithm for optimal assignment.
try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    print("scipy not found. Install: pip install scipy")
    raise

# Kalman Filter (optional)
# If filterpy exists, we use a proper KF; else a dummy stub is used.
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

def is_near_border(bbox, frame_shape, border_ratio=0.04):
    """Check if a box is close to any border of the frame."""
    x1, y1, x2, y2 = bbox
    H, W = frame_shape[:2]
    border_x = W * border_ratio
    border_y = H * border_ratio
    return (
        x1 <= border_x or y1 <= border_y or
        x2 >= (W - border_x) or y2 >= (H - border_y)
    )

def xyxy_to_cxcywh(box):
    """Convert [x1,y1,x2,y2] to [cx,cy,w,h]."""
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=np.float32)


def cxcywh_to_xyxy(box):
    """Convert [cx,cy,w,h] to [x1,y1,x2,y2]."""
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
    Create a Kalman filter with 8D state [cx, cy, w, h, vx, vy, vw, vh].
    If filterpy is missing, fall back to a minimal stub that stores last state only.
    """
    if HAS_FILTERPY:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0

        # State transition (constant velocity in all four observed dims).
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i + 4] = dt

        # Measurement function: we directly observe [cx, cy, w, h].
        kf.H = np.zeros((4, 8), dtype=np.float32)
        kf.H[0, 0] = 1
        kf.H[1, 1] = 1
        kf.H[2, 2] = 1
        kf.H[3, 3] = 1

        # Covariances: moderate prior uncertainty and noise.
        kf.P *= 10.0
        kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        q = 1.0
        kf.Q = np.eye(8, dtype=np.float32) * q

        kf.x[:4, 0] = initial_cxcywh.reshape(4)
        return kf
    else:
        # Lightweight stub: keeps the last measurement as the "state".
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
                # Return shape-compatible vector like filterpy's state.
                return np.concatenate([self.state, np.zeros(4, dtype=np.float32)])[:, None]

        return DummyKF(initial_cxcywh.astype(np.float32))


class Track:
    """
    Single target track. Wraps KF state + metadata (id, class, conf) and a short trajectory.
    """
    _next_id = 1

    def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1

        self.cls = int(cls_id)           # last class
        self.conf = float(conf)          # last confidence
        self.hits = 1                    # number of successful updates
        self.age = 1                     # total frames since init
        self.time_since_update = 0       # frames since last matched update
        self.last_frame = frame_idx      # last frame index that updated this track

        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
        self.kf = make_kf(cxcywh)
        # Ensure state initialized even if Dummy
        if HAS_FILTERPY:
            self.kf.predict()

        # For drawing: store recent center points.
        self.history = deque(maxlen=TRAJ_MAX_LEN)
        cx, cy, w, h = cxcywh
        self.history.append((int(cx), int(cy)))

    def predict(self):
        """Advance KF one step and return predicted box in xyxy."""
        if HAS_FILTERPY:
            self.kf.predict()
        # else: state remains
        pred_xyxy = cxcywh_to_xyxy(self.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state)
        return pred_xyxy

    def update(self, bbox_xyxy, cls_id, conf, frame_idx):
        """Correct KF with new measurement and update bookkeeping."""
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
        """Increase the miss counters when not matched this frame."""
        self.time_since_update += 1
        self.age += 1


class ByteTrackLike:
    """
    Minimal ByteTrack-like manager:
      1) Associate high-confidence detections first.
      2) Then low-confidence detections to the remaining tracks.
      3) Start new tracks only from unmatched high-confidence detections.
      4) Age and remove tracks after MAX_AGE misses.
    """
    def __init__(self, iou_gate= IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []

    def _match(self, tracks, dets):
        """
        Hungarian on cost = 1 - IoU. Reject pairs below iou_gate.
        Returns:
          matches:      list of (track_idx, det_idx)
          unmatched_t:  list of unmatched track indices
          unmatched_d:  list of unmatched det indices
        """
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        # Predict each track to build association boxes.
        track_boxes = np.array([t.predict() for t in tracks], dtype=np.float32)
        det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
        iou = iou_xyxy(track_boxes, det_boxes)
        cost = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)
        matches, unmatched_t, unmatched_d = [], [], []

        # Tracks not selected by Hungarian are unmatched.
        for r, t in enumerate(tracks):
            if r not in row_ind:
                unmatched_t.append(r)
        # Detections not selected by Hungarian are unmatched.
        for c, _ in enumerate(dets):
            if c not in col_ind:
                unmatched_d.append(c)

        # Keep only pairs above IoU gate.
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_gate:
                matches.append((r, c))
            else:
                unmatched_t.append(r)
                unmatched_d.append(c)

        return matches, unmatched_t, unmatched_d

    def update(self, detections, frame_idx):
        """
        Update the tracker with a new frame of detections.
        detections: list of dicts with keys: bbox (xyxy), cls, conf
        """
        # Split detections by confidence as in ByteTrack.
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        # Stage 1: match high-score dets to existing tracks.
        matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
        for ti, di in matches:
            t = self.tracks[ti]
            d = high[di]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

        # Stage 2: the tracks that remained unmatched can try to match low-score dets.
        remain_tracks = [self.tracks[i] for i in unmatched_t]
        m2, unmatched_t2, unmatched_low = self._match(remain_tracks, low)
        for local_ti, di in m2:
            t = remain_tracks[local_ti]
            d = low[di]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

        # Start new tracks only from unmatched high-score detections (ByteTrack policy).
        new_high_idx = [i for i in unmatched_high]
        for di in new_high_idx:
            d = high[di]
            self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        # Tracks not matched in either stage are marked as missed.
        matched_global_t_idx = {ti for ti, _ in matches}
        # add those matched in stage 2 (convert local indices)
        matched_stage2_global = {unmatched_t[i] for i, _ in m2}
        for idx, t in enumerate(self.tracks):
            if idx not in matched_global_t_idx and idx not in matched_stage2_global:
                t.mark_missed()

        # Remove dead tracks (too many consecutive misses).
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self.tracks


class KalmanBoxTracker:
    """
    Placeholder for a SORT-like box tracker (not used in this file).
    Left as a stub to indicate prior experiments with bg-sub + SORT.
    """
    count = 0
    def __init__(self, bbox):
        ...
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 0
        self.no_losses = 0
        self.history = []

        # --- New fields for anti-jerk ---
        self.last_box = np.array([x1,y1,x2,y2], float)
        self.last_area = w*h
        self.last_center = np.array([cx, cy], float)
        self.last_speed = 0.0

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

        # --- update auxiliaries ---
        cur_center = np.array([cx, cy], float)
        self.last_speed = float(np.linalg.norm(cur_center - self.last_center))
        self.last_center = cur_center
        self.last_box = np.array([x1,y1,x2,y2], float)
        self.last_area = s


class Sort:
    """
    Placeholder for a SORT manager (not used here). Kept for reference.
    """
    ...
    def _valid_size(self, box, prev_area=None, conf=None):
        x1,y1,x2,y2 = box
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        area = w*h
        ar = max(w/h, h/w)
        if area < SIZE_MIN_AREA or ar > SIZE_MAX_ASPECT:
            return False
        if prev_area is not None and conf is not None:
            grow = area / max(1.0, prev_area)
            if conf < HIGH_CONF_RELAX and (grow > SIZE_CHANGE_MAX or grow < SIZE_CHANGE_MIN):
                return False
        return True

    def _valid_motion(self, tracker, new_box, conf):
        # allow if not enough history
        if tracker.hits < 1:
            return True
        x1,y1,x2,y2 = new_box
        cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
        cur_speed = float(np.linalg.norm(np.array([cx,cy]) - tracker.last_center))
        # if confidence is high, relax
        if conf >= HIGH_CONF_RELAX:
            return True
        return abs(cur_speed - tracker.last_speed) <= MAX_SPEED_DELTA


# =========================
# ======  GT PARSER  ======
# =========================

def parse_sdd_annotations(path):
    """
    Parse SDD-style annotations.
    Returns dict:
      frames[frame_idx] -> list of dicts:
         { 'id': int, 'bbox': [x1,y1,x2,y2], 'lost':0/1, 'occluded':0/1, 'label':str }
    Also returns inferred reference (W_ref, H_ref) used by the annotations file.
    We infer the canvas size from the max x2,y2 seen in the file.
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
    """Scale a bbox from annotation canvas to the current video resolution."""
    x1, y1, x2, y2 = bbox
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


# =========================
# ======  METRICS    ======
# =========================

def mse_per_frame(gt_centers, pred_centers):
    """
    Compute per-frame MSE after matching GT centers to predicted centers.
    Use Hungarian on squared euclidean distance. If no pairs, return None.
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
    """Draw a rectangle with class name, id, and optional confidence."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
    text = f"{cls_name} ID:{tid}"
    if conf is not None:
        text += f" {conf:.2f}"
    cv2.putText(frame, text, (x1, max(0, y1 - 5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def draw_traj(frame, pts, color=(255,255,255)):
    """Draw a polyline through the stored center points of a track."""
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], color, 2)


# =========================
# ======  MAIN LOOP  ======
# =========================

def main():
    # Open video and read basic properties.
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load and index GT per frame. Also infer annotation canvas size for rescaling.
    frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
    scale_x = W / float(W_ref if W_ref > 0 else W)
    scale_y = H / float(H_ref if H_ref > 0 else H)

    # Create detector.
    model = YOLO(YOLO_WEIGHTS)

    # Create tracker.
    tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

    # Trajectory buffers for drawing removal after misses
    # For tracker: store per track.id -> deque of points and last_seen frame
    track_paths = {}   # id -> deque[(x,y)]
    track_last_seen = {}  # id -> frame_idx
    track_cls = {}     # id -> class id

    # For GT: per object id, store its trajectory centers for the left window.
    gt_paths = defaultdict(deque)  # id -> deque[(x,y)]

    # For metrics
    traj_rows = []  # video_id,track_id,frame,x,y
    mse_values = []

    frame_idx = 0

    # --- Added: playback state + seek helpers ---
    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i: int) -> int:
        """Clamp frame index into valid range."""
        return max(0, min(total_frames - 1, i))

    def _reset_tracking_state():
        """Reset tracker and on-screen trajectories after seeking."""
        nonlocal tracker, track_paths, track_last_seen, track_cls
        tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
        track_paths = {}
        track_last_seen = {}
        track_cls = {}

    def _seek_to(target_idx: int):
        """Jump to a target frame and reset tracker state to avoid drift artifacts."""
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

        # Prepare left (GT) and right (Det+Track) views.
        vis_gt = frame.copy()
        vis_det = frame.copy()

        # -----------------------
        # Ground Truth window
        # -----------------------
        gt_boxes = []
        gt_centers = []
        if frame_idx in frames_gt:
            for ann in frames_gt[frame_idx]:
                # Optionally skip occluded/lost GT to avoid penalizing detector on invisible targets.
                if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1):
                    continue
                # Rescale GT box to the actual video resolution.
                bb = scale_bbox(ann["bbox"], scale_x, scale_y)
                cx, cy, _, _ = xyxy_to_cxcywh(np.array(bb, dtype=np.float32))
                gt_boxes.append(bb)
                gt_centers.append((cx, cy))
                # Accumulate GT trajectory for visualization.
                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN:
                    gt_paths[ann["id"]].popleft()
                lbl = ann["label"]
                gt_cls_id = LABEL_TO_ID.get(lbl, 0)
                color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))  # fallback color if unmapped
                draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

        # -----------------------
        # Detection + Tracking
        # -----------------------
        # YOLO inference for this frame (Ultralytics applies NMS internally).
        yolo_res = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
        detections = []
        if len(yolo_res) > 0:
            r = yolo_res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                # Build list of detection dicts for the tracker.
                for b, s, c in zip(xyxy, conf, cls):
                    detections.append({
                        "bbox": b.astype(np.float32),
                        "conf": float(s),
                        "cls": int(c),
                    })

        # Update tracker with current frame detections.
        tracks = tracker.update(detections, frame_idx)

        # Prepare arrays for per-frame predicted centers for MSE.
        pred_centers = []

        # Draw tracked boxes and their trajectories.
        for t in tracks:
            # Optionally show only confirmed tracks (past MIN_HITS) or those just updated.
            if t.hits >= MIN_HITS or t.time_since_update == 0:
                box = t.predict()  # current KF state box (after predict/update in this frame)
                if t.time_since_update > MAX_AGE or  is_near_border(box, frame.shape):
                    continue
                cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.kf.state)
                pred_centers.append((float(cx), float(cy)))

                # Update short path for drawing.
                if t.id not in track_paths:
                    track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
                track_paths[t.id].append((int(cx), int(cy)))
                track_last_seen[t.id] = frame_idx
                track_cls[t.id] = t.cls

                color = CLASS_COLORS.get(t.cls, (200, 200, 200))
                draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_det, list(track_paths[t.id]), color=color)

                # Save one trajectory row per visible track for potential export.
                traj_rows.append(["video3", t.id, frame_idx, float(cx), float(cy)])

        # Remove stale trajectories if track not updated for too long (UI cleanup only).
        stale_ids = []
        for tid, last_seen in track_last_seen.items():
            if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH:
                stale_ids.append(tid)
        for tid in stale_ids:
            track_paths.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_cls.pop(tid, None)

        # MSE for this frame (only if we have both GT and predictions).
        if COMPUTE_MSE:
            frame_mse = mse_per_frame(gt_centers, pred_centers)
            if frame_mse is not None:
                mse_values.append(frame_mse)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Show two windows side by side (user toggles below).
        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        cv2.imshow("Detections + ByteTrack + Hungarian + trajectories", vis_det)

        # key handling: 0ms wait when paused, 1ms when playing
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            # Quit
            break

        # ignore space explicitly
        if key == ord(' '):
            # Do nothing on space to avoid accidental toggles.
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

    # Print final MSE over all frames with valid matches.
    if COMPUTE_MSE and len(mse_values) > 0:
        print(f"Overall MSE: {np.mean(mse_values):.3f}")


if __name__ == "__main__":
    # Minimal guard for missing OpenCV GUI support
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()



                                    # import cv2

                                    # """
                                    # Human Motion Analysis: SDD video0/video3
                                    # Two synchronized windows:
                                    # 1) Ground-truth (rescaled) + GT trajectories
                                    # 2) YOLOv8 detections + ByteTrack-like tracking (Hungarian) + trajectories
                                    # """

                                    # # =========================
                                    # # ====== HYPERPARAMS ======
                                    # # =========================

                                    # # --- Paths ---
                                    # YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

                                    # VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
                                    # ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

                                    # # --- Classes and colors (6 classes as in SDD) ---
                                    # CLASS_NAMES = {
                                    #     0: "Pedestrian",
                                    #     1: "Biker",
                                    #     2: "Skater",
                                    #     3: "Cart",
                                    #     4: "Car",
                                    #     5: "Bus",
                                    # }
                                    # LABEL_TO_ID = {
                                    #     "Pedestrian": 0,
                                    #     "Biker": 1,
                                    #     "Skater": 2,
                                    #     "Cart": 3,
                                    #     "Car": 4,
                                    #     "Bus": 5,
                                    # }
                                    # CLASS_COLORS = {
                                    #     0: (0, 255, 0),
                                    #     1: (255, 0, 0),
                                    #     2: (0, 0, 255),
                                    #     3: (255, 255, 0),
                                    #     4: (255, 0, 255),
                                    #     5: (0, 255, 255),
                                    # }

                                    # # --- Detection / NMS ---
                                    # DET_CONF_THRES = 0.60
                                    # DET_IOU_NMS = 0.70

                                    # # --- ByteTrack-like association ---
                                    # BYTE_HIGH_THRES = 0.50
                                    # BYTE_LOW_THRES = 0.10
                                    # IOU_GATE = 0.20
                                    # MAX_AGE = 30
                                    # MIN_HITS = 3

                                    # # --- Drawing / Trajectories ---
                                    # MISS_FRAMES_TO_DROP_PATH = 5
                                    # TRAJ_MAX_LEN = 2000
                                    # FONT = cv2.FONT_HERSHEY_SIMPLEX
                                    # FONT_SCALE = 0.5
                                    # THICKNESS = 2

                                    # # --- Metrics / Output ---
                                    # COMPUTE_MSE = True

                                    # # --- Playback ---
                                    # SKIP_GT_OCCLUDED = True
                                    # PAUSE_ON_START = False

                                    # # ---- Anti-jerk gates (small changes only) ----
                                    # MAX_SPEED_DELTA = 2.5
                                    # SIZE_CHANGE_MAX = 2.0
                                    # SIZE_CHANGE_MIN = 0.5
                                    # HIGH_CONF_RELAX = 0.55

                                    # # ---- New-track birth rule near existing tracks ----
                                    # BIRTH_MIN_FRAMES = 10
                                    # BIRTH_AVG_CONF = 0.80
                                    # BIRTH_POS_TOL = 20
                                    # NEW_NEAR_IOU = 0.30
                                    # NEW_NEAR_DIST = 25
                                    # CANDIDATE_MAX_AGE = 15

                                    # # ---- Border & revival rules ----
                                    # BORDER_MARGIN_PX = 25           # near-border margin
                                    # LOST_MAX_AGE = 120              # keep lost tracks this long for revival
                                    # REVIVE_DIST = 35                # max distance to predicted lost position
                                    # REVIVE_DIR_COS = 0.0            # require non-opposite direction (>=0)
                                    # REVIVE_IOU = 0.10               # loose IoU for revival

                                    # # =========================
                                    # # ======  IMPORTS   =======
                                    # # =========================
                                    # import os, sys, math, time
                                    # import numpy as np
                                    # import pandas as pd
                                    # from collections import deque, defaultdict
                                    # from ultralytics import YOLO
                                    # from scipy.optimize import linear_sum_assignment

                                    # # Kalman Filter (optional)
                                    # HAS_FILTERPY = True
                                    # try:
                                    #     from filterpy.kalman import KalmanFilter
                                    # except Exception:
                                    #     HAS_FILTERPY = False

                                    # # =========================
                                    # # ======  HELPERS    ======
                                    # # =========================

                                    # def iou_xyxy(a, b):
                                    #     N = a.shape[0]; M = b.shape[0]
                                    #     if N == 0 or M == 0:
                                    #         return np.zeros((N, M), dtype=np.float32)
                                    #     x11, y11, x12, y12 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
                                    #     x21, y21, x22, y22 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]
                                    #     inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
                                    #     inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
                                    #     inter = inter_w * inter_h
                                    #     area_a = np.maximum(0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
                                    #     area_b = np.maximum(0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
                                    #     union = area_a + area_b - inter
                                    #     return np.where(union > 0, inter / union, 0.0).astype(np.float32)

                                    # def xyxy_to_cxcywh(box):
                                    #     x1, y1, x2, y2 = box
                                    #     w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                                    #     return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=np.float32)

                                    # def cxcywh_to_xyxy(box):
                                    #     cx, cy, w, h = box
                                    #     return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

                                    # def center_of_box(b):
                                    #     x1,y1,x2,y2 = b
                                    #     return (0.5*(x1+x2), 0.5*(y1+y2))

                                    # def near_border(box, W, H, margin=BORDER_MARGIN_PX):
                                    #     x1,y1,x2,y2 = box
                                    #     cx, cy = center_of_box(box)
                                    #     return (cx < margin) or (cx > W - margin) or (cy < margin) or (cy > H - margin)

                                    # def nms_per_class(dets, iou_thr):
                                    #     # dets: list of dict {bbox, conf, cls}
                                    #     out = []
                                    #     for c in set(d["cls"] for d in dets):
                                    #         group = [d for d in dets if d["cls"] == c]
                                    #         if not group:
                                    #             continue
                                    #         boxes = np.array([g["bbox"] for g in group], dtype=np.float32)
                                    #         scores = np.array([g["conf"] for g in group], dtype=np.float32)
                                    #         idxs = scores.argsort()[::-1]
                                    #         keep = []
                                    #         while len(idxs) > 0:
                                    #             i = idxs[0]
                                    #             keep.append(i)
                                    #             if len(idxs) == 1:
                                    #                 break
                                    #             rest = idxs[1:]
                                    #             ious = iou_xyxy(boxes[i:i+1], boxes[rest])[0]
                                    #             idxs = rest[ious < iou_thr]
                                    #         out.extend([group[i] for i in keep])
                                    #     return out

                                    # # =========================
                                    # # ======  TRACKER    ======
                                    # # =========================

                                    # def make_kf(initial_cxcywh):
                                    #     if HAS_FILTERPY:
                                    #         kf = KalmanFilter(dim_x=8, dim_z=4); dt = 1.0
                                    #         kf.F = np.eye(8, dtype=np.float32)
                                    #         for i in range(4): kf.F[i, i + 4] = dt
                                    #         kf.H = np.zeros((4, 8), dtype=np.float32); kf.H[0,0]=kf.H[1,1]=kf.H[2,2]=kf.H[3,3]=1
                                    #         kf.P *= 10.0
                                    #         kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
                                    #         kf.Q = np.eye(8, dtype=np.float32) * 1.0
                                    #         kf.x[:4, 0] = initial_cxcywh.reshape(4)
                                    #         kf.predict()
                                    #         return kf
                                    #     else:
                                    #         class DummyKF:
                                    #             def __init__(self, init_state): self.state = init_state.copy()
                                    #             def predict(self): return self.state
                                    #             def update(self, z): self.state = z.copy()
                                    #             @property
                                    #             def x(self): return np.concatenate([self.state, np.zeros(4, dtype=np.float32)])[:, None]
                                    #         return DummyKF(initial_cxcywh.astype(np.float32))

                                    # class Track:
                                    #     _next_id = 1
                                    #     def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
                                    #         self.id = Track._next_id; Track._next_id += 1
                                    #         self.cls = int(cls_id); self.conf = float(conf)
                                    #         self.hits = 1; self.age = 1; self.time_since_update = 0; self.last_frame = frame_idx
                                    #         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
                                    #         self.kf = make_kf(cxcywh)
                                    #         self.history = deque(maxlen=TRAJ_MAX_LEN)
                                    #         cx, cy, w, h = (self.kf.x[:4, 0] if HAS_FILTERPY else self.kf.state)
                                    #         self.history.append((int(cx), int(cy)))
                                    #         # anti-jerk state
                                    #         x1,y1,x2,y2 = bbox_xyxy
                                    #         self.last_box = np.array([x1,y1,x2,y2], dtype=np.float32)
                                    #         self.last_area = max(1.0, (x2-x1)*(y2-y1))
                                    #         self.last_center = np.array([cx, cy], dtype=np.float32)
                                    #         self.last_speed = 0.0
                                    #         # velocity vector (for revival)
                                    #         self.vel = np.array([0.0, 0.0], dtype=np.float32)

                                    #     def predict(self):
                                    #         if HAS_FILTERPY: self.kf.predict()
                                    #         return (cxcywh_to_xyxy(self.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state))

                                    #     def update(self, bbox_xyxy, cls_id, conf, frame_idx):
                                    #         prev_center = self.last_center.copy()
                                    #         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
                                    #         if HAS_FILTERPY: self.kf.update(cxcywh)
                                    #         else: self.kf.update(cxcywh)
                                    #         self.cls = int(cls_id); self.conf = float(conf)
                                    #         self.hits += 1; self.time_since_update = 0; self.last_frame = frame_idx
                                    #         cx, cy, w, h = (self.kf.x[:4, 0] if HAS_FILTERPY else self.kf.state)
                                    #         self.history.append((int(cx), int(cy)))
                                    #         x1,y1,x2,y2 = bbox_xyxy
                                    #         cur_center = np.array([cx, cy], dtype=np.float32)
                                    #         self.last_speed = float(np.linalg.norm(cur_center - self.last_center))
                                    #         self.vel = cur_center - prev_center
                                    #         self.last_center = cur_center
                                    #         self.last_box = np.array([x1,y1,x2,y2], dtype=np.float32)
                                    #         self.last_area = max(1.0, (x2-x1)*(y2-y1))

                                    #     def mark_missed(self):
                                    #         self.time_since_update += 1; self.age += 1

                                    # class ByteTrackLike:
                                    #     def __init__(self, iou_gate=0.2, max_age=30, min_hits=3):
                                    #         self.iou_gate = iou_gate; self.max_age = max_age; self.min_hits = min_hits
                                    #         self.tracks = []
                                    #         # birth candidates: key -> dict(count,sum_conf,last_frame)
                                    #         self.birth_pool = {}
                                    #         # lost pool: id -> dict(track, age)
                                    #         self.lost = {}

                                    #     def _match(self, tracks, dets):
                                    #         if len(tracks) == 0 or len(dets) == 0:
                                    #             return [], list(range(len(tracks))), list(range(len(dets)))
                                    #         track_boxes = np.array([t.predict() for t in tracks], dtype=np.float32)
                                    #         det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
                                    #         iou = iou_xyxy(track_boxes, det_boxes); cost = 1.0 - iou
                                    #         row_ind, col_ind = linear_sum_assignment(cost)
                                    #         matches, unmatched_t, unmatched_d = [], [], []
                                    #         tr_set = set(row_ind.tolist()); dt_set = set(col_ind.tolist())
                                    #         for r in range(len(tracks)):
                                    #             if r not in tr_set: unmatched_t.append(r)
                                    #         for c in range(len(dets)):
                                    #             if c not in dt_set: unmatched_d.append(c)
                                    #         for r, c in zip(row_ind, col_ind):
                                    #             if iou[r, c] >= self.iou_gate: matches.append((r, c))
                                    #             else: unmatched_t.append(r); unmatched_d.append(c)
                                    #         return matches, unmatched_t, unmatched_d

                                    #     def _gate_ok(self, t: Track, det_bbox, det_conf):
                                    #         if t.hits < 2 or det_conf >= HIGH_CONF_RELAX:
                                    #             return True
                                    #         x1,y1,x2,y2 = det_bbox
                                    #         area = max(1.0, (x2-x1)*(y2-y1))
                                    #         grow = area / max(1.0, t.last_area)
                                    #         if grow > SIZE_CHANGE_MAX or grow < SIZE_CHANGE_MIN:
                                    #             return False
                                    #         cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
                                    #         cur_speed = float(np.linalg.norm(np.array([cx,cy], dtype=np.float32) - t.last_center))
                                    #         if abs(cur_speed - t.last_speed) > MAX_SPEED_DELTA:
                                    #             return False
                                    #         return True

                                    #     def _near_existing(self, det_bbox, W, H):
                                    #         if not self.tracks:
                                    #             return False
                                    #         d_cx, d_cy = center_of_box(det_bbox)
                                    #         for t in self.tracks:
                                    #             tb = t.predict()
                                    #             if iou_xyxy(np.array([tb]), np.array([det_bbox]))[0,0] >= NEW_NEAR_IOU:
                                    #                 return True
                                    #             tcx, tcy = center_of_box(tb)
                                    #             if np.hypot(d_cx - tcx, d_cy - tcy) <= NEW_NEAR_DIST:
                                    #                 return True
                                    #         return False

                                    #     def _birth_key(self, det_bbox):
                                    #         cx, cy = center_of_box(det_bbox)
                                    #         return (int(round(cx / BIRTH_POS_TOL)), int(round(cy / BIRTH_POS_TOL)))

                                    #     def _update_birth_pool(self, key, conf, frame_idx):
                                    #         slot = self.birth_pool.get(key, {"count":0, "sum_conf":0.0, "last_frame":frame_idx-1})
                                    #         if slot["last_frame"] != frame_idx-1:
                                    #             slot = {"count":0, "sum_conf":0.0, "last_frame":frame_idx-1}
                                    #         slot["count"] += 1
                                    #         slot["sum_conf"] += conf
                                    #         slot["last_frame"] = frame_idx
                                    #         self.birth_pool[key] = slot
                                    #         return slot

                                    #     def _purge_birth_pool(self, frame_idx):
                                    #         kill = [k for k,v in self.birth_pool.items() if frame_idx - v["last_frame"] > CANDIDATE_MAX_AGE]
                                    #         for k in kill: self.birth_pool.pop(k, None)

                                    #     def _predict_lost_center(self, t: Track, k_steps=1):
                                    #         return t.last_center + k_steps * t.vel

                                    #     def _try_revive(self, unmatched_dets, frame_idx):
                                    #         revived = set()
                                    #         if not self.lost or not unmatched_dets:
                                    #             return revived
                                    #         lost_ids = list(self.lost.keys())
                                    #         for di in list(unmatched_dets):
                                    #             db = unmatched_dets[di]["bbox"]
                                    #             dcx, dcy = center_of_box(db)
                                    #             best_id, best_score = None, 1e9
                                    #             for lid in lost_ids:
                                    #                 item = self.lost[lid]
                                    #                 t = item["track"]
                                    #                 age = item["age"]
                                    #                 if age > LOST_MAX_AGE:
                                    #                     continue
                                    #                 pred = self._predict_lost_center(t, k_steps=min(5, age+1))
                                    #                 dist = np.hypot(dcx - pred[0], dcy - pred[1])
                                    #                 if dist > REVIVE_DIST:
                                    #                     continue
                                    #                 # direction check
                                    #                 v = t.vel
                                    #                 to_det = np.array([dcx, dcy], dtype=np.float32) - t.last_center
                                    #                 nv = np.linalg.norm(v) + 1e-6
                                    #                 nd = np.linalg.norm(to_det) + 1e-6
                                    #                 cos = float(np.dot(v, to_det) / (nv*nd))
                                    #                 if cos < REVIVE_DIR_COS:
                                    #                     continue
                                    #                 # small distance wins
                                    #                 if dist < best_score:
                                    #                     best_score = dist
                                    #                     best_id = lid
                                    #             if best_id is not None:
                                    #                 t = self.lost[best_id]["track"]
                                    #                 d = unmatched_dets[di]
                                    #                 # revive: reuse track id and KF
                                    #                 t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                                    #                 self.tracks.append(t)
                                    #                 self.lost.pop(best_id, None)
                                    #                 revived.add(di)
                                    #         return revived

                                    #     def update(self, detections, frame_idx, frame_size):
                                    #         W, H = frame_size
                                    #         self._purge_birth_pool(frame_idx)

                                    #         # split
                                    #         high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
                                    #         low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

                                    #         # stage 1
                                    #         matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
                                    #         gated_tracks_s1, gated_dets_s1 = set(), set()
                                    #         for ti, di in matches:
                                    #             t = self.tracks[ti]; d = high[di]
                                    #             if self._gate_ok(t, d["bbox"], d["conf"]):
                                    #                 t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                                    #             else:
                                    #                 gated_tracks_s1.add(ti); gated_dets_s1.add(di)

                                    #         remain_indices = unmatched_t + list(gated_tracks_s1)
                                    #         remain_tracks = [self.tracks[i] for i in remain_indices]
                                    #         unmatched_high = unmatched_high + list(gated_dets_s1)

                                    #         # stage 2 with low-score on remaining tracks
                                    #         m2, unmatched_t2_local, unmatched_low = self._match(remain_tracks, low)
                                    #         matched_stage2_global = set()
                                    #         for local_ti, di in m2:
                                    #             t_global = remain_indices[local_ti]
                                    #             t = self.tracks[t_global]; d = low[di]
                                    #             if self._gate_ok(t, d["bbox"], d["conf"]):
                                    #                 t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                                    #                 matched_stage2_global.add(t_global)
                                    #             else:
                                    #                 unmatched_low.append(di)

                                    #         # convert unmatched_high to dict index->det for revival and birth logic
                                    #         uh_dict = {i: high[i] for i in unmatched_high}
                                    #         # first try revival against lost pool
                                    #         revived_idx = self._try_revive(uh_dict, frame_idx)
                                    #         for di in sorted(list(revived_idx), reverse=True):
                                    #             uh_dict.pop(di, None)

                                    #         # controlled new-track creation
                                    #         for di, d in uh_dict.items():
                                    #             allow_birth = False
                                    #             if frame_idx == 0:
                                    #                 allow_birth = True
                                    #             elif near_border(d["bbox"], W, H, BORDER_MARGIN_PX):
                                    #                 allow_birth = True
                                    #             else:
                                    #                 if self._near_existing(d["bbox"], W, H):
                                    #                     key = self._birth_key(d["bbox"])
                                    #                     slot = self._update_birth_pool(key, d["conf"], frame_idx)
                                    #                     avg_conf = slot["sum_conf"] / max(1, slot["count"])
                                    #                     if slot["count"] >= BIRTH_MIN_FRAMES and avg_conf >= BIRTH_AVG_CONF:
                                    #                         allow_birth = True
                                    #                 else:
                                    #                     # far from existing, still require persistence (strict birth rule)
                                    #                     key = self._birth_key(d["bbox"])
                                    #                     slot = self._update_birth_pool(key, d["conf"], frame_idx)
                                    #                     avg_conf = slot["sum_conf"] / max(1, slot["count"])
                                    #                     if slot["count"] >= BIRTH_MIN_FRAMES and avg_conf >= BIRTH_AVG_CONF:
                                    #                         allow_birth = True
                                    #             if allow_birth:
                                    #                 self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

                                    #         # mark missed and move non-border endings to lost pool
                                    #         matched_global_t_idx = {ti for ti, _ in matches}
                                    #         for idx, t in enumerate(self.tracks):
                                    #             if idx not in matched_global_t_idx and idx not in matched_stage2_global:
                                    #                 t.mark_missed()

                                    #         # collect survivors and move expired to lost or drop if at border
                                    #         survivors = []
                                    #         for t in self.tracks:
                                    #             if t.time_since_update <= self.max_age:
                                    #                 survivors.append(t)
                                    #             else:
                                    #                 if near_border(t.last_box, W, H, BORDER_MARGIN_PX):
                                    #                     # ended at border: drop
                                    #                     pass
                                    #                 else:
                                    #                     # move to lost pool for possible revival
                                    #                     self.lost[t.id] = {"track": t, "age": 0}
                                    #         self.tracks = survivors

                                    #         # age lost pool
                                    #         for lid in list(self.lost.keys()):
                                    #             self.lost[lid]["age"] += 1
                                    #             if self.lost[lid]["age"] > LOST_MAX_AGE:
                                    #                 self.lost.pop(lid, None)

                                    #         return self.tracks

                                    # # =========================
                                    # # ======  GT PARSER  ======
                                    # # =========================

                                    # def parse_sdd_annotations(path):
                                    #     frames = defaultdict(list); max_x = 0; max_y = 0
                                    #     with open(path, "r") as f:
                                    #         for line in f:
                                    #             line = line.strip()
                                    #             if not line: continue
                                    #             parts = line.split()
                                    #             if len(parts) < 10: continue
                                    #             tid = int(parts[0]); x1 = float(parts[1]); y1 = float(parts[2])
                                    #             x2 = float(parts[3]); y2 = float(parts[4]); frame = int(parts[5])
                                    #             lost = int(parts[6]); occl = int(parts[7])
                                    #             label_raw = " ".join(parts[9:])
                                    #             label = label_raw.strip().strip('"')
                                    #             frames[frame].append({"id": tid,"bbox":[x1,y1,x2,y2],"lost":lost,"occluded":occl,"label":label})
                                    #             max_x = max(max_x, x2); max_y = max(max_y, y2)
                                    #     W_ref = int(math.ceil(max_x)); H_ref = int(math.ceil(max_y))
                                    #     return frames, (W_ref, H_ref)

                                    # def scale_bbox(bbox, scale_x, scale_y):
                                    #     x1,y1,x2,y2 = bbox
                                    #     return [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]

                                    # # =========================
                                    # # ======  METRICS    ======
                                    # # =========================

                                    # def mse_per_frame(gt_centers, pred_centers):
                                    #     if len(gt_centers) == 0 or len(pred_centers) == 0:
                                    #         return None
                                    #     G = np.array(gt_centers, dtype=np.float32)
                                    #     P = np.array(pred_centers, dtype=np.float32)
                                    #     diff = G[:, None, :] - P[None, :, :]
                                    #     cost = np.sum(diff * diff, axis=2)
                                    #     r, c = linear_sum_assignment(cost)
                                    #     if len(r) == 0:
                                    #         return None
                                    #     return float(np.mean(cost[r, c]))

                                    # # =========================
                                    # # ======  DRAWING    ======
                                    # # =========================

                                    # def draw_box_id(frame, bbox, cls, tid, conf=None, color=(0,255,0), label_map=None):
                                    #     x1,y1,x2,y2 = map(int, bbox)
                                    #     cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                                    #     cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
                                    #     text = f"{cls_name} ID:{tid}"
                                    #     if conf is not None: text += f" {conf:.2f}"
                                    #     cv2.putText(frame, text, (x1, max(0, y1-5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

                                    # def draw_traj(frame, pts, color=(255,255,255)):
                                    #     if len(pts) < 2: return
                                    #     for i in range(1, len(pts)):
                                    #         cv2.line(frame, pts[i-1], pts[i], color, 2)

                                    # # =========================
                                    # # ======  MAIN LOOP  ======
                                    # # =========================

                                    # def main():
                                    #     cap = cv2.VideoCapture(VIDEO_PATH)
                                    #     if not cap.isOpened():
                                    #         print(f"Cannot open video: {VIDEO_PATH}"); return
                                    #     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    #     FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                                    #     # Load GT
                                    #     frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
                                    #     scale_x = W / float(W_ref if W_ref > 0 else W)
                                    #     scale_y = H / float(H_ref if H_ref > 0 else H)

                                    #     # YOLO
                                    #     model = YOLO(YOLO_WEIGHTS)

                                    #     # Tracker
                                    #     tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

                                    #     # Trajectory buffers
                                    #     track_paths = {}
                                    #     track_last_seen = {}
                                    #     track_cls = {}
                                    #     gt_paths = defaultdict(deque)

                                    #     traj_rows = []
                                    #     mse_values = []
                                    #     frame_idx = 0

                                    #     paused = PAUSE_ON_START
                                    #     did_seek = False

                                    #     def _clamp(i: int) -> int: return max(0, min(total_frames - 1, i))
                                    #     def _reset_tracking_state():
                                    #         nonlocal tracker, track_paths, track_last_seen, track_cls
                                    #         tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
                                    #         track_paths = {}; track_last_seen = {}; track_cls = {}
                                    #     def _seek_to(target_idx: int):
                                    #         nonlocal frame_idx, did_seek
                                    #         frame_idx = _clamp(target_idx); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                    #         _reset_tracking_state(); did_seek = True

                                    #     if PAUSE_ON_START: print("Paused. Press 'r' to resume, 'p' to pause again.")

                                    #     while True:
                                    #         ret, frame = cap.read()
                                    #         if not ret: break
                                    #         vis_gt = frame.copy(); vis_det = frame.copy()

                                    #         # ---- GT ----
                                    #         gt_centers = []
                                    #         if frame_idx in frames_gt:
                                    #             for ann in frames_gt[frame_idx]:
                                    #                 if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1): continue
                                    #                 bb = scale_bbox(ann["bbox"], scale_x, scale_y)
                                    #                 cx, cy, _, _ = xyxy_to_cxcywh(np.array(bb, dtype=np.float32))
                                    #                 gt_centers.append((cx, cy))
                                    #                 gt_paths[ann["id"]].append((int(cx), int(cy)))
                                    #                 if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN: gt_paths[ann["id"]].popleft()
                                    #                 lbl = ann["label"]; gt_cls_id = LABEL_TO_ID.get(lbl, 0)
                                    #                 color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))
                                    #                 draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                                    #                 draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

                                    #         # ---- Detection (Ultralytics already NMS; add per-class NMS safeguard) ----
                                    #         yolo_res = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
                                    #         detections = []
                                    #         if len(yolo_res) > 0:
                                    #             r = yolo_res[0]
                                    #             if r.boxes is not None and len(r.boxes) > 0:
                                    #                 xyxy = r.boxes.xyxy.cpu().numpy()
                                    #                 conf = r.boxes.conf.cpu().numpy()
                                    #                 cls  = r.boxes.cls.cpu().numpy().astype(int)
                                    #                 for b, s, c in zip(xyxy, conf, cls):
                                    #                     detections.append({"bbox": b.astype(np.float32), "conf": float(s), "cls": int(c)})
                                    #         # extra NMS to kill doubles next to the same target
                                    #         detections = nms_per_class(detections, DET_IOU_NMS)

                                    #         # ---- Tracking ----
                                    #         tracks = tracker.update(detections, frame_idx, (W, H))

                                    #         # ---- Draw tracks + per-frame MSE ----
                                    #         pred_centers = []
                                    #         for t in tracks:
                                    #             if t.hits >= MIN_HITS or t.time_since_update == 0:
                                    #                 box = t.predict()
                                    #                 cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.kf.state)
                                    #                 pred_centers.append((float(cx), float(cy)))
                                    #                 if t.id not in track_paths: track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
                                    #                 track_paths[t.id].append((int(cx), int(cy)))
                                    #                 track_last_seen[t.id] = frame_idx; track_cls[t.id] = t.cls
                                    #                 color = CLASS_COLORS.get(t.cls, (200, 200, 200))
                                    #                 draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
                                    #                 draw_traj(vis_det, list(track_paths[t.id]), color=color)
                                    #                 traj_rows.append(["video0_or_3", t.id, frame_idx, float(cx), float(cy)])

                                    #         stale_ids = [tid for tid, last_seen in track_last_seen.items()
                                    #                     if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH]
                                    #         for tid in stale_ids:
                                    #             track_paths.pop(tid, None); track_last_seen.pop(tid, None); track_cls.pop(tid, None)

                                    #         if COMPUTE_MSE:
                                    #             frame_mse = mse_per_frame(gt_centers, pred_centers)
                                    #             if frame_mse is not None:
                                    #                 mse_values.append(frame_mse)
                                    #                 cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                                    #                 cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                                    #         cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
                                    #         cv2.imshow("Detections + ByteTrack + Hungarian + trajectories", vis_det)

                                    #         key = cv2.waitKey(0 if paused else 1) & 0xFF
                                    #         if key == ord('q'): break
                                    #         if key == ord(' '): pass
                                    #         if key == ord('p'): paused = True
                                    #         elif key == ord('r'): paused = False
                                    #         if paused:
                                    #             if key == ord('o'): _seek_to(frame_idx + 1)
                                    #             elif key == ord('i'): _seek_to(frame_idx - 1)
                                    #             elif key == ord('l'): _seek_to(frame_idx + 100)
                                    #             elif key == ord('k'): _seek_to(frame_idx - 100)
                                    #         if not paused and not did_seek: frame_idx += 1
                                    #         did_seek = False

                                    #     cap.release(); cv2.destroyAllWindows()
                                    #     if COMPUTE_MSE and len(mse_values) > 0:
                                    #         print(f"Overall MSE: {np.mean(mse_values):.3f}")

                                    # if __name__ == "__main__":
                                    #     if not hasattr(cv2, "imshow"):
                                    #         print("OpenCV built without HighGUI. Install opencv-python."); sys.exit(1)
                                    #     main()



# s

# # TRACK and LOST  : i want it to be like if i have a frame going somewhere. don't make a new frame just next to him out of no where. unless the confidence average for 3-5 consecutive frames are above 0.8

# # # import cv2

# # # """
# # # Human Motion Analysis: SDD video3
# # # Left: GT (rescaled) + trajectories
# # # Right: YOLOv8 + ByteTrack-like tracking + trajectories
# # # """

# # # # =========================
# # # # ====== HYPERPARAMS ======
# # # # =========================

# # # # --- Paths ---
# # # YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"
# # # VIDEO_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
# # # ANNOT_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"
# # # TRAJ_CSV     = r"C:\Users\morte\ComputerVisionProject\outputs\video3_traj.csv"

# # # # --- Classes and colors (6 classes as in SDD) ---
# # # CLASS_NAMES = {0:"Pedestrian",1:"Biker",2:"Skater",3:"Cart",4:"Car",5:"Bus"}
# # # LABEL_TO_ID = {"Pedestrian":0,"Biker":1,"Skater":2,"Cart":3,"Car":4,"Bus":5}
# # # CLASS_COLORS={0:(0,255,0),1:(255,0,0),2:(0,0,255),3:(255,255,0),4:(255,0,255),5:(0,255,255)}

# # # # --- Detection / NMS ---
# # # DET_CONF_THRES = 0.45      # lower to reduce misses
# # # DET_IOU_NMS    = 0.60

# # # # --- ByteTrack-like association ---
# # # BYTE_HIGH_THRES = 0.50
# # # BYTE_LOW_THRES  = 0.05
# # # IOU_GATE        = 0.20
# # # MAX_AGE         = 30
# # # MIN_HITS        = 3

# # # # --- Drawing / Trajectories ---
# # # MISS_FRAMES_TO_DROP_PATH = 5
# # # TRAJ_MAX_LEN = 2000
# # # FONT = cv2.FONT_HERSHEY_SIMPLEX
# # # FONT_SCALE = 0.5
# # # THICKNESS = 2

# # # # --- Metrics / Output ---
# # # COMPUTE_MSE = True

# # # # --- Playback ---
# # # SKIP_GT_OCCLUDED = True
# # # PAUSE_ON_START   = False

# # # # ---- Anti-jerk gates ----
# # # MAX_SPEED_DELTA = 2.5
# # # SIZE_CHANGE_MAX = 2.0
# # # SIZE_CHANGE_MIN = 0.5
# # # HIGH_CONF_RELAX = 0.55

# # # # ---- Birth rule (only 3 cases allowed) ----
# # # BIRTH_MIN_FRAMES = 10
# # # BIRTH_AVG_CONF   = 0.80
# # # BIRTH_POS_TOL    = 20
# # # NEW_NEAR_IOU     = 0.30
# # # NEW_NEAR_DIST    = 25
# # # CANDIDATE_MAX_AGE= 15

# # # # ---- Border & lost-found rules ----
# # # BORDER_MARGIN_PX = 25
# # # LOST_TO_POOL     = 3      # after this many missed frames, move to lost pool
# # # LOST_MAX_AGE     = 120
# # # REVIVE_DIST      = 35
# # # REVIVE_DIR_COS   = 0.0
# # # REVIVE_IOU       = 0.10

# # # # =========================
# # # # ======  IMPORTS   =======
# # # # =========================
# # # import os, sys, math, time
# # # import numpy as np
# # # import pandas as pd
# # # from collections import deque, defaultdict
# # # from ultralytics import YOLO
# # # from scipy.optimize import linear_sum_assignment

# # # HAS_FILTERPY = True
# # # try:
# # #     from filterpy.kalman import KalmanFilter
# # # except Exception:
# # #     HAS_FILTERPY = False

# # # # =========================
# # # # ======  HELPERS    ======
# # # # =========================

# # # def iou_xyxy(a, b):
# # #     N = a.shape[0]; M = b.shape[0]
# # #     if N == 0 or M == 0:
# # #         return np.zeros((N, M), dtype=np.float32)
# # #     x11,y11,x12,y12 = a[:,0][:,None],a[:,1][:,None],a[:,2][:,None],a[:,3][:,None]
# # #     x21,y21,x22,y22 = b[:,0][None,:],b[:,1][None,:],b[:,2][None,:],b[:,3][None,:]
# # #     inter_w = np.maximum(0, np.minimum(x12,x22)-np.maximum(x11,x21))
# # #     inter_h = np.maximum(0, np.minimum(y12,y22)-np.maximum(y11,y21))
# # #     inter = inter_w*inter_h
# # #     area_a = np.maximum(0,(a[:,2]-a[:,0])*(a[:,3]-a[:,1]))[:,None]
# # #     area_b = np.maximum(0,(b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]
# # #     union = area_a + area_b - inter
# # #     return np.where(union>0, inter/union, 0.0).astype(np.float32)

# # # def xyxy_to_cxcywh(box):
# # #     x1,y1,x2,y2 = box
# # #     w = max(0.0,x2-x1); h = max(0.0,y2-y1)
# # #     return np.array([x1+w/2.0, y1+h/2.0, w, h], dtype=np.float32)

# # # def cxcywh_to_xyxy(box):
# # #     cx,cy,w,h = box
# # #     return np.array([cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0], dtype=np.float32)

# # # def center_of_box(b):
# # #     x1,y1,x2,y2 = b
# # #     return (0.5*(x1+x2), 0.5*(y1+y2))

# # # def near_border(box, W, H, margin=BORDER_MARGIN_PX):
# # #     cx, cy = center_of_box(box)
# # #     return (cx < margin) or (cx > W - margin) or (cy < margin) or (cy > H - margin)

# # # def nms_per_class(dets, iou_thr):
# # #     out = []
# # #     classes = set(d["cls"] for d in dets)
# # #     for c in classes:
# # #         group = [d for d in dets if d["cls"] == c]
# # #         if not group: continue
# # #         boxes = np.array([g["bbox"] for g in group], dtype=np.float32)
# # #         scores= np.array([g["conf"] for g in group], dtype=np.float32)
# # #         idxs = scores.argsort()[::-1]
# # #         keep = []
# # #         while len(idxs) > 0:
# # #             i = idxs[0]; keep.append(i)
# # #             if len(idxs)==1: break
# # #             rest = idxs[1:]
# # #             ious = iou_xyxy(boxes[i:i+1], boxes[rest])[0]
# # #             idxs = rest[ious < iou_thr]
# # #         out.extend([group[i] for i in keep])
# # #     return out

# # # # =========================
# # # # ======  TRACKER    ======
# # # # =========================

# # # def make_kf(initial_cxcywh):
# # #     if HAS_FILTERPY:
# # #         kf = KalmanFilter(dim_x=8, dim_z=4); dt = 1.0
# # #         kf.F = np.eye(8, dtype=np.float32)
# # #         for i in range(4): kf.F[i, i+4] = dt
# # #         kf.H = np.zeros((4,8), dtype=np.float32); kf.H[0,0]=kf.H[1,1]=kf.H[2,2]=kf.H[3,3]=1
# # #         kf.P *= 10.0
# # #         kf.R = np.diag([1.0,1.0,10.0,10.0]).astype(np.float32)
# # #         kf.Q = np.eye(8, dtype=np.float32)*1.0
# # #         kf.x[:4,0] = initial_cxcywh.reshape(4); kf.predict()
# # #         return kf
# # #     else:
# # #         class DummyKF:
# # #             def __init__(self, init_state): self.state = init_state.copy()
# # #             def predict(self): return self.state
# # #             def update(self, z): self.state = z.copy()
# # #             @property
# # #             def x(self): return np.concatenate([self.state,np.zeros(4,dtype=np.float32)])[:,None]
# # #         return DummyKF(initial_cxcywh.astype(np.float32))

# # # class Track:
# # #     _next_id = 1
# # #     def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
# # #         self.id = Track._next_id; Track._next_id += 1
# # #         self.cls = int(cls_id); self.conf=float(conf)
# # #         self.hits=1; self.age=1; self.time_since_update=0; self.last_frame=frame_idx
# # #         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
# # #         self.kf = make_kf(cxcywh)
# # #         self.history = deque(maxlen=TRAJ_MAX_LEN)
# # #         cx,cy,w,h = (self.kf.x[:4,0] if HAS_FILTERPY else self.kf.state)
# # #         self.history.append((int(cx),int(cy)))
# # #         x1,y1,x2,y2 = bbox_xyxy
# # #         self.last_box = np.array([x1,y1,x2,y2], np.float32)
# # #         self.last_area = max(1.0,(x2-x1)*(y2-y1))
# # #         self.last_center = np.array([cx,cy], np.float32)
# # #         self.last_speed = 0.0
# # #         self.vel = np.array([0.0,0.0], np.float32)

# # #     def predict(self):
# # #         if HAS_FILTERPY: self.kf.predict()
# # #         return (cxcywh_to_xyxy(self.kf.x[:4,0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state))

# # #     def update(self, bbox_xyxy, cls_id, conf, frame_idx):
# # #         prev_center = self.last_center.copy()
# # #         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
# # #         if HAS_FILTERPY: self.kf.update(cxcywh)
# # #         else: self.kf.update(cxcywh)
# # #         self.cls=int(cls_id); self.conf=float(conf)
# # #         self.hits+=1; self.time_since_update=0; self.last_frame=frame_idx
# # #         cx,cy,w,h = (self.kf.x[:4,0] if HAS_FILTERPY else self.kf.state)
# # #         self.history.append((int(cx),int(cy)))
# # #         x1,y1,x2,y2 = bbox_xyxy
# # #         cur_center = np.array([cx,cy], np.float32)
# # #         self.last_speed = float(np.linalg.norm(cur_center - self.last_center))
# # #         self.vel = cur_center - prev_center
# # #         self.last_center = cur_center
# # #         self.last_box = np.array([x1,y1,x2,y2], np.float32)
# # #         self.last_area = max(1.0,(x2-x1)*(y2-y1))

# # #     def mark_missed(self):
# # #         self.time_since_update += 1; self.age += 1

# # # class ByteTrackLike:
# # #     def __init__(self, iou_gate=0.2, max_age=30, min_hits=3):
# # #         self.iou_gate=iou_gate; self.max_age=max_age; self.min_hits=min_hits
# # #         self.tracks=[]; self.birth_pool={}; self.lost={}  # id -> {"track":t,"age":k}

# # #     def _match(self, tracks, dets):
# # #         if len(tracks)==0 or len(dets)==0:
# # #             return [], list(range(len(tracks))), list(range(len(dets)))
# # #         track_boxes=np.array([t.predict() for t in tracks], np.float32)
# # #         det_boxes  =np.array([d["bbox"] for d in dets], np.float32)
# # #         iou = iou_xyxy(track_boxes, det_boxes); cost = 1.0 - iou
# # #         row_ind, col_ind = linear_sum_assignment(cost)
# # #         matches=[]; unmatched_t=[]; unmatched_d=[]
# # #         tr_set=set(row_ind.tolist()); dt_set=set(col_ind.tolist())
# # #         for r in range(len(tracks)):
# # #             if r not in tr_set: unmatched_t.append(r)
# # #         for c in range(len(dets)):
# # #             if c not in dt_set: unmatched_d.append(c)
# # #         for r,c in zip(row_ind, col_ind):
# # #             if iou[r,c] >= self.iou_gate: matches.append((r,c))
# # #             else: unmatched_t.append(r); unmatched_d.append(c)
# # #         return matches, unmatched_t, unmatched_d

# # #     def _gate_ok(self, t: Track, det_bbox, det_conf):
# # #         if t.hits < 2 or det_conf >= HIGH_CONF_RELAX:
# # #             return True
# # #         x1,y1,x2,y2 = det_bbox
# # #         area = max(1.0,(x2-x1)*(y2-y1))
# # #         grow = area / max(1.0, t.last_area)
# # #         if grow > SIZE_CHANGE_MAX or grow < SIZE_CHANGE_MIN: return False
# # #         cx=(x1+x2)/2.0; cy=(y1+y2)/2.0
# # #         cur_speed = float(np.linalg.norm(np.array([cx,cy],np.float32)-t.last_center))
# # #         return abs(cur_speed - t.last_speed) <= MAX_SPEED_DELTA

# # #     def _near_existing(self, det_bbox, W, H):
# # #         if not self.tracks: return False
# # #         d_cx,d_cy = center_of_box(det_bbox)
# # #         for t in self.tracks:
# # #             tb=t.predict()
# # #             if iou_xyxy(np.array([tb]), np.array([det_bbox]))[0,0] >= NEW_NEAR_IOU: return True
# # #             tcx,tcy = center_of_box(tb)
# # #             if np.hypot(d_cx-tcx, d_cy-tcy) <= NEW_NEAR_DIST: return True
# # #         return False

# # #     def _birth_key(self, det_bbox):
# # #         cx,cy = center_of_box(det_bbox)
# # #         return (int(round(cx/BIRTH_POS_TOL)), int(round(cy/BIRTH_POS_TOL)))

# # #     def _update_birth_pool(self, key, conf, frame_idx):
# # #         slot = self.birth_pool.get(key, {"count":0,"sum_conf":0.0,"last_frame":frame_idx-1})
# # #         if slot["last_frame"] != frame_idx-1:
# # #             slot = {"count":0,"sum_conf":0.0,"last_frame":frame_idx-1}
# # #         slot["count"] += 1; slot["sum_conf"] += conf; slot["last_frame"]=frame_idx
# # #         self.birth_pool[key]=slot; return slot

# # #     def _purge_birth_pool(self, frame_idx):
# # #         kill=[k for k,v in self.birth_pool.items() if frame_idx-v["last_frame"]>CANDIDATE_MAX_AGE]
# # #         for k in kill: self.birth_pool.pop(k, None)

# # #     def _predict_lost_center(self, t: Track, k_steps=1):
# # #         return t.last_center + k_steps * t.vel

# # #     def _try_revive(self, unmatched_dets_dict, frame_idx):
# # #         revived=set()
# # #         if not self.lost or not unmatched_dets_dict: return revived
# # #         for di,d in list(unmatched_dets_dict.items()):
# # #             db = d["bbox"]; dcx,dcy = center_of_box(db)
# # #             best_id=None; best_dist=1e9
# # #             for lid, item in self.lost.items():
# # #                 t=item["track"]; age=item["age"]
# # #                 if age>LOST_MAX_AGE: continue
# # #                 pred = self._predict_lost_center(t, k_steps=min(5, age+1))
# # #                 dist = np.hypot(dcx-pred[0], dcy-pred[1])
# # #                 if dist>REVIVE_DIST: continue
# # #                 # direction consistency
# # #                 v=t.vel; to_det=np.array([dcx,dcy],np.float32)-t.last_center
# # #                 nv=np.linalg.norm(v)+1e-6; nd=np.linalg.norm(to_det)+1e-6
# # #                 if float(np.dot(v,to_det)/(nv*nd)) < REVIVE_DIR_COS: continue
# # #                 if dist<best_dist: best_dist=dist; best_id=lid
# # #             if best_id is not None:
# # #                 t=self.lost[best_id]["track"]
# # #                 t.update(db, d["cls"], d["conf"], frame_idx)
# # #                 self.tracks.append(t)
# # #                 self.lost.pop(best_id, None)
# # #                 revived.add(di)
# # #         return revived

# # #     def update(self, detections, frame_idx, frame_size):
# # #         W,H = frame_size
# # #         self._purge_birth_pool(frame_idx)

# # #         # split by score
# # #         high=[d for d in detections if d["conf"]>=BYTE_HIGH_THRES]
# # #         low =[d for d in detections if BYTE_LOW_THRES<=d["conf"]<BYTE_HIGH_THRES]

# # #         # stage 1: high
# # #         matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
# # #         matched_global=set([ti for ti,_ in matches])
# # #         gated_tracks_s1=set(); gated_dets_s1=set()
# # #         for ti,di in matches:
# # #             t=self.tracks[ti]; d=high[di]
# # #             if self._gate_ok(t,d["bbox"],d["conf"]): t.update(d["bbox"],d["cls"],d["conf"],frame_idx)
# # #             else: gated_tracks_s1.add(ti); gated_dets_s1.add(di)

# # #         # stage 2: low on remaining tracks
# # #         remain_indices = list(set(unmatched_t)|gated_tracks_s1)
# # #         remain_tracks  = [self.tracks[i] for i in remain_indices]
# # #         m2, unmatched_t2_local, unmatched_low = self._match(remain_tracks, low)
# # #         matched_stage2_global=set()
# # #         for lti,di in m2:
# # #             gi=remain_indices[lti]; t=self.tracks[gi]; d=low[di]
# # #             if self._gate_ok(t,d["bbox"],d["conf"]):
# # #                 t.update(d["bbox"],d["cls"],d["conf"],frame_idx)
# # #                 matched_stage2_global.add(gi)
# # #             else:
# # #                 unmatched_low.append(di)

# # #         # Build dicts of unmatched detections for revival/birth
# # #         uh_dict={i:high[i] for i in (set(unmatched_high)|gated_dets_s1)}
# # #         ul_dict={i:low[i]  for i in unmatched_low}

# # #         # try revival from lost pool first (high then low)
# # #         revived_h = self._try_revive(uh_dict, frame_idx)
# # #         for i in revived_h: uh_dict.pop(i, None)
# # #         revived_l = self._try_revive(ul_dict, frame_idx)
# # #         for i in revived_l: ul_dict.pop(i, None)

# # #         # controlled births (only 3 allowed cases)
# # #         def maybe_birth_from_dict(det_dict):
# # #             for di,d in det_dict.items():
# # #                 allow=False
# # #                 if frame_idx==0:
# # #                     allow=True
# # #                 elif near_border(d["bbox"], W, H, BORDER_MARGIN_PX):
# # #                     allow=True
# # #                 else:
# # #                     near = self._near_existing(d["bbox"], W, H)
# # #                     key  = self._birth_key(d["bbox"])
# # #                     slot = self._update_birth_pool(key, d["conf"], frame_idx)
# # #                     avg_conf = slot["sum_conf"]/max(1,slot["count"])
# # #                     if slot["count"]>=BIRTH_MIN_FRAMES and avg_conf>=BIRTH_AVG_CONF:
# # #                         allow=True
# # #                 if allow:
# # #                     self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))
# # #                     # clear candidate if exists
# # #                     k = self._birth_key(d["bbox"])
# # #                     if k in self.birth_pool: self.birth_pool.pop(k, None)
# # #         maybe_birth_from_dict(uh_dict)
# # #         maybe_birth_from_dict(ul_dict)

# # #         # mark missed and move to lost if needed
# # #         matched_all = matched_global | matched_stage2_global
# # #         survivors=[]
# # #         for idx,t in enumerate(self.tracks):
# # #             if idx not in matched_all:
# # #                 t.mark_missed()
# # #                 # move to lost pool once missed enough and not at border
# # #                 if t.time_since_update >= LOST_TO_POOL and not near_border(t.last_box, W, H, BORDER_MARGIN_PX):
# # #                     self.lost[t.id]={"track":t,"age":0}
# # #                 # keep alive until MAX_AGE passes
# # #                 if t.time_since_update <= self.max_age:
# # #                     survivors.append(t)
# # #             else:
# # #                 survivors.append(t)
# # #         self.tracks = survivors

# # #         # age lost pool and drop stale
# # #         for lid in list(self.lost.keys()):
# # #             self.lost[lid]["age"] += 1
# # #             if self.lost[lid]["age"] > LOST_MAX_AGE:
# # #                 self.lost.pop(lid, None)

# # #         return self.tracks

# # # # =========================
# # # # ======  GT PARSER  ======
# # # # =========================

# # # def parse_sdd_annotations(path):
# # #     frames=defaultdict(list); max_x=0; max_y=0
# # #     with open(path,"r") as f:
# # #         for line in f:
# # #             s=line.strip()
# # #             if not s: continue
# # #             p=s.split()
# # #             if len(p)<10: continue
# # #             tid=int(p[0]); x1=float(p[1]); y1=float(p[2]); x2=float(p[3]); y2=float(p[4])
# # #             frame=int(p[5]); lost=int(p[6]); occl=int(p[7])
# # #             label=" ".join(p[9:]).strip().strip('"')
# # #             frames[frame].append({"id":tid,"bbox":[x1,y1,x2,y2],"lost":lost,"occluded":occl,"label":label})
# # #             max_x=max(max_x,x2); max_y=max(max_y,y2)
# # #     return frames, (int(math.ceil(max_x)), int(math.ceil(max_y)))

# # # def scale_bbox(b, sx, sy):
# # #     x1,y1,x2,y2=b; return [x1*sx,y1*sy,x2*sx,y2*sy]

# # # # =========================
# # # # ======  METRICS    ======
# # # # =========================

# # # def mse_per_frame(gt_centers, pred_centers):
# # #     if len(gt_centers)==0 or len(pred_centers)==0: return None
# # #     G=np.array(gt_centers,np.float32); P=np.array(pred_centers,np.float32)
# # #     diff=G[:,None,:]-P[None,:,:]; cost=np.sum(diff*diff,axis=2)
# # #     r,c=linear_sum_assignment(cost)
# # #     if len(r)==0: return None
# # #     return float(np.mean(cost[r,c]))

# # # # =========================
# # # # ======  DRAWING    ======
# # # # =========================

# # # def draw_box_id(frame, bbox, cls, tid, conf=None, color=(0,255,0), label_map=None):
# # #     x1,y1,x2,y2 = map(int,bbox)
# # #     cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
# # #     name = label_map.get(cls,str(cls)) if label_map else str(cls)
# # #     txt=f"{name} ID:{tid}"; 
# # #     if conf is not None: txt+=f" {conf:.2f}"
# # #     cv2.putText(frame,txt,(x1,max(0,y1-5)),FONT,FONT_SCALE,color,THICKNESS,cv2.LINE_AA)

# # # def draw_traj(frame, pts, color=(255,255,255)):
# # #     if len(pts)<2: return
# # #     for i in range(1,len(pts)):
# # #         cv2.line(frame, pts[i-1], pts[i], color, 2)

# # # # =========================
# # # # ======  MAIN LOOP  ======
# # # # =========================

# # # def main():
# # #     cap=cv2.VideoCapture(VIDEO_PATH)
# # #     if not cap.isOpened():
# # #         print(f"Cannot open video: {VIDEO_PATH}"); return
# # #     W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #     FPS=cap.get(cv2.CAP_PROP_FPS) or 30.0; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # #     frames_gt,(Wref,Href)=parse_sdd_annotations(ANNOT_PATH)
# # #     sx=W/float(Wref if Wref>0 else W); sy=H/float(Href if Href>0 else H)

# # #     model=YOLO(YOLO_WEIGHTS)
# # #     tracker=ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

# # #     track_paths={}; track_last_seen={}; track_cls={}
# # #     gt_paths=defaultdict(deque)

# # #     traj_rows=[]; mse_vals=[]
# # #     idx=0; paused=PAUSE_ON_START; did_seek=False

# # #     def _clamp(i): return max(0,min(total-1,i))
# # #     def _reset():
# # #         nonlocal tracker,track_paths,track_last_seen,track_cls
# # #         tracker=ByteTrackLike(iou_gate=IOU_GATE,max_age=MAX_AGE,min_hits=MIN_HITS)
# # #         track_paths={}; track_last_seen={}; track_cls={}
# # #     def _seek(to):
# # #         nonlocal idx,did_seek
# # #         idx=_clamp(to); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); _reset(); did_seek=True

# # #     if PAUSE_ON_START: print("Paused. 'r' resume.")

# # #     while True:
# # #         ok,frame=cap.read()
# # #         if not ok: break
# # #         visL=frame.copy(); visR=frame.copy()

# # #         # ---- GT ----
# # #         gt_cent=[]
# # #         if idx in frames_gt:
# # #             for a in frames_gt[idx]:
# # #                 if SKIP_GT_OCCLUDED and (a["lost"]==1 or a["occluded"]==1): continue
# # #                 bb=scale_bbox(a["bbox"], sx, sy)
# # #                 cx,cy,_,_=xyxy_to_cxcywh(np.array(bb,np.float32))
# # #                 gt_cent.append((cx,cy))
# # #                 gt_paths[a["id"]].append((int(cx),int(cy)))
# # #                 if len(gt_paths[a["id"]])>TRAJ_MAX_LEN: gt_paths[a["id"]].popleft()
# # #                 cls_id=LABEL_TO_ID.get(a["label"],0); col=CLASS_COLORS.get(cls_id,(0,255,0))
# # #                 draw_box_id(visL,bb,cls=cls_id,tid=a["id"],color=col,label_map=CLASS_NAMES)
# # #                 draw_traj(visL,list(gt_paths[a["id"]]),color=col)

# # #         # ---- YOLO + extra NMS ----
# # #         res=model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
# # #         dets=[]
# # #         if len(res)>0 and res[0].boxes is not None and len(res[0].boxes)>0:
# # #             xyxy=res[0].boxes.xyxy.cpu().numpy()
# # #             conf=res[0].boxes.conf.cpu().numpy()
# # #             cls =res[0].boxes.cls.cpu().numpy().astype(int)
# # #             for b,s,c in zip(xyxy,conf,cls):
# # #                 dets.append({"bbox":b.astype(np.float32),"conf":float(s),"cls":int(c)})
# # #         dets=nms_per_class(dets, DET_IOU_NMS)

# # #         # ---- Track ----
# # #         tracks=tracker.update(dets, idx, (W,H))

# # #         pred_cent=[]
# # #         for t in tracks:
# # #             if t.hits>=MIN_HITS or t.time_since_update==0:
# # #                 box=t.predict()
# # #                 cx,cy,w,h=(t.kf.x[:4,0] if HAS_FILTERPY else t.kf.state)
# # #                 pred_cent.append((float(cx),float(cy)))
# # #                 if t.id not in track_paths: track_paths[t.id]=deque(maxlen=TRAJ_MAX_LEN)
# # #                 track_paths[t.id].append((int(cx),int(cy)))
# # #                 track_last_seen[t.id]=idx; track_cls[t.id]=t.cls
# # #                 col=CLASS_COLORS.get(t.cls,(200,200,200))
# # #                 draw_box_id(visR,box,cls=t.cls,tid=t.id,conf=t.conf,color=col,label_map=CLASS_NAMES)
# # #                 draw_traj(visR,list(track_paths[t.id]),color=col)
# # #                 # save trajectory row: track_id, frame, x, y
# # #                 traj_rows.append([t.id, idx, float(cx), float(cy)])

# # #         # remove stale small paths
# # #         stale=[tid for tid,last in track_last_seen.items() if idx-last>MISS_FRAMES_TO_DROP_PATH]
# # #         for tid in stale:
# # #             track_paths.pop(tid,None); track_last_seen.pop(tid,None); track_cls.pop(tid,None)

# # #         if COMPUTE_MSE:
# # #             fm=mse_per_frame(gt_cent, pred_cent)
# # #             if fm is not None:
# # #                 mse_vals.append(fm)
# # #                 cv2.putText(visR,f"MSE:{fm:.2f}",(10,20),FONT,0.6,(0,0,0),3,cv2.LINE_AA)
# # #                 cv2.putText(visR,f"MSE:{fm:.2f}",(10,20),FONT,0.6,(255,255,255),1,cv2.LINE_AA)

# # #         # status overlay
# # #         cv2.putText(visR,f"Tracks:{len(tracks)} Lost:{len(tracker.lost)}",
# # #                     (10,40),FONT,0.6,(255,255,255),1,cv2.LINE_AA)

# # #         cv2.imshow("GT (rescaled + trajectories)", visL)
# # #         cv2.imshow("Detections + Tracking + trajectories", visR)

# # #         key=cv2.waitKey(0 if paused else 1)&0xFF
# # #         if key==ord('q'): break
# # #         if key==ord('p'): paused=True
# # #         elif key==ord('r'): paused=False
# # #         if paused:
# # #             if key==ord('o'): _seek(idx+1)
# # #             elif key==ord('i'): _seek(idx-1)
# # #             elif key==ord('l'): _seek(idx+100)
# # #             elif key==ord('k'): _seek(idx-100)
# # #         if not paused and not did_seek: idx+=1
# # #         did_seek=False

# # #     cap.release(); cv2.destroyAllWindows()

# # #     # Save trajectories
# # #     if traj_rows:
# # #         os.makedirs(os.path.dirname(TRAJ_CSV), exist_ok=True)
# # #         pd.DataFrame(traj_rows, columns=["track_id","frame","x","y"]).to_csv(TRAJ_CSV, index=False)
# # #         print(f"Trajectories saved: {TRAJ_CSV}")

# # #     if COMPUTE_MSE and len(mse_vals)>0:
# # #         print(f"Overall MSE: {np.mean(mse_vals):.3f}")

# # # if __name__ == "__main__":
# # #     if not hasattr(cv2,"imshow"):
# # #         print("OpenCV built without HighGUI"); sys.exit(1)
# # #     main()
