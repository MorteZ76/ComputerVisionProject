# import cv2
# from collections import deque


# """
# Human Motion Analysis: SDD video0/video3
# Two synchronized windows:
#   1) Ground-truth (rescaled) + GT trajectories
#   2) YOLOv8 detections + ByteTrack-like tracking (Hungarian) + trajectories

# Author: you
# """

# # =========================
# # ====== HYPERPARAMS ======
# # =========================

# # --- Paths ---
# # Path to your trained YOLOv8 weights and the target video + its SDD annotation file.
# YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

# # VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
# # ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# # To switch to video0, set:
# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
# ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"

# # --- Classes and colors (6 classes as in SDD) ---
# # Mapping between numeric class ids and human-readable names (must match your training class order).
# CLASS_NAMES = {
#     0: "Pedestrian",
#     1: "Biker",
#     2: "Skater",
#     3: "Cart",
#     4: "Car",
#     5: "Bus",
# }

# # Map SDD label strings (from annotations.txt) to your training class indices.
# LABEL_TO_ID = {
#     "Pedestrian": 0,
#     "Biker": 1,      # or "biker" in your file
#     "Skater": 2,     # or "skater"
#     "Cart": 3,
#     "Car": 4,
#     "Bus": 5,
# }

# # BGR colors for drawing boxes per class on the frames.
# CLASS_COLORS = {
#     0: (0, 255, 0),
#     1: (255, 0, 0),
#     2: (0, 0, 255),
#     3: (255, 255, 0),
#     4: (255, 0, 255),
#     5: (0, 255, 255),
# }

# # Smoothed kinematics
# track_speed = {}   # id -> float
# track_heading = {} # id -> deg (-180..180)

# track_vel_hist = {}  # id -> deque[(vx,vy)] of last N velocities

# lost_now = set()  # track ids LOST in the current frame


# # Kinematics averaging
# KINEMA_WINDOW = 10

# # LOST rendering
# LOST_COLOR = (0, 165, 255)  # orange




# # --- Speed/Direction display ---
# SPEED_EWMA_ALPHA = 0.6   # 1.0 = no smoothing
# DIR_EWMA_ALPHA   = 0.6
# ARROW_LEN        = 35
# SHOW_UNITS       = "px/s"  # change to "m/s" if you add calibration



# # --- Detection / NMS ---
# # YOLO confidence threshold and NMS IoU used at inference time.
# # Lowering DET_CONF_THRES yields more detections; raising removes low-confidence ones.
# # DET_CONF_THRES = 0.30

# # # checked 8 18
# # DET_CONF_THRES = 0.65
# # DET_IOU_NMS = 0.7
# DET_CONF_THRES = 0.45
# DET_IOU_NMS = 0.6

# # --- ByteTrack-like association ---
# # Parameters for the tracker association logic.
# # BYTE_HIGH_THRES = 0.50   # high-score set
# BYTE_HIGH_THRES = 0.68 # high-score set
# # BYTE_LOW_THRES = 0.10    # low-score set
# BYTE_LOW_THRES = 0.58 # low-score set
# # IOU_GATE = 0.20        # minimum IoU to consider a match
# IOU_GATE = 0.1        # minimum IoU to consider a match
# # MAX_AGE = 30             # frames to keep "alive" without updates
# MAX_AGE = 30             # frames to keep "alive" without updates
# MIN_HITS = 3             # warm-up before rendering id (optional usage)
# BORDER_MARGIN = 5  # pixels from edge to consider 'exit'
# # #checked 8 18 
# # BYTE_HIGH_THRES = 0.85 # high-score set
# # BYTE_LOW_THRES = 0.70 # low-score set
# # IOU_GATE = 0.1         # minimum IoU to consider a match
# # MAX_AGE = 30             # frames to keep "alive" without updates
# # MIN_HITS = 3             # warm-up before rendering id (optional usage)
# # BORDER_MARGIN = 15  # pixels from edge to consider 'exit'

# # --- Drawing / Trajectories ---
# # Limits for trajectory memory and simple path cleanup.
# MISS_FRAMES_TO_DROP_PATH = 1  # delete trajectory if not seen for 10 frames
# TRAJ_MAX_LEN = 2000            # cap stored points per track to avoid memory growth
# FONT = cv2.FONT_HERSHEY_SIMPLEX
# FONT_SCALE = 0.5
# THICKNESS = 2

# # --- Metrics / Output ---
# # If True, compute per-frame MSE between GT centers and tracked centers.
# # TRAJ_CSV = "video3_trajectoriesNEW.csv"  # video_id,track_id,frame,x,y
# TRAJ_CSV = "video0_trajectoriesNEW.csv"  # video_id,track_id,frame,x,y

# COMPUTE_MSE = True

# # --- Playback ---
# # Controls for visual playback and whether to skip occluded/lost GT boxes.
# SKIP_GT_OCCLUDED = True  # set True to skip occluded==1 or lost==1 GT boxes
# PAUSE_ON_START = False    # press any key to start

# # ---- Anti-jerk gates (small changes only) ----
# # Heuristics to reject implausible size/speed jumps when updating tracks.
# MAX_SPEED_DELTA = 2.5     # max change in speed (px/frame) between consecutive updates
# SIZE_CHANGE_MAX = 2.0     # max area growth ratio allowed (unless high conf)
# SIZE_CHANGE_MIN = 0.5     # max area shrink ratio allowed (unless high conf)
# HIGH_CONF_RELAX = 0.55    # if conf >= this, allow larger changes


# # =========================
# # ======  IMPORTS   =======
# # =========================
# import os
# import cv2
# import sys
# import math
# import time
# import numpy as np
# import pandas as pd
# from collections import deque, defaultdict

# # Ultralytics YOLOv8 for detection.
# try:
#     from ultralytics import YOLO
# except Exception as e:
#     print("Ultralytics not found. Install: pip install ultralytics")
#     raise

# # Hungarian algorithm for optimal assignment.
# try:
#     from scipy.optimize import linear_sum_assignment
# except Exception as e:
#     print("scipy not found. Install: pip install scipy")
#     raise

# # Kalman Filter (optional)
# # If filterpy exists, we use a proper KF; else a dummy stub is used.
# HAS_FILTERPY = True
# try:
#     from filterpy.kalman import KalmanFilter
# except Exception:
#     HAS_FILTERPY = False


# # =========================
# # ======  HELPERS    ======
# # =========================

# def exp_smooth(cur, prev, alpha=0.6):
#     return cur if prev is None else alpha*cur + (1-alpha)*prev

# def angle_wrap_deg(a):
#     # wrap to [-180, 180)
#     a = (a + 180.0) % 360.0 - 180.0
#     return a

# def smooth_angle(cur_deg, prev_deg, alpha=0.6):
#     if prev_deg is None:
#         return cur_deg
#     # shortest path around wrap
#     delta = angle_wrap_deg(cur_deg - prev_deg)
#     return angle_wrap_deg(prev_deg + alpha*delta)

# def angle_to_compass(deg):
#     # 8-way
#     dirs = ["E","NE","N","NW","W","SW","S","SE"]
#     idx = int(((deg % 360) + 22.5)//45) % 8
#     return dirs[idx]


# def iou_xyxy(a, b):
#     """IoU for [x1,y1,x2,y2] boxes. a: (N,4) b:(M,4) -> (N,M)"""
#     N = a.shape[0]
#     M = b.shape[0]
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
#     iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
#     return iou

# def is_near_border(bbox, frame_shape, border_ratio=0.04):
#     """Check if a box is close to any border of the frame."""
#     x1, y1, x2, y2 = bbox
#     H, W = frame_shape[:2]
#     border_x = W * border_ratio
#     border_y = H * border_ratio
#     return (
#         x1 <= border_x or y1 <= border_y or
#         x2 >= (W - border_x) or y2 >= (H - border_y)
#     )

# def xyxy_to_cxcywh(box):
#     """Convert [x1,y1,x2,y2] to [cx,cy,w,h]."""
#     x1, y1, x2, y2 = box
#     w = max(0.0, x2 - x1)
#     h = max(0.0, y2 - y1)
#     return np.array([x1 + w / 2.0, y1 + h / 2.0, w, h], dtype=np.float32)


# def cxcywh_to_xyxy(box):
#     """Convert [cx,cy,w,h] to [x1,y1,x2,y2]."""
#     cx, cy, w, h = box
#     x1 = cx - w / 2.0
#     y1 = cy - h / 2.0
#     x2 = cx + w / 2.0
#     y2 = cy + h / 2.0
#     return np.array([x1, y1, x2, y2], dtype=np.float32)


# # =========================
# # ======  TRACKER    ======
# # =========================

# def make_kf(initial_cxcywh):
#     """
#     Create a Kalman filter with 8D state [cx, cy, w, h, vx, vy, vw, vh].
#     If filterpy is missing, fall back to a minimal stub that stores last state only.
#     """
#     if HAS_FILTERPY:
#         kf = KalmanFilter(dim_x=8, dim_z=4)
#         dt = 1.0

#         # State transition (constant velocity in all four observed dims).
#         kf.F = np.eye(8, dtype=np.float32)
#         for i in range(4):
#             kf.F[i, i + 4] = dt

#         # Measurement function: we directly observe [cx, cy, w, h].
#         kf.H = np.zeros((4, 8), dtype=np.float32)
#         kf.H[0, 0] = 1
#         kf.H[1, 1] = 1
#         kf.H[2, 2] = 1
#         kf.H[3, 3] = 1

#         # Covariances: moderate prior uncertainty and noise.
#         kf.P *= 10.0
#         kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
#         q = 1.0
#         kf.Q = np.eye(8, dtype=np.float32) * q

#         kf.x[:4, 0] = initial_cxcywh.reshape(4)
#         return kf
#     else:
#         # Lightweight stub: keeps the last measurement as the "state".
#         class DummyKF:
#             def __init__(self, init_state):
#                 self.state = init_state.copy()

#             def predict(self):
#                 # No motion model. Keeps last state.
#                 return self.state

#             def update(self, z):
#                 self.state = z.copy()

#             @property
#             def x(self):
#                 # Return shape-compatible vector like filterpy's state.
#                 return np.concatenate([self.state, np.zeros(4, dtype=np.float32)])[:, None]

#         return DummyKF(initial_cxcywh.astype(np.float32))


# class Track:
#     """
#     Single target track. Wraps KF state + metadata (id, class, conf) and a short trajectory.
#     """
#     _next_id = 1

#     def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
#         self.id = Track._next_id
#         Track._next_id += 1

#         self.cls = int(cls_id)           # last class
#         self.conf = float(conf)          # last confidence
#         self.hits = 1                    # number of successful updates
#         self.age = 1                     # total frames since init
#         self.time_since_update = 0       # frames since last matched update
#         self.last_frame = frame_idx      # last frame index that updated this track

#         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
#         self.kf = make_kf(cxcywh)
#         # Ensure state initialized even if Dummy
#         if HAS_FILTERPY:
#             self.kf.predict()

#         # For drawing: store recent center points.
#         self.history = deque(maxlen=TRAJ_MAX_LEN)
#         cx, cy, w, h = cxcywh
#         self.history.append((int(cx), int(cy)))

#     def predict(self):
#         """Advance KF one step and return predicted box in xyxy."""
#         if HAS_FILTERPY:
#             self.kf.predict()
#         # else: state remains
#         pred_xyxy = cxcywh_to_xyxy(self.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state)
#         return pred_xyxy

#     def update(self, bbox_xyxy, cls_id, conf, frame_idx):
#         """Correct KF with new measurement and update bookkeeping."""
#         cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
#         if HAS_FILTERPY:
#             self.kf.update(cxcywh)
#         else:
#             self.kf.update(cxcywh)
#         self.cls = int(cls_id)
#         self.conf = float(conf)
#         self.hits += 1
#         self.time_since_update = 0
#         self.last_frame = frame_idx
#         cx, cy, w, h = (self.kf.x[:4, 0] if HAS_FILTERPY else self.kf.state)
#         self.history.append((int(cx), int(cy)))

#     def mark_missed(self):
#         """Increase the miss counters when not matched this frame."""
#         self.time_since_update += 1
#         self.age += 1


# class ByteTrackLike:
#     """
#     Minimal ByteTrack-like manager:
#       1) Associate high-confidence detections first.
#       2) Then low-confidence detections to the remaining tracks.
#       3) Start new tracks only from unmatched high-confidence detections.
#       4) Age and remove tracks after MAX_AGE misses.
#     """
#     def __init__(self, iou_gate= IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
#         self.iou_gate = iou_gate
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.tracks = []

#     def _match(self, tracks, dets):
#         """
#         Hungarian on cost = 1 - IoU. Reject pairs below iou_gate.
#         Returns:
#           matches:      list of (track_idx, det_idx)
#           unmatched_t:  list of unmatched track indices
#           unmatched_d:  list of unmatched det indices
#         """
#         if len(tracks) == 0 or len(dets) == 0:
#             return [], list(range(len(tracks))), list(range(len(dets)))

#         # Predict each track to build association boxes.
#         track_boxes = np.array([t.predict() for t in tracks], dtype=np.float32)
#         det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)
#         iou = iou_xyxy(track_boxes, det_boxes)
#         cost = 1.0 - iou

#         row_ind, col_ind = linear_sum_assignment(cost)
#         matches, unmatched_t, unmatched_d = [], [], []

#         # Tracks not selected by Hungarian are unmatched.
#         for r, t in enumerate(tracks):
#             if r not in row_ind:
#                 unmatched_t.append(r)
#         # Detections not selected by Hungarian are unmatched.
#         for c, _ in enumerate(dets):
#             if c not in col_ind:
#                 unmatched_d.append(c)

#         # Keep only pairs above IoU gate.
#         for r, c in zip(row_ind, col_ind):
#             if iou[r, c] >= self.iou_gate:
#                 matches.append((r, c))
#             else:
#                 unmatched_t.append(r)
#                 unmatched_d.append(c)

#         return matches, unmatched_t, unmatched_d

#     def update(self, detections, frame_idx):
#         """
#         Update the tracker with a new frame of detections.
#         detections: list of dicts with keys: bbox (xyxy), cls, conf
#         """
#         # Split detections by confidence as in ByteTrack.
#         high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
#         low = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

#         # Stage 1: match high-score dets to existing tracks.
#         matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
#         for ti, di in matches:
#             t = self.tracks[ti]
#             d = high[di]
#             t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

#         # Stage 2: the tracks that remained unmatched can try to match low-score dets.
#         remain_tracks = [self.tracks[i] for i in unmatched_t]
#         m2, unmatched_t2, unmatched_low = self._match(remain_tracks, low)
#         for local_ti, di in m2:
#             t = remain_tracks[local_ti]
#             d = low[di]
#             t.update(d["bbox"], d["cls"], d["conf"], frame_idx)

#         # Start new tracks only from unmatched high-score detections (ByteTrack policy).
#         new_high_idx = [i for i in unmatched_high]
#         for di in new_high_idx:
#             d = high[di]
#             self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

#         # Tracks not matched in either stage are marked as missed.
#         matched_global_t_idx = {ti for ti, _ in matches}
#         # add those matched in stage 2 (convert local indices)
#         matched_stage2_global = {unmatched_t[i] for i, _ in m2}
#         for idx, t in enumerate(self.tracks):
#             if idx not in matched_global_t_idx and idx not in matched_stage2_global:
#                 t.mark_missed()

#         # Remove dead tracks (too many consecutive misses).
#         self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

#         return self.tracks


# class KalmanBoxTracker:
#     """
#     Placeholder for a SORT-like box tracker (not used in this file).
#     Left as a stub to indicate prior experiments with bg-sub + SORT.
#     """
#     count = 0
#     def __init__(self, bbox):
#         ...
#         KalmanBoxTracker.count += 1
#         self.id = KalmanBoxTracker.count
#         self.hits = 0
#         self.no_losses = 0
#         self.history = []

#         # --- New fields for anti-jerk ---
#         self.last_box = np.array([x1,y1,x2,y2], float)
#         self.last_area = w*h
#         self.last_center = np.array([cx, cy], float)
#         self.last_speed = 0.0

#     def update(self, bbox):
#         x1,y1,x2,y2 = bbox
#         w = max(1.0, x2 - x1)
#         h = max(1.0, y2 - y1)
#         cx = x1 + w/2.0
#         cy = y1 + h/2.0
#         s = w*h
#         r = w/max(1.0, h)
#         z = np.array([[cx],[cy],[s],[r]], dtype=np.float32)
#         self.kf.correct(z)
#         self.hits += 1
#         self.no_losses = 0

#         # --- update auxiliaries ---
#         cur_center = np.array([cx, cy], float)
#         self.last_speed = float(np.linalg.norm(cur_center - self.last_center))
#         self.last_center = cur_center
#         self.last_box = np.array([x1,y1,x2,y2], float)
#         self.last_area = s


# class Sort:
#     """
#     Placeholder for a SORT manager (not used here). Kept for reference.
#     """
#     ...
#     def _valid_size(self, box, prev_area=None, conf=None):
#         x1,y1,x2,y2 = box
#         w = max(1.0, x2-x1); h = max(1.0, y2-y1)
#         area = w*h
#         ar = max(w/h, h/w)
#         if area < SIZE_MIN_AREA or ar > SIZE_MAX_ASPECT:
#             return False
#         if prev_area is not None and conf is not None:
#             grow = area / max(1.0, prev_area)
#             if conf < HIGH_CONF_RELAX and (grow > SIZE_CHANGE_MAX or grow < SIZE_CHANGE_MIN):
#                 return False
#         return True

#     def _valid_motion(self, tracker, new_box, conf):
#         # allow if not enough history
#         if tracker.hits < 1:
#             return True
#         x1,y1,x2,y2 = new_box
#         cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
#         cur_speed = float(np.linalg.norm(np.array([cx,cy]) - tracker.last_center))
#         # if confidence is high, relax
#         if conf >= HIGH_CONF_RELAX:
#             return True
#         return abs(cur_speed - tracker.last_speed) <= MAX_SPEED_DELTA


# # =========================
# # ======  GT PARSER  ======
# # =========================

# def parse_sdd_annotations(path):
#     """
#     Parse SDD-style annotations.
#     Returns dict:
#       frames[frame_idx] -> list of dicts:
#          { 'id': int, 'bbox': [x1,y1,x2,y2], 'lost':0/1, 'occluded':0/1, 'label':str }
#     Also returns inferred reference (W_ref, H_ref) used by the annotations file.
#     We infer the canvas size from the max x2,y2 seen in the file.
#     """
#     frames = defaultdict(list)
#     max_x = 0
#     max_y = 0
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split()
#             if len(parts) < 10:
#                 # Some files may have more columns, but we need at least 10
#                 continue
#             tid = int(parts[0])
#             x1 = float(parts[1])
#             y1 = float(parts[2])
#             x2 = float(parts[3])
#             y2 = float(parts[4])
#             frame = int(parts[5])
#             lost = int(parts[6])
#             occl = int(parts[7])
#             # parts[8] = generated (unused here)
#             # parts[9] = label in quotes
#             label_raw = " ".join(parts[9:])
#             label = label_raw.strip().strip('"')
#             # print (label) 
#             frames[frame].append({
#                 "id": tid,
#                 "bbox": [x1, y1, x2, y2],
#                 "lost": lost,
#                 "occluded": occl,
#                 "label": label
#             })
#             max_x = max(max_x, x2)
#             max_y = max(max_y, y2)
#     # Heuristic "reference size" from annotations’ max coords
#     W_ref = int(math.ceil(max_x))
#     H_ref = int(math.ceil(max_y))
#     return frames, (W_ref, H_ref)


# def scale_bbox(bbox, scale_x, scale_y):
#     """Scale a bbox from annotation canvas to the current video resolution."""
#     x1, y1, x2, y2 = bbox
#     return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


# # =========================
# # ======  METRICS    ======
# # =========================

# def mse_per_frame(gt_centers, pred_centers):
#     """
#     Compute per-frame MSE after matching GT centers to predicted centers.
#     Use Hungarian on squared euclidean distance. If no pairs, return None.
#     """
#     if len(gt_centers) == 0 or len(pred_centers) == 0:
#         return None
#     G = np.array(gt_centers, dtype=np.float32)
#     P = np.array(pred_centers, dtype=np.float32)
#     # cost = squared euclidean distance
#     diff = G[:, None, :] - P[None, :, :]
#     cost = np.sum(diff * diff, axis=2)
#     r, c = linear_sum_assignment(cost)
#     if len(r) == 0:
#         return None
#     matched_costs = cost[r, c]
#     return float(np.mean(matched_costs))


# # =========================
# # ======  DRAWING    ======
# # =========================

# def draw_direction_arrow(frame, cx, cy, deg, length=35, color=(255,255,255)):
#     rad = math.radians(deg)
#     ex = int(cx + length * math.cos(rad))
#     ey = int(cy + length * math.sin(rad))
#     cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), color, 2, tipLength=0.3)


# def draw_box_id(frame, bbox, cls, tid, conf=None, color=(0,255,0), label_map=None):
#     """Draw a rectangle with class name, id, and optional confidence."""
#     x1, y1, x2, y2 = map(int, bbox)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#     cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
#     text = f"{cls_name} ID:{tid}"
#     if conf is not None:
#         text += f" {conf:.2f}"
#     cv2.putText(frame, text, (x1, max(0, y1 - 5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


# def draw_traj(frame, pts, color=(255,255,255)):
#     """Draw a polyline through the stored center points of a track."""
#     if len(pts) < 2:
#         return
#     for i in range(1, len(pts)):
#         cv2.line(frame, pts[i-1], pts[i], color, 2)


# # =========================
# # ======  MAIN LOOP  ======
# # =========================

# def main():
#     # Open video and read basic properties.
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print(f"Cannot open video: {VIDEO_PATH}")
#         return

#     W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Load and index GT per frame. Also infer annotation canvas size for rescaling.
#     frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
#     scale_x = W / float(W_ref if W_ref > 0 else W)
#     scale_y = H / float(H_ref if H_ref > 0 else H)

#     # Create detector.
#     model = YOLO(YOLO_WEIGHTS)

#     # Create tracker.
#     tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

#     # Trajectory buffers for drawing removal after misses
#     # For tracker: store per track.id -> deque of points and last_seen frame
#     track_paths = {}   # id -> deque[(x,y)]
#     track_last_seen = {}  # id -> frame_idx
#     track_cls = {}     # id -> class id

#     # For GT: per object id, store its trajectory centers for the left window.
#     gt_paths = defaultdict(deque)  # id -> deque[(x,y)]

#     # For metrics
#     traj_rows = []  # video_id,track_id,frame,x,y
#     mse_values = []

#     frame_idx = 0

#     # --- Added: playback state + seek helpers ---
#     paused = PAUSE_ON_START
#     did_seek = False

#     def _clamp(i: int) -> int:
#         """Clamp frame index into valid range."""
#         return max(0, min(total_frames - 1, i))

#     def _reset_tracking_state():
#         """Reset tracker and on-screen trajectories after seeking."""
#         nonlocal tracker, track_paths, track_last_seen, track_cls
#         tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
#         track_paths = {}
#         track_last_seen = {}
#         track_cls = {}

#     def _seek_to(target_idx: int):
#         """Jump to a target frame and reset tracker state to avoid drift artifacts."""
#         nonlocal frame_idx, did_seek
#         frame_idx = _clamp(target_idx)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         _reset_tracking_state()
#         did_seek = True
#     # --- End Added ---

#     if PAUSE_ON_START:
#         print("Paused. Press 'r' to resume, 'p' to pause again.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Prepare left (GT) and right (Det+Track) views.
#         vis_gt = frame.copy()
#         vis_det = frame.copy()

#         # -----------------------
#         # Ground Truth window
#         # -----------------------
#         gt_boxes = []
#         gt_centers = []
#         if frame_idx in frames_gt:
#             for ann in frames_gt[frame_idx]:
#                 # Optionally skip occluded/lost GT to avoid penalizing detector on invisible targets.
#                 if SKIP_GT_OCCLUDED and (ann["lost"] == 1 or ann["occluded"] == 1):
#                     continue
#                 # Rescale GT box to the actual video resolution.
#                 bb = scale_bbox(ann["bbox"], scale_x, scale_y)
#                 cx, cy, _, _ = xyxy_to_cxcywh(np.array(bb, dtype=np.float32))
#                 gt_boxes.append(bb)
#                 gt_centers.append((cx, cy))
#                 # Accumulate GT trajectory for visualization.
#                 gt_paths[ann["id"]].append((int(cx), int(cy)))
#                 if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN:
#                     gt_paths[ann["id"]].popleft()
#                 lbl = ann["label"]
#                 gt_cls_id = LABEL_TO_ID.get(lbl, 0)
#                 color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))  # fallback color if unmapped
#                 draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
#                 draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

#         # -----------------------
#         # Detection + Tracking
#         # -----------------------
#         # YOLO inference for this frame (Ultralytics applies NMS internally).
#         yolo_res = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
#         detections = []
#         if len(yolo_res) > 0:
#             r = yolo_res[0]
#             if r.boxes is not None and len(r.boxes) > 0:
#                 xyxy = r.boxes.xyxy.cpu().numpy()
#                 conf = r.boxes.conf.cpu().numpy()
#                 cls = r.boxes.cls.cpu().numpy().astype(int)
#                 # Build list of detection dicts for the tracker.
#                 for b, s, c in zip(xyxy, conf, cls):
#                     detections.append({
#                         "bbox": b.astype(np.float32),
#                         "conf": float(s),
#                         "cls": int(c),
#                     })

#         # Update tracker with current frame detections.
#         tracks = tracker.update(detections, frame_idx)

#         # Prepare arrays for per-frame predicted centers for MSE.
#         pred_centers = []

#         # Draw tracked boxes and their trajectories.
#         for t in tracks:
#             # Optionally show only confirmed tracks (past MIN_HITS) or those just updated.
#             if t.hits >= MIN_HITS or t.time_since_update == 0:
#                 box = t.predict()  # current KF state box (after predict/update in this frame)
#                 if t.time_since_update > MAX_AGE or  is_near_border(box, frame.shape):
#                     continue
#                 cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.kf.state)
#                 pred_centers.append((float(cx), float(cy)))

#                 # Update short path for drawing.
#                 if t.id not in track_paths:
#                     track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
#                 track_paths[t.id].append((int(cx), int(cy)))
#                 track_last_seen[t.id] = frame_idx
#                 track_cls[t.id] = t
                
#                 # Skip dead or border-near as before
#                 if t.time_since_update > MAX_AGE:
#                     continue
#                 if is_near_border(box, frame.shape):
#                     # near borders are never marked LOST
#                     continue

#                 # A track is LOST if:
#                 #  - it is confirmed (enough hits),
#                 #  - and either it wasn't matched this frame OR it was matched but with low conf,
#                 #  - and it's not near the borders (checked above).
#                 is_confirmed = (t.hits >= MIN_HITS)
#                 failed_high_conf_match = (t.time_since_update > 0) or (t.time_since_update == 0 and t.conf < BYTE_HIGH_THRES)
#                 is_lost = is_confirmed and failed_high_conf_match

#                 if :
#                     lost_now.add(t.id)


#                 # --- Kinematics ---
#                 # --- Instant velocity (px/frame) from KF or centers ---
#                 if HAS_FILTERPY:
#                     vx = float(t.kf.x[4, 0])
#                     vy = float(t.kf.x[5, 0])
#                 else:
#                     hx = list(track_paths.get(t.id, []))[-2:]
#                     if len(hx) >= 2:
#                         (x0,y0),(x1,y1) = hx
#                         vx = float(x1 - x0)
#                         vy = float(y1 - y0)
#                     else:
#                         vx = vy = 0.0

#                 # --- Gate history update: only on high-confidence matched frames ---
#                 if t.time_since_update == 0 and t.conf >= BYTE_HIGH_THRES:
#                     if t.id not in track_vel_hist:
#                         track_vel_hist[t.id] = deque(maxlen=KINEMA_WINDOW)
#                     track_vel_hist[t.id].append((vx, vy))

#                 # --- Mean over last <=10 high-confidence samples ---
#                 hist = track_vel_hist.get(t.id, [])
#                 if len(hist) > 0:
#                     vx_mean = float(np.mean([v[0] for v in hist]))
#                     vy_mean = float(np.mean([v[1] for v in hist]))
#                     speed_mean = math.hypot(vx_mean, vy_mean) * FPS       # px/s
#                     angle_mean = math.degrees(math.atan2(vy_mean, vx_mean))  # deg
#                 else:
#                     speed_mean = 0.0
#                     angle_mean = 0.0

#                 # --- Draw using the mean kinematics ---
#                 color = LOST_COLOR if t.id in lost_now else CLASS_COLORS.get(t.cls, (200,200,200))

#                 # box + id (+ LOST tag)
#                 tag = " LOST" if t.id in lost_now else ""
#                 draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
#                 cv2.putText(vis_det, tag, (int(box[0]), max(0, int(box[1]) - 20)), FONT, 0.5, color, 2, cv2.LINE_AA)

#                 # trajectory + direction arrow + kinematics text
#                 draw_traj(vis_det, list(track_paths[t.id]), color=color)
#                 draw_direction_arrow(vis_det, cx, cy, angle_mean, length=ARROW_LEN, color=color)
#                 comp = angle_to_compass((angle_mean + 360) % 360)
#                 cv2.putText(vis_det, f"v:{speed_mean:.1f} px/s  dir:{comp}",
#                             (int(cx)+5, int(cy)+15), FONT, 0.5, color, 2, cv2.LINE_AA)



#                 color = CLASS_COLORS.get(t.cls, (200, 200, 200))
#                 draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
#                 draw_traj(vis_det, list(track_paths[t.id]), color=color)

#                 # Save one trajectory row per visible track for potential export.
#                 traj_rows.append(["video0", t.id, frame_idx, float(cx), float(cy)])

#         # Remove stale trajectories if track not updated for too long (UI cleanup only).
#         stale_ids = []
#         for tid, last_seen in track_last_seen.items():
#             if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH:
#                 stale_ids.append(tid)
#         for tid in stale_ids:
#             lost_now.discard(tid)
#             track_vel_hist.pop(tid, None)
#             track_speed.pop(tid, None)
#             track_heading.pop(tid, None)
#             track_paths.pop(tid, None)
#             track_last_seen.pop(tid, None)
#             track_cls.pop(tid, None)

#         # MSE for this frame (only if we have both GT and predictions).
#         if COMPUTE_MSE:
#             frame_mse = mse_per_frame(gt_centers, pred_centers)
#             if frame_mse is not None:
#                 mse_values.append(frame_mse)
#                 cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
#                 cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

#         # Show two windows side by side (user toggles below).
#         cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
#         cv2.imshow("Detections + ByteTrack + Hungarian + trajectories", vis_det)

#         # key handling: 0ms wait when paused, 1ms when playing
#         key = cv2.waitKey(0 if paused else 1) & 0xFF

#         if key == ord('q'):
#             # Quit
#             break

#         # ignore space explicitly
#         if key == ord(' '):
#             # Do nothing on space to avoid accidental toggles.
#             pass

#         # pause / resume
#         if key == ord('p'):
#             paused = True
#         elif key == ord('r'):
#             paused = False

#         # step and jumps ONLY when paused
#         if paused:
#             if key == ord('o'):      # next frame
#                 _seek_to(frame_idx + 1)
#             elif key == ord('i'):    # previous frame
#                 _seek_to(frame_idx - 1)
#             elif key == ord('l'):    # +100
#                 _seek_to(frame_idx + 100)
#             elif key == ord('k'):    # -100
#                 _seek_to(frame_idx - 100)

#         # advance on play
#         if not paused and not did_seek:
#             frame_idx += 1
#         did_seek = False

#     cap.release()
#     cv2.destroyAllWindows()

#     # Save trajectories CSV
#     if traj_rows:
#         pd.DataFrame(traj_rows, columns=["video_id", "track_id", "frame", "x", "y"]).to_csv(TRAJ_CSV, index=False)
#         print(f"Trajectories saved to {TRAJ_CSV}")

#     # Print final MSE over all frames with valid matches.
#     if COMPUTE_MSE and len(mse_values) > 0:
#         print(f"Overall MSE: {np.mean(mse_values):.3f}")


# if __name__ == "__main__":
#     # Minimal guard for missing OpenCV GUI support
#     if not hasattr(cv2, "imshow"):
#         print("OpenCV built without HighGUI. Install opencv-python.")
#         sys.exit(1)
#     main()



import os
import cv2
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# Ultralytics YOLOv8
try:
    from ultralytics import YOLO
except Exception:
    print("Ultralytics not found. Install: pip install ultralytics")
    raise

# Hungarian
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    print("scipy not found. Install: pip install scipy")
    raise

# Kalman
HAS_FILTERPY = True
try:
    from filterpy.kalman import KalmanFilter
except Exception:
    HAS_FILTERPY = False


"""
Human Motion Analysis: SDD video0/video3
Two synchronized windows:
  1) Ground-truth (rescaled) + GT trajectories
  2) YOLOv8 detections + ByteTrack-like tracking (Hungarian) + trajectories
  + Mean kinematics over last 10 high-confidence updates
  + LOST state managed inside tracker
Author: you
"""

# =========================
# ====== HYPERPARAMS ======
# =========================

# --- Paths ---
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
# ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"

# --- Classes and colors (6 classes as in SDD) ---
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

LOST_COLOR = (0, 165, 255)  # orange

# --- Detection / NMS ---
DET_CONF_THRES = 0.45
DET_IOU_NMS = 0.6

# --- ByteTrack-like association ---
BYTE_HIGH_THRES = 0.68
BYTE_LOW_THRES = 0.58
IOU_GATE = 0.1
MAX_AGE = 30
MIN_HITS = 3
BORDER_MARGIN = 5  # pixels from edge to consider 'exit'

# --- Drawing / Trajectories ---
MISS_FRAMES_TO_DROP_PATH = 5
TRAJ_MAX_LEN = 2000
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
ARROW_LEN = 35  # direction arrow length

# --- Kinematics averaging ---
KINEMA_WINDOW = 10  # last N high-confidence updates
# SHOW_UNITS = "px/s"  # for speed text
SHOW_UNITS = "px/frame"


# --- Metrics / Output ---
TRAJ_CSV = "video3_trajectoriesNEW.csv"
COMPUTE_MSE = True

# --- Playback ---
SKIP_GT_OCCLUDED = True
PAUSE_ON_START = False

# ---- Anti-jerk gates (small changes only) ----
MAX_SPEED_DELTA = 2.5
SIZE_CHANGE_MAX = 1.1
SIZE_CHANGE_MIN = 0.9
HIGH_CONF_RELAX = BYTE_HIGH_THRES


# --- Area-based tracking averaging ---
AREA_WINDOW = 10          # same as KINEMA_WINDOW
AREA_W_MEAN = 0.7         # weight on mean of history
AREA_W_LAST = 0.3         # weight on last sample





# =========================
# ======  HELPERS    ======
# =========================


def _area_xyxy(box):
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h

def _blended_ref_area(hist, w_mean=AREA_W_MEAN, w_last=AREA_W_LAST):
    if not hist: return None
    a_mean = float(np.mean(hist)); a_last = float(hist[-1])
    return w_mean*a_mean + w_last*a_last

def _valid_size_change(prev_box, det_box, conf):
    # Allow if confident enough
    if conf >= HIGH_CONF_RELAX:
        return True
    prev_a = _area_xyxy(prev_box)
    if prev_a <= 1.0:
        return True
    ratio = _area_xyxy(det_box) / prev_a
    return (SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX)

def _track_box_now(t):
    return cxcywh_to_xyxy(t.kf.x[:4, 0])


def iou_xyxy(a, b):
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
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def is_near_border(bbox, frame_shape, border_ratio=0.03):
    x1, y1, x2, y2 = bbox
    H, W = frame_shape[:2]
    border_x = W * border_ratio
    border_y = H * border_ratio
    return (x1 <= border_x or y1 <= border_y or x2 >= (W - border_x) or y2 >= (H - border_y))

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

def angle_to_compass(deg):
    dirs = ["E","NE","N","NW","W","SW","S","SE"]
    idx = int(((deg % 360) + 22.5)//45) % 8
    return dirs[idx]

def draw_direction_arrow(frame, cx, cy, deg, length=ARROW_LEN, color=(255,255,255)):
    rad = math.radians(deg)
    ex = int(cx + length * math.cos(rad))
    ey = int(cy + length * math.sin(rad))
    cv2.arrowedLine(frame, (int(cx), int(cy)), (ex, ey), color, 2, tipLength=0.3)


# =========================
# ======  TRACKER    ======
# =========================

def make_kf(initial_cxcywh):
    if HAS_FILTERPY:
        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.F[i, i + 4] = dt
        kf.H = np.zeros((4, 8), dtype=np.float32)
        kf.H[0, 0] = 1; kf.H[1, 1] = 1; kf.H[2, 2] = 1; kf.H[3, 3] = 1
        kf.P *= 10.0
        kf.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        kf.Q = np.eye(8, dtype=np.float32) * 1.0
        kf.x[:4, 0] = initial_cxcywh.reshape(4)
        return kf
    else:
        class DummyKF:
            def __init__(self, init_state):
                self.state = init_state.copy()
            def predict(self):
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
        self.id = Track._next_id; Track._next_id += 1
        self.cls = int(cls_id)
        self.conf = float(conf)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.matched_this_frame = False
        self.high_conf_match = False

        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, dtype=np.float32))
        self.kf = make_kf(cxcywh)
        if HAS_FILTERPY:
            self.kf.predict()

        self.history = deque(maxlen=TRAJ_MAX_LEN)
        cx, cy, w, h = cxcywh
        self.history.append((int(cx), int(cy)))

    def predict(self):
        if HAS_FILTERPY:
            self.kf.predict()
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
    def __init__(self, iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.lost_ids = set()  # persistent LOST set
        self.area_hist = {}  # track_id -> deque[area] of last high-conf matches


    def _match(self, tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        track_boxes = np.array([_track_box_now(t) for t in tracks], dtype=np.float32)
        det_boxes = np.array([d["bbox"] for d in dets], dtype=np.float32)

        iou = iou_xyxy(track_boxes, det_boxes)
        cost = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, unmatched_t, unmatched_d = [], [], []
        for r in range(len(tracks)):
            if r not in row_ind:
                unmatched_t.append(r)
        for c in range(len(dets)):
            if c not in col_ind:
                unmatched_d.append(c)
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_gate:
                matches.append((r, c))
            else:
                unmatched_t.append(r)
                unmatched_d.append(c)
        return matches, unmatched_t, unmatched_d

    def update(self, detections, frame_idx, frame_shape=None):
         # 1) single predict step for ALL tracks
        for t in self.tracks:
            if HAS_FILTERPY:
                t.kf.predict()
            # split by confidence
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        # clear per-frame match flags
        for t in self.tracks:
            t.matched_this_frame = False
            t.high_conf_match = False

        # stage 1: high
        matches, unmatched_t, unmatched_high = self._match(self.tracks, high)
        for ti, di in matches:
            t = self.tracks[ti]; d = high[di]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
            # update area history with this high-confidence detection
            a = _area_xyxy(d["bbox"])
            dq = self.area_hist.get(t.id)
            if dq is None:
                dq = deque(maxlen=AREA_WINDOW)
                self.area_hist[t.id] = dq
            dq.append(a)

            t.matched_this_frame = True
            t.high_conf_match = True

        # ---------- Stage 2: LOW with size gate vs blended area ----------
        remain_tracks = [self.tracks[i] for i in unmatched_t]
        accepted_pairs = []                             # [(local_ti, di)]
        unmatched_t2 = list(range(len(remain_tracks)))  # default: all unmatched
        unmatched_low = []

        if low and remain_tracks:
            m2, unmatched_t2_raw, unmatched_low_raw = self._match(remain_tracks, low)
            unmatched_t2 = set(unmatched_t2_raw)
            unmatched_low = list(unmatched_low_raw)

            for local_ti, di in m2:
                t = remain_tracks[local_ti]; d = low[di]
                det_box  = d["bbox"]
                det_area = _area_xyxy(det_box)

                ref_hist = self.area_hist.get(t.id)
                ref_area = _blended_ref_area(ref_hist)
                if ref_area is None:
                    ref_area = _area_xyxy(_track_box_now(t))

                ratio = det_area / max(1.0, ref_area)
                if SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX:
                    accepted_pairs.append((local_ti, di))
                    if local_ti in unmatched_t2:
                        unmatched_t2.remove(local_ti)
                else:
                    unmatched_low.append(di)

            for local_ti, di in accepted_pairs:
                t = remain_tracks[local_ti]; d = low[di]
                t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                t.matched_this_frame = True
                t.high_conf_match = False

            unmatched_t2 = list(unmatched_t2)



        # new tracks from unmatched high
        for di in unmatched_high:
            d = high[di]
            self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        # mark missed
        matched_global_t_idx = {ti for ti, _ in matches}
        matched_stage2_global = {unmatched_t[local_ti] for local_ti, _ in accepted_pairs}
        for idx, t in enumerate(self.tracks):
            if idx not in matched_global_t_idx and idx not in matched_stage2_global:
                t.mark_missed()


        # LOST maintenance before pruning
        new_lost = set(self.lost_ids)
        for t in self.tracks:
            box_now = _track_box_now(t)
            near_border = is_near_border(box_now, frame_shape, border_ratio = 0.08) if frame_shape is not None else False

            if t.matched_this_frame and t.high_conf_match:
                new_lost.discard(t.id)  # recovered
            else:
                if (t.hits >= self.min_hits) and (not near_border):
                    new_lost.add(t.id)
                else:
                    new_lost.discard(t.id)

        # remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # keep LOST only for alive tracks
        alive_ids = {t.id for t in self.tracks}
        self.lost_ids = {tid for tid in new_lost if tid in alive_ids}

        return self.tracks


# =========================
# ======  GT PARSER  ======
# =========================

def parse_sdd_annotations(path):
    frames = defaultdict(list)
    max_x = 0; max_y = 0
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
            lost = int(parts[6])
            occl = int(parts[7])
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
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return None
    G = np.array(gt_centers, dtype=np.float32)
    P = np.array(pred_centers, dtype=np.float32)
    diff = G[:, None, :] - P[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    r, c = linear_sum_assignment(cost)
    if len(r) == 0:
        return None
    return float(np.mean(cost[r, c]))


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
    scale_x = W / float(W_ref if W_ref > 0 else W)
    scale_y = H / float(H_ref if H_ref > 0 else H)

    model = YOLO(YOLO_WEIGHTS)
    tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

    track_paths = {}        # id -> deque[(x,y)]
    track_last_seen = {}    # id -> frame_idx
    track_cls = {}          # id -> class id
    track_vel_hist = {}     # id -> deque[(vx,vy)] of last high-conf updates

    gt_paths = defaultdict(deque)

    traj_rows = []
    mse_values = []
    frame_idx = 0

    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i: int) -> int:
        return max(0, min(total_frames - 1, i))

    def _reset_tracking_state():
        nonlocal tracker, track_paths, track_last_seen, track_cls, track_vel_hist
        tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
        track_paths = {}
        track_last_seen = {}
        track_cls = {}
        track_vel_hist = {}

    def _seek_to(target_idx: int):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(target_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _reset_tracking_state()
        did_seek = True

    if PAUSE_ON_START:
        print("Paused. Press 'r' to resume, 'p' to pause again.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis_gt = frame.copy()
        vis_det = frame.copy()

        # -------- GT window --------
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
                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN:
                    gt_paths[ann["id"]].popleft()
                lbl = ann["label"]
                gt_cls_id = LABEL_TO_ID.get(lbl, 0)
                color = CLASS_COLORS.get(gt_cls_id, (0, 255, 0))
                draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

        # -------- Detections + Tracking --------
        yolo_res = model.predict(source=frame, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)
        detections = []
        if len(yolo_res) > 0:
            r = yolo_res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for b, s, c in zip(xyxy, conf, cls):
                    detections.append({"bbox": b.astype(np.float32), "conf": float(s), "cls": int(c)})

        tracks = tracker.update(detections, frame_idx, frame.shape)

        pred_centers = []

        for t in tracks:
            # draw only confirmed or just updated
            if t.hits < MIN_HITS and t.time_since_update != 0:
                continue

            # box = t.predict()
            box = _track_box_now(t)
            cx, cy, w, h = t.kf.x[:4, 0]
            if t.time_since_update > MAX_AGE or is_near_border(box, frame.shape):
                continue

            # center for display and export
            cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.state)
            pred_centers.append((float(cx), float(cy)))

            # path bookkeeping
            if t.id not in track_paths:
                track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
            track_paths[t.id].append((int(cx), int(cy)))
            track_last_seen[t.id] = frame_idx
            track_cls[t.id] = t.cls

            # --- Instant velocity (px/frame) from KF or last two centers ---
            if HAS_FILTERPY:
                vx = float(t.kf.x[4, 0]); vy = float(t.kf.x[5, 0])
            else:
                hx = list(track_paths.get(t.id, []))[-2:]
                if len(hx) >= 2:
                    (x0, y0), (x1, y1) = hx
                    vx = float(x1 - x0); vy = float(y1 - y0)
                else:
                    vx = vy = 0.0

            # --- Update velocity history only on high-confidence matched frames ---
            if t.matched_this_frame and t.high_conf_match:
                if t.id not in track_vel_hist:
                    track_vel_hist[t.id] = deque(maxlen=KINEMA_WINDOW)
                track_vel_hist[t.id].append((vx, vy))

            # --- Mean kinematics over last <=10 high-confidence samples ---
            hist = track_vel_hist.get(t.id, [])
            if len(hist) > 0:
                vx_mean = float(np.mean([v[0] for v in hist]))
                vy_mean = float(np.mean([v[1] for v in hist]))
                vx_last, vy_last = hist[-1]

                # blend
                vx_blend = 0.7 * vx_mean + 0.3 * vx_last
                vy_blend = 0.7 * vy_mean + 0.3 * vy_last

                # speed_val = math.hypot(vx_blend, vy_blend) * FPS
                speed_val = math.hypot(vx_blend, vy_blend)  # px/frame
                angle_val = math.degrees(math.atan2(vy_blend, vx_blend))
            else:
                speed_val = 0.0
                angle_val = 0.0


            # --- Draw (LOST-aware color) ---
            color = LOST_COLOR if (t.id in tracker.lost_ids) else CLASS_COLORS.get(t.cls, (200, 200, 200))
            tag = " LOST" if (t.id in tracker.lost_ids) else ""
            draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
            if tag:
                cv2.putText(vis_det, tag, (int(box[0]), max(0, int(box[1]) - 20)), FONT, 0.5, color, 2, cv2.LINE_AA)

            draw_traj(vis_det, list(track_paths[t.id]), color=color)
            draw_direction_arrow(vis_det, cx, cy, angle_val, length=ARROW_LEN, color=color)
            comp = angle_to_compass((angle_val + 360) % 360)
            cv2.putText(vis_det, f"v:{speed_val:.1f} {SHOW_UNITS}  dir:{comp}",
                        (int(cx)+5, int(cy)+15), FONT, 0.5, color, 2, cv2.LINE_AA)

            # export row
            traj_rows.append(["video3", t.id, frame_idx, float(cx), float(cy)])

        # prune stale per-track UI trails
        stale_ids = []
        for tid, last_seen in list(track_last_seen.items()):
            if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH:
                stale_ids.append(tid)
        for tid in stale_ids:
            track_paths.pop(tid, None)
            track_last_seen.pop(tid, None)
            track_cls.pop(tid, None)
            track_vel_hist.pop(tid, None)

        # frame MSE
        if COMPUTE_MSE:
            frame_mse = mse_per_frame(gt_centers, pred_centers)
            if frame_mse is not None:
                mse_values.append(frame_mse)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10, 20), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # show
        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        cv2.imshow("Detections + ByteTrack + Hungarian + trajectories", vis_det)

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

    if traj_rows:
        pd.DataFrame(traj_rows, columns=["video_id", "track_id", "frame", "x", "y"]).to_csv(TRAJ_CSV, index=False)
        print(f"Trajectories saved to {TRAJ_CSV}")

    if COMPUTE_MSE and len(mse_values) > 0:
        print(f"Overall MSE: {np.mean(mse_values):.3f}")


if __name__ == "__main__":
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()
