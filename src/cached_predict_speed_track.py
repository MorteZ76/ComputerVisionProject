import os
import cv2
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict

# Hungarian Algorithm
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    print("scipy not found. Install: pip install scipy")
    raise

# Kalman Filter
HAS_FILTERPY = True
try:
    from filterpy.kalman import KalmanFilter
except Exception:
    HAS_FILTERPY = False

"""
Human Motion Analysis on SDD
Left: GT (rescaled) + GT trajectories
Right: Cached detections + ByteTrack-like tracking + trajectories
Adds: constant-velocity prediction while lost, removal after 20 lost frames,
revival only via high-confidence near match; HOTA + MSE.

MODS: Trajectories only store high-confidence points. Lost/predicted points are
NOT appended or saved. When a track is revived (high-conf after being lost),
the last stored high-conf point is connected to the first new high-conf point
automatically by the deque geometry (no intermediate lost points are stored).
"""

# =========================
# ====== HYPERPARAMS ======
# =========================

# --- Paths ---
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"
DET_PATH = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\detected\video0_detections.parquet"

# --- Class Definitions ---
CLASS_NAMES = {
    0: "Pedestrian",
    1: "Biker",
    2: "Skater",
    3: "Cart",
    4: "Car",
    5: "Bus"
}
LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255)
}

# --- Visual Settings ---
LOST_COLOR = (0, 165, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
ARROW_LEN = 35
SHOW_UNITS = "px/frame"

# --- Tracking Settings ---
PRED_EXPAND = 1.05
PRED_HORIZON_LOST = 100
LOST_SIZE_INFLATE = 1.05
SIZE_EMA = 0.25
REVIVE_IOU_THRES = 0.2
REVIVE_CENTER_MULT = 0.90
SPEED_CAP_ABS = 4.0
SPEED_CAP_DIAG_FRAC = 0.6
MIN_SPEED = 0.2

# --- Detection/NMS Settings ---
DET_CONF_THRES = 0.50
DET_IOU_NMS = 0.60
AGNOSTIC_NMS = True

# --- ByteTrack Association ---
BYTE_HIGH_THRES = 0.68
BYTE_LOW_THRES = 0.45
IOU_GATE = 0.08
MIN_HITS = 3

# --- Lost Object Handling ---
LOST_KEEP = 150
DISABLE_LOW_WHEN_LOST = True
CENTER_DIST_GATE = 75.0

# --- Drawing Settings ---
MISS_FRAMES_TO_DROP_PATH = 5
TRAJ_MAX_LEN = 2000

# --- Kinematic Averaging ---
KINEMA_WINDOW = 10

# --- Evaluation & Output ---
TRAJ_CSV = "video0_trajectoriesNEW.csv"
HOTA_CSV = "video0_hota_breakdown.csv"
COMPUTE_MSE = True
HOTA_TAUS = [i / 20 for i in range(1, 20)]  # 0.05 to 0.95

# --- Playback Options ---
SKIP_GT_OCCLUDED = True
PAUSE_ON_START = False

# --- Anti-Jerk Thresholds ---
SIZE_CHANGE_MAX = 1.1
SIZE_CHANGE_MIN = 0.9
HIGH_CONF_RELAX = BYTE_HIGH_THRES

# --- Area Consistency ---
AREA_WINDOW = 10
AREA_W_MEAN = 0.7
AREA_W_LAST = 0.3


# ===================================
# === LOGIC STARTS BELOW THIS POINT ===
# ===================================

# =========================
# ======  HELPERS    ======
# =========================

def _area_xyxy(box):
    w = max(0.0, box[2] - box[0]); h = max(0.0, box[3] - box[1])
    return w * h

def _blended_ref_area(hist, w_mean=AREA_W_MEAN, w_last=AREA_W_LAST):
    if not hist: return None
    a_mean = float(np.mean(hist)); a_last = float(hist[-1])
    return w_mean*a_mean + w_last*a_last

def _track_box_now(t):
    return cxcywh_to_xyxy(t.kf.x[:4, 0]) if HAS_FILTERPY else cxcywh_to_xyxy(t.kf.state)

def iou_xyxy(a, b):
    N = a.shape[0]; M = b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    x11,y11,x12,y12 = a[:,0][:,None],a[:,1][:,None],a[:,2][:,None],a[:,3][:,None]
    x21,y21,x22,y22 = b[:,0][None,:],b[:,1][None,:],b[:,2][None,:],b[:,3][None,:]
    inter_w = np.maximum(0, np.minimum(x12, x22) - np.maximum(x11, x21))
    inter_h = np.maximum(0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = inter_w * inter_h
    area_a = np.maximum(0, (a[:,2]-a[:,0])*(a[:,3]-a[:,1]))[:,None]
    area_b = np.maximum(0, (b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]
    union = area_a + area_b - inter
    return np.where(union > 0, inter/union, 0.0).astype(np.float32)

def is_near_border(bbox, frame_shape, border_ratio=0.04):
    x1, y1, x2, y2 = bbox
    H, W = frame_shape[:2]
    bx = W * border_ratio; by = H * border_ratio
    return (x1 <= bx or y1 <= by or x2 >= (W - bx) or y2 >= (H - by))

def xyxy_to_cxcywh(box):
    x1,y1,x2,y2 = box
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    return np.array([x1+w/2.0, y1+h/2.0, w, h], dtype=np.float32)

def cxcywh_to_xyxy(box):
    cx,cy,w,h = box
    return np.array([cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0], dtype=np.float32)

def _expand_xyxy(box, scale):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)

def draw_box_id(frame, bbox, cls, tid, conf=None, color=(0,255,0), label_map=None):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cls_name = label_map.get(cls, str(cls)) if label_map else str(cls)
    text = f"{cls_name} ID:{tid}" + (f" {conf:.2f}" if conf is not None else "")
    cv2.putText(frame, text, (x1, max(0, y1-5)), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def draw_traj(frame, pts, color=(255,255,255)):
    if len(pts) < 2: return
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

# ===== Cached dets + NMS =====

def iou_nms_xyxy(boxes, scores, iou_thres):
    if len(boxes) == 0: return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep

def nms_frame(rows, conf_thres, iou_thres, agnostic=False):
    # rows: Nx7 [frame,x1,y1,x2,y2,score,cls]
    if rows.size == 0: return []
    fr = rows[rows[:,5] >= conf_thres]
    if fr.size == 0: return []
    out = []
    if agnostic:
        boxes = fr[:,1:5].astype(np.float32); scores = fr[:,5].astype(np.float32)
        keep = iou_nms_xyxy(boxes, scores, iou_thres)
        for i in keep:
            out.append({"bbox": boxes[i], "conf": float(scores[i]), "cls": int(fr[i,6])})
    else:
        for c in np.unique(fr[:,6]).astype(int):
            sc = fr[fr[:,6]==c]
            boxes = sc[:,1:5].astype(np.float32); scores = sc[:,5].astype(np.float32)
            keep = iou_nms_xyxy(boxes, scores, iou_thres)
            for i in keep:
                out.append({"bbox": boxes[i], "conf": float(scores[i]), "cls": int(c)})
    return out

def load_dets(path):
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_csv(path)
    cols = ["frame","x1","y1","x2","y2","score","cls"]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"Det file missing required columns: {cols}")
    per_frame = defaultdict(list)
    for _, r in df.iterrows():
        per_frame[int(r["frame"])].append([
            int(r["frame"]), float(r["x1"]), float(r["y1"]),
            float(r["x2"]), float(r["y2"]), float(r["score"]), int(r["cls"])
        ])
    for k in list(per_frame.keys()):
        per_frame[k] = np.array(per_frame[k], dtype=np.float32)
    return per_frame


# =========================
# ======  TRACKER    ======
# =========================

def make_kf(initial_cxcywh):
    if HAS_FILTERPY:
        kf = KalmanFilter(dim_x=8, dim_z=4); dt = 1.0
        kf.F = np.eye(8, dtype=np.float32)
        for i in range(4): kf.F[i, i+4] = dt
        kf.H = np.zeros((4,8), dtype=np.float32)
        kf.H[0,0]=kf.H[1,1]=kf.H[2,2]=kf.H[3,3]=1
        kf.P *= 10.0
        kf.R = np.diag([1.0,1.0,10.0,10.0]).astype(np.float32)
        kf.Q = np.eye(8, dtype=np.float32) * 1.0
        kf.x[:4, 0] = initial_cxcywh.reshape(4)
        return kf
    else:
        class DummyKF:
            def __init__(self, init_state): self.state = init_state.copy()
            def predict(self): return self.state
            def update(self, z): self.state = z.copy()
            @property
            def x(self): return np.concatenate([self.state, np.zeros(4, np.float32)])[:, None]
        return DummyKF(initial_cxcywh.astype(np.float32))

class Track:
    _next_id = 1
    def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = Track._next_id; Track._next_id += 1
        self.cls = int(cls_id); self.conf = float(conf)
        self.hits = 1; self.age = 1
        self.time_since_update = 0; self.last_frame = frame_idx
        self.matched_this_frame = False; self.high_conf_match = False
        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, np.float32))
        self.kf = make_kf(cxcywh)
        if HAS_FILTERPY: self.kf.predict()
        self.history = deque(maxlen=TRAJ_MAX_LEN)
        cx, cy, _, _ = cxcywh; self.history.append((int(cx), int(cy)))
        self.det_hist = deque(maxlen=KINEMA_WINDOW)
        self.det_hist.append((float(cx), float(cy), int(frame_idx)))  # first measurement

    def predict(self):
        if HAS_FILTERPY: self.kf.predict()
        return cxcywh_to_xyxy(self.kf.x[:4,0]) if HAS_FILTERPY else cxcywh_to_xyxy(self.kf.state)

    def update(self, bbox_xyxy, cls_id, conf, frame_idx):
        cxcywh = xyxy_to_cxcywh(np.array(bbox_xyxy, np.float32))
        self.kf.update(cxcywh)
        self.cls = int(cls_id); self.conf = float(conf)
        self.hits += 1; self.time_since_update = 0; self.last_frame = frame_idx
        cx, cy, _, _ = (self.kf.x[:4,0] if HAS_FILTERPY else self.kf.state)
        self.history.append((int(cx), int(cy)))
        self.det_hist.append((float(cx), float(cy), int(frame_idx)))

    def mark_missed(self):
        self.time_since_update += 1; self.age += 1

class ByteTrackLike:
    def __init__(self, iou_gate=IOU_GATE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.min_hits = min_hits
        self.tracks = []
        self.lost_ids = set()
        self.area_hist = {}                 # id -> deque of areas (from high-conf matches)
        self.future = defaultdict(dict)     # frame_idx -> {track_id: bbox_xyxy} scheduled predictions
        self.vel_hist = {}                  # id -> deque of (vx, vy) from matches
        self.size_ref = {}                  # id -> (w, h) from last high-conf match (EMA)

    # ---------- helpers ----------

    def _update_size_ref(self, t: 'Track', det_bbox):
        # use detection's size (more stable than KF state), with EMA smoothing
        cx, cy, w, h = xyxy_to_cxcywh(np.array(det_bbox, np.float32))
        if t.id not in self.size_ref:
            self.size_ref[t.id] = (float(w), float(h))
        else:
            pw, ph = self.size_ref[t.id]
            self.size_ref[t.id] = (
                (1.0 - SIZE_EMA)*pw + SIZE_EMA*float(w),
                (1.0 - SIZE_EMA)*ph + SIZE_EMA*float(h)
            )

    def _cap_velocity(self, t, vx, vy):
        # dynamic cap: min(abs cap, fraction of diag)
        diag = self._diag_len(_track_box_now(t))
        cap = min(SPEED_CAP_ABS, SPEED_CAP_DIAG_FRAC * diag)
        speed = math.hypot(vx, vy)
        if speed <= MIN_SPEED:
            return 0.0, 0.0
        if speed > cap:
            s = cap / (speed + 1e-6)
            return vx * s, vy * s
        return vx, vy

    def _append_obs_velocity(self, t: 'Track'):
        # need at least two detection samples
        if len(t.det_hist) < 2:
            return
        (x0, y0, f0), (x1, y1, f1) = t.det_hist[-2], t.det_hist[-1]
        dt = max(1, int(f1 - f0))  # normalize by frame gap
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        # reject obvious spikes before storing
        cap = min(SPEED_CAP_ABS, SPEED_CAP_DIAG_FRAC * self._diag_len(_track_box_now(t)))
        if math.hypot(vx, vy) > 1.2 * cap:
            return

        vx, vy = self._cap_velocity(t, vx, vy)  # per-sample cap + dead-zone

        dqv = self.vel_hist.get(t.id)
        if dqv is None:
            dqv = deque(maxlen=KINEMA_WINDOW)
            self.vel_hist[t.id] = dqv
        dqv.append((vx, vy))
    
    def _clear_future_for(self, tid, from_frame):
        for f in list(self.future.keys()):
            if f < from_frame:
                continue
            self.future[f].pop(tid, None)
            if not self.future[f]:
                self.future.pop(f, None)

    def _get_mean_velocity(self, t):
        hist = self.vel_hist.get(t.id)
        if hist and len(hist) > 0:
            vx = float(np.mean([v[0] for v in hist]))
            vy = float(np.mean([v[1] for v in hist]))
        else:
            if HAS_FILTERPY:
                vx, vy = float(t.kf.x[4,0]), float(t.kf.x[5,0])
            else:
                vx = vy = 0.0
        return self._cap_velocity(t, vx, vy)

    def _schedule_future(self, t: 'Track', cur_frame: int, horizon: int = PRED_HORIZON_LOST):
        if HAS_FILTERPY:
            cx, cy, w, h = [float(v) for v in t.kf.x[:4, 0]]
        else:
            cx, cy, w, h = [float(v) for v in t.state]

        # velocity for center propagation
        vx, vy = self._get_mean_velocity(t)

        # keep size near last stable high-conf size (no per-step shrink)
        if t.id in self.size_ref:
            w, h = self.size_ref[t.id]
        # small inflate for visibility, but constant over horizon
        w = max(1.0, w * LOST_SIZE_INFLATE)
        h = max(1.0, h * LOST_SIZE_INFLATE)

        for k in range(1, horizon + 1):
            cxk = cx + k * vx
            cyk = cy + k * vy
            vbox = cxcywh_to_xyxy(np.array([cxk, cyk, w, h], np.float32))
            self.future[cur_frame + k][t.id] = vbox

    def _center_dists(self, boxes_a, boxes_b):
        acx = (boxes_a[:, 0:1] + boxes_a[:, 2:3]) * 0.5
        acy = (boxes_a[:, 1:2] + boxes_a[:, 3:4]) * 0.5
        bcx = (boxes_b[:, 0:1] + boxes_b[:, 2:3]) * 0.5
        bcy = (boxes_b[:, 1:2] + boxes_b[:, 3:4]) * 0.5
        return np.hypot(acx - bcx.T, acy - bcy.T)

    CHI2_POS_99 = 9.21  # 2 DoF, ~99% gate

    def _diag_len(self, box):
        w = max(0.0, box[2]-box[0]); h = max(0.0, box[3]-box[1])
        return math.hypot(w, h)

    def _dyn_center_gate(self, track_box):
        return max(CENTER_DIST_GATE, 0.35 * self._diag_len(track_box))

    def _maha2(self, track, det_cx, det_cy):
        if not HAS_FILTERPY:
            return None
        H = np.zeros((2,8), np.float32); H[0,0]=1; H[1,1]=1
        x = track.kf.x.reshape(-1,1)
        z = np.array([[det_cx],[det_cy]], np.float32)
        y = z - H @ x
        S = H @ track.kf.P @ H.T + np.eye(2, dtype=np.float32)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return None
        return float((y.T @ Sinv @ y)[0,0])

    def _match(self, tracks, dets):
        if len(tracks)==0 or len(dets)==0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        t_boxes = np.array([_track_box_now(t) for t in tracks], np.float32)
        d_boxes = np.array([d["bbox"] for d in dets], np.float32)
        iou = iou_xyxy(t_boxes, d_boxes)
        cost = 1.0 - iou  # base cost

        t_cx = (t_boxes[:,0] + t_boxes[:,2]) * 0.5
        t_cy = (t_boxes[:,1] + t_boxes[:,3]) * 0.5
        d_cx = (d_boxes[:,0] + d_boxes[:,2]) * 0.5
        d_cy = (d_boxes[:,1] + d_boxes[:,3]) * 0.5

        cost[iou < self.iou_gate] = 1e6

        for i, t in enumerate(tracks):
            gate_px = self._dyn_center_gate(t_boxes[i])
            for j in range(len(dets)):
                dx = float(abs(t_cx[i] - d_cx[j]))
                dy = float(abs(t_cy[i] - d_cy[j]))
                if dx*dx + dy*dy > gate_px*gate_px:
                    cost[i, j] = 1e6
                    continue
                m2 = self._maha2(t, d_cx[j], d_cy[j])
                if m2 is not None and m2 > self.CHI2_POS_99:
                    cost[i, j] = 1e6

        row, col = linear_sum_assignment(cost)
        matches, un_t, un_d = [], [], []
        rset, cset = set(row.tolist()), set(col.tolist())
        for i in range(len(tracks)):
            if i not in rset: un_t.append(i)
        for j in range(len(dets)):
            if j not in cset: un_d.append(j)
        for i, j in zip(row, col):
            if cost[i, j] < 1e5:
                matches.append((i, j))
            else:
                un_t.append(i); un_d.append(j)
        return matches, un_t, un_d

    # revival using scheduled predictions for THIS frame only, high-conf only
    def _revive_from_future(self, dets_high, frame_idx):
        if frame_idx not in self.future or len(self.future[frame_idx]) == 0:
            return set(), set()

        pred_items = list(self.future[frame_idx].items())
        tids = [tid for tid, _ in pred_items]
        pboxes = np.stack([box for _, box in pred_items], axis=0).astype(np.float32)

        if len(dets_high) == 0:
            return set(), set()

        dboxes = np.stack([d["bbox"] for d in dets_high], axis=0).astype(np.float32)
        dist = self._center_dists(pboxes, dboxes)
        # NEW: IoU and joint gate
        iou  = iou_xyxy(pboxes, dboxes)
        gated = dist.copy()
        for r in range(len(pboxes)):
            diag = self._diag_len(pboxes[r])
            gate = max(CENTER_DIST_GATE, 0.50 * diag)
            gated[r, :] = np.where(gated[r, :] <= gate, gated[r, :], 1e6)

        r, c = linear_sum_assignment(gated)
        used_det_idx = set()
        revived_tids = set()
        for ri, ci in zip(r, c):
            if gated[ri, ci] >= 1e5:
                continue
            tid = tids[ri]
            revived_tids.add(tid)
            used_det_idx.add(ci)

            for t in self.tracks:
                if t.id == tid:
                    d = dets_high[ci]
                    t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                    self._append_obs_velocity(t)
                    self._update_size_ref(t, d["bbox"])  # <-- remember stable size
                    t.matched_this_frame = True
                    t.high_conf_match = True

                    a = _area_xyxy(d["bbox"])
                    dq = self.area_hist.get(t.id)
                    if dq is None:
                        dq = deque(maxlen=AREA_WINDOW); self.area_hist[t.id] = dq
                    dq.append(a)

                    break

        for tid in revived_tids:
            self._clear_future_for(tid, frame_idx)

        return revived_tids, used_det_idx

    def update(self, detections, frame_idx, frame_shape=None):
        # 0) predict
        for t in self.tracks:
            if HAS_FILTERPY: t.kf.predict()

        # 1) split dets
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        # reset flags
        for t in self.tracks:
            t.matched_this_frame = False
            t.high_conf_match = False

        used_high = set()
        matched_global = set()

        # ---------- STAGE A: match HIGH to ACTIVE tracks only ----------
        active_idx = [i for i,t in enumerate(self.tracks) if t.time_since_update == 0]
        active_trs = [self.tracks[i] for i in active_idx]
        if active_trs and high:
            mA, un_act, un_hA = self._match(active_trs, high)
            for ai, dj in mA:
                ti = active_idx[ai]
                t = self.tracks[ti]; d = high[dj]
                t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                self._append_obs_velocity(t)
                # area hist
                a = _area_xyxy(d["bbox"])
                dq = self.area_hist.get(t.id)
                if dq is None: dq = deque(maxlen=AREA_WINDOW); self.area_hist[t.id] = dq
                dq.append(a)

                self._update_size_ref(t, d["bbox"])  # <-- remember stable size

                t.matched_this_frame = True
                t.high_conf_match = True
                self._clear_future_for(t.id, frame_idx)
                used_high.add(dj)
                matched_global.add(ti)

        # ---------- STAGE B: revive LOST tracks from FUTURE using leftover HIGH ----------
        if len(high) > 0 and len(used_high) > 0:
            idx_map = [i for i in range(len(high)) if i not in used_high]
            high_left = [high[i] for i in idx_map]
        else:
            idx_map = list(range(len(high)))
            high_left = high

        revived_tids, used_det_idx = self._revive_from_future(high_left, frame_idx)
        if used_det_idx:
            for k in used_det_idx:
                used_high.add(idx_map[k])
        if revived_tids:
            for i, t in enumerate(self.tracks):
                if t.id in revived_tids:
                    matched_global.add(i)

        # ---------- STAGE C: match LOW to ACTIVE tracks only ----------
        if low:
            still_unmatched_act = [i for i in active_idx if i not in matched_global]
            if still_unmatched_act:
                trs_lo = [self.tracks[i] for i in still_unmatched_act]
                mC, _, _ = self._match(trs_lo, low)
                accepted = []
                for li, dj in mC:
                    t = trs_lo[li]; d = low[dj]
                    det_area = _area_xyxy(d["bbox"])
                    ref_hist = self.area_hist.get(t.id)
                    ref_area = _blended_ref_area(ref_hist)
                    if ref_area is None: ref_area = _area_xyxy(_track_box_now(t))
                    ratio = det_area / max(1.0, ref_area)
                    if SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX:
                        accepted.append((li, dj))
                for li, dj in accepted:
                    ti = still_unmatched_act[li]
                    t = self.tracks[ti]; d = low[dj]
                    t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                    self._append_obs_velocity(t)
                    t.matched_this_frame = True
                    t.high_conf_match = False
                    self._clear_future_for(t.id, frame_idx)
                    matched_global.add(ti)

        # Before Stage D
        if frame_idx in self.future and self.future[frame_idx]:
            tids = list(self.future[frame_idx].keys())
            pboxes = np.stack(list(self.future[frame_idx].values()), axis=0).astype(np.float32)
            if high:
                dboxes = np.stack([d["bbox"] for d in high], axis=0).astype(np.float32)
                dist = self._center_dists(pboxes, dboxes)
                iou  = iou_xyxy(pboxes, dboxes)
                for j, d in enumerate(high):
                    if j in used_high:
                        continue
                    ok = False
                    for r in range(len(pboxes)):
                        diag = self._diag_len(pboxes[r])
                        gate = max(CENTER_DIST_GATE, REVIVE_CENTER_MULT * diag)
                        if (dist[r, j] <= gate) and (iou[r, j] >= REVIVE_IOU_THRES):
                            tid = tids[r]
                            # find the track and revive
                            for t in self.tracks:
                                if t.id == tid:
                                    t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                                    self._append_obs_velocity(t)
                                    self._update_size_ref(t, d["bbox"])
                                    t.matched_this_frame = True
                                    t.high_conf_match = True
                                    self._clear_future_for(t.id, frame_idx)
                                    used_high.add(j)
                                    ok = True
                                    break
                        if ok: break

        # ---------- STAGE D: births from completely unused HIGH ----------
        for j in range(len(high)):
            if j not in used_high:
                d = high[j]
                self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        # ---------- STAGE E: mark missed + schedule predictions ----------
        for i, t in enumerate(self.tracks):
            if i not in matched_global:
                prev_tsu = t.time_since_update
                t.mark_missed()
                if prev_tsu == 0:
                    # force KF velocity to true mean so predictions move correctly
                    vx_mean, vy_mean = self._get_mean_velocity(t)
                    if HAS_FILTERPY:
                        t.kf.x[4, 0] = vx_mean
                        t.kf.x[5, 0] = vy_mean
                    self._schedule_future(t, frame_idx, horizon=PRED_HORIZON_LOST)

        # LOST set for UI
        new_lost = set()
        for t in self.tracks:
            near_b = is_near_border(_track_box_now(t), frame_shape, border_ratio=0.04) if frame_shape is not None else False
            if (t.hits >= self.min_hits) and (t.time_since_update >= 2) and (not near_b):
                new_lost.add(t.id)
        self.lost_ids = new_lost

        # drop very-long-lost
        survivors = []
        for t in self.tracks:
            if t.time_since_update <= LOST_KEEP:
                survivors.append(t)
            else:
                self._clear_future_for(t.id, frame_idx + 1)
                self.vel_hist.pop(t.id, None)
        self.tracks = survivors

        return self.tracks


# =========================
# ======  GT PARSER  ======
# =========================

def parse_sdd_annotations(path):
    frames = defaultdict(list); max_x = 0; max_y = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            p = s.split()
            if len(p) < 10: continue
            tid = int(p[0]); x1=float(p[1]); y1=float(p[2]); x2=float(p[3]); y2=float(p[4])
            fr  = int(p[5]); lost=int(p[6]); occl=int(p[7])
            label = " ".join(p[9:]).strip().strip('"')
            frames[fr].append({"id":tid,"bbox":[x1,y1,x2,y2],"lost":lost,"occluded":occl,"label":label})
            max_x = max(max_x, x2); max_y = max(max_y, y2)
    return frames, (int(math.ceil(max_x)), int(math.ceil(max_y)))

def scale_bbox(bbox, sx, sy):
    x1,y1,x2,y2 = bbox
    return [x1*sx, y1*sy, x2*sx, y2*sy]


# =========================
# ======  METRICS    ======
# =========================

def mse_per_frame(gt_centers, pred_centers, max_dist=60.0):
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return None
    G = np.array(gt_centers, np.float32)
    P = np.array(pred_centers, np.float32)
    diff = G[:, None, :] - P[None, :, :]
    cost = np.sum(diff * diff, axis=2)  # squared pixels
    r, c = linear_sum_assignment(cost)
    if len(r) == 0:
        return None
    sel = cost[r, c] <= (max_dist * max_dist)
    if not np.any(sel):
        return None
    return float(np.mean(cost[r[sel], c[sel]]))


def build_gt_by_frame(frames_gt, scale_x, scale_y, skip_occ=True):
    out = {}
    for f, anns in frames_gt.items():
        cur = []
        for ann in anns:
            if skip_occ and (ann["lost"]==1 or ann["occluded"]==1): continue
            bb = scale_bbox(ann["bbox"], scale_x, scale_y)
            cur.append((int(ann["id"]), np.array(bb, np.float32)))
        if cur: out[int(f)] = cur
    return out

def _match_frame(g_ids, g_boxes, t_ids, t_boxes, tau):
    if len(g_ids)==0 or len(t_ids)==0:
        return [], list(range(len(g_ids))), list(range(len(t_ids)))
    iou = iou_xyxy(g_boxes, t_boxes)
    cost = 1.0 - iou; cost[iou < tau] = 1e6
    r, c = linear_sum_assignment(cost)
    matches, un_g, un_t = [], [], []
    rset, cset = set(r.tolist()), set(c.tolist())
    for i in range(len(g_ids)):
        if i not in rset: un_g.append(i)
    for j in range(len(t_ids)):
        if j not in cset: un_t.append(j)
    for i, j in zip(r, c):
        if iou[i, j] >= tau: matches.append((i, j))
        else: un_g.append(i); un_t.append(j)
    return matches, un_g, un_t

def eval_hota(gt_by_frame, pred_by_frame, total_frames, taus):
    rows = []
    for tau in taus:
        TP=FP=FN=0; g2t={}
        for f in range(total_frames):
            g_list = gt_by_frame.get(f, []); t_list = pred_by_frame.get(f, [])
            g_ids = [gid for gid,_ in g_list]; t_ids = [tid for tid,_ in t_list]
            g_boxes = np.array([b for _,b in g_list], np.float32) if g_list else np.zeros((0,4),np.float32)
            t_boxes = np.array([b for _,b in t_list], np.float32) if t_list else np.zeros((0,4),np.float32)
            m, ug, ut = _match_frame(g_ids, g_boxes, t_ids, t_boxes, tau)
            TP += len(m); FP += len(ut); FN += len(ug)
            g2t[f] = { g_ids[i]: t_ids[j] for i,j in m }
        det_den = TP + 0.5*(FP+FN)
        DetA = (TP/det_den) if det_den>0 else 0.0
        pairs = {(gid,tid) for f,mm in g2t.items() for gid,tid in mm.items()}
        accs = []
        for gid, tid in pairs:
            IDTP=IDFP=IDFN=0
            for f in range(total_frames):
                g_present = any(g==gid for g,_ in gt_by_frame.get(f, []))
                t_present = any(t==tid for t,_ in pred_by_frame.get(f, []))
                if g_present and t_present:
                    if g2t.get(f, {}).get(gid, None) == tid: IDTP += 1
                    else: IDFP += 1; IDFN += 1
                elif g_present and not t_present: IDFN += 1
                elif t_present and not g_present: IDFP += 1
            denom = IDTP + 0.5*(IDFP+IDFN)
            if denom>0: accs.append(IDTP/denom)
        AssA = float(np.mean(accs)) if accs else 0.0
        HOTA = math.sqrt(max(0.0, DetA)*max(0.0, AssA))
        rows.append({"tau":tau, "DetA":DetA, "AssA":AssA, "HOTA":HOTA})
    df = pd.DataFrame(rows)
    return df, float(df["DetA"].mean()), float(df["AssA"].mean()), float(df["HOTA"].mean())


# =========================
# ======  MAIN LOOP  ======
# =========================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}"); return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # GT and scaling
    frames_gt, (W_ref, H_ref) = parse_sdd_annotations(ANNOT_PATH)
    scale_x = W / float(W_ref if W_ref > 0 else W)
    scale_y = H / float(H_ref if H_ref > 0 else H)
    gt_by_frame = build_gt_by_frame(frames_gt, scale_x, scale_y, SKIP_GT_OCCLUDED)
    pred_by_frame = defaultdict(list)

    # Cached detections
    dets_by_frame = load_dets(DET_PATH)

    tracker = ByteTrackLike(iou_gate=IOU_GATE, min_hits=MIN_HITS)

    track_paths = {}         # stores only HIGH-CONF centers per track
    track_last_seen = {}     # last frame when we appended a HIGH-CONF point
    track_cls = {}
    gt_paths = defaultdict(deque)

    traj_rows = []
    mse_values = []
    frame_idx = 0

    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i:int)->int: return max(0, min(total_frames-1, i))

    def _reset_tracking_state():
        nonlocal tracker, track_paths, track_last_seen, track_cls
        tracker = ByteTrackLike(iou_gate=IOU_GATE, min_hits=MIN_HITS)
        track_paths = {}; track_last_seen = {}; track_cls = {}

    def _seek_to(target_idx:int):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(target_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _reset_tracking_state(); did_seek = True

    if PAUSE_ON_START: print("Paused. Press 'r' to resume, 'p' to pause again.")

    while True:
        # store previous lost set to detect revival this frame
        prev_lost = tracker.lost_ids.copy()

        ret, frame = cap.read()
        if not ret: break

        vis_gt = frame.copy()
        vis_det = frame.copy()

        # -------- GT window --------
        gt_boxes = []; gt_centers = []
        if frame_idx in frames_gt:
            for ann in frames_gt[frame_idx]:
                if SKIP_GT_OCCLUDED and (ann["lost"]==1 or ann["occluded"]==1): continue
                bb = scale_bbox(ann["bbox"], scale_x, scale_y)
                cx, cy, _, _ = xyxy_to_cxcywh(np.array(bb, np.float32))
                gt_boxes.append(bb); gt_centers.append((cx, cy))
                gt_paths[ann["id"]].append((int(cx), int(cy)))
                if len(gt_paths[ann["id"]]) > TRAJ_MAX_LEN: gt_paths[ann["id"]].popleft()
                gt_cls_id = LABEL_TO_ID.get(ann["label"], 0)
                color = CLASS_COLORS.get(gt_cls_id, (0,255,0))
                draw_box_id(vis_gt, bb, cls=gt_cls_id, tid=ann["id"], conf=None, color=color, label_map=CLASS_NAMES)
                draw_traj(vis_gt, list(gt_paths[ann["id"]]), color=color)

        # -------- Cached detections + NMS --------
        rows = dets_by_frame.get(frame_idx, np.empty((0,7), np.float32))
        detections = nms_frame(rows, DET_CONF_THRES, DET_IOU_NMS, AGNOSTIC_NMS)

        # -------- Tracking --------
        tracks = tracker.update(detections, frame_idx, frame.shape)

        pred_centers = []

        for t in tracks:
            # draw confirmed or just born; but we still predict while lost
            if t.hits < MIN_HITS and t.time_since_update != 0:
                continue

            box = _track_box_now(t)
            cx, cy, w, h = (t.kf.x[:4, 0] if HAS_FILTERPY else t.state)

            # log for HOTA: include when matched (keep behavior unchanged)
            if t.matched_this_frame:
                pred_by_frame[frame_idx].append((int(t.id), box.astype(np.float32).copy()))

            # centers for MSE: only when updated by a detection this frame
            if t.matched_this_frame and t.time_since_update == 0:
                pred_centers.append((float(cx), float(cy)))

            # --- TRAJECTORY STORAGE (MODIFIED) ---
            # Only store high-confidence matched centers (t.high_conf_match == True).
            # This prevents storing predicted/lost points. When a track is revived
            # from lost -> high_conf_match True, appending this point will create a
            # connection from the last stored high-conf point to the new one.
            if t.high_conf_match:
                if t.id not in track_paths:
                    track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
                # Append the new high-conf center (this will connect to the previous high-conf center,
                # if present, and will NOT include any lost-only points).
                track_paths[t.id].append((int(cx), int(cy)))
                track_last_seen[t.id] = frame_idx
                track_cls[t.id] = t.cls
                # save only the high-conf points to CSV rows
                traj_rows.append(["video0", t.id, frame_idx, float(cx), float(cy)])

            # UI drawing: choose display box based on lost state but do not use predicted points in stored traj
            if t.id in tracker.lost_ids and t.id in tracker.size_ref:
                sw, sh = tracker.size_ref[t.id]
                draw_box = cxcywh_to_xyxy(np.array([cx, cy, sw * LOST_SIZE_INFLATE, sh * LOST_SIZE_INFLATE], np.float32))
            elif t.id in tracker.lost_ids:
                draw_box = _expand_xyxy(box, PRED_EXPAND)
            else:
                draw_box = box

            color = LOST_COLOR if (t.id in tracker.lost_ids) else CLASS_COLORS.get(t.cls, (200,200,200))
            tag = " LOST" if (t.id in tracker.lost_ids) else ""
            draw_box_id(vis_det, draw_box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
            if tag:
                cv2.putText(vis_det, tag, (int(draw_box[0]), max(0, int(draw_box[1])-20)), FONT, 0.5, color, 2, cv2.LINE_AA)

            # draw trajectory using only stored high-conf points (no lost/pred points)
            if t.id in track_paths:
                draw_traj(vis_det, list(track_paths[t.id]), color=color)

            # velocity / direction display (from tracker vel_hist) - unchanged
            hist = tracker.vel_hist.get(t.id, [])
            if len(hist) > 0:
                vx_mean = float(np.mean([v[0] for v in hist])); vy_mean = float(np.mean([v[1] for v in hist]))
                vx_disp, vy_disp = tracker._cap_velocity(t, vx_mean, vy_mean)
                speed_raw = math.hypot(vx_disp, vy_disp)
                if speed_raw < MIN_SPEED:
                    speed_val = 0.0
                    angle_val = 0.0
                else:
                    speed_val = speed_raw
                    angle_val = math.degrees(math.atan2(vy_disp, vx_disp))
            else:
                speed_val = 0.0; angle_val = 0.0

            if speed_val >= MIN_SPEED:
                draw_direction_arrow(vis_det, cx, cy, angle_val, length=ARROW_LEN, color=color)
                comp = angle_to_compass((angle_val + 360) % 360)
            else:
                comp = "--"

            cv2.putText(vis_det, f"v:{speed_val:.1f} {SHOW_UNITS}  dir:{comp}",
                        (int(cx)+5, int(cy)+15), FONT, 0.5, color, 2, cv2.LINE_AA)

        # prune stale trails (UI only) based on last stored high-conf point
        stale_ids = [tid for tid,last_seen in list(track_last_seen.items())
                     if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH]
        for tid in stale_ids:
            track_paths.pop(tid, None); track_last_seen.pop(tid, None)
            track_cls.pop(tid, None)

        # frame MSE
        if COMPUTE_MSE:
            frame_mse = mse_per_frame(gt_centers, pred_centers)
            if frame_mse is not None:
                mse_values.append(frame_mse)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10,20), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis_det, f"MSE: {frame_mse:.2f}", (10,20), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # show
        cv2.imshow("GT (rescaled annotations + trajectories)", vis_gt)
        cv2.imshow("Cached dets + ByteTrack + trajectories", vis_det)

        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): pass
        if key == ord('p'): paused = True
        elif key == ord('r'): paused = False
        if paused:
            if key == ord('o'): _seek_to(frame_idx + 1)
            elif key == ord('i'): _seek_to(frame_idx - 1)
            elif key == ord('l'): _seek_to(frame_idx + 100)
            elif key == ord('k'): _seek_to(frame_idx - 100)
        if not paused and not did_seek: frame_idx += 1
        did_seek = False

    cap.release()
    cv2.destroyAllWindows()

    # Save only stored high-confidence trajectory points (lost/pred points were never appended)
    if traj_rows:
        pd.DataFrame(traj_rows, columns=["video_id","track_id","frame","x","y"]).to_csv(TRAJ_CSV, index=False)
        print(f"Trajectories saved to {TRAJ_CSV}")

    processed_frames = frame_idx + 1
    if gt_by_frame:
        df_hota, mean_DetA, mean_AssA, mean_HOTA = eval_hota(
            gt_by_frame, pred_by_frame, processed_frames, HOTA_TAUS
        )
        df_hota.to_csv(HOTA_CSV, index=False)
        print(f"HOTA saved to {HOTA_CSV}")
        print(f"Mean DetA: {mean_DetA:.4f}  Mean AssA: {mean_AssA:.4f}  Mean HOTA: {mean_HOTA:.4f}")
    else:
        print("HOTA skipped: no GT.")

    if COMPUTE_MSE and len(mse_values) > 0:
        print(f"Overall MSE: {np.mean(mse_values):.3f}")


if __name__ == "__main__":
    if not hasattr(cv2, "imshow"):
        print("OpenCV built without HighGUI. Install opencv-python.")
        sys.exit(1)
    main()
