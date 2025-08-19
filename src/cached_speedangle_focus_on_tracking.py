import os
import cv2
import sys
import math
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict

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
  2) Cached detections + ByteTrack-like tracking (Hungarian) + trajectories
  + Mean kinematics over last 10 high-confidence updates
  + LOST state managed inside tracker
Author: you
"""

# =========================
# ====== HYPERPARAMS ======
# =========================

# --- Paths ---
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"
DET_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\detected\video3_detections.parquet"  # cached file

# --- Classes and colors (6 classes as in SDD) ---
CLASS_NAMES = {0:"Pedestrian",1:"Biker",2:"Skater",3:"Cart",4:"Car",5:"Bus"}
LABEL_TO_ID = {"Pedestrian":0,"Biker":1,"Skater":2,"Cart":3,"Car":4,"Bus":5}
CLASS_COLORS = {
    0:(0,255,0), 1:(255,0,0), 2:(0,0,255),
    3:(255,255,0), 4:(255,0,255), 5:(0,255,255)
}
LOST_COLOR = (0,165,255)

# --- Detection / NMS ---
DET_CONF_THRES = 0.47
DET_IOU_NMS = 0.60
AGNOSTIC_NMS = False

# --- ByteTrack-like association ---
BYTE_HIGH_THRES = 0.68
BYTE_LOW_THRES  = 0.58
IOU_GATE        = 0.08
MAX_AGE         = 35
MIN_HITS        = 3

# new gates
CLASS_MATCH = True
CENTER_DIST_GATE = 60.0  # px

# --- ORU revival & gap interpolation ---
ORU_MAX_GAP  = 6          # max frames to interpolate on revival
REVIVE_DIST  = 28.0        # px radius to revive from lost pool
LOST_MAX_AGE = 60         # keep lost tracks for possible revival


# --- Drawing / Trajectories ---
MISS_FRAMES_TO_DROP_PATH = 5
TRAJ_MAX_LEN = 2000
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 2
ARROW_LEN = 35

# --- Kinematics averaging ---
KINEMA_WINDOW = 10
SHOW_UNITS = "px/frame"

# --- Metrics / Output ---
TRAJ_CSV = "video3_trajectoriesNEW.csv"
COMPUTE_MSE = True
HOTA_CSV = "video3_hota_breakdown.csv"
HOTA_TAUS = [i/20 for i in range(1, 20)]  # 0.05..0.95

# --- Playback ---
SKIP_GT_OCCLUDED = True
PAUSE_ON_START = False

# ---- Anti-jerk gates ----
MAX_SPEED_DELTA = 2.5
SIZE_CHANGE_MAX = 1.1
SIZE_CHANGE_MIN = 0.9
HIGH_CONF_RELAX = BYTE_HIGH_THRES

# --- Area-based tracking averaging ---
AREA_WINDOW = 10
AREA_W_MEAN = 0.7
AREA_W_LAST = 0.3


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

def _valid_size_change(prev_box, det_box, conf):
    if conf >= HIGH_CONF_RELAX: return True
    prev_a = _area_xyxy(prev_box)
    if prev_a <= 1.0: return True
    ratio = _area_xyxy(det_box) / prev_a
    return (SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX)

def _track_box_now(t):
    return cxcywh_to_xyxy(t.kf.x[:4, 0])

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

    def mark_missed(self):
        self.time_since_update += 1; self.age += 1

class ByteTrackLike:
    def __init__(self, iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.lost_ids = set()      # alive but currently lost (for coloring)
        self.area_hist = {}        # id -> deque[area] from high-conf matches
        self.lost_pool = {}        # id -> (Track, lost_age) after removal
        self.virtual_boxes = []    # [{"frame":f,"tid":id,"bbox":xyxy}] filled on each update
        

    def _match(self, tracks, dets):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        t_boxes = np.array([_track_box_now(t) for t in tracks], np.float32)
        d_boxes = np.array([d["bbox"] for d in dets], np.float32)

        # cost = 1 - IoU
        iou = iou_xyxy(t_boxes, d_boxes)
        cost = 1.0 - iou

        # class gate
        if CLASS_MATCH:
            t_cls = np.array([t.cls for t in tracks])[:, None]
            d_cls = np.array([d["cls"] for d in dets])[None, :]
            cost[t_cls != d_cls] = 1e6

        # center distance gate
        tcx = (t_boxes[:, 0:1] + t_boxes[:, 2:3]) * 0.5
        tcy = (t_boxes[:, 1:2] + t_boxes[:, 3:4]) * 0.5
        dcx = (d_boxes[None, :, 0] + d_boxes[None, :, 2]) * 0.5
        dcy = (d_boxes[None, :, 1] + d_boxes[None, :, 3]) * 0.5
        dist = np.hypot(tcx - dcx, tcy - dcy)
        cost[dist > CENTER_DIST_GATE] = 1e6

        # IoU gate
        cost[iou < self.iou_gate] = 1e6

        r, c = linear_sum_assignment(cost)
        matches, un_t, un_d = [], [], []
        rset, cset = set(r.tolist()), set(c.tolist())
        for i in range(len(tracks)):
            if i not in rset: un_t.append(i)
        for j in range(len(dets)):
            if j not in cset: un_d.append(j)
        for i, j in zip(r, c):
            if cost[i, j] < 1e5:
                matches.append((i, j))
            else:
                un_t.append(i); un_d.append(j)
        return matches, un_t, un_d


    def _try_revive(self, dets, frame_idx):
        revived_det_indices = []
        for j, d in enumerate(dets):
            db = d["bbox"]
            dcx, dcy, dw, dh = xyxy_to_cxcywh(db)
            best_id, best_dist = None, 1e9
            for lid, (lt, age) in self.lost_pool.items():
                if age > LOST_MAX_AGE: 
                    continue
                if CLASS_MATCH and lt.cls != d["cls"]:
                    continue
                tcx, tcy, _, _ = (lt.kf.x[:4, 0] if HAS_FILTERPY else lt.state)
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist < best_dist and dist <= REVIVE_DIST:
                    best_dist, best_id = dist, lid

            if best_id is None:
                continue

            trk, _ = self.lost_pool.pop(best_id)
            gap = frame_idx - trk.last_frame - 1
            if 0 < gap <= ORU_MAX_GAP:
                scx, scy, sw, sh = (trk.kf.x[:4, 0] if HAS_FILTERPY else trk.state)
                for k in range(1, gap + 1):
                    alpha = k / (gap + 1.0)
                    cx = (1 - alpha) * scx + alpha * dcx
                    cy = (1 - alpha) * scy + alpha * dcy
                    w  = (1 - alpha) * sw  + alpha * dw
                    h  = (1 - alpha) * sh  + alpha * dh
                    vbox = cxcywh_to_xyxy(np.array([cx, cy, w, h], np.float32))
                    self.virtual_boxes.append({"frame": trk.last_frame + k, "tid": trk.id, "bbox": vbox})
                    trk.history.append((int(cx), int(cy)))

            trk.update(db, d["cls"], d["conf"], frame_idx)
            self.tracks.append(trk)
            revived_det_indices.append(j)

        return revived_det_indices


    def update(self, detections, frame_idx, frame_shape=None):
        self.virtual_boxes = []  # clear per frame

        # one-step predict for all tracks
        for t in self.tracks:
            if HAS_FILTERPY: t.kf.predict()

        # split by confidence
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        # reset flags
        for t in self.tracks:
            t.matched_this_frame = False
            t.high_conf_match = False

        # stage 1: high
        matches, un_t, un_h = self._match(self.tracks, high)
        for ti, dj in matches:
            t = self.tracks[ti]; d = high[dj]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
            # high-conf area history
            a = _area_xyxy(d["bbox"])
            dq = self.area_hist.get(t.id)
            if dq is None: dq = deque(maxlen=AREA_WINDOW); self.area_hist[t.id] = dq
            dq.append(a)
            t.matched_this_frame = True
            t.high_conf_match = True

        # stage 2: low with size gating against blended area ref
        remain_tracks = [self.tracks[i] for i in un_t]
        accepted = []; un_t2 = list(range(len(remain_tracks))); un_l = []
        if low and remain_tracks:
            m2, un_t2_raw, un_l_raw = self._match(remain_tracks, low)
            un_t2 = set(un_t2_raw); un_l = list(un_l_raw)
            for lti, dj in m2:
                t = remain_tracks[lti]; d = low[dj]
                det_area = _area_xyxy(d["bbox"])
                ref_hist = self.area_hist.get(t.id)
                ref_area = _blended_ref_area(ref_hist)
                if ref_area is None:
                    ref_area = _area_xyxy(_track_box_now(t))
                ratio = det_area / max(1.0, ref_area)
                if SIZE_CHANGE_MIN <= ratio <= SIZE_CHANGE_MAX:
                    accepted.append((lti, dj))
                    if lti in un_t2: un_t2.remove(lti)
                else:
                    un_l.append(dj)
            for lti, dj in accepted:
                t = remain_tracks[lti]; d = low[dj]
                t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
                t.matched_this_frame = True
                t.high_conf_match = False
            un_t2 = list(un_t2)

        # try to revive from lost_pool using unmatched HIGH dets
        uh = [high[j] for j in un_h]
        revived = self._try_revive(uh, frame_idx)
        for idx in sorted(revived, reverse=True):
            uh.pop(idx)

        # births from remaining unmatched high
        for d in uh:
            self.tracks.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        # mark missed
        matched_g1 = {ti for ti, _ in matches}
        matched_g2 = {un_t[lti] for lti, _ in accepted}
        for i, t in enumerate(self.tracks):
            if i not in matched_g1 and i not in matched_g2:
                t.mark_missed()

        # update LOST ids for alive tracks
        new_lost = set(self.lost_ids)
        for t in self.tracks:
            near_b = is_near_border(_track_box_now(t), frame_shape, border_ratio=0.04) if frame_shape is not None else False
            if t.matched_this_frame and t.high_conf_match:
                new_lost.discard(t.id)
            else:
                if (t.hits >= self.min_hits) and (not near_b):
                    new_lost.add(t.id)
                else:
                    new_lost.discard(t.id)

        # move expired to lost_pool, then prune
        survivors = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            else:
                # keep for potential revival if it didn't die at the border
                if not is_near_border(_track_box_now(t), frame_shape, border_ratio=0.04):
                    self.lost_pool[t.id] = (t, 0)
        self.tracks = survivors

        # age lost_pool
        for lid in list(self.lost_pool.keys()):
            trk, age = self.lost_pool[lid]
            age += 1
            if age > LOST_MAX_AGE:
                self.lost_pool.pop(lid, None)
            else:
                self.lost_pool[lid] = (trk, age)

        # keep LOST only for alive ids
        alive_ids = {t.id for t in self.tracks}
        self.lost_ids = {tid for tid in new_lost if tid in alive_ids}

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

def mse_per_frame(gt_centers, pred_centers):
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return None
    G = np.array(gt_centers, np.float32)
    P = np.array(pred_centers, np.float32)
    diff = G[:,None,:] - P[None,:,:]
    cost = np.sum(diff*diff, axis=2)
    r, c = linear_sum_assignment(cost)
    if len(r) == 0: return None
    return float(np.mean(cost[r, c]))

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

    # Cached detections indexed by frame
    dets_by_frame = load_dets(DET_PATH)

    tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

    track_paths = {}
    track_last_seen = {}
    track_cls = {}
    track_vel_hist = {}
    gt_paths = defaultdict(deque)

    traj_rows = []
    mse_values = []
    frame_idx = 0

    paused = PAUSE_ON_START
    did_seek = False

    def _clamp(i:int)->int: return max(0, min(total_frames-1, i))

    def _reset_tracking_state():
        nonlocal tracker, track_paths, track_last_seen, track_cls, track_vel_hist
        tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)
        track_paths = {}; track_last_seen = {}; track_cls = {}; track_vel_hist = {}

    def _seek_to(target_idx:int):
        nonlocal frame_idx, did_seek
        frame_idx = _clamp(target_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _reset_tracking_state(); did_seek = True

    if PAUSE_ON_START: print("Paused. Press 'r' to resume, 'p' to pause again.")

    while True:
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
        # retro-fill virtual boxes created by ORU for missed frames
        for vb in tracker.virtual_boxes:
            pred_by_frame[vb["frame"]].append((int(vb["tid"]), vb["bbox"].astype(np.float32)))

        pred_centers = []

        for t in tracks:
            if t.hits < MIN_HITS and t.time_since_update != 0:
                continue
            box = _track_box_now(t)
            cx, cy, w, h = t.kf.x[:4, 0]
            if t.time_since_update > MAX_AGE or is_near_border(box, frame.shape):
                continue

            # log for HOTA
            pred_by_frame[frame_idx].append((int(t.id), box.astype(np.float32).copy()))

            pred_centers.append((float(cx), float(cy)))

            if t.id not in track_paths:
                track_paths[t.id] = deque(maxlen=TRAJ_MAX_LEN)
            track_paths[t.id].append((int(cx), int(cy)))
            track_last_seen[t.id] = frame_idx
            track_cls[t.id] = t.cls

            # velocity history on high-conf matches
            if t.matched_this_frame and t.high_conf_match:
                pred_by_frame[frame_idx].append((int(t.id), box.astype(np.float32).copy()))
                if t.id not in track_vel_hist:
                    track_vel_hist[t.id] = deque(maxlen=KINEMA_WINDOW)
                # KF velocity if available, else finite diff
                if HAS_FILTERPY:
                    vx = float(t.kf.x[4,0]); vy = float(t.kf.x[5,0])
                else:
                    hx = list(track_paths[t.id])[-2:]
                    if len(hx) >= 2:
                        (x0,y0),(x1,y1) = hx; vx = x1-x0; vy = y1-y0
                    else:
                        vx = vy = 0.0
                track_vel_hist[t.id].append((vx, vy))

            # mean kinematics
            hist = track_vel_hist.get(t.id, [])
            if len(hist) > 0:
                vx_mean = float(np.mean([v[0] for v in hist])); vy_mean = float(np.mean([v[1] for v in hist]))
                vx_last, vy_last = hist[-1]
                vx_blend = 0.7*vx_mean + 0.3*vx_last; vy_blend = 0.7*vy_mean + 0.3*vy_last
                speed_val = math.hypot(vx_blend, vy_blend)      # px/frame
                angle_val = math.degrees(math.atan2(vy_blend, vx_blend))
            else:
                speed_val = 0.0; angle_val = 0.0

            color = LOST_COLOR if (t.id in tracker.lost_ids) else CLASS_COLORS.get(t.cls, (200,200,200))
            tag = " LOST" if (t.id in tracker.lost_ids) else ""
            draw_box_id(vis_det, box, cls=t.cls, tid=t.id, conf=t.conf, color=color, label_map=CLASS_NAMES)
            if tag:
                cv2.putText(vis_det, tag, (int(box[0]), max(0, int(box[1])-20)), FONT, 0.5, color, 2, cv2.LINE_AA)
            draw_traj(vis_det, list(track_paths[t.id]), color=color)
            draw_direction_arrow(vis_det, cx, cy, angle_val, length=ARROW_LEN, color=color)
            comp = angle_to_compass((angle_val + 360) % 360)
            cv2.putText(vis_det, f"v:{speed_val:.1f} {SHOW_UNITS}  dir:{comp}",
                        (int(cx)+5, int(cy)+15), FONT, 0.5, color, 2, cv2.LINE_AA)

            traj_rows.append(["video3", t.id, frame_idx, float(cx), float(cy)])

        # prune stale trails
        stale_ids = [tid for tid,last_seen in list(track_last_seen.items())
                     if frame_idx - last_seen > MISS_FRAMES_TO_DROP_PATH]
        for tid in stale_ids:
            track_paths.pop(tid, None); track_last_seen.pop(tid, None)
            track_cls.pop(tid, None); track_vel_hist.pop(tid, None)

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
