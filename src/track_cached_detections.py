# track_cached_detections.py — ByteTrack(+ORU) + HOTA eval
import cv2, numpy as np, pandas as pd, math
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment

# ========= PATHS =========
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
DET_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\detected\video3_detections.parquet"
ANNOT_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"  # "" to disable
TRAJ_CSV   = r"C:\Users\morte\ComputerVisionProject\outputs\video3_tracks_byte_oru.csv"
HOTA_CSV   = r"C:\Users\morte\ComputerVisionProject\outputs\video3_hota_breakdown.csv"

# ========= CLASSES =========
CLASS_NAMES = {0:"Pedestrian",1:"Biker",2:"Skater",3:"Cart",4:"Car",5:"Bus"}
LABEL_TO_ID = {"Pedestrian":0,"Biker":1,"Skater":2,"Cart":3,"Car":4,"Bus":5}
CLASS_COLORS= {0:(0,255,0),1:(255,0,0),2:(0,0,255),3:(255,255,0),4:(255,0,255),5:(0,255,255)}

# ========= DET/NMS =========
DET_CONF_THRES = 0.50
DET_IOU_NMS    = 0.60
AGNOSTIC_NMS   = False

# ========= BYTETRACK CORE =========
BYTE_HIGH_THRES = 0.68
BYTE_LOW_THRES  = 0.58
IOU_GATE        = 0.10
MAX_AGE         = 30
MIN_HITS        = 3

# ========= OC-SORT STYLE EXTRAS =========
ORU_MAX_GAP   = 10
REVIVE_DIST   = 40.0
LOST_MAX_AGE  = 100
BORDER_MARGIN = 5

# ========= UI & DRAW =========
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICK = 2
TRAJ_MAX_LEN = 2000
SHOW_IDS_ONLY_AFTER_MIN_HITS = True

# ========= HELPERS =========
def draw_txt(img, txt, org, col=(255,255,255)):
    cv2.putText(img, txt, org, FONT, FONT_SCALE, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, FONT_SCALE, col, 1, cv2.LINE_AA)

def iou_xyxy(a, b):
    # a: (N,4) [x1,y1,x2,y2], b: (M,4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    x11 = a[:, 0][:, None]; y11 = a[:, 1][:, None]
    x12 = a[:, 2][:, None]; y12 = a[:, 3][:, None]
    x21 = b[:, 0][None, :]; y21 = b[:, 1][None, :]
    x22 = b[:, 2][None, :]; y22 = b[:, 3][None, :]

    iw = np.maximum(0.0, np.minimum(x12, x22) - np.maximum(x11, x21))
    ih = np.maximum(0.0, np.minimum(y12, y22) - np.maximum(y11, y21))
    inter = iw * ih

    area_a = np.maximum(0.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = np.maximum(0.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter + 1e-9

    return (inter / union).astype(np.float32)


def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(0.0, x2-x1); h = max(0.0, y2-y1)
    return np.array([x1+w/2.0, y1+h/2.0, w, h], np.float32)

def cxcywh_to_xyxy(b):
    cx,cy,w,h = b
    return np.array([cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0], np.float32)

def center_of(b):
    return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)

def near_border(box, shape, margin=BORDER_MARGIN):
    H, W = shape[:2]
    cx, cy = center_of(box)
    return cx<margin or cy<margin or cx>W-margin or cy>H-margin

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
    if rows.size == 0: return []
    fr = rows[rows[:,5] >= conf_thres]
    if fr.size == 0: return []
    out = []
    if agnostic:
        boxes = fr[:,1:5].astype(np.float32)
        scores = fr[:,5].astype(np.float32)
        keep = iou_nms_xyxy(boxes, scores, iou_thres)
        for i in keep:
            out.append({"bbox":boxes[i], "conf":float(scores[i]), "cls":int(fr[i,6])})
    else:
        for c in np.unique(fr[:,6]).astype(int):
            sc = fr[fr[:,6]==c]
            boxes = sc[:,1:5].astype(np.float32)
            scores= sc[:,5].astype(np.float32)
            keep = iou_nms_xyxy(boxes, scores, iou_thres)
            for i in keep:
                out.append({"bbox":boxes[i], "conf":float(scores[i]), "cls":int(c)})
    return out

def load_dets(path):
    try: df = pd.read_parquet(path)
    except Exception: df = pd.read_csv(path)
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
            frames[fr].append({"id":tid, "bbox":[x1,y1,x2,y2], "lost":lost, "occluded":occl, "label":label})
            max_x = max(max_x, x2); max_y = max(max_y, y2)
    return frames, (int(math.ceil(max_x)), int(math.ceil(max_y)))

def scale_bbox(b, sx, sy):
    x1,y1,x2,y2 = b
    return [x1*sx, y1*sy, x2*sx, y2*sy]

# ========= TRACKER =========
class Track:
    _next_id = 1
    def __init__(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.id = Track._next_id; Track._next_id += 1
        self.cls = int(cls_id); self.conf = float(conf)
        self.hits = 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.state = xyxy_to_cxcywh(bbox_xyxy.astype(np.float32))
        self.history = deque(maxlen=TRAJ_MAX_LEN)  # (frame, x, y, is_virtual)
        cx, cy, _, _ = self.state
        self.history.append((frame_idx, int(cx), int(cy), False))

    def predict_xyxy(self):
        return cxcywh_to_xyxy(self.state)

    def update(self, bbox_xyxy, cls_id, conf, frame_idx):
        self.state = xyxy_to_cxcywh(bbox_xyxy.astype(np.float32))
        self.cls = int(cls_id); self.conf = float(conf)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        cx, cy, _, _ = self.state
        self.history.append((frame_idx, int(cx), int(cy), False))

    def mark_missed(self):
        self.time_since_update += 1

class ByteTrackLike:
    def __init__(self, iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS):
        self.iou_gate = iou_gate
        self.max_age = max_age
        self.min_hits = min_hits
        self.active = []
        self.lost = {}   # id -> (Track, lost_age)

    def _match(self, tracks, dets):
        if len(tracks)==0 or len(dets)==0:
            return [], list(range(len(tracks))), list(range(len(dets)))
        t_boxes = np.array([t.predict_xyxy() for t in tracks], np.float32)
        d_boxes = np.array([d["bbox"] for d in dets], np.float32)
        iou = iou_xyxy(t_boxes, d_boxes)
        cost = 1.0 - iou
        r, c = linear_sum_assignment(cost)
        matches, un_t, un_d = [], [], []
        rset, cset = set(r.tolist()), set(c.tolist())
        for i in range(len(tracks)):
            if i not in rset: un_t.append(i)
        for j in range(len(dets)):
            if j not in cset: un_d.append(j)
        for i, j in zip(r, c):
            if iou[i, j] >= self.iou_gate: matches.append((i, j))
            else: un_t.append(i); un_d.append(j)
        return matches, un_t, un_d

    def _try_revive(self, dets, frame_idx):
        revived = []
        if not self.lost or not dets: return revived
        for j, d in list(enumerate(dets)):
            db = d["bbox"]; dcx, dcy = center_of(db)
            best_id, best_dist = None, 1e9
            for lid, (lt, age) in list(self.lost.items()):
                if age > LOST_MAX_AGE: continue
                tcx, tcy, _, _ = lt.state
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist < best_dist and dist <= REVIVE_DIST:
                    best_dist, best_id = dist, lid
            if best_id is not None:
                trk, _ = self.lost.pop(best_id)
                gap = frame_idx - trk.last_frame - 1
                if 0 < gap <= ORU_MAX_GAP:
                    scx, scy = trk.history[-1][1], trk.history[-1][2]
                    ecx, ecy = int(dcx), int(dcy)
                    for k in range(1, gap+1):
                        tframe = trk.last_frame + k
                        alpha = k/(gap+1.0)
                        ix = int((1-alpha)*scx + alpha*ecx)
                        iy = int((1-alpha)*scy + alpha*ecy)
                        trk.history.append((tframe, ix, iy, True))
                trk.update(db, d["cls"], d["conf"], frame_idx)
                self.active.append(trk)
                revived.append(j)
        return revived

    def update(self, detections, frame_idx, frame_shape):
        H, W = frame_shape[:2]
        high = [d for d in detections if d["conf"] >= BYTE_HIGH_THRES]
        low  = [d for d in detections if BYTE_LOW_THRES <= d["conf"] < BYTE_HIGH_THRES]

        matches, un_t, un_h = self._match(self.active, high)
        matched_idx = set()
        for ti, dj in matches:
            t = self.active[ti]; d = high[dj]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
            matched_idx.add(ti)

        remain_tracks = [self.active[i] for i in un_t]
        m2, un_t2, un_l = self._match(remain_tracks, low)
        for lti, dj in m2:
            gi = un_t[lti]
            t = self.active[gi]; d = low[dj]
            t.update(d["bbox"], d["cls"], d["conf"], frame_idx)
            matched_idx.add(gi)

        uh = [high[j] for j in un_h]
        revived_h = self._try_revive(uh, frame_idx)
        for idx in sorted(revived_h, reverse=True): uh.pop(idx)

        for d in uh:
            self.active.append(Track(d["bbox"], d["cls"], d["conf"], frame_idx))

        survivors = []
        for i, t in enumerate(self.active):
            if i not in matched_idx:
                t.mark_missed()
            if t.time_since_update <= self.max_age:
                survivors.append(t)
            else:
                if not near_border(t.predict_xyxy(), (H,W), BORDER_MARGIN):
                    self.lost[t.id] = (t, 0)
        self.active = survivors

        for lid in list(self.lost.keys()):
            trk, age = self.lost[lid]
            age += 1
            if age > LOST_MAX_AGE:
                self.lost.pop(lid, None)
            else:
                self.lost[lid] = (trk, age)

        return self.active

# ========= HOTA EVALUATION =========
def build_gt_by_frame(frames_gt, sx, sy):
    out = {}
    for f, lst in frames_gt.items():
        cur = []
        for ann in lst:
            if ann["lost"]==1 or ann["occluded"]==1: 
                continue
            bb = scale_bbox(ann["bbox"], sx, sy)
            cur.append((int(ann["id"]), np.array(bb, np.float32)))
        if cur:
            out[int(f)] = cur
    return out

def match_frame(g_ids, g_boxes, t_ids, t_boxes, tau):
    if len(g_ids)==0 or len(t_ids)==0:
        return [], list(range(len(g_ids))), list(range(len(t_ids)))
    iou = iou_xyxy(g_boxes, t_boxes)
    # block pairs below tau
    cost = 1.0 - iou
    cost[iou < tau] = 1e6
    r, c = linear_sum_assignment(cost)
    matches, un_g, un_t = [], [], []
    rset, cset = set(r.tolist()), set(c.tolist())
    for i in range(len(g_ids)):
        if i not in rset: un_g.append(i)
    for j in range(len(t_ids)):
        if j not in cset: un_t.append(j)
    for i, j in zip(r, c):
        if iou[i, j] >= tau:
            matches.append((i, j))
        else:
            un_g.append(i); un_t.append(j)
    return matches, un_g, un_t

def eval_hota(gt_by_frame, pred_by_frame, total_frames, taus=None):
    if taus is None:
        taus = [i/20 for i in range(1,20)]  # 0.05..0.95

    results = []
    for tau in taus:
        # per-frame matching maps
        g2t = {}
        t2g = {}
        TP = 0; FP = 0; FN = 0

        for f in range(total_frames):
            g_list = gt_by_frame.get(f, [])
            t_list = pred_by_frame.get(f, [])
            if not g_list and not t_list:
                continue
            g_ids = [gid for gid,_ in g_list]
            g_boxes = np.array([b for _,b in g_list], np.float32)
            t_ids = [tid for tid,_ in t_list]
            t_boxes = np.array([b for _,b in t_list], np.float32)

            m, ug, ut = match_frame(g_ids, g_boxes, t_ids, t_boxes, tau)
            TP += len(m); FP += len(ut); FN += len(ug)

            if m:
                g2t[f] = { g_ids[i]: t_ids[j] for i,j in m }
                t2g[f] = { t_ids[j]: g_ids[i] for i,j in m }
            else:
                g2t[f] = {}
                t2g[f] = {}

        # DetA
        det_den = TP + 0.5*(FP + FN)
        DetA = (TP / det_den) if det_den>0 else 0.0

        # association for pairs seen at least once
        pairs = set()
        for f in g2t:
            for gid, tid in g2t[f].items():
                pairs.add((gid, tid))

        pair_acc = []
        for gid, tid in pairs:
            IDTP = 0; IDFP = 0; IDFN = 0
            for f in range(total_frames):
                g_present = any((gid == g for g,_ in gt_by_frame.get(f, [])))
                t_present = any((tid == t for t,_ in pred_by_frame.get(f, [])))
                if g_present and t_present:
                    mt = g2t.get(f, {}).get(gid, None)
                    if mt == tid:
                        IDTP += 1
                    else:
                        IDFP += 1
                        IDFN += 1
                elif g_present and not t_present:
                    IDFN += 1
                elif t_present and not g_present:
                    IDFP += 1
            denom = IDTP + 0.5*(IDFP + IDFN)
            if denom > 0:
                pair_acc.append(IDTP / denom)

        AssA = float(np.mean(pair_acc)) if pair_acc else 0.0
        HOTA = math.sqrt(max(0.0, DetA) * max(0.0, AssA))

        results.append({"tau":tau, "DetA":DetA, "AssA":AssA, "HOTA":HOTA})

    df = pd.DataFrame(results)
    return df, float(df["DetA"].mean()), float(df["AssA"].mean()), float(df["HOTA"].mean())

# ========= MAIN =========
def main():
    dets = load_dets(DET_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_gt = bool(ANNOT_PATH)
    if use_gt:
        frames_gt, (Wref, Href) = parse_sdd_annotations(ANNOT_PATH)
        sx = W / float(Wref if Wref>0 else W); sy = H / float(Href if Href>0 else H)
        gt_by_frame = build_gt_by_frame(frames_gt, sx, sy)
    else:
        gt_by_frame = {}

    tracker = ByteTrackLike(iou_gate=IOU_GATE, max_age=MAX_AGE, min_hits=MIN_HITS)

    f = 0
    conf_th = DET_CONF_THRES
    iou_th  = DET_IOU_NMS
    agn     = AGNOSTIC_NMS

    rows_out = []   # video_id, track_id, frame, x, y, virtual
    pred_by_frame = defaultdict(list)  # frame -> list of (track_id, bbox)

    print("Controls: +/- conf | [/] IoU | a=agnostic | c=class-wise | q=quit")

    while True:
        ok, frame = cap.read()
        if not ok: break

        # cached dets + NMS
        rows = dets.get(f, np.empty((0,7), np.float32))
        det_list = nms_frame(rows, conf_th, iou_th, agn)

        # tracking
        tracks = tracker.update(det_list, f, frame.shape)

        # draw + record predictions
        for t in tracks:
            if SHOW_IDS_ONLY_AFTER_MIN_HITS and (t.hits < MIN_HITS and t.time_since_update>0):
                continue
            box = t.predict_xyxy()
            if near_border(box, frame.shape): 
                continue

            cx, cy, _, _ = t.state
            color = CLASS_COLORS.get(t.cls, (200,200,200))
            x1,y1,x2,y2 = map(int, box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{t.id} {t.conf:.2f}", (x1, max(0,y1-6)), FONT, 0.5, color, 1, cv2.LINE_AA)

            pred_by_frame[f].append((int(t.id), box.copy()))

            rows_out.append(["video3", t.id, f, float(cx), float(cy), 0])
            hpts = [(x,y) for (_fr,x,y,_v) in t.history]
            for k in range(1, len(hpts)):
                cv2.line(frame, hpts[k-1], hpts[k], color, 2)

        cv2.imshow("Cached dets -> ByteTrack (+ORU) tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in (ord('+'), ord('=')): conf_th = min(0.99, conf_th + 0.02)
        elif key in (ord('-'), ord('_')): conf_th = max(0.00, conf_th - 0.02)
        elif key == ord('['): iou_th = max(0.00, iou_th - 0.02)
        elif key == ord(']'): iou_th = min(0.99, iou_th + 0.02)
        elif key == ord('a'): agn = True
        elif key == ord('c'): agn = False

        f += 1

    cap.release()
    cv2.destroyAllWindows()

    if rows_out:
        df = pd.DataFrame(rows_out, columns=["video_id","track_id","frame","x","y","virtual"])
        df.to_csv(TRAJ_CSV, index=False)
        print(f"Trajectories saved: {TRAJ_CSV}")

    # HOTA evaluation
    if gt_by_frame:
        taus = [i/20 for i in range(1,20)]  # 0.05..0.95
        df_hota, mean_DetA, mean_AssA, mean_HOTA = eval_hota(gt_by_frame, pred_by_frame, f, taus)
        df_hota.to_csv(HOTA_CSV, index=False)
        print(f"HOTA saved: {HOTA_CSV}")
        print(f"Mean DetA: {mean_DetA:.4f}  Mean AssA: {mean_AssA:.4f}  Mean HOTA: {mean_HOTA:.4f}")

if __name__ == "__main__":
    main()
