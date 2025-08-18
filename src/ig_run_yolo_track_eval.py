# scripts/run_yolo_track_eval.py
# Detection (YOLO) + Tracking (SORT) + Trajectories CSV + MSE vs GT + Most-frequent path + Bonus reconstruction.
# Keys (when SHOW=True): p=pause, r=resume, o=next, i=prev, q=quit.

import os, csv, math, time, json, collections, random
from pathlib import Path
import numpy as np
import cv2

# =========================
# HYPERPARAMETERS (EDIT)
# =========================
# Inputs
VIDEO_INFOS = [
    {
        "name": "video0",
        "video_path": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4",
        "ann_path":   r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt",
    },
    {
        "name": "video3",
        "video_path": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4",
        "ann_path":   r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt",
    },
]

# Your trained YOLO weights
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

# YOLO inference
YOLO_CONF = 0.25
YOLO_IOU  = 0.70
YOLO_IMGSZ = 960
ALLOW_CLASSES = None  # e.g. {0,1} to keep only person/bike; None=all

# Tracker (SORT-like)
IOU_MATCH_THR = 0.2
MAX_AGE       = 30
MIN_HITS      = 3

# Output
OUT_DIR = r"C:\Users\morte\ComputerVisionProject\outputs_yolo_sort"
TRAJ_CSV_TMPL = "{video}_traj.csv"
REPORT_TXT_TMPL = "{video}_report.txt"
RECON_CSV_TMPL = "{video}_reconstruct_example.csv"

# Entry/Exit binning
BINS_PER_SIDE = 8

# Viewer
SHOW = False
LEFT_WIN  = "GT (rescaled)"
RIGHT_WIN = "YOLO+SORT"
WINDOW_W, WINDOW_H = 1280, 720
# =========================


# ---------- utils ----------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def iou_xyxy(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xx1, yy1 = max(ax1,bx1), max(ay1,by1)
    xx2, yy2 = min(ax2,bx2), min(ay2,by2)
    w, h = max(0, xx2-xx1), max(0, yy2-yy1)
    inter = w*h
    ua = max(1e-6, (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
    return inter/ua

def centers_from_boxes(boxes):
    if len(boxes)==0: return np.zeros((0,2), dtype=np.float32)
    b = np.array(boxes, dtype=np.float32)
    return np.stack([(b[:,0]+b[:,2])/2.0, (b[:,1]+b[:,3])/2.0], axis=1)

def linear_sum_assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        return lsa(cost)
    except Exception:
        # greedy fallback
        C = cost.copy()
        n, m = C.shape
        rows, cols = [], []
        for _ in range(min(n,m)):
            r,c = np.unravel_index(np.argmin(C), C.shape)
            rows.append(r); cols.append(c)
            C[r,:] = np.inf; C[:,c] = np.inf
        return np.array(rows), np.array(cols)

def parse_annotations_txt(path):
    """
    Each line: track_id xmin ymin xmax ymax frame lost occluded generated "label"
    Returns: dict[frame] -> list of dict(id, bbox(xyxy float), label, lost), and max extents for scaling.
    """
    per_frame = collections.defaultdict(list)
    max_x = 1.0; max_y = 1.0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().split()
            if len(s) < 10: continue
            tid = int(s[0])
            xmin,ymin,xmax,ymax = map(float, s[1:5])
            fr = int(s[5]); lost = int(s[6])
            label = " ".join(s[9:]).strip().strip('"')
            max_x = max(max_x, xmin, xmax); max_y = max(max_y, ymin, ymax)
            per_frame[fr].append(dict(id=tid, bbox=np.array([xmin,ymin,xmax,ymax], float), label=label, lost=lost))
    return per_frame, max_x, max_y

def draw_box(img, xyxy, color, txt=None, thick=2):
    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick, cv2.LINE_AA)
    if txt:
        cv2.putText(img, txt, (x1, max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def label_color(name):
    table = {
        "Pedestrian": (50,200,50),
        "Biker": (60,140,255),
        "Skater": (180,160,50),
        "Cart": (200,50,200),
        "Car": (60,60,220),
        "Bus": (0,120,200),
    }
    return table.get(name, (180,180,180))

# ---------- SORT ----------
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # state [cx, cy, s, r, vx, vy, vs]; measure [cx,cy,s,r]
        x1,y1,x2,y2 = bbox
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        cx, cy = x1+w/2.0, y1+h/2.0
        s, r = w*h, w/h

        self.kf = cv2.KalmanFilter(7,4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4,7, dtype=np.float32)
        self.kf.processNoiseCov   = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.statePost = np.array([[cx],[cy],[s],[r],[0],[0],[0]], dtype=np.float32)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 1
        self.no_losses = 0
        self.history = []

    def predict(self):
        self.kf.predict()
        self.no_losses += 1
        b = self.get_state()
        self.history.append(b)
        if len(self.history) > 32: self.history.pop(0)
        return b

    def update(self, bbox):
        x1,y1,x2,y2 = bbox
        w = max(1.0, x2-x1); h = max(1.0, y2-y1)
        cx, cy = x1+w/2.0, y1+h/2.0
        s, r = w*h, w/max(1.0,h)
        z = np.array([[cx],[cy],[s],[r]], dtype=np.float32)
        self.kf.correct(z)
        self.hits += 1
        self.no_losses = 0

    def get_state(self):
        cx, cy, s, r = self.kf.statePost[:4,0]
        w = math.sqrt(max(1.0, s*r))
        h = s / max(1.0, w)
        return np.array([cx-w/2.0, cy-h/2.0, cx+w/2.0, cy+h/2.0], dtype=float)

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_thresh=0.2):
        self.max_age=max_age; self.min_hits=min_hits; self.iou_thresh=iou_thresh
        self.trackers=[]

    def update(self, dets):  # dets: list of xyxy
        for t in self.trackers: t.predict()
        N, M = len(self.trackers), len(dets)
        matches=[]; unmatched_t=list(range(N)); unmatched_d=list(range(M))
        if N>0 and M>0:
            iou_mat = np.zeros((N,M), np.float32)
            for i,t in enumerate(self.trackers):
                tb = t.get_state()
                for j,d in enumerate(dets):
                    iou_mat[i,j] = iou_xyxy(tb, d)
            used_t, used_d = set(), set()
            for _ in range(min(N,M)):
                i,j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i,j] < self.iou_thresh or i in used_t or j in used_d:
                    iou_mat[i,j] = -1; continue
                matches.append((i,j)); used_t.add(i); used_d.add(j)
                iou_mat[i,:] = -1; iou_mat[:,j] = -1
            unmatched_t = [i for i in range(N) if i not in used_t]
            unmatched_d = [j for j in range(M) if j not in used_d]
        for i,j in matches: self.trackers[i].update(dets[j])
        for j in unmatched_d: self.trackers.append(KalmanBoxTracker(dets[j]))
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]
        outs=[]
        for t in self.trackers:
            if t.hits >= self.min_hits or t.no_losses == 0:
                outs.append((t.get_state(), t.id, list(t.history)))
        return outs

# ---------- YOLO ----------
def yolo_load(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def yolo_detect(model, frame):
    res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    dets=[]
    names = res.names if hasattr(res, "names") else {}
    if res.boxes is None: return dets, names
    for b in res.boxes:
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1
        if ALLOW_CLASSES is not None and cls_id not in ALLOW_CLASSES: continue
        x1,y1,x2,y2 = b.xyxy[0].tolist()
        dets.append([x1,y1,x2,y2])
    return dets, names

# ---------- Metrics ----------
def mse_rmse_vs_gt(traj_csv, ann_by_frame, sx, sy):
    # load predictions per frame
    pred_by = collections.defaultdict(list)
    with open(traj_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            fr = int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            pred_by[fr].append([x,y])

    se_sum=0.0; n=0
    common_frames = sorted(set(pred_by.keys()) & set(ann_by_frame.keys()))
    for fr in common_frames:
        P = np.array(pred_by[fr], dtype=np.float32)
        # GT centers from ann
        Gc=[]
        for item in ann_by_frame[fr]:
            if item["lost"]==1: continue
            x1,y1,x2,y2 = item["bbox"]
            x1*=sx; y1*=sy; x2*=sx; y2*=sy
            Gc.append([(x1+x2)/2.0, (y1+y2)/2.0])
        if not Gc: continue
        G = np.array(Gc, dtype=np.float32)
        # Hungarian on Euclidean
        D = np.linalg.norm(P[:,None,:]-G[None,:,:], axis=2)
        ri, ci = linear_sum_assignment(D)
        for r,c in zip(ri, ci):
            d = P[r]-G[c]
            se_sum += float(d[0]*d[0]+d[1]*d[1])
            n+=1
    mse = se_sum / max(1,n)
    rmse = math.sqrt(mse)
    return mse, rmse, n

def entry_exit_bins(traj_csv, W, H, K=8):
    per_track = collections.defaultdict(list)
    with open(traj_csv,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            tid=int(row["track_id"]); fr=int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            per_track[tid].append((fr,x,y))
    def bin_point(x,y):
        d = [y, W-x, H-y, x]  # top,right,bottom,left dist
        side = int(np.argmin(d))
        if side==0:  pos=x/W;  return ('T', min(K-1, max(0,int(pos*K))))
        if side==1:  pos=y/H;  return ('R', min(K-1, max(0,int(pos*K))))
        if side==2:  pos=x/W;  return ('B', min(K-1, max(0,int(pos*K))))
        else:        pos=y/H;  return ('L', min(K-1, max(0,int(pos*K))))
    cnt=collections.Counter()
    for tid, pts in per_track.items():
        pts.sort(key=lambda t:t[0])
        x0,y0=pts[0][1], pts[0][2]
        x1,y1=pts[-1][1], pts[-1][2]
        e=bin_point(x0,y0); x=bin_point(x1,y1)
        if e and x: cnt[(e,x)]+=1
    top3 = cnt.most_common(3)
    return cnt, top3

# ---------- Bonus reconstruction ----------
def build_library_from_gt(ann_by_frame, sx, sy, W, H, K=8, resample_n=20):
    # collect GT trajectories per (entry,exit)
    per_track = collections.defaultdict(list)  # track_id -> [(fr,x,y)]
    # infer track id set
    for fr, items in ann_by_frame.items():
        for it in items:
            if it["lost"]==1: continue
            x1,y1,x2,y2 = it["bbox"]; x1*=sx; y1*=sy; x2*=sx; y2*=sy
            cx,cy=(x1+x2)/2.0,(y1+y2)/2.0
            per_track[it["id"]].append((fr,cx,cy))
    def bin_point(x,y):
        d=[y, W-x, H-y, x]
        side=int(np.argmin(d))
        if side==0: pos=x/W;  return ('T', int(min(K-1,max(0,int(pos*K)))))
        if side==1: pos=y/H;  return ('R', int(min(K-1,max(0,int(pos*K)))))
        if side==2: pos=x/W;  return ('B', int(min(K-1,max(0,int(pos*K)))))
        return ('L', int(min(K-1,max(0,int(pos*K)))))
    def resample(poly, n):
        if len(poly)<2: return [poly[0]]*n
        # uniform along index
        xs=np.array([p[0] for p in poly]); ys=np.array([p[1] for p in poly])
        idx=np.linspace(0,len(poly)-1,n)
        xsr=np.interp(idx, np.arange(len(poly)), xs)
        ysr=np.interp(idx, np.arange(len(poly)), ys)
        return list(zip(xsr,ysr))
    lib = collections.defaultdict(list)  # (e,x) -> [traj of Nx2]
    for tid, pts in per_track.items():
        pts.sort(key=lambda t:t[0])
        xs=[p[1] for p in pts]; ys=[p[2] for p in pts]
        e=bin_point(xs[0],ys[0]); x=bin_point(xs[-1],ys[-1])
        traj = resample(list(zip(xs,ys)), resample_n)
        lib[(e,x)].append(traj)
    # centroids
    centroids={}
    for k, L in lib.items():
        A = np.array(L)  # M x n x 2
        centroids[k] = A.mean(axis=0)  # n x 2
    return centroids  # dict[(e,x)] -> n x 2

def reconstruct_segment(start_xy, end_xy, entry_exit_key, lib_centroids, n_points=20):
    # choose centroid path for the pair, inject start/end
    C = lib_centroids.get(entry_exit_key)
    if C is None:
        # simple quadratic Bezier with control at midpoint
        p0 = np.array(start_xy); p2 = np.array(end_xy)
        pc = (p0+p2)/2.0
        ts=np.linspace(0,1,n_points)
        out=[]
        for t in ts:
            q=(1-t)*(1-t)*p0 + 2*(1-t)*t*pc + t*t*p2
            out.append((float(q[0]), float(q[1])))
        return out
    # use centroid midpoints as guidance (skip first/last)
    mid = C[1:-1]
    path = [start_xy] + [(float(x),float(y)) for (x,y) in mid] + [end_xy]
    return path

# ---------- main per video ----------
def process_video(info):
    name = info["name"]
    vpath = info["video_path"]; apath = info["ann_path"]
    ensure_dir(OUT_DIR)
    traj_csv = os.path.join(OUT_DIR, TRAJ_CSV_TMPL.format(video=name))
    report_txt = os.path.join(OUT_DIR, REPORT_TXT_TMPL.format(video=name))
    recon_csv = os.path.join(OUT_DIR, RECON_CSV_TMPL.format(video=name))

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened(): raise RuntimeError("Cannot open video: "+vpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # GT
    ann_by_frame, max_ax, max_ay = parse_annotations_txt(apath)
    sx = W / (max_ax if max_ax>0 else 1.0)
    sy = H / (max_ay if max_ay>0 else 1.0)

    # YOLO + SORT
    yolo = yolo_load(YOLO_WEIGHTS)
    tracker = Sort(MAX_AGE, MIN_HITS, IOU_MATCH_THR)

    if SHOW:
        cv2.namedWindow(LEFT_WIN, cv2.WINDOW_NORMAL)
        cv2.namedWindow(RIGHT_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(LEFT_WIN, WINDOW_W, WINDOW_H)
        cv2.resizeWindow(RIGHT_WIN, WINDOW_W, WINDOW_H)
        paused=False; frame_idx=0; cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # write trajectories
    with open(traj_csv, "w", newline="") as fcsv:
        wr = csv.writer(fcsv); wr.writerow(["track_id","frame","x","y"])
        while True:
            if SHOW:
                if not paused:
                    ok, frame = cap.read()
                    if not ok: break
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                else:
                    ok, frame = cap.read()
                    if not ok: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            else:
                ok, frame = cap.read()
                if not ok: break
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            t0=time.time()
            dets, names = yolo_detect(yolo, frame)
            tracks = tracker.update(dets)

            # write centers
            for (box, tid, hist) in tracks:
                cx=(box[0]+box[2])/2.0; cy=(box[1]+box[3])/2.0
                wr.writerow([tid, frame_idx, f"{cx:.2f}", f"{cy:.2f}"])

            if SHOW:
                # left = GT
                left = frame.copy()
                for item in ann_by_frame.get(frame_idx, []):
                    if item["lost"]==1: continue
                    x1,y1,x2,y2 = item["bbox"]
                    x1*=sx; y1*=sy; x2*=sx; y2*=sy
                    draw_box(left, (x1,y1,x2,y2), label_color(item["label"]), f'{item["label"]}#{item["id"]}', 2)
                # right = detections + tracks
                right = frame.copy()
                for d in dets:
                    draw_box(right, d, (0,255,255), "det", 1)
                for (box, tid, hist) in tracks:
                    draw_box(right, box, (0,255,0), f"id{tid}", 2)
                    if len(hist)>=2:
                        pts=[(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in hist]
                        for i in range(1,len(pts)):
                            cv2.line(right, pts[i-1], pts[i], (0,200,0), 2)
                fps=1.0/max(1e-6,time.time()-t0)
                cv2.putText(left, f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(right, f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}", (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(LEFT_WIN, cv2.resize(left,(WINDOW_W,WINDOW_H)))
                cv2.imshow(RIGHT_WIN, cv2.resize(right,(WINDOW_W,WINDOW_H)))
                k=cv2.waitKey(1 if not paused else 0)&0xFF
                if k in (ord('q'),27): break
                elif k==ord('p'): paused=True
                elif k==ord('r'): paused=False
                elif paused and k==ord('o'):
                    frame_idx=min(total-1, frame_idx+1); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                elif paused and k==ord('i'):
                    frame_idx=max(0, frame_idx-1); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    # evaluation
    mse, rmse, n_pairs = mse_rmse_vs_gt(traj_csv, ann_by_frame, sx, sy)
    counts, top3 = entry_exit_bins(traj_csv, W, H, K=BINS_PER_SIDE)

    # bonus: reconstruct example between first and last point of a short track
    lib = build_library_from_gt(ann_by_frame, sx, sy, W, H, K=BINS_PER_SIDE, resample_n=20)
    # choose a random track from predictions for demo
    per_track = collections.defaultdict(list)
    with open(traj_csv,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            tid=int(row["track_id"]); fr=int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            per_track[tid].append((fr,x,y))
    demo=None
    for tid, pts in per_track.items():
        pts.sort(key=lambda t:t[0])
        if len(pts) >= 6:
            demo=(tid, pts[0], pts[-1]); break
    if demo:
        _, s, e = demo
        def bin_point(x,y):
            d=[y, W-x, H-y, x]
            side=int(np.argmin(d))
            if side==0: pos=x/W;  return ('T', int(min(BINS_PER_SIDE-1,max(0,int(pos*BINS_PER_SIDE)))))
            if side==1: pos=y/H;  return ('R', int(min(BINS_PER_SIDE-1,max(0,int(pos*BINS_PER_SIDE)))))
            if side==2: pos=x/W;  return ('B', int(min(BINS_PER_SIDE-1,max(0,int(pos*BINS_PER_SIDE)))))
            return ('L', int(min(BINS_PER_SIDE-1,max(0,int(pos*BINS_PER_SIDE)))))
        ek = (bin_point(s[1],s[2]), bin_point(e[1],e[2]))
        recon = reconstruct_segment((s[1],s[2]), (e[1],e[2]), ek, lib, n_points=20)
        with open(recon_csv,"w",newline="") as f:
            wr=csv.writer(f); wr.writerow(["x","y"]); wr.writerows([[f"{x:.2f}",f"{y:.2f}"] for (x,y) in recon])

    # report
    with open(report_txt,"w") as f:
        f.write(f"Video: {name}\n")
        f.write(f"Frames matched (pairs): {n_pairs}\n")
        f.write(f"MSE (px^2): {mse:.2f}\n")
        f.write(f"RMSE (px): {rmse:.2f}\n")
        f.write(f"Entry/Exit bins K={BINS_PER_SIDE}\nTop-3 paths:\n")
        for ((e,x), cnt) in top3:
            f.write(f"  {e[0]}{e[1]} -> {x[0]}{x[1]} : {cnt}\n")
        if demo:
            f.write(f"\nBonus reconstruction saved: {os.path.basename(recon_csv)}\n")

    cap.release()
    if SHOW: cv2.destroyAllWindows()
    print(f"[OK] {name}: traj -> {traj_csv} | report -> {report_txt}")

# ---------- run both ----------
def main():
    ensure_dir(OUT_DIR)
    for info in VIDEO_INFOS:
        process_video(info)

if __name__ == "__main__":
    main()
