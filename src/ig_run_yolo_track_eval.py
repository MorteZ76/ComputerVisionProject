# scripts/yolo_track_eval_paths.py
# Detect → Track → Save trajectories → MSE vs GT → Most-frequent entry→exit → Bonus gap reconstruction.
# Keys (viewer): p pause, r resume, o next, i prev, q quit.

import os, sys, csv, math, time, json
import numpy as np
import cv2
from collections import defaultdict, Counter, deque

# =======================
# HYPERPARAMETERS (EDIT)
# =======================
VIDEO_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
ANN_PATH     = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"  # GT txt
YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

# Annotation original size if known; else auto from max coords
ANNOT_ORIG_SIZE = None  # (W,H) or None

# Detector
YOLO_IMGSZ = 960
YOLO_CONF  = 0.25
YOLO_IOU   = 0.70
ALLOW_CLASSES = None  # e.g., {0,1} or None = all classes from the trained model

# Tracker (BYTE-like over SORT)
BYTE_HIGH_THR = 0.5
BYTE_LOW_THR  = 0.1
SORT_IOU_GATE = 0.2
SORT_MAX_AGE  = 30
SORT_MIN_HITS = 3

# Output
OUT_DIR       = r"C:\Users\morte\ComputerVisionProject\outputs"
RUN_TAG       = "video0_yolo_byte"
TRAJ_CSV      = os.path.join(OUT_DIR, f"{RUN_TAG}_traj.csv")
REPORT_TXT    = os.path.join(OUT_DIR, f"{RUN_TAG}_report.txt")

# Entry/exit
BINS_PER_EDGE = 8

# Viewer
SHOW_VIEWER   = True
LEFT_WIN      = "GT (rescaled)"
RIGHT_WIN     = "YOLO+BYTE"
WIN_W, WIN_H  = 1280, 720
THICK         = 2
FS            = 0.5
# =======================

# ---------- utils ----------
def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)

def iou_xyxy(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0,x2-x1); ih=max(0,y2-y1); inter=iw*ih
    ua=max(1e-6,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    return inter/ua

def centers_from_boxes(boxes):
    if len(boxes)==0: return np.zeros((0,2),dtype=np.float32)
    b=np.asarray(boxes,float)
    return np.stack([(b[:,0]+b[:,2])*0.5,(b[:,1]+b[:,3])*0.5],axis=1)

def linear_sum(cost):
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        C=cost.copy(); n,m=C.shape; rows=[]; cols=[]
        for _ in range(min(n,m)):
            r,c=np.unravel_index(np.argmin(C),C.shape)
            rows.append(r); cols.append(c)
            C[r,:]=np.inf; C[:,c]=np.inf
        return np.array(rows), np.array(cols)

def parse_annotations(path, out_W, out_H):
    per_frame=defaultdict(list); maxx=1.0; maxy=1.0
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            s=ln.strip().split()
            if len(s)<10: continue
            tid=int(s[0]); x1=float(s[1]); y1=float(s[2]); x2=float(s[3]); y2=float(s[4])
            fr=int(s[5]); lost=int(s[6]); label=" ".join(s[9:]).strip().strip('"')
            maxx=max(maxx,x1,x2); maxy=max(maxy,y1,y2)
            per_frame[fr].append(dict(id=tid, bbox=np.array([x1,y1,x2,y2],float), lost=lost, label=label))
    if ANNOT_ORIG_SIZE:
        sx=out_W/float(ANNOT_ORIG_SIZE[0]); sy=out_H/float(ANNOT_ORIG_SIZE[1])
    else:
        sx=out_W/maxx; sy=out_H/maxy
    # scale now
    for fr, lst in per_frame.items():
        for it in lst:
            b=it["bbox"]; it["bbox"]=np.array([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],float)
    return per_frame

def color_for_label(name):
    table={"Pedestrian":(50,200,50),"Biker":(60,140,255),"Skater":(180,160,50),"Cart":(200,50,200),"Car":(60,60,220),"Bus":(0,120,200)}
    if name=="Truck": name="Car"
    return table.get(name,(180,180,180))

def draw_box(img, b, color, txt=None):
    x1,y1,x2,y2=map(int,b)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,THICK,cv2.LINE_AA)
    if txt: cv2.putText(img,txt,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,FS,color,2,cv2.LINE_AA)

# ---------- YOLO ----------
def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def yolo_detect(model, frame):
    r=model.predict(frame,imgsz=YOLO_IMGSZ,conf=YOLO_CONF,iou=YOLO_IOU,verbose=False)[0]
    out=[]
    names=getattr(r,"names",{})
    if r.boxes is None: return out,names
    for b in r.boxes:
        cls=int(b.cls[0].item()) if b.cls is not None else -1
        if ALLOW_CLASSES is not None and cls not in ALLOW_CLASSES: continue
        x1,y1,x2,y2=b.xyxy[0].tolist(); sc=float(b.conf[0].item()) if b.conf is not None else 0.0
        out.append([x1,y1,x2,y2,sc,cls])
    return out,names

# ---------- SORT/BYTE-like ----------
class KfTrack:
    _next=1
    def __init__(self,b):
        self.id=KfTrack._next; KfTrack._next+=1
        self.kf=cv2.KalmanFilter(7,4)
        self.kf.transitionMatrix=np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],np.float32)
        self.kf.measurementMatrix=np.eye(4,7,dtype=np.float32)
        self.kf.processNoiseCov=np.eye(7,dtype=np.float32)*1e-2
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float32)*1e-1
        x1,y1,x2,y2=b; w=max(1.0,x2-x1); h=max(1.0,y2-y1); cx=x1+w/2; cy=y1+h/2; s=w*h; r=w/h
        self.kf.statePost=np.array([[cx],[cy],[s],[r],[0],[0],[0]],np.float32)
        self.hits=1; self.miss=0; self.hist=[]
    def predict(self):
        self.kf.predict(); self.miss+=1
        self.hist.append(self.get_box()); 
        if len(self.hist)>32: self.hist.pop(0)
    def update(self,b):
        x1,y1,x2,y2=b; w=max(1.0,x2-x1); h=max(1.0,y2-y1); cx=x1+w/2; cy=y1+h/2; s=w*h; r=w/max(1.0,h)
        z=np.array([[cx],[cy],[s],[r]],np.float32); self.kf.correct(z); self.hits+=1; self.miss=0
    def get_box(self):
        cx,cy,s,r=self.kf.statePost[:4,0]; w=math.sqrt(max(1.0,s*r)); h=max(1.0,s/max(1.0,w))
        return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],float)

class Sort:
    def __init__(self,iou_thr=0.2,max_age=30,min_hits=3):
        self.iou_thr=iou_thr; self.max_age=max_age; self.min_hits=min_hits; self.tracks=[]
    def update(self,dets):  # dets: list xyxy
        for t in self.tracks: t.predict()
        N=len(self.tracks); M=len(dets)
        matches=[]; ut=list(range(N)); ud=list(range(M))
        if N>0 and M>0:
            mat=np.zeros((N,M),np.float32)
            for i,t in enumerate(self.tracks):
                tb=t.get_box()
                for j,d in enumerate(dets): mat[i,j]=iou_xyxy(tb,d)
            used_t=set(); used_d=set()
            for _ in range(min(N,M)):
                i,j=np.unravel_index(np.argmax(mat),mat.shape)
                if mat[i,j]<self.iou_thr or i in used_t or j in used_d:
                    mat[i,j]=-1; continue
                matches.append((i,j)); used_t.add(i); used_d.add(j); mat[i,:]=-1; mat[:,j]=-1
            ut=[i for i in range(N) if i not in used_t]; ud=[j for j in range(M) if j not in used_d]
        for i,j in matches: self.tracks[i].update(dets[j])
        for j in ud: self.tracks.append(KfTrack(dets[j]))
        self.tracks=[t for t in self.tracks if t.miss<=self.max_age]
        out=[]
        for t in self.tracks:
            if t.hits>=self.min_hits or t.miss==0: out.append((t.get_box(),t.id,list(t.hist)))
        return out

class ByteLike:
    def __init__(self): self.sort=Sort(SORT_IOU_GATE,SORT_MAX_AGE,SORT_MIN_HITS)
    def step(self,dets_xyxyscore):
        if len(dets_xyxyscore)==0:
            return self.sort.update([])
        dets=np.array([d[:4] for d in dets_xyxyscore],float)
        scrs=np.array([d[4] for d in dets_xyxyscore],float)
        high=dets[scrs>=BYTE_HIGH_THR].tolist()
        low =dets[(scrs<BYTE_HIGH_THR)&(scrs>=BYTE_LOW_THR)].tolist()
        self.sort.update(high)
        self.sort.update(low)
        return self.sort.update([])

# ---------- metrics ----------
def mse_rmse_vs_gt(traj_csv, gt_by_frame):
    # pred: frame,x,y per row id-agnostic via per-frame Hungarian on centers
    P=defaultdict(list)
    with open(traj_csv,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            fr=int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            P[fr].append([x,y])
    se=0.0; n=0
    for fr in sorted(set(P.keys()) & set(gt_by_frame.keys())):
        p=np.array(P[fr],float); g=np.array(gt_by_frame[fr],float)
        if len(p)==0 or len(g)==0: continue
        D=np.linalg.norm(p[:,None,:]-g[None,:,:],axis=2)
        ri,ci=linear_sum(D)
        for r,c in zip(ri,ci):
            d=p[r]-g[c]; se+=float(d[0]**2+d[1]**2); n+=1
    mse=se/max(1,n); rmse=math.sqrt(mse)
    return mse, rmse, n

def entry_exit_modes(traj_csv, W,H,K=8):
    by_id=defaultdict(list)
    with open(traj_csv,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            by_id[int(row["track_id"])].append((int(row["frame"]), float(row["x"]), float(row["y"])))
    def bin_point(x,y):
        d=[y, W-x, H-y, x]  # top,right,bottom,left distances
        s=int(np.argmin(d))
        if s==0: pos=x/W; return ('T', min(K-1,max(0,int(pos*K))))
        if s==1: pos=y/H; return ('R', min(K-1,max(0,int(pos*K))))
        if s==2: pos=x/W; return ('B', min(K-1,max(0,int(pos*K))))
        pos=y/H; return ('L', min(K-1,max(0,int(pos*K))))
    cnt=Counter()
    for tid, pts in by_id.items():
        pts.sort(key=lambda t:t[0]); x0,y0=pts[0][1],pts[0][2]; x1,y1=pts[-1][1],pts[-1][2]
        e=bin_point(x0,y0); x=bin_point(x1,y1)
        if e and x: cnt[(e,x)]+=1
    top3=cnt.most_common(3)
    return cnt, top3

# ---------- bonus: gap reconstruction ----------
def resample_polyline(pts, M=20):
    pts=np.asarray(pts,float)
    if len(pts)<2: return pts
    d=np.sqrt(((pts[1:]-pts[:-1])**2).sum(1)); s=np.insert(np.cumsum(d),0,0.0)
    if s[-1]==0: return np.repeat(pts[:1],M,axis=0)
    t=np.linspace(0,s[-1],M)
    x=np.interp(t,s,pts[:,0]); y=np.interp(t,s,pts[:,1])
    return np.stack([x,y],1)

def build_pair_centroids_from_gt(gt_by_id, W,H,K=8):
    # group GT trajectories by entry/exit bin and compute centroid polyline (length M=20)
    groups=defaultdict(list)
    for tid,pts in gt_by_id.items():
        pts=sorted(pts,key=lambda t:t[0])
        poly=np.array([[p[1],p[2]] for p in pts],float)
        if len(poly)<2: continue
        rr=resample_polyline(poly,20)
        # bins
        def bp(x,y):
            d=[y,W-x,H-y,x]; s=int(np.argmin(d))
            if s==0: return ('T', min(K-1,max(0,int(x/W*K))))
            if s==1: return ('R', min(K-1,max(0,int(y/H*K))))
            if s==2: return ('B', min(K-1,max(0,int(x/W*K))))
            return ('L', min(K-1,max(0,int(y/H*K))))
        e=bp(rr[0,0],rr[0,1]); x=bp(rr[-1,0],rr[-1,1])
        groups[(e,x)].append(rr)
    centroids={}
    for k, arrs in groups.items():
        centroids[k]=np.mean(np.stack(arrs,0),0)
    return centroids  # dict[(e,x)] -> 20x2 polyline

def reconstruct_gap(start_xy, end_xy, entry_bin, exit_bin, centroids):
    key=(entry_bin, exit_bin)
    if key not in centroids:
        # simple quadratic Bezier with control at midpoint
        p0=np.array(start_xy); p2=np.array(end_xy); p1=(p0+p2)/2.0
        t=np.linspace(0,1,20)[:,None]
        rr=(1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2
        return rr
    rr=centroids[key].copy()
    # condition endpoints
    rr[0]=start_xy; rr[-1]=end_xy
    return rr
# -----------------------------------------------

def main():
    ensure_dir(TRAJ_CSV); ensure_dir(REPORT_TXT)

    cap=cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt_by_frame=parse_annotations(ANN_PATH,W,H)        # scaled boxes per frame
    # also centers by frame for metrics
    GT_CENTERS=defaultdict(list)
    GT_BY_ID=defaultdict(list)
    for fr,lst in gt_by_frame.items():
        for it in lst:
            if it["lost"]==1: continue
            b=it["bbox"]; cx=(b[0]+b[2])/2; cy=(b[1]+b[3])/2
            GT_CENTERS[fr].append([cx,cy])
            GT_BY_ID[it["id"]].append((fr,cx,cy))

    # build detector+tracker
    yolo=load_yolo(YOLO_WEIGHTS)
    tracker=ByteLike()

    # viewer
    if SHOW_VIEWER:
        cv2.namedWindow(LEFT_WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(LEFT_WIN,WIN_W,WIN_H)
        cv2.namedWindow(RIGHT_WIN,cv2.WINDOW_NORMAL); cv2.resizeWindow(RIGHT_WIN,WIN_W,WIN_H)

    paused=False; frame_idx=0; cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)

    # write trajectories
    with open(TRAJ_CSV,"w",newline="") as fcsv:
        wr=csv.writer(fcsv); wr.writerow(["track_id","frame","x","y"])
        while True:
            if not paused:
                ok,frame=cap.read()
                if not ok: break
                frame_idx=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            else:
                ok,frame=cap.read()
                if not ok: break
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)

            t0=time.time()

            # GT view
            L=frame.copy()
            for it in gt_by_frame.get(frame_idx,[]):
                if it["lost"]==1: continue
                c=color_for_label(it["label"])
                draw_box(L,it["bbox"],c,f'{it["label"]}#{it["id"]}')

            # detect
            dets,names=yolo_detect(yolo,frame)  # [x1,y1,x2,y2,score,cls]
            # track
            tracks=tracker.step(dets)  # -> list (box,id,hist)

            # draw + dump centers
            R=frame.copy()
            for (box,tid,hist) in tracks:
                x1,y1,x2,y2=map(int,box)
                # name by nearest det
                best="Pedestrian"; best_i=0.0
                for x1d,y1d,x2d,y2d,sc,cl in dets:
                    i=iou_xyxy(box,[x1d,y1d,x2d,y2d])
                    if i>best_i:
                        best_i=i
                        nm = names.get(int(cl),"obj") if isinstance(names,dict) else "obj"
                        best = "Car" if nm=="truck" else nm.capitalize()
                col=color_for_label(best)
                draw_box(R,box,col,f"id{tid}:{best}")
                # center
                cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
                wr.writerow([tid,frame_idx,f"{cx:.2f}",f"{cy:.2f}"])
                # trail
                if len(hist)>=2:
                    pts=[(int((b[0]+b[2])/2),int((b[1]+b[3])/2)) for b in hist]
                    for i in range(1,len(pts)): cv2.line(R,pts[i-1],pts[i],col,2)

            if SHOW_VIEWER:
                fps=1.0/max(1e-6,time.time()-t0)
                cv2.putText(L,f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(R,f"Frame {frame_idx}/{total-1}  FPS~{fps:.1f}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow(LEFT_WIN,cv2.resize(L,(WIN_W,WIN_H)))
                cv2.imshow(RIGHT_WIN,cv2.resize(R,(WIN_W,WIN_H)))
                k=cv2.waitKey(1 if not paused else 0)&0xFF
                if k in (ord('q'),27): break
                elif k==ord('p'): paused=True
                elif k==ord('r'): paused=False
                elif paused and k==ord('o'):
                    frame_idx=min(total-1,frame_idx+1); cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                elif paused and k==ord('i'):
                    frame_idx=max(0,frame_idx-1); cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)

    cap.release()
    if SHOW_VIEWER: cv2.destroyAllWindows()

    # ---- Metrics ----
    mse,rmse,n_pairs=mse_rmse_vs_gt(TRAJ_CSV,GT_CENTERS)

    # ---- Most frequent entry→exit ----
    counts, top3 = entry_exit_modes(TRAJ_CSV,W,H,BINS_PER_EDGE)

    # ---- Bonus reconstruction (library from GT) ----
    centroids = build_pair_centroids_from_gt(GT_BY_ID,W,H,BINS_PER_EDGE)
    # Example usage: reconstruct a gap between first and last observed points for a chosen track
    # (you can integrate this where you detect a gap)
    # rec = reconstruct_gap((x0,y0),(x1,y1), ('L',0), ('T',3), centroids)

    ensure_dir(REPORT_TXT)
    with open(REPORT_TXT,"w") as r:
        r.write(f"Video: {os.path.basename(VIDEO_PATH)}\n")
        r.write(f"Traj CSV: {TRAJ_CSV}\n")
        r.write(f"Pairs matched: {n_pairs}\n")
        r.write(f"MSE (px^2): {mse:.2f}\n")
        r.write(f"RMSE (px): {rmse:.2f}\n")
        r.write(f"Top entry→exit (K={BINS_PER_EDGE} per edge):\n")
        for (pair,cnt) in top3:
            (e_bin,x_bin)=pair
            r.write(f"  {e_bin[0]}{e_bin[1]} -> {x_bin[0]}{x_bin[1]} : {cnt}\n")
        r.write(f"\nBonus: stored {len(centroids)} entry-exit centroids for gap reconstruction.\n")

    print("[OK] Trajectories:", TRAJ_CSV)
    print("[OK] Report:", REPORT_TXT)

if __name__=="__main__":
    main()
