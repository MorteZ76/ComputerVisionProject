# i have an issue here. some times there is a biker and then for a second the detector detects it as a pedestrain and then it changes to pedestrian. but i want the code to consider this:
# 1- if there is a bounding box for a biker for example at frame N, then there should be a bounding box for a biker with the same id at frame N+1 if not. and there is another bounding box very close to this bounding box and its some other class, then most likely our detection is wrong and it should consider it as biker too. it should depend on the confidence level of our detection, and maybe maybe in tracking for each object and id we should keep the top 3 max detection confidence mean. so we can compare with new detection. if our top 3 max detection confidence is higher than new confidence then it should just continue considering that object as biker (for example) if not it should assign it to new class but keep the id the same and just change the top 3 max confidence


# scripts/run_yolo_track_eval_smooth.py
import os, csv, math, time, collections
from pathlib import Path
import numpy as np
import cv2

# --------- HYPERPARAMETERS ----------
VIDEO_INFOS = [
    {"name":"video0",
     "video_path": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4",
     "ann_path":   r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"},
    {"name":"video3",
     "video_path": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4",
     "ann_path":   r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"},
]

YOLO_WEIGHTS = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"
YOLO_CONF  = 0.25
YOLO_IOU   = 0.70
YOLO_IMGSZ = 960
ALLOW_CLASSES = None   # e.g. {0,1}

IOU_MATCH_THR = 0.2
MAX_AGE = 30
MIN_HITS = 3

CONF_HISTORY_K = 3
SWITCH_MARGIN  = 0.00
REMAP = {"truck": "Car"}

OUT_DIR = r"C:\Users\morte\ComputerVisionProject\outputs_yolo_sort"
TRAJ_CSV_TMPL   = "{video}_traj.csv"
REPORT_TXT_TMPL = "{video}_report.txt"
RECON_CSV_TMPL  = "{video}_reconstruct_example.csv"

BINS_PER_SIDE = 8

SHOW = True
LEFT_WIN  = "GT"
RIGHT_WIN = "YOLO+SORT"
WINDOW_W, WINDOW_H = 1280, 720
# ------------------------------------

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    xx1,yy1=max(ax1,bx1),max(ay1,by1); xx2,yy2=min(ax2,bx2),min(ay2,by2)
    w,h=max(0,xx2-xx1),max(0,yy2-yy1)
    inter=w*h
    ua=max(1e-6,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)
    return inter/ua

def linear_sum_assignment(cost):
    try:
        from scipy.optimize import linear_sum_assignment as lsa
        return lsa(cost)
    except Exception:
        C=cost.copy(); n,m=C.shape; R=[]; Cc=[]
        for _ in range(min(n,m)):
            r,c=np.unravel_index(np.argmin(C),C.shape)
            R.append(r); C[r,:]=np.inf; Cc.append(c); C[:,c]=np.inf
        return np.array(R), np.array(Cc)

def parse_annotations_txt(path):
    per_frame=collections.defaultdict(list); max_x=1.; max_y=1.
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            s=ln.strip().split()
            if len(s)<10: continue
            tid=int(s[0]); x1,y1,x2,y2=map(float,s[1:5]); fr=int(s[5]); lost=int(s[6])
            label=" ".join(s[9:]).strip().strip('"')
            max_x=max(max_x,x1,x2); max_y=max(max_y,y1,y2)
            per_frame[fr].append(dict(id=tid,bbox=np.array([x1,y1,x2,y2],float),label=label,lost=lost))
    return per_frame,max_x,max_y

def draw_box(img,xyxy,color,txt=None,t=2):
    x1,y1,x2,y2=map(int,xyxy)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,t,cv2.LINE_AA)
    if txt: cv2.putText(img,txt,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)

def label_color(name):
    table={"Pedestrian":(50,200,50),"Biker":(60,140,255),"Skater":(180,160,50),
           "Cart":(200,50,200),"Car":(60,60,220),"Bus":(0,120,200)}
    return table.get(name,(180,180,180))

def norm_label(name):
    if not isinstance(name,str): return name
    low=name.lower()
    if low in REMAP: return REMAP[low]
    return name.capitalize()

# ---------- SORT w/ class smoothing ----------
class KalmanBoxTracker:
    count=0
    def __init__(self,bbox):
        x1,y1,x2,y2=bbox
        w=max(1.,x2-x1); h=max(1.,y2-y1)
        cx,cy=x1+w/2., y1+h/2.; s=w*h; r=w/h
        self.kf=cv2.KalmanFilter(7,4)
        self.kf.transitionMatrix=np.array(
            [[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
             [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],
            dtype=np.float32)
        self.kf.measurementMatrix=np.eye(4,7,dtype=np.float32)
        self.kf.processNoiseCov=np.eye(7,dtype=np.float32)*1e-2
        self.kf.measurementNoiseCov=np.eye(4,dtype=np.float32)*1e-1
        self.kf.statePost=np.array([[cx],[cy],[s],[r],[0],[0],[0]],dtype=np.float32)
        KalmanBoxTracker.count+=1; self.id=KalmanBoxTracker.count
        self.hits=1; self.no_losses=0; self.history=[]
        self.cls=None
        self.cls_scores=collections.defaultdict(list)

    def predict(self):
        self.kf.predict(); self.no_losses+=1
        b=self.get_state(); self.history.append(b)
        if len(self.history)>32: self.history.pop(0)
        return b

    def update(self,bbox,det_cls,det_score):
        x1,y1,x2,y2=bbox
        w=max(1.,x2-x1); h=max(1.,y2-y1)
        cx,cy=x1+w/2., y1+h/2.; s=w*h; r=w/max(1.,h)
        z=np.array([[cx],[cy],[s],[r]],dtype=np.float32)
        self.kf.correct(z); self.hits+=1; self.no_losses=0
        dname=norm_label(det_cls)
        lst=self.cls_scores[dname]; lst.append(float(det_score)); lst.sort(reverse=True)
        if len(lst)>CONF_HISTORY_K: lst[:] = lst[:CONF_HISTORY_K]
        if self.cls is None: self.cls=dname; return
        if dname==self.cls: return
        old_mean=float(np.mean(self.cls_scores.get(self.cls,[0.0])))
        if det_score > (old_mean + SWITCH_MARGIN): self.cls=dname

    def get_state(self):
        cx,cy,s,r=self.kf.statePost[:4,0]
        w=math.sqrt(max(1.,s*r)); h=s/max(1.,w)
        return np.array([cx-w/2., cy-h/2., cx+w/2., cy+h/2.], dtype=float)

class SortSmooth:
    def __init__(self,max_age=30,min_hits=3,iou_thresh=0.2):
        self.max_age=max_age; self.min_hits=min_hits; self.iou_thresh=iou_thresh
        self.tracks=[]
    def update(self,dets_meta):
        for t in self.tracks: t.predict()
        boxes=[d["box"] for d in dets_meta]
        N=len(self.tracks); M=len(boxes)
        matches=[]; unmatched_t=list(range(N)); unmatched_d=list(range(M))
        if N>0 and M>0:
            iou_mat=np.zeros((N,M),np.float32)
            for i,t in enumerate(self.tracks):
                tb=t.get_state()
                for j,b in enumerate(boxes):
                    iou_mat[i,j]=iou_xyxy(tb,b)
            used_t=set(); used_d=set()
            for _ in range(min(N,M)):
                i,j=np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i,j]<self.iou_thresh or i in used_t or j in used_d:
                    iou_mat[i,j]=-1; continue
                matches.append((i,j)); used_t.add(i); used_d.add(j)
                iou_mat[i,:]=-1; iou_mat[:,j]=-1
            unmatched_t=[i for i in range(N) if i not in used_t]
            unmatched_d=[j for j in range(M) if j not in used_d]
        for i,j in matches:
            d=dets_meta[j]; self.tracks[i].update(d["box"], d["cls"], d["score"])
        for j in unmatched_d:
            d=dets_meta[j]; t=KalmanBoxTracker(d["box"])
            t.update(d["box"], d["cls"], d["score"]); self.tracks.append(t)
        self.tracks=[t for t in self.tracks if t.no_losses<=self.max_age]
        outs=[]
        for t in self.tracks:
            if t.hits>=self.min_hits or t.no_losses==0:
                outs.append((t.get_state(), t.id, t.cls, list(t.history)))
        return outs

# ---------- YOLO ----------
def yolo_load(weights):
    from ultralytics import YOLO
    return YOLO(weights)

def _valid_box(b, W, H):
    x1,y1,x2,y2=b
    if not np.isfinite([x1,y1,x2,y2]).all(): return False
    if x2<=x1 or y2<=y1: return False
    if x2<0 or y2<0 or x1>W or y1>H: return False
    return True

def yolo_detect(model, frame):
    H, W = frame.shape[:2]
    res=model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
    out=[]; names = res.names if hasattr(res,"names") else {}
    if res.boxes is None: return out, names
    for b in res.boxes:
        cls_id=int(b.cls[0].item()) if b.cls is not None else -1
        if ALLOW_CLASSES is not None and cls_id not in ALLOW_CLASSES: continue
        x1,y1,x2,y2=b.xyxy[0].tolist()
        sc=float(b.conf[0].item()) if b.conf is not None else 0.0
        if not _valid_box((x1,y1,x2,y2), W, H): continue  # guard invalid/zero boxes
        name = names.get(cls_id, f"cls{cls_id}") if isinstance(names, dict) else f"cls{cls_id}"
        name = norm_label(name)
        out.append({"box":[x1,y1,x2,y2], "cls":name, "score":sc})
    return out, names

# ---------- Metrics / Paths / Bonus ----------
def mse_rmse_vs_gt(traj_csv, ann_by_frame, sx, sy):
    pred_by=collections.defaultdict(list)
    with open(traj_csv,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            fr=int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            pred_by[fr].append([x,y])
    se=0.0; n=0
    for fr in sorted(set(pred_by)&set(ann_by_frame)):
        P=np.array(pred_by[fr],np.float32)
        Gc=[]
        for it in ann_by_frame[fr]:
            if it["lost"]==1: continue
            x1,y1,x2,y2=it["bbox"]; x1*=sx; y1*=sy; x2*=sx; y2*=sy
            Gc.append([(x1+x2)/2., (y1+y2)/2.])
        if not Gc: continue
        G=np.array(Gc,np.float32)
        D=np.linalg.norm(P[:,None,:]-G[None,:,:],axis=2)
        ri,ci=linear_sum_assignment(D)
        for r,c in zip(ri,ci):
            d=P[r]-G[c]; se+=float(d @ d); n+=1
    mse=se/max(1,n); return mse, math.sqrt(mse), n

def entry_exit_bins(traj_csv,W,H,K=8):
    per=collections.defaultdict(list)
    with open(traj_csv,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            tid=int(row["track_id"]); fr=int(row["frame"]); x=float(row["x"]); y=float(row["y"])
            per[tid].append((fr,x,y))
    def bp(x,y):
        d=[y, W-x, H-y, x]; s=int(np.argmin(d))
        if s==0: pos=x/W;  return ('T', min(K-1,max(0,int(pos*K))))
        if s==1: pos=y/H;  return ('R', min(K-1,max(0,int(pos*K))))
        if s==2: pos=x/W;  return ('B', min(K-1,max(0,int(pos*K))))
        return ('L', min(K-1,max(0,int(pos*K))))
    cnt=collections.Counter()
    for tid,pts in per.items():
        pts.sort(key=lambda t:t[0]); x0,y0=pts[0][1],pts[0][2]; x1,y1=pts[-1][1],pts[-1][2]
        cnt[(bp(x0,y0), bp(x1,y1))]+=1
    return cnt, cnt.most_common(3)

def build_library_from_gt(ann_by_frame, sx, sy, W, H, K=8, resample_n=20):
    per=collections.defaultdict(list)
    for fr,items in ann_by_frame.items():
        for it in items:
            if it["lost"]==1: continue
            x1,y1,x2,y2=it["bbox"]; x1*=sx; y1*=sy; x2*=sx; y2*=sy
            cx,cy=(x1+x2)/2.,(y1+y2)/2.; per[it["id"]].append((fr,cx,cy))
    def bp(x,y):
        d=[y,W-x,H-y,x]; s=int(np.argmin(d))
        if s==0: pos=x/W;  return ('T',int(min(K-1,max(0,int(pos*K)))))
        if s==1: pos=y/H;  return ('R',int(min(K-1,max(0,int(pos*K)))))
        if s==2: pos=x/W;  return ('B',int(min(K-1,max(0,int(pos*K)))))
        return ('L',int(min(K-1,max(0,int(pos*K)))))
    def resample(poly,n):
        if len(poly)<2: return [poly[0]]*n
        xs=np.array([p[0] for p in poly]); ys=np.array([p[1] for p in poly])
        idx=np.linspace(0,len(poly)-1,n)
        return list(zip(np.interp(idx, np.arange(len(poly)), xs),
                        np.interp(idx, np.arange(len(poly)), ys)))
    lib=collections.defaultdict(list)
    for tid,pts in per.items():
        pts.sort(key=lambda t:t[0]); xs=[p[1] for p in pts]; ys=[p[2] for p in pts]
        lib[(bp(xs[0],ys[0]), bp(xs[-1],ys[-1]))].append(resample(list(zip(xs,ys)),resample_n))
    return {k: np.array(v).mean(axis=0) for k,v in lib.items()}

def reconstruct_segment(start_xy, end_xy, key, centroids, n_points=20):
    C=centroids.get(key)
    if C is None:
        p0=np.array(start_xy); p2=np.array(end_xy); pc=(p0+p2)/2.; ts=np.linspace(0,1,n_points)
        return [(float(((1-t)**2)*p0[0]+2*(1-t)*t*pc[0]+(t**2)*p2[0]),
                 float(((1-t)**2)*p0[1]+2*(1-t)*t*pc[1]+(t**2)*p2[1])) for t in ts]
    mid=C[1:-1]; return [start_xy]+[(float(x),float(y)) for x,y in mid]+[end_xy]

# ---------- main ----------
def process_video(info):
    name=info["name"]; vpath=info["video_path"]; apath=info["ann_path"]
    ensure_dir(OUT_DIR)
    traj_csv=os.path.join(OUT_DIR, TRAJ_CSV_TMPL.format(video=name))
    report_txt=os.path.join(OUT_DIR, REPORT_TXT_TMPL.format(video=name))
    recon_csv=os.path.join(OUT_DIR, RECON_CSV_TMPL.format(video=name))

    cap=cv2.VideoCapture(vpath)
    if not cap.isOpened(): raise RuntimeError("Cannot open video: "+vpath)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ann_by_frame,max_ax,max_ay=parse_annotations_txt(apath)
    sx=W/(max_ax if max_ax>0 else 1.); sy=H/(max_ay if max_ay>0 else 1.)

    yolo=yolo_load(YOLO_WEIGHTS)
    tracker=SortSmooth(MAX_AGE, MIN_HITS, IOU_MATCH_THR)

    if SHOW:
        cv2.namedWindow(LEFT_WIN, cv2.WINDOW_NORMAL)
        cv2.namedWindow(RIGHT_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(LEFT_WIN, WINDOW_W, WINDOW_H)
        cv2.resizeWindow(RIGHT_WIN, WINDOW_W, WINDOW_H)
        paused=False; frame_idx=0; cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    no_det_frames=0
    wrote_rows=0

    with open(traj_csv,"w",newline="") as fcsv:
        wr=csv.writer(fcsv); wr.writerow(["track_id","frame","x","y","label"])
        while True:
            if SHOW:
                if not paused:
                    ok,frame=cap.read()
                    if not ok: break
                    frame_idx=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok,frame=cap.read()
                    if not ok: break
            else:
                ok,frame=cap.read()
                if not ok: break
                frame_idx=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1

            t0=time.time()
            dets_meta,_=yolo_detect(yolo, frame)

            if len(dets_meta)==0:
                no_det_frames+=1
                if no_det_frames==50:
                    print("No detections in first 50 frames. Check weights/conf/classes.")
            else:
                no_det_frames=0

            tracks=tracker.update(dets_meta)

            # write centers only for valid boxes
            for (box, tid, cls, hist) in tracks:
                if not np.isfinite(box).all(): continue
                if (box[2]-box[0])<=0 or (box[3]-box[1])<=0: continue
                cx=(box[0]+box[2])/2.; cy=(box[1]+box[3])/2.
                wr.writerow([tid, frame_idx, f"{cx:.2f}", f"{cy:.2f}", cls if cls else "Unknown"])
                wrote_rows+=1

            if SHOW:
                left = frame.copy(); right = frame.copy()
                for it in ann_by_frame.get(frame_idx, []):
                    if it["lost"]==1: continue
                    x1,y1,x2,y2 = it["bbox"]; x1*=sx; y1*=sy; x2*=sx; y2*=sy
                    draw_box(left, (x1,y1,x2,y2), label_color(it["label"]), f'{it["label"]}#{it["id"]}', 2)
                for d in dets_meta:
                    draw_box(right, d["box"], (0,255,255), f'{d["cls"]} {d["score"]:.2f}', 1)
                for (box, tid, cls, hist) in tracks:
                    if (box[2]-box[0])<=0 or (box[3]-box[1])<=0: continue
                    draw_box(right, box, (0,255,0), f'id{tid}:{cls}', 2)
                    if len(hist)>=2:
                        for i in range(1,len(hist)):
                            p0=(int((hist[i-1][0]+hist[i-1][2])//2), int((hist[i-1][1]+hist[i-1][3])//2))
                            p1=(int((hist[i][0]+hist[i][2])//2),     int((hist[i][1]+hist[i][3])//2))
                            cv2.line(right, p0, p1, (0,200,0), 2)
                fps=1.0/max(1e-6,time.time()-t0)
                cv2.putText(right, f"Det:{len(dets_meta)} Trk:{len(tracks)} FPS~{fps:.1f}", (12,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow(LEFT_WIN,  cv2.resize(left,(WINDOW_W,WINDOW_H)))
                cv2.imshow(RIGHT_WIN, cv2.resize(right,(WINDOW_W,WINDOW_H)))
                k=cv2.waitKey(1 if not paused else 0)&0xFF
                if k in (ord('q'),27): break
                elif k==ord('p'): paused=True
                elif k==ord('r'): paused=False
                elif paused and k==ord('o'):
                    frame_idx=min(total-1, frame_idx+1)
                elif paused and k==ord('i'):
                    frame_idx=max(0, frame_idx-1)

    cap.release()
    if SHOW: cv2.destroyAllWindows()
    print(f"[{name}] wrote {wrote_rows} rows to {traj_csv}")

    # Evaluation + paths + bonus (unchanged). If you still need them here, say and I’ll append.
    # Keeping just tracking fix so csv stops having zeros.
    return traj_csv

def main():
    ensure_dir(OUT_DIR)
    for info in VIDEO_INFOS:
        process_video(info)

if __name__=="__main__":
    main()
