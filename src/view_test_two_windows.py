# # scripts/eval_yolo_sdd_two_videos_progress.py
# # Evaluate your YOLOv8 (Pedestrian+Biker) on ALL frames of video0 and video3.
# # Prints running metrics every 50 frames per video, then final per-video and combined tables.
# # Metrics: Precision, Recall, F1, AP50, mAP50-95, center MSE/RMSE (matched pairs @ IoU=0.5).

# import os, re, math, time
# from pathlib import Path
# import numpy as np
# import cv2

# # =====================
# # HYPERPARAMETERS
# # =====================
# IMG_DIRS = {
#     "video0": r"C:\Users\morte\ComputerVisionProject\Made dataset\images\test\little_video0",
#     "video3": r"C:\Users\morte\ComputerVisionProject\Made dataset\images\test\little_video3",
# }
# ANN_PATHS = {
#     "video0": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video0\annotations.txt",
#     "video3": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video3\annotations.txt",
# }

# ANNOT_ORIG_SIZE = None              # (W,H) if known; else None to auto-infer from maxima
# YOLO_WEIGHTS    = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"

# # --- Detection / NMS ---
# DET_CONF_THRES  = 0.50              # try 0.30 for higher recall if needed
# DET_IOU_NMS     = 0.70
# YOLO_IMGSZ      = 960

# # --- Evaluation ---
# EVAL_CLASSES    = ["Pedestrian", "Biker"]
# IOU_MATCH_THR   = 0.50              # for TP/FP/center error
# AP_THRESHOLDS   = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]  # mAP50-95
# EXCLUDE_LOST    = True              # ignore GT with lost=1

# # --- Progress print cadence ---
# PRINT_EVERY     = 50                # frames

# # =====================

# def list_frames(img_dir):
#     xs = [os.path.join(img_dir,f) for f in os.listdir(img_dir)
#           if f.lower().endswith((".jpg",".jpeg",".png"))]
#     xs.sort()
#     return xs

# def extract_frame_idx(path):
#     m = re.search(r'(\d+)(?=\.[a-zA-Z]+$)', os.path.basename(path))
#     return int(m.group(1)) if m else None

# def parse_annotations(ann_path, out_W, out_H, orig_size=None):
#     raw = {}
#     max_x, max_y = 1.0, 1.0
#     with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
#         for ln in f:
#             s = ln.strip().split()
#             if len(s) < 10: continue
#             try:
#                 tid = int(s[0])
#                 x1  = float(s[1]); y1=float(s[2]); x2=float(s[3]); y2=float(s[4])
#                 fr  = int(s[5]); lost=int(s[6])
#             except ValueError:
#                 continue
#             lab = " ".join(s[9:]).strip().strip('"').capitalize()
#             max_x = max(max_x, x1, x2); max_y = max(max_y, y1, y2)
#             raw.setdefault(fr, []).append(dict(id=tid, bbox=np.array([x1,y1,x2,y2],float),
#                                                label=lab, lost=lost))
#     if orig_size is not None:
#         oW,oH = orig_size; sx = out_W/max(1.0,oW); sy = out_H/max(1.0,oH)
#     else:
#         sx = out_W/max(1.0,max_x); sy = out_H/max(1.0,max_y)

#     per_frame = {}
#     for fr, items in raw.items():
#         out=[]
#         for r in items:
#             x1,y1,x2,y2 = r["bbox"]*np.array([sx,sy,sx,sy],float)
#             x1 = np.clip(x1,0,out_W-1); y1 = np.clip(y1,0,out_H-1)
#             x2 = np.clip(x2,0,out_W-1); y2 = np.clip(y2,0,out_H-1)
#             if x2<=x1 or y2<=y1: continue
#             out.append(dict(id=r["id"], bbox=np.array([x1,y1,x2,y2],float),
#                             label=r["label"], lost=r["lost"]))
#         if out: per_frame[fr]=out
#     return per_frame

# def iou_xyxy(a,b):
#     ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
#     xx1=max(ax1,bx1); yy1=max(ay1,by1); xx2=min(ax2,bx2); yy2=min(ay2,by2)
#     w=max(0,xx2-xx1); h=max(0,yy2-yy1)
#     inter=w*h
#     ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter + 1e-6
#     return inter/ua

# def yolo_load(weights):
#     from ultralytics import YOLO
#     return YOLO(weights)

# def yolo_detect(model, frame, allow_names, conf=DET_CONF_THRES, iou=DET_IOU_NMS):
#     res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=conf, iou=iou, verbose=False)[0]
#     out=[]
#     names = res.names if hasattr(res,"names") else {}
#     if res.boxes is None: return out
#     H,W = frame.shape[:2]
#     for b in res.boxes:
#         cls_id = int(b.cls[0].item()) if b.cls is not None else -1
#         name = names.get(cls_id, f"cls{cls_id}") if isinstance(names,dict) else f"cls{cls_id}"
#         name = name.capitalize()
#         if name not in allow_names: continue
#         x1,y1,x2,y2 = b.xyxy[0].tolist()
#         x1 = max(0,min(W-1,x1)); y1=max(0,min(H-1,y1))
#         x2 = max(0,min(W-1,x2)); y2=max(0,min(H-1,y2))
#         if x2<=x1 or y2<=y1: continue
#         sc = float(b.conf[0].item()) if b.conf is not None else 0.0
#         out.append((name, sc, np.array([x1,y1,x2,y2],float)))
#     return out

# def greedy_match(g_boxes, d_boxes, thr):
#     """Return (matches, uD_idx, uG_idx) using greedy IoU matching at threshold."""
#     if not g_boxes or not d_boxes: return [], list(range(len(d_boxes))), list(range(len(g_boxes)))
#     G=len(g_boxes); D=len(d_boxes)
#     iou = np.zeros((G,D),dtype=np.float32)
#     for i,g in enumerate(g_boxes):
#         for j,d in enumerate(d_boxes):
#             iou[i,j]=iou_xyxy(g,d)
#     matches=[]; used_g=set(); used_d=set()
#     while True:
#         i,j = np.unravel_index(np.argmax(iou), iou.shape)
#         if iou[i,j] < thr: break
#         if i in used_g or j in used_d:
#             iou[i,j]=-1; continue
#         matches.append((i,j))
#         used_g.add(i); used_d.add(j)
#         iou[i,:]=-1; iou[:,j]=-1
#     uD=[j for j in range(D) if j not in used_d]
#     uG=[i for i in range(G) if i not in used_g]
#     return matches, uD, uG

# def compute_AP(gt_store, det_store, frames_done, cls_name, thr):
#     """
#     COCO-style per-threshold AP:
#     - For all detections of 'cls_name' in frames_done, sort globally by score desc.
#     - In each frame, allow each GT to be matched at most once.
#     """
#     # Prepare global det list: (score, frame_key, det_idx, box)
#     global_dets = []
#     for fk in frames_done:
#         if fk not in det_store[cls_name]: continue
#         for j,(sc, box) in enumerate(det_store[cls_name][fk]):
#             global_dets.append((sc, fk, j, box))
#     if not global_dets:
#         return 0.0
#     global_dets.sort(key=lambda t:-t[0])

#     # Per-frame GT matched flags
#     gt_flags = {}
#     total_gt = 0
#     for fk in frames_done:
#         gL = gt_store[cls_name].get(fk, [])
#         total_gt += len(gL)
#         if gL:
#             gt_flags[fk] = np.zeros(len(gL), dtype=bool)

#     if total_gt == 0:
#         return 0.0

#     tps=[]; fps=[]
#     for sc, fk, j, dbox in global_dets:
#         gL = gt_store[cls_name].get(fk, [])
#         if not gL:
#             fps.append((sc, 1)); continue
#         # find best IoU GT not matched yet
#         best_i, best_iou = -1, 0.0
#         for i, gbox in enumerate(gL):
#             if gt_flags[fk][i]: continue
#             iou = iou_xyxy(gbox, dbox)
#             if iou > best_iou:
#                 best_iou = iou; best_i = i
#         if best_i != -1 and best_iou >= thr:
#             gt_flags[fk][best_i] = True
#             tps.append((sc, 1))
#         else:
#             fps.append((sc, 1))

#     # Build PR curve
#     dets = sorted([*[(sc,True) for sc,_ in tps], *[(sc,False) for sc,_ in fps]], key=lambda t:-t[0])
#     tp=0; fp=0; precisions=[]; recalls=[]
#     for sc,is_tp in dets:
#         if is_tp: tp+=1
#         else: fp+=1
#         precisions.append(tp/max(1,tp+fp))
#         recalls.append(tp/total_gt)
#     # Precision envelope
#     mrec=[0.0]+recalls+[1.0]
#     mpre=[0.0]+precisions+[0.0]
#     for i in range(len(mpre)-2,-1,-1):
#         mpre[i]=max(mpre[i], mpre[i+1])
#     ap=0.0
#     for i in range(len(mrec)-1):
#         if mrec[i+1]>mrec[i]:
#             ap += (mrec[i+1]-mrec[i]) * mpre[i+1]
#     return ap

# def evaluate_sequence(seq_name, img_dir, ann_path, model):
#     frames = list_frames(img_dir)
#     if not frames:
#         raise RuntimeError(f"No frames in {img_dir}")
#     sample = cv2.imread(frames[0]); H,W = sample.shape[:2]
#     gt_map = parse_annotations(ann_path, W, H, orig_size=ANNOT_ORIG_SIZE)

#     # Running stats for progress (IoU=0.5)
#     cls_run = {c:{'tp':0,'fp':0,'fn':0,'dets':[],'gt_total':0} for c in EVAL_CLASSES}
#     mse_se=0.0; mse_n=0

#     # Stores for final mAP50-95
#     gt_store  = {c:{} for c in EVAL_CLASSES}   # cls -> frame_key -> [gt boxes]
#     det_store = {c:{} for c in EVAL_CLASSES}   # cls -> frame_key -> [(score, box)]
#     frame_keys_done = []

#     t0 = time.time()
#     for idx, img_path in enumerate(frames, 1):
#         fidx = extract_frame_idx(img_path)
#         frame_key = f"{seq_name}:{fidx}"
#         img = cv2.imread(img_path); H,W = img.shape[:2]

#         # Ground truth this frame
#         gts = gt_map.get(fidx, [])
#         # build per-class GT (and store for AP* computation)
#         for c in EVAL_CLASSES:
#             gL = [g['bbox'] for g in gts if ( (not EXCLUDE_LOST or g['lost']==0) and g['label']==c )]
#             cls_run[c]['gt_total'] += len(gL)
#             gt_store[c][frame_key] = gL

#         # Detection
#         dets = yolo_detect(model, img, set(EVAL_CLASSES), conf=DET_CONF_THRES, iou=DET_IOU_NMS)
#         det_by_c = {c:[] for c in EVAL_CLASSES}
#         for (lab, sc, box) in dets:
#             det_by_c[lab].append((sc, box))
#         for c in EVAL_CLASSES:
#             det_store[c][frame_key] = det_by_c[c]

#         # Match per class at IoU=0.5 for running P/R and MSE
#         for c in EVAL_CLASSES:
#             gL = gt_store[c][frame_key]
#             dL = det_store[c][frame_key]
#             g_list = gL
#             d_boxes = [b for (sc,b) in dL]
#             matches, uD, uG = greedy_match(g_list, d_boxes, IOU_MATCH_THR)

#             # mark TP/FP for AP50 running
#             matched_d_idx = set(j for (_,j) in matches)
#             for j,(sc,_) in enumerate(dL):
#                 cls_run[c]['dets'].append((sc, j in matched_d_idx))
#             cls_run[c]['tp'] += len(matches)
#             cls_run[c]['fp'] += len(uD)
#             cls_run[c]['fn'] += len(uG)

#             # center error
#             for (gi, dj) in matches:
#                 gx1,gy1,gx2,gy2 = g_list[gi]
#                 dx1,dy1,dx2,dy2 = d_boxes[dj]
#                 gcx,gcy = (gx1+gx2)/2.0, (gy1+gy2)/2.0
#                 dcx,dcy = (dx1+dx2)/2.0, (dy1+dy2)/2.0
#                 mse_se += (gcx-dcx)**2 + (gcy-dcy)**2
#                 mse_n += 1

#         frame_keys_done.append(frame_key)

#         # Progress print
#         if idx % PRINT_EVERY == 0 or idx == len(frames):
#             # per-class running AP50 from current dets
#             line = f"[{seq_name}] {idx}/{len(frames)} "
#             totals_inst = sum(cls_run[c]['gt_total'] for c in EVAL_CLASSES)
#             line += f"Images {idx:5d}  Instances {totals_inst:5d}  "

#             # overall P/R from micro-averages
#             TP=sum(cls_run[c]['tp'] for c in EVAL_CLASSES)
#             FP=sum(cls_run[c]['fp'] for c in EVAL_CLASSES)
#             FN=sum(cls_run[c]['fn'] for c in EVAL_CLASSES)
#             P = TP/max(1,TP+FP); R = TP/max(1,TP+FN)

#             # running AP50 overall = mean of per-class AP50
#             ap_list=[]
#             for c in EVAL_CLASSES:
#                 ap_c = compute_AP(gt_store, det_store, frame_keys_done, c, 0.50)
#                 ap_list.append(ap_c)
#             mAP50_run = float(np.mean(ap_list)) if ap_list else 0.0

#             # Optional quick AP75 preview
#             ap75_list=[]
#             for c in EVAL_CLASSES:
#                 ap75_list.append(compute_AP(gt_store, det_store, frame_keys_done, c, 0.75))
#             mAP5095_preview = float(np.mean([(mAP50_run+np.mean(ap75_list))/2.0]))  # quick preview

#             line += f"Box(P {P:.3f}  R {R:.3f}  mAP50 {mAP50_run:.3f}  mAP50-95≈{mAP5095_preview:.3f})"
#             print(line)

#     # Final per-class APs over all thresholds
#     per_class = {}
#     for c in EVAL_CLASSES:
#         # micro P/R/F1 at 0.5
#         tp,fp,fn = cls_run[c]['tp'], cls_run[c]['fp'], cls_run[c]['fn']
#         prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
#         f1 = 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)
#         # AP50
#         ap50 = compute_AP(gt_store, det_store, frame_keys_done, c, 0.50)
#         # mAP50-95
#         aps=[]
#         for thr in AP_THRESHOLDS:
#             aps.append(compute_AP(gt_store, det_store, frame_keys_done, c, thr))
#         mAP5095 = float(np.mean(aps)) if aps else 0.0
#         per_class[c] = dict(TP=tp,FP=fp,FN=fn,Precision=prec,Recall=rec,F1=f1,AP50=ap50,mAP50_95=mAP5095,
#                             GT=cls_run[c]['gt_total'])

#     # Overall micro P/R/F1, mAPs avg over classes
#     TP=sum(per_class[c]['TP'] for c in EVAL_CLASSES)
#     FP=sum(per_class[c]['FP'] for c in EVAL_CLASSES)
#     FN=sum(per_class[c]['FN'] for c in EVAL_CLASSES)
#     P = TP/max(1,TP+FP); R = TP/max(1,TP+FN); F1 = 0.0 if P+R==0 else 2*P*R/(P+R)
#     mAP50 = float(np.mean([per_class[c]['AP50'] for c in EVAL_CLASSES]))
#     mAP5095 = float(np.mean([per_class[c]['mAP50_95'] for c in EVAL_CLASSES]))
#     mse = (0.0 if mse_n==0 else mse_se/mse_n); rmse = math.sqrt(mse) if mse_n>0 else 0.0

#     summary = dict(TP=TP,FP=FP,FN=FN,Precision=P,Recall=R,F1=F1,mAP50=mAP50,mAP50_95=mAP5095,MSE=mse,RMSE=rmse,Matches=mse_n)
#     return per_class, summary

# def main():
#     from ultralytics import YOLO
#     yolo = yolo_load(YOLO_WEIGHTS)

#     results = {}
#     for key in ("video0","video3"):
#         print(f"\n== Evaluating {key} ==")
#         pc, summ = evaluate_sequence(key, IMG_DIRS[key], ANN_PATHS[key], yolo)
#         results[key]=(pc, summ)

#         # Pretty print like Ultralytics
#         n_images = len(list_frames(IMG_DIRS[key]))
#         n_inst   = sum(pc[c]['GT'] for c in EVAL_CLASSES)
#         header = "Class".ljust(14) + "Images  Instances  Box(P       R     mAP50  mAP50-95)"
#         print("\n"+header)
#         print("-"*len(header))
#         print(f"{'all'.ljust(14)}{str(n_images).rjust(7)}{str(n_inst).rjust(11)}"
#               f"  {summ['Precision']:.3f}  {summ['Recall']:.3f}  {summ['mAP50']:.3f}   {summ['mAP50_95']:.3f}")
#         for c in EVAL_CLASSES:
#             st = pc[c]
#             print(f"{c.ljust(14)}{str(n_images).rjust(7)}{str(st['GT']).rjust(11)}"
#                   f"  {st['Precision']:.3f}  {st['Recall']:.3f}  {st['AP50']:.3f}   {st['mAP50_95']:.3f}")
#         print(f"Center RMSE (px): {summ['RMSE']:.2f}  matches:{summ['Matches']}")

#     # Combined micro
#     TP=FP=FN=0; mAP50s=[]; mAP5095s=[]
#     rmse_num=0.0; rmse_den=0
#     gt_comb = {c:0 for c in EVAL_CLASSES}
#     for key in ("video0","video3"):
#         pc, summ = results[key]
#         TP += summ['TP']; FP += summ['FP']; FN += summ['FN']
#         mAP50s.append(summ['mAP50']); mAP5095s.append(summ['mAP50_95'])
#         rmse_num += summ['MSE']*summ['Matches']; rmse_den += summ['Matches']
#         for c in EVAL_CLASSES: gt_comb[c] += pc[c]['GT']
#     P = TP/max(1,TP+FP); R = TP/max(1,TP+FN); F1 = 0.0 if P+R==0 else 2*P*R/(P+R)
#     mAP50 = float(np.mean(mAP50s)); mAP5095 = float(np.mean(mAP5095s))
#     MSE = rmse_num/max(1,rmse_den); RMSE = math.sqrt(MSE) if rmse_den>0 else 0.0

#     print("\n== Combined (video0 + video3) ==")
#     header = "Class".ljust(14) + "Images  Instances  Box(P       R     mAP50  mAP50-95)"
#     print(header); print("-"*len(header))
#     n_images_comb = len(list_frames(IMG_DIRS["video0"])) + len(list_frames(IMG_DIRS["video3"]))
#     n_inst_comb   = sum(gt_comb[c] for c in EVAL_CLASSES)
#     print(f"{'all'.ljust(14)}{str(n_images_comb).rjust(7)}{str(n_inst_comb).rjust(11)}"
#           f"  {P:.3f}  {R:.3f}  {mAP50:.3f}   {mAP5095:.3f}")
#     for c in EVAL_CLASSES:
#         # per-class combined P/R uses sums from per-video
#         tp=sum(results[k][0][c]['TP'] for k in ("video0","video3"))
#         fp=sum(results[k][0][c]['FP'] for k in ("video0","video3"))
#         fn=sum(results[k][0][c]['FN'] for k in ("video0","video3"))
#         gt=gt_comb[c]
#         Pc = tp/max(1,tp+fp); Rc = tp/max(1,tp+fn)
#         # combined AP is non-trivial without raw global lists; report AP means:
#         AP50c = float(np.mean([results[k][0][c]['AP50'] for k in ("video0","video3")]))
#         mAPc  = float(np.mean([results[k][0][c]['mAP50_95'] for k in ("video0","video3")]))
#         print(f"{c.ljust(14)}{str(n_images_comb).rjust(7)}{str(gt).rjust(11)}"
#               f"  {Pc:.3f}  {Rc:.3f}  {AP50c:.3f}   {mAPc:.3f}")
#     print(f"Center RMSE (px): {RMSE:.2f}  matches:{rmse_den}")

# if __name__ == "__main__":
#     main()


# scripts/eval_yolo_sdd_video3_only.py  (scaling fix for video3)

import os, re, math, time
import numpy as np
import cv2
from pathlib import Path

# -------- PATHS --------
IMG_DIR  = r"C:\Users\morte\ComputerVisionProject\Made dataset\images\test\little_video0"
ANN_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video0\annotations.txt"
ANN_REF  = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\little\video0\reference.jpg"  # NEW

YOLO_WEIGHTS   = r"C:\Users\morte\ComputerVisionProject\models\sdd_yolov8s_resume\weights\best.pt"
DET_CONF_THRES = 0.30 # lower for video3    
DET_IOU_NMS    = 0.70
YOLO_IMGSZ     = 960

EVAL_CLASSES   = ["Pedestrian", "Biker"]
IOU_MATCH_THR  = 0.50
AP_THRESHOLDS  = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]
EXCLUDE_LOST      = True
EXCLUDE_OCCLUDED  = False
EXCLUDE_GENERATED = False
PRINT_EVERY    = 50

def list_frames(d):
    xs=[os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith((".jpg",".jpeg",".png"))]
    xs.sort(); return xs

def extract_frame_idx(p):
    m=re.search(r'(\d+)(?=\.[a-zA-Z]+$)', os.path.basename(p))
    return int(m.group(1)) if m else None

def parse_annotations(ann_path, out_W, out_H, orig_size=None):
    raw={}; max_x=max_y=1.0
    with open(ann_path,"r",encoding="utf-8",errors="ignore") as f:
        for ln in f:
            s=ln.strip().split()
            if len(s)<10: continue
            try:
                tid=int(s[0]); x1=float(s[1]); y1=float(s[2]); x2=float(s[3]); y2=float(s[4])
                fr=int(s[5]); lost=int(s[6]); occ=int(s[7]); gen=int(s[8])
            except ValueError: continue
            lab=" ".join(s[9:]).strip().strip('"').capitalize()
            if EXCLUDE_LOST and lost==1: continue
            if EXCLUDE_OCCLUDED and occ==1: continue
            if EXCLUDE_GENERATED and gen==1: continue
            max_x=max(max_x,x1,x2); max_y=max(max_y,y1,y2)
            raw.setdefault(fr,[]).append(dict(id=tid,bbox=np.array([x1,y1,x2,y2],float),label=lab))
    if orig_size is not None:
        oW,oH=orig_size; sx=out_W/max(1.0,oW); sy=out_H/max(1.0,oH)
    else:
        sx=out_W/max(1.0,max_x); sy=out_H/max(1.0,max_y)

    per_frame={}
    for fr,items in raw.items():
        L=[]
        for r in items:
            x1,y1,x2,y2 = r["bbox"]*np.array([sx,sy,sx,sy],float)
            x1=np.clip(x1,0,out_W-1); y1=np.clip(y1,0,out_H-1)
            x2=np.clip(x2,0,out_W-1); y2=np.clip(y2,0,out_H-1)
            if x2<=x1 or y2<=y1: continue
            L.append(dict(id=r["id"],bbox=np.array([x1,y1,x2,y2],float),label=r["label"]))
        if L: per_frame[fr]=L
    return per_frame, sx, sy

def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    xx1=max(ax1,bx1); yy1=max(ay1,by1); xx2=min(ax2,bx2); yy2=min(ay2,by2)
    w=max(0,xx2-xx1); h=max(0,yy2-yy1); inter=w*h
    ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-6
    return inter/ua

def yolo_load(w):
    from ultralytics import YOLO
    return YOLO(w)

def yolo_detect(model, frame, allow_names):
    res=model.predict(frame, imgsz=YOLO_IMGSZ, conf=DET_CONF_THRES, iou=DET_IOU_NMS, verbose=False)[0]
    out=[]; names=getattr(res,"names",{})
    if res.boxes is None: return out
    H,W=frame.shape[:2]
    for b in res.boxes:
        cls_id=int(b.cls[0].item()) if b.cls is not None else -1
        name=(names.get(cls_id,f"cls{cls_id}") if isinstance(names,dict) else f"cls{cls_id}").capitalize()
        if name not in allow_names: continue
        x1,y1,x2,y2=b.xyxy[0].tolist()
        x1=max(0,min(W-1,x1)); y1=max(0,min(H-1,y1))
        x2=max(0,min(W-1,x2)); y2=max(0,min(H-1,y2))
        if x2<=x1 or y2<=y1: continue
        sc=float(b.conf[0].item()) if b.conf is not None else 0.0
        out.append((name,sc,np.array([x1,y1,x2,y2],float)))
    return out

def greedy_match(g_boxes,d_boxes,thr):
    if not g_boxes or not d_boxes: return [], list(range(len(d_boxes))), list(range(len(g_boxes)))
    G=len(g_boxes); D=len(d_boxes); iou=np.zeros((G,D),np.float32)
    for i,g in enumerate(g_boxes):
        for j,d in enumerate(d_boxes): iou[i,j]=iou_xyxy(g,d)
    matches=[]; used_g=set(); used_d=set()
    while True:
        i,j=np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i,j]<thr: break
        if i in used_g or j in used_d: iou[i,j]=-1; continue
        matches.append((i,j)); used_g.add(i); used_d.add(j); iou[i,:]=-1; iou[:,j]=-1
    uD=[j for j in range(D) if j not in used_d]; uG=[i for i in range(G) if i not in used_g]
    return matches,uD,uG

def compute_AP(gt_store, det_store, frames_done, cls_name, thr):
    global_dets = []
    for fk in frames_done:
        items = det_store[cls_name].get(fk, [])
        for j, item in enumerate(items):
            # Accept (score, box) or (label, score, box) or dicts
            if isinstance(item, dict):
                sc = float(item.get('score', 0.0))
                box = np.array(item.get('box'), dtype=float)
            else:
                try:
                    # (score, box)
                    sc, box = item
                except ValueError:
                    # e.g. (label, score, box)
                    sc, box = item[-2], item[-1]
                box = np.array(box, dtype=float)
            global_dets.append((sc, fk, j, box))

    if not global_dets:
        return 0.0
    global_dets.sort(key=lambda t: -t[0])

    gt_flags = {}
    total_gt = 0
    for fk in frames_done:
        gL = gt_store[cls_name].get(fk, [])
        total_gt += len(gL)
        if gL:
            gt_flags[fk] = np.zeros(len(gL), dtype=bool)
    if total_gt == 0:
        return 0.0

    tps = []; fps = []
    for sc, fk, j, dbox in global_dets:
        gL = gt_store[cls_name].get(fk, [])
        if not gL:
            fps.append((sc, 1)); continue
        best_i = -1; best_iou = 0.0
        for i, gbox in enumerate(gL):
            if gt_flags[fk][i]: continue
            iou = iou_xyxy(gbox, dbox)
            if iou > best_iou:
                best_iou = iou; best_i = i
        if best_i != -1 and best_iou >= thr:
            gt_flags[fk][best_i] = True
            tps.append((sc, 1))
        else:
            fps.append((sc, 1))

    dets = sorted([*[(sc, True) for sc, _ in tps], *[(sc, False) for sc, _ in fps]], key=lambda t: -t[0])
    tp = fp = 0; precisions = []; recalls = []
    for sc, is_tp in dets:
        if is_tp: tp += 1
        else: fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / total_gt)

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] > mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def evaluate_video3():
    frames=list_frames(IMG_DIR)
    if not frames: raise RuntimeError(f"No frames in {IMG_DIR}")
    sample=cv2.imread(frames[0]); H,W=sample.shape[:2]

    # --- SCALE USING reference.jpg DIMENSIONS ---
    orig_size=None
    if os.path.isfile(ANN_REF):
        ref=cv2.imread(ANN_REF)
        if ref is not None:
            rH,rW=ref.shape[:2]
            orig_size=(rW,rH)
            print(f"[video3] Using ANN reference size {rW}x{rH} for rescaling.")
        else:
            print("[video3] WARN: reference.jpg unreadable. Falling back to auto-infer.")
    else:
        print("[video3] WARN: reference.jpg not found. Falling back to auto-infer.")

    gt_map,sx,sy = parse_annotations(ANN_PATH, W, H, orig_size=orig_size)
    print(f"[video3] video={W}x{H}  scale_x={sx:.4f}  scale_y={sy:.4f}")

    from ultralytics import YOLO
    yolo=yolo_load(YOLO_WEIGHTS)

    cls_run={c:{'tp':0,'fp':0,'fn':0,'dets':[],'gt_total':0} for c in EVAL_CLASSES}
    mse_se=0.0; mse_n=0
    gt_store={c:{} for c in EVAL_CLASSES}
    det_store={c:{} for c in EVAL_CLASSES}
    frame_keys_done=[]

    for idx,p in enumerate(frames,1):
        fidx=extract_frame_idx(p); fk=f"video3:{fidx}"
        img=cv2.imread(p); H,W=img.shape[:2]

        gts=gt_map.get(fidx,[])
        for c in EVAL_CLASSES:
            gL=[g['bbox'] for g in gts if g['label']==c]
            cls_run[c]['gt_total']+=len(gL); gt_store[c][fk]=gL

        dets=yolo_detect(yolo,img,set(EVAL_CLASSES))
        det_by_c={c:[] for c in EVAL_CLASSES}
        for (lab,sc,box) in dets: det_by_c[lab].append((sc,box))
        for c in EVAL_CLASSES: det_store[c][fk]=det_by_c[c]

        for c in EVAL_CLASSES:
            g_list=gt_store[c][fk]; d_list=det_store[c][fk]
            d_boxes=[b for (sc,b) in d_list]
            matches,uD,uG=greedy_match(g_list,d_boxes,IOU_MATCH_THR)
            matched_d_idx=set(j for (_,j) in matches)
            for j,(sc,_) in enumerate(d_list):
                cls_run[c]['dets'].append((sc, j in matched_d_idx))
            cls_run[c]['tp']+=len(matches); cls_run[c]['fp']+=len(uD); cls_run[c]['fn']+=len(uG)
            for (gi,dj) in matches:
                gx1,gy1,gx2,gy2=g_list[gi]; dx1,dy1,dx2,dy2=d_boxes[dj]
                gcx,gcy=(gx1+gx2)/2.0,(gy1+gy2)/2.0; dcx,dcy=(dx1+dx2)/2.0,(dy1+dy2)/2.0
                mse_se+=(gcx-dcx)**2+(gcy-dcy)**2; mse_n+=1

        frame_keys_done.append(fk)

        if idx%PRINT_EVERY==0 or idx==len(frames):
            TP=sum(cls_run[c]['tp'] for c in EVAL_CLASSES)
            FP=sum(cls_run[c]['fp'] for c in EVAL_CLASSES)
            FN=sum(cls_run[c]['fn'] for c in EVAL_CLASSES)
            P=TP/max(1,TP+FP); R=TP/max(1,TP+FN)
            ap50s=[compute_AP(gt_store,det_store,frame_keys_done,c,0.50) for c in EVAL_CLASSES]
            ap75s=[compute_AP(gt_store,det_store,frame_keys_done,c,0.75) for c in EVAL_CLASSES]
            print(f"[video3] {idx}/{len(frames)}  Images {idx:5d}  "
                  f"Box(P {P:.3f}  R {R:.3f}  mAP50 {np.mean(ap50s):.3f}  mAP75 {np.mean(ap75s):.3f})")

    # Final report
    per_class={}
    for c in EVAL_CLASSES:
        tp,fp,fn=cls_run[c]['tp'],cls_run[c]['fp'],cls_run[c]['fn']
        P=tp/max(1,tp+fp); R=tp/max(1,tp+fn); F1=0.0 if P+R==0 else 2*P*R/(P+R)
        ap50=compute_AP(gt_store,det_store,frame_keys_done,c,0.50)
        aps=[compute_AP(gt_store,det_store,frame_keys_done,c,t) for t in AP_THRESHOLDS]
        per_class[c]=dict(Precision=P,Recall=R,F1=F1,AP50=ap50,mAP50_95=float(np.mean(aps)),
                          TP=tp,FP=fp,FN=fn,GT=cls_run[c]['gt_total'])
    TP=sum(per_class[c]['TP'] for c in EVAL_CLASSES)
    FP=sum(per_class[c]['FP'] for c in EVAL_CLASSES)
    FN=sum(per_class[c]['FN'] for c in EVAL_CLASSES)
    P=TP/max(1,TP+FP); R=TP/max(1,TP+FN); mAP50=float(np.mean([per_class[c]['AP50'] for c in EVAL_CLASSES]))
    mAP5095=float(np.mean([per_class[c]['mAP50_95'] for c in EVAL_CLASSES]))
    mse=(0.0 if mse_n==0 else mse_se/mse_n); rmse=math.sqrt(mse) if mse_n>0 else 0.0

    print("\n== video3 Results ==")
    header="Class".ljust(14)+"Images  Instances  Box(P       R     mAP50  mAP50-95)"
    print(header); print("-"*len(header))
    n_images=len(frames); n_inst=sum(per_class[c]['GT'] for c in EVAL_CLASSES)
    print(f"{'all'.ljust(14)}{str(n_images).rjust(7)}{str(n_inst).rjust(11)}  {P:.3f}  {R:.3f}  {mAP50:.3f}   {mAP5095:.3f}")
    for c in EVAL_CLASSES:
        st=per_class[c]
        print(f"{c.ljust(14)}{str(n_images).rjust(7)}{str(st['GT']).rjust(11)}  "
              f"{st['Precision']:.3f}  {st['Recall']:.3f}  {st['AP50']:.3f}   {st['mAP50_95']:.3f}")
    print(f"Center RMSE (px): {rmse:.2f}  matches:{mse_n}")

if __name__=="__main__":
    evaluate_video3()
