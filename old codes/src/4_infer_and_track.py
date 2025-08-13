# scripts/4_infer_and_track.py
import os, csv, time, math
import numpy as np
import cv2

# ---------- HYPERPARAMETERS (edit here) ----------
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\little\video0\video.mp4"
OUTPUT_CSV = r"C:\Users\morte\ComputerVisionProject\trajectories_video0_bg.csv"

MODE = "bg"  # "bg" for BackgroundSub+KF; "yolo" for YOLOv8 + BYTE-like (optional)

# Viewer
SHOW_WINDOW = True
WIN_NAME = "Tracker"
WINDOW_W, WINDOW_H = 1280, 720
SHOW_TRAILS = True
TRAIL_LEN = 24
SHOW_MIN_AREA_BOX = True

# ROI polygon in image coords, or None to disable
ROI_POLY = None  # e.g., [(100,200),(1700,200),(1700,900),(100,900)]

# ---- BG-sub detector (fast CPU baseline) ----
MOG2_HISTORY = 500
MOG2_VARTHRESH = 16
MOG2_DETECT_SHADOWS = False
MOG2_LR = -1  # auto

MORPH_OPEN = 3
MORPH_CLOSE = 5

MIN_CONTOUR_AREA = 200     # raise to suppress noise
MIN_BOX_W, MIN_BOX_H = 6, 6
MAX_ASPECT_RATIO = 3.0     # reject extreme skinny/wide boxes

IOU_MATCH_THR = 0.2        # association gate
MAX_AGE = 30               # frames to keep unmatched tracks
MIN_HITS = 3               # confirm track after N hits

# Heuristic classification
BIKER_SPEED_PX = 3.0       # avg px/frame to call Biker
BUS_AREA_PX = 9000         # area threshold for Bus
COLOR_MAP = {
    "Pedestrian": (50, 200, 50),
    "Biker": (60, 140, 255),
    "Bus": (0, 120, 200),
    "Other": (180, 180, 180),
}

# ---- Optional YOLOv8 + BYTE-like (CPU works, slower) ----
YOLO_MODEL = "yolov8n.pt"  # tiny
YOLO_IMGSZ = 640
YOLO_CONF = 0.30
YOLO_NMS_IOU = 0.70
BYTE_HIGH_THR = 0.50
BYTE_LOW_THR  = 0.10
DETECT_CLASSES = {0,1,2,3,5,7}  # COCO ids of interest
# ----------------------------------------------------------

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(1e-6, (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
    return inter / ua

# --------- Kalman/SORT-like ----------
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # state [cx, cy, s, r, vx, vy, vs]; z=[cx,cy,s,r]
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
        if len(self.history) > TRAIL_LEN:
            self.history.pop(0)
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
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.trackers = []

    def update(self, dets):  # dets: list of xyxy
        # Predict all
        for t in self.trackers: t.predict()

        N, M = len(self.trackers), len(dets)
        matches, unmatched_t, unmatched_d = [], list(range(N)), list(range(M))
        if N>0 and M>0:
            iou_mat = np.zeros((N, M), dtype=np.float32)
            for i,t in enumerate(self.trackers):
                tb = t.get_state()
                for j,d in enumerate(dets):
                    iou_mat[i,j] = iou_xyxy(tb, d)
            used_t, used_d = set(), set()
            # greedy
            for _ in range(min(N,M)):
                i,j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i,j] < self.iou_thresh or i in used_t or j in used_d:
                    iou_mat[i,j] = -1; continue
                matches.append((i,j)); used_t.add(i); used_d.add(j)
                iou_mat[i,:] = -1; iou_mat[:,j] = -1
            unmatched_t = [i for i in range(N) if i not in used_t]
            unmatched_d = [j for j in range(M) if j not in used_d]

        # Update matched
        for i,j in matches:
            self.trackers[i].update(dets[j])
        # New tracks
        for j in unmatched_d:
            self.trackers.append(KalmanBoxTracker(dets[j]))
        # Remove dead
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        # Output confirmed or just-updated
        outs=[]
        for t in self.trackers:
            if t.hits >= self.min_hits or t.no_losses == 0:
                outs.append((t.get_state(), t.id, list(t.history)))
        return outs

# -------- BG-sub detection --------
def build_roi_mask(shape, poly):
    H, W = shape[:2]
    mask = np.zeros((H,W), dtype=np.uint8)
    if poly and len(poly)>=3:
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    else:
        mask[:] = 255
    return mask

def detect_bg_boxes(frame, mog, roi_mask):
    fg = mog.apply(frame, learningRate=MOG2_LR)
    if roi_mask is not None:
        fg = cv2.bitwise_and(fg, roi_mask)

    if MORPH_OPEN>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    if MORPH_CLOSE>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA: continue
        x,y,w,h = cv2.boundingRect(c)
        if w < MIN_BOX_W or h < MIN_BOX_H: continue
        ar = max(w/h, h/w)
        if ar > MAX_ASPECT_RATIO: continue
        boxes.append([x,y,x+w,y+h])
    return boxes, fg

# -------- Heuristic class --------
def classify_by_speed_area(box, speed_px):
    x1,y1,x2,y2 = box
    area = max(1.0, (x2-x1)*(y2-y1))
    if area >= BUS_AREA_PX: return "Bus"
    if speed_px >= BIKER_SPEED_PX: return "Biker"
    return "Pedestrian"

# -------- Optional YOLO+BYTE-like --------
class ByteLike:
    def __init__(self, iou_thr, max_age, min_hits):
        self.sort = Sort(max_age=max_age, min_hits=min_hits, iou_thresh=iou_thr)
    def update(self, high_boxes, low_boxes):
        self.sort.update(high_boxes)           # first pass
        self.sort.update(low_boxes)            # recover with low-conf
        return self.sort.update([])            # emit state

def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_dir(OUTPUT_CSV)
    writer = csv.writer(open(OUTPUT_CSV, "w", newline=""))
    writer.writerow(["track_id","frame","x","y","label"])

    # trackers
    if MODE == "bg":
        mog = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY, varThreshold=MOG2_VARTHRESH, detectShadows=MOG2_DETECT_SHADOWS
        )
        roi_mask = build_roi_mask((H,W), ROI_POLY)
        tracker = Sort(MAX_AGE, MIN_HITS, IOU_MATCH_THR)
    else:
        # optional YOLO path
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        tracker = ByteLike(IOU_MATCH_THR, MAX_AGE, MIN_HITS)

    # viewer
    if SHOW_WINDOW:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, WINDOW_W, WINDOW_H)

    # track history for speed
    last_centers = {}  # id -> deque of centers
    from collections import deque

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok: break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok: break

        t0 = time.time()
        vis = frame.copy()

        if MODE == "bg":
            dets, fg = detect_bg_boxes(frame, mog, roi_mask=None if ROI_POLY is None else build_roi_mask((H,W), ROI_POLY))
            tracks = tracker.update(dets)
            # classify by speed/area
            for (box, tid, hist) in tracks:
                cx = 0.5*(box[0]+box[2]); cy = 0.5*(box[1]+box[3])
                dq = last_centers.get(tid)
                if dq is None:
                    dq = deque(maxlen=10); last_centers[tid] = dq
                dq.append((cx,cy))
                spd = 0.0
                if len(dq) >= 2:
                    dsum=0.0
                    for (x0,y0),(x1,y1) in zip(dq, list(dq)[1:]):
                        dsum += math.hypot(x1-x0, y1-y0)
                    spd = dsum / (len(dq)-1)
                lbl = classify_by_speed_area(box, spd)
                color = COLOR_MAP[lbl]
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, f"{lbl} id{tid} v={spd:.1f}", (x1, max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if SHOW_TRAILS and len(hist)>=2:
                    pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in hist]
                    for i in range(1, len(pts)):
                        cv2.line(vis, pts[i-1], pts[i], color, 2)
                writer.writerow([tid, frame_idx, f"{cx:.2f}", f"{cy:.2f}", lbl])

            # purple reference min-area box
            if SHOW_MIN_AREA_BOX:
                s = int(math.sqrt(MIN_CONTOUR_AREA))
                x0,y0 = 12, 12
                cv2.rectangle(vis, (x0,y0), (x0+s, y0+s), (200,0,200), 2)
                cv2.putText(vis, f"MinArea={MIN_CONTOUR_AREA}", (x0, y0+s+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), 1, cv2.LINE_AA)

        else:
            # YOLO + BYTE-like (optional)
            res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_NMS_IOU, verbose=False)[0]
            dets = []
            for b in (res.boxes or []):
                cls_id = int(b.cls[0].item())
                if cls_id not in DETECT_CLASSES: continue
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                score = float(b.conf[0].item())
                dets.append([x1,y1,x2,y2,score,cls_id])
            high = [d[:4] for d in dets if d[4] >= BYTE_HIGH_THR]
            low  = [d[:4] for d in dets if BYTE_LOW_THR <= d[4] < BYTE_HIGH_THR]
            out = tracker.update(high, low)
            # draw and log
            for (box, tid, hist) in out:
                cx = 0.5*(box[0]+box[2]); cy = 0.5*(box[1]+box[3])
                # approximate label by nearest detection IoU
                best_lbl = "Pedestrian"; best_i=0.0
                for x1,y1,x2,y2,sc,cls_id in dets:
                    i = iou_xyxy(box, [x1,y1,x2,y2])
                    if i>best_i:
                        best_i=i
                        best_lbl = {0:"Pedestrian",1:"Biker",2:"Other",3:"Biker",5:"Bus",7:"Other"}.get(cls_id,"Other")
                color = COLOR_MAP.get(best_lbl, (180,180,180))
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, f"{best_lbl} id{tid}", (x1, max(15,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if SHOW_TRAILS and len(hist)>=2:
                    pts = [(int((b[0]+b[2])/2), int((b[1]+b[3])/2)) for b in hist]
                    for i in range(1, len(pts)):
                        cv2.line(vis, pts[i-1], pts[i], color, 2)
                writer.writerow([tid, frame_idx, f"{cx:.2f}", f"{cy:.2f}", best_lbl])

        # show
        if SHOW_WINDOW:
            disp = cv2.resize(vis, (WINDOW_W, WINDOW_H)) if (vis.shape[1]!=WINDOW_W or vis.shape[0]!=WINDOW_H) else vis
            dt = time.time()-t0
            cv2.putText(disp, f"Frame {frame_idx}  FPS~{1.0/max(1e-3,dt):.1f}", (12,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow(WIN_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27):
                break
            elif key == ord('p'):
                paused = not paused
            elif paused and key == ord('o'):
                frame_idx = min(frame_idx+1, total-1)
            elif paused and key == ord('i'):
                frame_idx = max(frame_idx-1, 0)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("Wrote:", OUTPUT_CSV)

if __name__ == "__main__":
    run()
