# scripts/4_infer_and_track.py
import os, cv2, csv, numpy as np
from ultralytics import YOLO
from collections import defaultdict
from scripts.utils import ensure_dir

# Simple SORT implementation (minimal)
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        dt = 1.
        self.kf.F = np.array([[1,0,0,0,dt,0,0],
                              [0,1,0,0,0,dt,0],
                              [0,0,1,0,0,0,dt],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.eye(4,7)
        self.kf.R *= 10.
        self.kf.P *= 10.
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2; cy=(y1+y2)/2; s=(x2-x1)*(y2-y1); r=(x2-x1)/(y2-y1+1e-6)
        self.kf.x[:4,0] = np.array([cx,cy,s,r])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak=0
        self.time_since_update += 1
        return self.kf.x

    def update(self, bbox):
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2; cy=(y1+y2)/2; s=(x2-x1)*(y2-y1); r=(x2-x1)/(y2-y1+1e-6)
        z = np.array([cx,cy,s,r])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        x = self.kf.x[:4,0]
        cx,cy,s,r = x
        w = np.sqrt(s*r); h = s / (w+1e-6)
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        return [x1,y1,x2,y2]

# Simple IOU
def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0]); yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2]); yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2-xx1); h=max(0., yy2-yy1)
    inter = w*h
    areaA = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    areaB = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    union = areaA+areaB-inter
    return inter/union if union>0 else 0

class SimpleSORT:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):  # dets: list of [x1,y1,x2,y2,score]
        # Predict
        trks = []
        for t in self.trackers:
            t.predict()
            trks.append(t.get_state())
        matched, unmatched_dets, unmatched_trks = [], list(range(len(dets))), list(range(len(trks)))
        if len(trks)>0 and len(dets)>0:
            iou_mat = np.zeros((len(trks), len(dets)), dtype=np.float32)
            for t_i, trk in enumerate(trks):
                for d_i, det in enumerate(dets):
                    iou_mat[t_i, d_i] = iou(trk, det[:4])
            row, col = linear_sum_assignment(-iou_mat)
            for r, c in zip(row, col):
                if iou_mat[r, c] < self.iou_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append((r, c))
                    if c in unmatched_dets: unmatched_dets.remove(c)
                    if r in unmatched_trks: unmatched_trks.remove(r)
        # update matched
        for t_i, d_i in matched:
            self.trackers[t_i].update(dets[d_i][:4])
        # create new trackers for unmatched detections
        for d in unmatched_dets:
            trk = KalmanBoxTracker(dets[d][:4])
            self.trackers.append(trk)
        # remove dead
        new_trackers=[]
        for trk in self.trackers:
            if trk.time_since_update < self.max_age:
                new_trackers.append(trk)
        self.trackers = new_trackers
        # return for output: list of [x1,y1,x2,y2,id]
        outs=[]
        for trk in self.trackers:
            if (trk.hits >= self.min_hits) or trk.time_since_update==0:
                outs.append(trk.get_state()+[trk.id])
        return outs

# ---- INFERENCE + TRACKING ----
def run(video_path, output_csv, model_path=None, conf=0.25):
    model = YOLO(model_path if model_path else "yolov8s.pt")
    cap = cv2.VideoCapture(video_path)
    tracker = SimpleSORT()
    frame_id = 0
    rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=960, conf=conf, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
            confs = float(box.conf[0])
            dets.append([x1,y1,x2,y2,confs])
        outs = tracker.update(dets)
        for o in outs:
            x1,y1,x2,y2,tid = o
            cx = (x1+x2)/2; cy=(y1+y2)/2
            rows.append([int(tid), frame_id, float(cx), float(cy)])
        frame_id += 1
    cap.release()
    # save CSV
    import pandas as pd
    df = pd.DataFrame(rows, columns=["track_id","frame","x","y"])
    df.to_csv(output_csv, index=False)
    print("Wrote", output_csv)

if __name__ == "__main__":
    video = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\little\video0\video.mp4"
    outcsv = r"C:\Users\morte\ComputerVisionProject\trajectories_video0.csv"
    run(video, outcsv, model_path=None, conf=0.25)
