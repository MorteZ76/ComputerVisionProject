# scripts/5_evaluate_trajectories.py
import pandas as pd
import numpy as np
from scripts.utils import read_annotation_file, CLASS_MAP, get_entry_exit
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
import os

# Convert ground truth annotations to per-frame trackpoints center coords (for a given scene/vfolder)
def gt_to_points(ann_file, frame_offset=0):
    rows = read_annotation_file(ann_file)
    pts = defaultdict(list)  # track_id -> list of (frame, cx, cy)
    for r in rows:
        if r["lost"] == 1:
            continue
        xmin,ymin,xmax,ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        cx = (xmin + xmax)/2.0
        cy = (ymin + ymax)/2.0
        pts[r["track_id"]].append((r["frame"], cx, cy))
    return pts

def compute_mse(pred_csv, ann_file, img_w=None, img_h=None):
    pred = pd.read_csv(pred_csv)
    gt_pts = gt_to_points(ann_file)
    # turn pred into per-frame dict of detections: frame -> list of (trackid, x,y)
    pred_by_frame = defaultdict(list)
    for _,r in pred.iterrows():
        pred_by_frame[int(r['frame'])].append((int(r['track_id']), float(r['x']), float(r['y'])))
    # also ground truth per-frame
    gt_by_frame = defaultdict(list)
    for tid, pts in gt_pts.items():
        for f,x,y in pts:
            gt_by_frame[int(f)].append((tid,x,y))
    # for each frame compute bipartite assignment using euclidean distance and accumulate matched distances
    dists=[]
    for f in sorted(set(list(pred_by_frame.keys()) + list(gt_by_frame.keys()))):
        P = pred_by_frame.get(f, [])
        G = gt_by_frame.get(f, [])
        if len(P)==0 or len(G)==0:
            continue
        Pxy = np.array([[p[1],p[2]] for p in P])
        Gxy = np.array([[g[1],g[2]] for g in G])
        # cost matrix
        C = np.linalg.norm(Pxy[:,None,:] - Gxy[None,:,:], axis=2)
        row,col = linear_sum_assignment(C)
        for i,j in zip(row,col):
            dists.append(C[i,j]**2)  # squared error
    if len(dists)==0:
        return None
    mse = np.mean(dists)
    return mse

def most_frequent_path_from_gt(ann_file, frame_w, frame_h, margin=50):
    gt_pts = gt_to_points(ann_file)
    path_counts = Counter()
    for tid, pts in gt_pts.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        entry = get_entry_exit(pts_sorted[0][1], pts_sorted[0][2], frame_w, frame_h, margin)
        exit_ = get_entry_exit(pts_sorted[-1][1], pts_sorted[-1][2], frame_w, frame_h, margin)
        if entry and exit_:
            path_counts[(entry,exit_)] += 1
    return path_counts.most_common(10)

if __name__ == "__main__":
    # example
    pred_csv = r"C:\Users\morte\ComputerVisionProject\trajectories_video0.csv"
    ann_file = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\little\video0\annotations.txt"
    # you may want to read the image size from a sample frame
    import cv2
    img = cv2.imread(r"C:\Users\morte\ComputerVisionProject\dataset\images\test\little_video0\little_video0_frame000000.jpg")
    h,w = img.shape[:2]
    mse = compute_mse(pred_csv, ann_file)
    print("MSE (pixels^2):", mse)
    print("Top paths (GT):", most_frequent_path_from_gt(ann_file, w,h))
