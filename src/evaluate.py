import pandas as pd
import numpy as np

def mse(pred, gt):
    return np.mean((np.array(pred) - np.array(gt)) ** 2)

def evaluate(pred_csv, gt_csv):
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    # Match track IDs manually or via Hungarian algorithm
    # For now, assume IDs are aligned in CSV
    mse_values = []
    for tid in pred_df.track_id.unique():
        pred_points = pred_df[pred_df.track_id == tid][["x", "y"]].values
        gt_points = gt_df[gt_df.track_id == tid][["x", "y"]].values
        if len(pred_points) == len(gt_points):
            mse_values.append(mse(pred_points, gt_points))
    return np.mean(mse_values)
