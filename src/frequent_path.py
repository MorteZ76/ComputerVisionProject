import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# ===== CONFIG =====
CSV_FILES: List[str] = [
    r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\video0_trajectories.csv",
    r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\video3_trajectories.csv",
]
VIDEO_PATHS: Dict[str, str] = {
    "video0": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4",
    "video3": r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4",
}

AREA_ROWS_3 = ["top", "middle", "bottom"]
AREA_COLS_3 = ["left", "center", "right"]
AREA_ROWS_2 = ["top", "bottom"]
AREA_COLS_2 = ["left", "right"]
SIDE_ORDER = ["left", "right", "top", "bottom"]  # side->side paths

# ===== HELPERS =====
def load_video_size(path: str) -> Optional[Tuple[int, int]]:
    if not path: return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if (w > 0 and h > 0) else None

def infer_size_from_csv(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty: return 1, 1
    w = int(np.ceil(df["x"].max())); h = int(np.ceil(df["y"].max()))
    return max(w,1), max(h,1)

def bin_to_cell(x: float, y: float, w: int, h: int, grid: int) -> Tuple[int,int,int]:
    col = min(grid-1, max(0, int(grid * x / max(1, w))))
    row = min(grid-1, max(0, int(grid * y / max(1, h))))
    return row, col, row * grid + col

def cell_name(idx: int, grid: int) -> str:
    r, c = divmod(idx, grid)
    if grid == 3:
        return f"{AREA_ROWS_3[r]}-{AREA_COLS_3[c]}"
    else:
        return f"{AREA_ROWS_2[r]}-{AREA_COLS_2[c]}"

def side_of_point(x: float, y: float, w: int, h: int) -> str:
    d = {"left": x, "right": max(w-1,0) - x, "top": y, "bottom": max(h-1,0) - y}
    return min(SIDE_ORDER, key=lambda k: (d[k], SIDE_ORDER.index(k)))

def compute_counts(df_vid: pd.DataFrame, wh: Tuple[int,int]):
    W, H = wh
    df = df_vid.copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df.sort_values(["track_id", "frame"], inplace=True)

    firsts = df.groupby("track_id").first()[["x", "y"]]
    lasts  = df.groupby("track_id").last()[["x", "y"]]

    start_3 = np.zeros((3,3), dtype=int)
    end_3   = np.zeros((3,3), dtype=int)
    pair_9  = np.zeros((9,9), dtype=int)      # start_cell -> end_cell, excluding diagonal

    start_2 = np.zeros((2,2), dtype=int)
    end_2   = np.zeros((2,2), dtype=int)
    pair_4c = np.zeros((4,4), dtype=int)      # 2x2 cell pairs, excluding diagonal

    side_idx = {s:i for i,s in enumerate(SIDE_ORDER)}
    side_4   = np.zeros((4,4), dtype=int)     # side->side, excluding same side

    for tid, (sx, sy) in firsts.iterrows():
        ex, ey = lasts.loc[tid, ["x", "y"]]

        # 3x3
        r1, c1, i1 = bin_to_cell(float(sx), float(sy), W, H, 3)
        r2, c2, i2 = bin_to_cell(float(ex), float(ey), W, H, 3)
        start_3[r1, c1] += 1
        end_3[r2, c2]   += 1
        if i1 != i2:
            pair_9[i1, i2] += 1

        # 2x2
        r1q, c1q, i1q = bin_to_cell(float(sx), float(sy), W, H, 2)
        r2q, c2q, i2q = bin_to_cell(float(ex), float(ey), W, H, 2)
        start_2[r1q, c1q] += 1
        end_2[r2q, c2q]   += 1
        if i1q != i2q:
            pair_4c[i1q, i2q] += 1

        # sides (exclude same start=end)
        ss = side_of_point(float(sx), float(sy), W, H)
        es = side_of_point(float(ex), float(ey), W, H)
        if ss != es:
            side_4[side_idx[ss], side_idx[es]] += 1

    return start_3, end_3, pair_9, start_2, end_2, pair_4c, side_4

def most_frequent_side_path(side_4: np.ndarray) -> Tuple[Optional[str],Optional[str],int]:
    if side_4.sum() == 0:
        return None, None, 0
    i, j = np.unravel_index(np.argmax(side_4), side_4.shape)
    return SIDE_ORDER[i], SIDE_ORDER[j], int(side_4[i,j])

def mat_str_3x3(m: np.ndarray) -> str:
    lines = ["rows=top/middle/bottom, cols=left/center/right"]
    for r in range(3):
        lines.append(f"{AREA_ROWS_3[r]:>6}: " + "  ".join(f"{int(m[r,c]):4d}" for c in range(3)))
    return "\n".join(lines)

def mat_str_2x2(m: np.ndarray) -> str:
    lines = ["rows=top/bottom, cols=left/right"]
    for r in range(2):
        lines.append(f"{AREA_ROWS_2[r]:>6}: " + "  ".join(f"{int(m[r,c]):4d}" for c in range(2)))
    return "\n".join(lines)

def top_k_pairs(pair_mat: np.ndarray, grid: int, k: int = 3) -> List[Tuple[int,int,int]]:
    flat = []
    n = pair_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:  # excluded
                continue
            cnt = int(pair_mat[i,j])
            if cnt > 0:
                flat.append((cnt, i, j))
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(i, j, cnt) for cnt, i, j in flat[:k]]

def print_report(tag: str,
                 start_3: np.ndarray, end_3: np.ndarray, pair_9: np.ndarray,
                 start_2: np.ndarray, end_2: np.ndarray, pair_4c: np.ndarray,
                 side_4: np.ndarray):
    s_side, e_side, cnt = most_frequent_side_path(side_4)
    print(f"\n=== {tag} ===")
    print("Start 3x3:"); print(mat_str_3x3(start_3))
    print("End   3x3:"); print(mat_str_3x3(end_3))
    print("Start 2x2:"); print(mat_str_2x2(start_2))
    print("End   2x2:"); print(mat_str_2x2(end_2))
    if s_side is None:
        print("Most frequent path (sides, excluding same): none")
    else:
        print(f"Most frequent path (sides, excluding same): enter {s_side} and exit {e_side} (n={cnt})")

    top3_9  = top_k_pairs(pair_9, grid=3, k=3)
    print("Top 3 start→end 3x3 pairs (excluding same cell):")
    for i, j, c in top3_9:
        print(f"  {cell_name(i,3)} → {cell_name(j,3)} : n={c}")

    top3_2  = top_k_pairs(pair_4c, grid=2, k=3)
    print("Top 3 start→end 2x2 pairs (excluding same cell):")
    for i, j, c in top3_2:
        print(f"  {cell_name(i,2)} → {cell_name(j,2)} : n={c}")

# ===== MAIN =====
def main():
    dfs = []
    for p in CSV_FILES:
        if not Path(p).is_file():
            print(f"Warning: missing CSV: {p}"); continue
        df = pd.read_csv(p)
        need = {"video_id","track_id","frame","x","y"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        dfs.append(df[["video_id","track_id","frame","x","y"]])
    if not dfs:
        print("No CSVs loaded."); return
    data = pd.concat(dfs, ignore_index=True)

    # Per video
    results = {}
    for vid, df_vid in data.groupby("video_id"):
        size = load_video_size(VIDEO_PATHS.get(vid,"")) or infer_size_from_csv(df_vid)
        results[vid] = compute_counts(df_vid, size)

    # Print per video
    for vid in sorted(results.keys()):
        print_report(vid, *results[vid])

    # ALL
    size_all = infer_size_from_csv(data)
    all_counts = compute_counts(data, size_all)
    print_report("ALL", *all_counts)

if __name__ == "__main__":
    main()
