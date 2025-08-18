import cv2
import json
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
OUTPUT_XLSX = "new_video0_trajectory_summary.xlsx"
OUTPUT_JSON = "new_video0_trajectory_summary.json"

AREA_ROWS_3 = ["top", "middle", "bottom"]
AREA_COLS_3 = ["left", "center", "right"]
AREA_ROWS_2 = ["top", "bottom"]
AREA_COLS_2 = ["left", "right"]
SIDE_ORDER = ["left", "right", "top", "bottom"]  # side->side paths

# ===== HELPERS =====
def load_video_size(path: str) -> Optional[Tuple[int, int]]:
    if not path:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if (w > 0 and h > 0) else None

def infer_size_from_csv(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty:
        return 1, 1
    w = int(np.ceil(df["x"].max()))
    h = int(np.ceil(df["y"].max()))
    return max(w, 1), max(h, 1)

def bin_to_cell(x: float, y: float, w: int, h: int, grid: int) -> Tuple[int, int, int]:
    col = min(grid - 1, max(0, int(grid * x / max(1, w))))
    row = min(grid - 1, max(0, int(grid * y / max(1, h))))
    return row, col, row * grid + col

def cell_name(idx: int, grid: int) -> str:
    r, c = divmod(idx, grid)
    if grid == 3:
        return f"{AREA_ROWS_3[r]}-{AREA_COLS_3[c]}"
    else:
        return f"{AREA_ROWS_2[r]}-{AREA_COLS_2[c]}"

def side_of_point(x: float, y: float, w: int, h: int) -> str:
    d = {"left": x, "right": max(w - 1, 0) - x, "top": y, "bottom": max(h - 1, 0) - y}
    return min(SIDE_ORDER, key=lambda k: (d[k], SIDE_ORDER.index(k)))

def compute_counts(df_vid: pd.DataFrame, wh: Tuple[int, int]):
    W, H = wh
    df = df_vid.copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df.sort_values(["track_id", "frame"], inplace=True)

    firsts = df.groupby("track_id").first()[["x", "y"]]
    lasts = df.groupby("track_id").last()[["x", "y"]]

    start_3 = np.zeros((3, 3), dtype=int)
    end_3 = np.zeros((3, 3), dtype=int)
    pair_9 = np.zeros((9, 9), dtype=int)      # exclude same start=end

    start_2 = np.zeros((2, 2), dtype=int)
    end_2 = np.zeros((2, 2), dtype=int)
    pair_4c = np.zeros((4, 4), dtype=int)     # exclude same start=end

    side_idx = {s: i for i, s in enumerate(SIDE_ORDER)}
    side_4 = np.zeros((4, 4), dtype=int)      # exclude same start=end

    for tid, (sx, sy) in firsts.iterrows():
        ex, ey = lasts.loc[tid, ["x", "y"]]

        r1, c1, i1 = bin_to_cell(float(sx), float(sy), W, H, 3)
        r2, c2, i2 = bin_to_cell(float(ex), float(ey), W, H, 3)
        start_3[r1, c1] += 1
        end_3[r2, c2] += 1
        if i1 != i2:
            pair_9[i1, i2] += 1

        r1q, c1q, i1q = bin_to_cell(float(sx), float(sy), W, H, 2)
        r2q, c2q, i2q = bin_to_cell(float(ex), float(ey), W, H, 2)
        start_2[r1q, c1q] += 1
        end_2[r2q, c2q] += 1
        if i1q != i2q:
            pair_4c[i1q, i2q] += 1

        ss = side_of_point(float(sx), float(sy), W, H)
        es = side_of_point(float(ex), float(ey), W, H)
        if ss != es:
            side_4[side_idx[ss], side_idx[es]] += 1

    return start_3, end_3, pair_9, start_2, end_2, pair_4c, side_4

def most_frequent_side_path(side_4: np.ndarray):
    if side_4.sum() == 0:
        return None, None, 0
    i, j = np.unravel_index(np.argmax(side_4), side_4.shape)
    return SIDE_ORDER[i], SIDE_ORDER[j], int(side_4[i, j])

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

def top_k_pairs(pair_mat: np.ndarray, grid: int, k: int = 3) -> List[Tuple[int, int, int]]:
    flat = []
    n = pair_mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cnt = int(pair_mat[i, j])
            if cnt > 0:
                flat.append((cnt, i, j))
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(i, j, cnt) for cnt, i, j in flat[:k]]

def df3(m: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(m, index=AREA_ROWS_3, columns=AREA_COLS_3)

def df2(m: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(m, index=AREA_ROWS_2, columns=AREA_COLS_2)

def df_sides(m: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(m, index=SIDE_ORDER, columns=SIDE_ORDER)

def save_excel(results: Dict[str, dict], path: str):
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # Summary sheet
        summary_rows = []
        for tag, R in results.items():
            enter, exit_, n = R["most_frequent_side_path"]
            summary_rows.append({
                "video": tag,
                "enter_side": enter if enter is not None else "",
                "exit_side": exit_ if exit_ is not None else "",
                "count": n
            })
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="summary", index=False)

        # Per-video sheets
        for tag, R in results.items():
            # separate sheets for clarity
            df3(R["start_3x3"]).to_excel(xw, sheet_name=f"{tag}_start_3x3")
            df3(R["end_3x3"]).to_excel(xw, sheet_name=f"{tag}_end_3x3")
            # 9x9 with labels for start/end cells
            idx9 = [cell_name(i, 3) for i in range(9)]
            pd.DataFrame(R["pair_9x9"], index=idx9, columns=idx9).to_excel(xw, sheet_name=f"{tag}_pair_9x9")

            df2(R["start_2x2"]).to_excel(xw, sheet_name=f"{tag}_start_2x2")
            df2(R["end_2x2"]).to_excel(xw, sheet_name=f"{tag}_end_2x2")
            idx4 = [cell_name(i, 2) for i in range(4)]
            pd.DataFrame(R["pair_2x2"], index=idx4, columns=idx4).to_excel(xw, sheet_name=f"{tag}_pair_2x2")

            df_sides(R["side_4x4"]).to_excel(xw, sheet_name=f"{tag}_sides_4x4")

            # Top-3 tables
            top3_3 = pd.DataFrame([{
                "start_cell": cell_name(i, 3),
                "end_cell": cell_name(j, 3),
                "count": c
            } for i, j, c in R["top3_pairs_3x3"]])
            top3_2 = pd.DataFrame([{
                "start_cell": cell_name(i, 2),
                "end_cell": cell_name(j, 2),
                "count": c
            } for i, j, c in R["top3_pairs_2x2"]])
            top3_3.to_excel(xw, sheet_name=f"{tag}_top3_3x3", index=False)
            top3_2.to_excel(xw, sheet_name=f"{tag}_top3_2x2", index=False)

def save_json(results: Dict[str, dict], path: str):
    def ndarray_to_list(a): return a.astype(int).tolist()
    payload = {"videos": {}}
    for tag, R in results.items():
        payload["videos"][tag] = {
            "frame_size": R["frame_size"],
            "counts": {
                "start_3x3": ndarray_to_list(R["start_3x3"]),
                "end_3x3": ndarray_to_list(R["end_3x3"]),
                "pair_9x9_excl_same": ndarray_to_list(R["pair_9x9"]),
                "start_2x2": ndarray_to_list(R["start_2x2"]),
                "end_2x2": ndarray_to_list(R["end_2x2"]),
                "pair_2x2_excl_same": ndarray_to_list(R["pair_2x2"]),
                "side_4x4_excl_same": ndarray_to_list(R["side_4x4"]),
            },
            "most_frequent_side_path_excl_same": {
                "enter": R["most_frequent_side_path"][0],
                "exit": R["most_frequent_side_path"][1],
                "count": R["most_frequent_side_path"][2],
            },
            "top3_pairs_3x3_excl_same": [{
                "start_cell": cell_name(i, 3),
                "end_cell": cell_name(j, 3),
                "count": c
            } for i, j, c in R["top3_pairs_3x3"]],
            "top3_pairs_2x2_excl_same": [{
                "start_cell": cell_name(i, 2),
                "end_cell": cell_name(j, 2),
                "count": c
            } for i, j, c in R["top3_pairs_2x2"]],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def print_report(tag: str,
                 start_3: np.ndarray, end_3: np.ndarray, pair_9: np.ndarray,
                 start_2: np.ndarray, end_2: np.ndarray, pair_4c: np.ndarray,
                 side_4: np.ndarray):
    enter, exit_, cnt = most_frequent_side_path(side_4)
    print(f"\n=== {tag} ===")
    print("Start 3x3:"); print(mat_str_3x3(start_3))
    print("End   3x3:"); print(mat_str_3x3(end_3))
    print("Start 2x2:"); print(mat_str_2x2(start_2))
    print("End   2x2:"); print(mat_str_2x2(end_2))
    if enter is None:
        print("Most frequent path (sides, excl same): none")
    else:
        print(f"Most frequent path (sides, excl same): enter {enter} and exit {exit_} (n={cnt})")
    top3_9 = top_k_pairs(pair_9, grid=3, k=3)
    print("Top 3 start→end 3x3 pairs (excl same):")
    for i, j, c in top3_9:
        print(f"  {cell_name(i,3)} → {cell_name(j,3)} : n={c}")
    top3_2 = top_k_pairs(pair_4c, grid=2, k=3)
    print("Top 3 start→end 2x2 pairs (excl same):")
    for i, j, c in top3_2:
        print(f"  {cell_name(i,2)} → {cell_name(j,2)} : n={c}")

# ===== MAIN =====
def main():
    # Load CSVs
    dfs = []
    for p in CSV_FILES:
        if not Path(p).is_file():
            print(f"Warning: missing CSV: {p}")
            continue
        df = pd.read_csv(p)
        need = {"video_id", "track_id", "frame", "x", "y"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        dfs.append(df[["video_id", "track_id", "frame", "x", "y"]])
    if not dfs:
        print("No CSVs loaded.")
        return
    data = pd.concat(dfs, ignore_index=True)

    results: Dict[str, dict] = {}

    # Per video
    for vid, df_vid in data.groupby("video_id"):
        size = load_video_size(VIDEO_PATHS.get(vid, "")) or infer_size_from_csv(df_vid)
        s3, e3, p9, s2, e2, p4c, sides = compute_counts(df_vid, size)
        enter, exit_, cnt = most_frequent_side_path(sides)
        results[vid] = {
            "frame_size": list(size),
            "start_3x3": s3, "end_3x3": e3, "pair_9x9": p9,
            "start_2x2": s2, "end_2x2": e2, "pair_2x2": p4c,
            "side_4x4": sides,
            "most_frequent_side_path": (enter, exit_, cnt),
            "top3_pairs_3x3": top_k_pairs(p9, grid=3, k=3),
            "top3_pairs_2x2": top_k_pairs(p4c, grid=2, k=3),
        }
        print_report(vid, s3, e3, p9, s2, e2, p4c, sides)

    # ALL
    size_all = infer_size_from_csv(data)
    s3, e3, p9, s2, e2, p4c, sides = compute_counts(data, size_all)
    enter, exit_, cnt = most_frequent_side_path(sides)
    results["ALL"] = {
        "frame_size": list(size_all),
        "start_3x3": s3, "end_3x3": e3, "pair_9x9": p9,
        "start_2x2": s2, "end_2x2": e2, "pair_2x2": p4c,
        "side_4x4": sides,
        "most_frequent_side_path": (enter, exit_, cnt),
        "top3_pairs_3x3": top_k_pairs(p9, grid=3, k=3),
        "top3_pairs_2x2": top_k_pairs(p4c, grid=2, k=3),
    }
    print_report("ALL", s3, e3, p9, s2, e2, p4c, sides)

    # Save files
    save_excel(results, OUTPUT_XLSX)
    save_json(results, OUTPUT_JSON)
    print(f"\nSaved: {OUTPUT_XLSX}")
    print(f"Saved: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
