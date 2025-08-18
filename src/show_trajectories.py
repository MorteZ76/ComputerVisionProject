# import cv2
# import numpy as np
# import pandas as pd
# from collections import defaultdict, deque

# # --- Inputs ---
# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
# CSV_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\video0_trajectories.csv"

# # --- Viz params ---
# THICKNESS = 2
# POINT_RADIUS = 3
# TRAJ_MAX_LEN = 10000  # unlimited-ish

# def id_to_color(tid: int) -> tuple:
#     """Deterministic vivid color from track id using HSV cycling."""
#     h = (tid * 137) % 180  # golden-angle hop in OpenCV HSV hue space [0,179]
#     s, v = 200, 255
#     hsv = np.uint8([[[h, s, v]]])            # 1x1 HSV
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
#     return int(bgr[0]), int(bgr[1]), int(bgr[2])  # BGR

# def load_trajectories(csv_path):
#     df = pd.read_csv(csv_path)
#     # Expected columns: video_id, track_id, frame, x, y
#     # Convert frames to 0-based to match OpenCV iteration.
#     df["frame"] = df["frame"].astype(int) - 1
#     df["track_id"] = df["track_id"].astype(int)
#     # Build frame -> [(track_id, (x,y)), ...]
#     frames = defaultdict(list)
#     for tid, f, x, y in df[["track_id", "frame", "x", "y"]].itertuples(index=False):
#         frames[f].append((tid, (int(round(x)), int(round(y)))))
#     return frames

# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print(f"Cannot open video: {VIDEO_PATH}")
#         return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames_map = load_trajectories(CSV_PATH)

#     # Per-track accumulated points for polylines
#     paths = defaultdict(lambda: deque(maxlen=TRAJ_MAX_LEN))
#     colors = {}

#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Append any points for this frame
#         for tid, pt in frames_map.get(frame_idx, []):
#             paths[tid].append(pt)
#             if tid not in colors:
#                 colors[tid] = id_to_color(tid)

#         # Draw polylines for all tracks seen so far
#         for tid, pts in paths.items():
#             if len(pts) >= 2:
#                 pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
#                 cv2.polylines(frame, [pts_np], isClosed=False, color=colors[tid], thickness=THICKNESS)
#             # Draw current point dot
#             if len(pts) > 0:
#                 cv2.circle(frame, pts[-1], POINT_RADIUS, colors[tid], -1)

#         # HUD
#         cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

#         cv2.imshow("Trajectories Overlay", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#         frame_idx += 1

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# --- Inputs ---
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
CSV_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\video3_trajectoriesNEW.csv"

# --- Viz params ---
THICKNESS = 2
POINT_RADIUS = 3
TRAJ_MAX_LEN = 10000  # large cap

def id_to_color(tid: int) -> tuple:
    """Deterministic vivid color per track id."""
    h = (tid * 137) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def load_trajectories(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: video_id, track_id, frame, x, y
    df["track_id"] = df["track_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    # Auto-detect 1-based frames and convert to 0-based
    if df["frame"].min() > 0:
        df["frame"] = df["frame"] - 1

    # frame -> [(tid,(x,y)), ...]
    frames_map = defaultdict(list)
    for tid, f, x, y in df[["track_id", "frame", "x", "y"]].itertuples(index=False):
        frames_map[f].append((tid, (int(round(x)), int(round(y)))))

    # last frame per track to know when to remove finished lines
    last_frame = df.groupby("track_id")["frame"].max().to_dict()
    return frames_map, last_frame

def main():
    frames_map, last_frame_by_tid = load_trajectories(CSV_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, base = cap.read()
    if not ret:
        print("Failed to read first frame.")
        return

    # State
    frame_idx = 0
    paused = False
    paths = defaultdict(lambda: deque(maxlen=TRAJ_MAX_LEN))
    colors = {}

    # Add points for initial frame
    for tid, pt in frames_map.get(frame_idx, []):
        paths[tid].append(pt)
        if tid not in colors:
            colors[tid] = id_to_color(tid)

    window = "Trajectories Overlay"
    while True:
        # Draw overlay on a fresh copy of the current base frame
        frame = base.copy()

        # Remove tracks that are finished (past their last frame)
        finished = [tid for tid in list(paths.keys()) if frame_idx > last_frame_by_tid.get(tid, -1)]
        for tid in finished:
            paths.pop(tid, None)
            colors.pop(tid, None)

        # Draw polylines for active tracks
        for tid, pts in paths.items():
            if len(pts) >= 2:
                pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts_np], isClosed=False, color=colors[tid], thickness=THICKNESS)
            if len(pts) > 0:
                cv2.circle(frame, pts[-1], POINT_RADIUS, colors[tid], -1)

        cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}  {'PAUSED' if paused else 'PLAY'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}  {'PAUSED' if paused else 'PLAY'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window, frame)
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            paused = True
        elif key == ord('r'):
            paused = False

        # Advance only when playing
        if not paused:
            frame_idx += 1
            if frame_idx >= total_frames:
                break
            ret, base = cap.read()
            if not ret:
                break
            # Append points for the new frame
            for tid, pt in frames_map.get(frame_idx, []):
                # Ensure color exists
                if tid not in colors:
                    colors[tid] = id_to_color(tid)
                # Add point
                paths[tid].append(pt)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
