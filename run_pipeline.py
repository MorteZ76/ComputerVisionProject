import cv2
import pandas as pd
from src.detection import Detector
from src.tracking import Tracker
from src.utils import draw_tracks

video_path = "data/video0.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = Detector()
tracker = Tracker()

trajectories = {}
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)

    for t in tracks:
        track_id = t[4]
        cx = (t[0] + t[2]) / 2
        cy = (t[1] + t[3]) / 2
        trajectories.setdefault(track_id, []).append((frame_id, cx, cy))

    frame = draw_tracks(frame, tracks)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# Save trajectories
df = []
for tid, coords in trajectories.items():
    for f, x, y in coords:
        df.append([tid, f, x, y])
pd.DataFrame(df, columns=["track_id", "frame", "x", "y"]).to_csv("trajectories_video0.csv", index=False)
