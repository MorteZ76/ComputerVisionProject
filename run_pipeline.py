import cv2
import pandas as pd
import sys
from src.detection import Detector
from src.utils import * 

# from src.tracking import Tracker
# from src.utils import draw_tracks

print("Starting pipeline...")

video_path = "data/video0.mp4"
print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {frame_width}x{frame_height}")

print("Initializing detector...")
try:
    detector = Detector()
except Exception as e:
    print(f"Error initializing detector: {str(e)}")
    sys.exit(1)
# tracker = Tracker()

trajectories = {}
frame_id = 0

print("Starting frame processing...")
while True:
    try:
        ret, frame = cap.read()
        if not ret: 
            print("End of video reached")
            break

        print(f"Processing frame {frame_id}")
        try:
            detections = detector.detect(frame)
            print(f"Found {len(detections)} detections")
            
            # Draw detection boxes with class labels
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Get color based on class
                color = get_class_color(cls)
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw label with confidence
                label = f"{cls}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the frame
            cv2.imshow("Detections", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("User requested to quit")
                break
                
        except Exception as e:
            print(f"Error in detection for frame {frame_id}: {str(e)}")
            break

        # tracks = tracker.update(detections, frame)

        # for t in tracks:
        #     track_id = t[4]
        #     cx = (t[0] + t[2]) / 2
        #     cy = (t[1] + t[3]) / 2
        #     trajectories.setdefault(track_id, []).append((frame_id, cx, cy))

        # frame = draw_tracks(frame, tracks)
        frame_id += 1
    except Exception as e:
        print(f"Unexpected error in main loop: {str(e)}")
        break
    img = draw_bounding_boxes(frame, detections)
    cv2.imshow('Detection Results', img)

cap.release()
cv2.destroyAllWindows()

# # Save trajectories
# df = []
# for tid, coords in trajectories.items():
#     for f, x, y in coords:
#         df.append([tid, f, x, y])
# pd.DataFrame(df, columns=["track_id", "frame", "x", "y"]).to_csv("trajectories_video0.csv", index=False)
