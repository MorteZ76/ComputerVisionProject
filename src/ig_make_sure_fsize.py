import os
import cv2

def check_frame_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_size = None
    frame_count = 0
    mismatch_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        current_size = frame.shape[:2]  # (height, width)
        print (f"Frame {frame_count}: size = {current_size}")
        if frame_size is None:
            frame_size = current_size
        elif current_size != frame_size:
            print(f"Frame size mismatch at frame {frame_count}: {current_size} != {frame_size}")
            mismatch_found = True
            break

    cap.release()

    if not mismatch_found:
        print(f"All {frame_count} frames have the same size: {frame_size}")

video_path = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
check_frame_size(video_path)
