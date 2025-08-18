import cv2
import numpy as np
import pandas as pd
from src.utils import * 

def main(video_num, fps=32):
    # Get paths based on video number
    video_path, annotation_path = get_old_video_path(video_num)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Load annotations
    annotations = load_old_annotations(annotation_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video dimensions: {frame_width}x{frame_height}")
    
    # Calculate scaling factors if needed
    max_annotation_x = annotations[['xmin', 'xmax']].max().max()
    max_annotation_y = annotations[['ymin', 'ymax']].max().max()
    scale_x = frame_width / max_annotation_x if max_annotation_x > frame_width else 1
    scale_y = frame_height / max_annotation_y if max_annotation_y > frame_height else 1
    print(f"Scaling factors: x={scale_x:.2f}, y={scale_y:.2f}")
    
    frame_number = 0
    frame_by_frame = False
    frames_cache = {}
    
    while True:
        if frame_number not in frames_cache:
            # Read new frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                if frame_number > 0:
                    frame_number -= 1
                continue
            
            # Process frame and store in cache
            annotated_frame = process_frame(frame, frame_number, annotations, 
                                         frame_width, frame_height, scale_x, scale_y, fps)
            frames_cache[frame_number] = (frame, annotated_frame)
            
            # Limit cache size
            if len(frames_cache) > 100:
                min_key = min(frames_cache.keys())
                frames_cache.pop(min_key)
        else:
            # Get frame from cache
            frame, annotated_frame = frames_cache[frame_number]

        # Show annotated frame only
        cv2.imshow('Annotated Video', annotated_frame)

        # Handle keyboard input
        key = cv2.waitKey(1 if not frame_by_frame else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            frame_by_frame = not frame_by_frame
            print(f"{'Frame-by-Frame' if frame_by_frame else 'Normal Playback'} mode")
        elif frame_by_frame and key == ord('k'):
            frame_number = min(frame_number + 1, total_frames - 1)
        elif frame_by_frame and key == ord('j'):
            frame_number = max(frame_number - 1, 0)
        elif frame_by_frame and key == ord('o'):  # Skip forward 10 seconds
            frames_to_skip = fps * 10  # fps * 10 seconds
            frame_number = min(frame_number + frames_to_skip, total_frames - 1)
        elif frame_by_frame and key == ord('i'):  # Skip backward 10 seconds
            frames_to_skip = fps * 10  # fps * 10 seconds
            frame_number = max(frame_number - frames_to_skip, 0)
        elif not frame_by_frame:
            frame_number += 1
        
        print(frame.shape)
        # Handle video end
        if frame_number >= total_frames:
            frame_number = total_frames - 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change these values to customize the video playback
    VIDEO_NUM = 1  # 0 or 1 to select which video to play
    FPS = 32      # Frames per second of the video
    main(VIDEO_NUM, FPS)
