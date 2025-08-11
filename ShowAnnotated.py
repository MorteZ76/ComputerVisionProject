import cv2
import numpy as np
import pandas as pd

def load_annotations(annotation_path):
    # Read annotations file
    columns = ['track_id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
    annotations = pd.read_csv(annotation_path, header=None, names=columns, sep='\s+')  # Use whitespace separator
    print(f"Total annotations loaded: {len(annotations)}")
    print("First few annotations:")
    print(annotations.head())
    return annotations

def get_video_path(video_num):
    base_path = r"C:\Users\morte\Desktop\Computer Vision\Project Descryption\Videos and Annotations"
    if video_num == 0:
        video_path = fr"{base_path}\video0\video.mp4"
        annotation_path = fr"{base_path}\video0\annotations.txt"
    elif video_num == 1:
        video_path = fr"{base_path}\video3\video.mp4"
        annotation_path = fr"{base_path}\video3\annotations.txt"
    else:
        raise ValueError("Video number must be 0 or 1")
    
    return video_path, annotation_path

def get_class_color(label):
    # Different colors for different classes
    # Format: (B, G, R) for OpenCV
    color_map = {
        'Biker': (0, 255, 255),    # Yellow
        'Pedestrian': (0, 255, 0),     # Green
        'Skater': (255, 0, 255), # Magenta
        'Cart': (255, 128, 0),         # Light Blue
        'Car': (255, 0, 0),            # Blue
        'Bus': (0, 128, 255),          # Orange
    }
    # Remove quotes from label if present and get color, default to white if label not found
    clean_label = label.strip('"')
    return color_map.get(clean_label, (255, 255, 255))

def get_state_color(lost, occluded, generated):
    # Different colors for different combinations
    # Format: (B, G, R) for OpenCV
    if lost == 1:
        return (128, 128, 128)  # Gray for lost objects
    elif occluded == 1 and generated == 1:
        return (255, 165, 0)    # Orange for occluded + generated
    elif occluded == 1:
        return (0, 0, 255)      # Red for occluded only
    elif generated == 1:
        return (255, 0, 255)    # Purple for generated only
    else:
        return (0, 255, 0)      # Green for normal tracking

def draw_state_legend(frame):
    # Define legend position and formatting
    start_x = frame.shape[1] - 250  # 250 pixels from right
    start_y = 30
    font_scale = 0.4  # Reduced from 0.5
    thickness = 1
    line_height = 15  # Reduced from 20
    
    # Draw legend background
    cv2.rectangle(frame, (start_x - 10, start_y - 25), 
                 (frame.shape[1] - 10, start_y + 5 * line_height),
                 (0, 0, 0), -1)  # Black background
    
    # Draw legend entries
    legend_items = [
        ("Normal Tracking", (0, 255, 0)),
        ("Lost Object", (128, 128, 128)),
        ("Occluded", (0, 0, 255)),
        ("Generated", (255, 0, 255)),
        ("Occluded + Generated", (255, 165, 0))
    ]
    
    cv2.putText(frame, "State Colors:", (start_x - 10, start_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    for i, (text, color) in enumerate(legend_items):
        y = start_y + i * line_height
        # Draw color box
        cv2.rectangle(frame, (start_x, y - 10), (start_x + 20, y + 5), color, -1)
        # Draw text
        cv2.putText(frame, text, (start_x + 30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def draw_class_legend(frame):
    # Define legend position and formatting
    start_x = 30  # 30 pixels from left
    start_y = 30
    font_scale = 0.4  # Reduced from 0.5
    thickness = 1
    line_height = 15  # Reduced from 20
    
    # Class colors
    class_colors = [
        ("Biker", (0, 255, 255)),     # Yellow
        ("Pedestrian", (0, 255, 0)),      # Green
        ("Skater", (255, 0, 255)),  # Magenta
        ("Cart", (255, 128, 0)),          # Light Blue
        ("Car", (255, 0, 0)),             # Blue
        ("Bus", (0, 128, 255))            # Orange
    ]
    
    # Draw legend background
    cv2.rectangle(frame, (start_x - 10, start_y - 25), 
                 (start_x + 200, start_y + len(class_colors) * line_height),
                 (0, 0, 0), -1)  # Black background
    
    cv2.putText(frame, "Class Colors:", (start_x - 10, start_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    for i, (text, color) in enumerate(class_colors):
        y = start_y + i * line_height
        # Draw color box
        cv2.rectangle(frame, (start_x, y - 10), (start_x + 20, y + 5), color, -1)
        # Draw text
        cv2.putText(frame, text, (start_x + 30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def draw_hotkeys_legend(frame):
    # Define legend position and formatting
    start_x = 30  # 30 pixels from left
    start_y = frame.shape[0] - 100  # 100 pixels from bottom
    font_scale = 0.4
    thickness = 1
    line_height = 15
    
    # Hotkey descriptions
    hotkeys = [
        ("p", "Play/Pause"),
        ("j", "Previous Frame"),
        ("k", "Next Frame"),
        ("i", "Back 10 seconds"),
        ("o", "Forward 10 seconds"),
        ("q", "Quit")
    ]
    
    # Calculate background width based on longest text
    max_width = max([len(f"{key}: {desc}") for key, desc in hotkeys]) * 8
    
    # Draw legend background
    cv2.rectangle(frame, (start_x - 10, start_y - 25), 
                 (start_x + max_width, start_y + len(hotkeys) * line_height),
                 (0, 0, 0), -1)  # Black background
    
    cv2.putText(frame, "Keyboard Controls:", (start_x - 10, start_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    for i, (key, desc) in enumerate(hotkeys):
        y = start_y + i * line_height
        text = f"{key}: {desc}"
        cv2.putText(frame, text, (start_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def process_frame(frame, frame_number, annotations, frame_width, frame_height, scale_x, scale_y, fps):
    # Create a copy of the frame for drawing bounding boxes
    annotated_frame = frame.copy()
    
    # Get annotations for current frame
    frame_annotations = annotations[annotations['frame'] == frame_number]
    
    if len(frame_annotations) > 0:
        print(f"Frame {frame_number}: Found {len(frame_annotations)} annotations")
    
    # Draw bounding boxes
    for _, box in frame_annotations.iterrows():
        try:
            # Scale coordinates if needed
            xmin = int(box['xmin'] * scale_x)
            ymin = int(box['ymin'] * scale_y)
            xmax = int(box['xmax'] * scale_x)
            ymax = int(box['ymax'] * scale_y)
            
            # Ensure coordinates are within frame boundaries
            xmin = max(0, min(xmin, frame_width - 1))
            ymin = max(0, min(ymin, frame_height - 1))
            xmax = max(0, min(xmax, frame_width - 1))
            ymax = max(0, min(ymax, frame_height - 1))
            
            # Get colors based on state and class
            state_color = get_state_color(box['lost'], box['occluded'], box['generated'])
            class_color = get_class_color(box['label'])
            
            # Draw rectangle for state
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), state_color, 2)
            
            # Add track ID and label with better visibility
            label_y = max(ymin - 25, 20)  # Move text up a bit to accommodate both lines
            # Draw track ID
            cv2.putText(annotated_frame, f"ID: {int(box['track_id'])}", 
                      (xmin, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Draw class label
            label_text = box['label'].strip('"')
            cv2.putText(annotated_frame, label_text, 
                      (xmin, label_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
        except Exception as e:
            print(f"Error drawing box: {e}")
    
    # Add legends to frame
    draw_state_legend(annotated_frame)
    draw_class_legend(annotated_frame)
    draw_hotkeys_legend(annotated_frame)
    
    # Calculate time information
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    # Add frame number and time to the window
    time_text = f"Frame: {frame_number} | Time: {minutes:02d}:{seconds:02d}"
    
    # Get text size to center it
    (text_width, text_height), _ = cv2.getTextSize(time_text, 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1)  # Reduced font scale and thickness
    text_x = (frame_width - text_width) // 2
    
    # Draw black background for better visibility
    cv2.rectangle(annotated_frame, 
                 (text_x - 10, 8), 
                 (text_x + text_width + 10, 30),  # Reduced height of background
                 (0, 0, 0), -1)
    
    # Add centered text
    cv2.putText(annotated_frame, time_text,
              (text_x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Adjusted y-position, reduced scale and thickness
    
    return annotated_frame

def main(video_num, fps=32):
    # Get paths based on video number
    video_path, annotation_path = get_video_path(video_num)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Load annotations
    annotations = load_annotations(annotation_path)
    
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
