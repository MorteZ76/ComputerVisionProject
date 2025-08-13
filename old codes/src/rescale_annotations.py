import cv2
import os

# Paths
video_path = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\little\video3\video.mp4"
ann_path = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\little\video3\annotations.txt"
# Create output path in the same directory as the input annotation file
output_path = os.path.join(os.path.dirname(ann_path), "annotations_rescaled.txt")

# Read first frame of video to get actual dimensions
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
if not success:
    raise RuntimeError("Failed to read first frame from video")
actual_h, actual_w = frame.shape[:2]
cap.release()

# First pass: determine max values in original annotations
max_ann_x = 0
max_ann_y = 0
all_lines = []

with open(ann_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 11:
            continue  # skip malformed lines

        try:
            xmin = int(parts[1])
            ymin = int(parts[2])
            xmax = int(parts[3])
            ymax = int(parts[4])
        except ValueError:
            continue  # skip lines with non-integer bbox values

        max_ann_x = max(max_ann_x, xmax)
        max_ann_y = max(max_ann_y, ymax)
        all_lines.append(parts)

# Compute scaling factors - always scale to match video dimensions
scale_x = actual_w / max_ann_x
scale_y = actual_h / max_ann_y

# Print scaling information
print(f"Video dimensions: {actual_w}x{actual_h}")
print(f"Original annotation max coordinates: x={max_ann_x}, y={max_ann_y}")
print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

# Validate scale factors
if scale_x <= 0 or scale_y <= 0:
    raise ValueError("Invalid scale factors calculated. Check if annotations contain valid coordinates.")

# Apply scaling and write updated annotations
with open(output_path, 'w') as out:
    for parts in all_lines:
        try:
            # Scale coordinates
            parts[1] = str(int(int(parts[1]) * scale_x))  # xmin
            parts[2] = str(int(int(parts[2]) * scale_y))  # ymin
            parts[3] = str(int(int(parts[3]) * scale_x))  # xmax
            parts[4] = str(int(int(parts[4]) * scale_y))  # ymax
            
            # Ensure coordinates are within frame bounds
            parts[1] = str(max(0, min(actual_w, int(parts[1]))))  # xmin
            parts[2] = str(max(0, min(actual_h, int(parts[2]))))  # ymin
            parts[3] = str(max(0, min(actual_w, int(parts[3]))))  # xmax
            parts[4] = str(max(0, min(actual_h, int(parts[4]))))  # ymax
        except ValueError as e:
            print(f"Warning: Skipping malformed line: {','.join(parts)}")
            continue

        out.write(','.join(parts) + '\n')

print(f"Saved rescaled annotations to: {output_path}")
