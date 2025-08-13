import cv2
import os
import glob
import random

# ================= CONFIG =================
dataset_path = r"C:\Users\morte\ComputerVisionProject\dataset"
split = "all"  # change to 'val' or 'test'

images_path = os.path.join(dataset_path, "images", split)
labels_path = os.path.join(dataset_path, "labels", split)

# Class names in correct order (matching your YOLO dataset)
class_names = ["Pedestrian", "Biker", "Car", "Bus", "Skater", "Cart"]

# Assign fixed random colors for each class
colors = {cls: [random.randint(0, 255) for _ in range(3)] for cls in class_names}
# ===========================================

# Recursively collect all image files
image_files = sorted(glob.glob(os.path.join(images_path, "**", "*.jpg"), recursive=True))

frame_index = 0
paused = False

def draw_annotations(img, label_file):
    h, w, _ = img.shape
    print(f"H = {h}, W= {w}")
    if not os.path.exists(label_file):
        return img
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:])
            print (f"Class ID: {cls_id}, Center: ({x_center}, {y_center}), Size: ({bw}, {bh})")

            # Convert YOLO format to pixel coords
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            print(f"x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
            color = colors[class_names[cls_id]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[cls_id], (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

while True:
    # print(f"Image files: {image_files}")
    image_file = image_files[frame_index]

    # Compute label path (preserve subfolder structure)
    rel_path = os.path.relpath(image_file, images_path)
    label_file = os.path.join(labels_path, rel_path).replace(".jpg", ".txt")

    img = cv2.imread(image_file)
    img = draw_annotations(img, label_file)

    cv2.imshow("Annotation Viewer", img)
    key = cv2.waitKey(0 if paused else 30) & 0xFF
    frame_index = (frame_index + 1) % len(image_files)

    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused
    elif paused and key == ord("o"):
        frame_index = (frame_index + 1) % len(image_files)
    elif paused and key == ord("i"):
        frame_index = (frame_index - 1) % len(image_files)
    elif not paused and key != 255:
        frame_index = (frame_index + 1) % len(image_files)

cv2.destroyAllWindows()
