# scripts/1_convert_annotations_to_yolo.py
import os
from pathlib import Path
import cv2
import pandas as pd
from utils import *

RAW_VIDEO_ROOT = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video"
RAW_ANN_ROOT = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations"
OUT_ROOT = r"C:\Users\morte\ComputerVisionProject\dataset"
IMGS_ROOT = os.path.join(OUT_ROOT, "images")
LABELS_ROOT = os.path.join(OUT_ROOT, "labels")

def find_image_path(scene, vfolder, frame_idx):
    # looks for file we saved in extract step: scene_vfolder_frame{frame:06d}.jpg
    pattern = f"{scene}_{vfolder}_frame{frame_idx:06d}.jpg"
    # search train/val/test dirs:
    for split in ["train","val","test"]:
        for root, dirs, files in os.walk(os.path.join(IMGS_ROOT, split)):
            if pattern in files:
                return os.path.join(root, pattern)
    return None

def main():
    ensure_dir(LABELS_ROOT)
    for split in ["train","val","test"]:
        ensure_dir(os.path.join(LABELS_ROOT, split))

    # iterate annotation files
    for scene in os.listdir(RAW_ANN_ROOT):
        scene_ann_dir = os.path.join(RAW_ANN_ROOT, scene)
        if not os.path.isdir(scene_ann_dir): 
            continue

        for vfolder in os.listdir(scene_ann_dir):
            ann_file = os.path.join(scene_ann_dir, vfolder, "annotations.txt")
            if not os.path.exists(ann_file):
                print("no ann file", ann_file)
                continue

            rows = read_annotation_file(ann_file)

            # Compute scaling factors based on annotations & actual frame size
            max_annotation_x = max([max(r["xmin"], r["xmax"]) for r in rows])
            max_annotation_y = max([max(r["ymin"], r["ymax"]) for r in rows])

            # We'll get frame size later (per image), but if max_ann > img_size, scale_x/y will be < 1
            # This will be recalculated inside the frame loop for safety

            # group rows by frame
            by_frame = {}
            for r in rows:
                if r["lost"] == 1:
                    continue
                frame_idx = r["frame"]
                by_frame.setdefault(frame_idx, []).append(r)

            # for each annotated frame, find corresponding image
            for frame_idx, anns in by_frame.items():
                img_path = find_image_path(scene, vfolder, frame_idx)
                if img_path is None:
                    # frame might not have been extracted (due to step), skip
                    continue

                # determine split from image path
                if "\\images\\train\\" in img_path or "/images/train/" in img_path:
                    split = "train"
                elif "\\images\\val\\" in img_path or "/images/val/" in img_path:
                    split = "val"
                else:
                    split = "test"

                # label filename same base but .txt in labels/<split>/...
                base = os.path.basename(img_path)
                label_dir = os.path.join(LABELS_ROOT, split, os.path.basename(os.path.dirname(img_path)))
                ensure_dir(label_dir)
                label_file = os.path.join(label_dir, base.replace(".jpg", ".txt"))

                # get image size
                img = cv2.imread(img_path)
                h, w = img.shape[:2]

                # Calculate scaling factors for this video
                scale_x = w / max_annotation_x if max_annotation_x > w else 1
                scale_y = h / max_annotation_y if max_annotation_y > h else 1

                lines = []
                for a in anns:
                    lbl = a["label"]
                    if lbl not in CLASS_MAP:
                        continue
                    cls_id = CLASS_MAP[lbl]

                    # Apply scaling to annotation coords
                    xmin = int(a["xmin"] * scale_x)
                    ymin = int(a["ymin"] * scale_y)
                    xmax = int(a["xmax"] * scale_x)
                    ymax = int(a["ymax"] * scale_y)

                    # Convert to YOLO format
                    x_c, y_c, nw, nh = bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)

                    # Clamp
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    nw  = max(0.0, min(1.0, nw))
                    nh  = max(0.0, min(1.0, nh))

                    lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")

                if lines:
                    with open(label_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
