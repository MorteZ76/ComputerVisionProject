# scripts/show_image_lists_with_labels.py
import os
import cv2
import yaml
import random

# =========================
# HYPERPARAMETERS (EDIT)
# =========================
DATA_YAML_PATH = r"C:\Users\morte\ComputerVisionProject\data.yaml"
SPLIT = "train"            # "train", "val", or "test"
SHUFFLE = False            # True to randomize order
LIMIT = None               # e.g. 200 to cap number shown, or None
WINDOW_NAME = "YOLO labels viewer"
WINDOW_W, WINDOW_H = 1280, 720
THICKNESS = 2
FONT_SCALE = 0.5
# =========================

def load_lists_from_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    paths = {
        "train": data["train"],
        "val": data["val"],
        "test": data["test"]
    }
    names = data.get("names", [])
    return paths, names

def read_image_list(list_txt_path):
    with open(list_txt_path, "r", encoding="utf-8") as f:
        items = [ln.strip() for ln in f if ln.strip()]
    return items

def guess_label_path(img_path):
    # Convert: .../images/<split>/.../file.jpg  ->  .../labels/<split>/.../file.txt
    parts = img_path.replace("\\", "/").split("/")
    try:
        i = parts.index("images")
        split = parts[i+1]  # train/val/test
        rest = parts[i+2:]  # subdirs + filename
        label_path = "/".join(parts[:i] + ["labels", split] + rest)
        label_path = os.path.splitext(label_path)[0] + ".txt"
        return label_path
    except ValueError:
        # If "images" not found in path, try naive swap once
        label_path = img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        label_path = os.path.splitext(label_path)[0] + ".txt"
        return label_path

def yolo_to_xyxy(w, h, cx, cy, bw, bh):
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    return max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)

def color_for_class(c):
    # distinct but stable colors
    import numpy as np
    rng = np.random.default_rng(c*9973)
    r,g,b = rng.integers(60, 230, size=3).tolist()
    return int(b), int(g), int(r)

def draw_labels(img, label_path, class_names):
    if not os.path.isfile(label_path):
        return img, 0
    h, w = img.shape[:2]
    count = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 5:
                continue
            try:
                cls = int(s[0])
                cx, cy, bw, bh = map(float, s[1:5])
            except Exception:
                continue
            x1,y1,x2,y2 = yolo_to_xyxy(w, h, cx, cy, bw, bh)
            color = color_for_class(cls)
            cv2.rectangle(img, (x1,y1), (x2,y2), color, THICKNESS)
            label = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
            cv2.putText(img, f"{label}", (x1, max(12, y1-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 1, cv2.LINE_AA)
            count += 1
    return img, count

def main():
    paths_map, class_names = load_lists_from_yaml(DATA_YAML_PATH)
    list_txt = paths_map[SPLIT]
    images = read_image_list(list_txt)

    if SHUFFLE:
        random.shuffle(images)
    if LIMIT is not None:
        images = images[:LIMIT]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)

    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] cannot read {img_path}")
            idx += 1
            continue

        lab_path = guess_label_path(img_path)
        vis = img.copy()
        vis, nboxes = draw_labels(vis, lab_path, class_names)

        # footer
        text = f"{SPLIT} [{idx+1}/{len(images)}] boxes={nboxes}"
        cv2.putText(vis, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2, cv2.LINE_AA)

        disp = cv2.resize(vis, (WINDOW_W, WINDOW_H))
        cv2.imshow(WINDOW_NAME, disp)

        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):   # quit
            break
        elif k == ord('n'):       # next
            idx += 1
        elif k == ord('b'):       # back
            idx -= 1
        else:
            idx += 1  # default advance

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
