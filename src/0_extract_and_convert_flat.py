# scripts/extract_and_convert_flat.py
import os
import cv2
from collections import defaultdict
from utils import *  # expects: ensure_dir, read_annotation_file, CLASS_MAP, bbox_to_yolo

# =========================
# HYPERPARAMETERS (EDIT)
# =========================
# Absolute paths (no scene folders)
VIDEO0_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
VIDEO3_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
ANN0_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video0.txt"
ANN3_PATH   = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations\video3.txt"

OUT_ROOT    = r"C:\Users\morte\ComputerVisionProject\dataset"
FRAME_STEP  = 1                 # keep every k-th frame
IMG_EXT     = ".jpg"            # output format
# =========================

IMAGES_ALL = os.path.join(OUT_ROOT, "images", "all")
LABELS_ALL = os.path.join(OUT_ROOT, "labels", "all")

SOURCES = [
    {"name": "video0", "video": VIDEO0_PATH, "ann": ANN0_PATH},
    {"name": "video3", "video": VIDEO3_PATH, "ann": ANN3_PATH},
]

def extract_all_frames():
    ensure_dir(IMAGES_ALL)
    mapping = {}            # (name, local_frame_idx) -> abs image path
    per_video = {}          # name -> info
    global_idx = 0

    for s in SOURCES:
        name = s["name"]
        vpath = s["video"]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {vpath}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        first_name = None
        last_name = None
        saved = 0
        local_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if local_idx % FRAME_STEP == 0:
                fname = f"frame{global_idx:06d}{IMG_EXT}"
                fpath = os.path.join(IMAGES_ALL, fname)
                cv2.imwrite(fpath, frame)
                mapping[(name, local_idx)] = fpath
                last_name = fname
                if first_name is None:
                    first_name = fname
                saved += 1
                global_idx += 1
            local_idx += 1

        cap.release()
        per_video[name] = dict(first_name=first_name, last_name=last_name,
                               frames_saved=saved, width=W, height=H, total_frames=total)
        print(f"[{name}] first: {first_name}  last: {last_name}  saved: {saved}  (W={W}, H={H})")

    return mapping, per_video

def convert_annotations_to_yolo(mapping, per_video):
    ensure_dir(LABELS_ALL)

    for s in SOURCES:
        name = s["name"]
        ann_file = s["ann"]
        if not os.path.isfile(ann_file):
            print(f"[warn] missing ann: {ann_file}")
            continue

        rows = read_annotation_file(ann_file)
        if not rows:
            print(f"[{name}] no rows in ann")
            continue

        # per-video scale from annotation space to image size
        max_ax = max(max(r["xmin"], r["xmax"]) for r in rows)
        max_ay = max(max(r["ymin"], r["ymax"]) for r in rows)
        W = per_video[name]["width"]
        H = per_video[name]["height"]
        sx = W / max_ax if max_ax > 0 else 1.0
        sy = H / max_ay if max_ay > 0 else 1.0

        by_frame = defaultdict(list)
        for r in rows:
            if r["lost"] == 1:
                continue
            by_frame[r["frame"]].append(r)

        written = 0
        missing = 0
        for fidx, anns in by_frame.items():
            img_path = mapping.get((name, fidx))
            if img_path is None:
                missing += 1
                continue

            base = os.path.splitext(os.path.basename(img_path))[0]
            lab_path = os.path.join(LABELS_ALL, base + ".txt")

            lines = []
            for a in anns:
                lbl = a["label"]
                if lbl not in CLASS_MAP:
                    continue
                cls_id = CLASS_MAP[lbl]

                x1 = max(0, min(W - 1, a["xmin"] * sx))
                y1 = max(0, min(H - 1, a["ymin"] * sy))
                x2 = max(0, min(W - 1, a["xmax"] * sx))
                y2 = max(0, min(H - 1, a["ymax"] * sy))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx, cy, bw, bh = bbox_to_yolo(int(x1), int(y1), int(x2), int(y2), W, H)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if lines:
                with open(lab_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                written += 1

        print(f"[{name}] labels written: {written}  missing frames: {missing}")

def main():
    ensure_dir(IMAGES_ALL)
    ensure_dir(LABELS_ALL)
    mapping, per_video = extract_all_frames()
    convert_annotations_to_yolo(mapping, per_video)

    total_images = len(mapping)
    total_labels = len([f for f in os.listdir(LABELS_ALL) if f.lower().endswith(".txt")]) if os.path.isdir(LABELS_ALL) else 0
    print(f"\n[done] total images: {total_images}  total labels: {total_labels}")
    for name, info in per_video.items():
        print(f"  {name}: first={info['first_name']} last={info['last_name']} saved={info['frames_saved']}")

if __name__ == "__main__":
    main()
