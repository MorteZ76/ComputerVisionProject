# scripts/handpick_viewer_from_master_ann.py
# Browse frames that contain ANY target classes, determined from the SCENE'S master annotations.txt.
# Images/labels are loaded from your per-frame dataset folders.
# Keys: o=next, i=prev, l=+10, k=-10, s=save, a=toggle filter, q/ESC=quit.

import os, re, cv2, shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
# Master SDD annotations for the whole video:
MASTER_ANN_TXT = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotationsALL\deathCircle\video4\annotations.txt"

# Your extracted dataset roots and subfolder to browse:
IMAGES_ROOT = r"C:\Users\morte\ComputerVisionProject\Made dataset\images\val"
LABELS_ROOT = r"C:\Users\morte\ComputerVisionProject\Made dataset\labels\val"
SUBFOLDER   = r"deathCircle_video4"   # contains files like ...\bookstore_video2_frame000000.jpg

# Show only frames that contain ≥1 of these classes (from MASTER_ANN_TXT)
TARGET_CLASS_NAMES = ["Cart", "Bus", "Car", "Skater"]
FILTER_REQUIRED   = True            # toggle with 'a'
SHOW_ALL_IF_NOT_FOUND = True        # if no target frames, optionally show all

# Save accepted frames here
SAVE_IMG_DIR = r"C:\Users\morte\ComputerVisionProject\Made dataset\HANDPICKED\images"
SAVE_LBL_DIR = r"C:\Users\morte\ComputerVisionProject\Made dataset\HANDPICKED\labels"

# Misc
FPS = 32.0
WINDOW_NAME  = "Handpick Viewer (master-ann filter)"
WINDOW_SCALE = 1.0
THICK = 2
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FS    = 0.6

# Class list for YOLO label decoding (must match your training order)
CLASS_NAMES = ["Pedestrian", "Biker", "Car", "Bus", "Skater", "Cart"]
COLOR_MAP = {
    "Pedestrian": (50,200,50), "Biker": (60,140,255), "Car": (60,60,220),
    "Bus": (0,120,200), "Skater": (180,160,50), "Cart": (200,50,200), "Other": (180,180,180),
}
IMG_EXTS = (".jpg", ".jpeg", ".png")
# =========================

TARGET_CLASS_SET = set(TARGET_CLASS_NAMES)

def list_images(folder):
    if not os.path.isdir(folder):
        print(f"[error] missing folder: {folder}")
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files

def label_path_for_image(img_path):
    rel = os.path.relpath(img_path, IMAGES_ROOT)
    return os.path.splitext(os.path.join(LABELS_ROOT, rel))[0] + ".txt"

def parse_yolo_label(lbl_path, img_w, img_h):
    objs = []
    if not os.path.isfile(lbl_path): return objs
    with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().split()
            if len(s) < 5: continue
            try:
                cls_id = int(float(s[0]))
                cx = float(s[1]) * img_w
                cy = float(s[2]) * img_h
                w  = float(s[3]) * img_w
                h  = float(s[4]) * img_h
            except ValueError:
                continue
            x1 = max(0, min(img_w-1, cx - w/2.0))
            y1 = max(0, min(img_h-1, cy - h/2.0))
            x2 = max(0, min(img_w-1, cx + w/2.0))
            y2 = max(0, min(img_h-1, cy + h/2.0))
            tid = None
            if len(s) >= 6:
                try: tid = int(float(s[5]))
                except ValueError: tid = s[5]
            name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "Other"
            objs.append({"cls_id": cls_id, "name": name, "xyxy": (int(x1),int(y1),int(x2),int(y2)), "id": tid})
    return objs

def frame_index_from_filename(path):
    m = re.search(r'(\d+)(?=\.[a-zA-Z]+$)', os.path.basename(path))
    return int(m.group(1)) if m else None

def draw_boxes(img, objs):
    for o in objs:
        x1,y1,x2,y2 = o["xyxy"]
        col = COLOR_MAP.get(o["name"], COLOR_MAP["Other"])
        cv2.rectangle(img, (x1,y1), (x2,y2), col, THICK, cv2.LINE_AA)
        tag = f"{o['name']}" + (f"#{o['id']}" if o["id"] is not None else "")
        cv2.putText(img, tag, (x1, max(12,y1-6)), FONT, FS, col, 2, cv2.LINE_AA)

def parse_master_annotations(master_txt):
    """
    Reads SDD annotations.txt and returns a set of frame indices that contain ≥1 target class.
    Uses 'lost' to skip out-of-view boxes. Label column can contain spaces, enclosed in quotes.
    """
    keep_frames = set()
    found_any = False
    with open(master_txt, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip().split()
            if len(s) < 10: continue
            try:
                xmin = float(s[1]); ymin = float(s[2])
                xmax = float(s[3]); ymax = float(s[4])
                frame = int(s[5]); lost = int(s[6])
            except ValueError:
                continue
            label_raw = " ".join(s[9:]).strip().strip('"')
            if lost == 1:
                continue
            # allow simple label normalization
            lbl = label_raw.capitalize()
            if lbl in TARGET_CLASS_SET:
                keep_frames.add(frame)
                found_any = True
    return keep_frames, found_any

def ensure_dirs():
    Path(SAVE_IMG_DIR).mkdir(parents=True, exist_ok=True)
    Path(SAVE_LBL_DIR).mkdir(parents=True, exist_ok=True)

def unique_save_paths(dst_img_dir, dst_lbl_dir, basename):
    img_out = os.path.join(dst_img_dir, basename)
    lbl_out = os.path.join(dst_lbl_dir, os.path.splitext(basename)[0] + ".txt")
    if not os.path.exists(img_out) and not os.path.exists(lbl_out):
        return img_out, lbl_out
    stem, ext = os.path.splitext(basename); k = 1
    while True:
        new_base = f"{stem}_{k}{ext}"
        img_try = os.path.join(dst_img_dir, new_base)
        lbl_try = os.path.join(dst_lbl_dir, os.path.splitext(new_base)[0] + ".txt")
        if not os.path.exists(img_try) and not os.path.exists(lbl_try):
            return img_try, lbl_try
        k += 1

def build_candidates(images, require_target, frame_filter_set):
    kept = []
    for p in images:
        fidx = frame_index_from_filename(p)
        if fidx is None:
            continue
        if require_target:
            if fidx in frame_filter_set:
                kept.append(p)
        else:
            kept.append(p)
    return kept

def main():
    src_img_dir = os.path.join(IMAGES_ROOT, SUBFOLDER)
    images = list_images(src_img_dir)
    if not images:
        print("[error] no images found in:", src_img_dir)
        return

    frame_filter_set, found_any = parse_master_annotations(MASTER_ANN_TXT)
    if not found_any:
        print(f"There is no Target class {sorted(TARGET_CLASS_SET)} in master annotations for: {SUBFOLDER}")
        if not SHOW_ALL_IF_NOT_FOUND:
            return

    require_target = FILTER_REQUIRED and found_any
    candidates = build_candidates(images, require_target, frame_filter_set)

    if not candidates and require_target:
        print("[warn] no frames matched the master-ann filter. showing all frames.")
        require_target = False
        candidates = build_candidates(images, require_target, frame_filter_set)

    print(f"[info] total_images={len(images)}, filtered_frames={len(frame_filter_set)}, "
          f"show={len(candidates)}, filter={'ON' if require_target else 'OFF'}")

    ensure_dirs()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    idx, N = 0, len(candidates)
    while N > 0:
        img_path = candidates[idx]
        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % N
            continue
        H,W = img.shape[:2]
        lbl_path = label_path_for_image(img_path)
        objs = parse_yolo_label(lbl_path, W, H)

        vis = img.copy(); draw_boxes(vis, objs)

        fidx = frame_index_from_filename(img_path)
        if fidx is not None and FPS > 0:
            mm, ss = divmod(int(fidx / FPS), 60)
            time_str = f"{mm:02d}:{ss:02d}"
        else:
            time_str = "--:--"

        info1 = f"[{idx+1}/{N}] {SUBFOLDER} targets={list(TARGET_CLASS_SET)} filter={'ON' if require_target else 'OFF'} time={time_str}"
        info2 = f"frame={fidx if fidx is not None else '?'}  path={os.path.relpath(img_path, IMAGES_ROOT)}"
        info3 = "Keys: o=next, i=prev, l=+10, k=-10, s=save, a=toggle filter, q=quit"
        cv2.putText(vis, info1, (12, 24), FONT, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(vis, info2, (12, 48), FONT, 0.6, (220,220,220), 2, cv2.LINE_AA)
        cv2.putText(vis, info3, (12, 72), FONT, 0.55, (220,220,220), 2, cv2.LINE_AA)

        if WINDOW_SCALE != 1.0:
            vis = cv2.resize(vis, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(WINDOW_NAME, vis)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27): break
        elif k == ord('o'): idx = min(N-1, idx+1)
        elif k == ord('i'): idx = max(0, idx-1)
        elif k == ord('l'): idx = min(N-1, idx+10)
        elif k == ord('k'): idx = max(0, idx-10)
        elif k == ord('a'):
            require_target = not require_target
            candidates = build_candidates(images, require_target, frame_filter_set)
            if not candidates:
                print(f"[warn] no frames with filter={'ON' if require_target else 'OFF'}. reverting.")
                require_target = not require_target
                candidates = build_candidates(images, require_target, frame_filter_set)
            N = len(candidates); idx = min(idx, N-1)
            print(f"[info] now showing {N} frames. filter={'ON' if require_target else 'OFF'}.")
        elif k == ord('s'):
            base = os.path.basename(img_path)
            img_out, lbl_out = unique_save_paths(SAVE_IMG_DIR, SAVE_LBL_DIR, base)
            Path(os.path.dirname(img_out)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(lbl_out)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, img_out)
            if os.path.isfile(lbl_path): shutil.copy2(lbl_path, lbl_out)
            else: Path(lbl_out).write_text("", encoding="utf-8")
            print(f"[save] {SUBFOLDER}\\{base} -> {img_out}, {lbl_out}")

    if N == 0:
        print("[info] nothing to show after filtering.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
