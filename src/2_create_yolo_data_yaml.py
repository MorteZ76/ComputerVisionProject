# scripts/2_create_yolo_data_yaml.py
import os, re, yaml
from tqdm import tqdm  # pip install tqdm

# =======================
# HYPERPARAMETERS (EDIT)
# =======================
OUT_ROOT = r"C:\Users\morte\ComputerVisionProject\dataset"
DATA_YAML_PATH = r"C:\Users\morte\ComputerVisionProject\data.yaml"

# YOLO class ids (must match your conversion)
PED_ID = 0
BIKER_ID = 1

# Selection constraints
MIN_FRAME_GAP = 3                 # keep a frame only if >= this gap from the last kept in the SAME sequence
MAX_FRAMES_TOTAL = 1000           # hard cap across train+val+test
SPLIT_RATIOS = (0.7, 0.2, 0.1)    # train, val, test; must sum to 1.0

# Filenames considered images
IMG_EXTS = (".jpg", ".jpeg", ".png")

# =======================

def extract_frame_idx(path):
    """
    Extract an integer frame index from filename (uses the last digit run).
    Returns -1 if not found.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    m = list(re.finditer(r'(\d+)', name))
    return int(m[-1].group(1)) if m else -1

def has_biker_and_ped(label_path):
    """
    True if YOLO label file contains BOTH Pedestrian and Biker classes.
    Format per line: <cls> <cx> <cy> <w> <h>
    """
    if not os.path.isfile(label_path):
        return False
    ped = False
    bik = False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            try:
                cls_id = int(s[0])
            except ValueError:
                continue
            if cls_id == PED_ID:
                ped = True
            elif cls_id == BIKER_ID:
                bik = True
            if ped and bik:
                return True
    return False

def rel_from_images_root(abs_img_path, images_root):
    """Relative path of image from its split-specific images root."""
    return os.path.relpath(abs_img_path, images_root)

def gather_candidates():
    """
    Scan all three splits under OUT_ROOT/images/{train,val,test},
    keep frames with both biker+ped labels, and enforce MIN_FRAME_GAP per sequence.
    Returns a single ordered list of absolute image paths (capped by MAX_FRAMES_TOTAL).
    """
    splits = ["train", "val", "test"]
    candidates = []

    # Pass 1: collect and filter by labels
    for split in splits:
        images_dir = os.path.join(OUT_ROOT, "images", split)
        labels_dir = os.path.join(OUT_ROOT, "labels", split)
        if not os.path.isdir(images_dir):
            continue

        # Walk and collect images
        img_paths = []
        for root, _, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    img_paths.append(os.path.join(root, f))
        img_paths.sort()

        # Per-sequence gap control
        last_idx_by_seq = {}

        for img_path in tqdm(img_paths, desc=f"Scanning {split}", unit="img"):
            rel = rel_from_images_root(img_path, images_dir)
            lab_path = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
            if not has_biker_and_ped(lab_path):
                continue

            # Sequence key = parent folder(s) under this split to avoid cross-video interference
            seq_key = os.path.dirname(rel).replace("\\", "/")
            fidx = extract_frame_idx(img_path)

            if seq_key not in last_idx_by_seq:
                # keep first qualifying frame in this sequence
                candidates.append(img_path)
                last_idx_by_seq[seq_key] = fidx if fidx >= 0 else 0
            else:
                last_idx = last_idx_by_seq[seq_key]
                if fidx >= 0:
                    if fidx >= last_idx + MIN_FRAME_GAP:
                        candidates.append(img_path)
                        last_idx_by_seq[seq_key] = fidx
                else:
                    # if no index found, be conservative: require a gap of MIN_FRAME_GAP generic steps
                    # emulate by incrementing a virtual counter
                    last_idx_by_seq[seq_key] = last_idx + MIN_FRAME_GAP

    # Cap to MAX_FRAMES_TOTAL while preserving order
    if MAX_FRAMES_TOTAL is not None and MAX_FRAMES_TOTAL > 0:
        candidates = candidates[:MAX_FRAMES_TOTAL]

    # Normalize paths to forward slashes
    candidates = [p.replace("\\", "/") for p in candidates]
    return candidates

def split_list(candidates):
    """
    Split candidates into train/val/test according to SPLIT_RATIOS.
    """
    assert abs(sum(SPLIT_RATIOS) - 1.0) < 1e-6, "SPLIT_RATIOS must sum to 1.0"
    n = len(candidates)
    n_train = int(round(n * SPLIT_RATIOS[0]))
    n_val   = int(round(n * SPLIT_RATIOS[1]))
    # Ensure total matches n (assign remainder to test)
    n_test  = n - n_train - n_val
    train = candidates[:n_train]
    val   = candidates[n_train:n_train+n_val]
    test  = candidates[n_train+n_val:]
    return train, val, test

def write_list(paths, out_txt):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

def main():
    candidates = gather_candidates()
    train_list, val_list, test_list = split_list(candidates)

    base = os.path.dirname(DATA_YAML_PATH)
    os.makedirs(base, exist_ok=True)

    train_txt = os.path.join(base, "train_images.txt").replace("\\", "/")
    val_txt   = os.path.join(base, "val_images.txt").replace("\\", "/")
    test_txt  = os.path.join(base, "test_images.txt").replace("\\", "/")

    write_list(train_list, train_txt)
    write_list(val_list, val_txt)
    write_list(test_list, test_txt)

    data = {
        "path": "",  # not used since we pass full lists
        "train": train_txt,
        "val":   val_txt,
        "test":  test_txt,
        "names": ["Pedestrian", "Biker", "Car", "Bus", "Skater", "Cart"],
        "nc": 6
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("\nSummary")
    print(f"  MIN_FRAME_GAP       : {MIN_FRAME_GAP}")
    print(f"  MAX_FRAMES_TOTAL    : {MAX_FRAMES_TOTAL}")
    print(f"  SPLIT_RATIOS (t/v/e): {SPLIT_RATIOS}")
    print(f"  Total kept images   : {len(candidates)}")
    print(f"    train: {len(train_list)}  val: {len(val_list)}  test: {len(test_list)}")
    print(f"Wrote data yaml       : {DATA_YAML_PATH}")

if __name__ == "__main__":
    main()
