# scripts/2_create_yolo_data_yaml.py
import os, re, yaml, random
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
MIN_FRAME_GAP = 3                 # keep a frame only if >= this gap from the last kept (global index)
MAX_FRAMES_TOTAL = 1000           # hard cap across train+val+test
SPLIT_RATIOS = (0.7, 0.2, 0.1)    # train, val, test; must sum to 1.0

# Randomization
SHUFFLE_CANDIDATES = True
RANDOM_SEED = 42

# Filenames considered images
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Paths (flat)
IMAGES_ALL = os.path.join(OUT_ROOT, "images", "all")
LABELS_ALL = os.path.join(OUT_ROOT, "labels", "all")
# =======================


def extract_frame_idx(path):
    """
    Extract integer frame index from 'frame000123.jpg' (last digit run in basename).
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


def gather_candidates():
    """
    Scan OUT_ROOT/images/all, keep frames with both biker+ped labels,
    enforce MIN_FRAME_GAP on the GLOBAL frame index (from filename),
    cap to MAX_FRAMES_TOTAL.
    Returns an ordered list of absolute image paths.
    """
    if not os.path.isdir(IMAGES_ALL):
        raise RuntimeError(f"Images folder not found: {IMAGES_ALL}")
    if not os.path.isdir(LABELS_ALL):
        raise RuntimeError(f"Labels folder not found: {LABELS_ALL}")

    # Collect image paths
    img_paths = []
    for f in os.listdir(IMAGES_ALL):
        if f.lower().endswith(IMG_EXTS):
            img_paths.append(os.path.join(IMAGES_ALL, f))
    # Sort by global frame index (critical for MIN_FRAME_GAP)
    img_paths.sort(key=extract_frame_idx)

    kept = []
    last_kept_idx = None

    for img_path in tqdm(img_paths, desc="Scanning all", unit="img"):
        base = os.path.splitext(os.path.basename(img_path))[0]
        lab_path = os.path.join(LABELS_ALL, base + ".txt")
        if not has_biker_and_ped(lab_path):
            continue

        idx = extract_frame_idx(img_path)
        if idx < 0:
            continue  # skip if no index

        if last_kept_idx is None or idx >= last_kept_idx + MIN_FRAME_GAP:
            kept.append(img_path.replace("\\", "/"))
            last_kept_idx = idx

        if MAX_FRAMES_TOTAL and len(kept) >= MAX_FRAMES_TOTAL:
            break

    return kept


def split_list(candidates):
    """
    Shuffle candidates (optional) then split into train/val/test according to SPLIT_RATIOS.
    """
    if SHUFFLE_CANDIDATES:
        random.Random(RANDOM_SEED).shuffle(candidates)

    assert abs(sum(SPLIT_RATIOS) - 1.0) < 1e-6, "SPLIT_RATIOS must sum to 1.0"
    n = len(candidates)
    n_train = int(round(n * SPLIT_RATIOS[0]))
    n_val   = int(round(n * SPLIT_RATIOS[1]))
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
        "path": "",
        "train": train_txt,
        "val":   val_txt,
        "test":  test_txt,
        "names": ["Pedestrian", "Biker", "Car", "Bus", "Skater", "Cart"],
        "nc": 6
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("\nSummary")
    print(f"  MIN_FRAME_GAP          : {MIN_FRAME_GAP}")
    print(f"  MAX_FRAMES_TOTAL       : {MAX_FRAMES_TOTAL}")
    print(f"  SPLIT_RATIOS (train/val/test): {SPLIT_RATIOS}")
    print(f"  SHUFFLE_CANDIDATES     : {SHUFFLE_CANDIDATES}  (seed={RANDOM_SEED})")
    print(f"  Total kept images      : {len(candidates)}")
    print(f"    train: {len(train_list)}  val: {len(val_list)}  test: {len(test_list)}")
    print(f"Wrote data yaml          : {DATA_YAML_PATH}")

if __name__ == "__main__":
    main()
