# scripts/2_create_yolo_data_yaml.py
import os, yaml

# Match OUT_ROOT from the first two scripts
OUT_ROOT = r"C:\Users\morte\ComputerVisionProject\dataset"
DATA_YAML_PATH = r"C:\Users\morte\ComputerVisionProject\data.yaml"

def collect_images_list(split):
    images_dir = os.path.join(OUT_ROOT, "images", split)
    paths = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                paths.append(os.path.join(root, f).replace("\\", "/"))
    return sorted(paths)

def write_list(paths, out_txt):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(paths))

def main():
    train_list = collect_images_list("train")
    val_list   = collect_images_list("val")
    test_list  = collect_images_list("test")

    base = os.path.dirname(DATA_YAML_PATH)
    os.makedirs(base, exist_ok=True)

    train_txt = os.path.join(base, "train_images.txt")
    val_txt   = os.path.join(base, "val_images.txt")
    test_txt  = os.path.join(base, "test_images.txt")

    write_list(train_list, train_txt)
    write_list(val_list, val_txt)
    write_list(test_list, test_txt)

    data = {
        "path": "",  # not used since we give full image lists
        "train": train_txt.replace("\\", "/"),
        "val":   val_txt.replace("\\", "/"),
        "test":  test_txt.replace("\\", "/"),
        "names": ["Pedestrian", "Biker", "Car", "Bus", "Skater", "Cart"],
        "nc": 6
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Wrote {DATA_YAML_PATH}")

if __name__ == "__main__":
    main()
