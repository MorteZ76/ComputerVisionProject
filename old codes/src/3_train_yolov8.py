# scripts/3_train_yolov8.py
from ultralytics import YOLO
from pathlib import Path
import shutil, torch, os

DATA_YAML = Path(r"C:\Users\morte\ComputerVisionProject\data.yaml")
MODEL_SAVE_DIR = Path(r"C:\Users\morte\ComputerVisionProject\models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = Path(r"C:\Users\morte\ComputerVisionProject\checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def backup_ckpt(trainer):
    src = Path(trainer.save_dir) / "weights" / "last.pt"
    if src.exists():
        shutil.copy2(src, CKPT_DIR / "last.pt")

def train():
    model = YOLO("yolov8s.pt")
    model.add_callback("on_fit_epoch_end", backup_ckpt)

    device = 0 if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 1)

    model.train(
        data=str(DATA_YAML),
        epochs=40,
        imgsz=640,
        batch=8,
        project=str(MODEL_SAVE_DIR),
        name="sdd_yolov8s",
        verbose=True,
        plots=True,
        val=True,
        exist_ok=True,
        save_period=5,
        optimizer="SGD",
        device=device,
        workers=0,   # safest on Windows/CPU
    )

    print("\nTo monitor training:")
    print(f"tensorboard --logdir {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    train()
