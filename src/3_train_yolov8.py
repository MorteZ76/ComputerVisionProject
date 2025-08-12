# scripts/3_train_yolov8.py
from ultralytics import YOLO
import os

DATA_YAML = r"C:\Users\morte\ComputerVisionProject\data.yaml"
MODEL_SAVE_DIR = r"C:\Users\morte\ComputerVisionProject\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train():
    # Choose model size
    model = YOLO("yolov8s.pt")  # 'yolov8n.pt' for faster, 'yolov8m.pt' for better accuracy

    # Train with TensorBoard logging
    model.train(
        data=DATA_YAML,
        epochs=80,
        imgsz=960,
        batch=8,
        project=MODEL_SAVE_DIR,
        name="sdd_yolov8s",
        verbose=True,
        plots=True,       # save loss curves automatically
        val=True,         # run validation every epoch
        exist_ok=True,    # overwrite if exists
        # device=0,         # GPU
        save_period=5,    # save weights every 5 epochs
        # workers=4,
        optimizer="SGD",  # or 'AdamW'
    )

    # Optional: You can manually watch results in TensorBoard
    print("\nTo monitor training in TensorBoard, run:")
    print(f"tensorboard --logdir {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    train()
