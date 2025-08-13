# # scripts/3_train_yolov8.py
# from ultralytics import YOLO
# from pathlib import Path
# import shutil, torch, os

# DATA_YAML = Path(r"C:\Users\morte\ComputerVisionProject\data.yaml")
# MODEL_SAVE_DIR = Path(r"C:\Users\morte\ComputerVisionProject\models")
# MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# CKPT_DIR = Path(r"C:\Users\morte\ComputerVisionProject\checkpoints")
# CKPT_DIR.mkdir(parents=True, exist_ok=True)

# def backup_ckpt(trainer):
#     src = Path(trainer.save_dir) / "weights" / "last.pt"
#     if src.exists():
#         shutil.copy2(src, CKPT_DIR / "last.pt")

# def train():
#     model = YOLO("yolov8s.pt")
#     model.add_callback("on_fit_epoch_end", backup_ckpt)

#     device = 0 if torch.cuda.is_available() else "cpu"
#     torch.set_num_threads(os.cpu_count() or 1)

#     model.train(
#         data=str(DATA_YAML),
#         epochs=40,
#         imgsz=960,
#         batch=8,
#         project=str(MODEL_SAVE_DIR),
#         name="sdd_yolov8s",
#         verbose=True,
#         plots=True,
#         val=True,
#         exist_ok=True,
#         save_period=5,
#         optimizer="SGD",
#         device=device,
#         workers=0,   # safest on Windows/CPU
#     )

#     print("\nTo monitor training:")
#     print(f"tensorboard --logdir {MODEL_SAVE_DIR}")

# if __name__ == "__main__":
#     train()


# scripts/3_train_yolov8.py
from ultralytics import YOLO
from pathlib import Path
import shutil, torch, os

# =======================
# HYPERPARAMETERS (EDIT)
# =======================
DATA_YAML       = Path(r"C:\Users\morte\ComputerVisionProject\data.yaml")
BASE_MODEL      = "yolov8s.pt"       # used for fresh training
RUN_NAME        = "sdd_yolov8s"      # base run name
EPOCHS_FRESH    = 40                  # epochs for fresh training
EPOCHS_RESUME   = 32                  # extra epochs when resuming
IMGSZ           = 960
BATCH           = 8
OPTIMIZER       = "SGD"               # or "AdamW"
WORKERS         = 0                   # safest on Windows/CPU
# Paths
MODEL_SAVE_DIR  = Path(r"C:\Users\morte\ComputerVisionProject\models")
CKPT_DIR        = Path(r"C:\Users\morte\ComputerVisionProject\checkpoints")
RESUME_LAST_PT  = CKPT_DIR / "last.pt"  # auto-used if present
# =======================

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def backup_ckpt(trainer):
    src = Path(trainer.save_dir) / "weights" / "last.pt"
    if src.exists():
        shutil.copy2(src, CKPT_DIR / "last.pt")

def train():
    device = 0 if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(os.cpu_count() or 1)

    # choose mode: resume if checkpoint exists
    if RESUME_LAST_PT.exists():
        # load checkpoint and continue training for EPOCHS_RESUME
        model = YOLO(str(RESUME_LAST_PT))
        run_name = f"{RUN_NAME}_resume"
        epochs = EPOCHS_RESUME
        print(f"[resume] Using checkpoint: {RESUME_LAST_PT}")
    else:
        # fresh training
        model = YOLO(BASE_MODEL)
        run_name = RUN_NAME
        epochs = EPOCHS_FRESH
        print(f"[fresh] Using base model: {BASE_MODEL}")

    model.add_callback("on_fit_epoch_end", backup_ckpt)

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(MODEL_SAVE_DIR),
        name=run_name,
        verbose=True,
        plots=True,
        val=True,
        exist_ok=True,
        save_period=5,
        optimizer=OPTIMIZER,
        device=device,
        workers=WORKERS,
        # note: we load weights from RESUME_LAST_PT above; optimizer state is not resumed.
        # if you want exact-state resume of a prior run directory, use:
        # resume=True, project=<same>, name=<same>
    )

    print("\nTo monitor training:")
    print(f"tensorboard --logdir {MODEL_SAVE_DIR}")

if __name__ == "__main__":
    train()
