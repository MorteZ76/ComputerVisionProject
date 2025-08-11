# scripts/0_extract_frames_and_split.py
import os, shutil, cv2, math
from pathlib import Path
from utils import *

# ---- CONFIG ----
RAW_VIDEOS_ROOT = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video"
ANNOTATIONS_ROOT = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\annotations"
OUT_ROOT = r"C:\Users\morte\ComputerVisionProject\dataset"
TEST_FORCE = [ ( "little", "video0" ), ( "little", "video3" ) ]  # (scene, foldername)
FRAME_STEP = 1  # extract every frame (set >1 to subsample)
# -----------------

def is_forced_test(scene, vfolder):
    return (scene, vfolder) in TEST_FORCE

def main():
    images_train = os.path.join(OUT_ROOT, "images", "train")
    images_val   = os.path.join(OUT_ROOT, "images", "val")
    images_test  = os.path.join(OUT_ROOT, "images", "test")
    for p in [images_train, images_val, images_test]:
        ensure_dir(p)

    # iterate scenes
    for scene in os.listdir(RAW_VIDEOS_ROOT):
        scene_path = os.path.join(RAW_VIDEOS_ROOT, scene)
        if not os.path.isdir(scene_path): continue
        for vfolder in os.listdir(scene_path):
            vpath = os.path.join(scene_path, vfolder)
            video_file = os.path.join(vpath, "video.mp4")
            if not os.path.exists(video_file):
                print("missing video:", video_file)
                continue

            # decide split
            if is_forced_test(scene, vfolder):
                split = "test"
                out_dir = images_test
            else:
                # put others to train/val by 90/10 frames
                # we'll put entire video to train or val based on a modulo rule to avoid mixing partials:
                # simpler: most go to train, a few to val randomly deterministically
                split = "train" if (hash(scene+vfolder) % 10) < 9 else "val"
                out_dir = images_train if split=="train" else images_val

            # create a subfolder to avoid name collisions
            sub = os.path.join(out_dir, f"{scene}_{vfolder}")
            ensure_dir(sub)

            cap = cv2.VideoCapture(video_file)
            frame_id = 0
            saved = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % FRAME_STEP == 0:
                    fname = f"{scene}_{vfolder}_frame{frame_id:06d}.jpg"
                    cv2.imwrite(os.path.join(sub, fname), frame)
                    saved += 1
                frame_id += 1
            cap.release()
            print(f"Saved {saved} frames for {scene}/{vfolder} to {sub}")

if __name__ == "__main__":
    main()
