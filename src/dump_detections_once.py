# dump_detections_once.py
import argparse, cv2, numpy as np, pandas as pd
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--weights", required=True, help="YOLOv8 weights .pt")
    ap.add_argument("--out", default="detections_raw.parquet", help="Output Parquet")
    ap.add_argument("--low_conf", type=float, default=0.30, help="Keep boxes with score >= this")
    ap.add_argument("--max_det", type=int, default=30000, help="Max detections per frame to keep")
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame (1 = all)")
    args = ap.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    rows = []
    f = 0
    kept_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        ok, img = cap.read()
        if not ok:
            break
        if args.every > 1 and (f % args.every != 0):
            f += 1
            continue

        r = model.predict(
            source=img,
            conf=args.low_conf,
            iou=1.0,
            agnostic_nms=True,
            max_det=args.max_det,
            verbose=False
        )[0]

        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, conf, cls):
                rows.append([f, float(b[0]), float(b[1]),
                             float(b[2]), float(b[3]),
                             float(s), int(c)])

        # progress line
        if total_frames > 0:
            print(f"\rProcessed frame {f+1}/{total_frames}", end="", flush=True)

        f += 1

    print()  # newline after loop
        

    cap.release()

    if not rows:
        print("No detections saved.")
        return

    df = pd.DataFrame(rows, columns=["frame","x1","y1","x2","y2","score","cls"])
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} detections from {kept_frames} processed frames "
          f"({total_frames} total). File: {args.out}")

if __name__ == "__main__":
    main()
