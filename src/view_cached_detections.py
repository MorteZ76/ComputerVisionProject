# view_cached_detections.py  (no CLI args)
import cv2, numpy as np, pandas as pd
from collections import defaultdict

# ---- set your paths and initial thresholds here ----
# VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video0.mp4"
# DET_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\detected\video0_detections.parquet"
VIDEO_PATH = r"C:\Users\morte\Desktop\Computer Vision\FULL Dataset\video\video3.mp4"
DET_PATH   = r"C:\Users\morte\ComputerVisionProject\ComputerVisionProject\detected\video3_detections.parquet"

INIT_CONF = 0.50
INIT_IOU  = 0.60
INIT_AGNOSTIC = False  # False = class-wise (matches your original code)

# ---------- IoU + NMS ----------
def iou_nms_xyxy(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[order[1:],2]-boxes[order[1:],0]) * (boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / (area_i + area_j - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep

def nms_per_frame(frame_rows, conf_thres, iou_thres, agnostic=False):
    if frame_rows.size == 0:
        return []
    fr = frame_rows[frame_rows[:, 5] >= conf_thres]
    if fr.size == 0:
        return []
    out = []
    if agnostic:
        boxes = fr[:, 1:5].astype(np.float32)
        scores = fr[:, 5].astype(np.float32)
        keep = iou_nms_xyxy(boxes, scores, iou_thres)
        for i in keep:
            out.append((boxes[i], float(scores[i]), int(fr[i, 6])))
    else:
        for c in np.unique(fr[:, 6]).astype(int):
            sc = fr[fr[:, 6] == c]
            boxes = sc[:, 1:5].astype(np.float32)
            scores = sc[:, 5].astype(np.float32)
            keep = iou_nms_xyxy(boxes, scores, iou_thres)
            for i in keep:
                out.append((boxes[i], float(scores[i]), c))
    return out

def draw_text(img, txt, org, color=(255,255,255)):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def load_dets(path):
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_csv(path)
    cols = ["frame","x1","y1","x2","y2","score","cls"]
    if any(c not in df.columns for c in cols):
        raise ValueError(f"Det file missing required columns: {cols}")
    per_frame = defaultdict(list)
    for _, r in df.iterrows():
        per_frame[int(r["frame"])].append([
            int(r["frame"]), float(r["x1"]), float(r["y1"]),
            float(r["x2"]), float(r["y2"]), float(r["score"]), int(r["cls"])
        ])
    for k in list(per_frame.keys()):
        per_frame[k] = np.array(per_frame[k], dtype=np.float32)
    return per_frame

def main():
    per_frame = load_dets(DET_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    f = 0
    conf_thres = INIT_CONF
    iou_thres  = INIT_IOU
    agnostic   = INIT_AGNOSTIC

    print("Controls: +/- conf | [/] IoU | a=agnostic | c=class-wise | q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rows = per_frame.get(f, np.empty((0,7), dtype=np.float32))
        dets = nms_per_frame(rows, conf_thres, iou_thres, agnostic=agnostic)

        for box, score, cls in dets:
            x1,y1,x2,y2 = box.astype(int)
            color = (0,255,0) if not agnostic else (255,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            draw_text(frame, f"cls:{cls} s:{score:.2f}", (x1, max(0,y1-6)), color)

        mode = "agnostic" if agnostic else "class-wise"
        hud = f"frame {f+1}/{total} | conf={conf_thres:.2f} | iou={iou_thres:.2f} | {mode}"
        draw_text(frame, hud, (8, 20), (255,255,255))

        cv2.imshow("Cached detections", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in (ord('+'), ord('=')): conf_thres = min(0.99, conf_thres + 0.02)
        elif key in (ord('-'), ord('_')): conf_thres = max(0.00, conf_thres - 0.02)
        elif key == ord('['): iou_thres = max(0.00, iou_thres - 0.02)
        elif key == ord(']'): iou_thres = min(0.99, iou_thres + 0.02)
        elif key == ord('a'): agnostic = True
        elif key == ord('c'): agnostic = False

        f += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
