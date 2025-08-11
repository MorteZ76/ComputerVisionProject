from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), cls])
        return detections
