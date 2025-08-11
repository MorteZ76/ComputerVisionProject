from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        # Define the class mapping from YOLO classes to our custom classes
        self.class_mapping = {
            0: 'Pedestrian',  # person
            1: 'Biker',      # bicycle
            2: 'Car',        # car
            5: 'Bus',        # bus
            36: 'Skater',    # skateboard
            62: 'Cart'       # cart does not exist
        }
        print("Initialized detector with classes:", self.class_mapping.values())

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Only include detections for our classes of interest
            if cls in self.class_mapping:
                mapped_cls = self.class_mapping[cls]
                detections.append([
                    int(x1), int(y1), int(x2), int(y2),
                    float(conf),
                    mapped_cls
                ])
        return detections
