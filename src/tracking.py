from bytetrack import BYTETracker

class Tracker:
    def __init__(self):
        self.tracker = BYTETracker()

    def update(self, detections, frame):
        return self.tracker.update(detections, frame)
