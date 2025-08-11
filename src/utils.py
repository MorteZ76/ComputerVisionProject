import cv2

def get_entry_exit(x, y, w, h, margin=50):
    if x < margin: return "left"
    if x > w - margin: return "right"
    if y < margin: return "top"
    if y > h - margin: return "bottom"
    return None

def draw_tracks(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2, track_id, cls = t
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, str(track_id), (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame
