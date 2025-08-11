import pandas as pd
from collections import Counter
from .utils import get_entry_exit

def find_frequent_path(csv_path, frame_width, frame_height):
    df = pd.read_csv(csv_path)
    path_counts = Counter()

    for tid in df.track_id.unique():
        track = df[df.track_id == tid]
        entry = get_entry_exit(track.iloc[0]["x"], track.iloc[0]["y"], frame_width, frame_height)
        exit_ = get_entry_exit(track.iloc[-1]["x"], track.iloc[-1]["y"], frame_width, frame_height)
        if entry and exit_:
            path_counts[(entry, exit_)] += 1

    return path_counts.most_common(1)[0]
