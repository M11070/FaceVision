import json
import time
import uuid
from pathlib import Path

import numpy as np


class Measurement:
    def __init__(self, warmup, seconds):
        self.warmup, self.seconds = warmup, seconds
        self.opened = time.perf_counter()
        self.start = None
        self.end = None
        self.rows = []

    def add(self, start, read, detect, identity, end, sizes, owner, same):
        if start < self.opened + self.warmup:
            return False
        if self.start is None:
            self.start = start
        self.end = end
        self.rows.append(((read-start)*1000, (detect-read)*1000, (identity-detect)*1000,
                          (end-identity)*1000, (end-read)*1000, (end-start)*1000,
                          len(sizes), sizes[0] if len(sizes)==1 else None, owner, same))
        return end-self.start >= self.seconds

    def summary(self):
        if not self.rows:
            return {'frames': 0}
        cols = list(zip(*self.rows))
        duration = self.end-self.start
        def stats(seq):
            return [round(float(x), 2) for x in np.percentile(seq, [50, 95])]
        sizes = [v for v in cols[7] if v is not None]
        names = ('read_ms','detection_ms','recognition_tracking_ms','draw_gui_ms',
                 'read_return_to_gui_return_ms','full_cycle_ms')
        return dict(frames=len(self.rows), elapsed_s=duration,
                    completed_fps=len(self.rows)/duration,
                    consecutive_identical_frames=sum(cols[9]),
                    changed_frame_fps=(len(self.rows)-sum(cols[9]))/duration,
                    p50_p95={name:stats(col) for name,col in zip(names,cols)},
                    no_face_frames=cols[6].count(0), one_face_frames=len(sizes),
                    multiple_face_frames=sum(n>1 for n in cols[6]),
                    face_short_px_p50_p95=stats(sizes) if sizes else None,
                    size_eligible_frames=sum(v>=80 for v in sizes),
                    owner_frames=sum(cols[8]), owner_fraction=sum(cols[8])/len(self.rows))

    def write(self, meta):
        path = Path(__file__).resolve().parent/'results'
        path.mkdir(exist_ok=True)
        name = path/(uuid.uuid4().hex+'.json')
        name.write_text(json.dumps(meta | self.summary(), indent=2), encoding='utf-8')
        print('Result:', name, flush=True)
