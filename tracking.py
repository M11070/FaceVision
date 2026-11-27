"""Privremeni tragovi. Koordinate su u pikselima, vreme u sekundama."""
from dataclasses import dataclass
from math import hypot


@dataclass
class Track:
    number: int
    box: tuple
    seen: float
    velocity: tuple = (0., 0.)
    first_match: float | None = None
    visible: bool = True
    ambiguous: bool = False
    confirmed: bool = False

    def predict(self, now):
        x, y, w, h = self.box
        dt = min(.3, max(0., now - self.seen))
        return x + self.velocity[0]*dt, y + self.velocity[1]*dt, w, h

    def clear(self):
        self.first_match = None
        self.confirmed = False


def affinity(a, b):
    x, y, w, h = a
    u, v, s, t = b
    intersection = max(0, min(x+w, u+s)-max(x, u))*max(0, min(y+h, v+t)-max(y, v))
    iou = intersection / max(1., w*h+s*t-intersection)
    distance = hypot(x+w/2-u-s/2, y+h/2-v-t/2) / max(1., (w+h+s+t)/4)
    ratio = min(w*h, s*t)/max(1., w*h, s*t)
    if ratio < .35 or (iou < .1 and distance > .65):
        return 0.
    return .65*iou + .35*max(0., 1-distance)


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.counter = 0

    def update(self, boxes, now):
        self.tracks = {n: tr for n, tr in self.tracks.items() if now-tr.seen <= .3}
        edges = []
        by_face = [[] for _ in boxes]
        by_track = {}
        for n, tr in self.tracks.items():
            for i, box in enumerate(boxes):
                a = affinity(tr.predict(now), box)
                if a:
                    edges.append((a, n, i))
                    by_face[i].append(a)
                    by_track.setdefault(n, []).append(a)
        def close(scores):
            ordered = sorted(scores, reverse=True)
            return len(ordered) > 1 and ordered[0]-ordered[1] < .15
        assigned = [None]*len(boxes)
        used = set()
        for _, n, i in sorted(edges, reverse=True):
            if n in used or assigned[i] is not None:
                continue
            tr = self.tracks[n]
            old = tr.box
            dt = max(.001, now-tr.seen)
            tr.velocity = tuple(.5*speed+.5*(boxes[i][j]-old[j])/dt
                                for j, speed in enumerate(tr.velocity))
            tr.ambiguous = close(by_track[n]) or close(by_face[i])
            tr.box, tr.seen, tr.visible = boxes[i], now, True
            if tr.ambiguous:
                tr.clear()
            assigned[i] = tr
            used.add(n)
        for n, tr in self.tracks.items():
            if n not in used:
                tr.visible = False
                tr.clear()
        for i, tr in enumerate(assigned):
            if tr is None:
                self.counter += 1
                tr = Track(self.counter, boxes[i], now, ambiguous=bool(by_face[i]))
                self.tracks[tr.number] = assigned[i] = tr
        return assigned


def decide(tracks, scores, threshold, now):
    candidates = [tr for tr, value in zip(tracks, scores)
                  if value is not None and value >= threshold]
    for tr in tracks:
        if len(candidates) != 1 or candidates[0] is not tr or tr.ambiguous:
            tr.clear()
        elif tr.first_match is None:
            tr.first_match = now
        else:
            tr.confirmed = now - tr.first_match >= .10


class Enrollment:
    def __init__(self, number):
        self.number = number
        self.samples = []
        self.last = float('-inf')

    def observe(self, tracks, features, now):
        if (len(tracks) != 1 or tracks[0].number != self.number or
                tracks[0].ambiguous or features[0] is None):
            return 'cancel'
        feature = features[0]
        if self.samples and float(self.samples[0] @ feature) < .45:
            return 'cancel'
        if now-self.last >= .45:
            self.samples.append(feature.copy())
            self.last = now
        return 'save' if len(self.samples) == 12 else 'continue'
