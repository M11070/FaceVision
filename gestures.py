"""Independent, latest-frame hand/gesture pipeline. Never opens a camera."""
from collections import Counter, deque
from dataclasses import dataclass, replace
from itertools import product
from math import hypot, isfinite
from pathlib import Path
from threading import Condition, Thread
import hashlib
import sys
import time

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent / 'models' / 'gesture_recognizer.task'
MODEL_SHA256 = '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482'
LABELS = {
    'Open_Palm': 'Open palm', 'Closed_Fist': 'Closed fist',
    'Thumb_Up': 'Thumbs up', 'Thumb_Down': 'Thumbs down',
    'Victory': 'Victory', 'Pointing_Up': 'Pointing up', 'ILoveYou': 'I love you',
}


@dataclass(frozen=True)
class Hand:
    box: tuple  # normalized x, y, width, height in the original, unmirrored image
    handedness: str
    landmarks: tuple  # 21 normalized (x, y, z) points, not drawn
    world_landmarks: tuple
    gesture: str | None
    confidence: float


def decode(result, threshold=.60, margin=.15):
    """Keep None in the competition: filtering it out could force a gesture."""
    hands = []
    for i, landmarks in enumerate(result.hand_landmarks[:2]):
        points = tuple((float(p.x), float(p.y), float(p.z)) for p in landmarks)
        if len(points) != 21 or not all(isfinite(v) for p in points for v in p):
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        left, top = max(0., min(xs)-.025), max(0., min(ys)-.025)
        right, bottom = min(1., max(xs)+.025), min(1., max(ys)+.025)
        if right <= left or bottom <= top:
            continue
        categories = sorted(result.gestures[i], key=lambda c: c.score, reverse=True)
        best = categories[0] if categories else None
        second = categories[1].score if len(categories) > 1 else 0.
        gesture = (best.category_name if best and best.category_name in LABELS
                   and best.score >= threshold and best.score-second >= margin else None)
        sides = result.handedness[i]
        side = max(sides, key=lambda c: c.score) if sides else None
        handedness = (side.category_name if side and side.score >= .60
                      and side.category_name in ('Left', 'Right') else 'Hand')
        world = tuple((float(p.x), float(p.y), float(p.z))
                      for p in result.hand_world_landmarks[i])
        hands.append(Hand((left, top, right-left, bottom-top), handedness, points,
                          world, gesture, float(best.score) if gesture else 0.))
    return hands


class Stabilizer:
    """Match up to two hands spatially; confirm a label over time per hand."""
    def __init__(self, confirmation=.08, max_gap=.25):
        self.confirmation, self.max_gap = confirmation, max_gap
        self.tracks = []

    @staticmethod
    def affinity(a, b):
        x, y, w, h = a.box
        u, v, s, t = b.box
        distance = hypot(x+w/2-u-s/2, y+h/2-v-t/2)
        limit = max(.12, .8*max(hypot(w, h), hypot(s, t)))
        if distance >= limit:
            return 0.
        score = 1-distance/limit
        if a.handedness != 'Hand' and b.handedness != 'Hand':
            score += .35 if a.handedness == b.handedness else -.35
        return max(0., score)

    def update(self, hands, now):
        old = [tr for tr in self.tracks if 0 <= now-tr['seen'] <= self.max_gap]
        weights = [[self.affinity(h, tr['hand']) for tr in old] for h in hands]
        # Exhaustive assignment is tiny (at most 3**2), avoiding greedy swaps.
        options = product(range(-1, len(old)), repeat=len(hands))
        valid = [a for a in options if len([i for i in a if i >= 0]) == len(set(i for i in a if i >= 0))
                 and all(j < 0 or weights[i][j] > .15 for i, j in enumerate(a))]
        assignment = max(valid, key=lambda a: sum(weights[i][j] if j >= 0 else 0.
                                                for i, j in enumerate(a)), default=())
        current, views = [], []
        for i, (hand, j) in enumerate(zip(hands, assignment)):
            ranked = sorted(weights[i], reverse=True)
            ambiguous = len(ranked) > 1 and ranked[0]-ranked[1] < .15
            tr = old[j].copy() if j >= 0 and not ambiguous else {}
            history = [(t, label) for t, label in tr.get('history', [])
                       if 0 <= now-t <= self.max_gap][-4:] + [(now, hand.gesture)]
            agreed = [t for t, label in history if label == hand.gesture]
            stable = (hand.gesture if hand.gesture is not None and len(agreed) >= 3
                      and now-agreed[0] >= self.confirmation else None)
            tr['history'] = history
            views.append(replace(hand, gesture=stable))
            tr.update(hand=hand, seen=now)
            current.append(tr)
        # A missing or unknown hand loses its previous label immediately.
        self.tracks = current
        return tuple(views)


class GesturePipeline:
    """CPU worker with a one-frame mailbox, bounded results, and no GUI calls."""
    def __init__(self, enabled=True, max_fps=20., max_width=640, stale_after=.25):
        if max_fps <= 0 or max_width < 1 or stale_after <= 0:
            raise ValueError('Neispravna konfiguracija gestova.')
        self.max_fps, self.max_width, self.stale_after = max_fps, max_width, stale_after
        self._condition = Condition()
        self._enabled = False
        self._closed = False
        self._generation = 0
        self._pending = None
        self._latest = (0., ())
        self._thread = None
        self._error = None
        self._ready = False
        self._stats = Counter()
        self._latencies = deque(maxlen=5000)
        self._inference = deque(maxlen=5000)
        self.set_enabled(enabled)

    @property
    def enabled(self):
        with self._condition:
            return self._enabled

    def set_enabled(self, enabled):
        with self._condition:
            if self._closed:
                return False
            self._enabled = bool(enabled)
            self._generation += 1
            self._pending, self._latest = None, (0., ())
            if enabled and (self._thread is None or not self._thread.is_alive()):
                self._error = None
                self._thread = Thread(target=self._work, name='FaceVision-hands', daemon=True)
                self._thread.start()
            self._condition.notify_all()
            return self._enabled

    def submit(self, frame, captured):
        # OpenCV returns a new array per read; app.py never draws into this array.
        with self._condition:
            if not self._enabled or self._closed:
                return
            self._stats['submitted'] += 1
            self._stats['replaced_pending'] += int(self._pending is not None)
            self._pending = (frame, captured, self._generation)
            self._condition.notify()

    def snapshot(self, now=None):
        now = time.perf_counter() if now is None else now
        with self._condition:
            captured, hands = self._latest
            return hands if self._enabled and 0 <= now-captured <= self.stale_after else ()

    def stats(self):
        with self._condition:
            def percentiles(values):
                return [round(float(x), 2) for x in np.percentile(values, [50, 95])] if values else None
            return dict(self._stats) | dict(
                enabled=self._enabled, ready=self._ready, error=self._error,
                max_inference_fps=self.max_fps, max_input_width=self.max_width,
                inference_ms_p50_p95=percentiles(self._inference),
                capture_to_result_ms_p50_p95=percentiles(self._latencies),
                stats_scope='entire session including warmup; last 5000 timings',
            )

    def _work(self):
        recognizer = None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.components.processors import classifier_options
            if not MODEL.is_file() or hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
                raise RuntimeError('Nedostaje ili je ostecen models/gesture_recognizer.task.')
            options = vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL),
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=.6,
                min_hand_presence_confidence=.6, min_tracking_confidence=.5,
                canned_gesture_classifier_options=classifier_options.ClassifierOptions(max_results=-1),
            )
            recognizer = vision.GestureRecognizer.create_from_options(options)
            with self._condition:
                self._ready = True
                self._condition.notify_all()
            stabilizer, generation, last_start, timestamp = Stabilizer(), -1, 0., -1
            while True:
                with self._condition:
                    while not self._closed:
                        delay = 1/self.max_fps-(time.perf_counter()-last_start)
                        if self._enabled and self._pending is not None and delay <= 0:
                            break
                        self._condition.wait(timeout=max(.001, delay) if self._enabled and self._pending else None)
                    if self._closed:
                        return
                    frame, captured, current_generation = self._pending
                    self._pending = None
                if time.perf_counter()-captured > self.stale_after:
                    continue
                if generation != current_generation:
                    stabilizer, generation = Stabilizer(), current_generation
                last_start = time.perf_counter()
                h, w = frame.shape[:2]
                if w > self.max_width:
                    frame = cv2.resize(frame, (self.max_width, round(h*self.max_width/w)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(timestamp+1, int(captured*1000))
                infer_start = time.perf_counter()
                result = recognizer.recognize_for_video(image, timestamp)
                hands = stabilizer.update(decode(result), captured)
                finished = time.perf_counter()
                with self._condition:
                    self._stats['inferences'] += 1
                    self._stats['two_hand_results'] += int(len(hands) == 2)
                    self._inference.append((finished-infer_start)*1000)
                    self._latencies.append((finished-captured)*1000)
                    for hand in hands:
                        self._stats['hand_'+hand.handedness] += 1
                        if hand.gesture:
                            self._stats['gesture_'+hand.gesture] += 1
                    if self._enabled and generation == self._generation and not self._closed:
                        self._latest = (captured, hands)
        except Exception as exc:
            with self._condition:
                self._error = f'{type(exc).__name__}: {exc}'
                self._enabled, self._pending, self._latest = False, None, (0., ())
                self._condition.notify_all()
            print(f'Gestovi nisu dostupni: {self._error} Lica nastavljaju da rade.', file=sys.stderr, flush=True)
        finally:
            if recognizer is not None:
                recognizer.close()
            with self._condition:
                self._ready = False

    def close(self):
        with self._condition:
            self._closed, self._enabled = True, False
            self._pending, self._latest = None, (0., ())
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.)


def draw_hands(canvas, hands, size, top=42):
    """Only a thin box and a small label, within the existing camera area."""
    width, height = size
    for hand in hands:
        x, y, w, h = hand.box
        x1, y1 = max(0, int(x*width)), max(0, int(y*height))
        x2, y2 = min(width-1, int((x+w)*width)), min(height-1, int((y+h)*height))
        label = hand.handedness
        if hand.gesture:
            label += ': '+LABELS[hand.gesture]
        color = (210, 190, 100)
        cv2.rectangle(canvas, (x1, y1+top), (x2, y2+top), color, 1)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
        label_x = max(0, min(x1, width-text_width-2))
        label_y = y1-6 if y1 >= 20 else min(height-5, y2+17)
        cv2.putText(canvas, label, (label_x, label_y+top), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, color, 1, cv2.LINE_AA)
