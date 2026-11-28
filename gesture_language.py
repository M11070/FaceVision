"""Gesture Language v0.1: a small FSM over confirmed, tracked observations."""
from collections import Counter
from math import isfinite
from threading import Lock, Thread
import webbrowser

CHATGPT_URL = 'https://chatgpt.com/'
SEQUENCE = ('Open_Palm', 'Victory', 'Thumb_Up')
READY = 'GESTURE LANGUAGE: READY'


def open_chatgpt():
    """The only executable action. No caller-supplied URL or command."""
    return webbrowser.open(CHATGPT_URL, new=2)


class BrowserAction:
    """Keep browser startup off the camera thread; at most one call in flight."""
    def __init__(self):
        self._lock = Lock()
        self._thread = None
        self._attempts = 0
        self._opened = 0
        self._error = None

    def submit(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._attempts += 1
            self._error = None
            self._thread = Thread(target=self._run, name='FaceVision-browser', daemon=True)
            self._thread.start()
            return True

    @property
    def busy(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self):
        try:
            opened = bool(open_chatgpt())
            with self._lock:
                self._opened += int(opened)
                if not opened:
                    self._error = 'The default browser did not acknowledge the launch.'
            print('ACTION: OPEN CHATGPT' if opened else 'ACTION: OPEN CHATGPT failed.', flush=True)
        except Exception as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
            print(f'ACTION: OPEN CHATGPT failed: {exc}', flush=True)

    def wait(self, timeout=2.):
        """For tests/diagnostics only; the camera loop never waits for a browser."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def stats(self):
        with self._lock:
            return dict(attempts=self._attempts, opened=self._opened, error=self._error,
                        pending=self._thread is not None and self._thread.is_alive())


class GestureLanguage:
    """Return True exactly once per completed sequence; perform no I/O here.

    Input hands must expose track_id, observed_at, confirmed_since,
    association_ambiguous, and a confirmed gesture (or None).
    """
    def __init__(self, hold_seconds=.300, timeout=3., cooldown=5., max_sample_age=.25):
        if not all(isfinite(v) and v > 0 for v in (hold_seconds, timeout, cooldown, max_sample_age)):
            raise ValueError('Invalid gesture language configuration.')
        self.hold_seconds, self.timeout = hold_seconds, timeout
        self.cooldown, self.max_sample_age = cooldown, max_sample_age
        self.stage = 0
        self._track_id = None
        self._candidate = None
        self._candidate_since = None
        self._last_sample = None
        self._last_accepted = None
        self._last_gesture = None
        self._cooldown_until = float('-inf')
        self._fresh_after = float('-inf')
        self._completed_at = None
        self._counts = Counter()

    def reset(self, now, reason='reset'):
        self.stage = 0
        self._track_id = None
        self._candidate = self._candidate_since = None
        self._last_sample = self._last_accepted = self._last_gesture = None
        # G/enrollment cannot bypass a cooldown or reuse an already held pose.
        self._fresh_after = max(now, self._cooldown_until)
        self._completed_at = None
        self._counts['reset_'+reason] += 1

    def _clear_candidate(self):
        self._candidate = self._candidate_since = None

    def update(self, hands, now, *, enabled=True, enrolling=False, action_pending=False):
        if not isfinite(now):
            raise ValueError('Time must be finite.')
        if not enabled or enrolling:
            if self._track_id is not None or self.stage:
                self.reset(now, 'enrollment' if enrolling else 'disabled')
            else:
                self._fresh_after = max(now, self._cooldown_until)
                self._completed_at = None
            return False
        if action_pending:
            # Preserve CONFIRMED/ACTION feedback while a browser starts slowly.
            self._fresh_after = max(now, self._cooldown_until)
            return False
        if now < self._cooldown_until:
            return False
        if self.stage and now-self._last_accepted > self.timeout:
            self.reset(now, 'timeout')
            return False

        if self._track_id is None:
            candidates = [h for h in hands if h.track_id is not None
                          and not h.association_ambiguous and h.gesture == SEQUENCE[0]]
            if len(candidates) != 1:
                self._clear_candidate()
                return False
            hand = candidates[0]
            if hand.confirmed_since is None or hand.confirmed_since < self._fresh_after:
                return False
            self._track_id = hand.track_id
        else:
            matching = [h for h in hands if h.track_id == self._track_id]
            if len(matching) != 1:
                self.reset(now, 'hand_lost')
                return False
            hand = matching[0]

        if hand.association_ambiguous:
            self.reset(now, 'ambiguous')
            return False
        observed = hand.observed_at
        if not isfinite(observed) or not 0 <= now-observed <= self.max_sample_age:
            self.reset(now, 'stale')
            return False
        if observed < self._fresh_after:
            return False
        if self._last_sample is not None:
            if observed < self._last_sample:
                self.reset(now, 'out_of_order')
                return False
            if observed == self._last_sample:
                return False  # cached UI snapshot is not another inference
            if observed-self._last_sample > self.max_sample_age:
                self.reset(now, 'sample_gap')
                return False
        self._last_sample = observed

        gesture, since = hand.gesture, hand.confirmed_since
        if (gesture is None or since is None or not isfinite(since)
                or since > observed or gesture == self._last_gesture):
            self._clear_candidate()
            return False
        candidate = (gesture, since)
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = observed
        if observed-self._candidate_since+1e-9 < self.hold_seconds:
            return False
        if gesture != SEQUENCE[self.stage]:
            self.reset(now, 'wrong_gesture')
            return False

        self.stage += 1
        self._last_gesture, self._last_accepted = gesture, now
        self._counts['accepted_'+gesture] += 1
        self._clear_candidate()
        if self.stage < len(SEQUENCE):
            return False

        self._counts['completions'] += 1
        self._cooldown_until = now+self.cooldown
        self.reset(now, 'completed')
        self._completed_at = now
        return True

    def status(self, now):
        if self._completed_at is not None and now < self._cooldown_until:
            return ('STEP 3/3 — CONFIRMED' if now-self._completed_at < .4
                    else 'ACTION: OPEN CHATGPT')
        return {1: 'STEP 1/3 — OPEN PALM', 2: 'STEP 2/3 — VICTORY'}.get(self.stage, READY)

    def stats(self):
        return dict(self._counts) | dict(stage=self.stage, hold_ms=self.hold_seconds*1000,
                                         timeout_s=self.timeout, cooldown_s=self.cooldown)
