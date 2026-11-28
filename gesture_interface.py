"""Four fixed applications and an edge-triggered menu over confirmed hands."""
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from threading import Lock, Thread
import webbrowser


@dataclass(frozen=True)
class Application:
    name: str
    destination: str


APPLICATIONS = (
    Application('ChatGPT', 'https://chatgpt.com/'),
    Application('Browser', 'https://www.google.com/'),
    Application('YouTube', 'https://www.youtube.com/'),
    Application('GitHub', 'https://github.com/'),
)


def open_application(index):
    if type(index) is not int or not 0 <= index < len(APPLICATIONS):
        raise ValueError('Only the four predefined applications are allowed.')
    return bool(webbrowser.open(APPLICATIONS[index].destination,new=2))


class LauncherAction:
    """One nonblocking browser request at a time; no URLs accepted from callers."""
    def __init__(self):
        self._lock, self._thread = Lock(), None
        self._attempts, self._opened = Counter(), Counter()
        self._error = None

    @property
    def busy(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def submit(self, index):
        if type(index) is not int or not 0 <= index < len(APPLICATIONS):
            raise ValueError('Unknown application.')
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._attempts[APPLICATIONS[index].name] += 1
            self._error = None
            self._thread = Thread(target=self._run, args=(index,), name='FaceVision-launch', daemon=True)
            self._thread.start()
            return True

    def _run(self, index):
        try:
            opened = open_application(index)
            with self._lock:
                self._opened[APPLICATIONS[index].name] += int(opened)
                if not opened:
                    self._error = 'The browser did not acknowledge the launch.'
            print('LAUNCH:', APPLICATIONS[index].name, 'OK' if opened else 'FAILED', flush=True)
        except Exception as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
            print('LAUNCH FAILED:', exc, flush=True)

    def wait(self, timeout=2.):
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def stats(self):
        with self._lock:
            return dict(attempts=dict(self._attempts), opened=dict(self._opened), error=self._error,
                        pending=self._thread is not None and self._thread.is_alive())


class MenuController:
    """Pure FSM. No AI, GUI or I/O. One event per confirmed gesture transition."""
    def __init__(self, hold_seconds=.3, inactivity=10., cooldown=5., max_sample_age=.25):
        if not all(isfinite(v) and v > 0 for v in (hold_seconds, inactivity, max_sample_age)):
            raise ValueError('Invalid gesture hold or inactivity duration.')
        if not isfinite(cooldown) or cooldown < 0:
            raise ValueError('Invalid launch cooldown.')
        self.hold_seconds, self.inactivity = hold_seconds, inactivity
        self.cooldown, self.max_sample_age = cooldown, max_sample_age
        self.visible, self.selected = False, 0
        self._track_id = self._last_sample = None
        self._candidate = self._candidate_since = self._latched = None
        self._latched_at = None
        self._last_activity = None
        self._fresh_after = self._cooldown_until = float('-inf')
        self._counts = Counter()
        self.reason = 'ready'

    def close(self, now, reason='cancel'):
        if self.visible or self._track_id is not None:
            self._counts['close_'+reason] += 1
        self.visible, self.selected = False, 0
        self._track_id = self._last_sample = None
        self._candidate = self._candidate_since = self._latched = None
        self._latched_at = None
        self._last_activity = None
        self._fresh_after = max(now, self._cooldown_until)
        self.reason = reason

    def update(self, hands, now, *, enabled=True, enrolling=False, action_pending=False):
        if not isfinite(now):
            raise ValueError('Time must be finite.')
        if not enabled or enrolling:
            self.close(now, 'enrollment' if enrolling else 'disabled')
            return None
        if action_pending or now < self._cooldown_until:
            self._fresh_after = max(now, self._cooldown_until)
            return None
        if self.visible and now-self._last_activity >= self.inactivity:
            self.close(now, 'timeout')
            return None
        if self._track_id is None:
            candidates = [h for h in hands if h.track_id is not None and not h.association_ambiguous
                          and h.gesture == 'Open_Palm' and h.confirmed_since is not None
                          and h.confirmed_since >= self._fresh_after]
            if len(candidates) != 1:
                self._candidate = self._candidate_since = None
                return None
            hand = candidates[0]
            self._track_id = hand.track_id
        else:
            matches = [h for h in hands if h.track_id == self._track_id]
            if len(matches) != 1:
                self.close(now, 'hand_lost')
                return None
            hand = matches[0]
        if hand.association_ambiguous:
            self.close(now, 'ambiguous')
            return None
        observed = hand.observed_at
        if not isfinite(observed) or not 0 <= now-observed <= self.max_sample_age:
            self.close(now, 'stale')
            return None
        if observed < self._fresh_after:
            return None
        if self._last_sample is not None:
            if observed < self._last_sample or observed-self._last_sample > self.max_sample_age:
                self.close(now, 'sample_gap')
                return None
            if observed == self._last_sample:
                return None
        self._last_sample = observed
        gesture, since = hand.gesture, hand.confirmed_since
        if gesture is None or since is None or not isfinite(since) or since > observed:
            neutral_since = getattr(hand,'neutral_since',None)
            if (self.visible and self._latched is not None and neutral_since is not None
                    and isfinite(neutral_since) and self._latched_at <= neutral_since <= observed
                    and observed-neutral_since+1e-9 >= self.hold_seconds):
                # Confident model None is a release, never an action prediction.
                self._latched = self._latched_at = None
                self._counts['neutral_releases'] += 1
            self._candidate = self._candidate_since = None
            return None
        if gesture == self._latched:
            self._candidate = self._candidate_since = None
            return None
        candidate = (gesture, since)
        if candidate != self._candidate:
            self._candidate, self._candidate_since = candidate, observed
        if observed-self._candidate_since+1e-9 < self.hold_seconds:
            return None
        self._latched = gesture
        self._latched_at = observed
        self._candidate = self._candidate_since = None
        if not self.visible:
            if gesture == 'Open_Palm':
                self.visible, self.selected = True, 0
                self._last_activity, self.reason = now, 'open'
                self._counts['activations'] += 1
            return None
        if gesture == 'Thumb_Down':
            self.close(now, 'cancel')
        elif gesture == 'Thumb_Up':
            selection = self.selected
            self._counts['confirmed_'+APPLICATIONS[selection].name] += 1
            self._cooldown_until = now+self.cooldown
            self.close(now, 'launch')
            return selection
        elif gesture in ('Open_Palm', 'Victory'):
            self._last_activity = now
            if gesture == 'Victory':
                self.selected = (self.selected+1) % len(APPLICATIONS)
                self._counts['selection_changes'] += 1
                self.reason = 'next'
        return None

    def stats(self):
        return dict(self._counts) | dict(visible=self.visible, selected=self.selected, reason=self.reason,
                                         hold_ms=self.hold_seconds*1000, inactivity_s=self.inactivity,
                                         cooldown_s=self.cooldown)

    def hint(self):
        return 'Relax your hand, then repeat Victory.' if self._latched == 'Victory' else 'Ready for the next gesture.'
