import unittest
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import patch

from gesture_language import BrowserAction, CHATGPT_URL, GestureLanguage, READY


def observation(gesture, at, since, track=(1, 1), ambiguous=False):
    return NS(track_id=track, gesture=gesture, observed_at=at,
              confirmed_since=since, association_ambiguous=ambiguous)


class LanguageChecks(unittest.TestCase):
    def setUp(self):
        self.fsm = GestureLanguage()

    def hold(self, label, start, duration=.35, *, track=(1, 1), since=None, **kwargs):
        since = start if since is None else since
        return [self.fsm.update([observation(label, round(start+i*.05, 8), since, track)],
                                round(start+i*.05, 8), **kwargs)
                for i in range(round(duration/.05)+1)]

    def sequence(self, start=0.):
        return (self.hold('Open_Palm', start)+self.hold('Victory', start+.4)
                +self.hold('Thumb_Up', start+.8))

    def test_correct_sequence_and_status(self):
        self.assertEqual(self.fsm.status(0), READY)
        self.assertFalse(any(self.hold('Open_Palm', 0)))
        self.assertEqual(self.fsm.status(.35), 'STEP 1/3 — OPEN PALM')
        self.assertFalse(any(self.hold('Victory', .4)))
        self.assertEqual(self.fsm.status(.75), 'STEP 2/3 — VICTORY')
        self.assertEqual(sum(self.hold('Thumb_Up', .8)), 1)
        self.assertEqual(self.fsm.status(1.15), 'STEP 3/3 — CONFIRMED')
        self.assertEqual(self.fsm.status(1.6), 'ACTION: OPEN CHATGPT')
        self.assertEqual(self.fsm.status(6.2), READY)

    def test_each_confirmed_gesture_requires_300_ms(self):
        for t in (0., .1, .2, .29):
            self.assertFalse(self.fsm.update([observation('Open_Palm', t, 0)], t))
            self.assertEqual(self.fsm.stage, 0)
        self.fsm.update([observation('Open_Palm', .3, 0)], .3)
        self.assertEqual(self.fsm.stage, 1)

    def test_incorrect_gesture_order(self):
        self.assertFalse(any(self.hold('Victory', 0)+self.hold('Thumb_Up', .4)))
        self.hold('Open_Palm', 1.)
        self.hold('Thumb_Up', 1.4)
        self.assertEqual(self.fsm.stage, 0)
        self.assertEqual(self.fsm.stats()['reset_wrong_gesture'], 1)
        self.assertFalse(any(self.hold('Victory', 1.8)))

    def test_sequence_timeout_and_held_pose_does_not_restart(self):
        self.hold('Open_Palm', 0, duration=3.7)
        self.assertEqual(self.fsm.stage, 0)
        self.assertEqual(self.fsm.stats()['reset_timeout'], 1)
        self.assertFalse(any(self.hold('Victory', 3.75)))

    def test_three_second_boundary_allows_acceptance(self):
        self.hold('Open_Palm', 0, duration=2.95)
        self.hold('Victory', 3., duration=.3)
        self.assertEqual(self.fsm.stage, 2)

    def test_repeated_gesture_frames_never_advance_another_stage(self):
        self.hold('Open_Palm', 0, duration=1.)
        self.assertEqual(self.fsm.stage, 1)
        self.hold('Open_Palm', 1.05, since=0., duration=1.)
        self.assertEqual(self.fsm.stage, 1)
        self.assertEqual(self.fsm.stats()['accepted_Open_Palm'], 1)

    def test_cached_snapshot_cannot_prove_duration(self):
        sample = observation('Open_Palm', 0., 0.)
        for t in (0., .05, .1, .2, .25):
            self.fsm.update([sample], t)
        self.assertEqual(self.fsm.stage, 0)
        self.fsm.update([sample], .31)
        self.assertEqual(self.fsm.stage, 0)

    def test_accidental_short_poses_do_not_activate(self):
        rows = self.hold('Open_Palm', 0., .15)+self.hold('Victory', .2, .15)+self.hold('Thumb_Up', .4, .15)
        self.assertFalse(any(rows))
        self.assertEqual(self.fsm.stage, 0)

    def test_uncertain_gap_restarts_dwell_even_if_ui_missed_unknown_result(self):
        self.hold('Open_Palm', 0., .2)
        self.fsm.update([observation('Open_Palm', .25, .24)], .25)
        self.fsm.update([observation('Open_Palm', .35, .24)], .35)
        self.assertEqual(self.fsm.stage, 0)
        self.hold('Open_Palm', .4, .2, since=.24)
        self.assertEqual(self.fsm.stage, 1)

    def test_unknown_is_neither_wrong_gesture_nor_a_new_transition(self):
        self.hold('Open_Palm', 0.)
        self.hold(None, .4, .15)
        self.hold('Open_Palm', .6)
        self.assertEqual(self.fsm.stage, 1)
        self.assertEqual(self.fsm.stats()['accepted_Open_Palm'], 1)

    def test_different_hand_cannot_continue_sequence(self):
        self.hold('Open_Palm', 0.)
        self.assertFalse(any(self.hold('Victory', .4, track=(1, 2))))
        self.assertFalse(any(self.hold('Thumb_Up', .8, track=(1, 2))))
        self.assertEqual(self.fsm.stage, 0)

    def test_ambiguous_hand_association(self):
        self.hold('Open_Palm', 0.)
        self.fsm.update([observation('Victory', .4, .4, ambiguous=True)], .4)
        self.assertEqual(self.fsm.stage, 0)
        self.assertEqual(self.fsm.stats()['reset_ambiguous'], 1)

    def test_hand_disappearance_and_epoch_change_reset(self):
        self.hold('Open_Palm', 0.)
        self.fsm.update([], .4)
        self.assertEqual(self.fsm.stage, 0)
        self.hold('Open_Palm', .5)
        self.hold('Victory', .9, track=(2, 1))
        self.assertEqual(self.fsm.stage, 0)

    def test_reordered_two_hands_preserve_locked_hand(self):
        self.hold('Open_Palm', 0.)
        for i in range(8):
            t=round(.4+i*.05,8)
            a=observation('Victory',t,.4)
            b=observation('Thumb_Down',t,0.,track=(1,2))
            self.fsm.update([a,b] if i%2 else [b,a],t)
        self.assertEqual(self.fsm.stage, 2)
        self.assertEqual(sum(self.hold('Thumb_Up', .8)), 1)

    def test_two_open_palms_do_not_arbitrarily_choose_a_command_hand(self):
        for i in range(10):
            t=i*.05
            self.fsm.update([observation('Open_Palm',t,0.),
                             observation('Open_Palm',t,0.,track=(1,2))],t)
        self.assertEqual(self.fsm.stage, 0)

    def test_cooldown_and_fresh_sequence_required(self):
        self.assertEqual(sum(self.sequence()), 1)
        self.assertFalse(any(self.sequence(1.5)))
        # A palm started during cooldown cannot be carried across its boundary.
        self.assertFalse(any(self.hold('Open_Palm', 3., 3.3)))
        self.assertEqual(self.fsm.stage, 0)
        self.hold('Victory', 6.4)
        self.assertEqual(sum(self.sequence(6.8)), 1)
        self.assertEqual(self.fsm.stats()['completions'], 2)

    def test_single_browser_activation_with_fixed_url(self):
        browser = BrowserAction()
        with patch('gesture_language.webbrowser.open', return_value=True) as mocked:
            for trigger in self.sequence()+self.hold('Thumb_Up',1.2,2.,since=.8):
                if trigger:
                    browser.submit()
            browser.wait()
            mocked.assert_called_once_with(CHATGPT_URL,new=2)
        self.assertEqual(browser.stats()['attempts'],1)

    def test_disabled_recognition_resets_partial_sequence(self):
        self.hold('Open_Palm', 0.)
        self.hold('Victory', .4, enabled=False)
        self.assertEqual(self.fsm.stage, 0)
        self.assertFalse(any(self.hold('Victory', .8)+self.hold('Thumb_Up', 1.2)))

    def test_cooldown_survives_disable_reset(self):
        self.sequence()
        self.fsm.update([],1.2,enabled=False)
        self.assertFalse(any(self.sequence(1.5)))

    def test_enrollment_disables_and_requires_fresh_pose(self):
        self.hold('Open_Palm', 0.)
        self.hold('Victory', .4, enrolling=True)
        self.assertFalse(any(self.hold('Open_Palm', .8, since=0.)))
        self.assertEqual(self.fsm.stage,0)
        self.hold('Victory',1.2)
        self.assertEqual(sum(self.sequence(1.6)),1)

    def test_timestamp_gap_and_out_of_order_results_reset(self):
        self.hold('Open_Palm', 0.)
        self.fsm.update([observation('Victory', .7, .7)], .7)
        self.assertEqual(self.fsm.stage,0)
        self.hold('Open_Palm', .8)
        self.fsm.update([observation('Victory', 1., 1.)], 1.15)
        self.assertEqual(self.fsm.stage,0)

    def test_browser_failure_does_not_retry_or_crash(self):
        browser = BrowserAction()
        with patch('gesture_language.webbrowser.open',side_effect=OSError('test')) as mocked:
            self.assertTrue(browser.submit())
            browser.wait()
            self.assertEqual(mocked.call_count,1)
            self.assertIn('OSError',browser.stats()['error'])

    def test_browser_startup_does_not_block_or_create_parallel_calls(self):
        browser = BrowserAction()
        started, release = Event(), Event()
        def slow(*args, **kwargs):
            started.set()
            release.wait(2.)
            return True
        with patch('gesture_language.webbrowser.open',side_effect=slow) as mocked:
            try:
                self.assertTrue(browser.submit())
                self.assertTrue(started.wait(1.))
                self.assertFalse(browser.submit())
            finally:
                release.set()
                browser.wait()
            self.assertEqual(mocked.call_count,1)

    def test_pending_action_preserves_status_and_rejects_new_sequences(self):
        self.sequence()
        self.fsm.update([],1.2,action_pending=True)
        self.assertEqual(self.fsm.status(1.2),'STEP 3/3 — CONFIRMED')
        self.fsm.update([],2.,action_pending=True)
        self.assertEqual(self.fsm.status(2.),'ACTION: OPEN CHATGPT')
        self.assertFalse(any(self.hold('Open_Palm',6.5,action_pending=True)))
        self.assertEqual(self.fsm.stats()['completions'],1)
        self.fsm.update([],7.,enabled=False,action_pending=True)
        self.assertEqual(self.fsm.status(7.),READY)


if __name__ == '__main__':
    unittest.main(verbosity=2)
