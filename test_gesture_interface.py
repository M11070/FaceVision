import unittest
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import patch, call

from gesture_interface import APPLICATIONS, LauncherAction, MenuController, open_application


def sample(gesture, at, since, track=(1,1), ambiguous=False):
    return NS(gesture=gesture,observed_at=at,confirmed_since=since,track_id=track,
              association_ambiguous=ambiguous)


class MenuChecks(unittest.TestCase):
    def setUp(self):
        self.menu=MenuController()

    def hold(self,label,start,duration=.35,*,track=(1,1),since=None,**kwargs):
        since=start if since is None else since
        return [self.menu.update([sample(label,round(start+i*.05,8),since,track)],
                                 round(start+i*.05,8),**kwargs)
                for i in range(round(duration/.05)+1)]

    def test_exact_four_apps_in_required_order(self):
        self.assertEqual(tuple(a.name for a in APPLICATIONS),('ChatGPT','Browser','YouTube','GitHub'))
        self.assertEqual(tuple(a.destination for a in APPLICATIONS),
                         ('https://chatgpt.com/','https://www.google.com/','https://www.youtube.com/','https://github.com/'))

    def test_activate_defaults_to_chatgpt_and_requires_300ms(self):
        self.assertFalse(self.menu.visible)
        self.hold('Open_Palm',0.,.25)
        self.assertFalse(self.menu.visible)
        self.menu.update([sample('Open_Palm',.3,0.)],.3)
        self.assertTrue(self.menu.visible)
        self.assertEqual(self.menu.selected,0)

    def test_cycle_all_four_and_wrap(self):
        self.hold('Open_Palm',0.)
        for i in range(4):
            if i:
                self.hold('Closed_Fist',.8+(i-1)*.8)
            self.hold('Victory',.4+i*.8)
            self.assertEqual(self.menu.selected,(i+1)%4)
        self.assertEqual(self.menu.stats()['selection_changes'],4)

    def test_held_victory_has_only_one_event(self):
        self.hold('Open_Palm',0.)
        self.hold('Victory',.4,2.5)
        self.assertEqual(self.menu.selected,1)
        self.assertEqual(self.menu.stats()['selection_changes'],1)

    def test_classification_dropouts_cannot_repeat_held_victory(self):
        self.hold('Open_Palm',0.)
        self.hold('Victory',.4)
        self.hold(None,.8,.5)
        self.hold('Victory',1.35)
        self.assertEqual(self.menu.selected,1)

    def test_confident_neutral_release_rearms_victory_without_another_command(self):
        self.hold('Open_Palm',0.)
        self.hold('Victory',.4)
        self.assertTrue(self.menu.hint().startswith('Relax'))
        for i in range(8):
            t=round(.8+i*.05,8)
            neutral=sample(None,t,None)
            neutral.neutral_since=.8
            self.menu.update([neutral],t)
        self.assertEqual(self.menu.selected,1)
        self.assertEqual(self.menu.hint(),'Ready for the next gesture.')
        self.hold('Victory',1.2)
        self.assertEqual(self.menu.selected,2)
        self.assertEqual(self.menu.stats()['neutral_releases'],1)

    def test_open_palm_in_open_menu_preserves_selection_and_rearms_victory(self):
        self.hold('Open_Palm',0.)
        self.hold('Victory',.4)
        self.hold('Open_Palm',.8)
        self.assertEqual(self.menu.selected,1)
        self.hold('Victory',1.2)
        self.assertEqual(self.menu.selected,2)

    def test_launch_each_selected_app_exactly_once(self):
        for index in range(4):
            with self.subTest(index=index):
                self.menu=MenuController()
                self.hold('Open_Palm',0.)
                for step in range(index):
                    self.hold('Victory',.4+step*.8)
                    self.hold('Closed_Fist',.8+step*.8)
                events=self.hold('Thumb_Up',.4+index*.8,2.)
                self.assertEqual([e for e in events if e is not None],[index])
                self.assertFalse(self.menu.visible)

    def test_cancel_and_reopen_resets_selection(self):
        self.hold('Open_Palm',0.)
        self.hold('Victory',.4)
        self.assertTrue(all(e is None for e in self.hold('Thumb_Down',.8)))
        self.assertFalse(self.menu.visible)
        self.hold('Open_Palm',1.2)
        self.assertTrue(self.menu.visible)
        self.assertEqual(self.menu.selected,0)

    def test_timeout_closes_and_held_palm_does_not_reopen(self):
        self.hold('Open_Palm',0.,10.6)
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.stats()['activations'],1)
        self.assertEqual(self.menu.stats()['close_timeout'],1)

    def test_command_activity_refreshes_timeout_but_holding_does_not(self):
        self.hold('Open_Palm',0.,8.)
        self.hold('Victory',8.05)
        self.hold('Victory',8.45,9.3,since=8.05)
        self.assertTrue(self.menu.visible)
        self.hold('Victory',17.8,.7,since=8.05)
        self.assertFalse(self.menu.visible)

    def test_unclassified_and_short_gestures_cannot_activate_or_launch(self):
        self.hold('Thumb_Up',0.)
        self.hold('Victory',.4)
        self.hold('Open_Palm',.8,.15)
        self.hold('Thumb_Up',1.,.15)
        self.assertFalse(self.menu.visible)
        self.assertFalse(any(k.startswith('confirmed_') for k in self.menu.stats()))

    def test_lost_hand_closes_menu_and_other_hand_cannot_confirm(self):
        self.hold('Open_Palm',0.)
        self.menu.update([],.4)
        self.assertFalse(self.menu.visible)
        self.assertTrue(all(e is None for e in self.hold('Thumb_Up',.45,track=(1,2))))

    def test_different_hand_or_epoch_cannot_take_over(self):
        for track in ((1,2),(2,1)):
            self.menu=MenuController()
            self.hold('Open_Palm',0.)
            self.hold('Victory',.4,track=track)
            self.assertFalse(self.menu.visible)

    def test_ambiguous_association_closes(self):
        self.hold('Open_Palm',0.)
        self.menu.update([sample('Victory',.4,.4,ambiguous=True)],.4)
        self.assertFalse(self.menu.visible)
        self.assertEqual(self.menu.reason,'ambiguous')

    def test_two_open_palms_cannot_arbitrarily_select_command_hand(self):
        for i in range(10):
            t=i*.05
            self.menu.update([sample('Open_Palm',t,0.),sample('Open_Palm',t,0.,track=(1,2))],t)
        self.assertFalse(self.menu.visible)

    def test_other_hand_and_result_order_do_not_change_selection(self):
        self.hold('Open_Palm',0.)
        for i in range(8):
            t=round(.4+i*.05,8)
            a,b=sample('Victory',t,.4),sample('Thumb_Up',t,.4,track=(1,2))
            self.menu.update([a,b] if i%2 else [b,a],t)
        self.assertTrue(self.menu.visible)
        self.assertEqual(self.menu.selected,1)

    def test_cached_stale_and_out_of_order_results_are_rejected(self):
        h=sample('Open_Palm',0.,0.)
        for t in (0.,.1,.2,.25):
            self.menu.update([h],t)
        self.assertFalse(self.menu.visible)
        self.menu.update([h],.31)
        self.assertEqual(self.menu.reason,'stale')
        self.hold('Open_Palm',.4)
        self.menu.update([sample('Victory',.6,.6)],.8)
        self.assertFalse(self.menu.visible)

    def test_missing_samples_or_unknown_interrupt_confirmation(self):
        self.hold('Open_Palm',0.,.2)
        self.menu.update([sample('Open_Palm',.25,.24)],.25)
        self.menu.update([sample('Open_Palm',.4,.24)],.4)
        self.assertFalse(self.menu.visible)
        self.menu.update([sample('Open_Palm',.55,.24)],.55)
        self.assertTrue(self.menu.visible)
        self.menu.update([sample('Thumb_Up',.9,.9)],.9)
        self.assertFalse(self.menu.visible)

    def test_disabled_enrollment_and_pending_action_block_commands(self):
        for flag in ('enabled','enrolling','action_pending'):
            self.menu=MenuController()
            self.hold('Open_Palm',0.)
            kwargs={flag:False if flag=='enabled' else True}
            self.assertTrue(all(e is None for e in self.hold('Thumb_Up',.4,**kwargs)))
            if flag!='action_pending':
                self.assertFalse(self.menu.visible)

    def test_cooldown_requires_fresh_palm_and_cannot_be_bypassed_by_g(self):
        self.hold('Open_Palm',0.)
        self.assertIn(0,self.hold('Thumb_Up',.4))
        self.menu.update([],1.,enabled=False)
        self.hold('Open_Palm',1.1,5.)
        self.assertFalse(self.menu.visible)
        self.hold('Closed_Fist',6.15)
        self.hold('Open_Palm',6.55)
        self.assertTrue(self.menu.visible)

    def test_configurable_hold_and_inactivity(self):
        self.menu=MenuController(hold_seconds=.5,inactivity=1.,cooldown=0)
        self.hold('Open_Palm',0.,.4)
        self.assertFalse(self.menu.visible)
        self.hold('Open_Palm',.45,.1,since=0.)
        self.assertTrue(self.menu.visible)
        self.hold('Open_Palm',.6,1.,since=0.)
        self.assertFalse(self.menu.visible)

    def test_invalid_configuration_and_arbitrary_actions_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                MenuController(hold_seconds=value)
            with self.assertRaises(ValueError):
                MenuController(inactivity=value)
        for value in (-1,4,True,'https://example.com/','cmd.exe'):
            with self.assertRaises(ValueError):
                open_application(value)
            with self.assertRaises(ValueError):
                LauncherAction().submit(value)


class ActionChecks(unittest.TestCase):
    def test_fixed_destinations_and_no_browser_side_effects_in_tests(self):
        launcher=LauncherAction()
        with patch('gesture_interface.webbrowser.open',return_value=True) as browser:
            for index in range(4):
                self.assertTrue(launcher.submit(index))
                launcher.wait()
            self.assertEqual(browser.call_args_list,[call(a.destination,new=2) for a in APPLICATIONS])
        self.assertEqual(launcher.stats()['opened'],{a.name:1 for a in APPLICATIONS})

    def test_browser_opens_google_without_protocol_or_registry_workarounds(self):
        with patch('gesture_interface.webbrowser.open',return_value=True) as browser:
            self.assertTrue(open_application(1))
            browser.assert_called_once_with('https://www.google.com/',new=2)

    def test_failure_is_reported_without_retry(self):
        launcher=LauncherAction()
        with patch('gesture_interface.webbrowser.open',side_effect=OSError('test')) as browser:
            launcher.submit(0)
            launcher.wait()
            browser.assert_called_once_with('https://chatgpt.com/',new=2)
            self.assertIn('OSError',launcher.stats()['error'])

    def test_slow_launch_is_nonblocking_and_cannot_run_in_parallel(self):
        started,release=Event(),Event()
        def slow(*args,**kwargs):
            started.set(); release.wait(2.); return True
        launcher=LauncherAction()
        with patch('gesture_interface.webbrowser.open',side_effect=slow) as browser:
            try:
                self.assertTrue(launcher.submit(2))
                self.assertTrue(started.wait(1.))
                self.assertTrue(launcher.busy)
                self.assertFalse(launcher.submit(3))
            finally:
                release.set(); launcher.wait()
            self.assertEqual(browser.call_count,1)


if __name__=='__main__':
    unittest.main(verbosity=2)
