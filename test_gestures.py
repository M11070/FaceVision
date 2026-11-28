import time
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from threading import Event
from unittest.mock import patch

import numpy as np

from gestures import Hand, GesturePipeline, Stabilizer, decode, draw_hands


def hand(x=.2, side='Left', gesture='Open_Palm'):
    points = tuple((x+i*.005, .3+i*.01, 0.) for i in range(21))
    return Hand((x, .3, .15, .25), side, points, points, gesture, .95)


class GestureChecks(unittest.TestCase):
    def test_confidence_unknown_and_margin(self):
        def result(categories):
            return NS(hand_landmarks=[[NS(x=.2+i*.005,y=.3+i*.01,z=0) for i in range(21)]],
                      hand_world_landmarks=[[NS(x=0,y=0,z=0) for _ in range(21)]],
                      handedness=[[NS(category_name='Right',score=.99)]],
                      gestures=[[NS(category_name=name,score=score) for name,score in categories]])
        for categories in ([('Open_Palm',.59)], [('None',.95),('Open_Palm',.8)],
                           [('Open_Palm',.8),('Victory',.73)], [('Other',.99)], []):
            self.assertIsNone(decode(result(categories))[0].gesture)
        decoded = decode(result([('Thumb_Up',.95),('None',.03)]))[0]
        self.assertEqual(decoded.gesture,'Thumb_Up')
        self.assertEqual(decoded.handedness,'Right')
        self.assertEqual(len(decoded.landmarks),21)

    def test_motion_order_and_independent_labels(self):
        s=Stabilizer()
        left, right=hand(),hand(.65,'Right','Victory')
        for i in range(5):
            a=replace(left,box=(.2+i*.015,.3,.15,.25))
            views=s.update([right,a] if i%2 else [a,right],i*.05)
        self.assertEqual({h.handedness:h.gesture for h in views},
                         {'Left':'Open_Palm','Right':'Victory'})
        views=s.update([replace(a,gesture=None),right],.26)
        self.assertIsNone(views[0].gesture)
        self.assertEqual(views[1].gesture,'Victory')
        # A different gesture must be confirmed again, never inherit the old label.
        views=s.update([replace(a,gesture='Closed_Fist'),right],.31)
        self.assertIsNone(views[0].gesture)

    def test_one_uncertain_sample_does_not_restart_confirmation(self):
        s=Stabilizer()
        for now in (0,.05,.1):
            views=s.update([hand()],now)
        self.assertEqual(views[0].gesture,'Open_Palm')
        self.assertIsNone(s.update([hand(gesture=None)],.15)[0].gesture)
        self.assertEqual(s.update([hand()],.2)[0].gesture,'Open_Palm')

    def test_tracking_provenance_and_continuous_confirmation(self):
        s=Stabilizer(epoch=4)
        for i in range(7):
            views=s.update([hand()],i*.05)
        first_id=views[0].track_id
        self.assertEqual(first_id[0],4)
        self.assertAlmostEqual(views[0].confirmed_since,.1)
        self.assertAlmostEqual(views[0].observed_at,.3)
        self.assertFalse(views[0].association_ambiguous)
        uncertain=s.update([hand(gesture=None)],.35)[0]
        self.assertEqual(uncertain.track_id,first_id)
        self.assertIsNone(uncertain.confirmed_since)
        restored=s.update([hand()],.4)[0]
        self.assertEqual(restored.track_id,first_id)
        self.assertAlmostEqual(restored.confirmed_since,.4)
        s.update([],.45)
        self.assertNotEqual(s.update([hand()],.5)[0].track_id,first_id)

    def test_neutral_release_provenance_requires_confident_none_not_low_gesture_score(self):
        def result(name,score):
            return NS(hand_landmarks=[[NS(x=.2+i*.005,y=.3+i*.01,z=0) for i in range(21)]],
                      hand_world_landmarks=[[NS(x=0,y=0,z=0) for _ in range(21)]],
                      handedness=[[NS(category_name='Right',score=.99)]],
                      gestures=[[NS(category_name=name,score=score)]])
        self.assertTrue(decode(result('None',.95))[0].neutral)
        self.assertFalse(decode(result('Victory',.55))[0].neutral)
        self.assertFalse(decode(result('None',.65))[0].neutral)
        s=Stabilizer()
        for i in range(5):
            view=s.update(decode(result('None',.95)),i*.05)[0]
        self.assertEqual(view.neutral_since,0.)
        self.assertIsNone(view.gesture)
        self.assertIsNone(s.update(decode(result('Victory',.55)),.25)[0].neutral_since)
        self.assertEqual(s.update(decode(result('None',.95)),.3)[0].neutral_since,.3)

    def test_tracking_provenance_fails_closed_for_two_to_one_and_one_to_two(self):
        s=Stabilizer()
        s.update([hand(.2),hand(.24)],0)
        self.assertTrue(s.update([hand(.22)],.05)[0].association_ambiguous)
        s=Stabilizer()
        s.update([hand(.22)],0)
        views=s.update([hand(.2),hand(.24)],.05)
        self.assertTrue(any(h.association_ambiguous for h in views))

    def test_known_handedness_flip_breaks_command_continuity(self):
        s=Stabilizer()
        old=s.update([hand()],0)[0]
        changed=s.update([hand(side='Right')],.05)[0]
        self.assertNotEqual(old.track_id,changed.track_id)
        self.assertTrue(changed.association_ambiguous)
        self.assertIsNone(changed.confirmed_since)

    def test_global_assignment_tie_breaks_continuity_without_local_ties(self):
        s=Stabilizer()
        s.update([hand(.2),hand(.6)],0)
        weights={(0.25,0.2):1., (0.25,0.6):.7,
                 (0.55,0.2):.65, (0.55,0.6):.35}
        # Both total assignments score 1.35, despite separated row/column scores.
        with patch.object(s,'affinity',side_effect=lambda a,b:weights[a.box[0],b.box[0]]):
            views=s.update([hand(.25),hand(.55)],.05)
        self.assertTrue(all(h.association_ambiguous for h in views))
        self.assertTrue(all(h.confirmed_since is None for h in views))

    def test_brief_candidate_and_disappearance(self):
        s=Stabilizer()
        for now in (0,.05,.1):
            views=s.update([hand()],now)
        self.assertEqual(views[0].gesture,'Open_Palm')
        self.assertEqual(s.update([],.15),())
        self.assertIsNone(s.update([hand()],.2)[0].gesture)
        s.update([hand()],.25)
        self.assertIsNone(s.update([hand()],1.)[0].gesture)

    def test_overlay_stays_inside_camera_and_source_is_unchanged(self):
        source=np.zeros((480,640,3),np.uint8)
        canvas=np.pad(source,((42,78),(0,0),(0,0)),constant_values=17)
        draw_hands(canvas,[hand()],(640,480))
        self.assertTrue((canvas[:42]==17).all())
        self.assertTrue((canvas[522:]==17).all())
        self.assertFalse(source.any())

    def test_disabled_never_starts_model_and_stale_results_expire(self):
        with patch.object(GesturePipeline,'_work') as work:
            p=GesturePipeline(enabled=False)
            try:
                p.submit(np.zeros((10,10,3),np.uint8),time.perf_counter())
                self.assertIsNone(p._thread)
                work.assert_not_called()
                p._enabled=True
                p._latest=(10.,(hand(),))
                self.assertEqual(len(p.snapshot(10.1)),1)
                self.assertEqual(p.snapshot(10.3),())
                p.set_enabled(False)
                self.assertEqual(p.snapshot(10.1),())
            finally: p.close()

    def test_real_model_blank_frame_and_toggle(self):
        p=GesturePipeline()
        frame=np.zeros((480,640,3),np.uint8)
        try:
            deadline=time.perf_counter()+10
            while time.perf_counter()<deadline:
                p.submit(frame,time.perf_counter())
                if p.stats().get('inferences',0)>=2 or p.stats()['error']:
                    break
                time.sleep(.03)
            self.assertIsNone(p.stats()['error'])
            self.assertGreaterEqual(p.stats().get('inferences',0),2)
            self.assertEqual(p.snapshot(),())
            p.set_enabled(False)
            time.sleep(.1)  # allow any in-flight inference to complete
            before=p.stats()['inferences']
            p.submit(frame,time.perf_counter())
            time.sleep(.12)
            self.assertEqual(p.stats()['inferences'],before)
            p.set_enabled(True)
            p.submit(frame,time.perf_counter())
            deadline=time.perf_counter()+2
            while p.stats()['inferences']==before and time.perf_counter()<deadline:
                time.sleep(.02)
            self.assertGreater(p.stats()['inferences'],before)
        finally: p.close()
        self.assertFalse(p._thread.is_alive())

    def test_toggle_during_inference_discards_old_result(self):
        from mediapipe.tasks.python import vision
        started, release = Event(), Event()
        result=NS(hand_landmarks=[[NS(x=.2+i*.005,y=.3+i*.01,z=0) for i in range(21)]],
                  hand_world_landmarks=[[NS(x=0,y=0,z=0) for _ in range(21)]],
                  handedness=[[NS(category_name='Right',score=.99)]],
                  gestures=[[NS(category_name='Open_Palm',score=.99)]])
        class SlowRecognizer:
            def recognize_for_video(self, image, timestamp):
                started.set()
                release.wait(2)
                return result
            def close(self):
                pass
        with patch.object(vision.GestureRecognizer,'create_from_options',return_value=SlowRecognizer()):
            p=GesturePipeline()
            try:
                p.submit(np.zeros((480,640,3),np.uint8),time.perf_counter())
                self.assertTrue(started.wait(3))
                p.set_enabled(False)
                p.set_enabled(True)
                release.set()
                deadline=time.perf_counter()+2
                while p.stats().get('inferences',0)==0 and time.perf_counter()<deadline:
                    time.sleep(.01)
                self.assertEqual(p.stats().get('inferences'),1)
                self.assertEqual(p.snapshot(),())
            finally:
                release.set()
                p.close()


if __name__=='__main__':
    unittest.main(verbosity=2)
