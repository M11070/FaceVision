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
