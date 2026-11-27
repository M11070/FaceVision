import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import face_vision as base
from metrics import Measurement
from tracking import Tracker, Enrollment, decide
from app import usable


class Checks(unittest.TestCase):
    def test_models(self):
        a,b = base.get_models()
        det = cv2.FaceDetectorYN.create(a,'',(640,480),.85,.3,100)
        _,faces = det.detect(np.zeros((480,640,3),np.uint8))
        self.assertIsNone(faces)
        sf = cv2.FaceRecognizerSF.create(b,'')
        v = base.normalize(sf.feature(np.full((112,112,3),127,np.uint8)))
        self.assertAlmostEqual(base.score(v,v),1,places=5)

    def test_clock(self):
        m = Measurement(0,1)
        m.opened = 0.
        for i in range(30):
            t = i/30
            m.add(t,t+.002,t+.010,t+.013,(i+1)/30,[96],True,i==1)
        s = m.summary()
        self.assertEqual(s['completed_fps'],30)
        self.assertEqual(s['changed_frame_fps'],29)
        self.assertEqual(s['p50_p95']['read_ms'],[2,2])
        self.assertEqual(s['owner_frames'],30)

    def test_track(self):
        t = Tracker()
        a,b = (0,0,100,100),(300,0,100,100)
        u,v = t.update([a,b],0)
        self.assertEqual(t.update([b,a],.03),[v,u])
        t.update([],.06)
        self.assertFalse(u.visible)
        self.assertIs(t.update([a],.20)[0],u)
        self.assertNotEqual(t.update([a],.60)[0].number,u.number)

    def test_identity(self):
        t = Tracker()
        tracks = t.update([(0,0,100,100),(300,0,100,100)],0)
        for now in (0,.12):
            decide(tracks,[.62,None],.45,now)
        self.assertTrue(tracks[0].confirmed)
        decide(tracks,[None,None],.45,.14)
        self.assertFalse(tracks[0].confirmed)
        for now in (.15,.30):
            decide(tracks,[.62,.61],.45,now)
        self.assertFalse(any(x.confirmed for x in tracks))

    def test_ambiguity(self):
        t = Tracker()
        t.update([(0,0,90,90),(12,0,90,90)],0)
        tracks = t.update([(5,0,90,90)],.03)
        self.assertTrue(tracks[0].ambiguous)
        decide(tracks,[.90],.45,.03)
        self.assertFalse(tracks[0].confirmed)

    def test_enrollment(self):
        t = Tracker()
        tracks = t.update([(0,0,100,100)],0)
        e = Enrollment(tracks[0].number)
        v,w = np.eye(128,dtype=np.float32)[:2]
        self.assertEqual(e.observe(tracks,[v],0),'continue')
        self.assertEqual(e.observe([],[],.1),'cancel')
        self.assertEqual(e.observe(tracks,[w],.2),'cancel')
        self.assertEqual(e.observe(tracks,[None],.3),'cancel')
        self.assertEqual(e.observe(tracks*2,[v,v],.4),'cancel')
        other = t.update([(400,0,100,100)],.5)
        self.assertEqual(e.observe(other,[v],.5),'cancel')
        for i in range(1,12):
            state=e.observe(tracks,[v],i*.5)
        self.assertEqual(state,'save')
        with tempfile.TemporaryDirectory() as root, patch.object(base,'PROFILE',Path(root)/'owner.npy'):
            base.save_profile(e.samples)
            np.testing.assert_allclose(base.load_profile(),v)
            base.PROFILE.unlink()
            self.assertIsNone(base.load_profile())

    def test_small_face(self):
        f = np.zeros(15,np.float32)
        f[:4] = (100,100,79,110)
        self.assertFalse(usable(np.zeros((480,640,3),np.uint8),f))


if __name__ == '__main__':
    unittest.main(verbosity=2)
