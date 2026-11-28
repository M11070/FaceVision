from threading import Event
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import app


class EngineBoundaryChecks(unittest.TestCase):
    def test_one_camera_no_cv_gui_no_legacy_action_and_frame_processing_continues(self):
        args=NS(size='640x480',camera=0,reference=None,distance=None,no_gestures=False,
                gesture_fps=20.,seconds=0,warmup=0,mode='exp',threshold=.45,no_gesture_language=False)
        stop=Event()
        frame=np.zeros((480,640,3),np.uint8)
        cap=Mock()
        cap.read.return_value=True,frame
        cap.get.return_value=0.
        cap.getBackendName.return_value='MOCK'
        detector=Mock()
        detector.detect.return_value=0,None
        pipeline=Mock(enabled=True)
        pipeline.snapshot.return_value=()
        received=[]
        def observer(**values):
            received.append(values)
            if len(received)==3:
                stop.set()
        with patch('app.camera',return_value=cap) as camera, \
             patch('app.base.get_models',return_value=('unused','unused')), \
             patch('app.base.load_profile',return_value=None), \
             patch('app.cv2.FaceDetectorYN.create',return_value=detector), \
             patch('app.cv2.FaceRecognizerSF.create'), \
             patch('app.GesturePipeline',return_value=pipeline), \
             patch('app.GestureLanguage') as legacy, patch('app.BrowserAction') as action, \
             patch('app.cv2.namedWindow') as named, patch('app.cv2.imshow') as show, \
             patch('app.cv2.waitKey') as wait, patch('app.cv2.destroyAllWindows') as destroy:
            app.run(args,frame_observer=observer,stop_event=stop)
        camera.assert_called_once_with(0,(640,480))
        self.assertEqual(cap.read.call_count,3)
        self.assertEqual(detector.detect.call_count,3)
        self.assertEqual(pipeline.submit.call_count,3)
        legacy.assert_not_called(); action.assert_not_called()
        for gui in (named,show,wait,destroy):
            gui.assert_not_called()
        self.assertEqual(len(received),3)
        self.assertFalse(any('frame' in v or 'image' in v for v in received))
        cap.release.assert_called_once()
        pipeline.close.assert_called_once()


if __name__=='__main__':
    unittest.main(verbosity=2)
