"""Check real app-loop boundaries without a camera, GUI or browser."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import app


class AppBoundaryChecks(unittest.TestCase):
    def run_keys(self, keys, with_face=False):
        frame=np.full((480,640,3),127,np.uint8)
        cap=Mock()
        cap.read.return_value=(True,frame)
        cap.get.return_value=0.
        cap.getBackendName.return_value='MOCK'
        detector=Mock()
        face=np.array([[100,100,120,140]+[0]*11],np.float32)
        detector.detect.return_value=(0,face if with_face else None)
        recognizer=Mock()
        recognizer.alignCrop.return_value=frame[:112,:112]
        recognizer.feature.return_value=np.eye(128,dtype=np.float32)[0]
        pipeline=Mock(enabled=True)
        pipeline.snapshot.return_value=()
        def toggle(enabled):
            pipeline.enabled=enabled
            return enabled
        pipeline.set_enabled.side_effect=toggle
        language=Mock()
        language.update.return_value=True  # completed immediately before a key event
        language.status.return_value='GESTURE LANGUAGE: READY'
        browser=Mock(busy=False)
        args=NS(size='640x480',camera=0,reference=None,distance=None,no_gestures=False,
                gesture_fps=20.,seconds=0,warmup=0,mode='exp',threshold=.45,
                no_gesture_language=False)
        with ExitStack() as stack:
            replacements=[('app.camera',cap),('app.GesturePipeline',pipeline),
                          ('app.GestureLanguage',language),('app.BrowserAction',browser),
                          ('app.cv2.FaceDetectorYN.create',detector),
                          ('app.cv2.FaceRecognizerSF.create',recognizer)]
            for target,value in replacements:
                stack.enter_context(patch(target,return_value=value))
            stack.enter_context(patch('app.base.get_models',return_value=('unused','unused')))
            stack.enter_context(patch('app.base.load_profile',return_value=None))
            stack.enter_context(patch('app.usable',return_value=True))
            stack.enter_context(patch('app.cv2.Laplacian',return_value=NS(var=lambda:100.)))
            for name in ('namedWindow','resizeWindow','imshow','destroyAllWindows'):
                stack.enter_context(patch('app.cv2.'+name))
            stack.enter_context(patch('app.cv2.getWindowProperty',return_value=1))
            stack.enter_context(patch('app.cv2.waitKey',side_effect=keys))
            app.run(args)
        return language,browser,pipeline

    def test_g_discards_completion_from_the_same_frame(self):
        for key in ('g','G'):
            with self.subTest(key=key):
                language,browser,pipeline=self.run_keys([ord(key),ord('q')])
                browser.submit.assert_not_called()
                self.assertFalse(pipeline.enabled)
                self.assertTrue(any(c.args[1]=='toggle' for c in language.reset.call_args_list))

    def test_enrollment_discards_pending_completion_and_blocks_following_frames(self):
        language,browser,_=self.run_keys([ord('e'),0,ord('q')],with_face=True)
        browser.submit.assert_not_called()
        self.assertTrue(language.update.call_args_list[1].kwargs['enrolling'])

    def test_normal_completion_dispatches_once_and_quit_does_not_dispatch(self):
        _,browser,_=self.run_keys([0,ord('q')])
        browser.submit.assert_called_once_with()


if __name__=='__main__':
    unittest.main(verbosity=2)
