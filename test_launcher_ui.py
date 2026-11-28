from queue import Queue
from threading import Event, Thread
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from gesture_interface import APPLICATIONS
from launcher_ui import CameraMailbox, LauncherWindow, run_interface, CYAN, BORDER


def arguments():
    return NS(gesture_hold_ms=300.,menu_timeout=10.,launch_cooldown=5.,seconds=0)


class TkWindowChecks(unittest.TestCase):
    def setUp(self):
        self.window=LauncherWindow(arguments(),CameraMailbox(),Event(),Queue(),action=Mock(busy=False))
        self.window.root.update_idletasks()

    def tearDown(self):
        if not self.window._closed:
            self.window.shutdown()

    def test_menu_hidden_initially_and_exact_icon_labels(self):
        self.assertEqual(self.window.overlay.state(),'withdrawn')
        names=[self.window.canvas.itemcget(i,'text') for i in self.window.canvas.find_all()
               if self.window.canvas.type(i)=='text']
        self.assertEqual([n for n in names if n in {a.name for a in APPLICATIONS}],
                         [a.name for a in APPLICATIONS])

    def test_selected_card_is_cyan_and_cancel_withdraws_menu(self):
        self.window.render_menu(True,2)
        self.window.root.update_idletasks()
        self.assertEqual([self.window.canvas.itemcget(i,'outline') for i in self.window._cards],
                         [BORDER,BORDER,CYAN,BORDER])
        self.assertEqual(self.window.overlay.state(),'normal')
        self.window.cancel_menu()
        self.assertEqual(self.window.overlay.state(),'withdrawn')
        self.assertFalse(self.window.stop_event.is_set())

    def test_native_close_has_manual_exit_handler(self):
        command=self.window.root.protocol('WM_DELETE_WINDOW')
        self.window.root.tk.call(command)
        self.assertTrue(self.window._closed)
        self.assertTrue(self.window.stop_event.is_set())
        self.assertEqual(self.window.stats()['shutdown_reason'],'manual')

    def test_cross_thread_tk_access_is_rejected_before_tk_call(self):
        errors=[]
        def wrong_thread():
            try:
                self.window.render_menu(True,0)
            except RuntimeError as error:
                errors.append(str(error))
        worker=Thread(target=wrong_thread)
        worker.start(); worker.join()
        self.assertEqual(len(errors),1)
        self.assertEqual(self.window.overlay.state(),'withdrawn')

    def test_g_cancels_menu_and_queues_only_one_camera_command(self):
        self.window.render_menu(True,0)
        self.window.toggle_gestures()
        self.assertEqual(self.window.overlay.state(),'withdrawn')
        self.assertEqual(self.window.commands.get_nowait(),ord('g'))


class RuntimeChecks(unittest.TestCase):
    def test_gui_remains_responsive_during_slow_camera_and_exit_stops_worker(self):
        observed=[]
        class TimedWindow(LauncherWindow):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.root.after(130,self.check_and_exit)
            def check_and_exit(self):
                observed.append(self.indicator.cget('text'))
                self.shutdown()
        def slow_camera(args,*,frame_observer,stop_event,commands):
            now=time.perf_counter()
            frame_observer(hands=(),captured_at=now,produced_at=now,gestures_enabled=True,
                           enrolling=False,faces=1,owner=True)
            stop_event.wait(2.)
        with patch('gesture_interface.webbrowser.open') as browser:
            started=time.perf_counter()
            result=run_interface(arguments(),engine=slow_camera,window_factory=TimedWindow)
        browser.assert_not_called()
        self.assertLess(time.perf_counter()-started,1.5)
        self.assertEqual(observed,['Camera active'])
        self.assertTrue(result['camera_worker_stopped'])
        self.assertEqual(result['shutdown_reason'],'manual')
        self.assertIsNone(result['callback_error'])
        self.assertFalse(result['legacy_sequence_enabled'])

    def test_camera_error_is_visible_and_exit_remains_available(self):
        indicators=[]
        class TimedWindow(LauncherWindow):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.root.after(100,self.capture)
            def capture(self):
                indicators.append(self.indicator.cget('text'))
                self.shutdown()
        def failed_camera(*args,**kwargs):
            raise RuntimeError('test camera')
        result=run_interface(arguments(),engine=failed_camera,window_factory=TimedWindow)
        self.assertTrue(indicators[0].startswith('Camera error'))
        self.assertIn('test camera',result['camera']['error'])
        self.assertTrue(result['camera_worker_stopped'])


if __name__=='__main__':
    unittest.main(verbosity=2)
