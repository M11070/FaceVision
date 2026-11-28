"""Tk runs only on the main thread; the existing camera engine runs once."""
from collections import Counter, deque
from dataclasses import dataclass
import json
from math import cos, sin, pi
from pathlib import Path
from queue import Queue, Full
from threading import Event, Lock, Thread, get_ident
import time
import tkinter as tk
import uuid

from gesture_interface import APPLICATIONS, LauncherAction, MenuController

BG, CARD, BORDER, CYAN, TEXT, MUTED = '#0b121b', '#111e2c', '#293b4d', '#22d3ee', '#eef5fb', '#99adbf'


@dataclass(frozen=True)
class FrameState:
    hands: tuple
    captured_at: float
    produced_at: float
    gestures_enabled: bool
    enrolling: bool
    faces: int
    owner: bool


class CameraMailbox:
    """One latest immutable result, no image arrays or unbounded queues."""
    def __init__(self):
        self._lock = Lock()
        self._latest, self._error, self._done = None, None, False
        self._counts = Counter()

    def publish(self, **values):
        frame = FrameState(**values)
        with self._lock:
            self._latest = frame
            self._counts['frames'] += 1
            self._counts['face_frames'] += int(frame.faces > 0)
            self._counts['owner_frames'] += int(frame.owner)
            self._counts['two_hand_frames'] += int(len(frame.hands) == 2)
            self._counts['owner_and_two_hands'] += int(frame.owner and len(frame.hands) == 2)

    def finish(self, error=None):
        with self._lock:
            self._done, self._error = True, error

    def read(self):
        with self._lock:
            return self._latest, self._error, self._done

    def stats(self):
        with self._lock:
            return dict(self._counts) | dict(error=self._error, done=self._done)


def draw_icon(canvas, index, x, y, color):
    """Small bundled vector symbols; no fonts, downloads or image dependencies."""
    if index == 0:
        for i in range(6):
            angle = i*pi/3
            pts = []
            for px, py in ((-5,-20),(10,-20),(18,-7),(10,7),(-3,7),(-10,-5)):
                pts += [x+px*cos(angle)-py*sin(angle), y+px*sin(angle)+py*cos(angle)]
            canvas.create_line(*pts, fill=color, width=2, smooth=True)
    elif index == 1:
        canvas.create_oval(x-23,y-23,x+23,y+23,outline=color,width=2)
        canvas.create_oval(x-10,y-23,x+10,y+23,outline=color,width=2)
        for dy in (-9,9):
            canvas.create_line(x-21,y+dy,x+21,y+dy,fill=color,width=2)
    elif index == 2:
        canvas.create_polygon(x-27,y-18,x+27,y-18,x+27,y+18,x-27,y+18,
                              fill=color,smooth=True,splinesteps=20)
        canvas.create_polygon(x-6,y-10,x-6,y+10,x+11,y,fill=CARD)
    else:
        canvas.create_polygon(x-20,y-8,x-19,y-24,x-6,y-18,x+6,y-18,x+19,y-24,x+20,y-8,
                              fill=color)
        canvas.create_oval(x-23,y-19,x+23,y+16,fill=color,outline=color)
        canvas.create_oval(x-13,y-5,x+13,y+9,fill=CARD,outline=CARD)
        canvas.create_line(x,y+12,x,y+25,fill=color,width=7)
        canvas.create_line(x-3,y+20,x-19,y+20,x-25,y+12,fill=color,width=4,smooth=True)


class LauncherWindow:
    def __init__(self, args, mailbox, stop_event, commands, *, action=None):
        self._ui_thread = get_ident()
        self.args, self.mailbox, self.stop_event, self.commands = args, mailbox, stop_event, commands
        self.controller = MenuController(hold_seconds=getattr(args,'gesture_hold_ms',300.)/1000,
                                         inactivity=getattr(args,'menu_timeout',10.),
                                         cooldown=getattr(args,'launch_cooldown',5.))
        self.action = action if action is not None else LauncherAction()
        self._closed, self._callback_error = False, None
        self._shutdown_reason = None
        self._rendered_state, self._indicator_state = None, None
        self._ticks, self._last_tick = deque(maxlen=5000), None
        self.root = tk.Tk()
        self.root.title('FACEVISION')
        self.root.configure(bg=BG)
        self.root.resizable(False,False)
        self.root.attributes('-topmost',True)
        try:
            self.root.attributes('-toolwindow',True)
        except tk.TclError:
            pass
        self.root.geometry(f'320x60+{max(0,self.root.winfo_screenwidth()-344)}+{max(0,self.root.winfo_screenheight()-148)}')
        self.root.protocol('WM_DELETE_WINDOW',self.shutdown)
        self.root.report_callback_exception = self._callback_exception
        self.led = tk.Canvas(self.root,width=14,height=14,bg=BG,highlightthickness=0)
        self.led.place(x=12,y=24)
        self._led_id = self.led.create_oval(2,2,12,12,fill=MUTED,outline='')
        tk.Label(self.root,text='FACEVISION',font=('Segoe UI',10,'bold'),fg=TEXT,bg=BG).place(x=35,y=6)
        self.indicator = tk.Label(self.root,text='Starting camera...',font=('Segoe UI',9),fg=MUTED,bg=BG)
        self.indicator.place(x=35,y=29)
        tk.Button(self.root,text='Exit',command=self.shutdown,bg=CARD,fg=TEXT,activebackground=BORDER,
                  activeforeground=TEXT,relief='flat',font=('Segoe UI',9),padx=10,pady=5).place(x=252,y=16)
        self.overlay = tk.Toplevel(self.root)
        self.overlay.withdraw()
        self.overlay.title('FACEVISION — applications')
        self.overlay.configure(bg=BG)
        self.overlay.overrideredirect(True)
        self.overlay.attributes('-topmost',True)
        self.overlay.protocol('WM_DELETE_WINDOW',self.cancel_menu)
        self.overlay.geometry(f'560x282+{max(0,(self.root.winfo_screenwidth()-560)//2)}+{max(0,(self.root.winfo_screenheight()-282)//2)}')
        self.canvas = tk.Canvas(self.overlay,width=560,height=282,bg=BG,highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack()
        self.canvas.create_text(22,24,text='FACEVISION',anchor='w',fill=CYAN,font=('Segoe UI',13,'bold'))
        self.canvas.create_text(22,49,text='Select an application',anchor='w',fill=MUTED,font=('Segoe UI',10))
        tk.Button(self.overlay,text='×',command=self.cancel_menu,bg=BG,fg=MUTED,activebackground=CARD,
                  activeforeground=TEXT,relief='flat',font=('Segoe UI',16),padx=6,pady=0).place(x=523,y=9)
        self._cards = []
        for i, application in enumerate(APPLICATIONS):
            x = 20+i*132
            card = self.canvas.create_rectangle(x,77,x+124,190,fill=CARD,outline=BORDER,width=2)
            self._cards.append(card)
            draw_icon(self.canvas,i,x+62,119,TEXT)
            self.canvas.create_text(x+62,168,text=application.name,fill=TEXT,font=('Segoe UI',11,'bold'))
        self.canvas.create_text(280,216,text='Open palm: menu     Victory: next',fill=TEXT,font=('Segoe UI',10))
        self.canvas.create_text(280,237,text='Thumbs up: open     Thumbs down: cancel',fill=TEXT,font=('Segoe UI',10))
        self._hint_text = self.canvas.create_text(280,262,text='Ready for the next gesture.',fill=MUTED,font=('Segoe UI',9))
        self._last_hint = 'Ready for the next gesture.'
        self.root.bind_all('<Control-q>',lambda event:self.shutdown())
        self.root.bind_all('<Escape>',lambda event:self.cancel_menu())
        self.root.bind_all('<g>',lambda event:self.toggle_gestures())
        self.root.bind_all('<G>',lambda event:self.toggle_gestures())
        self._after_id = self.root.after(20,self._tick)

    def _assert_ui(self):
        if get_ident() != self._ui_thread:
            raise RuntimeError('Tkinter is restricted to the main GUI thread.')

    def _callback_exception(self, kind, value, traceback):
        self._callback_error = f'{kind.__name__}: {value}'
        self.shutdown('gui_error')

    def _desired_menu(self):
        return self.controller.visible, self.controller.selected

    def render_menu(self, visible, selected):
        self._assert_ui()
        state = visible, selected
        if state == self._rendered_state:
            return
        for i, card in enumerate(self._cards):
            self.canvas.itemconfigure(card,outline=CYAN if i == selected else BORDER)
        if visible:
            if not self._rendered_state or not self._rendered_state[0]:
                self.overlay.deiconify()
                self.overlay.lift()
                self.overlay.focus_force()
        else:
            self.overlay.withdraw()
        self._rendered_state = state

    def _tick(self):
        self._assert_ui()
        if self._closed:
            return
        frame, error, done = self.mailbox.read()
        now = time.perf_counter()
        if self._last_tick is not None:
            self._ticks.append((now-self._last_tick)*1000)
        self._last_tick = now
        active = frame is not None and 0 <= now-frame.produced_at <= 1. and not done
        fresh = frame is not None and 0 <= now-frame.produced_at <= .25
        selected = self.controller.update(frame.hands if fresh else (),now,
                                           enabled=bool(frame and frame.gestures_enabled and not done),
                                           enrolling=bool(frame and frame.enrolling),action_pending=self.action.busy)
        self.render_menu(*self._desired_menu())
        hint = self.controller.hint()
        if hint != self._last_hint:
            self.canvas.itemconfigure(self._hint_text,text=hint)
            self._last_hint = hint
        if selected is not None and not self.stop_event.is_set():
            self.action.submit(selected)
        if error:
            indicator = ('Camera error — see console','#fb7185')
        elif done:
            indicator = ('Processing stopped',MUTED)
        elif active:
            indicator = ('Camera active' if frame.gestures_enabled else 'Camera active · G: disabled',CYAN)
        else:
            indicator = ('Starting / waiting for camera',MUTED)
        if indicator != self._indicator_state:
            self.indicator.configure(text=indicator[0])
            self.led.itemconfigure(self._led_id,fill=indicator[1])
            self._indicator_state = indicator
        if done and getattr(self.args,'seconds',0):
            self.shutdown('complete')
            return
        self._after_id = self.root.after(20,self._tick)

    def cancel_menu(self):
        self._assert_ui()
        if not self._closed:
            self.controller.close(time.perf_counter(),'manual_cancel')
            self.render_menu(False,0)

    def toggle_gestures(self):
        self._assert_ui()
        self.cancel_menu()
        try:
            self.commands.put_nowait(ord('g'))
        except Full:
            pass

    def shutdown(self, reason='manual'):
        self._assert_ui()
        if not self._closed:
            self._closed = True
            self._shutdown_reason = reason
            self.stop_event.set()
            self.controller.close(time.perf_counter(),'exit')
            self.root.after_cancel(self._after_id)
            self.root.destroy()

    def stats(self):
        values = sorted(self._ticks)
        return dict(menu=self.controller.stats(),actions=self.action.stats(),
                    camera=self.mailbox.stats(),callback_error=self._callback_error,shutdown_reason=self._shutdown_reason,
                    poll_ms_p50_p95=([values[len(values)//2],values[min(len(values)-1,int(len(values)*.95))]]
                                    if values else None))


def run_interface(args, *, engine=None, window_factory=LauncherWindow):
    if engine is None:
        from app import run
        engine = run
    mailbox, stop_event, commands = CameraMailbox(), Event(), Queue(maxsize=8)
    window = window_factory(args,mailbox,stop_event,commands)
    def process_camera():
        error = None
        try:
            engine(args,frame_observer=mailbox.publish,stop_event=stop_event,commands=commands)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            print('CAMERA:',error,flush=True)
        finally:
            mailbox.finish(error)
    worker = Thread(target=process_camera,name='FaceVision-camera',daemon=True)
    worker.start()
    try:
        window.root.mainloop()
    finally:
        if not window._closed:
            window.shutdown('cleanup')
        stop_event.set()
        worker.join(timeout=2.)
    result = window.stats() | dict(camera_worker_stopped=not worker.is_alive(),legacy_sequence_enabled=False)
    if getattr(args,'seconds',0):
        folder = Path(__file__).resolve().parent/'results'
        folder.mkdir(exist_ok=True)
        path = folder/('gesture_interface_live_'+uuid.uuid4().hex+'.json')
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Interface:',path,flush=True)
    return result
