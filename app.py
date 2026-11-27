import argparse
import platform
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import face_vision as base
from metrics import Measurement
from tracking import Tracker, Enrollment, decide
from gestures import GesturePipeline, draw_hands

WINDOW = 'FACE VISION | EXP'
SIZES = ('640x480', '1280x720', '1920x1080')


def usable(frame, face):
    x, y, w, h = face[:4]
    rows, cols = frame.shape[:2]
    if min(w,h) < 80 or min(x,y) < 0 or x+w > cols or y+h > rows:
        return False
    points = face[4:14].reshape(5,2)
    if (not np.isfinite(points).all() or np.linalg.norm(points[0]-points[1]) < .18*w
            or (points[:,0]<x).any() or (points[:,0]>x+w).any()
            or (points[:,1]<y).any() or (points[:,1]>y+h).any()):
        return False
    return True


def camera(index, size):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            ok, frame = cap.read()
            if ok and frame.shape[1::-1] == size:
                return cap
        cap.release()
    raise RuntimeError(f'Kamera {index} nije isporucila {size}.')


def run(args):
    size = tuple(map(int, args.size.split('x')))
    detector_path, recognizer_path = base.get_models()
    detector = cv2.FaceDetectorYN.create(detector_path, '', size, .85, .3, 100)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_path, '')
    reference = (base.normalize(np.load(args.reference, allow_pickle=False))
                 if args.reference else base.load_profile())
    if args.distance and reference is None:
        raise ValueError('Za test dometa prvo registruj vlasnika tasterom E.')
    cap = camera(args.camera, size)
    gestures = GesturePipeline(enabled=not args.no_gestures, max_fps=args.gesture_fps)
    measure = Measurement(args.warmup, args.seconds) if args.seconds else None
    hand_frames = dict(one_hand=0, two_hands=0, face_and_two_hands=0, owner_and_two_hands=0)
    gesture_toggles = 0
    tracker, enroll = Tracker(), None
    ticks = deque()
    previous = None
    message = ''
    meta = dict(mode=args.mode, requested=args.size, actual=size, distance_m=args.distance,
                profile_loaded=reference is not None, threshold=args.threshold,
                python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                backend=cap.getBackendName(), negotiated_fps=cap.get(cv2.CAP_PROP_FPS),
                fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)), threads=cv2.getNumThreads(),
                camera=args.camera, warmup_s=args.warmup, requested_s=args.seconds)
    def result_meta(complete):
        return meta | dict(complete=complete, gestures=gestures.stats(),
                           gesture_toggles=gesture_toggles, measured_hand_frames=hand_frames)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW,800,650)
        while True:
            start = time.perf_counter()
            ok, frame = cap.read()
            read = time.perf_counter()
            if not ok or frame is None:
                raise RuntimeError('Prekinuto citanje kamere.')
            if frame.shape[1::-1] != size:
                raise RuntimeError('Kamera je promenila rezoluciju tokom merenja.')
            gestures.submit(frame, read)
            _, output = detector.detect(frame)
            faces = [] if output is None else output
            detected = time.perf_counter()
            boxes = [tuple(map(float,f[:4])) for f in faces]
            exp = args.mode == 'exp'
            tracks = tracker.update(boxes, read) if exp else []
            sizes = [min(b[2:]) for b in boxes]
            features = []
            eligible = []
            for i, face in enumerate(faces):
                good = (usable(frame,face) and not tracks[i].ambiguous) if exp else sizes[i]>=80
                eligible.append(good)
                feature = None
                if good and (reference is not None or enroll is not None):
                    aligned = recognizer.alignCrop(frame,face)
                    if not exp or (10 < float(aligned.mean()) < 245 and
                                   cv2.Laplacian(cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY),cv2.CV_32F).var() >= 12):
                        feature = base.normalize(recognizer.feature(aligned))
                features.append(feature)
            scores = [base.score(v,reference) if v is not None and reference is not None
                      else None for v in features]
            if exp:
                decide(tracks, scores, args.threshold, read)
            owners = ([tr.confirmed for tr in tracks] if exp else
                      [v is not None and v>=args.threshold for v in scores])
            if enroll is not None:
                state = enroll.observe(tracks, features, read)
                if state == 'cancel':
                    enroll = None
                    message = 'Enrollment cancelled'
                elif state == 'save':
                    reference = base.save_profile(enroll.samples)
                    enroll = None
                    message = 'Profile saved'
                    for tr in tracker.tracks.values():
                        tr.clear()
                owners = [False]*len(faces)
            identified = time.perf_counter()
            canvas = cv2.copyMakeBorder(frame,42,78,0,0,cv2.BORDER_CONSTANT,value=(20,25,32))
            def draw(box,label,color):
                x,y,w,h = map(int,box)
                cv2.rectangle(canvas,(x,y+42),(x+w,y+h+42),color,2)
                base.text(canvas,label,(max(0,x),max(58,y+38)),color)
            for i,box in enumerate(boxes):
                label = f'ID {tracks[i].number}' if exp else 'FACE'
                if owners[i]:
                    label += ' OWNER'
                if enroll is not None:
                    label += f' {len(enroll.samples)}/12'
                draw(box,label,(70,220,110) if owners[i] else (0,190,240))
            for tr in tracker.tracks.values():
                if not tr.visible:
                    draw(tr.predict(read),f'ID {tr.number}',(145,145,145))
            hands = gestures.snapshot()
            draw_hands(canvas, hands, size)
            if measure and start >= measure.opened + measure.warmup:
                hand_frames['one_hand'] += int(len(hands) == 1)
                hand_frames['two_hands'] += int(len(hands) == 2)
                hand_frames['face_and_two_hands'] += int(len(faces) > 0 and len(hands) == 2)
                hand_frames['owner_and_two_hands'] += int(any(owners) and len(hands) == 2)
            h = frame.shape[0]
            fps = (len(ticks)-1)/(ticks[-1]-ticks[0]) if len(ticks)>1 else 0
            base.text(canvas,'FACE VISION / EXP  |  LOCAL',(12,26))
            status = message or ('Profile ready' if reference is not None else 'E: enroll your face')
            base.text(canvas,status,(12,h+65))
            base.text(canvas,f'Faces: {len(faces)}   FPS: {fps:.1f}   Threshold: {args.threshold:.2f}',(12,h+86))
            base.text(canvas,'Q: stop test' if measure else 'E: enroll  X: erase profile  Q: quit  ESC: cancel',(12,h+109))
            same = bool(measure and previous is not None and np.array_equal(frame,previous))
            previous = frame if measure else None
            cv2.imshow(WINDOW,canvas)
            key = cv2.waitKey(1) & 255
            end = time.perf_counter()
            ticks.append(end)
            while len(ticks)>2 and ticks[1]<end-1:
                ticks.popleft()
            if measure and measure.add(start,read,detected,identified,end,sizes,any(owners),same):
                measure.write(result_meta(True))
                return
            if key==ord('q') or (key==27 and enroll is None) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                if measure:
                    measure.write(result_meta(False))
                    raise SystemExit(2)
                return
            if key in (ord('g'), ord('G')):
                gestures.set_enabled(not gestures.enabled)
                gesture_toggles += 1
                print('Gestovi:', 'ukljuceni' if gestures.enabled else 'iskljuceni', flush=True)
            if measure:
                continue
            if key==ord('e'):
                if len(tracks)==1 and eligible[0]:
                    enroll = Enrollment(tracks[0].number)
                    message = 'Enrollment'
                else:
                    enroll = None
                    message = 'One clear face needed'
            elif key==27:
                enroll = None
                message = 'Enrollment cancelled'
            elif key==ord('x'):
                base.PROFILE.unlink(missing_ok=True)
                reference, enroll, message = None, None, 'Profile erased'
                for tr in tracker.tracks.values():
                    tr.clear()
    finally:
        gestures.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=SIZES, default=SIZES[0])
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--mode', choices=['base','exp'], default='exp')
    p.add_argument('--threshold', type=float, default=.45)
    p.add_argument('--warmup', type=float, default=3)
    p.add_argument('--seconds', type=float, default=0)
    p.add_argument('--distance', type=float, choices=(.5,1,2,3))
    p.add_argument('--reference', type=Path, help='Postojeci profil samo za citanje u benchmarku.')
    p.add_argument('--no-gestures', action='store_true', help='Pokretanje bez obrade ruku; G je ukljucuje.')
    p.add_argument('--gesture-fps', type=float, default=20., help='Najvise obrada ruku u sekundi (podrazumevano 20).')
    args = p.parse_args()
    if not 0<args.threshold<1 or args.seconds<0 or args.warmup<0 or not 1<=args.gesture_fps<=30:
        p.error('Neispravni parametri.')
    if (args.reference or args.mode=='base' or args.distance) and not args.seconds:
        p.error('Ova opcija zahteva vremenski ogranicen benchmark.')
    run(args)
