import argparse
import hashlib
import os
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
PROFILE = ROOT / 'data' / 'owner.npy'
MODELS_INFO = [
    ('face_detection_yunet', 'face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    ('face_recognition_sface', 'face_recognition_sface_2021dec.onnx', '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
]
SAMPLES = 12
WINDOW = 'FACE VISION | v0.1'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def get_models():
    MODELS.mkdir(exist_ok=True)
    paths = []
    for repo, filename, sha in MODELS_INFO:
        path = MODELS / filename
        if not path.exists() or digest(path) != sha:
            url = f'https://huggingface.co/opencv/{repo}/resolve/main/{filename}'
            temp = path.with_suffix('.part')
            print(f'Preuzimanje modela: {filename}', flush=True)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'FACE-VISION/0.1'})
                with urllib.request.urlopen(req, timeout=90) as src, temp.open('wb') as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                if digest(temp) != sha:
                    raise RuntimeError('SHA256 modela se ne podudara.')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        paths.append(str(path))
    return paths


def normalize(feature):
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.size != 128 or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Neispravan vektor lica.')
    return vector / norm


def score(feature, reference):
    return float(np.clip(np.dot(feature, reference), -1.0, 1.0))


def load_profile():
    if not PROFILE.exists():
        return None
    return normalize(np.load(PROFILE, allow_pickle=False))


def save_profile(samples):
    reference = normalize(np.mean(samples, axis=0))
    PROFILE.parent.mkdir(exist_ok=True)
    temp = PROFILE.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.save(f, reference, allow_pickle=False)
    temp.replace(PROFILE)
    return reference


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    raise RuntimeError('Kamera nije dostupna. Zatvori aplikacije koje je koriste ili probaj --camera 1.')


def text(frame, value, pos, color=(235, 235, 235), size=0.55):
    cv2.putText(frame, value, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Lokalna demonstracija prepoznavanja sopstvenog lica.')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.45)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('--threshold mora biti izmedju 0 i 1')
    if args.self_test:
        a = normalize(np.eye(128)[0]); b = normalize(np.eye(128)[1])
        assert abs(score(a, a) - 1) < 1e-6
        assert abs(score(a, b)) < 1e-6
        print('SELF-TEST OK: normalizacija i kosinusna slicnost.')
        return
    detector_path, recognizer_path = get_models()
    detector = cv2.FaceDetectorYN.create(detector_path, '', (640, 480), 0.85, 0.3, 100)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_path, '')
    reference = load_profile()
    cap = open_camera(args.camera)
    samples = None
    last_sample = 0.0
    fps = 0.0
    failure_count = 0
    print('E = registracija, X = brisanje profila, Q = izlaz. Esc = otkazivanje registracije.', flush=True)
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 800, 650)
        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                failure_count += 1
                if failure_count >= 10:
                    raise RuntimeError('Kamera ne salje sliku.')
                time.sleep(0.05)
                continue
            failure_count = 0
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            faces = [] if faces is None else faces
            display = frame.copy()
            status = 'E: enroll your face' if reference is None else 'Profile ready'
            for face in faces:
                x, y, fw, fh = map(int, face[:4])
                good_size = min(fw, fh) >= 80
                feature = None
                if good_size and (samples is not None or reference is not None):
                    aligned = recognizer.alignCrop(frame, face)
                    feature = normalize(recognizer.feature(aligned))
                color = (0, 190, 240)
                label = 'FACE DETECTED' if good_size else 'MOVE CLOSER'
                if samples is not None:
                    if len(faces) == 1 and feature is not None:
                        now = time.monotonic()
                        if now - last_sample >= 0.45:
                            samples.append(feature)
                            last_sample = now
                        label = f'ENROLLING {len(samples)}/{SAMPLES}'
                        if len(samples) >= SAMPLES:
                            reference = save_profile(samples)
                            samples = None
                            label = 'PROFILE SAVED'
                            color = (70, 220, 110)
                    else:
                        label = 'ONE FACE ONLY' if len(faces) != 1 else 'MOVE CLOSER'
                elif reference is not None and feature is not None:
                    similarity = score(feature, reference)
                    match = similarity >= args.threshold
                    label = f'{"OWNER MATCH" if match else "UNKNOWN"}  {similarity:.2f}'
                    color = (70, 220, 110) if match else (80, 90, 245)
                cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 2)
                ty = max(20, y - 7)
                cv2.rectangle(display, (max(0, x), max(0, ty-19)), (min(w-1, max(0,x)+int(len(label)*10+10)), min(h-1,ty+4)), (25,25,25), -1)
                text(display, label, (max(2,x+3), ty), color)
            if samples is not None:
                status = f'Enrollment {len(samples)}/{SAMPLES} | Turn head slightly' if len(faces) == 1 else 'Enrollment: keep exactly one face visible'
            elif len(faces) == 0:
                status = 'No face detected | ' + status
            elapsed = time.perf_counter() - started
            fps = 1/elapsed if not fps else .9*fps + .1/elapsed
            canvas = cv2.copyMakeBorder(display, 42, 78, 0, 0, cv2.BORDER_CONSTANT, value=(20,25,32))
            text(canvas, 'FACE VISION / 0.1  |  LOCAL', (12, 26), (120,230,170), .62)
            text(canvas, f'{status}', (12, h+65), size=.48)
            text(canvas, f'Faces: {len(faces)}   Processing FPS: {fps:.1f}   Threshold: {args.threshold:.2f}', (12,h+86), size=.48)
            text(canvas, 'E: enroll  X: erase profile  Q: quit  ESC: cancel', (12,h+109), size=.46)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or (key == 27 and samples is None):
                break
            if key == ord('e'):
                samples = []; last_sample = 0.0
            if key == 27:
                samples = None
            if key == ord('x'):
                PROFILE.unlink(missing_ok=True)
                reference = None; samples = None
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'GRESKA: {exc}', flush=True)
        raise SystemExit(1)
