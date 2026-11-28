"""Compare archived stable source against the experiment with identical pixels.

Real YuNet/SFace/MediaPipe and OpenCV GUI are used. Camera input is replaced
by a fixed public face/two-hand fixture; webbrowser.open is always mocked.
This measures processing throughput, not webcam FPS or recognition accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    sys.path.insert(0, str(args.project))
    import cv2
    import app
    source = cv2.imread(str(args.fixture))
    if source is None or source.shape[:2] != (960, 640):
        raise ValueError('Expected the original woman_hands.jpg (640x960).')
    frame = source[250:730, :640].copy()
    class ReplayCamera:
        def read(self):
            return True, frame.copy()
        def get(self, prop):
            return 0.
        def getBackendName(self):
            return 'FIXED_FIXTURE_UNPACED'
        def release(self):
            pass
    folder = args.project / 'results'
    folder.mkdir(exist_ok=True)
    before = set(folder.glob('*.json'))
    config = SimpleNamespace(size='640x480', camera=0, reference=args.profile,
                             distance=None, no_gestures=False, gesture_fps=20.,
                             seconds=args.seconds, warmup=args.warmup, mode='exp',
                             threshold=.45, no_gesture_language=False)
    with patch.object(app, 'camera', return_value=ReplayCamera()), patch('webbrowser.open', return_value=True) as browser:
        app.run(config)
    created = set(folder.glob('*.json'))-before
    if len(created) != 1:
        raise RuntimeError('Ambiguous measurement result.')
    row = json.loads(created.pop().read_text(encoding='utf-8'))
    if not row['complete'] or row['gesture_toggles'] or row['gestures']['error']:
        raise RuntimeError('Measurement was interrupted or modified.')
    if row['one_face_frames'] < .95*row['frames'] or row['measured_hand_frames']['two_hands'] < .95*row['frames']:
        raise RuntimeError('The fixture does not consistently exercise one face and both hands.')
    row['browser_mock_calls'] = browser.call_count
    row['source_sha256'] = {name: digest(args.project/name) for name in
                            ('app.py', 'gestures.py', 'gesture_language.py') if (args.project/name).exists()}
    args.output.write_text(json.dumps(row, indent=2), encoding='utf-8')


def fsm_timings():
    import numpy as np
    from gesture_language import GestureLanguage, open_chatgpt
    fsm, timings = GestureLanguage(), []
    labels = ('Open_Palm', 'Victory', 'Thumb_Up')
    with patch('gesture_language.webbrowser.open', return_value=True) as browser:
        for i in range(5000):
            phase, now = (i % 160)//12, i*.05
            label = labels[phase] if phase < 3 else None
            since = ((i//160)*160+phase*12)*.05 if label else None
            hand = SimpleNamespace(track_id=(1,1), gesture=label, observed_at=now,
                                   confirmed_since=since, association_ambiguous=False)
            started = time.perf_counter_ns()
            event = fsm.update((hand,), now)
            timings.append((time.perf_counter_ns()-started)/1000)
            if event:
                open_chatgpt()
    return dict(samples=len(timings), update_us_p50_p95=np.percentile(timings,[50,95]).tolist(),
                simulated_completions=fsm.stats().get('completions',0),
                mocked_browser_calls=browser.call_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT.parent/'backup/test-assets/woman_hands.jpg')
    parser.add_argument('--baseline-source', type=Path,
                        default=ROOT.parent/'backup/gesture-language-baseline-53e1b8c/source.zip')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--warmup', type=float, default=3.)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--project', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--profile', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.fixture = args.fixture.resolve()
    if args.worker:
        worker(args)
        return
    if args.seconds <= 0 or args.warmup < 0 or args.rounds < 1:
        parser.error('Invalid parameters.')
    if not args.fixture.is_file() or not args.baseline_source.is_file():
        parser.error('Missing fixture image or archived baseline source.')
    result_dir = ROOT/'results'
    result_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = (ROOT.parent/'backup').resolve()
    backup.mkdir(exist_ok=True)
    runs = []
    with tempfile.TemporaryDirectory(prefix='language-benchmark-', dir=backup) as temporary:
        temp = Path(temporary).resolve()
        if not temp.is_relative_to(backup):
            raise RuntimeError('Temporary directory is outside backup.')
        with zipfile.ZipFile(args.baseline_source) as archive:
            if any(not (temp/n).resolve().is_relative_to(temp) for n in archive.namelist()):
                raise RuntimeError('Invalid baseline source archive path.')
            archive.extractall(temp)
        baseline = temp/'FaceVision'
        (baseline/'models').mkdir(exist_ok=True)
        for model in (ROOT/'models').iterdir():
            if not model.is_file():
                continue
            try:
                os.link(model, baseline/'models'/model.name)
            except OSError:
                shutil.copyfile(model, baseline/'models'/model.name)
        for repeat in range(args.rounds):
            order = ('before','after') if repeat%2 == 0 else ('after','before')
            for version in order:
                project = baseline if version == 'before' else ROOT
                output = result_dir/f'language_fixture_{stamp}_{repeat+1}_{version}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--project', str(project), '--fixture', str(args.fixture),
                           '--output', str(output), '--seconds', str(args.seconds),
                           '--warmup', str(args.warmup)]
                profile = ROOT/'data/owner.npy'
                if profile.exists():
                    command += ['--profile', str(profile)]
                print(f'Identical fixture: {version}, run {repeat+1}/{args.rounds}', flush=True)
                subprocess.run(command, cwd=project, check=True)
                row = json.loads(output.read_text(encoding='utf-8'))
                row.update(version=version, result_file=output.name)
                runs.append(row)
    groups = {version: statistics.median(r['completed_fps'] for r in runs if r['version'] == version)
              for version in ('before','after')}
    delta = 100*(groups['after']/groups['before']-1)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), baseline_commit='53e1b8c',
                  workload='fixed public image replay, unpaced; real DNN and GUI; browser mocked',
                  fixture_source='https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg',
                  fixture_sha256=digest(args.fixture), crop_xywh=[0,250,640,480],
                  conditions=dict(warmup_s=args.warmup, seconds=args.seconds, rounds=args.rounds,
                                  size='640x480', gesture_fps=20., face_mode='exp'),
                  median_completed_fps=groups, fps_change_percent=delta,
                  degradation_under_5_percent=delta > -5., fsm_microbenchmark=fsm_timings(), runs=runs,
                  limits='This is processing throughput on a static public image, not webcam FPS. '
                         'It does not measure sensor/driver latency or sequence accuracy.')
    output = result_dir/f'gesture_language_ab_{stamp}.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('median_completed_fps','fps_change_percent',
                                         'degradation_under_5_percent','fsm_microbenchmark')}, indent=2))
    print('Comparison:', output)


if __name__ == '__main__':
    main()
