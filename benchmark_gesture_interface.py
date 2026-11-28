"""Controlled v0.1/v0.2 replay: identical face/two-hand pixels, real models/UI."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import zipfile

from benchmark_gesture_language import digest

ROOT=Path(__file__).resolve().parent


def worker(args):
    sys.path.insert(0,str(args.project))
    import cv2
    import app
    source=cv2.imread(str(args.fixture))
    if source is None or source.shape[:2]!=(960,640):
        raise ValueError('Expected woman_hands.jpg at 640x960.')
    frame=source[250:730,:640].copy()
    class ReplayCamera:
        def read(self): return True,frame.copy()
        def get(self,prop): return 0.
        def getBackendName(self): return 'FIXED_FIXTURE_UNPACED'
        def release(self): pass
    config=NS(size='640x480',camera=0,reference=args.profile,distance=None,no_gestures=False,
              gesture_fps=20.,seconds=args.seconds,warmup=args.warmup,mode='exp',threshold=.45,
              no_gesture_language=False,gesture_hold_ms=300.,menu_timeout=10.,launch_cooldown=5.)
    folder=args.project/'results'
    folder.mkdir(exist_ok=True)
    before=set(folder.glob('*.json'))
    interface=None
    with patch.object(app,'camera',return_value=ReplayCamera()),patch('webbrowser.open',return_value=True) as browser:
        if args.version=='baseline':
            app.run(config)
        else:
            from launcher_ui import LauncherWindow,run_interface
            class BenchmarkWindow(LauncherWindow):
                def _desired_menu(self):
                    # Isolate rendering load; do not synthesize a command or launch.
                    return args.version=='visible',0
            interface=run_interface(config,engine=app.run,window_factory=BenchmarkWindow)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in set(folder.glob('*.json'))-before]
    rows=[r for r in rows if 'completed_fps' in r]
    if len(rows)!=1:
        raise RuntimeError('Missing or ambiguous processing result.')
    row=rows[0]
    if not row['complete'] or row['gesture_toggles'] or row['gestures']['error']:
        raise RuntimeError('Measurement was interrupted or modified.')
    if row['one_face_frames']<row['frames']*.95 or row['measured_hand_frames']['two_hands']<row['frames']*.95:
        raise RuntimeError('The fixture face and hands were not consistently detected.')
    if interface and (interface['callback_error'] or not interface['camera_worker_stopped']):
        raise RuntimeError('GUI or camera worker did not finish correctly.')
    names=('app.py','gestures.py','gesture_language.py','gesture_interface.py','launcher_ui.py')
    row.update(interface=interface,browser_mock_calls=browser.call_count,
               source_sha256={name:digest(args.project/name) for name in names if (args.project/name).is_file()})
    args.output.write_text(json.dumps(row,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=ROOT.parent/'backup/test-assets/woman_hands.jpg')
    parser.add_argument('--baseline-source',type=Path,
                        default=ROOT.parent/'backup/gesture-interface-baseline-1b3bd81/source.zip')
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--warmup',type=float,default=3.)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--project',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--profile',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--version',choices=('baseline','hidden','visible'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.fixture=args.fixture.resolve()
    if args.worker:
        worker(args)
        return
    if args.seconds<=0 or args.warmup<0 or args.rounds<1 or not args.fixture.is_file() or not args.baseline_source.is_file():
        parser.error('Check the duration, public fixture and baseline source archive.')
    folder=ROOT/'results'
    folder.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=(ROOT.parent/'backup').resolve()
    runs=[]
    with tempfile.TemporaryDirectory(prefix='interface-benchmark-',dir=backup) as temporary:
        temp=Path(temporary).resolve()
        if not temp.is_relative_to(backup):
            raise RuntimeError('Temporary directory is outside backup.')
        with zipfile.ZipFile(args.baseline_source) as archive:
            if any(not (temp/n).resolve().is_relative_to(temp) for n in archive.namelist()):
                raise RuntimeError('Invalid archive path.')
            archive.extractall(temp)
        baseline=temp/'FaceVision'
        (baseline/'models').mkdir(exist_ok=True)
        for model in (ROOT/'models').iterdir():
            if model.is_file():
                try: os.link(model,baseline/'models'/model.name)
                except OSError: shutil.copyfile(model,baseline/'models'/model.name)
        for repeat in range(args.rounds):
            order=('baseline','hidden','visible') if repeat%2==0 else ('visible','hidden','baseline')
            for version in order:
                project=baseline if version=='baseline' else ROOT
                output=folder/f'interface_fixture_{stamp}_{repeat+1}_{version}.json'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--version',version,
                         '--project',str(project),'--fixture',str(args.fixture),'--output',str(output),
                         '--seconds',str(args.seconds),'--warmup',str(args.warmup)]
                if (ROOT/'data/owner.npy').exists():
                    command+=['--profile',str(ROOT/'data/owner.npy')]
                print(f'Identical frame: {version}, run {repeat+1}/{args.rounds}',flush=True)
                subprocess.run(command,cwd=project,check=True)
                row=json.loads(output.read_text(encoding='utf-8'))
                row.update(version=version,result_file=output.name)
                runs.append(row)
    medians={version:statistics.median(r['completed_fps'] for r in runs if r['version']==version)
             for version in ('baseline','hidden','visible')}
    changes={version:100*(medians[version]/medians['baseline']-1) for version in ('hidden','visible')}
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline_commit='1b3bd81',
                workload='identical public static frame, unpaced; real models and GUIs; browser mocked',
                fixture_source='https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg',
                fixture_sha256=digest(args.fixture),crop_xywh=[0,250,640,480],
                conditions=dict(seconds=args.seconds,warmup=args.warmup,rounds=args.rounds,
                                size='640x480',gesture_fps=20.),
                median_processing_fps=medians,change_percent=changes,
                degradation_under_5_percent=all(delta > -5. for delta in changes.values()),runs=runs,
                limits='Static fixture throughput, not physical webcam FPS or gesture accuracy. '
                       'Visible benchmark forces the menu renderer open without synthesizing gestures.')
    output=folder/f'gesture_interface_ab_{stamp}.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('median_processing_fps','change_percent','degradation_under_5_percent')},indent=2))
    print('Comparison:',output)


if __name__=='__main__':
    main()
