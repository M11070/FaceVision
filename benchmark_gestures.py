"""Compare actual camera/GUI loops, with a balanced disabled/enabled order."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def compare(runs):
    groups = {}
    for enabled in (False, True):
        rows = [r for r in runs if r['gestures']['enabled'] is enabled]
        groups['on' if enabled else 'off'] = dict(
            runs=len(rows),
            median_completed_fps=statistics.median(r['completed_fps'] for r in rows),
            median_full_cycle_p95_ms=statistics.median(r['p50_p95']['full_cycle_ms'][1] for r in rows),
            face_and_two_hands_frames=sum(r['measured_hand_frames']['face_and_two_hands'] for r in rows),
            owner_and_two_hands_frames=sum(r['measured_hand_frames']['owner_and_two_hands'] for r in rows),
        )
    groups['fps_change_percent'] = 100*(groups['on']['median_completed_fps']/groups['off']['median_completed_fps']-1)
    return groups


def main():
    parser = argparse.ArgumentParser(description='Live A/B benchmark with identical camera settings.')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--size', choices=('640x480', '1280x720', '1920x1080'), default='640x480')
    parser.add_argument('--gesture-fps', type=float, default=20)
    args = parser.parse_args()
    if args.rounds < 1 or args.seconds <= 0 or args.warmup < 0:
        parser.error('Invalid parameters.')
    folder = ROOT / 'results'
    folder.mkdir(exist_ok=True)
    runs = []
    for round_number in range(args.rounds):
        order = (False, True) if round_number % 2 == 0 else (True, False)
        for enabled in order:
            print(f"Run {round_number+1}/{args.rounds}: gestures {'ENABLED' if enabled else 'DISABLED'}. "
                  'Keep your face and both hands in view; change gestures. G invalidates the run.', flush=True)
            before = set(folder.glob('*.json'))
            command = [sys.executable, str(ROOT/'app.py'), '--size', args.size,
                       '--camera', str(args.camera), '--warmup', str(args.warmup),
                       '--seconds', str(args.seconds), '--gesture-fps', str(args.gesture_fps)]
            if not enabled:
                command.append('--no-gestures')
            subprocess.run(command, cwd=ROOT, check=True)
            created = set(folder.glob('*.json'))-before
            if len(created) != 1:
                raise RuntimeError('No unique result file was found for the run.')
            path = created.pop()
            row = json.loads(path.read_text(encoding='utf-8'))
            if not row.get('complete') or row['gesture_toggles'] or row['gestures']['error']:
                raise RuntimeError(f'Invalid run: {path.name}')
            if enabled and not row['gestures']['ready']:
                raise RuntimeError('Hand processing was not ready.')
            row['result_file'] = path.name
            runs.append(row)
            time.sleep(.3)  # release the existing camera between processes
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  conditions=vars(args), summary=compare(runs), runs=runs,
                  limits='Live scene may vary. Software timings exclude sensor, driver and monitor latency. '
                         'Hand/face counts measure availability, not classification accuracy.')
    path = folder / ('gestures_ab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print('Comparison:', path)


if __name__ == '__main__':
    main()
