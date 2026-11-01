#!/usr/bin/env python3
"""Alternate affected Flash-Next histories; retain main controls and all samples."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
OUT = ROOT / 'artifacts/model-extensions/card03'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', action='append', choices=('off', 'mtp'), required=True)
args = parser.parse_args()
invocations = json.loads((OUT / 'functional-invocations.json').read_text())
controls = []
sys.path.insert(0, str(ROOT / 'tests/functional'))
from run import compare_runs
from metrics import qualify
for mode in args.mode:
    for arm in ('baseline', 'candidate'):
        initial = next(row for row in invocations if row['label'] == f'{arm}-{mode}-idle')
        command = list(initial['command'])
        label = f'{arm}-{mode}-repeat1'
        output = OUT / label
        command[command.index('--output') + 1] = str(output)
        if arm == 'candidate':
            command[command.index('--baseline') + 1] = str(OUT / f'baseline-{mode}-repeat1')
        row = {'label': label, 'command': command, 'status': 'running'}
        controls.append(row)
        (OUT / 'control-invocations.json').write_text(json.dumps(controls, indent=2) + '\n')
        print('START', label, flush=True)
        with (OUT / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((output / 'report.json').read_text())
        row.update(status='finished', returncode=result.returncode,
                   functional_status=report.get('functional_status', report['status']))
        (OUT / 'control-invocations.json').write_text(json.dumps(controls, indent=2) + '\n')
        print('FINISH', label, result.returncode, row['functional_status'], flush=True)
        if row['functional_status'] != 'passed':
            raise SystemExit(result.returncode or 1)
    initial = json.loads((OUT / f'candidate-{mode}-idle/comparison.json').read_text())
    repeated = json.loads((OUT / f'candidate-{mode}-repeat1/comparison.json').read_text())
    control = compare_runs(OUT / f'baseline-{mode}-idle', OUT / f'baseline-{mode}-repeat1')
    (OUT / f'{mode}-unchanged-main-comparison.json').write_text(json.dumps(control, indent=2) + '\n')
    qualified = qualify([initial, repeated], [control])
    (OUT / f'{mode}-qualified.json').write_text(json.dumps(qualified, indent=2) + '\n')
    print('QUALIFIED', mode, qualified['status'], flush=True)
