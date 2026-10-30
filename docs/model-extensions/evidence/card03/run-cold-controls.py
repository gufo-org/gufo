#!/usr/bin/env python3
"""Focused cold-schema histories, preserving original disk policy and prefix."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
OUT = ROOT / 'artifacts/model-extensions/card03'
initial = json.loads((OUT / 'functional-invocations.json').read_text())
rows = []
for index in (1, 2):
    for arm in ('baseline', 'candidate'):
        previous = next(row for row in initial if row['label'] == f'{arm}-mtp-idle')
        command = list(previous['command'])
        label = f'{arm}-mtp-cold-control{index}'
        output = OUT / label
        command[command.index('--output') + 1] = str(output)
        command[command.index('--')] = '--through-case'
        insertion = command.index('--through-case') + 1
        command[insertion:insertion] = ['long-context:long_schema_cold', '--']
        if arm == 'candidate':
            command[command.index('--baseline') + 1] = str(OUT / f'baseline-mtp-cold-control{index}')
        row = {'label': label, 'command': command, 'status': 'running'}
        rows.append(row)
        (OUT / 'cold-control-invocations.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('START', label, flush=True)
        with (OUT / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads((output / 'report.json').read_text())
        row.update(status='finished', returncode=result.returncode,
                   functional_status=report.get('functional_status', report['status']))
        (OUT / 'cold-control-invocations.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('FINISH', label, result.returncode, row['functional_status'], flush=True)
        if row['functional_status'] != 'passed':
            raise SystemExit(result.returncode or 1)
