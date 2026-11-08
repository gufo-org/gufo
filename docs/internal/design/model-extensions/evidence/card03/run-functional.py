#!/usr/bin/env python3
"""Retain ordered, isolated baseline/candidate Flash-Next spike captures."""
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
OUT = ROOT / 'artifacts/model-extensions/card03'
WEIGHTS = ROOT / 'artifacts/flash-next-full-20261027/weights'
BINARIES = {'baseline': ROOT / 'build/release/gufo',
            'candidate': Path('/tmp/gufo-card03-spike/build/release/gufo')}
results = []
for mode in ('off', 'mtp'):
    for arm in ('baseline', 'candidate'):
        label = f'{arm}-{mode}-idle'
        output = OUT / label
        command = [sys.executable, str(ROOT / 'tests/functional/run.py'),
                   '--output', str(output), '--sampling-preset', 'qwen38',
                   '--suite', 'state-edges', '--suite', 'long-context',
                   '--suite', 'tool-reasoning', '--suite', 'cache', '--tier', 'pr',
                   '--startup-timeout', '600', '--request-timeout', '240']
        command += (['--record-baseline'] if arm == 'baseline' else
                    ['--baseline', str(OUT / f'baseline-{mode}-idle')])
        command += ['--', str(BINARIES[arm]), 'serve', 'llm', '--model',
                    str(WEIGHTS / 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'),
                    '--mmproj', str(WEIGHTS / 'mmproj-BF16.gguf'),
                    '--speculative', mode, '--context', '8192', '--sessions', '4']
        if mode == 'mtp':
            command += ['--mtp-model', str(WEIGHTS / 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf')]
        row = {'label': label, 'command': command, 'status': 'running'}
        results.append(row)
        (OUT / 'functional-invocations.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'START {label}', flush=True)
        with (OUT / f'{label}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row.update(status='finished', returncode=result.returncode)
        report = json.loads((output / 'report.json').read_text())
        row['functional_status'] = report.get('functional_status', report['status'])
        (OUT / 'functional-invocations.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'FINISH {label}: exit={result.returncode} functional={row["functional_status"]}', flush=True)
        if row['functional_status'] != 'passed':
            raise SystemExit(result.returncode or 1)
