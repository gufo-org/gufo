#!/usr/bin/env python3
"""Bounded standard HTTP and direct CLI controls for card 03 (run alone)."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
OUT = ROOT / 'artifacts/model-extensions/card03/speed'
OUT.mkdir(exist_ok=False)
WEIGHTS = ROOT / 'artifacts/flash-next-full-20261027/weights'
MODEL = str(WEIGHTS / 'UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf')
MTP = str(WEIGHTS / 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf')
BINARIES = {'baseline': ROOT / 'build/release/gufo',
            'candidate': Path('/tmp/gufo-card03-spike/build/release/gufo')}
rows = []
def run(label, command):
    row = {'label': label, 'command': [str(s) for s in command], 'status': 'running'}
    rows.append(row)
    (OUT / 'commands.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('START', label, flush=True)
    start = time.monotonic()
    with (OUT / (label + '.stdout.txt')).open('w') as stdout, \
         (OUT / (label + '.stderr.txt')).open('w') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    row.update(status='finished', returncode=result.returncode,
               process_wall_ms=(time.monotonic() - start) * 1000)
    (OUT / 'commands.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('FINISH', label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
for arm, binary in BINARIES.items():
    run(arm + '-http', [sys.executable, str(ROOT / 'docs/model-extensions/evidence/card03/capture-standard-speed.py'),
        str(OUT / (arm + '-http-observations.json')), '--model', 'qwen3.8-flash-next', '--gufo', str(binary), '--gguf', MODEL,
        '--mtp', MTP, '--log-dir', str(OUT / (arm + '-http-logs')),
        '--artifacts-dir', str(OUT / (arm + '-http')), 'run', '--target', 'gufo',
        '--table', 'single-ar,single-mtp', '--depths', '0', '--context', '4096',
        '--repetitions', '1'])
for mode in ('off', 'mtp'):
    for arm, binary in BINARIES.items():
        speculation = ['--speculative', mode]
        if mode == 'mtp':
            speculation += ['--mtp-model', MTP]
        run(f'{arm}-bench-{mode}', [str(binary), 'bench', '--model', MODEL,
            '-p', '2048', '-n', '128', '-r', '1', '--seed', '1', '-v', *speculation])
        run(f'{arm}-prompt-{mode}', [str(binary), 'prompt', '--model', MODEL,
            '--prompt', 'Explain why a bounded queue needs backpressure in two short sentences.',
            '--think', 'off', '--no-display-prompt', '-n', '64', '--temperature', '0',
            '--seed', '1', '-v', *speculation])
run('candidate-prefill-validation', [str(BINARIES['candidate']), 'bench', '--model', MODEL,
    '-p', '0', '-n', '0', '--validate-prefill', '16'])
for mode in ('off', 'mtp'):
    normalized = []
    for arm in BINARIES:
        output = (OUT / f'{arm}-prompt-{mode}.stdout.txt').read_text()
        output = re.sub(r'^\[Model Load\]: .*\n', '', output, flags=re.MULTILINE)
        normalized.append(output)
    if normalized[0] != normalized[1]:
        raise SystemExit(f'CLI prompt mismatch for {mode}')
    print('MATCH prompt', mode, hashlib.sha256(normalized[0].encode()).hexdigest(), flush=True)
