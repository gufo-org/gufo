#!/usr/bin/env python3
"""Retain individual standard-benchmark observations, never rate averages."""
import json
import hashlib
import re
from pathlib import Path

ROOT = Path('/home/mixer/gufo/artifacts/model-extensions/card03/speed')
left, right = [json.loads((ROOT / f'{arm}-http-observations.json').read_text())
               for arm in ('baseline', 'candidate')]
assert len(left['requests']) == len(right['requests'])
observations = []
for index, (a, b) in enumerate(zip(left['requests'], right['requests'])):
    assert a['request_sha256'] == b['request_sha256'], index
    assert a['history_sha256'] == b['history_sha256'], index
    x, y = a['observation'], b['observation']
    invariants = ('prompt_tokens', 'cached_prompt_tokens', 'completion_tokens',
                  'prefill_tokens', 'draft_tokens', 'draft_accepted_tokens',
                  'completion_sha256', 'execution_plan', 'physical_execution_width')
    changes = {key: [x[key], y[key]] for key in invariants if x[key] != y[key]}
    phases = {}
    for metric in ('wall_ms', 'client_ttft_ms', 'client_event_inter_token_ms',
                   'queue_ms', 'prefill_ms', 'decode_ms', 'server_ttft_ms',
                   'server_inter_token_ms', 'server_max_inter_token_ms'):
        aa, bb = x[metric], y[metric]
        if aa is None or bb is None:
            phases[metric] = {'baseline': aa, 'candidate': bb, 'status': 'unavailable'}
            continue
        allowance = max(aa * 0.05, 10)
        phases[metric] = {'baseline': aa, 'candidate': bb,
                          'allowed_increase_ms': allowance,
                          'status': 'flagged' if bb - aa > allowance else 'passed'}
    observations.append({'index': index, 'request_sha256': a['request_sha256'],
                         'history_sha256': a['history_sha256'],
                         'baseline_server_log': a['server_log'],
                         'candidate_server_log': b['server_log'],
                         'invariant_changes': changes, 'phases': phases})
result = {'scope': 'All returned HTTP observations including calibration/warmup; per request/phase 5% and 10 ms. Flags require investigation, not an aggregate pass.',
          'requests': observations}
(ROOT / 'http-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print('requests', len(observations))
print('invariant changes', sum(bool(r['invariant_changes']) for r in observations))
print('flagged phases', sum(p['status'] == 'flagged' for r in observations for p in r['phases'].values()))

cli = []
for mode in ('off', 'mtp'):
    bench = []
    prompt = []
    traces = []
    for arm in ('baseline', 'candidate'):
        text = (ROOT / f'{arm}-bench-{mode}.stdout.txt').read_text()
        rates = {name: float(rate) for name, rate in
                 re.findall(r'\|\s*(pp2048|tg128)\s*\|\s*([0-9.]+)', text)}
        assert set(rates) == {'pp2048', 'tg128'}
        bench.append({name: int(name[2:]) / rate * 1000 for name, rate in rates.items()})
        stdout = (ROOT / f'{arm}-prompt-{mode}.stdout.txt').read_text()
        normalized = re.sub(r'^\[Model Load\]: .*\n', '', stdout, flags=re.MULTILINE)
        stderr = (ROOT / f'{arm}-prompt-{mode}.stderr.txt').read_text()
        count, rate = re.search(r'Generated (\d+) tokens \(([0-9.]+) tok/s\)', stderr).groups()
        prompt.append({'output_sha256': hashlib.sha256(normalized.encode()).hexdigest(),
                       'tokens': int(count), 'decode_ms': int(count) / float(rate) * 1000})
        traces.append((ROOT / f'{arm}-bench-{mode}.stderr.txt').read_text())
    metrics = []
    for name in ('pp2048', 'tg128'):
        a, b = bench[0][name], bench[1][name]
        metrics.append({'metric': name, 'baseline_ms': a, 'candidate_ms': b,
                        'status': 'flagged' if b - a > max(10, a * .05) else 'passed'})
    a, b = [p['decode_ms'] for p in prompt]
    metrics.append({'metric': 'prompt_decode', 'baseline_ms': a, 'candidate_ms': b,
                    'status': 'flagged' if b - a > max(10, a * .05) else 'passed'})
    hashes = [re.search(r'output_sha256=([0-9a-f]+)', t).group(1) for t in traces]
    cli.append({'mode': mode, 'phases': metrics, 'prompt': prompt,
                'bench_output_sha256': hashes,
                'output_equal': prompt[0]['output_sha256'] == prompt[1]['output_sha256']
                    and prompt[0]['tokens'] == prompt[1]['tokens'] and hashes[0] == hashes[1]})
(ROOT / 'cli-comparison.json').write_text(json.dumps({
    'scope': 'One matched run per CLI phase. Durations derived from printed rounded rates; no averaging. Prompt driver exposes decode only; HTTP separately covers prefill/TTFT.',
    'modes': cli}, indent=2) + '\n')
print('CLI', [(r['mode'], r['output_equal'], [p['status'] for p in r['phases']]) for r in cli])
