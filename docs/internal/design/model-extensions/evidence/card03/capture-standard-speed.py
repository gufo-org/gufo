#!/usr/bin/env python3
"""Run the standard model benchmark and retain each returned observation."""
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path('/home/mixer/gufo')
sys.path.insert(0, str(ROOT / 'tools'))
from gufo.model_bench import llm
from gufo.model_bench.cli import main
output = Path(sys.argv[1])
original = llm.Session.request
rows = []
history = []
def capture(self, base_url, prompt, max_tokens, **kwargs):
    result = original(self, base_url, prompt, max_tokens, **kwargs)
    signature = {'prompt': prompt, 'max_tokens': max_tokens, **kwargs}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    history.append(fingerprint)
    rows.append({'request_sha256': fingerprint,
                 'history_sha256': hashlib.sha256(json.dumps(history).encode()).hexdigest(),
                 'server_log': str(self.active_log),
                 'observation': dataclasses.asdict(result)})
    return result
llm.Session.request = capture
try:
    code = main(sys.argv[2:])
finally:
    output.write_text(json.dumps({'command': sys.argv[2:], 'requests': rows}, indent=2) + '\n')
raise SystemExit(code)
