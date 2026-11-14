#!/usr/bin/env python3
"""Check every decoder transfer against snapshots of the original schedule."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'decoder_layout', ROOT / 'src/core/npu/kernels/qwen3dec/layout.py')
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)
# Capture TAP operands without importing IRON or touching an NPU.
LAYOUT.t = lambda total, offset, sizes, strides: [total, offset, sizes, strides]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class DecoderPlanTest(unittest.TestCase):
    def test_all_transfers_match_original_schedule(self):
        fixture = json.loads((ROOT / 'tests/fixtures/npu/qwen3dec_phases.json').read_text())
        for case in fixture['cases']:
            with self.subTest(model=case['model'], dispatch=case['dispatch']):
                layout = LAYOUT.Layout(case['model'])
                phases = [{'heads': [p.heads(s) for s in range(16)],
                           'stream': [p.stream(s) for s in range(16)],
                           'drains': [p.drains(c) for c in range(8)]}
                          for p in layout.phases(case['dispatch'])]
                self.assertEqual(len(phases), case['phase_count'])
                self.assertEqual(digest(layout.items()), case['layout_sha256'])
                self.assertEqual(digest(phases), case['phases_sha256'])


if __name__ == '__main__':
    unittest.main()
