#!/usr/bin/env python3
"""Check the rolled prefill schedule against the host's full layer layout."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prefill_plan", ROOT / "src/core/npu/kernels/qwen3pre/plan.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


class PrefillPlanTest(unittest.TestCase):
    def test_every_phase_matches_host_layout(self):
        for rows in (256, 512, 1024):
            plan = PLAN.Plan(rows)
            expanded = [plan.phases[0]]
            for layer in range(PLAN.LAYERS):
                expanded.extend(plan.layer_phases(layer))
            with self.subTest(rows=rows):
                self.assertEqual(len(expanded), 309)
                self.assertEqual(expanded, plan.phases)


if __name__ == "__main__":
    unittest.main()
