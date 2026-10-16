#!/usr/bin/env python3
"""CPU-only checks for the Devlab functional PR report."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report


BENCH = """| model | size | backend | test | t/s |
| ----- | ---- | ------- | ---- | --- |
| flash-next | 50 GiB | ROCm | pp2048 @ d0 | 1300.00 ± 0.00 |
| flash-next | 50 GiB | ROCm | tg128 @ d0 | 20.00 ± 0.00 |
"""


class ReportTest(unittest.TestCase):
    def run_report(self, directory, plan, summary=None, authorize="success", smoke="success",
                   benchmark=False, bench=None):
        root = Path(directory)
        (root / "plan.json").write_text(json.dumps(plan))
        results = root / "results"
        results.mkdir()
        if summary is not None:
            (results / "summary.json").write_text(json.dumps(summary))
        for name, text in (bench or {}).items():
            (results / name).write_text(text)
        argv = ["report.py", "--plan", str(root / "plan.json"), "--results", str(results),
                "--authorize", authorize, "--smoke", smoke, "--run-url", "https://run",
                "--comment", str(root / "comment.md"), "--status", str(root / "status.env")]
        if benchmark:
            argv.append("--benchmark")
        with patch.object(sys, "argv", argv):
            self.assertEqual(report.main(), 0)
        status = dict(line.split("=", 1) for line in (root / "status.env").read_text().splitlines())
        return status, (root / "comment.md").read_text()

    def setUp(self):
        self.plan = {"base": "a" * 40, "head": "b" * 40, "deferred": [], "out_of_scope": [],
                     "groups": [{"profile": "flash-next-q4/off"}]}

    def test_failed_group_is_a_failure_with_detail(self):
        summary = {"groups": [
            {"profile": "flash-next-q4/off", "configuration": "standard",
             "suites": ["stops", "tools", "discovery"], "status": "failed", "error": None,
             "failures": {"stops": "sync close", "tools": "sync close"}, "log_tail": "noise"},
            {"profile": "flash-next-q4/mtp", "configuration": "cache-pressure",
             "suites": ["cache-bridge"], "status": "failed", "error": "server exited 1",
             "failures": {}, "log_tail": "Failed to open GGUF"},
            {"profile": "flash-next-q4/mtp", "configuration": "standard", "suites": ["stops"],
             "status": "passed"}]}
        with tempfile.TemporaryDirectory() as directory:
            status, comment = self.run_report(directory, self.plan, summary, smoke="failure")
        self.assertEqual(status["state"], "failure")
        self.assertTrue(comment.startswith(report.MARKER))
        self.assertIn("| ❌ | AR · standard | 1/3 | `stops`, `tools` |", comment)
        self.assertIn("| ⚠️ | MTP · cache-pressure | — | server or runner error |", comment)
        self.assertIn("| ✅ | MTP · standard | 1/1 |  |", comment)
        self.assertIn("- `stops`, `tools`: sync close", comment)
        self.assertEqual(report.suite_list(["a", "b", "c", "d"], 3), "`a`, `b`, `c` +1 more")
        self.assertIn("Failed to open GGUF", comment)
        self.assertNotIn("noise", comment)
        self.assertIn("never blocks merging", comment)

    def test_cluster_problems_are_errors_not_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _ = self.run_report(directory, self.plan, smoke="failure")
        self.assertEqual(status["state"], "error")
        with tempfile.TemporaryDirectory() as directory:
            status, _ = self.run_report(directory, self.plan, authorize="failure", smoke="skipped")
        self.assertEqual(status["state"], "error")

    def test_no_impact_skip_passes(self):
        plan = dict(self.plan, groups=[])
        with tempfile.TemporaryDirectory() as directory:
            status, _ = self.run_report(directory, plan, smoke="skipped")
        self.assertEqual(status["state"], "success")

    def test_benchmark_reports_deltas_against_base(self):
        summary = {"groups": []}
        head = BENCH.replace("1300.00", "1430.00")
        with tempfile.TemporaryDirectory() as directory:
            status, comment = self.run_report(
                directory, self.plan, summary, benchmark=True,
                bench={"bench-base.md": BENCH, "bench-head.md": head})
        self.assertEqual(status["state"], "success")
        self.assertIn("| `pp2048 @ d0` | 1300.00 | 1430.00 | +10.0% |", comment)
        self.assertIn("| `tg128 @ d0` | 20.00 | 20.00 | +0.0% |", comment)


if __name__ == "__main__":
    unittest.main()
