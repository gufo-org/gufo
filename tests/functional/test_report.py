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

import phases
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
                "--authorize", authorize, "--smoke", smoke,
                "--run-url", "https://run",
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
        self.assertIn("only the automatic release merge waits for it", comment)

    def test_pr_tier_is_named_with_its_release_suites(self):
        summary = {"groups": [{"profile": "flash-next-q4/mtp", "configuration": "standard",
                               "suites": ["stops"], "status": "passed"}]}
        plan = {**self.plan, "tier": "pr", "release_only": [{"suite": "cache-stalls"}]}
        with tempfile.TemporaryDirectory() as directory:
            _, comment = self.run_report(directory, plan, summary)
        self.assertIn("### ✅ Devlab functional checks · PR tier: 1 configuration groups passed",
                      comment)
        self.assertIn("0 deferred, 1 for the release run, 0 out of scope", comment)
        with tempfile.TemporaryDirectory() as directory:
            _, comment = self.run_report(directory, self.plan, summary)
        self.assertIn("### ✅ Devlab functional checks: 1 configuration", comment)

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

    def test_phase_verdicts_need_every_repeat(self):
        self.assertEqual(phases.verdict([(100, 110), (100, 120)]), "slower")
        self.assertEqual(phases.verdict([(100, 110), (100, 101)]), "noisy")
        self.assertEqual(phases.verdict([(100, 90), (100, 80)]), "faster")
        # Above 5% but within 3 ms is noise, not a regression.
        self.assertEqual(phases.verdict([(20, 22), (20, 22.5)]), "same")
        self.assertEqual(phases.verdict([]), "missing")

    def test_phase_pipeline_is_drawn_with_verdict_colors(self):
        def arm(revision, prefill, text="same"):
            request = {"cache_restore_ms": 0, "prompt_ms": prefill, "ttft_ms": prefill + 20,
                       "predicted_ms": 2000, "cache_snapshot_ms": 50, "prompt_tokens": 2048,
                       "text_sha256": text}
            return {"revision": revision, "startup": {
                        "init": 900, "weights": 14000, "vision": 540, "structured": None,
                        "listen": 12},
                    "requests": {key: dict(request) for key in ("short", "long", "resume")}}
        arms = [arm("base", 1500), arm("head", 1700, "other"),
                arm("base", 1500), arm("head", 1680, "other")]
        arms[1]["requests"]["resume"]["cache_restore_ms"] = 40
        found, diverged = phases.summarize(arms)
        verdicts = {phase["key"]: phase["verdict"] for phase in found}
        self.assertEqual(verdicts["short.prompt_ms"], "slower")
        self.assertEqual(verdicts["startup.weights"], "same")
        self.assertEqual(verdicts["startup.structured"], "missing")
        self.assertEqual(diverged, ["long", "resume", "short"])
        data = {"settings": {"repeats": 2, "slowdown": .05, "noise_ms": 3,
                             "requests": {"short": "Short prompt", "long": "Long prompt",
                                          "resume": "Follow-up"}},
                "wall_s": 600, "arms": arms, "phases": found, "diverged": diverged}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            (results / "phases.json").write_text(json.dumps(data))
            (root / "plan.json").write_text(json.dumps(self.plan))
            argv = ["report.py", "--plan", str(root / "plan.json"), "--results", str(results),
                    "--authorize", "success", "--smoke", "success", "--run-url", "https://run",
                    "--comment", str(root / "comment.md"), "--status", str(root / "status.env")]
            with patch.object(sys, "argv", argv):
                self.assertEqual(report.main(), 0)
            comment = (root / "comment.md").read_text()
        self.assertIn("```mermaid\nflowchart LR", comment)
        self.assertIn('subgraph g1["Short prompt · 2,048 tok"]', comment)
        self.assertIn('["Prefill<br/>1.50 s → 1.69 s<br/>+12.7%"]:::slower', comment)
        self.assertIn('["Weights and model<br/>14.00 s → 14.00 s<br/>+0.0%"]:::same', comment)
        # Unexercised phases stay out; a one-sided restore is inconclusive.
        self.assertNotIn("Structured output", comment)
        self.assertNotIn("Short prompt · 2,048 tok · Cache restore", comment)
        self.assertIn("| Follow-up · 2,048 tok · Cache restore | 0.0 ms, 0.0 ms "
                      "| 40 ms, 0.0 ms | noisy |", comment)
        self.assertIn("g0 --> g1 --> g2 --> g3", comment)
        self.assertIn("Measured in 10.0 min", comment)
        self.assertIn("⚠️ Generated text differs", comment)


if __name__ == "__main__":
    unittest.main()
