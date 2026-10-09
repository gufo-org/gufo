"""Fast failure-injection checks for the card 01 workload contracts."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_compaction import ARCHIVE_LINE
from cache_workloads import Workload, check_cache_transforms, check_cache_pressure
from cache_messages_loop import check_cache_messages_loop, chat_messages
from cache_lifecycle import Lifecycle, published_tokens


def reply(total, cached, code="ALPHA", tools=None):
    return {"text": code, "reasoning": "", "tools": tools or [],
            "finish": "tool_calls" if tools else "stop", "usage": {
                "prompt_tokens": total, "cached_tokens": cached,
                "completion_tokens": 12 if tools else 2,
                "gufo": {"prefill_tokens": total - cached, "cache_snapshot_bytes": 300}}}


class CacheWorkloadsTest(unittest.TestCase):
    def transforms(self, lost=False):
        calls, checks, seen, seeds, large = [], {}, set(), {}, {}

        def send(client, body):
            calls.append(deepcopy(body))
            messages = body["messages"]
            label = messages[0]["content"].splitlines()[0]
            total = 2048 + len(messages) * 32 + 14 * sum(
                m.get("content", "").count(ARCHIVE_LINE) for m in messages)
            seeds.setdefault(label, total)
            if len(messages) == 6 and messages[-1]["content"].endswith("ALPHA."):
                large.setdefault(label, total)
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            key = json.dumps(messages)
            if cold or len(messages) == 2:
                cached = 0
            elif key in seen:
                cached = total
            elif len(messages) == 6 and messages[-1]["content"].endswith("ALPHA."):
                cached = seeds[label] - 32
            elif messages[4]["content"].startswith("[Old"):
                cached = 0 if lost else seeds[label] - 32
            else:
                cached = large[label] - 32
            seen.add(key)
            code = re.findall(r"only (\w+)\.", messages[-1]["content"])[-1]
            return reply(total, cached, code)

        with contextlib.redirect_stderr(io.StringIO()):
            try:
                check_cache_transforms(None, "fixture", checks, send, 32768)
            finally:
                self.calls, self.checks = calls, checks
        return calls, checks

    def test_transforms_only_change_the_intended_history_and_delay_controls(self):
        calls, checks = self.transforms()
        self.assertEqual(len(calls), 28)
        self.assertTrue(all(b.get("extra_body", {}).get("cache_prompt") is False
                            for b in calls[14:]))
        prune = calls[4]["messages"]
        self.assertEqual(prune[4]["content"], "[Old tool output cleared by the client.]")
        self.assertEqual(prune[:4], calls[3]["messages"][:4])
        self.assertEqual(prune[5:], calls[3]["messages"][5:])
        self.assertEqual(checks["transform_micro_deep"]["usage"]["prompt_tokens"] -
                         checks["transform_micro_edited"]["usage"]["prompt_tokens"], 224)
        self.assertEqual(calls[11], calls[12])
        self.assertEqual(calls[11]["messages"][:-1], calls[13]["messages"][:-1])

    def test_transforms_keep_all_controls_when_pruning_loses_its_boundary(self):
        with self.assertRaisesRegex(AssertionError, "transform_prune_edited"):
            self.transforms(lost=True)
        self.assertEqual(len(self.calls), 28)
        self.assertTrue(self.checks["transforms_evidence"]["failures"])

    def pressure(self, regression=None):
        calls, checks, last, visits = [], {}, {}, {}

        def send(client, body):
            calls.append(deepcopy(body))
            messages = body["messages"]
            label = messages[0]["content"].splitlines()[0]
            cold = body.get("extra_body", {}).get("cache_prompt") is False
            total = 2500 + len(messages) * 32 + 14 * sum(
                m.get("content", "").count(ARCHIVE_LINE) for m in messages)
            visits[label] = visits.get(label, 0) + 1
            cached = 0 if cold or label not in last else min(last[label] - 8, total)
            if label == "pressure_compaction" and len(messages) == 2 and visits[label] > 1:
                cached = 0 if cold else 2000
            if label.startswith("nonce_") and regression == "false_header" and not cold:
                cached = 2000
            if label == "pressure_live_ALPHA" and len(messages) > 2 and not cold:
                if regression == "lost":
                    cached = 0
            if label == "pressure_compaction" and visits[label] == 3 and regression == "abandoned":
                cached = total
            last[label] = total
            code = re.findall(r"only (\w+)\.", messages[-1]["content"])[-1]
            return reply(total, cached, code)

        with contextlib.redirect_stderr(io.StringIO()):
            try:
                check_cache_pressure(None, "fixture", checks, send, 1000, 2)
            finally:
                self.calls, self.checks = calls, checks
        return calls, checks

    def test_pressure_fills_budget_replays_ui_and_resumes_after_children(self):
        calls, checks = self.pressure()
        fill = checks["pressure_pressure"]
        self.assertGreaterEqual(fill["published_bytes"], 2 * fill["capacity_bytes"])
        self.assertEqual(checks["pressure_shape"]["concurrency"], 2)
        self.assertEqual(len(checks["pressure_fanout"]), 3)
        first_cold = next(i for i, b in enumerate(calls)
                          if b.get("extra_body", {}).get("cache_prompt") is False)
        self.assertEqual(first_cold * 2, len(calls))
        self.assertTrue(all(b.get("extra_body", {}).get("cache_prompt") is False
                            for b in calls[first_cold:]))
        self.assertIn("Parent resumed", calls[first_cold - 1]["messages"][-1]["content"])
        self.assertTrue(any("Generate title for this chat" in b["messages"][-1]["content"]
                            for b in calls[:first_cold]))

    def test_pressure_rejects_lost_live_history_false_headers_and_abandoned_retention(self):
        for regression in ("lost", "false_header", "abandoned"):
            with self.subTest(regression=regression), self.assertRaises(AssertionError):
                self.pressure(regression)
            self.assertTrue(self.checks["pressure_evidence"]["failures"])
            self.assertEqual(len(self.calls) % 2, 0)

    def messages_loop(self, lost=False):
        checks, requests, previous = {}, [], 0

        def tokens(system, messages):
            texts = [system]
            for m in messages:
                content = m["content"] or ""
                texts += [content] if isinstance(content, str) else [
                    b.get("content", b.get("text", "")) for b in content]
            return 1000 + len(messages) * 30 + 14 * sum(t.count(ARCHIVE_LINE) for t in texts)

        def stream(client, body):
            nonlocal previous
            requests.append(deepcopy(body))
            messages = body["messages"]
            latest = messages[-1]["content"]
            call = isinstance(latest, str)
            total = tokens(body["system"], messages)
            cached = max(0, previous - 8) if not lost else 0
            previous = total
            turn = len(messages) // 4
            blocks = [{"type": "tool_use", "id": f"call-{turn}", "name": "read_archive",
                       "input": {"turn": turn}}] if call else [{"type": "text", "text": "BETA"}]
            return {"blocks": blocks, "finish": "tool_use" if call else "end_turn",
                    "usage": {"input_tokens": total, "cache_read_input_tokens": cached,
                              "output_tokens": 12 if call else 2}}

        def cold(client, body):
            requests.append(deepcopy(body))
            self.assertIs(body["extra_body"]["cache_prompt"], False)
            system, messages = body["messages"][0]["content"], body["messages"][1:]
            total = tokens(system, messages)
            call = messages[-1]["role"] == "user"
            turn = len(messages) // 4
            calls = [{"function": {"name": "read_archive", "arguments": json.dumps({"turn": turn})}}]
            return reply(total, 0, "" if call else "BETA", calls if call else None)

        with contextlib.redirect_stderr(io.StringIO()):
            try:
                check_cache_messages_loop(None, "fixture", checks, cold, 32768, stream)
            finally:
                self.calls, self.checks = requests, checks
        return requests, checks

    def test_messages_loop_uses_actual_calls_and_delays_all_cold_controls(self):
        requests, checks = self.messages_loop()
        self.assertEqual(len(requests), 48)
        for turn in range(12):
            call = checks[f"messages_loop_call_{turn:02d}"]["tools"][0]
            result = requests[turn * 2 + 1]["messages"][-1]["content"][0]
            self.assertEqual(result["tool_use_id"], call["id"])
            self.assertEqual(call["input"], {"turn": turn})
        self.assertGreaterEqual(checks["messages_loop_evidence"]["prompt_tokens"], 10000)

    def test_messages_loop_preserves_all_evidence_on_cache_failures(self):
        with self.assertRaisesRegex(AssertionError, "messages_loop"):
            self.messages_loop(lost=True)
        self.assertEqual(len(self.calls), 48)

    def test_history_converter_preserves_tool_input_types_and_pairing(self):
        source = [{"role": "assistant", "content": [{"type": "tool_use", "id": "c",
                   "name": "read_archive", "input": {"turn": 7}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c",
                   "content": "result"}]}]
        converted = chat_messages("system", source)
        call = converted[1]["tool_calls"][0]
        self.assertEqual(call["id"], converted[2]["tool_call_id"])
        self.assertEqual(json.loads(call["function"]["arguments"]), {"turn": 7})

    def test_publication_boundaries_ignore_queued_skipped_and_unpublished_entries(self):
        log = ("event=disk_cache action=stored reason=saved file_bytes=1 tokens=3000\n"
               "event=disk_cache action=skipped reason=staging_capacity file_bytes=2 tokens=8000\n"
               "event=disk_cache action=restored reason=hit file_bytes=1 tokens=3000\n")
        self.assertEqual(published_tokens(log), [3000])

    def test_lifecycle_removes_its_disk_after_a_failed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fail(runner, seed, case_root):
                runner.disk.mkdir()
                (runner.disk / "unpublished").write_text("temporary")
                raise RuntimeError("injected")

            with patch.object(Lifecycle, "crash_case", fail), \
                    patch("cache_lifecycle.provenance", return_value={"harness_sha256": "fixture"}):
                runner = Lifecycle(["gufo", "serve", "llm", "--speculative", "off"],
                                   root, 1, root / "library.so")
                with patch("cache_lifecycle.shutil.disk_usage") as usage:
                    usage.return_value.free = 32 * 1024**3
                    runner.run_case("crash")
                self.assertEqual(runner.report["cases"]["crash"]["status"], "failed")
                self.assertFalse((root / "crash/disk").exists())


if __name__ == "__main__":
    unittest.main()
