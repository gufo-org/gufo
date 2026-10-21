"""CPU failure injection for deep-history evidence; does not qualify a model."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from qwen_deep_options import assert_cold_control, assert_publication_wait, verify_execution_log


def response(cached=0):
    return {
        "choices": [{"message": {"role": "assistant", "content": "READY"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20000, "completion_tokens": 1,
                  "prompt_tokens_details": {"cached_tokens": cached},
                  "gufo": {"prefill_tokens": 20000 - cached,
                           "component_cache": {"selected_tokens": cached}}},
    }


WAIT = (
    "event=publication_wait request=7 stage=start checkpoint_id=42 "
    "pending_tokens=17000 available_tokens=10000 limit_ms=100\n"
    "event=published tokens=17000 checkpoint_id=42 payload_bytes=100\n"
    "event=publication_wait request=7 stage=release checkpoint_id=42 "
    "pending_tokens=17000 available_tokens=10000 limit_ms=100 wait_ms=80 release=not_pending\n"
)


class QwenRecoveryEvidenceTest(unittest.TestCase):
    def test_actual_drafts_exclude_repeated_progress_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory, "server.log")
            log.write_text(
                "event=load_completed kind=text speculative=dflash2\n"
                "[progress] phase=decode draft_proposed=100\n"
                "[http] event=completed method=POST draft_proposed=7\n"
                "[progress] phase=decode draft_proposed=7\n")
            verify_execution_log(directory, "server.log", "dflash2")
            record = json.loads(Path(directory, "server.log.execution.json").read_text())
            self.assertEqual(record["proposed"], 7)
            log.write_text(log.read_text().replace("draft_proposed=7\n", "draft_proposed=0\n"))
            with self.assertRaises(RuntimeError):
                verify_execution_log(directory, "server.log", "dflash2")

    def test_exact_uncached_control(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_cold_control(directory, "control", response(17000), response())
            self.assertTrue(json.loads(Path(directory, "control.cold-comparison.json").read_text())["qualified"])

    def test_output_and_cold_work_failures_are_retained(self):
        for kind in ("output", "cache", "prefill", "missing", "boolean"):
            cold = response()
            if kind == "output":
                cold["choices"][0]["message"]["content"] = "CHANGED"
            elif kind == "cache":
                cold = response(1)
            elif kind == "prefill":
                cold["usage"]["gufo"]["prefill_tokens"] -= 1
            elif kind == "missing":
                cold["usage"]["prompt_tokens_details"].clear()
            else:
                cold["usage"]["prompt_tokens_details"]["cached_tokens"] = False
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    assert_cold_control(directory, "control", response(17000), cold)
                self.assertFalse(json.loads(Path(directory, "control.cold-comparison.json").read_text())["qualified"])

    def test_publication_and_deeper_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            assert_publication_wait(directory, "rewrite", WAIT, response(17000))
            self.assertTrue(json.loads(Path(directory, "rewrite.publication-wait.json").read_text())["qualified"])

    def test_wait_without_successful_publication_is_rejected(self):
        for segment in ("", WAIT.replace("not_pending", "cost_bound"),
                        WAIT.replace("event=published", "event=queued"),
                        WAIT.replace("available_tokens=10000", "available_tokens=15000"),
                        WAIT.replace("request=7 stage=release", "request=8 stage=release"),
                        WAIT.replace("event=published tokens=17000 checkpoint_id=42",
                                     "event=published tokens=17000 checkpoint_id=43")):
            with self.subTest(segment=segment), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    assert_publication_wait(directory, "rewrite", segment, response(17000))
                self.assertFalse(json.loads(Path(directory, "rewrite.publication-wait.json").read_text())["qualified"])

    def test_publication_without_matching_response_reuse_is_rejected(self):
        for key in ("cached", "selected"):
            result = deepcopy(response(17000))
            if key == "cached":
                result["usage"]["prompt_tokens_details"]["cached_tokens"] = 10000
            else:
                result["usage"]["gufo"]["component_cache"]["selected_tokens"] = 10000
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(RuntimeError):
                    assert_publication_wait(directory, "rewrite", WAIT, result)


if __name__ == "__main__":
    unittest.main()
