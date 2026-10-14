"""CPU failure injection for deep-history evidence; does not qualify a model."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from qwen_deep_options import assert_cold_control, assert_publication_wait


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
    "event=publication_wait_begin id=7 pending_tokens=17000 cached_tokens=10000\n"
    "event=published tokens=17000 payload_bytes=100\n"
    "event=publication_wait_end id=7 reason=pending_settled cached_tokens=17000 wait_ms=80\n"
)


class QwenRecoveryEvidenceTest(unittest.TestCase):
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
        for segment in ("", WAIT.replace("pending_settled", "timeout"),
                        WAIT.replace("event=published", "event=queued"),
                        WAIT.replace("cached_tokens=10000", "cached_tokens=15000"),
                        WAIT.replace("wait_end id=7", "wait_end id=8")):
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
