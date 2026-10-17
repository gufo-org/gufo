"""Long-context replay contract checks; run in nix develop for the OpenAI SDK."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import openai_sdk


class ReplayChecked(Exception):
    """Stop before the unrelated Responses/thinking/cancellation checks."""


def response(cached, prefilled):
    return {"text": '{"keyword":"LANTERN"}', "reasoning": "", "finish": "stop",
            "tools": [], "usage": {"prompt_tokens": 3143, "cached_tokens": cached,
                                   "gufo": {"prefill_tokens": prefilled}}}


def refusal(reason="byte_capacity", tokens=3143):
    return (f"schema=component-cache-v1 event=capture_refused reason={reason} "
            f"tokens={tokens}\n")


class LongContextRetryTest(unittest.TestCase):
    def run_replay(self, replay, *, before="", cold="", during_replay="", with_log=True):
        checks = {}
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=ReplayChecked)))
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(before)

            def chat(_client, body, streaming=False):
                with log.open("a") as output:
                    output.write(during_replay if streaming else cold)
                return deepcopy(replay if streaming else response(0, 3143))

            with patch.object(openai_sdk, "chat_result", side_effect=chat) as send, \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    openai_sdk.check_long_context(
                        client, "fixture", checks, 8192, server_log=log if with_log else None)
                except ReplayChecked:
                    self.assertEqual(send.call_count, 2)
                    self.assertEqual(send.call_args_list[0].args[1],
                                     send.call_args_list[1].args[1])
                    self.assertEqual(checks["long_schema_cached"], replay)
                    return
        self.fail("long-context did not reach the next phase")

    def test_full_replay_needs_no_capacity_exception(self):
        self.run_replay(response(3143, 0), with_log=False)

    def test_framing_suffix_requires_the_previous_requests_capacity_refusal(self):
        for reason in ("byte_capacity", "entry_capacity"):
            with self.subTest(reason=reason):
                self.run_replay(response(3136, 7), cold=refusal(reason))

    def test_unrelated_or_failed_captures_do_not_excuse_prefill(self):
        for evidence in (
                {}, {"with_log": False},
                {"before": refusal()}, {"during_replay": refusal()},
                {"cold": refusal(tokens=3142)},
                {"cold": "schema=component-cache-v1 event=capture_failed reason=transfer tokens=3143\n"},
                {"cold": "schema=component-cache-v1 event=snapshot_skipped reason=3 tokens=3143\n"}):
            with self.subTest(evidence=evidence), self.assertRaises(AssertionError):
                self.run_replay(response(3136, 7), **evidence)

    def test_capacity_refusal_does_not_hide_context_loss_or_changed_output(self):
        wrong_output = response(3136, 7)
        wrong_output["text"] = '{"keyword":"WRONG"}'
        wrong_depth = response(3136, 7)
        wrong_depth["usage"]["prompt_tokens"] += 1
        for replay in (response(3126, 17), response(0, 3143), response(0, 7),
                       wrong_output, wrong_depth):
            with self.subTest(replay=replay), self.assertRaises(AssertionError):
                self.run_replay(replay, cold=refusal())


if __name__ == "__main__":
    unittest.main()
