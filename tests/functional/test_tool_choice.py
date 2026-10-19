"""Checks the tool-choice cache retry contract. Run it in nix develop for the OpenAI SDK."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import tool_choice

PROMPT = 299


class CaseChecked(Exception):
    """Stop after the first case, before the unrelated cases."""


def result(finish="stop", text="DONE", tools=(), cached=0, prompt=PROMPT):
    return {"text": text, "reasoning": "", "finish": finish, "tools": list(tools),
            "usage": {"prompt_tokens": prompt, "completion_tokens": 5, "cached_tokens": cached}}


def refusal(reason="byte_capacity", tokens=PROMPT):
    return f"schema=component-cache-v1 event=capture_refused reason={reason} tokens={tokens}\n"


class ToolChoiceRetryTest(unittest.TestCase):
    def run_case(self, retry_cached, *, suffix=23, first_log="", continued="", retry_log="", with_log=True):
        call = {"id": "c1", "function": {"name": "read", "arguments": '{"path":"fixture.txt"}'}}
        first = result("tool_calls", "", [call], prompt=100)
        replies = [first, result(cached=105), result(cached=retry_cached)]
        seen = []
        client = Mock()
        client.post.return_value = {"input_tokens": suffix}
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("")

            def chat(_client, body, streaming=False):
                seen.append(body)
                with log.open("a") as output:
                    output.write({0: first_log, 1: continued, 2: retry_log}.get(len(seen) - 1, ""))
                if len(seen) > 3:
                    raise CaseChecked
                return deepcopy(replies[len(seen) - 1])

            with patch.object(tool_choice, "assert_native"), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    tool_choice.check_tool_choice_cache(
                        client, "fixture", {}, chat, log if with_log else None)
                except CaseChecked:
                    return
        self.fail("unexpected completion")

    def test_full_replay_needs_no_log(self):
        self.run_case(PROMPT, with_log=False)

    def test_suffix_requires_the_continued_requests_refusal(self):
        for reason in ("byte_capacity", "entry_capacity"):
            with self.subTest(reason=reason):
                self.run_case(276, continued=refusal(reason))

    def test_other_evidence_does_not_excuse_prefill(self):
        for evidence in ({}, {"with_log": False}, {"first_log": refusal()}, {"retry_log": refusal()},
                         {"continued": refusal(tokens=PROMPT - 1)},
                         {"continued": "event=capture_failed reason=transfer tokens=299\n"}):
            with self.subTest(evidence=evidence), self.assertRaises(AssertionError):
                self.run_case(276, **evidence)

    def test_prefill_beyond_the_measured_suffix_fails(self):
        with self.assertRaises(AssertionError):
            self.run_case(270, continued=refusal(), suffix=23)


if __name__ == "__main__":
    unittest.main()
