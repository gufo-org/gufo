"""Cache accounting must not mistake a client history edit for full reuse."""

import json
from pathlib import Path
import tempfile
import unittest

from agent_long import check_continuation
from opencode_agent import sse_output


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "reply.sse"
        self.output.write_text('data: {"choices":[{"delta":{"reasoning_content":"Done."}}]}\n\n')
        self.previous = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "metrics": {"prefill_chunks": 2, "prefill_tokens": 27, "max_prefill_chunk_tokens": 22},
        }
        self.old = {"messages": [{"role": "tool", "content": "done"}]}
        self.omitted = {"messages": self.old["messages"] + [{"role": "user", "content": "next"}]}

    def check(self, cached, body=None):
        return check_continuation(
            self.previous, {"index": 2, "usage": {"cached_tokens": cached}},
            self.old, body or self.omitted, self.output)

    def test_full_reuse(self):
        self.assertEqual(self.check(110)["status"], "full")

    def test_omitted_reasoning_reuses_exact_boundary(self):
        result = self.check(95)
        self.assertEqual(result["status"], "client_omitted_reasoning_only_reply")
        self.assertEqual(result["omitted_generation_tokens"], 10)
        self.assertEqual(result["generation_header_tokens"], 5)

    def test_omission_does_not_excuse_lost_history(self):
        with self.assertRaisesRegex(AssertionError, "lost preserved history"):
            self.check(94)

    def test_preserved_assistant_must_be_fully_reused(self):
        body = {"messages": self.old["messages"] + [
            {"role": "assistant", "reasoning_content": "Done.", "content": ""},
            {"role": "user", "content": "next"}]}
        with self.assertRaisesRegex(AssertionError, "not fully cached"):
            self.check(95, body)

    def test_visible_reply_is_not_a_reasoning_only_omission(self):
        self.output.write_text("data: " + json.dumps(
            {"choices": [{"delta": {"content": "Done.", "reasoning_content": "Done."}}]}) + "\n\n")
        with self.assertRaisesRegex(AssertionError, "not fully cached"):
            self.check(95)

    def test_split_native_header_is_visible_in_both_reasoning_transports(self):
        parts = ["Read it.\n<tool_", "call>\n<function=read>"]
        for responses in (False, True):
            events = [
                {"type": "response.reasoning_summary_text.delta", "delta": part}
                if responses else {"choices": [{"delta": {"reasoning_content": part}}]}
                for part in parts]
            self.output.write_text("".join(
                "data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n")
            reasoning = []
            content, calls = sse_output(self.output, reasoning=reasoning)
            self.assertEqual(content, "")
            self.assertEqual(calls, [])
            self.assertIn("<tool_call>\n<function=", "".join(reasoning))


if __name__ == "__main__":
    unittest.main()
