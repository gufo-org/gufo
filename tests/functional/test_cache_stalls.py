"""The stall workload must preserve ordering and reject incorrect cache work."""

from contextlib import redirect_stderr
import io
import unittest

from cache_stalls import PADDING_WORDS, TASKS, check_cache_stalls


class CacheStallsTests(unittest.TestCase):
    def run_fixture(self, defect=None):
        requests, checks = [], {}
        previous = None
        previous_words = 0

        def chat_result(client, body):
            nonlocal previous, previous_words
            text = body["messages"][0]["content"]
            words = text.count("context ")
            total = words + 76
            retry = text == previous
            cached = total if retry else max(0, min(words, previous_words) - 2048)
            if defect == "lost_prefix" and previous and not retry:
                cached = 0
            if defect == "extra_prefill" and retry:
                cached -= 1
            reasoning = "Reasoning." if not (retry and defect == "changed_output") else "Changed."
            requests.append(body)
            previous, previous_words = text, words
            return {"text": "", "reasoning": reasoning, "tools": [], "finish": "length",
                    "usage": {"prompt_tokens": total, "cached_tokens": cached,
                              "completion_tokens": 128,
                              "gufo": {"prefill_tokens": total - cached}}}

        with redirect_stderr(io.StringIO()):
            check_cache_stalls(None, "fixture", checks, chat_result, 2, 262144)
        return requests, checks

    def test_ordered_histories_and_fully_cached_repeats(self):
        requests, checks = self.run_fixture()
        expected = [words for words in PADDING_WORDS for _ in TASKS for _ in range(3)]
        self.assertEqual([r["messages"][0]["content"].count("context ") for r in requests],
                         expected)
        self.assertEqual(len(checks), 24)
        for i in range(0, len(requests), 3):
            self.assertEqual(requests[i:i + 3], [requests[i]] * 3)
        self.assertTrue(all(r["temperature"] == 0 and r["max_completion_tokens"] == 128
                            for r in requests))

    def test_rejects_lost_reuse_changed_replay_and_unexplained_prefill(self):
        for defect in ("lost_prefix", "changed_output", "extra_prefill"):
            with self.subTest(defect=defect), self.assertRaises(AssertionError):
                self.run_fixture(defect)


if __name__ == "__main__":
    unittest.main()
