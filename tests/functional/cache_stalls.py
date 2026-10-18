"""Ordered cold/partial/full cache histories for Flash-Next stall investigation."""

import sys

from cache_growth import check_unchanged_retry, log_offset


# #632 supplies these padding sizes and request order, but not its prompts.
# These public equivalents must not be described as the reporter's exact test.
PADDING_WORDS = (6400, 25600, 55600, 56000)
TASKS = {
    "code": "\nWrite a Python function that merges two sorted lists. Explain the algorithm "
            "and include tests for empty lists and duplicate values.",
    "agentic": "\nPlan how to inspect a repository, locate a failing cache test, fix the root "
               "cause, run focused tests, and report the change.",
}


def check_cache_stalls(client, model, checks, chat_result, concurrency, context,
                       server_log=None):
    assert concurrency >= 2 and context >= 65536, \
        "cache-stalls needs at least two sessions and 65536 context tokens"
    previous_words = 0
    for words in PADDING_WORDS:
        for kind, task in TASKS.items():
            request = {
                "model": model, "temperature": 0, "seed": 31,
                "max_completion_tokens": 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                "messages": [{"role": "user", "content": "context " * words + task}],
            }
            previous = None
            previous_start = None
            for repeat in range(3):
                label = f"cache_stalls_{words}_{kind}_{repeat + 1}"
                start = log_offset(server_log)
                # The reported workload is nonstreaming. The normal functional
                # recorder still checks server PP/TG, TTFT and maximum token gap
                # individually against a matched baseline, without averaging.
                result = chat_result(client, request)
                checks[label] = result
                print(f"CHECK {label}", file=sys.stderr, flush=True)
                usage = result["usage"]
                total, cached, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                            usage["gufo"]["prefill_tokens"])
                assert all(type(n) is int and n >= 0 for n in (total, cached, prefilled))
                assert cached + prefilled == total and words <= total < words + 256, result
                assert result["reasoning"].strip() and not result["tools"], result
                assert result["finish"] in ("stop", "length") \
                    and 0 < usage["completion_tokens"] <= 128, result
                if previous is None:
                    if previous_words:
                        assert cached >= min(words, previous_words) - 2048, result
                    else:
                        assert cached == 0, result
                else:
                    assert total == previous["usage"]["prompt_tokens"], result
                    check_unchanged_retry(label, result, prefilled, server_log,
                                          previous_start, start)
                    if prefilled == 0:
                        def signature(row):
                            return (row["text"], row["reasoning"], row["tools"], row["finish"],
                                    row["usage"]["completion_tokens"])
                        assert signature(result) == signature(previous), (previous, result)
                previous, previous_start = result, start
            previous_words = words
