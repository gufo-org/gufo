"""Check new conversations reuse the prefix they share with earlier ones.

Conversations arrive one after another with one system prompt and their own
task. The second one learns where they diverge and retains a checkpoint
there; later ones must restore the whole shared prefix, not only the nearest
grid checkpoint, and must answer as uncached controls do.
"""

from copy import deepcopy
import sys

from cache_concurrency import background, system_prompt

CODES = ("ALPHA", "BETA", "GAMMA", "DELTA")
# Tokens around the divergence point a restore may still miss: template
# framing and tokenizer merges where the user message starts.
BOUNDARY_ALLOWANCE = 96
# Below this gain the server keeps the nearest existing checkpoint instead of
# capturing another one (TextRunnerPool::Request::kSharedPrefixMinTokens).
LEARN_MIN_TOKENS = 512


def check_cache_shared_prefix(client, model, checks, chat_result):
    failures = []
    request = dict(model=model, temperature=0, seed=31, max_completion_tokens=16,
                   reasoning_effort="none")

    def chat(label, messages, cold=False):
        body = {**deepcopy(request), "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        result = chat_result(client, body)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        if cold:
            assert reused == 0, result
        return result

    def answer(result, code):
        assert result["text"].strip() == code and not result["tools"] \
            and result["finish"] == "stop", result

    def group(label, tail_lines, slack):
        system = system_prompt(f"cache_shared_prefix_{label}", 220)
        conversations = [[{"role": "system", "content": system},
                          {"role": "user", "content": background(code, tail_lines) +
                           f"Reply with only the code {code}."}]
                         for code in CODES]
        results = []
        for index, (messages, code) in enumerate(zip(conversations, CODES)):
            result = chat(f"{label}_{index}", messages)
            answer(result, code)
            results.append(result)
        # The shared prefix ends where the user message starts. An uncached
        # probe with a one-word user message measures it.
        stem = chat(f"{label}_stem", [conversations[0][0],
                                      {"role": "user", "content": "Z"}], cold=True)
        shared = stem["usage"]["prompt_tokens"] - BOUNDARY_ALLOWANCE - slack
        # The first conversation has nothing to share, and the second learns
        # the divergence point. From the third on it must be restored.
        for index in range(2, len(results)):
            cached = results[index]["usage"]["cached_tokens"]
            if cached < shared:
                failures.append(f"{label}_{index}: reused {cached} tokens, expected at least "
                                f"{shared} of the shared system prompt")
        for index, (messages, warm) in enumerate(zip(conversations, results)):
            cold = chat(f"{label}_cold_{index}", messages, cold=True)
            assert cold["usage"]["prompt_tokens"] == warm["usage"]["prompt_tokens"], (warm, cold)
            assert cold["text"].strip() == warm["text"].strip(), (warm, cold)

    # Long distinct tasks spread the grid checkpoints past the shared prefix:
    # only a learned boundary restores all of it.
    group("long_tasks", 700, 0)
    # Short tasks leave a grid checkpoint near the boundary; the server may
    # keep it when learning would gain less than LEARN_MIN_TOKENS.
    group("short_tasks", 4, LEARN_MIN_TOKENS)

    assert not failures, "\n".join(failures)
