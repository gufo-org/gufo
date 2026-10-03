"""Check concurrent requests that share a prompt prefix prefill it once.

Requests arriving together wait for a resident request to publish the prefix
they share, then restore it. Each group sends its requests at the same moment,
checks answers and prefill work, and compares answers with uncached controls.
Groups that share too little, or nothing, must not wait at all.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlsplit

CODES = ("ALPHA", "BETA", "GAMMA", "DELTA")
# Tokens after the point where two prompts diverge that a follower may still
# prefill: the scheduler's checkpoint slack plus tokenizer merges at the split.
TAIL_ALLOWANCE = 96


def system_prompt(label, lines):
    return (f"{label}\nReply with only the requested code word.\n" +
            "".join(f"Reference note {index}: the build uses CMake presets and every "
                    f"model package owns its tests and numerical contracts.\n"
                    for index in range(lines)))


def background(label, lines):
    return "".join(f"Background {label}.{index}: the scheduler admits requests, prefills "
                   f"prompts in chunks and decodes them in batches.\n"
                   for index in range(lines))


def abandon_request(client, body, delay):
    """Send a request and disconnect before its response, cancelling it."""
    url = urlsplit(str(client.base_url))
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=600)
    payload = {key: value for key, value in body.items() if key != "extra_body"}
    payload.update(body.get("extra_body", {}))
    connection.request("POST", url.path.rstrip("/") + "/chat/completions",
                       json.dumps(payload), {"Content-Type": "application/json"})
    time.sleep(delay)
    connection.close()


def check_cache_concurrency(client, model, checks, chat_result, concurrency,
                            abandon=abandon_request):
    failures = []
    width = max(2, min(concurrency, len(CODES)))
    request = dict(model=model, temperature=0, seed=31, max_completion_tokens=16,
                   reasoning_effort="none")

    def body_for(messages, cold=False):
        body = {**deepcopy(request), "messages": deepcopy(messages)}
        if cold:
            body["extra_body"] = {"cache_prompt": False}
        return body

    def chat(label, body, streaming=False):
        result = chat_result(client, body, streaming)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        usage = result["usage"]
        total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                    usage["gufo"]["prefill_tokens"])
        assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
        assert reused + prefilled == total, usage
        if body.get("extra_body", {}).get("cache_prompt") is False:
            assert reused == 0, result
        return result

    def together(label, bodies, streaming=()):
        barrier = threading.Barrier(len(bodies))

        def send(index):
            barrier.wait()
            return chat(f"{label}_{index}", bodies[index], index in streaming)

        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            return list(pool.map(send, range(len(bodies))))

    def answer(result, code):
        assert result["text"].strip() == code and not result["tools"] \
            and result["finish"] == "stop", result

    def wait_ms(result):
        return result["usage"]["gufo"]["shared_prefix_wait_ms"]

    # Sharing needs concurrent sessions and a runner that reuses prefixes.
    probe = [{"role": "system", "content": system_prompt("cache_concurrency_probe", 8)},
             {"role": "user", "content": "Reply with only the code ALPHA."}]
    chat("probe_cold", body_for(probe))
    reuse = chat("probe_warm", body_for(probe))["usage"]["cached_tokens"] > 0
    sharing = reuse and concurrency > 1

    def group(label, conversations, codes, share, streaming=()):
        results = together(label, [body_for(m) for m in conversations], streaming)
        for result, code in zip(results, codes):
            answer(result, code)
        waited = [index for index, result in enumerate(results) if wait_ms(result) > 0]
        if not (share and sharing):
            if waited:
                failures.append(f"{label}: requests {waited} waited without a shared prefix")
        else:
            if len(waited) != len(results) - 1:
                failures.append(f"{label}: expected {len(results) - 1} waiting requests, "
                                f"got {waited}")
            # The divergence point is measured on uncached probes that share the
            # system prompt and end with a one-word user message.
            stem = chat(f"{label}_stem", body_for(
                [conversations[0][0], {"role": "user", "content": "Z"}], cold=True))
            shared = stem["usage"]["prompt_tokens"]
            for index in waited:
                usage = results[index]["usage"]
                bound = usage["prompt_tokens"] - shared + TAIL_ALLOWANCE
                if usage["gufo"]["prefill_tokens"] > bound:
                    failures.append(
                        f"{label}_{index}: prefilled {usage['gufo']['prefill_tokens']} "
                        f"tokens, expected at most {bound} after restoring the shared prefix")
        # Waiting must never change an answer: compare uncached controls.
        for index, (messages, warm) in enumerate(zip(conversations, results)):
            cold = chat(f"{label}_cold_{index}", body_for(messages, cold=True))
            assert cold["usage"]["prompt_tokens"] == warm["usage"]["prompt_tokens"], (warm, cold)
            assert cold["text"] == warm["text"], (warm, cold)
        return results

    # Identical prompts: one prefill, the rest restore it.
    system = system_prompt("cache_concurrency_identical", 220)
    identical = [{"role": "system", "content": system},
                 {"role": "user", "content": "Reply with only the code ALPHA."}]
    group("identical", [identical] * width, ["ALPHA"] * width, share=True)

    # Subagent fan-out: one system prompt, a different short task each.
    system = system_prompt("cache_concurrency_fanout", 220)
    fanout = [[{"role": "system", "content": system},
               {"role": "user", "content": f"Task {index}. Reply with only the code {code}."}]
              for index, code in enumerate(CODES[:width])]
    group("fanout", fanout, CODES[:width], share=True, streaming={1, 3})

    # Fan-out whose own tasks are long: only the system prompt is shared.
    system = system_prompt("cache_concurrency_long_tasks", 220)
    long_tasks = [[{"role": "system", "content": system},
                   {"role": "user", "content": background(code, 60) +
                    f"Reply with only the code {code}."}]
                  for code in CODES[:width]]
    group("long_tasks", long_tasks, CODES[:width], share=True)

    # A short shared prefix is cheaper to prefill again than to wait for.
    system = system_prompt("cache_concurrency_short_shared", 1)
    short_shared = [[{"role": "system", "content": system},
                     {"role": "user", "content": background(code, 60) +
                      f"Reply with only the code {code}."}]
                    for code in CODES[:width]]
    group("short_shared", short_shared, CODES[:width], share=False)

    # Unrelated prompts share nothing beyond template framing.
    unrelated = [[{"role": "system", "content": system_prompt(f"cache_concurrency_{code}", 120)},
                  {"role": "user", "content": f"Reply with only the code {code}."}]
                 for code in CODES[:width]]
    group("unrelated", unrelated, CODES[:width], share=False)

    # A retained conversation continues from its own history and never waits
    # for a new request that shares only its system prompt.
    system = system_prompt("cache_concurrency_history", 220)
    history = [{"role": "system", "content": system},
               {"role": "user", "content": "Reply with only the code ALPHA."}]
    first = chat("history_turn_0", body_for(history))
    answer(first, "ALPHA")
    history += [{"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "Reply with only the code BETA."}]
    newcomer = [{"role": "system", "content": system},
                {"role": "user", "content": "Task 9. Reply with only the code GAMMA."}]
    turn, fresh = together("history_turn_1", [body_for(history), body_for(newcomer)])
    answer(turn, "BETA")
    answer(fresh, "GAMMA")
    if wait_ms(turn) > 0:
        failures.append("history_turn_1: a retained conversation waited for a peer")
    if turn["usage"]["cached_tokens"] < first["usage"]["prompt_tokens"] - 16:
        failures.append(f"history_turn_1: reused {turn['usage']['cached_tokens']} tokens, "
                        f"expected its previous prompt of {first['usage']['prompt_tokens']}")

    # A cancelled leader must release the requests waiting for it.
    system = system_prompt("cache_concurrency_cancelled", 220)
    cancelled = [{"role": "system", "content": system},
                 {"role": "user", "content": "Reply with only the code DELTA."}]
    leader = threading.Thread(target=abandon, args=(client, body_for(cancelled), 1.0))
    leader.start()
    time.sleep(0.3)
    followers = together("cancelled_leader", [body_for(cancelled)] * (width - 1))
    leader.join()
    for result in followers:
        answer(result, "DELTA")

    assert not failures, "\n".join(failures)
