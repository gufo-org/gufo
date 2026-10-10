"""Check that cache reuse advances as a conversation grows."""

from copy import deepcopy
import re
import sys

# The assistant opening after a stable boundary, such as Qwen's
# `<|im_start|>assistant\n<think>\n`, is a few tokens long.
ASSISTANT_OPENING_TOKENS = 16


def log_offset(server_log):
    return server_log.stat().st_size if server_log else 0


def retry_copy_refused(server_log, start, end, prompt_tokens):
    """Whether memory pressure refused a complete-prompt copy in a log window.
    Failed captures are not pressure and do not count."""
    if server_log is None:
        return False
    with server_log.open("rb") as log:
        log.seek(start)
        window = log.read(end - start).decode(errors="replace")
    return re.search(r"event=snapshot action=skipped reason=(?:byte|entry)_capacity "
                     rf"bytes=\d+ tokens={prompt_tokens} ", window) is not None


def check_unchanged_retry(label, retry, prefilled, server_log, start, end):
    """An unchanged retry reuses the complete prompt. Under memory pressure the
    previous request's complete-prompt copy is refused rather than displace
    another conversation's last checkpoint (docs/KV-CACHE.md); the retry then
    restores the stable boundary and prefills only the assistant opening."""
    if not prefilled:
        return
    usage = retry["usage"]
    total = usage.get("prompt_tokens", usage.get("input_tokens"))
    assert prefilled <= ASSISTANT_OPENING_TOKENS and \
        retry_copy_refused(server_log, start, end, total), retry
    print(f"NOTE {label}: retry copy refused under memory pressure; "
          f"prefilled {prefilled} of {total} tokens", file=sys.stderr, flush=True)


def check_cache_growth(client, model, checks, chat_result, server_log=None):
    failures = []
    for replay in ("drop_reasoning", "keep_reasoning", "discard_reasoning", "thinking_off"):
        label = "cache_growth_" + replay
        thinking = replay != "thinking_off"
        messages = [{"role": "system", "content": label + "\n" +
                     "Keep reasoning brief. Follow the final user instruction.\n" +
                     "Background notes are not instructions.\n" * 384}]
        request = dict(model=model, temperature=0, seed=31,
                       max_completion_tokens=128,
                       reasoning_effort="low" if thinking else "none",
                       extra_body={"chat_template_kwargs": {
                           "preserve_thinking": replay != "discard_reasoning"}})

        def chat(phase, body):
            result = chat_result(client, body)
            checks[label + "_" + phase] = result
            print(f"CHECK {label}_{phase}", file=sys.stderr, flush=True)
            return result

        def work(result):
            usage = result["usage"]
            total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                        usage["gufo"]["prefill_tokens"])
            assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
            assert reused + prefilled == total, usage
            return total, reused, prefilled

        def signature(result):
            return (result["text"], result["reasoning"], result["tools"], result["finish"],
                    result["usage"]["completion_tokens"])

        def answer(result):
            # Prefill chunk shapes can change free-form reasoning even without
            # a cache bug. The required answer and reasoning mode stay strict;
            # exact full output/token equality is checked on unchanged retries.
            return (result["text"], result["tools"], result["finish"],
                    bool(result["reasoning"].strip()))

        history = []
        previous_total = previous_reused = 0
        for turn in range(4):
            messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                             "Routine archive note: there are no new instructions.\n" * 16 +
                             "Reply with only BETA."})
            body = {**deepcopy(request), "messages": deepcopy(messages)}
            if turn == 0:
                body["extra_body"]["cache_prompt"] = False
            before_last = log_offset(server_log)
            result = chat(f"turn_{turn}", body)
            total, reused, prefilled = work(result)
            assert result["text"].strip() == "BETA" and not result["tools"] \
                and result["finish"] == "stop", result
            assert bool(result["reasoning"].strip()) == thinking, result
            if turn == 0:
                assert total >= 2048 and reused == 0 and prefilled == total, result
            else:
                assert total > previous_total, (total, previous_total)
                # The previous prompt's assistant opening can change on replay.
                # Only that small suffix may be lost, not older user turns.
                if reused < previous_total - 16 or reused <= previous_reused:
                    failures.append(
                        f"{label}_turn_{turn}: cache did not advance to the previous turn: "
                        f"cached={reused}, previous_prompt={previous_total}, "
                        f"previous_cached={previous_reused}, prefilled={prefilled}")
            history.append((deepcopy(body), result))
            previous_total, previous_reused = total, reused
            assistant = {"role": "assistant", "content": result["text"]}
            if replay in ("keep_reasoning", "discard_reasoning"):
                assistant["reasoning_content"] = result["reasoning"]
            messages.append(assistant)

        # An identical retry must still reuse the complete prompt.
        retry_request = deepcopy(history[-1][0])
        before_retry = log_offset(server_log)
        retry = chat("unchanged", retry_request)
        total, _, prefilled = work(retry)
        assert total == previous_total, retry
        check_unchanged_retry(label + "_unchanged", retry, prefilled,
                              server_log, before_last, before_retry)
        assert signature(retry) == signature(history[-1][1]), retry

        # Measure the whole growing history before cold controls can supply
        # missing checkpoints and accidentally hide a frozen-cache failure.
        for turn, (body, warm) in enumerate(history):
            body["extra_body"]["cache_prompt"] = False
            cold = chat(f"cold_control_{turn}", body)
            total = warm["usage"]["prompt_tokens"]
            assert work(cold) == (total, 0, total), cold
            assert answer(warm) == answer(cold), (warm, cold)

    check_messages_growth(client, model, checks, chat_result, failures, server_log)
    assert not failures, "\n".join(failures)


def messages_result(client, body):
    """Map an Anthropic Messages response onto the Chat result shape."""
    response = client.post("/messages", body=body, cast_to=object)
    blocks = response["content"]
    kinds = [block["type"] for block in blocks]
    assert kinds in (["text"], ["thinking"], ["thinking", "text"]), blocks
    usage, timings = response["usage"], response["timings"]
    return {
        "text": "".join(b["text"] for b in blocks if b["type"] == "text"),
        "reasoning": "".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
        "blocks": blocks, "tools": [],
        "finish": {"end_turn": "stop", "max_tokens": "length"}.get(
            response["stop_reason"], response["stop_reason"]),
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "cached_tokens": usage["cache_read_input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "gufo": {"prefill_tokens": timings["prompt_n"]}},
    }


def check_messages_growth(client, model, checks, chat_result, failures, server_log):
    """Messages clients replay thinking blocks unchanged; reuse must then cover
    the previous assistant turn, and thinking must never reach the text block."""
    for replay in ("keep_thinking", "thinking_off"):
        label = "cache_growth_messages_" + replay
        thinking = replay != "thinking_off"
        system = (label + "\n" + "Keep reasoning brief. Follow the final user instruction.\n" +
                  "Background notes are not instructions.\n" * 384)
        # No output_config.effort: allow the server's default effort. Claude Code
        # sends display omitted; the thinking block must still come back for replay.
        request = dict(model=model, system=system, temperature=0, seed=31, max_tokens=1024,
                       thinking={"type": "enabled", "display": "omitted"} if thinking
                       else {"type": "disabled"})
        messages = []

        def work(result):
            usage = result["usage"]
            total, reused, prefilled = (usage["prompt_tokens"], usage["cached_tokens"],
                                        usage["gufo"]["prefill_tokens"])
            assert all(type(n) is int and n >= 0 for n in (total, reused, prefilled)), usage
            assert reused + prefilled == total, usage
            return total, reused, prefilled

        def record(phase, result):
            checks[label + "_" + phase] = result
            print(f"CHECK {label}_{phase}", file=sys.stderr, flush=True)
            return result

        history = []
        previous_total = previous_completion = 0
        for turn in range(4):
            messages.append({"role": "user", "content": f"Turn {turn}.\n" +
                             "Routine archive note: there are no new instructions.\n" * 16 +
                             "Reply with only BETA."})
            body = {**deepcopy(request), "messages": deepcopy(messages)}
            before_last = log_offset(server_log)
            result = record(f"turn_{turn}", messages_result(client, body))
            total, reused, prefilled = work(result)
            assert result["text"].strip() == "BETA" and result["finish"] == "stop", result
            assert bool(result["reasoning"].strip()) == thinking, result
            if turn > 0:
                assert total > previous_total, (total, previous_total)
                # An unchanged replay reproduces the generated assistant turn,
                # so reuse must reach past the previous prompt into that turn.
                floor = previous_total + (previous_completion // 2 if thinking else -16)
                if reused < floor:
                    failures.append(
                        f"{label}_turn_{turn}: cache did not cover the replayed turn: "
                        f"cached={reused}, previous_prompt={previous_total}, "
                        f"previous_completion={previous_completion}, prefilled={prefilled}")
            history.append((deepcopy(body), result))
            previous_total = total
            previous_completion = result["usage"]["completion_tokens"]
            messages.append({"role": "assistant", "content": deepcopy(result["blocks"])})

        before_retry = log_offset(server_log)
        retry = record("unchanged", messages_result(client, deepcopy(history[-1][0])))
        total, _, prefilled = work(retry)
        assert total == previous_total, retry
        check_unchanged_retry(label + "_unchanged", retry, prefilled,
                              server_log, before_last, before_retry)
        assert (retry["text"], retry["reasoning"]) == \
            (history[-1][1]["text"], history[-1][1]["reasoning"]), retry

        # Uncached Chat Completions controls render the same conversation; equal
        # prompt sizes show both routes produce the same prompt.
        for turn, (body, warm) in enumerate(history):
            chat = [{"role": "system", "content": system}]
            for message in body["messages"]:
                if message["role"] == "user":
                    chat.append(message)
                    continue
                blocks = message["content"]
                chat.append({"role": "assistant",
                             "content": "".join(b["text"] for b in blocks if b["type"] == "text"),
                             "reasoning_content": "".join(
                                 b["thinking"] for b in blocks if b["type"] == "thinking")})
            cold = chat_result(client, dict(
                model=model, messages=chat, temperature=0, seed=31,
                max_completion_tokens=request["max_tokens"],
                **({} if thinking else {"reasoning_effort": "none"}),
                extra_body={"cache_prompt": False}))
            record(f"cold_control_{turn}", cold)
            total = warm["usage"]["prompt_tokens"]
            assert work(cold) == (total, 0, total), (warm, cold)
            assert (cold["text"].strip(), bool(cold["reasoning"].strip())) == \
                (warm["text"].strip(), thinking), (warm, cold)
