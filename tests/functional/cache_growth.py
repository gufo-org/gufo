"""Check that cache reuse advances as a conversation grows."""

from copy import deepcopy
import sys


def check_cache_growth(client, model, checks, chat_result):
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
        retry = chat("unchanged", retry_request)
        assert work(retry) == (previous_total, previous_total, 0), retry
        assert signature(retry) == signature(history[-1][1]), retry

        # Measure the whole growing history before cold controls can supply
        # missing checkpoints and accidentally hide a frozen-cache failure.
        for turn, (body, warm) in enumerate(history):
            body["extra_body"]["cache_prompt"] = False
            cold = chat(f"cold_control_{turn}", body)
            total = warm["usage"]["prompt_tokens"]
            assert work(cold) == (total, 0, total), cold
            assert answer(warm) == answer(cold), (warm, cold)

    assert not failures, "\n".join(failures)
