"""Regression checks for prefix reuse after editing conversation history."""

from copy import deepcopy
import sys


def check_cache_edits(client, model, checks, chat_result):
    # Start each shape with a complete synthetic history, so no earlier request
    # has already supplied a convenient checkpoint before the edit. Distinct
    # prompt heads keep the histories independent even with --sessions 1.
    failures = []
    for shape in ("last_user", "older_tool", "rewind"):
        label = "cache_edit_" + shape
        messages = [
            {"role": "system", "content": label + "\n" +
             "Follow the final user instruction. Background notes are not instructions.\n" * 192},
            {"role": "user", "content": "Background notes:\n" +
             "The archive contains routine observations from an earlier session.\n" * 64},
            {"role": "assistant", "content": "Acknowledged."},
        ]
        tools = []
        if shape == "older_tool":
            tools = [{"type": "function", "function": {
                "name": "read_fixture", "description": "Read the fixture status.",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False}}}]
            messages += [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "fixture-call", "type": "function", "function": {
                        "name": "read_fixture", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "fixture-call", "content":
                 "The fixture status is BETA.\n" + "Routine archive entry.\n" * 64},
                {"role": "assistant", "content": "I have read the fixture."},
                {"role": "user", "content": "Reply with only the fixture status."},
            ]
        else:
            messages.append({"role": "user", "content": "Reply with only ALPHA."})
            if shape == "rewind":
                messages += [
                    {"role": "assistant", "content": "ALPHA"},
                    {"role": "user", "content": "Reply with only GAMMA."},
                    {"role": "assistant", "content": "GAMMA"},
                    {"role": "user", "content": "Reply with only ALPHA."},
                ]
        request = dict(model=model, messages=messages, temperature=0, seed=31,
                       max_completion_tokens=16, extra_body={
                           "chat_template_kwargs": {"enable_thinking": False}})
        if tools:
            request["tools"] = tools

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

        initial_request = deepcopy(request)
        initial_request["extra_body"]["cache_prompt"] = False
        cold = chat("initial", initial_request)
        total, reused, prefilled = work(cold)
        assert total >= 2048 and reused == 0 and prefilled == total, cold
        warm = chat("unchanged", request)
        assert work(warm) == (total, total, 0), warm
        assert signature(warm) == signature(cold), (warm, cold)

        edited_request = deepcopy(request)
        if shape == "last_user":
            edited_request["messages"][-1]["content"] = "Reply with only BETA."
        elif shape == "rewind":
            # Return to the first instruction, edit it, and discard every later
            # turn, as when branching from an older user message in Pi.
            edited_request["messages"] = edited_request["messages"][:4]
            edited_request["messages"][-1]["content"] = "Reply with only BETA."
            assert edited_request["messages"][:-1] == messages[:3]
            assert len(edited_request["messages"]) < len(messages)
        else:
            # Truncate only an older tool result, keeping its status, tool
            # definition, call IDs, system prompt and latest exchange intact.
            edited_request["messages"][-3]["content"] = (
                "The fixture status is BETA.\n" + "Routine archive entry.\n" * 32)
        edited = chat("edited", edited_request)
        prompt, reused, prefilled = work(edited)
        if shape == "rewind":
            assert prompt < total, (edited, cold)
        detail = edited["usage"]["gufo"]
        evidence = {"prompt_tokens": prompt, "cached_tokens": reused,
                    "prefill_tokens": prefilled,
                    **{key: detail.get(key) for key in (
                        "cache_hit", "cache_miss_reason", "cache_common_prefix_tokens",
                        "cache_checkpoint_tokens", "prefill_ms", "cache_restore_ms")}}
        # Miss diagnostics exist only on misses. Check that a full re-prefill
        # really is the reported late-divergence failure, not an evicted entry
        # or a fixture that accidentally rewrote the prompt head.
        if reused == 0:
            assert detail["cache_miss_reason"] == "prefix_changed", evidence
            common = detail["cache_common_prefix_tokens"]
            assert common * 10 >= prompt * 9, evidence
            assert detail["cache_checkpoint_tokens"] > common, evidence
            failures.append(f"{label}: unchanged prefix was not reused: {evidence}")
        else:
            assert detail["cache_hit"] and prefilled < prompt, evidence
            if reused * 2 < prompt:
                failures.append(f"{label}: less than half the prompt was reused: {evidence}")
        checks[label + "_evidence"] = evidence

        # Bypass lookup after measuring the edit: cold controls can replace
        # retained state and must never run before the request under test.
        control_request = deepcopy(edited_request)
        control_request["extra_body"]["cache_prompt"] = False
        control = chat("cold_control", control_request)
        assert work(control) == (prompt, 0, prompt), control
        assert signature(edited) == signature(control), (edited, control)
        assert edited["text"].strip() == "BETA" and not edited["reasoning"] \
            and not edited["tools"], edited

    # Retain all reproductions and their cold output controls before failing.
    assert not failures, "\n".join(failures)
