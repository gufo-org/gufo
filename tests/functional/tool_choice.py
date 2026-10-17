"""Tool choice changes constraints without invalidating prompt history (#619)."""

import json
import sys

from cache_growth import check_unchanged_retry, log_offset
from tool_native import assert_native, cached


def check_tool_choice_cache(client, model, checks, chat_result, server_log=None):
    from messages_count_tokens import assert_count_response
    function = {
        "name": "read", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}
    common = dict(model=model, temperature=0, seed=619, reasoning_effort="none",
                  parallel_tool_calls=False, max_completion_tokens=128,
                  tools=[{"type": "function", "function": function}],
                  extra_body={"cache_prompt": True})

    def record(label, request, streaming=False):
        result = chat_result(client, request, streaming)
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        assert_native(result, label)
        return result

    # Separate system prefixes prevent another case from providing the cache
    # hit. Controls also retain the unchanged-choice behavior and timings.
    for index, (before, after) in enumerate((
            ("auto", "auto"), ("required", "required"),
            ("required", None), (None, "required"),
            ("required", "auto"), ("auto", "required"))):
        label = f"choice_{before or 'omitted'}_{after or 'omitted'}"
        messages = [
            {"role": "system", "content": f"Tool choice regression case {index}."},
            {"role": "user", "content": "Call read once with path fixture.txt."}]
        request = {**common, "messages": messages}
        if before is not None:
            request["tool_choice"] = before
        first = record(label + "_first", request, index % 2 == 0)
        assert first["finish"] == "tool_calls" and len(first["tools"]) == 1, first
        call = first["tools"][0]
        assert call["function"]["name"] == "read", first
        assert json.loads(call["function"]["arguments"]) == {"path": "fixture.txt"}, first
        continued = {**common, "messages": [
            *messages,
            {"role": "assistant", "content": first["text"] or None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "The file says DONE."},
            {"role": "user", "content": (
                "Call read once more with path verified.txt." if after == "required"
                else "Reply with only DONE. Do not call a tool.")}]}
        if after is not None:
            continued["tool_choice"] = after
        # Clients may replace the final user turn of an agent history. Count
        # that turn and its assistant opening in a separate request, as
        # tool-history does.
        suffix = client.post("/messages/count_tokens", body={
            "model": model, "thinking": {"type": "disabled"},
            "messages": [continued["messages"][-1]]}, cast_to=object)
        checks[label + "_mutable_suffix_count"] = suffix
        mutable_suffix_tokens = assert_count_response(suffix)
        start = log_offset(server_log)
        done = record(label + "_continued", continued, index % 2 != 0)
        end = log_offset(server_log)
        expected = first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"]
        assert cached(done["usage"]) >= expected, (
            label, "tool choice lost the generated frontier", expected, done)
        if after == "required":
            assert done["finish"] == "tool_calls" and len(done["tools"]) == 1, done
            assert done["tools"][0]["function"]["name"] == "read", done
            assert json.loads(done["tools"][0]["function"]["arguments"]) == {
                "path": "verified.txt"}, done
        else:
            assert done["finish"] == "stop" and not done["tools"], done
            assert done["text"].strip() == "DONE", done

        # An identical prompt under another tool choice must have identical
        # token counts and reuse its completed prompt without any prefill. If
        # the server logged that memory pressure refused the prompt copy, the
        # retry may prefill the final user turn and its assistant opening.
        retry = record(label + "_retry", {**continued, "tool_choice": "auto"})
        assert retry["usage"]["prompt_tokens"] == done["usage"]["prompt_tokens"], retry
        prompt_tokens = retry["usage"]["prompt_tokens"]
        check_unchanged_retry(label + "_retry", retry, prompt_tokens - cached(retry["usage"]),
                              server_log, start, end,
                              mutable_suffix_tokens=mutable_suffix_tokens)

    # Required calls remain mandatory even when the user asks for plain text.
    # The grammar must supply that guarantee without a system instruction.
    forced = record("choice_required_overrides_prose", {
        **common, "tool_choice": "required", "messages": [
            {"role": "user", "content": "Reply DONE without calling any tool."}]})
    assert forced["finish"] == "tool_calls" and forced["tools"], forced
    for call in forced["tools"]:
        assert call["function"]["name"] == "read", forced
        arguments = json.loads(call["function"]["arguments"])
        assert isinstance(arguments.get("path"), str), forced
