"""Messages prompt counts, including tools, reasoning and cache-independent sizing."""

from copy import deepcopy
import sys

from server_metrics import COUNTERS, DEFERRED, PROCESSING, ServerMetrics


def assert_count_response(response):
    assert isinstance(response, dict) and set(response) == {"input_tokens"}, response
    count = response["input_tokens"]
    assert type(count) is int and count > 0, response
    return count


def assert_prompt_count(count, usage):
    fields = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    values = [usage.get(field, 0) for field in fields]
    assert all(type(value) is int and value >= 0 for value in values), usage
    assert count == sum(values), (count, usage)


def check_messages_count_tokens(client, model, checks, context):
    import openai

    metrics = ServerMetrics(client.base_url)

    def record(label, value):
        checks["messages_count_tokens_" + label] = value
        print(f"CHECK messages_count_tokens_{label}", file=sys.stderr, flush=True)
        return value

    def count(label, body):
        # Counting must not prefill, generate, schedule drafts or update cache
        # accounting, even when the same conversation is already cached.
        before = metrics.wait(lambda m: m[PROCESSING] == m[DEFERRED] == 0, "idle")
        response = client.post("/messages/count_tokens", body=body, cast_to=object)
        after = metrics.read()
        record(label, {"response": response,
                       "counter_deltas": {key: after[key] - before[key] for key in COUNTERS}})
        assert all(after[key] == before[key] for key in COUNTERS), (before, after)
        assert after[PROCESSING] == after[DEFERRED] == 0, after
        return assert_count_response(response)

    def generate(label, body, expected):
        response = client.post("/messages", body={**body, "max_tokens": 32,
                                                  "temperature": 0}, cast_to=object)
        record(label, response)
        assert_prompt_count(expected, response["usage"])
        assert response["usage"]["output_tokens"] > 0, response
        assert expected == (response["timings"]["prompt_n"]
                            + response["usage"]["cache_read_input_tokens"]), response
        return response

    plain = {"model": model, "thinking": {"type": "disabled"},
             "messages": [{"role": "user", "content": "Reply with exactly: COUNT_OK"}]}
    initial = count("plain", plain)  # max_tokens is deliberately absent.
    assert count("plain_repeat", plain) == initial
    cold = generate("plain_cold", plain, initial)
    assert cold["usage"]["cache_read_input_tokens"] == 0, cold
    assert count("plain_warm", plain) == initial
    warm = generate("plain_cached", plain, initial)
    assert warm["usage"]["cache_read_input_tokens"] > 0, warm
    assert warm["content"] == cold["content"], (cold, warm)
    assert count("generation_options", {**plain, "max_tokens": 128, "temperature": .7,
                                       "stop_sequences": ["never_stop_here"],
                                       "stream": False}) == initial

    system = {"type": "text", "text": "Be concise. Preserve café, 東京 and 🦉."}
    history = [{"role": "user", "content": [{"type": "text", "text": "Remember: café."}]},
               {"role": "assistant", "content": [{"type": "text", "text": "Remembered."}]},
               {"role": "user", "content": "Repeat the remembered word."}]
    cases = [
        ("system_history", {**plain, "system": [system], "messages": history}),
        ("thinking", {**plain, "thinking": {"type": "enabled", "budget_tokens": 16}}),
        ("adaptive", {**plain, "thinking": {"type": "adaptive"},
                      "output_config": {"effort": "high"}}),
    ]
    tools = [{"name": "remember", "description": "Remember one word.",
              "input_schema": {"type": "object",
                               "properties": {"word": {"type": "string", "enum": ["café"]}},
                               "required": ["word"], "additionalProperties": False}}]
    tool_request = {**plain, "system": [system], "tools": tools,
                    "messages": [{"role": "user", "content": "Use remember for café."}]}
    for choice in ({"type": "none"}, {"type": "auto"}, {"type": "any"},
                   {"type": "tool", "name": "remember"}):
        cases.append(("tools_" + choice["type"], {**tool_request, "tool_choice": choice}))
    cases.append(("thinking_tool_history", {
        **tool_request, "thinking": {"type": "enabled", "budget_tokens": 16},
        "messages": [tool_request["messages"][0],
                     {"role": "assistant", "content": [
                         {"type": "thinking", "thinking": "I should remember the word."},
                         {"type": "tool_use", "id": "call_count_1",
                          "name": "remember", "input": {"word": "café"}}]},
                     {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": "call_count_1",
                          "content": [{"type": "text", "text": "Remembered café."}]}]}]}))
    for label, body in cases:
        expected = count(label, deepcopy(body))
        generate(label + "_generated", body, expected)
        if label == "system_history":
            assert count("system_string", {**body, "system": system["text"]}) == expected
        if label == "thinking_tool_history":
            without_thinking = deepcopy(body)
            without_thinking["messages"][1]["content"].pop(0)
            assert count("tool_history_without_thinking", without_thinking) < expected

    # A sizing endpoint must allow callers to discover that trimming is needed.
    # llama.cpp renders/tokenizes here without admission to the execution context.
    oversized = {**plain, "messages": [{"role": "user", "content": " counting" * (context * 2)}]}
    assert count("over_context", oversized) > context
    try:
        client.post("/messages", body={**oversized, "max_tokens": 1}, cast_to=object)
    except openai.APIStatusError as error:
        record("over_context_generation", {"status": error.status_code, "body": error.body})
        assert error.status_code == 400, error
    else:
        raise AssertionError("generation accepted a prompt beyond its context")

    for label, changes, status in (
        ("empty", {"messages": []}, 400),
        ("thinking_invalid", {"thinking": True}, 400),
        ("stream", {"stream": True}, 400),
        ("wrong_model", {"model": model + "-not-loaded"}, 404),
    ):
        try:
            client.post("/messages/count_tokens", body={**plain, **changes}, cast_to=object)
        except openai.APIStatusError as error:
            record(label, {"status": error.status_code, "body": error.body})
            assert error.status_code == status, error
        else:
            raise AssertionError(f"{label}: invalid request was accepted")
    assert count("recovery", plain) == initial
    recovered = generate("recovery_generated", plain, initial)
    assert recovered["content"] == cold["content"], (cold, recovered)
