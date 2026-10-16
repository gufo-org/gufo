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


def check_messages_image_counts(client, model, checks, context, image_content,
                                input_modalities):
    """Image counts must agree with generation and with model capabilities."""
    import openai
    from discovery import assert_model_listing
    from image_inputs import JPEG, WEBP_LOSSLESS, messages_color_requests
    from messages_tools import messages_stream

    metrics = ServerMetrics(client.base_url)

    def record(label, value):
        checks["messages_image_counts_" + label] = value
        print(f"CHECK messages_image_counts_{label}", file=sys.stderr, flush=True)
        return value

    listing = client.models.list().to_dict()
    assert_model_listing(listing, model, context, input_modalities)
    record("capabilities", listing)

    def idle():
        return metrics.wait(lambda m: m[PROCESSING] == m[DEFERRED] == 0, "idle")

    def unchanged(before):
        after = metrics.read()
        assert all(after[key] == before[key] for key in COUNTERS), (before, after)
        assert after[PROCESSING] == after[DEFERRED] == 0, after
        return {key: after[key] - before[key] for key in COUNTERS}

    def count(label, body):
        before = idle()
        response = client.post("/messages/count_tokens", body=body, cast_to=object)
        record(label, {"response": response, "counter_deltas": unchanged(before)})
        return assert_count_response(response)

    def reject(label, endpoint, body):
        before = idle()
        try:
            client.post(endpoint, body=body, cast_to=object)
        except openai.APIStatusError as error:
            record(label, {"status": error.status_code, "body": error.body,
                           "counter_deltas": unchanged(before)})
            assert error.status_code == 400, error
        else:
            raise AssertionError(f"{label}: invalid image request was accepted")

    def generate(label, body, expected, color, streaming=False):
        if streaming:
            response = messages_stream(client, body)
        else:
            raw = client.post("/messages", body=body, cast_to=object)
            assert raw["timings"]["prompt_n"] + raw["usage"]["cache_read_input_tokens"] == expected, raw
            response = {"blocks": raw["content"], "finish": raw["stop_reason"],
                        "usage": raw["usage"]}
        record(label, response)
        assert_prompt_count(expected, response["usage"])
        assert response["finish"] == "end_turn", response
        assert all(block["type"] == "text" for block in response["blocks"]), response
        text = "".join(block["text"] for block in response["blocks"])
        assert text.strip().lower().rstrip(".!") == color, response
        return response

    red = messages_color_requests(image_content("red")["image_url"]["url"])
    if "image" not in input_modalities:
        # This is a capability rejection check, not a vision workload on a
        # text-only model. Both routes must reject, including tool images.
        for place, request in red:
            body = {"model": model, **request}
            reject(place + "_count_unsupported", "/messages/count_tokens", body)
            reject(place + "_generate_unsupported", "/messages", body)
    else:
        for color in ("red", "blue"):
            png = image_content(color)["image_url"]["url"]
            for place, request in messages_color_requests(png):
                label = place + "_" + color
                body = {"model": model, **request}
                # The sizing endpoint does not require max_tokens.
                sizing = {key: value for key, value in body.items() if key != "max_tokens"}
                expected = count(label, sizing)
                assert count(label + "_repeat", sizing) == expected
                generate(label + "_generated", body, expected, color)
                assert count(label + "_warm", sizing) == expected
                warm = generate(label + "_streamed", body, expected, color, True)
                assert warm["usage"]["cache_read_input_tokens"] > 0, warm

        for name, media, payload in (("jpeg", "image/jpeg", JPEG),
                                      ("webp", "image/webp", WEBP_LOSSLESS)):
            body = {"model": model, **deepcopy(red[0][1])}
            body["messages"][0]["content"][0]["source"] = {
                "type": "base64", "media_type": media, "data": payload}
            generate(name + "_generated", body, count(name, body), "red")

        body = {"model": model, **deepcopy(red[0][1])}
        parts = body["messages"][0]["content"]
        blue = messages_color_requests(image_content("blue")["image_url"]["url"])[0][1]
        parts[:0] = [{"type": "text", "text": "café 東京 🦉"}, blue["messages"][0]["content"][0]]
        parts[-1]["text"] = "Name the color of the LAST image. Reply with one lowercase English color name only."
        generate("multi_image_generated", body, count("multi_image", body), "red")

        oversized = {"model": model, **deepcopy(red[0][1])}
        oversized["messages"][0]["content"].append({
            "type": "text", "text": " counting" * (context * 2)})
        assert count("over_context", oversized) > context
        reject("over_context_generation", "/messages", oversized)

        corrupt = {"model": model, **deepcopy(red[0][1])}
        corrupt["messages"][0]["content"][0]["source"]["data"] = "AQID"
        reject("corrupt_count", "/messages/count_tokens", corrupt)
        reject("corrupt_generation", "/messages", corrupt)

    recovery = {"model": model, "max_tokens": 16, "temperature": 0,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Reply with only BETA."}]}
    generate("text_recovery_generated", recovery, count("text_recovery", recovery), "beta")
