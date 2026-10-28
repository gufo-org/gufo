"""Chat/Responses sizing must match generation without inference side effects."""

from copy import deepcopy
import sys

from server_metrics import COUNTERS, DEFERRED, PROCESSING, ServerMetrics


def check_openai_count_tokens(client, model, checks, context, image_content,
                             input_modalities):
    import openai

    metrics = ServerMetrics(client.base_url)

    def record(label, value):
        checks["openai_count_tokens_" + label] = value
        print(f"CHECK openai_count_tokens_{label}", file=sys.stderr, flush=True)

    def count(label, endpoint, body):
        before = metrics.wait(lambda m: m[PROCESSING] == m[DEFERRED] == 0, "idle")
        response = client.post(endpoint + "/input_tokens", body=body, cast_to=object)
        after = metrics.read()
        record(label, {"response": response,
                       "counter_deltas": {key: after[key] - before[key] for key in COUNTERS}})
        assert set(response) == {"object", "input_tokens"}, response
        assert response["object"] == "response.input_tokens", response
        assert type(response["input_tokens"]) is int and response["input_tokens"] > 0, response
        assert all(after[key] == before[key] for key in COUNTERS), (before, after)
        assert after[PROCESSING] == after[DEFERRED] == 0, after
        return response["input_tokens"]

    def generate(label, endpoint, body, expected):
        responses = endpoint == "/responses"
        limit = "max_output_tokens" if responses else "max_tokens"
        response = client.post(endpoint, body={**body, limit: 32, "temperature": 0,
                                               "stream": False}, cast_to=object)
        record(label, response)
        usage = response["usage"]
        input_key, output_key = ("input_tokens", "output_tokens") if responses else (
            "prompt_tokens", "completion_tokens")
        assert usage[input_key] == expected, (expected, usage)
        assert usage[output_key] > 0, response
        return usage

    def reject(label, endpoint, body, status):
        try:
            client.post(endpoint, body=body, cast_to=object)
        except openai.APIStatusError as error:
            record(label, {"status": error.status_code, "body": error.body})
            assert error.status_code == status, error
        else:
            raise AssertionError(f"{label}: invalid request was accepted")

    for responses in (False, True):
        endpoint = "/responses" if responses else "/chat/completions"
        prefix = "responses_" if responses else "chat_"
        plain = ({"model": model, "input": "Reply with COUNT_OK.",
                  "reasoning": {"effort": "none"}} if responses else
                 {"model": model, "messages": [{"role": "user", "content": "Reply with COUNT_OK."}],
                  "reasoning_effort": "none"})
        initial = count(prefix + "plain", endpoint, plain)
        assert count(prefix + "stream_body", endpoint, {**plain, "stream": True}) == initial
        generate(prefix + "cold", endpoint, plain, initial)
        assert count(prefix + "warm_count", endpoint, plain) == initial
        usage = generate(prefix + "warm", endpoint, plain, initial)
        detail = "input_tokens_details" if responses else "prompt_tokens_details"
        assert usage[detail]["cached_tokens"] > 0, usage

        history = [{"role": "user", "content": "Remember café 東京."},
                   {"role": "assistant", "content": "Remembered."},
                   {"role": "user", "content": "Repeat the word café."}]
        system_history = ({**plain, "instructions": "Be concise.", "input": history}
                          if responses else {**plain, "messages": [
                              {"role": "system", "content": "Be concise."}, *history]})
        tool = {"name": "remember", "description": "Remember one word.",
                "parameters": {"type": "object", "properties": {"word": {"type": "string"}},
                               "required": ["word"], "additionalProperties": False}}
        tools = [{"type": "function", **tool} if responses else {"type": "function", "function": tool}]
        tool_input = ({"input": "Use remember for the word café."} if responses else
                      {"messages": [{"role": "user", "content": "Use remember for the word café."}]})
        tool_body = {**plain, **tool_input, "tools": tools}
        cases = [("system_history", system_history),
                 ("reasoning", {**plain, **({"reasoning": {"effort": "high"}} if responses else
                                           {"reasoning_effort": "high"})})]
        for choice in ("none", "auto", "required"):
            cases.append(("tools_" + choice, {**tool_body, "tool_choice": choice}))
        tool_history = deepcopy(tool_body)
        if responses:
            tool_history["input"] = [
                {"role": "user", "content": "Remember café."},
                {"type": "function_call", "call_id": "count_1", "name": "remember",
                 "arguments": '{"word":"café"}'},
                {"type": "function_call_output", "call_id": "count_1", "output": "Remembered."}]
        else:
            tool_history["messages"] = [
                {"role": "user", "content": "Remember café."},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "count_1", "type": "function", "function": {
                        "name": "remember", "arguments": '{"word":"café"}'}}]},
                {"role": "tool", "tool_call_id": "count_1", "content": "Remembered."}]
        cases.append(("tool_history", tool_history))
        schema = {"name": "answer", "description": "Return the answer as JSON.",
                  "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                             "required": ["answer"], "additionalProperties": False}}
        structured = ({"text": {"format": {"type": "json_schema", **schema}}} if responses else
                      {"response_format": {"type": "json_schema", "json_schema": schema}})
        cases.append(("structured", {**plain, **structured}))
        for label, body in cases:
            expected = count(prefix + label, endpoint, body)
            generate(prefix + label + "_generated", endpoint, body, expected)

        image = image_content("red")
        image_body = ({**plain, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Name the color."},
            {"type": "input_image", "image_url": image["image_url"]["url"]}]}]} if responses else
            {**plain, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Name the color."}, image]}]})
        if "image" in input_modalities:
            expected = count(prefix + "image", endpoint, image_body)
            generate(prefix + "image_generated", endpoint, image_body, expected)
        else:
            reject(prefix + "image_unsupported", endpoint + "/input_tokens", image_body, 400)

        oversized = {**plain, **({"input": " counting" * (context * 2)} if responses else
                                 {"messages": [{"role": "user", "content": " counting" * (context * 2)}]})}
        assert count(prefix + "over_context", endpoint, oversized) > context
        reject(prefix + "over_context_generation", endpoint,
               {**oversized, "max_output_tokens" if responses else "max_tokens": 1}, 400)
        reject(prefix + "wrong_model", endpoint + "/input_tokens",
               {**plain, "model": model + "-not-loaded"}, 404)
        assert count(prefix + "recovery", endpoint, plain) == initial
