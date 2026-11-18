"""Generated-token logprobs through the official SDK, on a real text server."""

import json
import math


def check_logprobs(client, model, checks, completion_frame):
    for chat in (False, True):
        endpoint = client.chat.completions if chat else client.completions
        for temperature in (0, 0.7):
            common = dict(model=model, temperature=temperature, seed=73,
                          max_tokens=24, extra_body={"cache_prompt": False})
            if chat:
                common.update(messages=[{"role": "user", "content": "Continue: red blue green red blue green"}],
                              reasoning_effort="none")
            else:
                common["prompt"] = "Continue: red blue green red blue green"
            baseline = endpoint.create(**common).model_dump()
            reference_choice = baseline["choices"][0]
            reference_text = reference_choice["message"] if chat else reference_choice["text"]
            for count, stream in ((0, False), (3, False), (3, True)):
                options = dict(logprobs=True, top_logprobs=count) if chat else dict(logprobs=count)
                name = f"logprobs_{'chat' if chat else 'completion'}_t{temperature}_k{count}_stream{stream}"
                if stream:
                    # The legacy SDK requires a terminal finish_reason even
                    # on intermediate chunks. Reuse the suite's nullable model.
                    stream_endpoint = endpoint if chat else client.copy(
                        _extra_kwargs={"_strict_response_validation": False}).completions
                    with stream_endpoint.create(**common, **options, stream=True,
                                                stream_options={"include_usage": True}) as response:
                        chunks = [chunk.model_dump() for chunk in response]
                    if not chat:
                        for chunk in chunks:
                            completion_frame.model_validate(chunk)
                    choices = [choice for chunk in chunks for choice in chunk["choices"]]
                    text = "".join((choice["delta"].get("content") or "") if chat else choice["text"]
                                   for choice in choices)
                    records = [choice["logprobs"] for choice in choices if choice.get("logprobs")]
                    assert text == (reference_text.get("content") or "" if chat else reference_text), chunks
                    checks[name] = {"chunks": chunks}
                else:
                    result = endpoint.create(**common, **options).model_dump()
                    choice = result["choices"][0]
                    assert (choice["message"] if chat else choice["text"]) == reference_text, result
                    records = [choice["logprobs"]]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        assert result["usage"][key] == baseline["usage"][key], result
                    checks[name] = result
                assert records, name
                seen = 0
                for record in records:
                    assert record is not None, name
                    if chat:
                        for token in record["content"]:
                            seen += 1
                            assert math.isfinite(token["logprob"]) and -9999 <= token["logprob"] <= 0
                            assert token["bytes"] == list(token["token"].encode("utf-8"))
                            assert len(token["top_logprobs"]) == count
                            for alternative in token["top_logprobs"]:
                                assert math.isfinite(alternative["logprob"]) and alternative["logprob"] <= 0
                                assert alternative["bytes"] == list(alternative["token"].encode("utf-8"))
                    else:
                        size = len(record["tokens"])
                        seen += size
                        assert all(len(record[key]) == size for key in ("token_logprobs", "top_logprobs", "text_offset"))
                        for token, value, alternatives in zip(record["tokens"], record["token_logprobs"], record["top_logprobs"]):
                            assert math.isfinite(value) and -9999 <= value <= 0
                            assert token in alternatives and alternatives[token] == value
                            assert 1 <= len(alternatives) <= count + 1
                assert seen >= baseline["usage"]["completion_tokens"], (name, seen, baseline["usage"])

    # Twenty candidates per token once exceeded the default 1 MiB text byte
    # limit near 560 tokens.
    long = client.completions.create(model=model, prompt="Count upward, one number per line:\n1\n",
                                     max_tokens=1200, temperature=0, logprobs=20,
                                     extra_body={"ignore_eos": True, "cache_prompt": False}).model_dump()
    checks["logprobs_completion_k20_long"] = {"usage": long["usage"],
                                              "finish_reason": long["choices"][0]["finish_reason"]}
    assert long["usage"]["completion_tokens"] == 1200, long["usage"]
    assert len(long["choices"][0]["logprobs"]["tokens"]) == 1200


def check_responses_logprobs(client, model, checks):
    """Validate Responses parts, SSE deltas/done events, and SDK accumulation."""
    def parts(response):
        return [part for item in response.output if item.type == "message"
                for part in item.content if part.type == "output_text"]

    def scores(response):
        return [token.model_dump() for part in parts(response) for token in part.logprobs]

    def event_scores(tokens):
        # SSE text-event schemas omit the bytes fields present on output parts.
        return [{"token": token["token"], "logprob": token["logprob"],
                 "top_logprobs": [{"token": top["token"], "logprob": top["logprob"]}
                                  for top in token["top_logprobs"]]} for token in tokens]

    for temperature in (0, 0.7):
        common = dict(model=model, input="Continue: red blue green red blue green",
                      reasoning={"effort": "none"}, max_output_tokens=24,
                      temperature=temperature, store=False,
                      extra_body={"seed": 73, "cache_prompt": False})
        baseline = client.responses.create(**common)
        checks[f"responses_logprobs_disabled_t{temperature}"] = baseline.model_dump()
        assert all(not part.logprobs for part in parts(baseline)), baseline
        for count in (0, 3, 20):
            options = dict(include=["message.output_text.logprobs"], top_logprobs=count)
            name = f"responses_logprobs_t{temperature}_k{count}"
            buffered = client.responses.create(**common, **options)
            checks[name + "_buffered"] = buffered.model_dump()
            assert buffered.output_text == baseline.output_text, (baseline, buffered)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                assert getattr(buffered.usage, key) == getattr(baseline.usage, key)
            expected = scores(buffered)
            assert expected, buffered
            for token in expected:
                assert math.isfinite(token["logprob"]) and -9999 <= token["logprob"] <= 0
                assert token["bytes"] == list(token["token"].encode("utf-8"))
                assert len(token["top_logprobs"]) == count
                for alternative in token["top_logprobs"]:
                    assert math.isfinite(alternative["logprob"]) and -9999 <= alternative["logprob"] <= 0
                    assert alternative["bytes"] == list(alternative["token"].encode("utf-8"))
            with client.responses.stream(**common, **options) as stream:
                events = list(stream)
                terminal = next(event.response for event in reversed(events)
                                if event.type in ("response.completed", "response.incomplete"))
                # SDK 2.41.1's helper exposes only completed responses.
                final = stream.get_final_response() if terminal.status == "completed" else terminal
            checks[name + "_stream"] = {"response": final.model_dump(),
                                        "events": [event.model_dump() for event in events]}
            assert final.output_text == buffered.output_text, (buffered, final)
            assert scores(final) == expected, (buffered, final)
            indices, deltas, emitted = [], {}, []
            for event in events:
                # The SDK adds local .done convenience events; only wire
                # events carry the server's sequence number.
                if getattr(event, "sequence_number", None) is not None:
                    indices.append(event.sequence_number)
                if event.type == "response.output_text.delta":
                    values = event_scores([token.model_dump() for token in event.logprobs])
                    key = event.output_index, event.content_index
                    deltas.setdefault(key, []).extend(values)
                    emitted.extend(values)
                elif event.type == "response.output_text.done":
                    values = event_scores([token.model_dump() for token in event.logprobs])
                    assert values == deltas.get((event.output_index, event.content_index), [])
            assert emitted == event_scores(expected), events
            assert indices == list(range(len(indices))), indices
            if count == 3:
                reference = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": common["input"]}],
                    reasoning_effort="none", temperature=temperature, max_completion_tokens=24,
                    logprobs=True, top_logprobs=count, extra_body=common["extra_body"])
                checks[name + "_chat_reference"] = reference.model_dump()
                assert reference.choices[0].message.content == buffered.output_text
                reference_scores = [token.model_dump() for token in reference.choices[0].logprobs.content]
                assert expected == reference_scores[:len(expected)], (expected, reference_scores)

    tool_request = dict(model=model, input="Call record with value repeat. No explanation.",
                        tools=[{"type": "function", "name": "record", "strict": True,
                                "parameters": {"type": "object", "properties": {
                                    "value": {"type": "string", "const": "repeat"}},
                                    "required": ["value"], "additionalProperties": False}}],
                        tool_choice="required", parallel_tool_calls=False,
                        reasoning={"effort": "none"}, temperature=0, max_output_tokens=128,
                        include=["message.output_text.logprobs"], top_logprobs=3,
                        extra_body={"seed": 73, "cache_prompt": False})
    tool = client.responses.create(**tool_request)
    checks["responses_logprobs_tool"] = tool.model_dump()
    assert tool.status == "completed" and not tool.output_text and not scores(tool), tool
    calls = [item for item in tool.output if item.type == "function_call"]
    assert len(calls) == 1 and json.loads(calls[0].arguments) == {"value": "repeat"}, tool

    reasoning = client.responses.create(
        model=model, input="What is two plus two? Reply briefly.",
        reasoning={"effort": "low"}, max_output_tokens=256, temperature=0,
        include=["message.output_text.logprobs"], top_logprobs=3,
        extra_body={"seed": 73, "cache_prompt": False})
    checks["responses_logprobs_reasoning"] = reasoning.model_dump()
    assert reasoning.status == "completed" and reasoning.output_text and scores(reasoning), reasoning
    assert any(item.type == "reasoning" for item in reasoning.output), reasoning
    assert all("logprobs" not in part.model_dump()
               for item in reasoning.output if item.type == "reasoning"
               for part in item.summary), reasoning
