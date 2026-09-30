#!/usr/bin/env python3
"""Check Gufo's text APIs with the official OpenAI Python SDK.

Run with nix develop -c python3. Start a Gufo text server first; all requests
go to the explicitly supplied loopback endpoint.
"""

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import sys
import zlib
from urllib.parse import urlsplit

import openai
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI


def chat_result(client, request, streaming=False):
    """Accumulate typed SDK chunks, including Gufo's reasoning/usage extensions."""
    result = client.chat.completions.create(
        **request, stream=streaming,
        **({"stream_options": {"include_usage": True}} if streaming else {}),
    )
    if not streaming:
        choice = result.choices[0]
        return {
            "text": choice.message.content or "",
            "reasoning": getattr(choice.message, "reasoning_content", "") or "",
            "tools": [t.to_dict() for t in choice.message.tool_calls or []],
            "finish": choice.finish_reason, "usage": result.usage.to_dict(),
        }
    text, reasoning, tools, finish, usage = "", "", [], None, None
    with result:
        for chunk in result:
            if chunk.usage:
                usage = chunk.usage.to_dict()
            for choice in chunk.choices:
                text += choice.delta.content or ""
                reasoning += getattr(choice.delta, "reasoning_content", "") or ""
                tools.extend(t.to_dict() for t in choice.delta.tool_calls or [])
                finish = choice.finish_reason or finish
    assert finish is not None and usage is not None, (finish, usage)
    return dict(text=text, reasoning=reasoning, tools=tools, finish=finish, usage=usage)


def check_stops(client, model, checks):
    # Start from actual greedy output: this tests filtering independently of
    # whether a particular quantization obeys a verbatim-copy instruction.
    request = dict(
        model=model,
        messages=[{"role": "user", "content":
                   "Copy exactly, without explanation: ALPHA BETA GAMMA DELTA"}],
        temperature=0, seed=42, max_completion_tokens=32,
        extra_body={"chat_template_kwargs": {"enable_thinking": False},
                    "cache_prompt": False},
    )

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def run(name, body, streaming=False):
        return record(name, chat_result(client, body, streaming))

    baseline = run("stop_baseline", request)
    text = baseline["text"]
    assert len(text) > 15 and not baseline["reasoning"] and not baseline["tools"], baseline
    marker = text[6:12]
    for streaming in (False, True):
        suffix = "stream" if streaming else "buffered"
        for label, stops in (
            ("string", marker),
            ("list", ["__NEVER_MATCH__", text[12:15], marker]),
            ("overlap", [text[6:12], text[6:9]]),
            ("empty_answer", text[0]),
        ):
            sequences = [stops] if isinstance(stops, str) else stops
            matches = [(text.index(s) + len(s), text.index(s))
                       for s in sequences if s in text]
            _, cut = min(matches)
            result = run(f"stop_{label}_{suffix}", {**request, "stop": stops}, streaming)
            assert result["text"] == text[:cut] and result["finish"] == "stop", result
            assert not result["tools"] and not result["reasoning"], result
        # The withheld prefix must be flushed when EOS/length wins.
        for label, stops in (
            ("partial_prefix", text[-4:] + "__NEVER_MATCH__"),
            ("null", None), ("empty_list", []),
        ):
            result = run(f"stop_{label}_{suffix}", {**request, "stop": stops}, streaming)
            assert (result["text"], result["finish"]) == (text, baseline["finish"]), result

    unicode_request = {**request, "messages": [{"role": "user", "content":
                       "Copy exactly, without explanation: 甲乙丙丁甲乙丙丁"}]}
    unicode_full = run("stop_unicode_baseline", unicode_request)
    assert "乙" in unicode_full["text"], unicode_full
    for streaming in (False, True):
        result = run(f"stop_unicode_{streaming}",
                     {**unicode_request, "stop": "乙"}, streaming)
        assert result["text"] == unicode_full["text"].split("乙")[0], result
        assert result["finish"] == "stop", result

    sampled = {**request, "temperature": .7, "top_p": .9,
               "messages": [{"role": "user", "content":
                             "Name ten animals, comma-separated."}]}
    full = run("stop_sampled_baseline", sampled)
    assert len(full["text"]) > 8, full
    marker_sampled = full["text"][4:8]
    result = run("stop_sampled", {**sampled, "stop": marker_sampled}, True)
    assert result["text"] == full["text"].split(marker_sampled)[0], result

    thinking = {
        **request, "messages": [{"role": "user", "content": "Compute 123 times 456."}],
        "extra_body": {**request["extra_body"],
                       "chat_template_kwargs": {"enable_thinking": True}},
    }
    thought = run("stop_reasoning_baseline", thinking)
    reasoning = thought["reasoning"]
    assert len(reasoning) > 12, thought
    thought_marker = reasoning[6:12]
    for streaming in (False, True):
        result = run(f"stop_reasoning_{streaming}",
                     {**thinking, "stop": thought_marker}, streaming)
        # Buffered chat trims reasoning, streamed deltas retain whitespace.
        assert result["reasoning"].strip() == reasoning.split(thought_marker)[0].strip(), result
        assert not result["text"] and not result["tools"] and result["finish"] == "stop", result

    tool_request = {
        **request, "max_completion_tokens": 96,
        "messages": [{"role": "user", "content":
                      "Call echo once with text exactly 'alpha SDK_STOP omega'."}],
        "tools": [{"type": "function", "function": {
            "name": "echo", "description": "Echo the supplied text.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string"}}, "required": ["text"]},
        }}], "tool_choice": "required",
    }
    full = run("stop_tool_baseline", tool_request)
    assert full["tools"] and "SDK_STOP" in full["tools"][0]["function"]["arguments"], full
    for streaming in (False, True):
        result = run(f"stop_tool_argument_{streaming}",
                     {**tool_request, "stop": "SDK_STOP"}, streaming)
        assert not result["tools"] and result["finish"] == "stop", result
        # DSML may have whitespace before the call; stopping must preserve it.
        assert not result["text"].strip() and not result["reasoning"], result

    with ThreadPoolExecutor(max_workers=2) as pool:
        stopped = pool.submit(chat_result, client, {**request, "stop": marker}, True)
        peer = pool.submit(chat_result, client, request, True)
        stopped, peer = stopped.result(), peer.result()
    assert stopped["text"] == text.split(marker)[0] and stopped["finish"] == "stop", stopped
    assert peer["text"] == text and peer["finish"] == baseline["finish"], peer
    record("stop_concurrent_isolation", [stopped, peer])

    cached = {**request, "extra_body": {**request["extra_body"], "cache_prompt": True},
              "messages": [{"role": "system", "content":
                            "The secret keyword is LANTERN. Follow instructions accurately. " * 32},
                           *request["messages"]]}
    full = run("stop_cache_baseline", cached)
    assert len(full["text"]) > 12, full
    stopped_request = {**cached, "stop": full["text"][6:12]}
    stopped = run("stop_cached", stopped_request)
    replay = run("stop_cached_replay", stopped_request, True)
    assert replay["text"] == stopped["text"] and replay["usage"]["cached_tokens"] > 0, replay
    continuation = {**cached, "max_completion_tokens": 16, "messages": [
        *cached["messages"], {"role": "assistant", "content": stopped["text"]},
        {"role": "user", "content": "Reply with only the secret keyword."},
    ]}
    warm = run("stop_continuation_warm", continuation)
    cold = run("stop_continuation_cold", {
        **continuation, "extra_body": {**continuation["extra_body"], "cache_prompt": False},
    })
    assert warm["usage"]["cached_tokens"] > 0, warm
    assert (warm["text"], warm["finish"]) == (cold["text"], cold["finish"]), (warm, cold)

    for value in ("", ["x"] * 5, ["ok", 1], 1):
        try:
            client.chat.completions.create(**request, stop=value)
        except openai.BadRequestError as error:
            assert error.status_code == 400 and error.code == "invalid_stop", error
        else:
            raise AssertionError(f"Invalid stop accepted: {value!r}")
    record("stop_invalid_schema", {"status": 400, "cases": 4})


def image_content(color):
    rgb = {"red": (255, 0, 0), "blue": (0, 0, 255)}[color]

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + bytes(rgb) * 128) * 128))
        + chunk(b"IEND", b"")
    )
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode()
        },
    }


def check_sampling_defaults(client, model, checks, preset, overrides, vision=False):
    """Omission and explicit presets must replay within each execution mode."""
    def signature(value):
        return value["text"].strip(), value["reasoning"].strip(), value["finish"]

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    if not overrides:
        # A default server must use the model's recommended thinking mode,
        # including the same effort instructions as an explicit request.
        effort = "high" if preset == "deepseek4" else "xhigh"
        request = {
            "model": model, "seed": 73, "max_completion_tokens": 8,
            "messages": [{"role": "user", "content": "What is two plus two?"}],
            "extra_body": {"cache_prompt": False},
        }
        default = chat_result(client, request)
        explicit = chat_result(client, {**request, "reasoning_effort": effort}, True)
        assert default["reasoning"], default
        assert signature(default) == signature(explicit), (default, explicit)
        record("default_thinking_effort", default)
        if preset == "deepseek4":
            xhigh = chat_result(client, {**request, "reasoning_effort": "xhigh"})
            assert signature(xhigh) == signature(explicit), (xhigh, explicit)
            record("deepseek_xhigh_maps_high", xhigh)

    for thinking in (False, True):
        qwen_off = preset == "qwen38" and not thinking
        expected = {
            "temperature": .7 if qwen_off else 1.,
            "top_p": .8 if qwen_off else .95,
            "presence_penalty": 1.5 if qwen_off else 0.,
            "frequency_penalty": 0.,
            "seed": 73,
        }
        native = {"top_k": 20 if preset == "qwen38" else 0,
                  "min_p": 0., "min_keep": 0, "repeat_penalty": 1.,
                  "repeat_last_n": 64}
        assert overrides.keys() <= (expected.keys() | native.keys()), overrides
        expected.update({key: value for key, value in overrides.items() if key in expected})
        native.update({key: value for key, value in overrides.items() if key in native})
        request = {
            "model": model, "seed": 73, "max_completion_tokens": 8,
            "messages": [{"role": "user", "content":
                          "Name three animals and describe each in one sentence."}],
            "extra_body": {"cache_prompt": False, "chat_template_kwargs": {
                "enable_thinking": thinking}},
        }
        if "seed" in overrides:
            del request["seed"]
        explicit = {**request, **expected,
                    "extra_body": {**request["extra_body"], **native}}
        inherited = chat_result(client, request)
        supplied = chat_result(client, explicit, True)
        assert signature(inherited) == signature(supplied), (inherited, supplied)
        nullable = chat_result(client, {**request, "temperature": None,
                                       "top_p": None, "presence_penalty": None,
                                       "frequency_penalty": None,
                                       **({"seed": None} if "seed" in overrides else {})})
        assert signature(nullable) == signature(inherited), (nullable, inherited)
        record(f"sampling_preset_thinking{thinking}", inherited)

        # Each change targets one independent control, with and without all
        # other defaults written out; this also checks explicit neutral values.
        for label, override, extra in (
            ("greedy", {"temperature": 0, "presence_penalty": 0}, {"top_k": 0}),
            ("nucleus", {"temperature": .5, "top_p": .7}, {}),
            ("min_p", {"temperature": 1.3, "top_p": 1.}, {"top_k": 0, "min_p": .1}),
            ("penalties", {"frequency_penalty": .3, "presence_penalty": .4},
             {"repeat_penalty": 1.1, "repeat_last_n": 16}),
            ("greedy_penalties", {"temperature": 0, "frequency_penalty": .3,
                                  "presence_penalty": .4},
             {"repeat_penalty": 1.1, "repeat_last_n": 16}),
        ):
            body = {**request, **override, "extra_body": {**request["extra_body"], **extra}}
            full = {**explicit, **override, "extra_body": {**explicit["extra_body"], **extra}}
            actual, reference = chat_result(client, body), chat_result(client, full, True)
            assert signature(actual) == signature(reference), (label, actual, reference)
            record(f"sampling_{label}_thinking{thinking}", actual)

        # Reusing model state must not restore the preceding request's sampler.
        cached = chat_result(client, {
            **request, "extra_body": {**request["extra_body"], "cache_prompt": True}}, True)
        assert signature(cached) == signature(inherited), (cached, inherited)
        assert cached["usage"]["cached_tokens"] > 0, cached
        assert cached["usage"]["gufo"]["prefill_tokens"] == 0, cached
        record(f"sampling_cached_preset_thinking{thinking}", cached)

        # Distinct request-owned samplers must replay when admitted together.
        peers = [request, {**request, "seed": 91, "temperature": .4}]
        def batch():
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(lambda body: chat_result(client, body, True), peers))
        first, second = batch(), batch()
        assert list(map(signature, first)) == list(map(signature, second)), (first, second)
        record(f"sampling_concurrent_thinking{thinking}", first)

        # Responses must select its preset after applying reasoning.effort.
        response_request = {
            "model": model, "input": request["messages"][0]["content"],
            "reasoning": {"effort": "low" if thinking else "none"},
            "max_output_tokens": 8, "store": False,
            "extra_body": {"cache_prompt": False},
        }
        if vision:
            response_request["input"] = [{"role": "user", "content": [
                *[{"type": "input_image",
                   "image_url": image_content(color)["image_url"]["url"]}
                  for color in ("red", "blue")],
                {"type": "input_text", "text":
                 "Reply with exactly two English color names in image order, separated by a comma."},
            ]}]
            # Allow a complete visual answer; reasoning remains deliberately bounded.
            response_request["max_output_tokens"] = 8 if thinking else 32
        if "seed" not in overrides:
            response_request["extra_body"]["seed"] = 73
        explicit_response = {
            **response_request,
            "temperature": expected["temperature"], "top_p": expected["top_p"],
            "extra_body": {
                **response_request["extra_body"], **native,
                **{key: expected[key] for key in ("seed", "presence_penalty", "frequency_penalty")},
            },
        }

        def response_signature(response):
            check_response(response, thinking)
            return response.status, [
                item.model_dump(exclude={"id", "status"}, exclude_none=True)
                for item in response.output
            ]

        omitted = client.responses.create(**response_request)
        with client.responses.stream(**explicit_response) as stream:
            events = list(stream)
        check_events(events, thinking)
        supplied = events[-1].response
        assert omitted.usage.input_tokens_details.cached_tokens == 0, omitted
        assert supplied.usage.input_tokens_details.cached_tokens == 0, supplied
        assert response_signature(omitted) == response_signature(supplied), (omitted, supplied)
        record(f"responses_preset_thinking{thinking}", check_response(omitted, thinking))
        cached_response = client.responses.create(**{
            **response_request,
            "extra_body": {**response_request["extra_body"], "cache_prompt": True},
        })
        assert cached_response.usage.input_tokens_details.cached_tokens > 0, cached_response
        assert response_signature(cached_response) == response_signature(omitted), (
            cached_response, omitted)
        if vision and not thinking:
            assert "red" in omitted.output_text.lower() and "blue" in omitted.output_text.lower(), omitted
        record(f"responses_cached_preset_thinking{thinking}",
               check_response(cached_response, thinking))

        # Exercise grammar+default resolution without requiring a long answer.
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                  "required": ["ok"], "additionalProperties": False}
        constrained = {
            **request, "max_completion_tokens": 32 if not thinking else 2,
            "messages": [{"role": "user", "content": "Return a JSON object with ok true."}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "sampling_defaults", "strict": True, "schema": schema}},
        }
        if vision:
            constrained["messages"][0]["content"] = [
                image_content("red"), {"type": "text", "text": "Return a JSON object with ok true."}]
        actual = chat_result(client, constrained, True)
        if thinking:
            assert actual["finish"] == "length" and actual["usage"]["completion_tokens"] == 2, actual
        else:
            from jsonschema import Draft202012Validator
            assert actual["finish"] == "stop", actual
            Draft202012Validator(schema).validate(json.loads(actual["text"]))
        record(f"sampling_schema_thinking{thinking}", actual)


def check_conversations(client, model, checks, vision=False):
    """Exercise thinking controls and cache reuse after a client disconnect."""

    def signature(result):
        return result["text"], result["reasoning"], result["finish"]

    def save(label, result):
        checks[label] = result
        print(f"CHECK {label}", file=sys.stderr, flush=True)
        return result

    def chat(label, request, stream=False):
        return save(label, chat_result(client, request, stream))

    def body(prompt="Compute 123 times 456.", thinking=False, **kwargs):
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 31,
            "max_completion_tokens": 12,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": thinking,
                    "reasoning_effort": "low",
                },
                "cache_prompt": False,
            },
            **kwargs,
        }

    for effort in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        request = body()
        request["extra_body"] = {"cache_prompt": False}
        request["reasoning_effort"] = effort
        r = chat("effort_" + effort, request, True)
        assert bool(r["reasoning"]) == (effort != "off"), r
    for enabled in (False, True):
        for shape in ("kwargs", "thinking"):
            request = body(thinking=enabled)
            if shape == "thinking":
                request["extra_body"] = {
                    "cache_prompt": False,
                    "thinking": {"type": "enabled" if enabled else "disabled"},
                }
            r = chat(f"{shape}_{enabled}", request)
            assert bool(r["reasoning"]) == enabled, r
    for override in (
        {"reasoning_effort": "bogus"},
        {
            "reasoning_effort": "high",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    ):
        try:
            client.chat.completions.create(**{**body(), **override})
        except openai.BadRequestError as e:
            assert e.code == "invalid_reasoning", e
        else:
            raise AssertionError("invalid reasoning accepted")
    save("invalid_reasoning", {"cases": 2, "status": 400})
    for thinking in (False, True):
        for retain in (False, True):
            label = f"cancel_thinking{thinking}_retain{retain}"
            request = body(
                "Derive the sum of the first 1000 squares step by step."
                if thinking
                else "Count from one to one hundred, separated by commas.",
                thinking=thinking,
                max_completion_tokens=128,
            )
            request["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": label
                    + ". "
                    + "Follow the user instruction carefully and answer accurately. "
                    * 32,
                },
            )
            request["extra_body"]["chat_template_kwargs"]["preserve_thinking"] = retain
            if vision:
                request["messages"][-1]["content"] = [
                    image_content("red"),
                    {"type": "text", "text": request["messages"][-1]["content"]},
                ]
            assistant = {"role": "assistant", "content": "", "reasoning_content": ""}
            count = 0
            with client.chat.completions.create(**request, stream=True) as stream:
                for chunk in stream:
                    for choice in chunk.choices:
                        a = choice.delta.content or ""
                        b = getattr(choice.delta, "reasoning_content", "") or ""
                        assistant["content"] += a
                        assistant["reasoning_content"] += b
                        count += bool(b if thinking else a)
                    if count >= 3:
                        break
                else:
                    raise AssertionError("never reached cancellation point")
            request["extra_body"]["cache_prompt"] = True
            request["messages"] += ([assistant] if retain else []) + [
                {
                    "role": "user",
                    "content": "Now reply with only the number 7." if retain else ".",
                }
            ]
            request["max_completion_tokens"] = 8
            warm = chat(label + "_resume", request)
            assert warm["usage"]["cached_tokens"] >= 128, warm
            if not retain:
                assert warm["usage"]["gufo"]["prefill_tokens"] <= 16, warm
            replay = chat(label + "_replay", request, True)
            assert (
                warm["text"].strip(),
                warm["reasoning"].strip(),
                warm["finish"],
            ) == (
                replay["text"].strip(),
                replay["reasoning"].strip(),
                replay["finish"],
            ), (warm, replay)
    candidates = []
    if vision:
        for color in ("red", "blue"):
            request = body(
                [
                    image_content(color),
                    {
                        "type": "text",
                        "text": "Name the dominant color in the image in a full sentence.",
                    },
                ],
                max_completion_tokens=24,
            )
            request["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "Describe the actual image carefully. " * 32,
                },
            )
            cold = chat("vision_" + color + "_cold", request)
            assert color in cold["text"].lower() and not cold["reasoning"], cold
            request["extra_body"]["cache_prompt"] = True
            warm = chat("vision_" + color + "_warm", request)
            repeated = chat("vision_" + color + "_replay", request, True)
            assert signature(cold) == signature(warm) == signature(repeated), (
                cold,
                warm,
                repeated,
            )
            assert (
                repeated["usage"]["cached_tokens"] > 0
                and repeated["usage"]["gufo"]["prefill_tokens"] == 0
            ), repeated
            candidates.append((request, cold))
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(lambda item: chat_result(client, item[0], True), candidates)
            )
        for (_, expected), r in zip(candidates, results):
            assert signature(r) == signature(expected), (r, expected)
        save("vision_concurrent", results)
        request, expected = candidates[0]
        marker = expected["text"][6:12]
        assert marker
        r = chat("vision_stop", {**request, "stop": marker}, True)
        assert (
            r["text"] == expected["text"].split(marker)[0] and r["finish"] == "stop"
        ), r


def check_structured_outputs(client, model, checks, vision=False):
    from typing import Literal
    from pydantic import BaseModel, Field

    class Item(BaseModel):
        name: Literal["cat", "dog"]
        count: int = Field(ge=1, le=5)

    class Reply(BaseModel):
        items: list[Item] = Field(min_length=1, max_length=2)
        note: str | None

    check_strict_tools(client, model, checks)

    def record(name, value):
        checks[name] = value
        print(f"CHECK {name}", file=sys.stderr, flush=True)

    def sdk_result(result):
        choice = result.choices[0]
        return {"content": choice.message.content,
                "parsed": choice.message.parsed.model_dump(),
                "finish": choice.finish_reason, "usage": result.usage.to_dict()}

    common = dict(model=model, messages=[{"role": "user", "content":
                  "Return one item named cat with count 1, and a null note."}],
                  temperature=0, seed=31, max_completion_tokens=128,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    for label, options in (
        ("greedy", {}),
        ("sampled", {"temperature": .7, "top_p": .8,
                     "presence_penalty": .3, "frequency_penalty": .2,
                     "extra_body": {"top_k": 20, "min_p": .05, "repeat_penalty": 1.1}}),
    ):
        body = {**common, **options}
        body["extra_body"] = {**common["extra_body"], **options.get("extra_body", {})}
        first = client.chat.completions.parse(**body, response_format=Reply)
        assert first.choices[0].message.parsed is not None, first
        assert first.choices[0].finish_reason == "stop", first
        with client.chat.completions.stream(**body, response_format=Reply,
                                           stream_options={"include_usage": True}) as stream:
            events = list(stream)
            repeated = stream.get_final_completion()
        assert repeated.choices[0].message.content == first.choices[0].message.content, (first, repeated)
        assert repeated.choices[0].message.parsed == first.choices[0].message.parsed
        assert any(e.type == "content.delta" for e in events), events
        record("schema_sdk_" + label, sdk_result(first))

    class Measurement(BaseModel):
        score: int = Field(ge=1, le=5)
        fraction: float = Field(ge=0, le=1, multiple_of=.25)
        tag: str = Field(pattern=r"^[A-Z]{2}$", min_length=2, max_length=2)

    measured = {**common, "messages": [{"role": "user", "content":
                "Return score 3, fraction 0.75, and tag OK."}]}
    for thinking in (False, True):
        body = {**measured, "max_completion_tokens": 768 if thinking else 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
        result = client.chat.completions.parse(**body, response_format=Measurement)
        assert result.choices[0].message.parsed is not None, result
        record(f"schema_bounds_thinking_{thinking}", sdk_result(result))
    response = client.responses.parse(
        model=model, input=measured["messages"][0]["content"],
        text_format=Measurement, reasoning={"effort": "none"},
        temperature=0, max_output_tokens=128)
    assert response.output_parsed is not None and response.status == "completed", response
    record("schema_responses", {"text": response.output_text,
                               "parsed": response.output_parsed.model_dump(),
                               "usage": response.usage.to_dict()})
    with client.responses.stream(
        model=model, input=measured["messages"][0]["content"],
        text_format=Measurement, reasoning={"effort": "none"},
        temperature=0, max_output_tokens=128) as stream:
        list(stream)
        streamed = stream.get_final_response()
    assert streamed.output_text == response.output_text, (response, streamed)
    record("schema_responses_stream", {"text": streamed.output_text})

    object_request = {**common, "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": "Return JSON with answer 42."}]}
    result = chat_result(client, object_request, True)
    assert result["finish"] == "stop" and isinstance(json.loads(result["text"]), dict), result
    record("json_object", result)

    unbounded = {
        **common,
        "messages": [{"role": "user", "content": "Return the first two positive integers."}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "Numbers", "description": "A list of the requested integers.",
            "strict": True, "schema": {
                "type": "object", "properties": {
                    "numbers": {"type": "array", "items": {"type": "integer"}}},
                "required": ["numbers"], "additionalProperties": False}}},
    }
    result = chat_result(client, unbounded, True)
    assert result["finish"] == "stop" and json.loads(result["text"]) == {"numbers": [1, 2]}, result
    record("schema_unbounded_array", result)

    def constant(text):
        return {"type": "json_schema", "json_schema": {"name": "constant", "strict": True,
                "schema": {"type": "object", "properties": {
                    "text": {"type": "string", "const": text}},
                    "required": ["text"], "additionalProperties": False}}}

    # Unicode and protocol-looking strings must not be reinterpreted by the
    # streaming tool/reasoning parser, including tokens split inside UTF-8.
    literal = '┌ <think>not reasoning</think> <tool_call>not a call</tool_call> "\\'
    constrained = {**common, "response_format": constant(literal)}
    for streaming in (False, True):
        result = chat_result(client, constrained, streaming)
        assert result["finish"] == "stop" and json.loads(result["text"]) == {"text": literal}, result
        assert not result["reasoning"] and not result["tools"], result
        record(f"schema_literal_{streaming}", result)

    # A new grammar starts at its root even when prompt/model state is reused.
    for text in ("alpha", "beta", "alpha"):
        body = {**common, "response_format": constant(text)}
        result = chat_result(client, body, True)
        assert json.loads(result["text"]) == {"text": text}, result
        record("schema_cache_" + text, result)
    assert checks["schema_cache_alpha"]["usage"]["cached_tokens"] > 0

    peers = [{**common, "response_format": constant(text), "temperature": .8, "seed": seed}
             for text, seed in (("peer-a", 11), ("peer-b", 19))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda body: chat_result(client, body, True), peers))
    assert [json.loads(r["text"])["text"] for r in results] == ["peer-a", "peer-b"], results
    record("schema_concurrent", results)
    plain = chat_result(client, {**common, "messages": [{"role": "user", "content":
                        "Reply with just the word hello."}], "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}}})
    assert "hello" in plain["text"].lower() and not plain["text"].startswith("{"), plain
    record("schema_does_not_leak", plain)

    interrupted = {**common, "response_format": constant("resume " * 24),
                   "messages": [{"role": "system", "content": "Follow the schema precisely. " * 48},
                                {"role": "user", "content": "Return the required object."}]}
    partial, chunks = "", 0
    with client.chat.completions.create(**interrupted, stream=True) as stream:
        for chunk in stream:
            for choice in chunk.choices:
                piece = choice.delta.content or ""
                partial += piece
                chunks += bool(piece)
            if chunks >= 3:
                break
        else:
            raise AssertionError("structured stream never reached cancellation point")
    resumed = {**interrupted, "messages": [*interrupted["messages"],
               {"role": "assistant", "content": partial},
               {"role": "user", "content": "Return a fresh complete object."}]}
    result = chat_result(client, resumed, True)
    assert json.loads(result["text"]) == {"text": "resume " * 24}, result
    assert result["usage"]["cached_tokens"] >= 128, result
    replay = chat_result(client, resumed)
    assert replay["text"] == result["text"], (result, replay)
    record("schema_cancel_resume", result)
    record("schema_cancel_replay", replay)

    truncated = chat_result(client, {**constrained, "max_completion_tokens": 1}, True)
    assert truncated["finish"] == "length", truncated
    record("schema_length", truncated)
    stopped = chat_result(client, {**constrained, "stop": "text"}, True)
    assert stopped["finish"] == "stop" and "text" not in stopped["text"], stopped
    try:
        json.loads(stopped["text"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("stop inside the required key should leave incomplete JSON")
    record("schema_explicit_stop", stopped)
    arguments = {
        "type": "object", "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "text": {"type": "string", "const": "<think>literal</think>"}},
        "required": ["score", "text"], "additionalProperties": False}
    tool_request = {**constrained, "tool_choice": "required",
                    "messages": [{"role": "user", "content": "Call the score tool with score 3."}],
                    "tools": [{"type": "function", "function": {
                        "name": "score", "description": "Report a score",
                        "parameters": arguments, "strict": True}}]}
    for streaming in (False, True):
        tool_result = chat_result(client, tool_request, streaming)
        assert tool_result["finish"] == "tool_calls" and len(tool_result["tools"]) == 1, tool_result
        function = tool_result["tools"][0]["function"]
        from jsonschema import Draft202012Validator
        Draft202012Validator(arguments).validate(json.loads(function["arguments"]))
        assert function["name"] == "score", tool_result
        record(f"schema_tool_{streaming}", tool_result)
    # A response schema leaves automatic tools enabled; an explicit `none`
    # must select the final-answer grammar even when the prompt asks for a tool.
    answer_only = {**tool_request, "tool_choice": "none",
                   "response_format": constant("ok")}
    for streaming in (False, True):
        result = chat_result(client, answer_only, streaming)
        assert result["finish"] == "stop" and not result["tools"], result
        assert json.loads(result["text"]) == {"text": "ok"}, result
        record(f"schema_tool_none_{streaming}", result)
    for invalid in (
        {"response_format": {"type": "json_schema", "json_schema": {
            "name": "bad", "schema": {"type": "object", "properties": {},
            "additionalProperties": False, "not": {}}}}},
    ):
        try:
            client.chat.completions.create(**{**constrained, **invalid}, stream=True)
        except openai.BadRequestError as error:
            assert error.code == "invalid_response_format", error
        else:
            raise AssertionError("unsupported structured combination accepted")
    record("schema_invalid", {"status": 400, "cases": 1})

    if vision:
        class Color(BaseModel):
            color: Literal["red", "blue"]
        requests = [{**common, "messages": [{"role": "user", "content": [
            image_content(color), {"type": "text", "text": "What is the dominant color?"}]}]}
            for color in ("red", "blue")]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda body: client.chat.completions.parse(
                **body, response_format=Color), requests))
        for expected, result in zip(("red", "blue"), results):
            assert result.choices[0].message.parsed.color == expected, result
        replay = client.chat.completions.parse(**requests[0], response_format=Color)
        assert replay.choices[0].message.parsed.color == "red", replay
        assert replay.usage.to_dict()["cached_tokens"] > 0, replay
        record("schema_vision", [sdk_result(r) for r in results])
        record("schema_vision_replay", sdk_result(replay))
        responses_image = client.responses.parse(
            model=model, input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": image_content("blue")["image_url"]["url"]},
                {"type": "input_text", "text": "What is the dominant color?"}]}],
            text_format=Color, reasoning={"effort": "none"},
            temperature=0, max_output_tokens=64)
        assert responses_image.output_parsed.color == "blue", responses_image
        record("schema_responses_image", {"text": responses_image.output_text,
                                         "usage": responses_image.usage.to_dict()})


def check_strict_tools(client, model, checks):
    """Strict arguments work without imposing a JSON response on ordinary text."""
    from concurrent.futures import ThreadPoolExecutor

    literal = "<think>literal</think></tool_call>"
    schema = {"type": "object", "properties": {
        "value": {"type": "string", "const": literal}},
        "required": ["value"], "additionalProperties": False}
    tool = {"type": "function", "function": {
        "name": "record", "strict": True, "parameters": schema}}
    common = dict(model=model, tools=[tool], parallel_tool_calls=False,
                  messages=[{"role": "user", "content":
                             "Call record once with the required value. No explanation."}],
                  temperature=0, seed=41, max_completion_tokens=96,
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def signature(result):
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == "record", result
        assert json.loads(function["arguments"]) == {"value": literal}, result
        assert not result["reasoning"], result
        return (result["text"], function["name"], function["arguments"])

    for choice in ("required", {"type": "function", "function": {"name": "record"}}):
        result = record("strict_tool_" + ("required" if isinstance(choice, str) else "forced"),
                        chat_result(client, {**common, "tool_choice": choice}, True))
        signature(result)
    sampled = {**common, "tool_choice": "required", "temperature": .7, "top_p": .8,
               "presence_penalty": .3, "frequency_penalty": .2,
               "extra_body": {**common["extra_body"], "top_k": 20, "min_p": .05,
                              "repeat_penalty": 1.1}}
    baseline = chat_result(client, sampled, False)
    expected = signature(baseline)
    record("strict_tool_sampled", baseline)
    parsed = client.chat.completions.parse(**sampled)
    assert parsed.choices[0].message.tool_calls[0].function.parsed_arguments == {"value": literal}, parsed
    record("strict_tool_sdk_parse", {"parsed": {"value": literal}})
    with ThreadPoolExecutor(2) as pool:
        concurrent = list(pool.map(lambda _: chat_result(client, sampled, True), range(2)))
    assert all(signature(result) == expected for result in concurrent), concurrent
    assert all(result["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
               for result in concurrent), concurrent
    record("strict_tool_concurrent_replay", concurrent)
    for limit in (2, 12):
        partial = chat_result(client, {**sampled, "max_completion_tokens": limit}, True)
        assert partial["finish"] == "length" and not partial["tools"], partial
        # Native DeepSeek calls may begin with ordinary whitespace. No partial
        # parameter or envelope markup may escape at the token limit.
        assert not partial["text"].strip(), partial
        record(f"strict_tool_limit_{limit}", partial)
    stopped = chat_result(client, {**sampled, "stop": "literal"}, True)
    assert stopped["finish"] == "stop" and not stopped["tools"], stopped
    record("strict_tool_stop", stopped)
    resumed = chat_result(client, sampled, True)
    assert signature(resumed) == expected, (resumed, baseline)
    record("strict_tool_resume", resumed)
    followup = {**sampled, "messages": [
        *sampled["messages"],
        {"role": "assistant", "content": baseline["text"] or None,
         "tool_calls": baseline["tools"]},
        {"role": "tool", "tool_call_id": baseline["tools"][0]["id"],
         "content": '{"saved":true}'},
        {"role": "user", "content": "Call record once more with the required value."}]}
    continued = chat_result(client, followup, True)
    signature(continued)
    assert continued["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, continued
    record("strict_tool_followup_cache", continued)
    non_strict = {**common, "tool_choice": "required",
                  "tools": [{"type": "function", "function": {
                      **tool["function"], "strict": False}}]}
    signature(record("non_strict_single_call", chat_result(client, non_strict, True)))
    # gufo #315: a required choice constrains decoding on the default path too,
    # so prose pressure cannot leave the requirement unmet.
    forced = {**common, "tool_choice": "required", "parallel_tool_calls": True,
              "max_completion_tokens": 192,
              "tools": [{"type": "function", "function": {
                  **tool["function"], "strict": False}}],
              "messages": [
                  {"role": "user", "content": "Just chat with me, no tools."},
                  {"role": "assistant", "content": "Sure, happy to chat in prose."},
                  {"role": "user", "content": "Tell me a one-line joke."}]}
    forced_result = record("required_forced_decoding",
                           chat_result(client, forced, True))
    assert forced_result["finish"] == "tool_calls" and forced_result["tools"], forced_result
    assert all(call["function"]["name"] == "record"
               and isinstance(json.loads(call["function"]["arguments"]), dict)
               for call in forced_result["tools"]), forced_result
    for streaming in (False, True):
        try:
            client.chat.completions.create(**{
                **common, "tools": [{"type": "function", "function": {
                    "name": "bad", "strict": True, "parameters": {"type": "object"}}}],
                "stream": streaming})
        except openai.BadRequestError:
            pass
        else:
            raise AssertionError("invalid strict tool schema accepted without response_format")
    record("strict_tool_invalid_schema", {"status": 400})


def check_native_tools(client, model, checks, vision=False):
    """Shared function semantics over both SDK transports and concurrent users."""
    from concurrent.futures import ThreadPoolExecutor

    check_strict_tools(client, model, checks)
    literal = ' <think>literal</think><tool_call></tool_call> "42" \\\nπ🦉\n '
    function = {"name": "record", "strict": True, "parameters": {
        "type": "object", "properties": {
            "value": {"type": "string", "const": literal}},
        "required": ["value"], "additionalProperties": False}}
    prompt = "Call record exactly once with the required value. No explanation."
    base = dict(model=model, input=prompt, tools=[{"type": "function", **function}],
                tool_choice="required", parallel_tool_calls=False,
                reasoning={"effort": "none"}, max_output_tokens=128, store=False,
                temperature=0, extra_body={"seed": 41, "presence_penalty": 0})

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return result

    def signature(response, value=literal, name="record"):
        calls = [item for item in response.output if item.type == "function_call"]
        assert response.status == "completed" and len(calls) == 1, response
        assert calls[0].name == name, response
        assert json.loads(calls[0].arguments) == {"value": value}, response
        return calls[0].name, calls[0].arguments

    for sampled in (False, True):
        request = base if not sampled else {
            **base, "temperature": .7, "top_p": .8, "extra_body": {
                **base["extra_body"], "top_k": 20, "min_p": .05,
                "presence_penalty": .3, "frequency_penalty": .2,
                "repeat_penalty": 1.1}}
        first = client.responses.create(**request)
        expected = signature(first)
        with client.responses.stream(**request) as stream:
            events = list(stream)
            replay = stream.get_final_response()
        assert signature(replay) == expected, (first, replay)
        assert replay.usage.input_tokens_details.cached_tokens > 0, replay
        assert [event.sequence_number for event in events] == list(range(len(events)))
        assert any(event.type == "response.function_call_arguments.delta" for event in events)
        record(f"responses_tool_sampled{sampled}", replay.to_dict())

    named = client.responses.create(**{
        **base, "tool_choice": {"type": "function", "name": "record"}})
    signature(named)
    record("responses_named_tool", named.to_dict())
    history = [{"role": "user", "content": prompt}, *named.output,
               {"type": "function_call_output",
                "call_id": next(item.call_id for item in named.output
                                if item.type == "function_call"),
                "output": '{"saved":true}'},
               {"role": "user", "content": prompt}]
    followup = client.responses.create(**{**base, "input": history})
    signature(followup)
    assert followup.usage.input_tokens_details.cached_tokens > 0, followup
    record("responses_tool_history_cache", followup.to_dict())
    for limit in (1, 12):
        with client.responses.create(
                **{**base, "max_output_tokens": limit}, stream=True) as stream:
            events = list(stream)
        final = events[-1].response
        assert final.status == "incomplete", final
        assert final.incomplete_details.reason == "max_output_tokens", final
        assert not any(item.type == "function_call" for item in final.output), final
        record(f"responses_tool_limit_{limit}", final.to_dict())

    def isolated(index):
        fn = {**function, "name": f"record_{index}"}
        request = {**base, "tools": [{"type": "function", **fn}],
                   "input": f"Call record_{index} once with its required value."}
        result = client.responses.create(**request)
        signature(result, name=fn["name"])
        return result.to_dict()

    with ThreadPoolExecutor(4) as pool:
        record("responses_tool_c4_isolation", list(pool.map(isolated, range(4))))

    # A framed delimiter cannot be represented as raw native parameter data.
    # The exact JSON fallback must preserve it, including literal backslashes.
    delimiter = "\n</parameter>\n</｜DSML｜parameter>\\"
    fallback = json.loads(json.dumps(function))
    fallback["parameters"]["properties"]["value"]["const"] = delimiter
    result = client.responses.create(**{
        **base, "tools": [{"type": "function", **fallback}]})
    signature(result, value=delimiter)
    record("responses_tool_delimiter_fallback", result.to_dict())

    fallback["parameters"]["properties"]["value"] = {
        "type": "string", "pattern": "^\\n</parameter>$"}
    result = client.responses.create(**{
        **base, "tools": [{"type": "function", **fallback}]})
    signature(result, value="\n</parameter>")
    record("responses_tool_pattern_fallback", result.to_dict())

    nested = '<tool_call>{"name":"record","arguments":{"value":"literal"}}</tool_call>'
    fallback["parameters"]["properties"]["value"] = {
        "type": "string", "const": nested}
    result = client.responses.create(**{
        **base, "tools": [{"type": "function", **fallback}]})
    signature(result, value=nested)
    record("responses_literal_call_argument", result.to_dict())

    thinking_function = json.loads(json.dumps(function))
    thinking_function["parameters"]["properties"]["value"] = {
        "type": "string", "const": "alpha"}
    thought = chat_result(client, dict(
        model=model, messages=[{"role": "user", "content": "Call record once."}],
        tools=[{"type": "function", "function": thinking_function}],
        tool_choice="required", parallel_tool_calls=False,
        reasoning_effort="low", temperature=0, max_completion_tokens=256,
        extra_body={"presence_penalty": 0}), True)
    assert thought["finish"] == "tool_calls" and len(thought["tools"]) == 1, thought
    assert json.loads(thought["tools"][0]["function"]["arguments"]) == {"value": "alpha"}, thought
    record("chat_tool_thinking_low", thought)
    with client.responses.stream(**{
            **base, "tools": [{"type": "function", **thinking_function}],
            "reasoning": {"effort": "high"}, "max_output_tokens": 256}) as stream:
        list(stream)
        thought = stream.get_final_response()
    signature(thought, value="alpha")
    record("responses_tool_thinking_high", thought.to_dict())

    if vision:
        fn = json.loads(json.dumps(function))
        fn["parameters"]["properties"]["value"] = {
            "type": "string", "enum": ["red", "blue"]}
        image = image_content("blue")
        result = client.responses.create(**{
            **base, "tools": [{"type": "function", **fn}],
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text":
                 "Call record with value equal to this image's color."},
                {"type": "input_image", "image_url": image["image_url"]["url"]}]}]})
        signature(result, value="blue")
        record("responses_image_tool", result.to_dict())


def check_structured_limits(client, model, checks, vision=False):
    """Short boundary checks; valid prefixes may be incomplete at a limit."""
    schema = {"type": "object", "properties": {
        "text": {"type": "string", "const": 'é😀 "\\ ' * 24}},
        "required": ["text"], "additionalProperties": False}
    specification = {"name": "boundary", "strict": True, "schema": schema}
    common = dict(model=model, temperature=0, seed=79,
                  messages=[{"role": "user", "content": "Return the required object."}],
                  response_format={"type": "json_schema", "json_schema": specification},
                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)

    # Ragged concurrent budgets straddle the maximum speculative block width.
    bodies = [{**common, "max_completion_tokens": n} for n in range(1, 9)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda body: chat_result(
            client, body, body["max_completion_tokens"] % 2 == 0), bodies))
    for n, result in enumerate(results, 1):
        assert result["finish"] == "length", result
        assert result["usage"]["completion_tokens"] == n, result
        assert not result["tools"] and not result["reasoning"], result
    record("schema_ragged_limits", results)

    sampled = {**common, "temperature": .7, "top_p": .8, "max_completion_tokens": 7,
               "extra_body": {**common["extra_body"], "top_k": 20, "min_p": .05}}
    first = chat_result(client, sampled, True)
    replay = chat_result(client, sampled)
    assert (first["text"], first["finish"]) == (replay["text"], replay["finish"]), (first, replay)
    assert first["usage"]["completion_tokens"] == replay["usage"]["completion_tokens"] == 7
    record("schema_limited_sampled_replay", replay)

    for thinking in (False, True):
        result = chat_result(client, {
            **common, "max_completion_tokens": 3,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}, True)
        assert result["finish"] == "length" and result["usage"]["completion_tokens"] == 3, result
        record(f"schema_limited_thinking_{thinking}", result)
        body = dict(model=model, input="Return the required object.",
                    temperature=0, max_output_tokens=3,
                    reasoning={"effort": "low" if thinking else "none"},
                    text={"format": {"type": "json_schema", **specification}})
        # Use raw typed events: SDK parse helpers correctly reject incomplete JSON.
        with client.responses.create(**body, stream=True) as stream:
            events = list(stream)
        final = events[-1].response
        assert events[-1].type == "response.incomplete" and final.status == "incomplete", final
        assert final.incomplete_details.reason == "max_output_tokens", final
        assert final.usage.output_tokens == 3, final
        record(f"schema_responses_limit_{thinking}", final.to_dict())

    tool = {"type": "function", "function": {"name": "echo", "strict": True,
            "parameters": schema}}
    for n in (1, 2, 7, 16):
        result = chat_result(client, {**common, "tools": [tool], "tool_choice": "required",
                                     "max_completion_tokens": n}, n % 2 == 0)
        assert result["finish"] == "length" and result["usage"]["completion_tokens"] == n, result
        assert not result["text"] and not result["tools"], result
        record(f"schema_tool_limit_{n}", result)

    # A non-strict tool schema can be broader than the response-format subset.
    result = chat_result(client, {**common, "tools": [{
        "type": "function", "function": {"name": "optional_tool", "strict": False,
        "parameters": {"type": "object", "additionalProperties": True,
                       "dependentRequired": {"a": ["b"]}}}}],
        "max_completion_tokens": 2}, True)
    assert result["finish"] == "length" and result["usage"]["completion_tokens"] == 2, result
    record("schema_non_strict_tool", result)
    for parameters in (
        {"type": "object", "properties": {}, "required": []},
        {"type": "object", "properties": {"x": {"type": "integer"}},
         "additionalProperties": False},
    ):
        try:
            client.chat.completions.create(**{
                **common, "max_completion_tokens": 2, "tool_choice": "required",
                "tools": [{"type": "function", "function": {
                    "name": "invalid", "strict": True, "parameters": parameters}}]}, stream=True)
        except openai.BadRequestError:
            pass
        else:
            raise AssertionError("malformed strict tool schema was accepted")
    record("schema_invalid_strict_tools", {"status": 400, "cases": 2})
    if vision:
        result = chat_result(client, {**common, "max_completion_tokens": 2,
            "messages": [{"role": "user", "content": [
                image_content("red"), {"type": "text", "text": "Return the required object."}]}]}, True)
        assert result["finish"] == "length" and result["usage"]["completion_tokens"] == 2, result
        record("schema_image_limit", result)

    # Request-local grammar and budget must reset after incomplete generations.
    fresh_schema = {"type": "object", "properties": {"ok": {"type": "boolean", "const": True}},
                    "required": ["ok"], "additionalProperties": False}
    fresh = {**common, "max_completion_tokens": 32,
             "response_format": {"type": "json_schema", "json_schema": {
                 "name": "fresh", "strict": True, "schema": fresh_schema}}}
    resumed = chat_result(client, fresh, True)
    assert resumed["finish"] == "stop" and json.loads(resumed["text"]) == {"ok": True}, resumed
    repeated = chat_result(client, fresh)
    assert repeated["text"] == resumed["text"] and repeated["usage"]["cached_tokens"] > 0, repeated
    record("schema_after_limits", repeated)
    bounded = {"type": "object", "properties": {
        "label": {"type": "string", "pattern": "^é😀$", "minLength": 2, "maxLength": 2},
        "value": {"type": "number", "minimum": -.5, "maximum": .5, "multipleOf": .25}},
        "required": ["label", "value"], "additionalProperties": False}
    checked = chat_result(client, {**fresh,
        "messages": [{"role": "user", "content": "Return label é😀 and value 0.25."}],
        "max_completion_tokens": 64,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "unicode_bounds", "strict": True, "schema": bounded}}}, True)
    from jsonschema import Draft202012Validator
    assert checked["finish"] == "stop", checked
    Draft202012Validator(bounded).validate(json.loads(checked["text"]))
    record("schema_unicode_bounds", checked)
    from jsonschema import FormatChecker
    for name, constraints, prompt, expected in (
        ("alternative_length", {"pattern": "^(?:(?:ab){2}|c)$", "maxLength": 3},
         "Return ab. Start the value with ab.", "c"),
        ("lookahead_length", {"pattern": "^(?=a|c)(?:(?:ab){2}|c)$", "maxLength": 3},
         "Return ab. Start the value with ab.", "c"),
        ("pattern_format", {"pattern": r"^(?:999|127)\.0\.0\.1$", "format": "ipv4"},
         "Return the address 999.0.0.1, exactly.", "127.0.0.1"),
        ("ecmascript_word", {"pattern": r"^\w+$", "enum": ["é", "a"],
                            "maxLength": 1},
         "Return the accented letter é.", "a"),
    ):
        bounded = {"type": "object", "properties": {"x": {"type": "string", **constraints}},
                   "required": ["x"], "additionalProperties": False}
        checked = chat_result(client, {**fresh, "max_completion_tokens": 32,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": bounded}}}, True)
        assert checked["finish"] == "stop", checked
        value = json.loads(checked["text"])
        Draft202012Validator(bounded, format_checker=FormatChecker()).validate(value)
        assert value["x"] == expected, checked
        record("schema_" + name, checked)

    # ECMA-262 '$' does not accept a trailing newline without multiline mode.
    # Unsatisfiable enum/pattern intersections must fail before streaming.
    for constraints in (
        {"pattern": r"^\w+$", "enum": ["é"]},
        {"pattern": r"^\d+$", "enum": ["١"]},
        {"pattern": "^a$", "enum": ["a\n"]},
    ):
        invalid = {"type": "object", "properties": {"x": {"type": "string", **constraints}},
                   "required": ["x"], "additionalProperties": False}
        try:
            client.chat.completions.create(**{**fresh, "max_completion_tokens": 1,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "invalid_pattern_intersection", "strict": True, "schema": invalid}}},
                stream=True)
        except openai.BadRequestError:
            pass
        else:
            raise AssertionError("empty ECMA-262 enum/pattern intersection was accepted")
    record("schema_ecmascript_invalid_intersections", {"status": 400, "cases": 3})


def check_response(response, reasoning):
    assert response.status in ("completed", "incomplete"), response
    assert response.parallel_tool_calls is False
    assert response.tool_choice == "none" and response.tools == []
    usage = response.usage
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert 0 <= usage.input_tokens_details.cached_tokens <= usage.input_tokens
    assert usage.input_tokens_details.cache_write_tokens >= 0
    count = usage.output_tokens_details.reasoning_tokens
    assert 0 <= count <= usage.output_tokens
    assert (count > 0) == reasoning, usage
    if response.status == "completed":
        assert response.output_text.strip(), response
    else:
        assert response.incomplete_details.reason == "max_output_tokens"
    return {"status": response.status, "usage": usage.to_dict()}


def check_events(events, reasoning):
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert [e.type for e in events[:2]] == [
        "response.created", "response.in_progress"
    ]
    assert events[-1].type in ("response.completed", "response.incomplete")
    final = events[-1].response
    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert text == final.output_text
    return check_response(final, reasoning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:PORT/v1")
    parser.add_argument("--model", required=True, help="Gufo served model name")
    parser.add_argument("--expect-reasoning", action="store_true")
    parser.add_argument("--suite", choices=("all", "stops", "responses", "conversation", "structured", "structured-limits", "tools", "sampling-defaults"), default="all")
    parser.add_argument("--vision", action="store_true",
                        help="Add image checks; the server needs its matching --mmproj")
    parser.add_argument("--sampling-preset", choices=("qwen38", "deepseek4"),
                        help="Expected text defaults; required for sampling-defaults suite")
    parser.add_argument("--sampling-overrides", type=json.loads, default={},
                        help="JSON object of explicit server sampling CLI values")
    args = parser.parse_args()
    if args.suite == "sampling-defaults" and not args.sampling_preset:
        parser.error("--sampling-preset is required for sampling-defaults")
    if not isinstance(args.sampling_overrides, dict):
        parser.error("--sampling-overrides must be a JSON object")
    if args.vision and args.suite not in ("all", "conversation", "structured", "structured-limits", "tools", "sampling-defaults"):
        parser.error("--vision requires a conversation or structured-output suite")
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "::1")
            or url.path.rstrip("/") != "/v1" or url.username or url.password
            or url.query or url.fragment):
        parser.error("--base-url must be an explicit loopback HTTP /v1 endpoint")

    def local_only(request):
        assert request.url.host == url.hostname
        assert (request.url.port or 80) == (url.port or 80)

    async def async_local_only(request):
        local_only(request)

    options = dict(
        api_key="local-test", base_url=args.base_url, max_retries=0, timeout=120,
        _strict_response_validation=True,
    )
    prompt = "What is two plus two? Reply briefly."
    request = dict(
        model=args.model, input=prompt, temperature=0, max_output_tokens=256,
        store=False,
    )
    report = {"sdk": openai.__version__, "model": args.model, "checks": {}}
    checks = report["checks"]
    with OpenAI(**options, http_client=DefaultHttpxClient(
        trust_env=False, event_hooks={"request": [local_only]}
    )) as client:
        if args.suite == "structured":
            check_structured_outputs(client, args.model, checks, args.vision)
            print(json.dumps(report, indent=2))
            return
        if args.suite == "structured-limits":
            check_structured_limits(client, args.model, checks, args.vision)
            print(json.dumps(report, indent=2))
            return
        if args.suite == "tools":
            check_native_tools(client, args.model, checks, args.vision)
            print(json.dumps(report, indent=2))
            return
        if args.suite == "sampling-defaults":
            check_sampling_defaults(client, args.model, checks, args.sampling_preset,
                                    args.sampling_overrides, args.vision)
            print(json.dumps(report, indent=2))
            return
        if args.suite in ("all", "stops"):
            check_stops(client, args.model, checks)
        if args.suite in ("all", "conversation"):
            check_conversations(client, args.model, checks, args.vision)
        if args.suite in ("stops", "conversation"):
            print(json.dumps(report, indent=2))
            return
        first = client.responses.create(**request)
        assert first.status == "completed", first
        checks["create"] = check_response(first, args.expect_reasoning)
        with client.responses.stream(**request) as stream:
            events = list(stream)
            final = stream.get_final_response()
        checks["stream_helper"] = check_events(events, args.expect_reasoning)
        assert final.output_text == first.output_text
        assert final.usage.input_tokens_details.cached_tokens > 0

        history = [
            {"role": "user", "content": prompt},
            *first.output,
            {"role": "user", "content": "And two plus three? Reply briefly."},
        ]
        replay = client.responses.create(**{**request, "input": history})
        checks["conversation_replay"] = check_response(replay, args.expect_reasoning)
        assert replay.usage.input_tokens > first.usage.input_tokens

        # The SDK accumulation helper requires response.completed; consume
        # typed events directly when testing a deliberately truncated response.
        with client.responses.create(**{**request, "max_output_tokens": 1},
                                     stream=True) as stream:
            events = list(stream)
        checks["incomplete"] = check_events(events, args.expect_reasoning)
        assert events[-1].response.usage.output_tokens == 1
        if args.expect_reasoning:
            assert events[-1].response.usage.output_tokens_details.reasoning_tokens == 1

        for label, override in [
            ("invalid_limit", {"max_output_tokens": 0}),
            ("unsupported_store", {"store": True}),
            ("unsupported_tools", {"tools": [{"type": "web_search"}]}),
        ]:
            try:
                client.responses.create(**{**request, **override})
            except openai.BadRequestError as error:
                assert error.code == ("invalid_tools" if label == "unsupported_tools"
                                      else "invalid_request")
                checks[label] = {"status": error.status_code, "code": error.code}
            else:
                raise AssertionError(f"{label} was accepted")

        with client.responses.create(
            **{**request, "input": "Count from one to one thousand."}, stream=True
        ) as stream:
            for event in stream:
                if event.type.endswith(".delta"):
                    break
            else:
                raise AssertionError("No generation to cancel")
        # A fresh request must still run after closing the unfinished stream.
        after = client.responses.create(**{**request, "max_output_tokens": 1})
        checks["after_disconnect"] = check_response(after, args.expect_reasoning)
        chat_request = dict(
            model=args.model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_completion_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        chat = client.chat.completions.create(**chat_request)
        assert chat.choices[0].message.content
        checks["chat_completions"] = {"finish_reason": chat.choices[0].finish_reason}

    async def concurrent():
        async with AsyncOpenAI(**options, http_client=DefaultAsyncHttpxClient(
            trust_env=False, event_hooks={"request": [async_local_only]}
        )) as client:
            async def generate(index):
                body = {**request, "input": f"Count from {index + 1} to one hundred.",
                        "max_output_tokens": 8}
                if index == 0:
                    return check_response(await client.responses.create(**body),
                                          args.expect_reasoning)
                async with await client.responses.create(**body, stream=True) as stream:
                    events = [event async for event in stream]
                return check_events(events, args.expect_reasoning)
            return await asyncio.gather(generate(0), generate(1))

    checks["async_concurrent"] = asyncio.run(concurrent())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
