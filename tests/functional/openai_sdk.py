#!/usr/bin/env python3
"""Functional text API checks with the official OpenAI Python SDK.

Run with nix develop -c python3. Start a Gufo text server first; all requests
go to the explicitly supplied loopback endpoint.
"""

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import sys
import threading
import time
import traceback
from typing import Literal
import zlib
from urllib.parse import urlsplit

import openai
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI
from openai.types import Completion, CompletionChoice
from metrics import CaseComplete, Recorder
from tool_reasoning import check_reasoning_separator, check_tool_reasoning, response_result
from discovery import check_discovery
from image_inputs import check_image_count, check_image_inputs
from tool_native import check_finite_argument_types, check_native_tool_schemas
from tool_agent import (check_tool_agent, check_tool_agent_loop, check_tool_history,
                        check_untyped_agent_tools, check_mixed_tool_schemas,
                        check_tool_schema_edges)
from cache_edits import check_cache_edits
from cache_concurrency import check_cache_concurrency
from cache_shared_prefix import check_cache_shared_prefix
from cache_bridge import check_cache_bridge
from cache_compaction import check_cache_compaction
from cache_workloads import check_cache_transforms, check_cache_pressure
from cache_messages_loop import check_cache_messages_loop
from system_injection import check_system_injection
from cache_growth import check_cache_growth
from messages_tools import check_messages_tools
from cache_depth import check_cache_depth
from cache_rotation import check_cache_rotation
from prefill_scheduling import check_prefill_scheduling


class CompletionStreamChoice(CompletionChoice):
    # openai-python shares its completed-choice type with legacy stream chunks,
    # although in-progress chunks legitimately carry finish_reason:null.
    finish_reason: Literal["stop", "length", "content_filter"] | None


class CompletionStreamFrame(Completion):
    choices: list[CompletionStreamChoice]


class CheckResults(dict):
    """Persist completed cases even if a later request fails or is interrupted."""

    def __init__(self, report, output, recorder):
        super().__init__()
        self.report, self.output = report, output
        self.recorder = recorder

    def save(self):
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report, indent=2) + "\n")
            temporary.replace(self.output)

    def __setitem__(self, name, value):
        if name in self:
            raise AssertionError(f"duplicate functional check: {name}")
        super().__setitem__(name, value)
        self.recorder.mark(name)
        self.save()


def chat_result(client, request, streaming=False, on_chunk=None, on_open=None):
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
        if on_open:
            on_open()
        for chunk in result:
            if on_chunk:
                on_chunk(chunk)
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


def check_sampling_defaults(client, model, checks, preset, overrides, vision=False,
                            server_thinking=None):
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
        thinking = server_thinking != "off"
        effort = ("high" if preset == "deepseek4" else "xhigh") if thinking else "none"
        request = {
            "model": model, "seed": 73, "max_completion_tokens": 8,
            "messages": [{"role": "user", "content": "What is two plus two?"}],
            "extra_body": {"cache_prompt": False},
        }
        default = chat_result(client, request)
        explicit = chat_result(client, {**request, "reasoning_effort": effort}, True)
        assert bool(default["reasoning"]) == thinking, default
        assert signature(default) == signature(explicit), (default, explicit)
        record("default_thinking_effort", default)
        if preset == "deepseek4" and thinking:
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
        # A server seeded from entropy has no replay contract until a request
        # supplies a seed. Still test that explicit request seed overrides -1.
        inherited_seed = "seed" in overrides and overrides["seed"] >= 0
        if expected["seed"] < 0:
            expected["seed"] = 73
        request = {
            "model": model, "seed": 73, "max_completion_tokens": 8,
            "messages": [{"role": "user", "content":
                          "Name three animals and describe each in one sentence."}],
            "extra_body": {"cache_prompt": False, "chat_template_kwargs": {
                "enable_thinking": thinking}},
        }
        if inherited_seed:
            del request["seed"]
        explicit = {**request, **expected,
                    "extra_body": {**request["extra_body"], **native}}
        inherited = chat_result(client, request)
        supplied = chat_result(client, explicit, True)
        assert signature(inherited) == signature(supplied), (inherited, supplied)
        nullable = chat_result(client, {**request, "temperature": None,
                                       "top_p": None, "presence_penalty": None,
                                       "frequency_penalty": None,
                                       **({"seed": None} if inherited_seed else {})})
        assert signature(nullable) == signature(inherited), (nullable, inherited)
        record(f"sampling_preset_thinking{thinking}", inherited)

        # Each change targets one independent control, with and without all
        # other defaults written out; this also checks explicit neutral values.
        for label, override, extra in (
            ("temperature_only", {"temperature": .4}, {}),
            ("top_p_only", {"top_p": .6}, {}),
            ("top_k_only", {}, {"top_k": 0}),
            ("presence_only", {"presence_penalty": 0}, {}),
            ("frequency_only", {"frequency_penalty": .2}, {}),
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
        if not inherited_seed:
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
        for label, override, extra in (
            ("temperature_only", {"temperature": 0}, {}),
            ("top_p_only", {"top_p": .6}, {}),
            ("top_k_only", {}, {"top_k": 0}),
            ("presence_only", {}, {"presence_penalty": 0}),
            ("frequency_only", {}, {"frequency_penalty": .2}),
            ("min_p_only", {}, {"min_p": .1}),
        ):
            partial = {**response_request, **override, "extra_body": {
                **response_request["extra_body"], **extra}}
            full = {**explicit_response, **override, "extra_body": {
                **explicit_response["extra_body"], **extra}}
            first = client.responses.create(**partial)
            with client.responses.create(**full, stream=True) as stream:
                events = list(stream)
            check_events(events, thinking)
            assert response_signature(first) == response_signature(events[-1].response), (
                label, first, events[-1].response)
            record(f"responses_{label}_thinking{thinking}", check_response(first, thinking))
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

        # Raw Completions has no chat template control; its preset follows the
        # server's configured thinking mode rather than this Chat request.
        if thinking == (server_thinking != "off"):
            raw = dict(model=model, prompt="One, two, three,",
                       max_tokens=8, seed=73, extra_body={})
            if inherited_seed:
                del raw["seed"]
            full = {**raw, **expected, "extra_body": {**raw["extra_body"], **native}}
            for label, change in (("defaults", {}), ("temperature_only", {"temperature": 0}),
                                  ("top_p_only", {"top_p": .6})):
                first = completion_result(client, {**raw, **change})
                second = completion_result(client, {**full, **change}, True)
                assert first["text"] == second["text"] and first["finish"] == second["finish"], (
                    first, second)
                record("completions_" + label, first)


def completion_result(client, request, streaming=False, on_chunk=None):
    # Use the SDK's ordinary legacy stream path, then validate the wire shape
    # with only its in-progress finish_reason made nullable. Keep the original
    # shared client immutable; other threads still use strict Chat/Responses.
    stream_client = client.copy(_extra_kwargs={"_strict_response_validation": False})
    result = (stream_client if streaming else client).completions.create(**request, stream=streaming,
        **({"stream_options": {"include_usage": True}} if streaming else {}))
    if not streaming:
        return {"text": result.choices[0].text, "finish": result.choices[0].finish_reason,
                "usage": result.usage.to_dict()}
    text, finish, usage = "", None, None
    with result:
        for chunk in result:
            if on_chunk:
                on_chunk(chunk)
            CompletionStreamFrame.model_validate(chunk.to_dict())
            if chunk.usage:
                usage = chunk.usage.to_dict()
            for choice in chunk.choices:
                text += choice.text
                finish = choice.finish_reason or finish
    assert finish is not None and usage is not None, (finish, usage)
    return {"text": text, "finish": finish, "usage": usage}


def check_sampling_ranges(client, model, checks):
    """Malformed inputs fail before SSE starts, consistently across transports."""
    endpoints = {
        "chat": (client.chat.completions.create, {
            "model": model, "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 1}),
        "responses": (client.responses.create, {
            "model": model, "input": "Hello", "max_output_tokens": 1, "store": False}),
        "completions": (client.completions.create, {
            "model": model, "prompt": "Hello", "max_tokens": 1}),
    }
    invalid = (
        ("temperature", -1), ("temperature", 2.01),
        ("temperature", "hot"), ("temperature", True),
        ("top_p", -0.1), ("top_p", 1.01), ("top_p", "0.9"),
        ("top_p", 1.00000001), ("temperature", 2.00000001),
        ("top_k", -1), ("top_k", 1.5), ("min_p", -0.1), ("min_p", 1.1),
        ("repeat_penalty", 0), ("repeat_last_n", -1), ("seed", -2),
        ("presence_penalty", 2.1), ("frequency_penalty", -2.1),
        ("presence_penalty", 2.00000001), ("frequency_penalty", -2.00000001),
        ("draft_temperature", .7), ("typical_p", .9),
    )
    for endpoint, (create, body) in endpoints.items():
        for streaming in (False, True):
            for index, (name, value) in enumerate(invalid):
                try:
                    response = create(**body, stream=streaming, extra_body={name: value})
                except openai.BadRequestError as error:
                    assert error.status_code == 400 and error.code, error
                    checks[f"range_{endpoint}_{streaming}_{index}_{name}"] = {
                        "status": 400, "code": error.code, "value": value}
                else:
                    if streaming:
                        response.close()
                    raise AssertionError(f"{endpoint} accepted invalid {name}={value!r}")
        # Boundaries are valid; use greedy selection to avoid probabilistic assertions.
        for top_p in (0., .0001, 1.):
            response = create(**body, temperature=0, top_p=top_p)
            checks[f"range_{endpoint}_valid_{top_p}"] = response.to_dict()
        response = create(**body, temperature=.7, top_p=0, extra_body={"seed": 42})
        checks[f"range_{endpoint}_sampled_zero"] = response.to_dict()


def check_batches(client, model, checks, width, vision=False, speculative="off"):
    """Different endpoints/samplers/grammars share a physical batch without state leakage."""
    cases = []
    values = []
    for index in range(width):
        name = f"row_{index}"
        value = name + ": " + " ".join(f"item{n}" for n in range(16))
        values.append(value)
        body = dict(model=model, temperature=0 if index % 2 == 0 else .7, seed=101 + index,
                    top_p=.85, presence_penalty=.2, frequency_penalty=.1,
                    max_completion_tokens=48,
                    messages=[{"role": "user", "content": "Count from one to twenty."}],
                    extra_body={"cache_prompt": True, "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": False}})
        if index % 3 == 1:
            body.update(tools=[{"type": "function", "function": {
                "name": name, "parameters": {"type": "object", "properties": {
                    "value": {"type": "string", "const": value}},
                    "required": ["value"], "additionalProperties": False}}}],
                tool_choice="auto", parallel_tool_calls=False, max_completion_tokens=128,
                messages=[{"role": "user", "content": f"Call {name} once with its required value."}])
        elif index % 3 == 2:
            body.update(response_format={"type": "json_schema", "json_schema": {
                "name": name, "strict": True, "schema": {"type": "object",
                "properties": {"value": {"type": "string", "const": value}},
                "required": ["value"], "additionalProperties": False}}},
                max_completion_tokens=96,
                messages=[{"role": "user", "content": "Return JSON with the required value."}])
        if vision and index == width - 1:
            content = body["messages"][0]["content"]
            body["messages"][0]["content"] = [
                image_content("red"), {"type": "text", "text": content}]
        # Cache all prompts before measuring concurrency, so short prefills cannot
        # turn the intended physical batch into sequential requests.
        chat_result(client, {**body, "max_completion_tokens": 1})
        cases.append(body)
    # Each row must outlive admission of its peers. A seven-token schema can
    # finish in one speculative cycle and cannot qualify the requested width.
    start = threading.Barrier(width)
    def simultaneous(body):
        start.wait(timeout=30)
        return chat_result(client, body, True)
    with ThreadPoolExecutor(width) as pool:
        first = list(pool.map(simultaneous, cases))
    assert max(row["usage"]["gufo"]["physical_execution_width"] for row in first) == width, first
    for index, (body, row) in enumerate(zip(cases, first)):
        replay = chat_result(client, body)
        def signature(result):
            return result["text"], result["reasoning"], result["finish"], [
                call["function"] for call in result["tools"]]
        if speculative == "dspark" and body["temperature"] > 0:
            # Filtered DSpark C>1 uses p/q proposals, while C1 uses a point
            # mass. Both preserve p, but their same-seed traces can differ.
            # Test exact cache replay with the same C1 execution configuration.
            repeated = chat_result(client, body, True)
            assert signature(replay) == signature(repeated), (replay, repeated)
        else:
            assert signature(row) == signature(replay), (row, replay)
        assert replay["usage"]["cached_tokens"] > 0, replay
        if index % 3 == 1:
            assert row["tools"] and all(
                call["function"]["name"] == f"row_{index}"
                and json.loads(call["function"]["arguments"]) == {"value": values[index]}
                for call in row["tools"]), row
        elif index % 3 == 2:
            assert json.loads(row["text"]) == {"value": values[index]}, row
    checks[f"batch_c{width}_independent_samplers_and_grammars"] = first

    def cross_endpoint(index):
        # Each endpoint must retain its own seeded output when mixed with the
        # other transports. Their prompt templates need not be byte-identical.
        if index % 3 == 0:
            value = chat_result(client, cases[0], True)
            return (value["text"], value["reasoning"], value["finish"])
        if index % 3 == 1:
            request = dict(model=model, input="Count from one to twenty.", temperature=.7,
                           top_p=1 if speculative == "dspark" else .85,
                           max_output_tokens=24, store=False,
                           reasoning={"effort": "none"}, extra_body={
                               "seed": 101 + index,
                               **({"top_k": 0} if speculative == "dspark" else {})})
            with client.responses.create(**request, stream=True) as stream:
                events = list(stream)
            check_events(events, False)
            response = events[-1].response
            return response.output_text, response.status
        result = completion_result(client, dict(
            model=model, prompt="One, two, three,", temperature=.7,
            top_p=1 if speculative == "dspark" else .85, seed=101 + index,
            max_tokens=24, extra_body=(
                {"top_k": 0} if speculative == "dspark" else {})), True)
        return result["text"], result["finish"]
    indices = range(max(3, width))
    expected = [cross_endpoint(index) for index in indices]
    with ThreadPoolExecutor(width) as pool:
        actual = list(pool.map(cross_endpoint, indices))
    assert actual == expected, (actual, expected)
    checks[f"batch_c{width}_chat_responses_completions_replay"] = actual
    if width == 1:
        return

    def cancel_peer(index):
        if index:
            return cross_endpoint(index)
        pieces = 0
        with client.chat.completions.create(**cases[0], stream=True) as stream:
            for chunk in stream:
                pieces += sum(bool(choice.delta.content) for choice in chunk.choices)
                if pieces >= 3:
                    break
            else:
                raise AssertionError("batch cancellation never reached visible output")
        return None

    with ThreadPoolExecutor(width) as pool:
        cancelled = list(pool.map(cancel_peer, indices))
    assert cancelled[1:] == expected[1:], (cancelled, expected)
    assert cross_endpoint(0) == expected[0], "cancelled row failed to replay from cache"
    checks[f"batch_c{width}_cancelled_peer_isolation"] = cancelled

    def invalid_peer(index):
        if index:
            return cross_endpoint(index)
        try:
            client.chat.completions.create(**{**cases[0], "top_p": -0.1}, stream=True)
        except openai.BadRequestError:
            return None
        raise AssertionError("invalid peer entered generation")

    with ThreadPoolExecutor(width) as pool:
        rejected = list(pool.map(invalid_peer, indices))
    assert rejected[1:] == expected[1:], (rejected, expected)
    checks[f"batch_c{width}_rejected_peer_isolation"] = rejected


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

        # Close after SSE headers, before consuming output from a longer image
        # prefill. Reuse may keep only completed checkpoints, never partial state.
        request = body([
            image_content("blue"),
            {"type": "text", "text": "Name this image's dominant color in one sentence."},
        ], max_completion_tokens=16, presence_penalty=0)
        request["messages"].insert(0, {
            "role": "system", "content": "Follow the final instruction carefully. " * 256})
        with client.chat.completions.create(**request, stream=True):
            time.sleep(.05)
        request["extra_body"]["cache_prompt"] = True
        resumed = chat("vision_prefill_cancel_resume", request, True)
        warm = chat("vision_prefill_cancel_warm", request)
        cold = chat("vision_prefill_cancel_cold", {
            **request, "extra_body": {**request["extra_body"], "cache_prompt": False}})
        assert signature(resumed) == signature(warm) == signature(cold), (resumed, warm, cold)
        assert warm["usage"]["cached_tokens"] > 0, warm


def check_long_context(client, model, checks, context, vision=False):
    """A bounded, multi-turn agent history; actual depth is retained in usage."""
    if context < 8192:
        raise ValueError("long-context requires a server context of at least 8192")
    archive = ("Archive entry: this ordinary record has no new instructions.\n"
               * (context // 32))
    messages = [
        {"role": "system", "content": "Remember the secret keyword LANTERN.\n" + archive},
        {"role": "user", "content": "Return the secret keyword in the requested JSON format."},
    ]
    if vision:
        messages[-1]["content"] = [
            image_content("red"),
            {"type": "text", "text": "Return the secret keyword in the requested JSON format."},
        ]
    schema = {"type": "object", "properties": {"keyword": {"type": "string"}},
              "required": ["keyword"], "additionalProperties": False}
    extra = {"cache_prompt": True,
             "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
             "top_k": 20, "min_p": 0}
    request = dict(model=model, messages=messages, temperature=0, top_p=.95,
                   presence_penalty=0, frequency_penalty=0, seed=42,
                   max_completion_tokens=32, extra_body=extra,
                   response_format={"type": "json_schema", "json_schema": {
                       "name": "remembered_keyword", "strict": True, "schema": schema}})

    def chat(label, body, stream=False):
        result = chat_result(client, body, stream)
        checks[label] = result
        return result

    def signature(result):
        return result["text"], result["reasoning"].strip(), result["finish"], result["tools"]

    first = chat("long_schema_cold", request)
    assert json.loads(first["text"]) == {"keyword": "LANTERN"}, first
    depth = first["usage"]["prompt_tokens"]
    assert context // 4 <= depth < context - 512, depth
    replay = chat("long_schema_cached", request, True)
    assert signature(replay) == signature(first), replay
    assert replay["usage"]["gufo"]["prefill_tokens"] == 0, replay

    # Reuse the same multimodal history through the Responses SDK, with sampled
    # output and a new turn. JSON validity alone would not catch lost memory.
    history = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [{"type": "input_image", "image_url": item["image_url"]["url"]}
                       if item["type"] == "image_url" else
                       {"type": "input_text", "text": item["text"]} for item in content]
        history.append({**message, "content": content})
    history += [{"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "Give the same keyword again as JSON."}]
    responses = dict(model=model, input=history, temperature=.7, top_p=.9,
                     presence_penalty=0, seed=42, max_output_tokens=32, store=False,
                     reasoning={"effort": "none"}, extra_body={"top_k": 20},
                     text={"format": {"type": "json_schema", "name": "remembered_keyword",
                                      "strict": True, "schema": schema}})
    # seed/presence are Gufo extensions on Responses, not SDK keyword arguments.
    responses["extra_body"].update(seed=responses.pop("seed"),
                                    presence_penalty=responses.pop("presence_penalty"))
    response = client.responses.create(**responses)
    checks["long_responses_sampled"] = response.to_dict()
    assert response.status == "completed" and json.loads(response.output_text) == {
        "keyword": "LANTERN"}, response
    assert response.usage.input_tokens_details.cached_tokens >= depth - 128, response
    with client.responses.create(**responses, stream=True) as stream:
        events = list(stream)
    check_events(events, False)
    repeated = events[-1].response
    assert repeated.output_text == response.output_text, repeated
    checks["long_responses_sampled_replay"] = repeated.to_dict()

    messages = [*messages, {"role": "assistant", "content": first["text"]},
                 {"role": "user", "content": "Give the same keyword again as JSON."},
                 {"role": "assistant", "content": response.output_text},
                 {"role": "user", "content": "Explain how to calculate 123 times 456."}]
    thinking = {**request, "messages": messages, "max_completion_tokens": 8,
                "reasoning_effort": "high", "extra_body": {
                    **extra, "chat_template_kwargs": {
                        "enable_thinking": True, "preserve_thinking": True}}}
    limited = chat("long_thinking_token_limit", thinking, True)
    assert limited["finish"] == "length" and limited["reasoning"], limited
    # Switching thinking changes the template's leading instructions; that
    # frontier may legitimately miss. Subsequent same-mode continuation must hit.
    count = 0
    with client.chat.completions.create(
            **{**thinking, "max_completion_tokens": 128}, stream=True) as stream:
        for chunk in stream:
            count += sum(bool(getattr(c.delta, "reasoning_content", "")) for c in chunk.choices)
            if count >= 3:
                break
        else:
            raise AssertionError("long thinking stream never reached cancellation point")
    checks["long_thinking_cancel"] = {"reasoning_deltas": count}
    # Discard the interrupted assistant, as Pi does, then continue immediately.
    continued = {**thinking, "max_completion_tokens": 192,
                 "messages": [*messages, {"role": "user", "content":
                 "Return only the secret keyword as JSON; do not explain the arithmetic."}]}
    resumed = chat("long_cancel_resume", continued, True)
    assert json.loads(resumed["text"]) == {"keyword": "LANTERN"}, resumed
    assert resumed["usage"]["cached_tokens"] >= depth - 128, resumed
    assert resumed["usage"]["gufo"]["prefill_tokens"] < 128, resumed
    repeated = chat("long_cancel_replay", continued)
    assert signature(resumed) == signature(repeated), (resumed, repeated)
    stopped = chat("long_json_stop", {**request, "stop": "TERN"}, True)
    assert stopped["finish"] == "stop" and "TERN" not in stopped["text"], stopped
    assert stopped["text"] == first["text"].split("TERN")[0], stopped


def check_structured_outputs(client, model, checks, vision=False):
    from typing import Literal
    from pydantic import BaseModel, Field

    class Item(BaseModel):
        name: Literal["cat", "dog"]
        count: int = Field(ge=1, le=5)

    class Reply(BaseModel):
        items: list[Item] = Field(min_length=1, max_length=2)
        note: str | None

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
    for index, text in enumerate(("alpha", "beta", "alpha")):
        body = {**common, "response_format": constant(text)}
        result = chat_result(client, body, True)
        assert json.loads(result["text"]) == {"text": text}, result
        record(f"schema_cache_{index}_{text}", result)
    assert checks["schema_cache_2_alpha"]["usage"]["cached_tokens"] > 0

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
    # As in llama.cpp the call stays native (#438): one call, a string value
    # and no framing in the output, never a switch to a JSON envelope.
    delimiter = "\n</parameter>\n</｜DSML｜parameter>\\"
    fallback = json.loads(json.dumps(function))
    # Non-strict: the value cannot be generated natively, which the harness's
    # strict-schema validation would otherwise report (llama.cpp alike).
    fallback["strict"] = False
    fallback["parameters"]["properties"]["value"]["const"] = delimiter

    def native_signature(response):
        calls = [item for item in response.output if item.type == "function_call"]
        text = "".join(part.text for item in response.output if item.type == "message"
                       for part in item.content if part.type == "output_text")
        assert not any(tag in text for tag in ("<tool_call>", "<function=", "<parameter=",
                                               "<｜DSML｜", '{"name"')), response
        # As in llama.cpp nothing guides an unenforceable string value, so the
        # model may not finish it within the limit; a call it completes is one
        # native record call carrying a string.
        if response.status == "incomplete":
            assert response.incomplete_details.reason == "max_output_tokens", response
            return
        assert response.status == "completed" and len(calls) == 1, response
        assert calls[0].name == "record", response
        assert isinstance(json.loads(calls[0].arguments)["value"], str), response

    result = client.responses.create(**{
        **base, "tools": [{"type": "function", **fallback}]})
    native_signature(result)
    record("responses_tool_delimiter_native", result.to_dict())

    fallback["parameters"]["properties"]["value"] = {
        "type": "string", "pattern": "^\\n</parameter>$"}
    result = client.responses.create(**{
        **base, "tools": [{"type": "function", **fallback}]})
    native_signature(result)
    record("responses_tool_pattern_native", result.to_dict())

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


def check_tool_edges(client, model, checks, sampling_preset=None):
    """Exercise schema-to-native-to-JSON conversion through the real model."""
    expected = {"n": 42, "b": True, "a": [1], "o": {"x": 2}, "s": "42", "z": None}
    definitions = {
        "number/type~": {"type": "integer", "enum": [42]},
        "b": {"type": "boolean", "enum": [True]},
        "a": {"type": "array", "items": {"type": "integer"}, "enum": [[1]]},
        "o": {"type": "object", "properties": {"x": {"type": "integer", "enum": [2]}},
              "required": ["x"], "additionalProperties": False},
        "s": {"type": "string", "enum": ["42"]}, "z": {"type": "null"},
    }
    definitions["Arguments"] = {
        "type": "object", "properties": {
            key: {"$ref": "#/$defs/" + ("number~1type~0" if key == "n" else key)}
            for key in expected},
        "required": list(expected), "additionalProperties": False}
    function = {"name": "record", "strict": True, "parameters": {
        "$ref": "#/$defs/Arguments", "$defs": definitions}}
    prompt = "Call record once using its required values. Do not explain."
    common = dict(model=model, temperature=0,
                  extra_body={"seed": 41, "presence_penalty": 0})
    chat = dict(**common, messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": function}],
                tool_choice="required", parallel_tool_calls=False,
                reasoning_effort="none", max_completion_tokens=160)
    for stream in (False, True):
        result = chat_result(client, chat, stream)
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        arguments = json.loads(result["tools"][0]["function"]["arguments"])
        assert arguments == expected and type(arguments["b"]) is bool, arguments
        checks[f"chat_referenced_types_stream{stream}"] = result

    responses = dict(**common, input=prompt, tools=[{"type": "function", **function}],
                     tool_choice={"type": "function", "name": "record"},
                     parallel_tool_calls=None, reasoning={"effort": "none"},
                     max_output_tokens=160, store=False)
    for stream in (False, True):
        if stream:
            with client.responses.stream(**responses) as events:
                list(events)
                result = events.get_final_response()
        else:
            result = client.responses.create(**responses)
        calls = [item for item in result.output if item.type == "function_call"]
        assert result.status == "completed" and len(calls) == 1, result
        assert json.loads(calls[0].arguments) == expected, result
        assert result.tool_choice.type == "function" and result.tool_choice.name == "record"
        assert result.parallel_tool_calls is False
        checks[f"responses_referenced_types_stream{stream}"] = result.to_dict()

    for key in ("value", " value "):
        literal = "literal\r"
        fn = {"name": "record", "strict": True, "parameters": {
            "type": "object", "properties": {key: {"type": "string", "const": literal}},
            "required": [key], "additionalProperties": False}}
        result = client.responses.create(**{
            **responses, "tools": [{"type": "function", **fn}]})
        calls = [item for item in result.output if item.type == "function_call"]
        assert result.status == "completed" and len(calls) == 1, result
        assert json.loads(calls[0].arguments) == {key: literal}, result
        checks[f"responses_literal_cr_key{key!r}"] = result.to_dict()

    # Prose may quote another dialect's envelope before a real call. Only the
    # admitted format's opener starts tool output; the quote stays text.
    read = {"name": "read", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"]}}
    literal = "<tool_calls></tool_calls>"
    prompt = (f"Reply with the exact text {literal} on the first line, then call "
              "the read tool with path fixture.xml.")
    for stream in (False, True):
        result = chat_result(client, dict(
            **common, messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": read}], tool_choice="auto",
            reasoning_effort="none", max_completion_tokens=200), stream)
        # Qwen fixtures emit the literal, so their run exercises the marker
        # case. Other models may omit it: record that rather than claim it.
        exercised = literal in result["text"]
        assert exercised or sampling_preset != "qwen38", result
        assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
        function = result["tools"][0]["function"]
        assert function["name"] == "read", result
        assert json.loads(function["arguments"]) == {"path": "fixture.xml"}, result
        assert not any(marker in result["text"] for marker in
                       ("<tool_call>", "<function=", "</parameter>")), result
        checks[f"foreign_marker_prose_stream{stream}"] = {
            **result, "foreign_marker_exercised": exercised}

    # Parallel calls must survive the native framing. As in llama.cpp, a
    # DeepSeek call block also ends the output, so no text streams after it.
    prompt = ("Call read twice in parallel, with path a.txt and with path b.txt. "
              "After the calls, write Done.")
    for endpoint in ("chat", "responses"):
        for stream in (False, True):
            order = []
            if endpoint == "chat":
                result = chat_result(client, dict(
                    **common, messages=[{"role": "user", "content": prompt}],
                    tools=[{"type": "function", "function": read}], tool_choice="auto",
                    parallel_tool_calls=True, reasoning_effort="none",
                    max_completion_tokens=256), stream, on_chunk=lambda chunk: order.extend(
                        "tool" if choice.delta.tool_calls else "text"
                        for choice in chunk.choices
                        if choice.delta.tool_calls or choice.delta.content))
            else:
                result = response_result(client, dict(
                    **common, input=prompt, tools=[{"type": "function", **read}],
                    tool_choice="auto", parallel_tool_calls=True,
                    reasoning={"effort": "none"}, max_output_tokens=256, store=False), stream)
            name = f"parallel_calls_{endpoint}_stream{stream}"
            checks[name] = {**result, "delta_order": order}
            print(f"CHECK {name}", file=sys.stderr, flush=True)
            assert result["finish"] == "tool_calls", result
            assert all(call["function"]["name"] == "read" for call in result["tools"]), result
            paths = sorted(json.loads(call["function"]["arguments"])["path"]
                           for call in result["tools"])
            assert paths == ["a.txt", "b.txt"], result
            assert "DSML" not in result["text"] and "<tool_call>" not in result["text"], result
            if sampling_preset == "deepseek4" and "tool" in order:
                assert "text" not in order[order.index("tool"):], result


def check_state_edges(client, model, checks, speculative, vision=False):
    """Mode proof and request-local grammar state across limits, errors and reuse."""
    common = dict(model=model, temperature=0, top_p=1, seed=91,
                  extra_body={"presence_penalty": 0, "frequency_penalty": 0,
                              "repeat_penalty": 1, "top_k": 0, "min_p": 0,
                              "chat_template_kwargs": {"enable_thinking": False}})
    probe = chat_result(client, {
        **common, "messages": [{"role": "user", "content":
        "Count from one to one hundred, separated by commas. Start with 1."}],
        "max_completion_tokens": 32}, True)
    assert probe["usage"]["completion_tokens"] == 32 and probe["finish"] == "length", probe
    proposed = probe["usage"]["draft_tokens"]
    assert (proposed == 0 if speculative == "off" else proposed > 0), (speculative, probe)
    checks["execution_mode_probe"] = probe

    empty = {"name": "ping", "strict": True}  # Omitted parameters means no arguments.
    tiny = chat_result(client, {
        **common, "messages": [{"role": "user", "content": "Call ping."}],
        "tools": [{"type": "function", "function": empty}], "tool_choice": "required",
        "max_completion_tokens": 1, "reasoning_effort": "low",
        "extra_body": {**common["extra_body"],
                       "chat_template_kwargs": {"enable_thinking": True}}}, True)
    assert tiny["finish"] == "length" and not tiny["tools"], tiny
    checks["thinking_tool_one_token"] = tiny

    options = {key: value for key, value in common.items() if key != "seed"}
    options["extra_body"] = {key: value for key, value in common["extra_body"].items()
                             if key != "chat_template_kwargs"}
    options["extra_body"]["seed"] = common["seed"]
    options.update(reasoning={"effort": "none"}, store=False, max_output_tokens=96)
    with client.responses.stream(**{
            **options, "input": "Call ping.", "tools": [{"type": "function", **empty}],
            "tool_choice": {"type": "function", "name": "ping"}}) as events:
        list(events)
        result = events.get_final_response()
    calls = [item for item in result.output if item.type == "function_call"]
    assert result.status == "completed" and len(calls) == 1
    assert calls[0].name == "ping" and json.loads(calls[0].arguments) == {}, result
    checks["zero_argument_tool_after_limit"] = result.to_dict()

    def function(value):
        return {"name": "record", "strict": True, "parameters": {
            "type": "object", "properties": {
                "payload": {"type": "object", "properties": {
                    "id": {"type": "string"}, "call_id": {"type": "string"}},
                    "required": ["id", "call_id"], "additionalProperties": False,
                    "enum": [{"id": value, "call_id": "literal"}]}},
            "required": ["payload"], "additionalProperties": False}}

    # The name and prompt are identical; only data inside the schema changes.
    for value in ("alpha", "beta"):
        with client.responses.stream(**{
                **options, "input": "Call record with its required payload.",
                "temperature": .7 if value == "beta" else 0,
                "top_p": .8, "tools": [{"type": "function", **function(value)}],
                "tool_choice": {"type": "function", "name": "record"}}) as events:
            list(events)
            result = events.get_final_response()
        calls = [item for item in result.output if item.type == "function_call"]
        assert result.status == "completed" and len(calls) == 1, result
        assert json.loads(calls[0].arguments) == {
            "payload": {"id": value, "call_id": "literal"}}, result
        checks[f"schema_payload_{value}"] = result.to_dict()

    content = "Call record with its required payload."
    if vision:
        content = [image_content("red"), {"type": "text", "text": content}]
    request = {**common, "messages": [{"role": "user", "content": content}],
               "tools": [{"type": "function", "function": function("alpha")}],
               "tool_choice": {"type": "function", "function": {"name": "record"}},
               "parallel_tool_calls": False, "max_completion_tokens": 96}
    stopped = chat_result(client, {**request, "stop": "alpha"}, True)
    assert stopped["finish"] == "stop" and not stopped["tools"], stopped
    checks["tool_argument_stop"] = stopped

    # A stop inside an argument leaves the call unfinished. A call quoted in
    # that value is argument data and must not become a separate call. Native
    # formats keep native framing here too, as llama.cpp (#438).
    literal = ("<tool_call><function=record><parameter=content>AAAA</parameter>"
               "</function></tool_call>")
    quoted = {"name": "record", "parameters": {"type": "object", "properties": {
        "content": {"type": "string", "const": literal + " ZZSTOP"},
        "tag": {"type": "string", "pattern": "^[a-z]+$"}},
        "required": ["content"], "additionalProperties": False}}
    quoted_request = {**common, "messages": [{"role": "user", "content": "Call record."}],
                      "tools": [{"type": "function", "function": quoted}],
                      "tool_choice": {"type": "function", "function": {"name": "record"}},
                      "max_completion_tokens": 96}
    for stream in (False, True):
        result = chat_result(client, {**quoted_request, "stop": "ZZSTOP"}, stream)
        assert result["finish"] == "stop" and not result["tools"], result
        assert "<tool_call>" not in result["text"] and "AAAA" not in result["text"], result
        checks[f"quoted_call_argument_stop_stream{stream}"] = result
    result = chat_result(client, quoted_request, True)
    assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
    assert json.loads(result["tools"][0]["function"]["arguments"])["content"] == (
        literal + " ZZSTOP"), result
    checks["quoted_call_argument_complete"] = result

    completed = chat_result(client, request, True)
    assert completed["finish"] == "tool_calls" and len(completed["tools"]) == 1, completed
    assert json.loads(completed["tools"][0]["function"]["arguments"]) == {
        "payload": {"id": "alpha", "call_id": "literal"}}, completed
    assert completed["usage"]["cached_tokens"] > 0, completed
    checks["tool_argument_stop_retry"] = completed

    invalid = {"name": "record", "strict": True, "parameters": {"$ref": "#"}}
    try:
        client.chat.completions.create(**{
            **request, "tools": [{"type": "function", "function": invalid}]})
    except openai.BadRequestError as error:
        assert error.code == "invalid_tools", error
        checks["cyclic_schema_rejected"] = {"status": 400, "code": error.code}
    else:
        raise AssertionError("nonproductive schema cycle was accepted")
    replay = chat_result(client, request)
    assert replay["tools"][0]["function"] == completed["tools"][0]["function"], replay
    assert replay["usage"]["gufo"]["prefill_tokens"] == 0, replay
    checks["schema_error_did_not_poison_cache"] = replay


def check_auto_tools(client, model, checks, vision=False):
    """Automatic calls retain prose, loose schemas, replay and both transports."""
    from concurrent.futures import ThreadPoolExecutor

    function = {"name": "record", "strict": False, "parameters": {
        "type": "object", "properties": {
            "value": {"type": "string"}, "optional": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False}}
    prompt = "Call record exactly once with value alpha. Omit optional. No explanation."
    common = dict(model=model, tools=[{"type": "function", "function": function}],
                  tool_choice="auto", parallel_tool_calls=True, temperature=0,
                  seed=61, max_completion_tokens=160,
                  messages=[{"role": "user", "content": prompt}],
                  extra_body={"presence_penalty": 0,
                              "chat_template_kwargs": {"enable_thinking": False}})

    def record(name, value):
        checks[name] = value
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        return value

    def signature(result):
        assert result["finish"] == "tool_calls" and result["tools"], result
        calls = [item["function"] for item in result["tools"]]
        assert all(call["name"] == "record"
                   and json.loads(call["arguments"]) == {"value": "alpha"}
                   for call in calls), result
        return result["text"], result["reasoning"], calls

    for sampled in (False, True):
        request = {**common, "temperature": .7 if sampled else 0}
        if sampled:
            request = {**request, "top_p": .85, "presence_penalty": .3,
                       "frequency_penalty": .2, "extra_body": {
                           **common["extra_body"], "top_k": 20, "min_p": .05,
                           "repeat_penalty": 1.1}}
        first = chat_result(client, request)
        replay = chat_result(client, request, True)
        assert signature(first) == signature(replay), (first, replay)
        assert replay["usage"]["cached_tokens"] > 0, replay
        record(f"auto_chat_sampled{sampled}_cache_replay", replay)
    prose = {**common, "messages": [{"role": "user", "content":
             "Do not call any tools. Reply with the single word Hello."}]}
    for thinking in (False, True):
        request = {**prose, "max_completion_tokens": 96, "extra_body": {
            "presence_penalty": 0,
            "chat_template_kwargs": {"enable_thinking": thinking}}}
        if thinking:
            request["reasoning_effort"] = "low"
        result = chat_result(client, request, True)
        assert not result["tools"] and "Hello" in result["text"], result
        record(f"auto_prose_thinking{thinking}", result)
    thought = chat_result(client, {
        **common, "reasoning_effort": "high", "max_completion_tokens": 256,
        "extra_body": {"presence_penalty": 0}}, True)
    signature(thought)
    record("auto_chat_thinking_high", thought)
    for limit in (2, 12):
        result = chat_result(client, {**common, "max_completion_tokens": limit}, True)
        assert result["finish"] == "length" and not result["tools"], result
        record(f"auto_chat_limit{limit}", result)
    stopped = chat_result(client, {**common, "stop": "alpha"}, True)
    assert stopped["finish"] == "stop" and not stopped["tools"], stopped
    signature(chat_result(client, common, True))
    record("auto_argument_stop_retry", stopped)

    response_request = dict(
        model=model, input=prompt, tools=[{"type": "function", **function}],
        tool_choice="auto", parallel_tool_calls=True, max_output_tokens=160,
        temperature=.7, top_p=.85, reasoning={"effort": "none"}, store=False,
        extra_body={"seed": 61, "top_k": 20, "presence_penalty": 0})

    def response_signature(result, value="alpha"):
        calls = [item for item in result.output if item.type == "function_call"]
        # Parallel auto permits more than one call; cardinality is enforced
        # separately by the nonparallel suite, not by prompt obedience.
        assert result.status == "completed" and calls, result
        assert all(call.name == "record" and json.loads(call.arguments) == {"value": value}
                   for call in calls), result
        return [(call.name, call.arguments) for call in calls]

    result = client.responses.create(**response_request)
    with client.responses.stream(**response_request) as stream:
        list(stream)
        replay = stream.get_final_response()
    assert response_signature(result) == response_signature(replay)
    record("auto_responses_sampled_replay", replay.to_dict())
    for name, schema in (
            ("open_object", {"type": "object", "additionalProperties": True}),
            ("unsupported_schema", {"type": "object", "unevaluatedProperties": True})):
        loose = {**function, "parameters": schema}
        result = client.responses.create(**{
            **response_request, "temperature": 0,
            "parallel_tool_calls": True,
            "tools": [{"type": "function", **loose}],
            "input": "Call record exactly once with the JSON object {\"value\":\"alpha\"}, then stop."})
        response_signature(result)
        # This prompt produces one call on the unconstrained baseline. A
        # previous JSON-envelope fallback instead looped until the token cap.
        assert len([item for item in result.output if item.type == "function_call"]) == 1, result
        record("auto_" + name, result.to_dict())

    def independent(index):
        name = f"record_{index}"
        request = {**common, "seed": 70 + index,
                   "tools": [{"type": "function", "function": {**function, "name": name}}],
                   "messages": [{"role": "user", "content":
                                 f"Call {name} exactly once with value alpha. Omit optional."}]}
        result = chat_result(client, request, True)
        assert result["tools"] and all(
            call["function"]["name"] == name
            and json.loads(call["function"]["arguments"]) == {"value": "alpha"}
            for call in result["tools"]), result
        return result

    with ThreadPoolExecutor(4) as pool:
        record("auto_c4_independent_schemas", list(pool.map(independent, range(4))))
    many_tools = [{"type": "function", "name": f"record_{index}", "strict": False,
                   "parameters": {"type": "object", "properties": {
                       "value": {"type": "integer"}}, "required": ["value"],
                       "additionalProperties": False}} for index in range(32)]
    result = client.responses.create(**{
        **response_request, "temperature": 0, "tools": many_tools,
        "parallel_tool_calls": False,
        "input": "Call record_31 exactly once with value 7."})
    calls = [item for item in result.output if item.type == "function_call"]
    assert len(calls) == 1 and calls[0].name == "record_31", result
    assert json.loads(calls[0].arguments) == {"value": 7}, result
    record("auto_many_declared_tools", result.to_dict())
    if vision:
        result = client.responses.create(**{
            **response_request, "temperature": 0,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text":
                 "Call record with value equal to the image color. Omit optional."},
                {"type": "input_image", "image_url":
                 image_content("blue")["image_url"]["url"]}]}]})
        response_signature(result, value="blue")
        record("auto_image_tool", result.to_dict())
        image_request = {**common, "messages": [{"role": "user", "content": [
            image_content("blue"), {"type": "text", "text":
            "Call record with value equal to the image color. Omit optional."}]}]}
        stopped = chat_result(client, {**image_request, "stop": "blue"}, True)
        assert stopped["finish"] == "stop" and not stopped["tools"], stopped
        retry = chat_result(client, image_request, True)
        assert retry["usage"]["cached_tokens"] > 0 and retry["tools"], retry
        assert all(call["function"]["name"] == "record"
                   and json.loads(call["function"]["arguments"]) == {"value": "blue"}
                   for call in retry["tools"]), retry
        record("auto_image_argument_stop_retry", {"stopped": stopped, "retry": retry})


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

    # Greedy GPU winners include penalties, but still obey the tool grammar.
    # Invalid speculative proposals must not mutate its state while staging
    # conditional penalty histories.
    for choice in ("auto", "required"):
        request = dict(model=model, temperature=0, seed=79,
            presence_penalty=1.5, frequency_penalty=.2,
            max_completion_tokens=96, tool_choice=choice, parallel_tool_calls=False,
            messages=[{"role": "user", "content": "Call echo with text alpha."}],
            tools=[{"type": "function", "function": {"name": "echo", "strict": True,
                "parameters": {"type": "object", "properties": {
                    "text": {"type": "string", "const": "alpha"}},
                    "required": ["text"], "additionalProperties": False}}}],
            extra_body={"repeat_penalty": 1.1,
                        "chat_template_kwargs": {"enable_thinking": False}})
        first = chat_result(client, request, True)
        replay = chat_result(client, request)
        for result in (first, replay):
            assert result["finish"] == "tool_calls" and len(result["tools"]) == 1, result
            call = result["tools"][0]["function"]
            assert call["name"] == "echo" and json.loads(call["arguments"]) == {"text": "alpha"}, result
        assert replay["usage"]["cached_tokens"] > 0, replay
        record(f"schema_greedy_tool_penalties_{choice}", [first, replay])

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
    assert events[-1].type in ("response.completed", "response.incomplete"), [
        event.to_dict() for event in events[-3:]]
    final = events[-1].response
    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert text == final.output_text
    return check_response(final, reasoning)


def check_responses(client, model, checks, options, async_local_only, expect_reasoning):
    prompt = "What is two plus two? Reply briefly."
    request = dict(model=model, input=prompt, temperature=0, max_output_tokens=256,
                   reasoning={"effort": "low" if expect_reasoning else "none"}, store=False)
    first = client.responses.create(**request)
    assert first.status == "completed", first
    checks["create"] = check_response(first, expect_reasoning)
    with client.responses.stream(**request) as stream:
        events = list(stream)
        final = stream.get_final_response()
    checks["stream_helper"] = check_events(events, expect_reasoning)
    assert final.output_text == first.output_text
    assert final.usage.input_tokens_details.cached_tokens > 0

    history = [
        {"role": "user", "content": prompt},
        *first.output,
        {"role": "user", "content": "And two plus three? Reply briefly."},
    ]
    replay = client.responses.create(**{**request, "input": history})
    checks["conversation_replay"] = check_response(replay, expect_reasoning)
    assert replay.usage.input_tokens > first.usage.input_tokens

    # The SDK accumulation helper requires response.completed; consume
    # typed events directly when testing a deliberately truncated response.
    with client.responses.create(**{**request, "max_output_tokens": 1},
                                 stream=True) as stream:
        events = list(stream)
    checks["incomplete"] = check_events(events, expect_reasoning)
    assert events[-1].response.usage.output_tokens == 1
    if expect_reasoning:
        assert events[-1].response.usage.output_tokens_details.reasoning_tokens == 1

    for label, override in [
        ("invalid_limit", {"max_output_tokens": 0}),
        ("unsupported_store", {"store": True}),
        ("unsupported_tools", {"tools": [{"type": "namespace", "name": "crm"}]}),
    ]:
        try:
            client.responses.create(**{**request, **override})
        except openai.BadRequestError as error:
            assert error.code == ("invalid_tools" if label == "unsupported_tools"
                                  else "invalid_request")
            checks[label] = {"status": error.status_code, "code": error.code}
        else:
            raise AssertionError(f"{label} was accepted")
    # Hosted tools only OpenAI can run are skipped, as llama.cpp does.
    hosted = client.responses.create(**{**request, "max_output_tokens": 1,
                                        "tools": [{"type": "web_search"}]})
    assert hosted.tools == [], hosted
    checks["hosted_tools_skipped"] = {"status": hosted.status}

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
    checks["after_disconnect"] = check_response(after, expect_reasoning)
    chat_request = dict(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_completion_tokens=16,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    chat = client.chat.completions.create(**chat_request)
    assert chat.choices[0].message.content
    checks["chat_completions"] = {"finish_reason": chat.choices[0].finish_reason}

    async def concurrent():
        async with AsyncOpenAI(**options, http_client=DefaultAsyncHttpxClient(
            trust_env=False, event_hooks={
                "request": [async_local_only, checks.recorder.async_request_hook],
                "response": [checks.recorder.async_response_hook]}
        )) as client:
            async def generate(index):
                body = {**request, "input": f"Count from {index + 1} to one hundred.",
                        "max_output_tokens": 8}
                if index == 0:
                    return check_response(await client.responses.create(**body),
                                          expect_reasoning)
                async with await client.responses.create(**body, stream=True) as stream:
                    events = [event async for event in stream]
                return check_events(events, expect_reasoning)
            return await asyncio.gather(generate(0), generate(1))

    checks["async_concurrent"] = asyncio.run(concurrent())


def check_stream_start(client, model, checks, width, context):
    from stream_start import check_stream_start as check

    check(client, model, checks, width, context)


def check_prompt_progress(client, model, checks, width, vision, allow_missing):
    from progress import ProgressTrace
    from server_metrics import ServerMetrics, assert_accounting

    metrics = ServerMetrics(client.base_url)
    before = metrics.idle()
    request_start = len(checks.recorder.rows)

    prompt = "Count from one to one hundred, separated by commas."
    common = dict(model=model, temperature=0, seed=42, max_completion_tokens=24,
                  messages=[{"role": "system", "content": "Follow the user's instructions. " * 48},
                            {"role": "user", "content": prompt}],
                  extra_body={"presence_penalty": 0, "cache_prompt": False,
                              "chat_template_kwargs": {"enable_thinking": False}})

    def chat(name, request, enabled=True):
        trace = ProgressTrace(enabled is True, allow_missing)
        result = chat_result(client, {**request, "extra_body": {
            **request.get("extra_body", {}), "return_progress": enabled}}, True, trace)
        trace.finish(checks.recorder.rows[-1]["metrics"])
        checks[name] = {**result, "progress": trace.updates}
        return result

    def signature(result):
        return result["text"], result["reasoning"], result["tools"], result["finish"]

    off = chat("progress_cold_off", common, False)
    on = chat("progress_cold_on", common)
    assert signature(off) == signature(on), (off, on)
    cached = {**common, "extra_body": {**common["extra_body"], "cache_prompt": True}}
    replay = chat("progress_cached", cached)
    assert signature(on) == signature(replay)
    assert replay["usage"]["gufo"]["prefill_tokens"] == 0
    sampled = {**cached, "temperature": .7, "top_p": .85, "presence_penalty": .3,
               "frequency_penalty": .2, "extra_body": {
                   **cached["extra_body"], "presence_penalty": .3, "top_k": 20, "min_p": .05}}
    a = chat("progress_sampled_off", sampled, False)
    b = chat("progress_sampled_on", sampled)
    assert signature(a) == signature(b), (a, b)
    thinking = {**cached, "max_completion_tokens": 2, "reasoning_effort": "high",
                "extra_body": {**cached["extra_body"], "chat_template_kwargs": {"enable_thinking": True}}}
    a = chat("progress_thinking_limit_off", thinking, False)
    b = chat("progress_thinking_limit_on", thinking)
    assert signature(a) == signature(b) and b["usage"]["completion_tokens"] == 2
    assert off["text"], off
    stopped = chat("progress_stop", {**cached, "stop": off["text"][:1]})
    assert stopped["finish"] == "stop"
    assert signature(chat("progress_after_stop", cached, None)) == signature(on)

    trace = ProgressTrace(True, allow_missing)
    result = completion_result(client, dict(
        model=model, prompt="One, two, three,", max_tokens=16, temperature=0,
        extra_body={"return_progress": True}), True, trace)
    trace.finish(checks.recorder.rows[-1]["metrics"])
    checks["progress_completions"] = {**result, "progress": trace.updates}
    trace = ProgressTrace(True, allow_missing)
    with client.responses.create(model=model, input=prompt, max_output_tokens=16,
                                 temperature=0, reasoning={"effort": "none"}, stream=True,
                                 extra_body={"return_progress": True}) as stream:
        events = []
        for event in stream:
            trace(event)
            events.append(event)
    trace.finish(checks.recorder.rows[-1]["metrics"])
    checks["progress_responses"] = {"result": check_events(events, False), "progress": trace.updates}

    if vision:
        image_request = {**cached, "messages": [{"role": "user", "content": [
            image_content("red"), {"type": "text", "text": "Name the color of this image."}]}]}
        a = chat("progress_image_off", image_request, False)
        b = chat("progress_image_on_cached", image_request)
        assert signature(a) == signature(b) and "red" in b["text"].lower()
        assert b["usage"]["cached_tokens"] > 0

    # Response headers precede model output, unlike coalesced body chunks.
    # Use them to order admissions while still overlapping prefills, so
    # main/PR timings compare the same queue position.
    admitted = [threading.Event() for _ in range(width)]

    def peer(index):
        if index:
            assert admitted[index - 1].wait(30), "peer stream did not start"
        trace = ProgressTrace(index % 2 == 0, allow_missing)

        request = {**cached, "extra_body": {**cached["extra_body"],
                                          "return_progress": index % 2 == 0}}
        result = chat_result(client, request, True, trace, admitted[index].set)
        return result, trace
    with ThreadPoolExecutor(width) as pool:
        results = list(pool.map(peer, range(width)))
    for result, trace in results:
        assert signature(result) == signature(on)
        # Usage is request-local; recorder completion order is concurrent here.
        trace.finish({"prompt_tokens": result["usage"]["prompt_tokens"],
                      "cached_tokens": result["usage"]["cached_tokens"]})
    assert_accounting(before, metrics.idle(), checks.recorder.rows[request_start:])
    checks["progress_batch"] = [{"result": result, "progress": trace.updates}
                               for result, trace in results]

    # Cancel during prompt processing, before text/reasoning; then retry the
    # same history. A revision without progress uses its first output chunk as
    # the control cancellation point and is never a feature qualification.
    interrupted = {**common, "max_completion_tokens": 256, "extra_body": {
        **common["extra_body"], "return_progress": True},
        "messages": [{"role": "system", "content": "Remember this context. " * 512},
                     {"role": "user", "content": prompt}]}
    received = False
    with client.chat.completions.create(**interrupted, stream=True) as stream:
        for chunk in stream:
            if getattr(chunk, "prompt_progress", None) is not None:
                received = True
                break
            if allow_missing and any(choice.delta.content for choice in chunk.choices):
                break
        else:
            raise AssertionError("did not reach cancellation point")
    assert received or allow_missing
    checks["progress_cancel"] = {"progress_received": received}
    recovered = chat("progress_cancel_resume", {
        **interrupted, "max_completion_tokens": 8,
        "extra_body": {**interrupted["extra_body"], "cache_prompt": True}})
    replay = chat("progress_cancel_replay", {
        **interrupted, "max_completion_tokens": 8,
        "extra_body": {**interrupted["extra_body"], "cache_prompt": True}})
    assert signature(recovered) == signature(replay)
    assert replay["usage"]["gufo"]["prefill_tokens"] == 0


def check_server_metrics(client, model, checks, width, context, speculative):
    from server_metrics import (ServerMetrics, assert_accounting, PROMPT, GENERATED,
                                PROCESSING, DEFERRED, PROMPT_SPEED, GENERATED_SPEED,
                                KV_USAGE, DRAFT_ROUNDS, DRAFTS, ACCEPTED, assert_slots)

    metrics = ServerMetrics(client.base_url)
    initial = metrics.idle()
    assert metrics.read("/v1/metrics") == initial
    def slots(path="/slots", *, idle=False):
        snapshot = metrics.slots(path)
        assert_slots(snapshot, width, model, context, speculative != "off")
        if idle:
            assert not any(slot["is_processing"] for slot in snapshot), snapshot
        return snapshot

    idle_slots = slots(idle=True)
    assert slots("/v1/slots", idle=True) == idle_slots
    assert initial[KV_USAGE] == 0, initial
    prompt = "Count from one to one hundred, with no explanation."
    common = dict(model=model, temperature=0, seed=42, max_completion_tokens=16,
                  messages=[{"role": "user", "content": prompt}],
                  extra_body={"presence_penalty": 0, "cache_prompt": True,
                              "chat_template_kwargs": {"enable_thinking": False}})

    def completed(name, operation):
        before = metrics.idle()
        start = len(checks.recorder.rows)
        result = operation()
        after = metrics.idle()
        rows = checks.recorder.rows[start:]
        assert rows, "accounting check performed no requests"
        assert_accounting(before, after, rows)
        assert after[KV_USAGE] == 0, after
        checks[name] = {"before": before, "after": after, "result": result,
                        "slots": slots(idle=True)}
        return result, rows, after

    first, _, cold = completed(
        "metrics_chat_cold", lambda: chat_result(client, common))
    replay, rows, warm = completed(
        "metrics_chat_stream_cached", lambda: chat_result(client, common, True))
    assert first["text"] == replay["text"] and first["reasoning"] == replay["reasoning"]
    assert rows[0]["metrics"]["cached_tokens"] > 0 and rows[0]["metrics"]["prefill_tokens"] == 0
    assert warm[PROMPT_SPEED] == cold[PROMPT_SPEED] > 0
    assert warm[GENERATED_SPEED] > 0

    for streaming in (False, True):
        def response():
            request = dict(model=model, input=prompt, temperature=0, max_output_tokens=16,
                           reasoning={"effort": "none"}, store=False,
                           extra_body={"seed": 42, "presence_penalty": 0})
            if streaming:
                with client.responses.create(**request, stream=True) as stream:
                    events = list(stream)
                return check_events(events, False)
            return check_response(client.responses.create(**request), False)
        completed(f"metrics_responses_stream{streaming}", response)
        completed(f"metrics_completions_stream{streaming}", lambda: completion_result(
            client, dict(model=model, prompt="One, two, three,", temperature=0,
                         seed=42, max_tokens=16), streaming))

    # Stop-filtered text still accounts for generated tokens, including the
    # withheld stop sequence. Match server usage, not retokenized visible text.
    assert first["text"], first
    completed("metrics_stop", lambda: chat_result(
        client, {**common, "stop": first["text"][:1]}, True))

    def rejected():
        try:
            client.chat.completions.create(**{**common, "top_p": 2})
        except openai.BadRequestError as error:
            return {"status": error.status_code}
        raise AssertionError("invalid sampling request was accepted")
    completed("metrics_rejected_request", rejected)

    # Prefix-sharing followers reserve admission capacity before acquiring a
    # runner. A newer arrival must remain queued rather than exceed --sessions
    # or become an active request missing from /slots.
    def shared_prefix_reservations():
        from cache_concurrency import system_prompt
        barrier = threading.Barrier(width + 1)
        body = {**common, "messages": [
            {"role": "system", "content": system_prompt("metrics_slot_reservations", 220)},
            {"role": "user", "content": "Reply with only the code ALPHA."}]}
        observations = []
        def send():
            barrier.wait()
            return chat_result(client, body)
        with ThreadPoolExecutor(width + 1) as pool:
            futures = [pool.submit(send) for _ in range(width + 1)]
            while not all(future.done() for future in futures):
                gauges = metrics.read()
                snapshot = slots()
                assert gauges[PROCESSING] <= width, gauges
                observations.append({"processing": gauges[PROCESSING],
                                     "deferred": gauges[DEFERRED], "slots": snapshot})
                time.sleep(.01)
            results = [future.result() for future in futures]
        assert any(o["processing"] == width and o["deferred"] > 0
                   for o in observations), observations
        if width > 1:
            assert any(r["usage"]["gufo"]["shared_prefix_wait_ms"] > 0 for r in results), results
        assert all(r["text"].strip() == "ALPHA" for r in results), results
        return {"results": results, "observations": observations}
    completed("metrics_shared_prefix_reservations", shared_prefix_reservations)

    # Exercise live totals and the deferred gauge with every session occupied.
    # A long constrained value prevents model-specific early EOS. Disconnect
    # after observing admission; do not finish generating this value.
    before = metrics.idle()
    ready = [threading.Event() for _ in range(width)]
    release = threading.Event()
    live = {**common, "max_completion_tokens": 2048, "response_format": {
        "type": "json_schema", "json_schema": {"name": "busy", "strict": True,
        "schema": {"type": "object", "properties": {"text": {
            "type": "string", "const": "tick " * 2048}},
            "required": ["text"], "additionalProperties": False}}}}

    def hold(index):
        request = {**live, "messages": [{"role": "user", "content":
                   f"Request {index}: emit the required JSON object."}]}
        with client.chat.completions.create(**request, stream=True) as stream:
            for chunk in stream:
                if any(choice.delta.content for choice in chunk.choices):
                    ready[index].set()
                if release.is_set():
                    return {"cancelled": True}
        raise AssertionError("live accounting request finished before cancellation")

    with ThreadPoolExecutor(width + 1) as pool:
        active = [pool.submit(hold, index) for index in range(width)]
        try:
            for event in ready:
                assert event.wait(30), "live stream did not start"
            busy = metrics.wait(lambda m: m[PROCESSING] == width and m[GENERATED] > before[GENERATED]
                                and m[PROMPT] > before[PROMPT], "live token accounting")
            assert busy[DEFERRED] == 0, busy
            busy_slots = slots()
            assert all(slot["is_processing"] for slot in busy_slots), busy_slots
            assert all(slot["next_token"][0]["n_decoded"] > 0 for slot in busy_slots), busy_slots
            assert all(slot["next_token"][0]["n_decoded"] +
                       slot["next_token"][0]["n_remain"] == 2048 for slot in busy_slots), busy_slots
            # Scrapes are separate HTTP snapshots. With every long request held,
            # their token counts can only advance, so bracket the ratio.
            ratio = metrics.read()[KV_USAGE]
            later_slots = slots("/v1/slots")
            def usage(snapshot):
                return sum(slot["n_prompt_tokens"] + slot["next_token"][0]["n_decoded"]
                           for slot in snapshot) / (width * context)
            assert 0 < usage(busy_slots) <= ratio <= usage(later_slots) <= 1, ratio
            queued_start = len(checks.recorder.rows)
            queued = pool.submit(chat_result, client, common, True)
            waiting = metrics.wait(lambda m: m[DEFERRED] == 1 and m[PROCESSING] == width,
                                   "one queued request")
            queued_slots = slots()
            assert {slot["id_task"] for slot in queued_slots} == {
                slot["id_task"] for slot in busy_slots}, queued_slots
        finally:
            release.set()
        cancelled = [future.result(timeout=30) for future in active]
        result = queued.result(timeout=30)
    after = metrics.idle()
    assert after[GENERATED] >= waiting[GENERATED]
    assert all(row["status"] == "complete" for row in checks.recorder.rows[queued_start:])
    checks["metrics_live_queue_cancel"] = {
        "before": before, "busy": busy, "queued": waiting, "after": after,
        "cancelled": cancelled, "completed_peer": result,
        "busy_slots": busy_slots, "queued_slots": queued_slots,
        "after_slots": slots(idle=True)}
    assert after[KV_USAGE] == 0, after
    # Cancellation must not leave stale gauge ownership or double-count the
    # completed peer when the next request reuses its prompt.
    _, _, final = completed("metrics_after_cancel_cached", lambda: chat_result(client, common))
    proposed = final[DRAFTS] - initial[DRAFTS]
    accepted = final[ACCEPTED] - initial[ACCEPTED]
    assert 0 <= accepted <= proposed, final
    rounds = final[DRAFT_ROUNDS] - initial[DRAFT_ROUNDS]
    assert rounds > 0 if speculative != "off" else rounds == 0, final
    assert proposed > 0 if speculative != "off" else proposed == 0, final


SDK_SUITES = ("discovery", "responses", "stops", "conversation", "image-inputs", "image-count", "structured", "structured-limits",
              "tool-reasoning", "reasoning-separator",
              "tools", "auto-tools", "tool-edges", "tool-agent", "tool-agent-loop", "tool-history", "messages-tools", "tool-untyped", "tool-mixed", "tool-native-schemas", "tool-native-types", "tool-schema-edges", "sampling-defaults", "sampling-ranges", "batch",
              "long-context", "state-edges", "progress", "stream-start", "prefill-scheduling", "metrics", "cache-edits", "cache-growth", "cache-depth", "cache-rotation", "cache-concurrency", "cache-shared-prefix",
              "cache-bridge", "cache-compaction", "cache-transforms", "cache-pressure", "cache-messages-loop", "system-injection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-input-modalities", choices=("text", "text,image"),
                        help="Expected loaded inputs for the discovery suite")
    parser.add_argument("--base-url", required=True, help="http://127.0.0.1:PORT/v1")
    parser.add_argument("--model", required=True, help="Gufo served model name")
    parser.add_argument("--expect-reasoning", action="store_true")
    parser.add_argument("--suite", choices=("all", *SDK_SUITES), default="all")
    parser.add_argument("--vision", action="store_true",
                        help="Add image checks; the server needs its matching --mmproj")
    parser.add_argument("--sampling-preset", choices=("qwen38", "deepseek4"),
                        help="Expected text defaults; required for all/sampling-defaults")
    parser.add_argument("--sampling-overrides", type=json.loads, default={},
                        help="JSON object of explicit server sampling CLI values")
    parser.add_argument("--server-thinking", choices=("on", "off"),
                        help="Explicit --think setting on the server, if any")
    parser.add_argument("--output", type=Path, help="Write a partial report after every case")
    parser.add_argument("--through-case",
                        help="Replay the suite prefix and stop before the next request")
    parser.add_argument("--allow-missing-progress", action="store_true",
                        help="Timing control only for revisions predating return_progress")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Requests in the batch suite; must fit server --sessions")
    parser.add_argument("--context", type=int, default=8192,
                        help="Server capacity; long-context fills roughly half, measured in usage")
    parser.add_argument("--speculative", choices=("off", "mtp", "dflash2", "dspark"),
                        default="off", help="Server mode; determines sampled replay guarantees")
    parser.add_argument("--snapshot-capacity-bytes", type=int,
                        help="Configured RAM checkpoint budget; required for cache-bridge")
    parser.add_argument("--server-log", type=Path,
                        help="Server log; shows retry copies refused under memory pressure")
    args = parser.parse_args()
    if args.suite in ("discovery", "all") and args.expected_input_modalities is None:
        parser.error("discovery requires --expected-input-modalities text or text,image")
    if args.suite in ("image-inputs", "image-count") and not args.vision:
        parser.error("image-inputs and image-count require --vision and a loaded projector")
    if args.suite in ("all", "sampling-defaults") and not args.sampling_preset:
        parser.error("--sampling-preset is required for all/sampling-defaults")
    if not isinstance(args.sampling_overrides, dict):
        parser.error("--sampling-overrides must be a JSON object")
    if not 1 <= args.concurrency <= 8:
        parser.error("--concurrency must be between 1 and 8")
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

    options = dict(api_key="local-test", base_url=args.base_url, max_retries=0,
                   timeout=120, _strict_response_validation=True)
    report = {"sdk": openai.__version__, "model": args.model, "suite": args.suite,
              "vision": args.vision, "speculative": args.speculative,
              "sampling_preset": args.sampling_preset,
              "sampling_overrides": args.sampling_overrides, "suites": {},
              "through_case": args.through_case, "status": "running"}
    report["expected_input_modalities"] = args.expected_input_modalities
    recorder = Recorder(args.output.with_suffix(".requests.json") if args.output else None,
                        args.through_case)
    checks = CheckResults(report, args.output, recorder)
    report["checks"] = checks
    checks.save()
    with OpenAI(**options, http_client=DefaultHttpxClient(
        trust_env=False, event_hooks={"request": [local_only, recorder.request_hook],
                                     "response": [recorder.response_hook]}
    )) as client:
        suites = {
            "discovery": lambda: check_discovery(
                client, args.model, checks, args.context,
                args.expected_input_modalities.split(",")),
            "responses": lambda: check_responses(client, args.model, checks, options,
                                                   async_local_only, args.expect_reasoning),
            "stops": lambda: check_stops(client, args.model, checks),
            "conversation": lambda: check_conversations(client, args.model, checks, args.vision),
            "image-inputs": lambda: check_image_inputs(
                client, args.model, checks, image_content, chat_result, response_result),
            "image-count": lambda: check_image_count(
                client, args.model, checks, image_content, chat_result, response_result,
                args.context, args.concurrency),
            "structured": lambda: check_structured_outputs(client, args.model, checks, args.vision),
            "structured-limits": lambda: check_structured_limits(client, args.model, checks, args.vision),
            "native-tools": lambda: check_native_tools(client, args.model, checks, args.vision),
            "auto-tools": lambda: check_auto_tools(client, args.model, checks, args.vision),
            "tool-edges": lambda: check_tool_edges(
                client, args.model, checks, args.sampling_preset),
            "tool-reasoning": lambda: check_tool_reasoning(
                client, args.model, checks, chat_result, args.sampling_preset),
            "reasoning-separator": lambda: check_reasoning_separator(
                client, args.model, checks, chat_result),
            "tool-agent": lambda: check_tool_agent(
                client, args.model, checks, chat_result, args.vision, image_content),
            "tool-agent-loop": lambda: check_tool_agent_loop(client, args.model, checks, chat_result),
            "tool-history": lambda: check_tool_history(
                client, args.model, checks, chat_result, args.vision, image_content),
            "messages-tools": lambda: check_messages_tools(
                client, args.model, checks, chat_result),
            "tool-untyped": lambda: check_untyped_agent_tools(
                client, args.model, checks, chat_result, args.vision, image_content),
            "tool-mixed": lambda: check_mixed_tool_schemas(
                client, args.model, checks, chat_result, args.vision, image_content,
                args.sampling_preset),
            "tool-native-schemas": lambda: check_native_tool_schemas(
                client, args.model, checks, chat_result, args.sampling_preset,
                image_content("red") if args.vision else None),
            "tool-native-types": lambda: check_finite_argument_types(
                client, args.model, checks, chat_result,
                image_content("red") if args.vision else None),
            "tool-schema-edges": lambda: check_tool_schema_edges(
                client, args.model, checks, chat_result, args.vision, image_content),
            "state-edges": lambda: check_state_edges(
                client, args.model, checks, args.speculative, args.vision),
            "sampling-defaults": lambda: check_sampling_defaults(
                client, args.model, checks, args.sampling_preset, args.sampling_overrides,
                args.vision, args.server_thinking),
            "sampling-ranges": lambda: check_sampling_ranges(client, args.model, checks),
            "batch": lambda: check_batches(
                client, args.model, checks, args.concurrency, args.vision, args.speculative),
            "long-context": lambda: check_long_context(
                client, args.model, checks, args.context, args.vision),
            "progress": lambda: check_prompt_progress(
                client, args.model, checks, args.concurrency, args.vision,
                args.allow_missing_progress),
            "stream-start": lambda: check_stream_start(
                client, args.model, checks, args.concurrency, args.context),
            "prefill-scheduling": lambda: check_prefill_scheduling(
                client, args.model, checks, chat_result, args.concurrency, args.context),
            "metrics": lambda: check_server_metrics(client, args.model, checks, args.concurrency,
                                                     args.context, args.speculative),
            "cache-edits": lambda: check_cache_edits(client, args.model, checks, chat_result),
            "cache-growth": lambda: check_cache_growth(
                client, args.model, checks, chat_result, args.server_log),
            "cache-depth": lambda: check_cache_depth(
                client, args.model, checks, chat_result, args.concurrency, args.server_log),
            "cache-rotation": lambda: check_cache_rotation(client, args.model, checks, chat_result),
            "cache-concurrency": lambda: check_cache_concurrency(
                client, args.model, checks, chat_result, args.concurrency),
            "cache-shared-prefix": lambda: check_cache_shared_prefix(
                client, args.model, checks, chat_result),
            "cache-bridge": lambda: check_cache_bridge(
                client, args.model, checks, chat_result, args.snapshot_capacity_bytes),
            "cache-compaction": lambda: check_cache_compaction(
                client, args.model, checks, chat_result, args.context, args.server_log),
            "cache-transforms": lambda: check_cache_transforms(
                client, args.model, checks, chat_result, args.context, args.server_log),
            "cache-pressure": lambda: check_cache_pressure(
                client, args.model, checks, chat_result, args.snapshot_capacity_bytes,
                args.concurrency),
            "cache-messages-loop": lambda: check_cache_messages_loop(
                client, args.model, checks, chat_result, args.context),
            "system-injection": lambda: check_system_injection(
                client, args.model, checks, chat_result),
        }
        selected = ([name for name in suites if name not in ("tool-native-types", "cache-bridge", "cache-compaction", "cache-transforms", "cache-pressure", "cache-messages-loop", "prefill-scheduling")
                     and (name not in ("image-inputs", "image-count") or args.vision)]
                    if args.suite == "all" else
                    ["native-tools", "auto-tools"] if args.suite == "tools" else [args.suite])
        for name in selected:
            started = time.monotonic()
            try:
                suites[name]()
                report["suites"][name] = {"status": "passed"}
            except CaseComplete:
                report["suites"][name] = {"status": "passed", "through_case": args.through_case}
            except Exception as error:
                report["suites"][name] = {"status": "failed", "error": str(error),
                                          "traceback": traceback.format_exc()}
                print(f"FAIL {name}: {error}", file=sys.stderr, flush=True)
            except KeyboardInterrupt:
                report["status"] = "interrupted"
                checks.save()
                raise
            report["suites"][name]["seconds"] = round(time.monotonic() - started, 3)
            checks.save()
            if args.through_case and args.through_case in checks:
                break
    if args.through_case and args.through_case not in checks:
        report["suites"]["requested-case"] = {
            "status": "failed", "error": f"case not reached: {args.through_case}"}
    report["status"] = ("passed" if all(row["status"] == "passed"
                         for row in report["suites"].values()) else "failed")
    checks.save()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
