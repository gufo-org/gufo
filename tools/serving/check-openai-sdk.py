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


def check_conversations(client, model, checks, vision=False):
    """Exercise thinking controls and cache reuse after a client disconnect."""

    def image(color):
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
                    image("red"),
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
                    image(color),
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
    parser.add_argument("--suite", choices=("all", "stops", "responses", "conversation"), default="all")
    parser.add_argument("--vision", action="store_true",
                        help="Add image checks; the server needs its matching --mmproj")
    args = parser.parse_args()
    if args.vision and args.suite not in ("all", "conversation"):
        parser.error("--vision requires --suite all or conversation")
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
                assert error.code == "invalid_request"
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
