#!/usr/bin/env python3
"""Check Gufo's text Responses subset with the official OpenAI Python SDK.

Run with nix develop -c python3. Start a Gufo text server first; all requests
go to the explicitly supplied loopback endpoint.
"""

import argparse
import asyncio
import json
from urllib.parse import urlsplit

import openai
from openai import AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient, OpenAI


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
    args = parser.parse_args()
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
        # Stop at the known greedy answer. This also exercises an empty visible
        # response and the stream terminator, using the SDK's own chunk parser.
        for streaming in (False, True):
            result = client.chat.completions.create(
                **chat_request, stop=chat.choices[0].message.content,
                stream=streaming,
            )
            if streaming:
                text, finish = "", None
                with result:
                    for chunk in result:
                        for choice in chunk.choices:
                            text += choice.delta.content or ""
                            finish = choice.finish_reason or finish
            else:
                text = result.choices[0].message.content or ""
                finish = result.choices[0].finish_reason
            assert text == "" and finish == "stop", (text, finish)
            checks[f"chat_stop_stream_{streaming}"] = {"finish_reason": finish}

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
