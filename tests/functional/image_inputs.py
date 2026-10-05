"""Image upload formats and errors through Chat and Responses."""

import base64
import re
import sys


# Solid red, 128x128. Fixed fixtures keep this suite independent of image tools.
JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYF"
    "BgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCACAAIADASIA"
    "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFgEB"
    "AQEAAAAAAAAAAAAAAAAAAAgJ/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8Ai8BK"
    "bfwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB//Z"
)
WEBP_LOSSLESS = "UklGRiIAAABXRUJQVlA4TBYAAAAvf8AfAAcQ/Y/+hwFICP//KxH9T/0D"
WEBP_LOSSY = (
    "UklGRnAAAABXRUJQVlA4IGQAAAAQBwCdASqAAIAAPjEYi0SiIaEQRAAgAwS0t3C6TLgPwA/AAAC1"
    "YSrOcgjYgqOe2IvEFRz2xF4gqOe2IvEFRz2wuAD+/zXPf/+xmP2jP8kr//+DMfwZj+DMf/CfG"
    "xI0QAAAAAAA"
)


def image_cases(image_content):
    png = image_content("red")["image_url"]["url"]
    payload = png.split(",", 1)[1]
    return (
        ("png", png),
        ("jpeg", "data:image/jpeg;base64," + JPEG),
        ("webp_lossless", "data:image/webp;base64," + WEBP_LOSSLESS),
        ("webp_lossy", "data:image/webp;base64," + WEBP_LOSSY),
        ("jpg_alias", "data:image/jpg;base64," + JPEG),
        ("uppercase", "data:IMAGE/PNG;BASE64," + payload),
        ("parameters", "data:image/png;name=upload.png;charset=utf-8;base64," + payload),
    )


def invalid_image_cases(image_content):
    payload = image_content("red")["image_url"]["url"].split(",", 1)[1]
    corrupt = base64.b64encode(b"RIFF\x04\x00\x00\x00WEBP").decode()
    return (
        ("unsupported", "data:image/gif;base64," + payload, "image/gif"),
        ("not_base64", "data:image/png," + payload, "base64"),
        ("empty", "data:image/png;base64,", "base64"),
        ("bad_base64", "data:image/png;base64,A===", "base64"),
        ("corrupt_webp", "data:image/webp;base64," + corrupt, "WebP"),
    )


def assert_color(result, color):
    assert re.fullmatch(color + r"[.!]?", result["text"].strip().lower()), result
    assert not result["reasoning"] and result["finish"] == "stop", result
    usage = result["usage"]
    tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details"))
    assert tokens > 0 and details["cached_tokens"] == 0, usage
    if "gufo" in usage:  # Responses exposes these counters outside usage.
        assert usage["gufo"]["prefill_tokens"] == tokens, usage


def check_image_inputs(client, model, checks, image_content, chat_result, response_result):
    from openai import BadRequestError

    def record(name, result):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)

    def request(endpoint, url, streaming):
        prompt = "Name the dominant color in this image. Reply with one lowercase English color name only."
        common = dict(model=model, temperature=0,
                      extra_body={"cache_prompt": False, "seed": 41})
        if endpoint == "chat":
            return chat_result(client, dict(
                **common, messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt}]}],
                max_completion_tokens=16, reasoning_effort="none"), streaming)
        return response_result(client, dict(
            **common, input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": url},
                {"type": "input_text", "text": prompt}]}],
            max_output_tokens=16, reasoning={"effort": "none"}, store=False), streaming)

    # Existing PNG/JPEG requests come first, allowing matched timing controls on
    # servers that do not yet support the other spellings or WebP.
    for name, url in image_cases(image_content):
        for endpoint in ("chat", "responses"):
            result = request(endpoint, url, name in ("jpeg", "webp_lossy", "uppercase"))
            assert_color(result, "red")
            record(f"image_inputs_{endpoint}_{name}", result)
    for endpoint in ("chat", "responses"):
        for name, url, message in invalid_image_cases(image_content):
            try:
                request(endpoint, url, False)
            except BadRequestError as error:
                assert error.status_code == 400 and error.code, error
                assert message.lower() in error.message.lower(), error
                record(f"image_inputs_{endpoint}_{name}",
                       {"status": error.status_code, "code": error.code, "message": error.message})
            else:
                raise AssertionError(f"Invalid image accepted: {endpoint}/{name}")
        result = request(endpoint, image_content("blue")["image_url"]["url"], True)
        assert_color(result, "blue")
        record(f"image_inputs_{endpoint}_recovery", result)


def check_image_count(client, model, checks, image_content, chat_result, response_result,
                      context, concurrency):
    """Image count is bounded by actual prompt size, including replayed history."""
    from concurrent.futures import ThreadPoolExecutor
    from openai import BadRequestError

    question = ("Name the dominant color of the LAST image. "
                "Reply with one lowercase English color name only.")

    def request(endpoint, count, color, split=False):
        parts = [image_content("red") for _ in range(count - 1)] + [image_content(color)]
        if endpoint == "responses":
            parts = [{"type": "input_image", "image_url": part["image_url"]["url"]}
                     for part in parts]
        text = {"type": "text" if endpoint == "chat" else "input_text", "text": question}
        messages = ([{"role": "user", "content": [part]} for part in parts]
                    if split else [{"role": "user", "content": parts}])
        messages[-1]["content"].append(text)
        common = dict(model=model, temperature=0,
                      extra_body={"seed": 354, "presence_penalty": 0, "frequency_penalty": 0})
        if endpoint == "chat":
            return dict(**common, messages=messages, reasoning_effort="none",
                        max_completion_tokens=16)
        return dict(**common, input=messages, reasoning={"effort": "none"},
                    max_output_tokens=16, store=False)

    def save(name, result, color):
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        assert re.fullmatch(color + r"[.!]?", result["text"].strip().lower()), result
        assert not result["tools"] and result["finish"] == "stop", result
        return result

    def counts(result):
        usage = result["usage"]
        total = usage.get("prompt_tokens", usage.get("input_tokens"))
        details = usage.get("prompt_tokens_details", usage.get("input_tokens_details"))
        return total, details["cached_tokens"]

    # Stable main/PR controls precede the newly supported cases, so timing
    # comparisons can stop here without asking main to accept >16 images.
    for count in (1, 16):
        body = request("chat", count, "red")
        for streaming in (False, True):
            result = save(f"image_count_control_{count}_{streaming}",
                          chat_result(client, body, streaming), "red")
            if streaming:
                total, cached = counts(result)
                assert cached == total, result

    for endpoint, call in (("chat", chat_result), ("responses", response_result)):
        body = request(endpoint, 17, "blue", split=endpoint == "responses")
        first = save(f"image_count_{endpoint}_17", call(client, body, False), "blue")
        retry = save(f"image_count_{endpoint}_17_retry", call(client, body, True), "blue")
        assert retry["text"] == first["text"] and counts(retry)[0] == counts(retry)[1], retry
        key = "messages" if endpoint == "chat" else "input"
        next_body = request(endpoint, 1, "red")
        next_body[key] = [*body[key], {"role": "assistant", "content": retry["text"]},
                          *next_body[key]]
        continued = save(f"image_count_{endpoint}_18",
                          call(client, next_body, True), "red")
        assert counts(continued)[1] >= counts(first)[0], (first, continued)

    body = request("chat", 17, "blue")
    schema = {"type": "object", "properties": {
        "color": {"type": "string", "enum": ["red", "blue"]}},
        "required": ["color"], "additionalProperties": False}
    sampled = {**body, "temperature": 0.7, "top_p": 0.95,
               "reasoning_effort": "low", "max_completion_tokens": 256,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "color", "strict": True, "schema": schema}}}
    import json
    for streaming in (False, True):
        result = chat_result(client, sampled, streaming)
        name = f"image_count_sampled_thinking_{streaming}"
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        assert json.loads(result["text"]) == {"color": "blue"} and result["reasoning"], result
        assert result["finish"] == "stop", result
        if streaming:
            assert counts(result)[0] == counts(result)[1], result

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [(color, pool.submit(chat_result, client,
                                          request("chat", 17, color), True))
                       for color in ("red", "blue")]
            results = {}
            for color, future in pending:
                result = future.result()
                assert re.fullmatch(color + r"[.!]?", result["text"].strip().lower()), result
                assert not result["tools"] and result["finish"] == "stop", result
                results[color] = result
            checks["image_count_parallel"] = results
            print("CHECK image_count_parallel", file=sys.stderr, flush=True)

    # Every 128px fixture expands to at least 64 image tokens. Admission
    # must reject the expanded prompt before running the vision encoder.
    try:
        chat_result(client, request("chat", context // 64 + 1, "red"), False)
    except BadRequestError as error:
        assert error.code == "context_length_exceeded", error
        checks["image_count_context_limit"] = {"status": 400, "code": error.code}
        print("CHECK image_count_context_limit", file=sys.stderr, flush=True)
    else:
        raise AssertionError("Accepted images exceeding the model context")
    save("image_count_recovery", chat_result(client, body, True), "blue")
