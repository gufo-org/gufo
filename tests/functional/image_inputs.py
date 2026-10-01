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
