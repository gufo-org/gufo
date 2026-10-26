#!/usr/bin/env python3
"""EmbeddingGemma 2 image HTTP parity, mixed inputs and rejection checks.

Needs a running gufo serve embedding and Pillow/numpy. Reference fixtures
come from the official transformers processor/model, not from Gufo.
"""
import argparse
import base64
import concurrent.futures
import io
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


def picture(spec, mode="RGB", encoding="PNG"):
    y, x, c = np.indices((spec["height"], spec["width"], 3), dtype=np.int64)
    rgb = ((x * 17 + y * 31 + c * 67 + (x * y + spec["seed"] * 13) % 251) % 256).astype(np.uint8)
    image = Image.fromarray(rgb).convert(mode)
    output = io.BytesIO()
    image.save(output, format=encoding, **({"lossless": True} if encoding == "WEBP" else {}))
    return f"data:image/{encoding.lower()};base64," + base64.b64encode(output.getvalue()).decode()


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/embedding_gemma2/image_golden.json").read_text())
    results = []

    def post(inputs, **options):
        request = urllib.request.Request(args.base_url.rstrip("/") + "/embeddings",
            data=json.dumps({"model": args.model, "input": inputs, **options}).encode(),
            headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
                status = response.status
        except urllib.error.HTTPError as error:
            status, body = error.code, json.load(error)
        results.append({"status": status, "elapsed_ms": (time.perf_counter()-start)*1000,
                        "usage": body.get("usage"), "error": body.get("error"),
                        "request": {"input": inputs, **options}, "response": body})
        return status, body

    inputs = []
    for case in fixture["cases"]:
        fragments = case["text"].split("<|image|>")
        content = []
        for i, fragment in enumerate(fragments):
            if fragment:
                content.append({"type": "text", "text": fragment})
            if i < len(case["images"]):
                content.append({"type": "image_url", "image_url": {"url": picture(case["images"][i])}})
        inputs.append({"content": content})
    try:
        # Each request is retained, including cold vision initialization.
        for case, item in zip(fixture["cases"], inputs):
            status, body = post([item])
            assert status == 200, body
            vector = body["data"][0]["embedding"]
            agreement = cosine(vector, case["embedding"])
            assert len(vector) == 768 and agreement >= 0.9999, (case["name"], agreement)
            assert body["usage"]["prompt_tokens"] == len(case["token_ids"]), body["usage"]
            assert abs(sum(v*v for v in vector)-1) < 1e-5
            results[-1]["case"] = case["name"]
            results[-1]["cosine"] = agreement
            print(case["name"], agreement, results[-1]["elapsed_ms"], flush=True)
        status, body = post([inputs[1], "hello", inputs[0], {"content": [{"type": "text", "text": "hello"}]}])
        assert status == 200, body
        assert [v["index"] for v in body["data"]] == list(range(4))
        assert cosine(body["data"][0]["embedding"], fixture["cases"][1]["embedding"]) >= 0.9999
        assert cosine(body["data"][2]["embedding"], fixture["cases"][0]["embedding"]) >= 0.9999
        assert cosine(body["data"][1]["embedding"], body["data"][3]["embedding"]) >= 0.99999
        for dimensions in (128, 256, 512):
            status, body = post([inputs[0]], dimensions=dimensions, encoding_format="base64")
            assert status == 200, body
            values = struct.unpack("<" + "f"*dimensions, base64.b64decode(body["data"][0]["embedding"]))
            assert cosine(values, fixture["cases"][0]["embedding"][:dimensions]) >= 0.9999
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda item: post([item]), [inputs[0], "hello", inputs[3]]))
        assert all(status == 200 for status, _ in responses)
        assert cosine(responses[0][1]["data"][0]["embedding"], fixture["cases"][0]["embedding"]) >= 0.9999
        # PNG alpha and lossless WebP must decode to the same RGB reference.
        for mode, encoding in [("RGBA", "PNG"), ("RGB", "WEBP")]:
            url = picture(fixture["cases"][0]["images"][0], mode, encoding)
            status, body = post([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])
            assert status == 200, body
            assert len(body["data"][0]["embedding"]) == 768
            assert cosine(body["data"][0]["embedding"], fixture["cases"][0]["embedding"]) >= 0.9999
        # Compare the HTTP JPEG decoder with an independently Pillow-decoded
        # lossless copy of exactly the same JPEG pixels.
        jpeg_url = picture(fixture["cases"][0]["images"][0], encoding="JPEG")
        decoded = Image.open(io.BytesIO(base64.b64decode(jpeg_url.split(",", 1)[1]))).convert("RGB")
        output = io.BytesIO()
        decoded.save(output, format="PNG")
        png_url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        status, body = post([{"content": [{"type": "image_url", "image_url": {"url": url}}]}
                             for url in (jpeg_url, png_url)])
        assert status == 200, body
        assert cosine(body["data"][0]["embedding"], body["data"][1]["embedding"]) >= 0.99999
        for url in ["file:///etc/passwd", "http://127.0.0.1/test.png", "https://127.0.0.1/test.png", "data:image/png;base64,AAAA"]:
            status, body = post([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])
            assert status == 400 and body["error"]["code"] == "invalid_image_input", body
        # A valid small image repeated in one input must be refused once its
        # expanded sequence exceeds the model context; it must not truncate.
        tiny = {"width": 1, "height": 1, "seed": 0}
        part = {"type": "image_url", "image_url": {"url": picture(tiny)}}
        status, body = post([{"content": [part] * 32}])
        assert status == 400 and body["error"]["code"] == "context_length_exceeded", body
        status, body = post([{"content": [part] * 33}])
        assert status == 400 and body["error"]["code"] == "too_many_images", body
        status, body = post([{"content": [part, {"type": "text", "text": "<|image|>"}]}])
        assert status == 400 and body["error"]["code"] == "invalid_image_input", body
        status, body = post([{"content": [{"type": "text", "text": "<|ima"},
                                          {"type": "text", "text": "ge|>"}, part]}])
        assert status == 400 and body["error"]["code"] == "invalid_image_input", body
        status, body = post(["hello"])
        assert status == 200, body
    finally:
        Path(args.output).write_text(json.dumps({"model_revision": fixture["revision"], "requests": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
