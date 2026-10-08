#!/usr/bin/env python3
"""Image-request latency of an OpenAI-compatible Gemma 4 server.

For each image: a cold request (a fresh nonce before the image, so nothing is
reused) streamed to measure time to first token, the server's prompt timings
and decode rate, then a follow-up turn that reuses the image prefix. Images
should already be resized to multiples of 48 inside the server's token range
so every server sees the same pixels and soft-token count.

    python3 tools/gemma4/image_bench.py --url http://127.0.0.1:8080 \\
        --model Gemma-4-31B-It --label gufo IMAGE.png [IMAGE.png ...] \\
        [--repetitions 3] [--output result.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

QUESTION = "Describe this image in detail."
FOLLOW_UP = "Summarize that in one sentence."


def stream(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps({**body, "stream": True,
                    "stream_options": {"include_usage": True}}).encode(),
        {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    text = []
    timings = {}
    usage = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    end = time.perf_counter()
    return {"ttft_s": (first or end) - start, "wall_s": end - start,
            "text": "".join(text), "timings": timings, "usage": usage}


def run_image(url: str, model: str, path: Path, output_tokens: int) -> dict:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    user = {"role": "user", "content": [
        {"type": "text", "text": f"Session {uuid.uuid4().hex}. {QUESTION}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]}
    common = {"model": model, "max_tokens": output_tokens, "temperature": 0,
              "seed": 1}
    cold = stream(url, {**common, "messages": [user]})
    follow = stream(url, {**common, "messages": [
        user, {"role": "assistant", "content": cold["text"]},
        {"role": "user", "content": FOLLOW_UP}]})
    t = cold["timings"]
    return {
        "prompt_tokens": cold["usage"].get("prompt_tokens") or t.get("prompt_n"),
        "cold_ttft_s": cold["ttft_s"],
        "cold_prompt_ms": t.get("prompt_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "follow_ttft_s": follow["ttft_s"],
        "follow_cached_tokens": (follow["usage"].get("prompt_tokens_details")
                                 or {}).get("cached_tokens",
                                             follow["timings"].get("cache_n")),
        "answer": cold["text"][:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for image in args.images:
        runs = [run_image(args.url, args.model, image, args.output_tokens)
                for _ in range(args.repetitions + 1)][1:]  # first run warms up
        summary = {"image": image.name, "server": args.label, "runs": runs}
        for key in ("cold_ttft_s", "cold_prompt_ms", "decode_tok_s",
                    "follow_ttft_s"):
            values = [r[key] for r in runs if r[key] is not None]
            summary[key] = statistics.median(values) if values else None
        summary["prompt_tokens"] = runs[0]["prompt_tokens"]
        results.append(summary)
        print(f"{args.label} {image.name}: {summary['prompt_tokens']} prompt "
              f"tokens, cold TTFT {summary['cold_ttft_s']:.3f} s, follow-up "
              f"TTFT {summary['follow_ttft_s']:.3f} s, decode "
              f"{summary['decode_tok_s'] or 0:.2f} tok/s")
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
