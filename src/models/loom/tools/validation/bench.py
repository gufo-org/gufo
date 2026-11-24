#!/usr/bin/env python3
"""Same requests against any OpenAI-compatible server (loom-serve or gufo serve).

Greedy (temperature 0), thinking off. Measures:
  correctness: greedy text of short prompts (compared across servers later)
  decode:      single-stream decode tok/s and TTFT over a long generation
  prefill:     time to first token of a ~2.5k-token document (unique per rep)
  c4:          aggregate tok/s of 4 concurrent generations
Usage: bench.py BASE_URL MODEL_ID OUT.json --doc PATH [--reps N]
"""
import concurrent.futures as cf
import http.client
import json
import statistics
import sys
import time
import urllib.parse

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
# Any ~2.5k-token English document (the published numbers used hrx-system's
# docs/serving HTTP profiles page): --doc PATH.
DOC = open(sys.argv[sys.argv.index("--doc") + 1]).read()
URL = urllib.parse.urlparse(BASE)

CORRECTNESS = [
    "What is the capital of France? Answer in one sentence.",
    "List the first five prime numbers, comma separated.",
    "Translate 'good morning' into Spanish, French and German.",
    "Write a haiku about autumn leaves.",
    "Explain in two sentences why the sky is blue.",
]


def stream(messages, max_tokens):
    """Streams one chat completion; returns text, usage and token timing."""
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    })
    conn = http.client.HTTPConnection(URL.hostname, URL.port, timeout=600)
    start = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    text, usage, first, last, deltas = "", None, None, None, 0
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        event = json.loads(line[5:])
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                deltas += 1
                text += piece
    conn.close()
    end = time.perf_counter()
    completion = (usage or {}).get("completion_tokens", deltas)
    ttft = (first or end) - start
    decode_s = (last - first) if first and last and last > first else 0
    return {
        "text": text, "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": completion, "ttft_s": ttft, "total_s": end - start,
        "decode_tok_s": (completion - 1) / decode_s if decode_s else None,
    }


def user(text):
    return [{"role": "user", "content": text}]


def main():
    results = {"base": BASE, "model": MODEL}
    stream(user("Say hi."), 8)  # warm-up: weights and compiled shapes
    results["correctness"] = [stream(user(p), 128)["text"] for p in CORRECTNESS]

    decode = [stream(user("Write a long, detailed story about a lighthouse keeper "
                          "and a storm. Use at least 800 words."), 512) for _ in range(REPS)]
    results["decode"] = {
        "tok_s": statistics.median(d["decode_tok_s"] for d in decode),
        "ttft_s": statistics.median(d["ttft_s"] for d in decode),
        "tokens": [d["completion_tokens"] for d in decode],
    }

    prefill = []
    for rep in range(REPS):
        prompt = f"Document {rep} ({time.time_ns()}).\n\n{DOC}\n\nSummarize the document in one sentence."
        prefill.append(stream(user(prompt), 16))
    results["prefill"] = {
        "prompt_tokens": prefill[0]["prompt_tokens"],
        "ttft_s": statistics.median(p["ttft_s"] for p in prefill),
        "tok_s": statistics.median(p["prompt_tokens"] / p["ttft_s"] for p in prefill),
    }

    topics = ["a lighthouse", "a desert caravan", "a space station", "a mountain village"]
    c4 = []
    for _ in range(REPS):
        start = time.perf_counter()
        with cf.ThreadPoolExecutor(4) as pool:
            runs = list(pool.map(lambda t: stream(user(f"Write a long story about {t}. "
                                                       "Use at least 600 words."), 256), topics))
        wall = time.perf_counter() - start
        c4.append(sum(r["completion_tokens"] for r in runs) / wall)
    results["c4"] = {"tok_s": statistics.median(c4)}

    json.dump(results, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "correctness"}, indent=1))


main()
