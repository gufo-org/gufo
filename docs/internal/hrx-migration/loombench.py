#!/usr/bin/env python3
"""Times prefill and decode against an OpenAI-compatible endpoint (loom-serve).

Prefill: a prompt of N tokens with max_tokens=1, wall time to completion.
Decode: max_tokens=G streamed; tokens/s from first to last content chunk.
Prompts are built from a fixed text so token counts are close to N; the
server's reported usage gives the exact count.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
LENGTHS = [int(x) for x in sys.argv[3].split(",")]
GEN = int(sys.argv[4])
REPS = int(sys.argv[5])

WORDS = ("the quick brown fox jumps over the lazy dog while seven wizards "
         "quietly judge boxing matches near a frozen river ").split()


def prompt(n):
    # Roughly one token per word for this vocabulary; usage reports the exact count.
    return " ".join(WORDS[i % len(WORDS)] for i in range(n))


def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def prefill(n):
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "user", "content": prompt(n)}]}
    t = time.perf_counter()
    with post(body) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t
    return dt, out.get("usage", {})


def decode(n, gen):
    body = {"model": MODEL, "max_tokens": gen, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content":
                          "Write a long, detailed essay of at least 800 words about "
                          "the history of rivers and cities. " + prompt(n)}]}
    stamps, text, usage = [], [], {}
    t0 = time.perf_counter()
    with post(body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    ttft = stamps[0] - t0 if stamps else float("nan")
    rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
    return ttft, rate, len(stamps), usage, "".join(text)


for n in LENGTHS:
    for r in range(REPS):
        dt, usage = prefill(n)
        pt = usage.get("prompt_tokens", n)
        print(f"prefill n={n} rep={r} prompt_tokens={pt} wall={dt*1000:.1f} ms "
              f"rate={pt/dt:.1f} tok/s", flush=True)
for r in range(REPS):
    ttft, rate, chunks, usage, text = decode(64, GEN)
    print(f"decode rep={r} ttft={ttft*1000:.1f} ms chunks={chunks} "
          f"completion_tokens={usage.get('completion_tokens')} rate={rate:.2f} chunks/s",
          flush=True)
print("sample:", repr(text[:300]))
