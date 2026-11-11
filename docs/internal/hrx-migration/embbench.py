#!/usr/bin/env python3
"""Sanity and throughput check for an OpenAI-style /v1/embeddings endpoint."""
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]


def embed(texts, dims=None):
    body = {"model": MODEL, "input": texts}
    if dims:
        body["dimensions"] = dims
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return [d["embedding"] for d in out["data"]], out.get("usage", {})


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


pairs = [
    ("The cat sat on the mat.", "A cat is sitting on a rug."),
    ("The cat sat on the mat.", "Quarterly revenue grew by eight percent."),
    ("How do I reset my password?", "Steps to change a forgotten login password"),
    ("How do I reset my password?", "The volcano erupted in 1883."),
]
for a, b in pairs:
    (ea, eb), _ = embed([a, b])
    print(f"cos={cos(ea, eb):.3f}  dim={len(ea)}  {a!r} vs {b!r}")

text = "Embeddings map text to vectors so that similar meanings lie close together."
for batch in (1, 4, 8):
    times = []
    for _ in range(5):
        t = time.perf_counter()
        _, usage = embed([text] * batch)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    print(f"batch {batch:2d}: {med*1000:7.1f} ms  {batch/med:6.1f} req/s  "
          f"prompt_tokens={usage.get('prompt_tokens')}")

# Concurrent single-input requests.
with concurrent.futures.ThreadPoolExecutor(16) as pool:
    t = time.perf_counter()
    list(pool.map(lambda _: embed([text]), range(64)))
    dt = time.perf_counter() - t
print(f"64 concurrent single requests: {dt*1000:.0f} ms  {64/dt:.1f} req/s")
