#!/usr/bin/env python3
"""Measure `gufo serve embedding` over HTTP with the standard library only.

  python3 tools/models/embedding_gemma2/http_bench.py \
      --base-url http://127.0.0.1:8080/v1 --model embeddinggemma-2

Each workload sends `requests` POSTs to /v1/embeddings from `concurrency`
threads, and every request carries `inputs` texts of `words` words. It prints
one JSON object per workload with the median and 95th-percentile request time,
the request rate, and input tokens per second. The server counts input tokens
in `usage.prompt_tokens`, which includes <bos> and <eos>. The benchmark
excludes one warm-up request per workload. Run it against a resident server on
an otherwise idle machine. The transport is loopback.
"""

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request

# (concurrency, inputs per request, words per input)
WORKLOADS = [
    (1, 1, 16),
    (1, 1, 128),
    (1, 1, 512),
    (1, 1, 2100),
    (1, 8, 128),
    (1, 32, 128),
    (4, 1, 128),
    (16, 1, 128),
    (16, 1, 16),
    (4, 8, 128),
    (16, 8, 128),
]


def post(base_url, model, texts, api_key):
    body = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        base_url + "/embeddings", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key})
    start = time.perf_counter()
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - start
    assert len(payload["data"]) == len(texts)
    return elapsed, payload["usage"]["prompt_tokens"]


def run(base_url, model, api_key, concurrency, inputs, words, requests):
    # Distinct words exercise the tokenizer, and the text is deterministic.
    texts = [" ".join(f"word{(i * 7 + j) % 997}" for j in range(words))
             for i in range(inputs)]
    post(base_url, model, texts, api_key)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda _: post(base_url, model, texts, api_key), range(requests)))
    wall = time.perf_counter() - start
    times = sorted(r[0] for r in results)
    tokens = sum(r[1] for r in results)
    return {
        "concurrency": concurrency, "inputs": inputs, "words": words,
        "requests": requests,
        "request_ms": [round(r[0] * 1000, 3) for r in results],
        "tokens_per_request": results[0][1],
        "median_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(times[math.ceil(len(times) * 0.95) - 1] * 1000, 2),
        "requests_per_s": round(requests / wall, 2),
        "tokens_per_s": round(tokens / wall, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="embeddinggemma-2")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--requests", type=int, default=60)
    args = parser.parse_args()
    for concurrency, inputs, words in WORKLOADS:
        print(json.dumps(run(args.base_url, args.model, args.api_key,
                             concurrency, inputs, words, args.requests)),
              flush=True)


if __name__ == "__main__":
    main()
