#!/usr/bin/env python3
"""Focused LFM3 HTTP qualification; run against an explicitly started server.

python tests/functional/lfm3.py --url http://127.0.0.1:18083 --output artifacts/lfm3/http.json
No model download or inference runs in hosted CI.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []

    def request(messages, *, seed=None, stream=False, cache=True):
        body = {"model": "LFM3-1.7B", "messages": messages, "max_tokens": 32,
                "temperature": 0 if seed is None else 0.7, "cache_prompt": cache,
                "chat_template_kwargs": {"enable_thinking": False}, "stream": stream}
        if seed is not None:
            body["seed"] = seed
        started = time.monotonic()
        req = urllib.request.Request(args.url + "/v1/chat/completions",
                                     json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode()
        if stream:
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            result = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "")
                             for c in chunks if c.get("choices"))
            record = {"request": body, "chunks": chunks}
        else:
            reply = json.loads(raw)
            result = reply["choices"][0]["message"]["content"]
            record = {"request": body, "response": reply}
        record["wall_s"] = time.monotonic() - started
        records.append(record)
        assert result, "empty nonthinking response"
        return result

    try:
        basic = [{"role": "user", "content": "What is the capital of France?"}]
        answer = request(basic, cache=False)
        assert "Paris" in answer, answer
        assert request(basic, stream=True, cache=False) == answer, "stream differs from nonstream"
        assert request(basic, seed=42, cache=False) == request(basic, seed=42, cache=False), "seed replay differs"
        followup = basic + [{"role": "assistant", "content": answer},
                            {"role": "user", "content": "Which country is that city in?"}]
        multi = request(followup)
        assert "France" in multi, multi
        assert request(followup, cache=False) == multi, "cached/fresh continuation differs"
        # Cross the 256-token sliding window and the 512-token prefill chunk.
        long = [{"role": "user", "content": "The secret word is pineapple. " +
                 "Please keep this in mind. " * 90 + "What is the secret word?"}]
        result = request(long, cache=False)
        assert "pineapple" in result.lower(), result
        other = [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}]
        control = request(other, cache=False)
        with ThreadPoolExecutor(max_workers=2) as pool:
            left = pool.submit(request, basic, cache=False)
            right = pool.submit(request, other, cache=False)
            assert left.result() == answer and right.result() == control, "concurrent sessions changed output"
        print("PASS LFM3 greedy/streaming, sampled replay, multi-turn/cache, long prompt and C2 isolation")
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
