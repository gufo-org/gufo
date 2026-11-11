#!/usr/bin/env python3
"""Speed and package power of an OpenAI-compatible chat server, per phase.

Phases (each sampled separately, sequential single-stream requests):
  idle     no requests
  prefill  ~N-token prompts with max_tokens=1
  decode   short prompt, G greedy tokens, streamed
Reports tokens/s, average socket power, power above idle, and energy per
token (total and above idle), from the amdgpu hwmon sensor.
Usage: httppower.py <base-url> <model> <prompt-tokens> <gen-tokens> <seconds>
"""
import glob
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
N, G, SECONDS = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
SENSOR = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average")[0]
WORDS = ("the quick brown fox jumps over the lazy dog while seven wizards "
         "quietly judge boxing matches near a frozen river ").split()


def watts():
    return int(open(SENSOR).read()) / 1e6


def sample(seconds=None, stop=None):
    out, end = [], time.time() + (seconds or 1e9)
    while time.time() < end and not (stop and stop.is_set()):
        out.append(watts())
        time.sleep(0.05)
    return out


def post(body):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def prefill_once():
    nonce = str(time.time_ns())  # defeat prefix reuse
    prompt = nonce + " " + " ".join(WORDS[i % len(WORDS)] for i in range(N))
    body = {"model": MODEL, "max_tokens": 1, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    with post(body) as r:
        return json.loads(r.read()).get("usage", {}).get("prompt_tokens", N)


def decode_once():
    body = {"model": MODEL, "max_tokens": G, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content":
                          "Write a long, detailed essay about rivers and cities. " + str(time.time_ns())}]}
    stamps = []
    with post(body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for c in json.loads(line[5:]).get("choices", []):
                d = c.get("delta", {})
                if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                    stamps.append(time.perf_counter())
    return len(stamps), (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0


def phase(name, work):
    stop = threading.Event()
    box = {}
    t = threading.Thread(target=lambda: box.update(s=sample(stop=stop)))
    t.start()
    start = time.time()
    tokens = busy = 0.0
    while time.time() - start < SECONDS:
        n, dt = work()
        tokens += n
        busy += dt
    elapsed = time.time() - start
    stop.set(); t.join()
    return name, tokens, busy or elapsed, statistics.mean(box["s"])


prefill_once(); decode_once()  # warm up
idle = statistics.mean(sample(SECONDS))
results = []
def timed_prefill():
    s = time.perf_counter()
    n = prefill_once()
    return n, time.perf_counter() - s


results.append(phase("prefill", timed_prefill))
results.append(phase("decode", decode_once))
for name, tokens, seconds, power in results:
    rate = tokens / seconds
    print(f"{name:7s} {rate:8.1f} tok/s  socket {power:5.1f} W  (+{power - idle:5.1f} W over idle {idle:.1f} W)  "
          f"{power / rate * 1000:7.2f} mJ/token total  {(power - idle) / rate * 1000:7.2f} mJ/token above idle")
