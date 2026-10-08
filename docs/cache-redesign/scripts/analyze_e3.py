#!/usr/bin/env python3
"""E3: missed reuse per request.

ideal = longest prefix the prompt shares with any earlier prompt+output in the
same workload (a cache that kept everything, with a checkpoint at every token).
actual = cached_tokens reported by the server. The gap is classified with the
server's own diagnostics (common_prefix_tokens, nearest checkpoint, restarts)
and converted to seconds with that request's measured prefill rate.
"""
import json
import pathlib
import re
import sys

import numpy as np

from paths import WORK as HERE
COMPLETED = re.compile(r"request=(\S+) event=completed .*?prefill_tokens=(\d+)"
                       r".*?prefill_tps=([\d.]+)")


def lcp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.nonzero(a[:n] != b[:n])[0]
    return int(diff[0]) if len(diff) else n


def main(model, workload):
    base = HERE / "results" / "e2" / model / workload
    meta = json.loads((base / "requests.json").read_text())
    data = np.load(base / "requests.npz")
    rates = {}
    for match in COMPLETED.finditer((base / "server.log").read_text(errors="replace")):
        rates[match.group(1)] = float(match.group(3))
    histories = []
    rows = []
    for i, m in enumerate(meta):
        prompt = data[f"p{i}"]
        ideal = max((lcp(prompt, h) for h in histories), default=0)
        # Re-tokenized text is a few tokens shorter than the server's own
        # tokens; express positions in the server's token count.
        scale = m["prompt_tokens"] / max(1, m["tokenized_prompt"])
        ideal = min(m["prompt_tokens"], round(ideal * scale))
        histories.append(np.concatenate([prompt, data[f"o{i}"]]))
        actual = m["cached_tokens"] or 0
        gap = max(0, ideal - actual)
        common = m.get("common_prefix_tokens")
        if gap < 256:
            cause = "none"
        elif i and m["server_lifetime"] != meta[i - 1]["server_lifetime"] and \
                m["cache"] != "disk":
            cause = "restart (disk had nothing usable)"
        elif common is not None and common >= ideal - 64:
            cause = "no checkpoint at the shared position"
        else:
            cause = "evicted or never retained"
        rate = rates.get(m["request"]) or None
        rows.append({"i": i, "prompt": m["prompt_tokens"], "ideal": ideal,
                     "cached": actual, "gap": gap, "cache": m["cache"],
                     "miss_reason": m.get("cache_miss_reason"),
                     "common": common, "cause": cause,
                     "gap_s": round(gap / rate, 2) if rate and gap else 0.0,
                     "tokenized_ok": m["tokenized_prompt"] == m["prompt_tokens"]})
    (base / "e3.json").write_text(json.dumps(rows, indent=1))
    total_prompt = sum(r["prompt"] for r in rows)
    total_ideal = sum(r["ideal"] for r in rows)
    total_cached = sum(r["cached"] for r in rows)
    by_cause = {}
    for r in rows:
        if r["cause"] != "none":
            entry = by_cause.setdefault(r["cause"], [0, 0, 0.0])
            entry[0] += 1
            entry[1] += r["gap"]
            entry[2] += r["gap_s"]
    print(f"{model}/{workload}: {len(rows)} requests, prompt {total_prompt}, "
          f"ideal reuse {total_ideal} ({total_ideal / max(1, total_prompt):.1%}), "
          f"actual {total_cached} ({total_cached / max(1, total_prompt):.1%}), "
          f"tokenization mismatches {sum(not r['tokenized_ok'] for r in rows)}")
    for cause, (count, tokens, seconds) in sorted(by_cause.items()):
        print(f"  {cause}: {count} requests, {tokens} tokens, ~{seconds:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
