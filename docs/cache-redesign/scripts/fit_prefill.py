#!/usr/bin/env python3
"""Fit the cost of prefilling N new tokens on top of D cached tokens.

From every E2 completion line: N = prefill_tokens, D = cached_tokens,
t = N / prefill_tps. Least squares on
    t = a + b*N + c*N*(D + N/2) + e*D
where c captures attention over the context and e checkpoint captures that
grow with depth.
"""
import json
import pathlib
import re

import numpy as np

from paths import WORK as HERE
LINE = re.compile(r"event=completed .*?prompt_tokens=(\d+) prefill_tokens=(\d+)"
                  r".*?cached_tokens=(\d+).*?prefill_tps=([\d.]+)")


def samples(model):
    rows = []
    for log in (HERE / "results" / "e2" / model).glob("*/server.log"):
        for p, n, d, tps in LINE.findall(log.read_text(errors="replace")):
            n, d, tps = int(n), int(d), float(tps)
            if n >= 32 and tps > 0:
                rows.append((n, d, n / tps))
    return np.array(rows, dtype=float)


def main():
    out = {}
    for model in ("fn", "q27"):
        data = samples(model)
        n, d, t = data[:, 0], data[:, 1], data[:, 2]
        x = np.stack([np.ones_like(n), n, n * (d + n / 2), d], axis=1)
        coef, *_ = np.linalg.lstsq(x, t, rcond=None)
        pred = x @ coef
        rel = np.abs(pred - t) / t
        out[model] = {"samples": len(t), "a_s": coef[0], "b_s_per_token": coef[1],
                      "c_s_per_token_ctx": coef[2], "e_s_per_ctx": coef[3],
                      "median_rel_error": float(np.median(rel)),
                      "p90_rel_error": float(np.percentile(rel, 90))}
        table = {}
        for depth in (6000, 50000, 100000, 150000):
            for gap in (256, 1024, 2048, 6634, 29500, 59000):
                v = np.array([1, gap, gap * (depth + gap / 2), depth])
                table[f"D{depth}_N{gap}"] = round(float(v @ coef), 2)
        out[model]["predicted_s"] = table
        print(model, json.dumps({k: v for k, v in out[model].items()
                                 if k != "predicted_s"}))
        for depth in (6000, 50000, 100000, 150000):
            print("  depth", depth, [table[f"D{depth}_N{g}"] for g in
                                     (256, 1024, 2048, 6634, 29500, 59000)])
    (HERE / "results" / "prefill_fit.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
