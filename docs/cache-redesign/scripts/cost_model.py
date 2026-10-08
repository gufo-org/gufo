#!/usr/bin/env python3
"""Per-operation cost of today's cache and the hybrid design, from measured
constants. Prints Markdown tables for docs/cache-redesign/cost-model.md."""
import json
import pathlib

from paths import WORK as HERE
FIT = json.loads((HERE / "results" / "prefill_fit.json").read_text())

MODELS = {
    # fixed bytes, bytes per token (E1), RAM budget and staging with 2 sessions (E2)
    "Flash-Next (MTP)": {"key": "fn", "F": 119.3e6, "r": 27460, "ram": 9.2e9, "staging": 2.3e9,
                         "capture_today": lambda b: 3.0, "capture_hybrid": 3.0},
    "27B (DFlash2)": {"key": "q27", "F": 243.7e6, "r": 65536, "ram": 23.0e9, "staging": 5.75e9,
                      "capture_today": lambda b: 66.6 + 16.9 * b / 1e9, "capture_hybrid": 14.0},
}
D2D = 100e9          # restore into a session: measured 96-110 GB/s
WRITE = 0.44e9       # gufo disk write incl. serialization (E2 logs; raw 0.59)
READ = 1.1e9         # cold read (diskbench); page-cache hits reach ~1.5
CHUNK_READ_PENALTY = 1.13
STEP = 2048          # new tokens per persisted checkpoint


def prefill(key, n, d=0):
    c = FIT[key]
    return c["a_s"] + c["b_s_per_token"] * n + c["c_s_per_token_ctx"] * n * (d + n / 2) \
        + c["e_s_per_ctx"] * d


def fmt_s(s):
    return f"{s * 1000:.0f} ms" if s < 1 else (f"{s:.1f} s" if s < 120 else f"{s / 60:.1f} min")


def main():
    for name, m in MODELS.items():
        print(f"\n#### {name}\n")
        print("| Operation | Depth | Today | Hybrid |")
        print("| --- | ---: | ---: | ---: |")
        for depth in (32_000, 100_000, 149_000):
            full = m["F"] + m["r"] * depth
            kv = m["r"] * depth
            step_bytes = m["F"] + m["r"] * STEP
            rows = [
                ("Checkpoint bytes", f"{full / 1e9:.2f} GB", f"{m['F'] / 1e6:.0f} MB + shared KV"),
                ("Capture (on the request path)", f"{m['capture_today'](full):.0f} ms",
                 f"~{m['capture_hybrid']:.0f} ms"),
                ("Persist one checkpoint", fmt_s(full / WRITE) + (" (skipped: > staging)"
                 if full > m["staging"] else ""), fmt_s(step_bytes / WRITE)),
                ("Restore from RAM", fmt_s(full / D2D), fmt_s(full / D2D)),
                ("Restore from disk (cold)", fmt_s(full / READ),
                 fmt_s(full * CHUNK_READ_PENALTY / READ)),
                ("Prefill instead (cold)", fmt_s(prefill(m["key"], depth)),
                 fmt_s(prefill(m["key"], depth))),
                ("Checkpoints of this conversation in the RAM budget",
                 f"{int(m['ram'] // full)}",
                 f"{max(0, int((m['ram'] - kv) // (m['F'] + m['r'] * 1024)))}"),
            ]
            for op, today, hybrid in rows:
                print(f"| {op} | {depth // 1000}k | {today} | {hybrid} |")


if __name__ == "__main__":
    main()
