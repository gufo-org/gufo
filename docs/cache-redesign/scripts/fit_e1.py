#!/usr/bin/env python3
"""E1: least-squares fit of payload bytes = fixed + per_token * tokens."""
import json
import pathlib

from paths import WORK as HERE

rows = []
for path in sorted(p for p in (HERE / "results" / "e1").glob("*.json") if p.name != "fit.json"):
    result = json.loads(path.read_text())
    samples = sorted({(s["tokens"], s["payload_bytes"]) for s in result.get("stored", [])})
    if len(samples) < 2:
        print(f"{result['name']}: {len(samples)} samples, skipped")
        continue
    n = len(samples)
    sx = sum(t for t, _ in samples)
    sy = sum(b for _, b in samples)
    sxx = sum(t * t for t, _ in samples)
    sxy = sum(t * b for t, b in samples)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    fixed = (sy - slope * sx) / n
    worst = max(abs(b - (fixed + slope * t)) / b for t, b in samples)
    rows.append({"name": result["name"], "samples": n, "fixed_mib": fixed / 2**20,
                 "per_token_kb": slope / 1000, "max_rel_residual": worst,
                 "points": samples})
    print(f"{result['name']:>14}: fixed {fixed / 2**20:8.1f} MiB  "
          f"per token {slope / 1000:6.2f} KB  ({n} samples, "
          f"max residual {worst:.2%}, tokens {samples[0][0]}..{samples[-1][0]})")
(HERE / "results" / "e1" / "fit.json").write_text(json.dumps(rows, indent=1))
