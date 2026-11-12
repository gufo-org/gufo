#!/usr/bin/env python3
"""Fit nonnegative prefill costs to CSV columns from,to,seconds (stdlib only)."""
import argparse
import csv
import json
import math
from pathlib import Path


def fit(samples):
    rows = [(float(b - a), float(b - a) * (float(b) + a) / 200000, seconds)
            for a, b, seconds in samples]
    if len(rows) < 2 or any(a < 0 or b <= a or not math.isfinite(seconds) or seconds <= 0
                            for a, b, seconds in samples):
        raise ValueError("need at least two finite, positive prefill intervals")
    xx = sum(x*x for x, y, t in rows)
    xy = sum(x*y for x, y, t in rows)
    yy = sum(y*y for x, y, t in rows)
    xt = sum(x*t for x, y, t in rows)
    yt = sum(y*t for x, y, t in rows)
    det = xx*yy - xy*xy
    if det <= 1e-12 * xx*yy:
        raise ValueError("prefill depths do not identify both coefficients")
    candidates = [(xt/xx, 0), (0, yt/yy)]
    d, e = (xt*yy - yt*xy)/det, (yt*xx - xt*xy)/det
    if d >= 0 and e >= 0:
        candidates.append((d, e))
    d, e = min(candidates, key=lambda p: sum((p[0]*x + p[1]*y - t)**2
                                            for x, y, t in rows))
    errors = [(d*x + e*y - t)/t for x, y, t in rows]
    return {"d": d, "e": e, "samples": len(rows),
            "rms_relative_error": math.sqrt(sum(v*v for v in errors)/len(errors)),
            "max_relative_error": max(map(abs, errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with args.csv.open() as source:
        samples = [(int(row["from"]), int(row["to"]), float(row["seconds"]))
                   for row in csv.DictReader(source)]
    print(json.dumps(fit(samples), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
