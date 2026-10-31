#!/usr/bin/env python3
"""Compare two Gemma 4 logit files (G4LG format) row by row.

Reports, over the positions both files contain: mean/max KL(reference ||
candidate) in nats on the full vocabulary at temperature 1, total variation,
top-1 agreement and the maximum absolute logit difference.

    nix develop -c python3 tools/gemma4/compare_logits.py REF.g4lg CAND.g4lg
"""
from __future__ import annotations

import argparse
import json
import struct
import sys

import numpy as np


def read(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        magic, version, rows, vocab = struct.unpack("<4I", handle.read(16))
        if magic != 0x474C3447 or version != 1:
            raise SystemExit(f"{path}: not a G4LG v1 logit file")
        positions = np.frombuffer(handle.read(4 * rows), dtype="<u4")
        logits = np.frombuffer(handle.read(4 * rows * vocab), dtype="<f4").reshape(rows, vocab)
    return positions, logits


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ref_pos, ref = read(args.reference)
    cand_pos, cand = read(args.candidate)
    common, ri, ci = np.intersect1d(ref_pos, cand_pos, return_indices=True)
    if common.size == 0:
        raise SystemExit("no common positions")
    lp = log_softmax(ref[ri])
    lq = log_softmax(cand[ci])
    p = np.exp(lp)
    kl = (p * (lp - lq)).sum(axis=-1)
    tv = 0.5 * np.abs(p - np.exp(lq)).sum(axis=-1)
    top1 = (ref[ri].argmax(axis=-1) == cand[ci].argmax(axis=-1))
    result = {
        "rows": int(common.size),
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "kl_max_position": int(common[int(kl.argmax())]),
        "tv_mean": float(tv.mean()), "tv_max": float(tv.max()),
        "top1_agreement": f"{int(top1.sum())}/{top1.size}",
        "max_abs_logit_diff": float(np.abs(ref[ri] - cand[ci]).max()),
        "nonfinite": int((~np.isfinite(cand[ci])).sum()),
    }
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        for key, value in result.items():
            print(f"{key:>20}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
