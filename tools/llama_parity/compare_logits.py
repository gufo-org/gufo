#!/usr/bin/env python3
"""Compares two float32 logit dumps position by position.

Both files hold one vocabulary-wide row per position (llama_logits.cpp and the
per-model reference probes write this layout). Reports top-1 agreement,
KL(reference || candidate) and the largest absolute logit difference.
Standard library only, so it runs outside the development environment.

  compare_logits.py REFERENCE.bin CANDIDATE.bin --vocab 131072
"""

import argparse
import math
import statistics
import sys
from array import array


def read_rows(path: str, vocab: int) -> list:
    values = array("f")
    with open(path, "rb") as f:
        values.frombytes(f.read())
    if len(values) % vocab:
        raise ValueError(f"{path}: size is not a multiple of the vocabulary")
    return [values[i:i + vocab] for i in range(0, len(values), vocab)]


def log_softmax(row) -> list:
    peak = max(row)
    log_denom = math.log(math.fsum(math.exp(v - peak) for v in row))
    return [v - peak - log_denom for v in row]


def argmax(row) -> int:
    return max(range(len(row)), key=row.__getitem__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--vocab", type=int, required=True)
    args = parser.parse_args()

    try:
        ref = read_rows(args.reference, args.vocab)
        cand = read_rows(args.candidate, args.vocab)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    rows = min(len(ref), len(cand))
    if len(ref) != len(cand):
        print(f"comparing the first {rows} of {len(ref)} / {len(cand)} rows")

    kls = []
    agree = 0
    print("pos  top1(ref) top1(cand)  p_ref(top1)        KL   max|dlogit|")
    for i in range(rows):
        lp_ref = log_softmax(ref[i])
        lp_cand = log_softmax(cand[i])
        kl = math.fsum(math.exp(a) * (a - b) for a, b in zip(lp_ref, lp_cand))
        top_ref, top_cand = argmax(ref[i]), argmax(cand[i])
        max_abs = max(abs(a - b) for a, b in zip(ref[i], cand[i]))
        kls.append(kl)
        agree += top_ref == top_cand
        mark = "" if top_ref == top_cand else "  <-- top-1 differs"
        print(f"{i:3d}  {top_ref:9d} {top_cand:10d}  "
              f"{math.exp(lp_ref[top_ref]):11.5f}  {kl:9.2e}  "
              f"{max_abs:11.4f}{mark}")
    p90 = sorted(kls)[min(rows - 1, int(0.9 * rows))]
    print(f"top-1 agreement {agree}/{rows}; KL median "
          f"{statistics.median(kls):.2e}, p90 {p90:.2e}, max {max(kls):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
