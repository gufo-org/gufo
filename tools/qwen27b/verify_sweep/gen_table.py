#!/usr/bin/env python3
"""Turn two verify_sweep runs into the verify_gemm_27b.hip table.

    python3 tools/qwen27b/verify_sweep/gen_table.py \\
      --target target.gguf --draft draft.gguf \\
      --first target1.log draft1.log --second target2.log draft2.log \\
      [--out table.inc]

The logs are the raw lines that sweep.py writes with --out, one target and one
draft log per independent sweep. For each format, shape and row count, the
candidate that saves the most time over `generic` in the first sweep is
listed when that saving is at least --min-gain percent and the same candidate
also saves at least --min-gain percent in the second sweep. The saving is
summed over target and draft, each weighted by the number of layers that run
the projection, so one entry serves both models. Saving is
(generic - candidate) / generic, the gain sweep.py prints. Equal first-sweep
savings go to the candidate the second sweep favours.

Prints the decision for every shape, then the LaunchQ4K, LaunchQ5K, LaunchQ6K
and LaunchIQ4XS functions of the table (to stdout or --out). Paste them into
the platform's block of verify_gemm_27b.hip. Check the new kernels for
scratch first; pass a candidate that spills with --skip so it is never
picked.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import TYPE_NAMES, projections  # noqa: E402

FUNCTIONS = {12: "LaunchQ4K", 13: "LaunchQ5K", 14: "LaunchQ6K", 23: "LaunchIQ4XS"}
ENUMS = {12: "kQ4_K", 13: "kQ5_K", 14: "kQ6_K", 23: "kIQ4_XS"}
# Projection names used in the comments of verify_gemm_27b.hip; other shapes
# take the tensor name from the GGUF.
ROLES = {(1024, 5120): "attn_k/attn_v", (1280, 5120): "draft conv",
         (4096, 5120): "draft attn_q", (5120, 4096): "draft attn_output",
         (5120, 17408): "ffn_down", (6144, 5120): "attn_gate",
         (10240, 5120): "attn_qkv", (12288, 5120): "attn_q",
         (17408, 5120): "ffn_gate/ffn_up",
         (34816, 5120): "joined ffn_gate+ffn_up",
         (5120, 6144): "ssm_out/attn_output"}
BASE = re.compile(r"^base type=(\d+) m=(\d+) k=(\d+) batch=(\d+) ")
CANDIDATE = re.compile(r"^cand (.+) ([\d.]+) us \d+ GB/s [+-][\d.]+%$")


def layers(model: Path, min_layers: int):
    """Return {(type, m, k): (layer count, tensor role)} as sweep.py sweeps them."""
    found = {}
    for kind, m, k, _offset, count, name in projections(model):
        if count >= min_layers:
            role = name if name.startswith("packed") else name.split(".")[2]
            found[(kind, m, k)] = (count, role)
    return found


def timings(log: Path):
    """Return {(type, m, k, rows): {candidate: us}} without inexact candidates."""
    result, current = {}, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if match := BASE.match(line):
            current = result.setdefault(tuple(int(v) for v in match.groups()), {})
        elif (match := CANDIDATE.match(line)) and current is not None:
            current[match.group(1)] = float(match.group(2))
    return result


def combine(logs, models):
    """Return {(type, m, k, rows): {candidate: layer-weighted us}} of one sweep.

    A candidate counts only where it is exact in every model that runs the
    shape."""
    parts = [(timings(log), model) for log, model in zip(logs, models)]
    combined = collections.defaultdict(dict)
    keys = {key for result, model in parts for key in result if key[:3] in model}
    for key in keys:
        used = [(result[key], model[key[:3]][0]) for result, model in parts
                if key in result and key[:3] in model]
        # Keep the sweep's candidate order, so a tie picks the same candidate
        # on every run.
        names = [n for n in used[0][0] if all(n in result for result, _ in used)]
        for name in names:
            combined[key][name] = sum(result[name] * count
                                      for result, count in used)
    return combined


def saving(times, name):
    if name not in times or "generic" not in times:
        return float("nan")
    return (times["generic"] - times[name]) / times["generic"] * 100


def emit(listed):
    lines = []
    for kind in (12, 13, 14, 23):
        name = FUNCTIONS[kind]
        lines += [f"bool {name}(const void* w, const float* x, float* y, "
                  "std::size_t batch,",
                  " " * (len(name) + 6)
                  + "std::size_t m, std::size_t k, hipStream_t stream) {"]
        shapes = sorted({(m, k) for (t, m, k, _) in listed if t == kind})
        if shapes:
            lines.append(f"  constexpr auto T = GgmlType::{ENUMS[kind]};")
        for m, k in shapes:
            rows = sorted(b for (t, mm, kk, b) in listed
                          if (t, mm, kk) == (kind, m, k))
            role = ROLES.get((m, k), listed[(kind, m, k, rows[0])][2])
            lines += [f"  if (Is(m, k, {m}, {k})) {{  // {role}",
                      "    switch (batch) {"]
            for batch in rows:
                launch, gain, _ = listed[(kind, m, k, batch)]
                lines += [f"      case {batch}:  // +{gain:.0f}%",
                          f"        return {launch}(w, x, y, m, k, stream);"]
            lines += ["      default:", "        return false;", "    }", "  }"]
        lines += ["  return false;", "}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=Path, required=True, help="target GGUF")
    parser.add_argument("--draft", type=Path, required=True,
                        help="DFlash2 draft GGUF")
    parser.add_argument("--first", type=Path, nargs=2, required=True,
                        metavar=("TARGET_LOG", "DRAFT_LOG"),
                        help="raw logs of the sweep that picks the shapes")
    parser.add_argument("--second", type=Path, nargs=2, required=True,
                        metavar=("TARGET_LOG", "DRAFT_LOG"),
                        help="raw logs of the independent sweep that confirms them")
    parser.add_argument("--min-gain", type=float, default=5.0,
                        help="percent saved over generic in both sweeps")
    parser.add_argument("--skip", action="append", default=[],
                        metavar="TYPE:LAUNCH",
                        help='never pick this candidate, e.g. '
                        '"Q4_K:Launch<T, 7, 4, 4, 1, 7, 8, 16>" (repeatable)')
    parser.add_argument("--out", type=Path, help="write the table here")
    args = parser.parse_args()

    type_ids = {name: kind for kind, name in TYPE_NAMES.items()}
    skip = set()
    for item in args.skip:
        type_name, _, launch = item.partition(":")
        if type_name not in type_ids or not launch.startswith("Launch<"):
            parser.error(f"--skip {item!r}: expected TYPE:Launch<...>")
        skip.add((type_ids[type_name], launch))
    # The same layer filters as sweep.py: the draft is swept with --min-layers 1.
    models = [layers(args.target, 2), layers(args.draft, 1)]
    first, second = combine(args.first, models), combine(args.second, models)

    listed = {}
    print("format  m x k         rows  candidate (first sweep)            "
          "first  second  layers  result")
    for key in sorted(first):
        times = first[key]
        names = [n for n in times if n != "generic" and (key[0], n) not in skip]
        if "generic" not in times or not names:
            continue
        # Ties (equal medians) go to the candidate the second sweep favours.
        best = max(names, key=lambda n: (round(saving(times, n), 6),
                                         saving(second.get(key, {}), n)))
        gain, again = saving(times, best), saving(second.get(key, {}), best)
        ok = gain >= args.min_gain and again >= args.min_gain
        counts = "/".join(str(model.get(key[:3], (0,))[0]) for model in models)
        result = "listed" if ok else "not reproduced" if gain >= args.min_gain else ""
        print(f"{TYPE_NAMES[key[0]]:6} {key[1]:>6}x{key[2]:<6} {key[3]}  "
              f"{best:36} {gain:+6.1f}% {again:+6.1f}%  {counts:>6}  {result}")
        if ok:
            role = next(model[key[:3]][1] for model in models if key[:3] in model)
            listed[key] = (best, gain, role)
    print(f"listed {len(listed)} of {len(first)}")

    table = emit(listed)
    if args.out:
        args.out.write_text(table, encoding="utf-8", newline="\n")
    else:
        print()
        print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
