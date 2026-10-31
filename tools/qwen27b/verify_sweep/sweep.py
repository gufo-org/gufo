#!/usr/bin/env python3
"""Sweep the Qwen verification launch geometries for one GGUF on this GPU.

    python3 tools/qwen27b/verify_sweep/sweep.py --binary <qwen27b_verify_sweep> <model.gguf>

Lists every distinct (type, m, k) projection of the transformer layers, runs
qwen27b_verify_sweep on one example tensor of each at every requested row
count, and prints the fastest exact candidate per projection and row count.
Gains of at least --min-gain percent are also printed in the Launch<> syntax
of src/families/qwen/kernels/verify_gemm_27b.hip. Gate/up pairs of one
format stored back to back are measured as the joined (2 * m) x k matrix the
decode multiplies. The GPU must be idle for the timings to mean anything.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import re
import struct
import subprocess
import sys

# GGUF type id -> bytes per 256 weights; the formats the sweep binary measures.
BLOCK_BYTES = {12: 144, 13: 176, 14: 210, 23: 136}
TYPE_NAMES = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 23: "IQ4_XS"}
CANDIDATE = re.compile(r"^cand (.+) [\d.]+ us (\d+) GB/s ([+-][\d.]+)%$")
BASE = re.compile(r"^base .* [\d.]+ us (\d+) GB/s$")


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_string(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def read_value(f, kind):
    scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
               10: "Q", 11: "q", 12: "d"}
    if kind in scalars:
        return read(f, scalars[kind])
    if kind == 8:
        return read_string(f)
    if kind == 9:
        inner = read(f, "I")
        for _ in range(read(f, "Q")):
            read_value(f, inner)
        return None
    raise ValueError(f"unknown GGUF value type {kind}")


def tensors(path):
    """Return [(name, type, ne, absolute offset)] without reading weights."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        read(f, "I")
        n_tensors, n_kv = read(f, "Q"), read(f, "Q")
        alignment = 32
        for _ in range(n_kv):
            key = read_string(f)
            value = read_value(f, read(f, "I"))
            if key == "general.alignment":
                alignment = value
        infos = []
        for _ in range(n_tensors):
            name = read_string(f)
            ne = [read(f, "Q") for _ in range(read(f, "I"))]
            kind = read(f, "I")
            infos.append((name, kind, ne, read(f, "Q")))
        start = (f.tell() + alignment - 1) // alignment * alignment
    return [(name, kind, ne, start + offset) for name, kind, ne, offset in infos]


def projections(path):
    """Return [(type, m, k, offset, layer_count, example_name)]."""
    layers = collections.defaultdict(dict)
    for name, kind, ne, offset in tensors(path):
        parts = name.split(".")
        if parts[0] != "blk" or len(ne) != 2 or kind not in BLOCK_BYTES:
            continue
        k, m = ne
        layers[parts[1]][parts[2]] = (kind, m, k, offset, name)
    found = collections.OrderedDict()
    for layer in layers.values():
        gate, up = layer.get("ffn_gate"), layer.get("ffn_up")
        packed = False
        if gate and up and gate[0] == up[0] and gate[1:3] == up[1:3]:
            size = gate[1] * (gate[2] // 256) * BLOCK_BYTES[gate[0]]
            packed = gate[3] + size == up[3]
        for role, (kind, m, k, offset, name) in layer.items():
            if packed and role in ("ffn_gate", "ffn_up"):
                continue  # multiplied as the joined matrix below
            found.setdefault((kind, m, k), [offset, 0, name])[1] += 1
        if packed:
            key = (gate[0], 2 * gate[1], gate[2])
            found.setdefault(key, [gate[3], 0, "packed ffn_gate+ffn_up"])[1] += 1
    return [(kind, m, k, offset, count, name)
            for (kind, m, k), (offset, count, name) in found.items()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--binary", type=Path, required=True,
                        help="qwen27b_verify_sweep executable")
    parser.add_argument("--batches", default="2,3,4,5,6,7,8")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--min-gain", type=float, default=5.0,
                        help="percent over production to suggest a shape")
    parser.add_argument("--min-layers", type=int, default=2,
                        help="skip projections used by fewer layers; "
                        "pass 1 for a DFlash2 draft")
    parser.add_argument("--out", type=Path, help="write the raw sweep lines")
    args = parser.parse_args()
    batches = [int(b) for b in args.batches.split(",")]

    log = []
    rows = []
    for kind, m, k, offset, count, name in projections(args.model):
        if count < args.min_layers:
            continue
        for batch in batches:
            lines = subprocess.run(
                [str(args.binary), str(args.model), str(offset), str(kind),
                 str(m), str(k), str(batch), str(args.rounds)],
                check=True, capture_output=True, text=True).stdout.splitlines()
            log += lines
            base = next((int(match.group(1)) for line in lines
                         if (match := BASE.match(line))), 0)
            best = max(((match.group(1), int(match.group(2)),
                         float(match.group(3))) for line in lines
                        if (match := CANDIDATE.match(line))),
                       key=lambda c: c[2], default=("-", 0, 0.0))
            generic = next((float(match.group(3)) for line in lines
                            if (match := CANDIDATE.match(line))
                            and match.group(1) == "generic"), 0.0)
            if "listed 1" not in lines:
                generic = 0.0  # production is the generic dispatch
            inexact = sum("NOT-EXACT" in line for line in lines)
            rows.append((TYPE_NAMES[kind], m, k, count, name, batch, base,
                         best[1], best[2], best[0], inexact, generic))
            print(f"{TYPE_NAMES[kind]:6} {m:>6} {k:>6} layers={count:<3} "
                  f"rows={batch} {base:>4} -> {best[1]:>4} GB/s "
                  f"{best[2]:+6.1f}%  {best[0]}"
                  + (f"  ({inexact} not exact)" if inexact else ""),
                  flush=True)
    if args.out:
        args.out.write_text("\n".join(log) + "\n", encoding="utf-8")

    print(f"\nGains of {args.min_gain:g}% or more, as verify_gemm_27b.hip "
          "entries (check the shape is not also a draft shape there):")
    for (type_name, m, k, count, name, batch, base, best_gbs, gain, launch,
         _, _) in rows:
        if gain >= args.min_gain and launch != "generic":
            print(f"  // {type_name} {m}x{k} ({count} layers, {name}), "
                  f"{batch} rows: {base} -> {best_gbs} GB/s, +{gain:g}%")
            print(f"  {launch}(w, x, y, m, k, stream);")
    slower = [row for row in rows if row[11] > 0]
    if slower:
        print("\nListed shapes slower than the generic dispatch; remove them "
              "or replace them with a shape that wins by --min-gain:")
        for (type_name, m, k, count, name, batch, *_, generic) in slower:
            print(f"  {type_name} {m}x{k} ({count} layers, {name}), "
                  f"{batch} rows: generic {generic:+g}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
