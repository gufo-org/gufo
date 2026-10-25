#!/usr/bin/env python3
"""Compare Gemma 4 vision encoder stage dumps (u32 dims[4] + f32 data).

    python3 tools/gemma4/compare_stages.py REF_DIR OTHER_DIR [--map llama]

With --map llama, OTHER_DIR is a llama_vision --dump-dir: stage "patch" is
read from pos_embd, "layer<i>" from layer_out-<i>, "pooled" from pooled and
"embedding" from projected.
"""
import argparse
import math
import struct
import sys
from pathlib import Path


def load(path: Path):
    raw = path.read_bytes()
    dims = struct.unpack_from("<4I", raw)
    count = (len(raw) - 16) // 4
    return dims, struct.unpack_from(f"<{count}f", raw, 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("--map", choices=["same", "llama"], default="same")
    args = parser.parse_args()
    stages = ["patch"] + [f"layer{i}" for i in range(27)] + ["pooled", "embedding"]
    names = {"patch": "pos_embd", "pooled": "pooled", "embedding": "projected"}
    for stage in stages:
        other_name = stage
        if args.map == "llama":
            other_name = names.get(stage, "layer_out-" + stage[5:])
        ref_path = args.reference / f"{stage}.f32"
        other_path = args.other / f"{other_name}.f32"
        if not ref_path.exists() or not other_path.exists():
            continue
        (_, a), (_, b) = load(ref_path), load(other_path)
        if len(a) != len(b):
            print(f"{stage:10s} size {len(a)} vs {len(b)}")
            continue
        err = sum((x - y) ** 2 for x, y in zip(a, b))
        ref = sum(x * x for x in a)
        worst = max(abs(x - y) for x, y in zip(a, b))
        print(f"{stage:10s} rel rms {math.sqrt(err / ref):.3e}  max abs {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
