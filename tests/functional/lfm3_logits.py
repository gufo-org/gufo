#!/usr/bin/env python3
"""Independent LFM3 full-logit check against a pinned llama.cpp FP32 teacher.

The teacher's GGUF is a lossless BF16 -> F32 conversion. This prevents
llama.cpp's input quantization from confounding model/operator parity.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess

CASES = {
    "capital": [597, 5205, 302, 3980, 355],
    "unicode": [9792, 25677, 31, 34998, 229, 11253, 5899, 207, 206, 20406, 229, 3005],
    "chat": [124894, 124899, 5922, 207, 35808, 124900, 207, 124899, 63514, 207, 124901],
    "sliding-wrap": ([597, 5205, 302, 3980, 355] * 103),
}


def read(path):
    data = path.read_bytes()
    return struct.unpack("<" + "f" * (len(data) // 4), data)


def compare(actual, teacher):
    assert len(actual) == len(teacher) == 128000
    assert all(math.isfinite(x) for x in actual + teacher), "nonfinite logits"
    def probabilities(values):
        maximum = max(values)
        p = [math.exp(x - maximum) for x in values]
        total = sum(p)
        return [x / total for x in p]
    p, q = probabilities(teacher), probabilities(actual)
    return {"max_abs": max(abs(a - b) for a, b in zip(actual, teacher)),
            "rms": math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, teacher)) / len(actual)),
            "kl_teacher_candidate": sum(a * math.log(a / b) for a, b in zip(p, q) if a and b),
            "total_variation": 0.5 * sum(abs(a - b) for a, b in zip(p, q)),
            "top1_equal": max(range(len(actual)), key=actual.__getitem__) ==
                          max(range(len(teacher)), key=teacher.__getitem__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--q8", type=Path, required=True)
    parser.add_argument("--teacher-f32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for name, tokens in CASES.items():
            def run(label, command):
                out = args.output / f"{name}-{label}.bin"
                with (args.output / f"{name}-{label}.log").open("w") as log:
                    subprocess.run(command(out) + list(map(str, tokens)), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                return read(out)
            teacher = run("teacher", lambda out: [str(args.llama_probe.resolve()), str(args.teacher_f32), str(out)])
            bf16 = run("bf16", lambda out: [str(args.probe.resolve()), str(args.bf16), "gpu", str(out)])
            q8 = run("q8", lambda out: [str(args.probe.resolve()), str(args.q8), "gpu", str(out)])
            parity, quantization = compare(bf16, teacher), compare(q8, bf16)
            results[name] = {"tokens": tokens, "bf16_vs_teacher": parity, "q8_vs_bf16": quantization}
            print(name, json.dumps(results[name] | {"tokens": len(tokens)}), flush=True)
            # Different FP32 reduction trees are expected; test probabilities
            # and the selected token independently of absolute logit offsets.
            assert parity["top1_equal"] and parity["kl_teacher_candidate"] < 1e-5
            assert parity["total_variation"] < 0.002
            assert quantization["top1_equal"] and quantization["kl_teacher_candidate"] < 0.02
    finally:
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
