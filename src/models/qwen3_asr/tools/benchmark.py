#!/usr/bin/env python3
"""Benchmark native Qwen3-ASR with one resident model runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gufo", type=Path, default=Path("result/bin/gufo"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median(runs: list[dict[str, float]], field: str) -> float:
    return statistics.median(float(run[field]) for run in runs)


def main() -> int:
    args = parse_args()
    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup non-negative")
    command = [
        str(args.gufo),
        "transcribe",
        "--model",
        str(args.model),
        "--audio",
        str(args.audio),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--format",
        "json",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=1800
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"native Qwen3-ASR benchmark failed: {detail}") from error
    native = json.loads(completed.stdout)
    runs = native["runs"]
    if not runs:
        raise ValueError("native Qwen3-ASR benchmark returned no measured runs")
    # Use the actual decoded/resampled input. Python's wave reader rejects
    # float WAV, which the native CLI supports.
    duration = native["audio_samples"] / native["sample_rate"]
    if duration <= 0:
        raise ValueError("native Qwen3-ASR benchmark returned empty audio")
    total_ms = median(runs, "total_ms")
    report = {
        "schema": "gufo.qwen3-asr.benchmark.v1",
        "model": str(args.model),
        "audio": str(args.audio),
        "audio_seconds": duration,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "load_ms": native["timings"]["load_ms"],
        "median_total_ms": total_ms,
        "median_realtime_factor": total_ms / (duration * 1000.0),
        "median_audio_seconds_per_wall_second": duration / (total_ms / 1000.0),
        "median_decode_audio_ms": median(runs, "decode_audio_ms"),
        "median_feature_extraction_ms": median(
            runs, "feature_extraction_ms"
        ),
        "median_audio_encoder_ms": median(runs, "audio_encoder_ms"),
        "median_text_decoder_ms": median(runs, "text_decoder_ms"),
        "transcript": native["text"],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
