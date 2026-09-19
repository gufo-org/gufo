#!/usr/bin/env python3
"""gufo-compare — compare candidate logit artifact against teacher artifact.

Validates matching schemas, vocabularies, prompt tokens, and positions,
then computes KL divergence, top-k agreement, logit RMSE, and perplexity.

Usage:
  gufo-compare --teacher DIR --candidate DIR [--max-kl FLOAT] [--min-top1 FLOAT] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo import manifest as gufo_manifest
from gufo import quality


def load_artifact_logits(artifact_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Validate and load full logits and tokens from an artifact directory."""
    manifest = gufo_manifest.validate_logit_artifact(artifact_dir)
    vocab_size = manifest["vocab_size"]

    tokens_path = artifact_dir / "tokens.u32"
    tokens = np.frombuffer(tokens_path.read_bytes(), dtype=np.uint32) if tokens_path.exists() else np.array([])

    dctx = zstandard.ZstdDecompressor()
    all_logits = []

    for chunk_info in manifest["chunks"]:
        chunk_file = artifact_dir / chunk_info["file"]
        compressed = chunk_file.read_bytes()
        expected_size = chunk_info["token_count"] * vocab_size * 4
        raw = dctx.decompress(compressed, max_output_size=expected_size + 1024)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(chunk_info["token_count"], vocab_size)
        all_logits.append(arr)

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, vocab_size), dtype=np.float32)
    return manifest, tokens, logits


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-compare", description="Compare candidate vs teacher logit artifacts")
    ap.add_argument("--teacher", required=True, help="Path to teacher artifact directory")
    ap.add_argument("--candidate", required=True, help="Path to candidate artifact directory")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k overlap parameter (default: 5)")
    ap.add_argument("--max-kl-mean", type=float, default=None, help="Maximum allowed mean KL divergence")
    ap.add_argument("--min-top1", type=float, default=None, help="Minimum allowed top-1 agreement fraction (e.g. 0.99)")
    ap.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    args = ap.parse_args(argv)

    teacher_dir = Path(args.teacher)
    candidate_dir = Path(args.candidate)

    try:
        t_manifest, t_tokens, t_logits = load_artifact_logits(teacher_dir)
        c_manifest, c_tokens, c_logits = load_artifact_logits(candidate_dir)
    except Exception as e:
        print(f"Artifact validation error: {e}", file=sys.stderr)
        return 1

    # Verify compatibility
    if t_manifest["vocab_size"] != c_manifest["vocab_size"]:
        print(f"Error: Vocab size mismatch: teacher {t_manifest['vocab_size']} vs candidate {c_manifest['vocab_size']}", file=sys.stderr)
        return 1

    if t_manifest["token_stream_hash"] != c_manifest["token_stream_hash"]:
        print("Error: Token stream hash mismatch between teacher and candidate", file=sys.stderr)
        return 1

    if t_logits.shape != c_logits.shape:
        print(f"Error: Logit shape mismatch: teacher {t_logits.shape} vs candidate {c_logits.shape}", file=sys.stderr)
        return 1

    # Compute comparison metrics
    target_tokens = t_tokens if len(t_tokens) == t_logits.shape[0] else None
    metrics = quality.compare_logits(t_logits, c_logits, target_tokens=target_tokens, top_k=args.top_k)

    report = {
        "schema": "gufo.logit-comparison.v1",
        "teacher_tag": t_manifest.get("model_tag", "teacher"),
        "candidate_tag": c_manifest.get("model_tag", "candidate"),
        "metrics": metrics,
    }

    if args.json or True:
        print(json.dumps(report, indent=2))

    # Evaluate gates if provided
    if args.max_kl_mean is not None and metrics.get("kl_mean", 0.0) > args.max_kl_mean:
        print(f"GATE FAILED: kl_mean {metrics['kl_mean']} > threshold {args.max_kl_mean}", file=sys.stderr)
        return 2

    if args.min_top1 is not None and metrics.get("top1_agreement", 0.0) < args.min_top1:
        print(f"GATE FAILED: top1_agreement {metrics['top1_agreement']} < threshold {args.min_top1}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
