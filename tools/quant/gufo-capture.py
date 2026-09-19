#!/usr/bin/env python3
"""gufo-capture — capture full-precision teacher logits + perplexity.

Teacher-forced matched-token capture per docs/TESTING.md. Logits are chunked
by position and zstd-compressed under an artifact-id directory (never
committed). Writes manifest.json, tokens.u32, logits-*.f32.zst, metrics.json,
checksums.sha256.

Usage:
  gufo-capture --source DIR --suite FILE --out DIR [--model-tag qwen3.5-4b-bf16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gufo import manifest as gufo_manifest
from gufo import model as strix_model
from gufo import quality


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-capture")
    ap.add_argument("--source", required=True, help="Directory containing source weights & tokenizer")
    ap.add_argument("--suite", required=True, help="Path to prompt evaluation suite JSON")
    ap.add_argument("--out", default="artifacts/teacher", help="Output artifact directory")
    ap.add_argument("--model-tag", default="qwen3.5-4b-bf16", help="Model descriptor identifier")
    args = ap.parse_args(argv)

    suite_path = Path(args.suite)
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes.decode("utf-8"))
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()

    model, tok = strix_model.load_teacher(args.source)
    vocab_size = int(getattr(model.config, "vocab_size", len(tok)))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Tokenize each prompt; teacher-force next-token prediction.
    all_tokens = []
    positions = []  # (prompt_idx, start)
    logit_chunks = []
    ctx = zstandard.ZstdCompressor(level=3, write_content_size=True)

    for pi, text in enumerate(suite["prompts"]):
        ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()  # [T]
        all_tokens.append(ids)
        positions.append({"prompt": pi, "offset": len(all_tokens) - 1, "tokens": len(ids)})
        with torch.no_grad():
            logits = strix_model.forward_logits(model, torch.tensor([ids], dtype=torch.long))

        # logits [1, T, vocab]
        logits = logits[0].float().numpy()  # [T, V]
        assert logits.shape[1] == vocab_size, f"Logit vocab width ({logits.shape[1]}) != model vocab ({vocab_size})"
        chunk = np.ascontiguousarray(logits, dtype=np.float32)
        logit_chunks.append(chunk.tobytes())

    tokens = np.concatenate([np.array(t, dtype=np.uint32) for t in all_tokens])
    token_stream_bytes = tokens.tobytes()
    token_stream_hash = hashlib.sha256(token_stream_bytes).hexdigest()
    (outdir / "tokens.u32").write_bytes(token_stream_bytes)

    with open(outdir / "positions.json", "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2)

    # Write logit chunks (zstd), one file per prompt
    chunks_meta = []
    for pi, b in enumerate(logit_chunks):
        chunk_fname = f"logits-{pi:05d}.f32.zst"
        compressed = ctx.compress(b)
        chunk_path = outdir / chunk_fname
        chunk_path.write_bytes(compressed)
        chunk_sha = hashlib.sha256(compressed).hexdigest()
        chunks_meta.append({
            "chunk_id": pi,
            "file": chunk_fname,
            "token_count": len(all_tokens[pi]),
            "vocab_size": vocab_size,
            "sha256": chunk_sha,
        })
        # Verify uncompressed byte length matches [T, V, 4]
        assert len(b) == len(all_tokens[pi]) * vocab_size * 4

    # Perplexity + metrics from teacher (NLL on target tokens)
    nlls = []
    dctx = zstandard.ZstdDecompressor()
    for pi, ids in enumerate(all_tokens):
        fbytes = (outdir / f"logits-{pi:05d}.f32.zst").read_bytes()
        decomp = dctx.decompress(fbytes, max_output_size=len(ids) * vocab_size * 4 + 1024)
        logits = np.frombuffer(decomp, dtype=np.float32).reshape(len(ids), vocab_size)
        lp = quality.log_softmax(logits)
        p = np.exp(lp)
        tgt = ids[1:]
        nlls.append(-np.log(np.clip(p[np.arange(len(tgt)), tgt], 1e-30, 1.0)))

    nll = np.concatenate(nlls)
    metrics = {
        "positions": int(len(nll)),
        "perplexity": float(np.exp(nll.mean())),
        "nll_mean": float(nll.mean()),
        "prompts": len(all_tokens),
        "vocab_size": vocab_size,
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write logit manifest
    gufo_manifest.write_logit_manifest(
        outdir,
        model_family="qwen35",
        model_tag=args.model_tag,
        source_dir=args.source,
        suite_hash=suite_hash,
        token_stream_hash=token_stream_hash,
        vocab_size=vocab_size,
        positions=positions,
        metrics=metrics,
        chunks=chunks_meta,
        storage_dtype="bfloat16",
        accumulation_dtype="float32",
        serialization_dtype="float32",
        reference_runtime=f"torch-{torch.__version__}",
    )

    # Checksums over all files in artifact
    lines = []
    for p in sorted(outdir.glob("*")):
        if p.name == "checksums.sha256":
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (outdir / "checksums.sha256").write_text("\n".join(lines) + "\n", "utf-8")

    # Validate output artifact
    gufo_manifest.validate_logit_artifact(outdir)

    print(json.dumps({"artifact": str(outdir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
