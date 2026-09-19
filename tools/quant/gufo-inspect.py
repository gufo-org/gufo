#!/usr/bin/env python3
"""gufo-inspect — validate a safetensors snapshot, write source-manifest.json.

Uses tools/gufo/safetensors.py for strict header, tensor, and shard validation.

Usage:
  gufo-inspect --source DIR [--out manifest.json] [--json]
"""
import sys
import os
import json
import hashlib
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo import safetensors
from gufo.manifest import write_source_manifest, sha256_text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gufo-inspect")
    ap.add_argument("--source", required=True, help="HF model snapshot directory")
    ap.add_argument("--out", default="artifacts/work/source-manifest.json")
    ap.add_argument("--repository", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--revision", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.source)
    index = src / "model.safetensors.index.json"
    tensors, files = safetensors.inspect_snapshot(src, index if index.exists() else None)

    file_records = []
    for p in files:
        file_records.append({
            "path": p.name,
            "size": p.stat().st_size,
            "sha256": safetensors.sha256_file(p),
        })

    tokenizer_sha = sha256_text((src / "tokenizer.json").read_bytes())
    template_sha = ""
    tp = src / "chat_template.jinja"
    if tp.exists():
        template_sha = sha256_text(tp.read_bytes())

    # Determine source storage dtype from config.json
    arch = ""
    storage = "BF16"
    cfgp = src / "config.json"
    if cfgp.exists():
        cfg = json.loads(cfgp.read_text("utf-8"))
        arch = cfg.get("architectures", [""])[0]
        storage = cfg.get("text_config", {}).get("dtype", "BF16")

    manifest = write_source_manifest(
        Path(args.out), repository=args.repository, revision=args.revision,
        architecture=arch, source_storage_dtype=storage, files=file_records,
        tokenizer_sha=tokenizer_sha, template_sha=template_sha)

    info = {
        "manifest": str(manifest),
        "tensor_count": len(tensors),
        "files": len(files),
        "architecture": arch,
        "source_dtype": storage,
    }
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"source manifest: {manifest}")
        print(f"tensors: {len(tensors)}, files: {len(files)}, arch: {arch}, dtype: {storage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
