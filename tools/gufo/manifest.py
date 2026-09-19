"""Manifest helpers (source-manifest.json, quantization-plan.json, logit artifacts)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zstandard


def sha256_text(s) -> str:
    data = s if isinstance(s, bytes) else s.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, sort_keys=True) + "\n", "utf-8")
    return path


def write_source_manifest(out: Path, *, repository, revision, architecture,
                          source_storage_dtype, files: list, tokenizer_sha,
                          template_sha):
    manifest = {
        "schema": "gufo.source.v1",
        "repository": repository,
        "revision": revision,
        "architecture": architecture,
        "source_storage_dtype": source_storage_dtype,
        "tokenizer_sha256": tokenizer_sha,
        "chat_template_sha256": template_sha,
        "files": files,
    }
    write_json(out, manifest)
    return out


def write_logit_manifest(out_dir: Path, *,
                         model_family: str = "qwen35",
                         model_tag: str,
                         source_dir: str,
                         suite_hash: str,
                         token_stream_hash: str,
                         vocab_size: int,
                         positions: list,
                         metrics: dict,
                         chunks: list,
                         storage_dtype: str = "bfloat16",
                         accumulation_dtype: str = "float32",
                         serialization_dtype: str = "float32",
                         reference_runtime: str = "torch-rocm/transformers"):
    """Write validated schema-v1 teacher logit artifact manifest."""
    source_p = Path(source_dir)
    source_files = {}
    if source_p.is_dir():
        for p in sorted(source_p.glob("*")):
            if p.is_file() and p.suffix in (".json", ".safetensors", ".txt"):
                source_files[p.name] = sha256_file(p)

    manifest = {
        "schema": "gufo.logit-artifact.v1",
        "model_family": model_family,
        "model_tag": model_tag,
        "source_files": source_files,
        "suite_hash": suite_hash,
        "token_stream_hash": token_stream_hash,
        "storage_dtype": storage_dtype,
        "accumulation_dtype": accumulation_dtype,
        "serialization_dtype": serialization_dtype,
        "reference_runtime": reference_runtime,
        "vocab_size": int(vocab_size),
        "positions": positions,
        "metrics": metrics,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def validate_logit_artifact(artifact_dir: Path) -> dict:
    """Validate a teacher logit artifact against its manifest and checksums."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {artifact_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("schema") != "gufo.logit-artifact.v1":
        raise ValueError(f"Invalid schema: {manifest.get('schema')}")

    vocab_size = manifest["vocab_size"]
    chunks = manifest.get("chunks", [])
    if len(chunks) != manifest.get("chunk_count", 0):
        raise ValueError("Chunk count mismatch in manifest")

    dctx = zstandard.ZstdDecompressor()

    for chunk_info in chunks:
        chunk_file = artifact_dir / chunk_info["file"]
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk file: {chunk_file}")

        file_sha = sha256_file(chunk_file)
        if file_sha != chunk_info["sha256"]:
            raise ValueError(f"Checksum mismatch for {chunk_file.name}: expected {chunk_info['sha256']}, got {file_sha}")

        # Verify decompression and shape
        compressed_bytes = chunk_file.read_bytes()
        expected_bytes = chunk_info["token_count"] * vocab_size * 4
        decompressed = dctx.decompress(compressed_bytes, max_output_size=expected_bytes + 1024)
        if len(decompressed) != expected_bytes:
            raise ValueError(f"Decompressed byte count mismatch for {chunk_file.name}: expected {expected_bytes}, got {len(decompressed)}")

    # Verify checksums.sha256 file if present
    checksums_file = artifact_dir / "checksums.sha256"
    if checksums_file.exists():
        for line in checksums_file.read_text("utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_sha, fname = parts[0], parts[1].strip()
                target_file = artifact_dir / fname
                if not target_file.exists():
                    raise FileNotFoundError(f"File listed in checksums.sha256 not found: {fname}")
                actual_sha = sha256_file(target_file)
                if actual_sha != expected_sha:
                    raise ValueError(f"Checksum verification failed for {fname}")

    return manifest
