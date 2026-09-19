"""Unit tests for teacher logit artifact generation and manifest validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import zstandard

from tools.gufo import manifest as gufo_manifest


class TestCaptureArtifact(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_artifact(self, vocab_size: int = 128, num_prompts: int = 2) -> Path:
        ctx = zstandard.ZstdCompressor(level=3)
        positions = []
        chunks = []
        all_tokens = []

        for pi in range(num_prompts):
            tokens = [10, 20, 30 + pi]
            all_tokens.extend(tokens)
            positions.append({"prompt": pi, "offset": pi * 3, "tokens": len(tokens)})

            # Synthetic logits [T, V]
            dummy_logits = np.random.randn(len(tokens), vocab_size).astype(np.float32)
            compressed = ctx.compress(dummy_logits.tobytes())
            chunk_file = f"logits-{pi:05d}.f32.zst"
            (self.artifact_dir / chunk_file).write_bytes(compressed)

            chunks.append({
                "chunk_id": pi,
                "file": chunk_file,
                "token_count": len(tokens),
                "vocab_size": vocab_size,
                "sha256": hashlib.sha256(compressed).hexdigest(),
            })

        tokens_arr = np.array(all_tokens, dtype=np.uint32)
        (self.artifact_dir / "tokens.u32").write_bytes(tokens_arr.tobytes())
        (self.artifact_dir / "positions.json").write_text(json.dumps(positions, indent=2), "utf-8")

        metrics = {
            "positions": len(all_tokens),
            "perplexity": 12.34,
            "nll_mean": 2.51,
            "prompts": num_prompts,
            "vocab_size": vocab_size,
        }
        (self.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

        gufo_manifest.write_logit_manifest(
            self.artifact_dir,
            model_family="qwen35",
            model_tag="qwen3.5-synthetic",
            source_dir=str(self.artifact_dir),
            suite_hash="abc123suite",
            token_stream_hash=hashlib.sha256(tokens_arr.tobytes()).hexdigest(),
            vocab_size=vocab_size,
            positions=positions,
            metrics=metrics,
            chunks=chunks,
        )

        lines = []
        for p in sorted(self.artifact_dir.glob("*")):
            if p.name == "checksums.sha256":
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}")
        (self.artifact_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", "utf-8")

        return self.artifact_dir

    def test_valid_artifact_validation(self):
        self._create_synthetic_artifact(vocab_size=64, num_prompts=3)
        manifest = gufo_manifest.validate_logit_artifact(self.artifact_dir)
        self.assertEqual(manifest["schema"], "gufo.logit-artifact.v1")
        self.assertEqual(manifest["vocab_size"], 64)
        self.assertEqual(manifest["chunk_count"], 3)

    def test_corrupted_chunk_detection(self):
        self._create_synthetic_artifact(vocab_size=64, num_prompts=2)
        # Corrupt the first chunk
        chunk_path = self.artifact_dir / "logits-00000.f32.zst"
        corrupted = bytearray(chunk_path.read_bytes())
        corrupted[10] ^= 0xFF
        chunk_path.write_bytes(bytes(corrupted))

        with self.assertRaises(ValueError):
            gufo_manifest.validate_logit_artifact(self.artifact_dir)

    def test_missing_chunk_detection(self):
        self._create_synthetic_artifact(vocab_size=64, num_prompts=2)
        # Delete chunk
        (self.artifact_dir / "logits-00001.f32.zst").unlink()

        with self.assertRaises(FileNotFoundError):
            gufo_manifest.validate_logit_artifact(self.artifact_dir)


if __name__ == "__main__":
    unittest.main()
