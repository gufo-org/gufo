"""Unit tests for matched-token full-logit comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import zstandard

from tools.gufo import manifest as gufo_manifest
from tools.gufo import quality


class TestLogitCompare(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_artifact(self, out_dir: Path, logits: np.ndarray, tokens: list[int], tag: str = "model") -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = zstandard.ZstdCompressor(level=3)
        vocab_size = logits.shape[1]

        compressed = ctx.compress(logits.astype(np.float32).tobytes())
        chunk_file = "logits-00000.f32.zst"
        (out_dir / chunk_file).write_bytes(compressed)

        tokens_arr = np.array(tokens, dtype=np.uint32)
        (out_dir / "tokens.u32").write_bytes(tokens_arr.tobytes())

        positions = [{"prompt": 0, "offset": 0, "tokens": len(tokens)}]
        (out_dir / "positions.json").write_text(json.dumps(positions, indent=2), "utf-8")

        chunks = [{
            "chunk_id": 0,
            "file": chunk_file,
            "token_count": len(tokens),
            "vocab_size": vocab_size,
            "sha256": hashlib.sha256(compressed).hexdigest(),
        }]

        metrics = {"positions": len(tokens), "prompts": 1, "vocab_size": vocab_size}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

        gufo_manifest.write_logit_manifest(
            out_dir,
            model_family="qwen35",
            model_tag=tag,
            source_dir=str(out_dir),
            suite_hash="suite_sha",
            token_stream_hash=hashlib.sha256(tokens_arr.tobytes()).hexdigest(),
            vocab_size=vocab_size,
            positions=positions,
            metrics=metrics,
            chunks=chunks,
        )

        lines = []
        for p in sorted(out_dir.glob("*")):
            if p.name == "checksums.sha256":
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}")
        (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", "utf-8")

        return out_dir

    def test_exact_match(self):
        np.random.seed(42)
        logits = np.random.randn(10, 64).astype(np.float32)
        tokens = list(range(10))

        metrics = quality.compare_logits(logits, logits, target_tokens=tokens)
        self.assertAlmostEqual(metrics["kl_mean"], 0.0, places=5)
        self.assertEqual(metrics["top1_agreement"], 1.0)
        self.assertAlmostEqual(metrics["logit_rmse"], 0.0, places=5)
        self.assertEqual(metrics["nonfinite_teacher"], 0)
        self.assertEqual(metrics["nonfinite_candidate"], 0)

    def test_slight_perturbation(self):
        np.random.seed(42)
        t_logits = np.random.randn(20, 64).astype(np.float32)
        c_logits = t_logits + (np.random.randn(20, 64).astype(np.float32) * 0.01)

        metrics = quality.compare_logits(t_logits, c_logits)
        self.assertLess(metrics["kl_mean"], 0.01)
        self.assertGreaterEqual(metrics["top1_agreement"], 0.95)
        self.assertLess(metrics["logit_rmse"], 0.02)

    def test_nonfinite_rejection(self):
        t_logits = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        c_logits = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        metrics = quality.compare_logits(t_logits, c_logits)
        self.assertFalse(metrics["valid"])
        self.assertEqual(metrics["nonfinite_teacher"], 1)


if __name__ == "__main__":
    unittest.main()
