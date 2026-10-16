#!/usr/bin/env python3
"""Tests for the pinned MiniMax H3 source-manifest contract."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gufo import control_tokens, h3_manifest, safetensors  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), "utf-8"
    )


def tensor_bytes(dtype: str, shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count * safetensors.dtype_size(dtype)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    payload_size = 0
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        size = tensor_bytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [payload_size, payload_size + size],
        }
        payload_size += size
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(payload_size)
    )


class H3ManifestFixture:
    def __init__(self, root: Path):
        self.root = root
        self.create()

    def create(self) -> None:
        root = self.root
        write_json(root / "model_index.json", {
            "_class_name": "MiniMaxH3ModularPipeline"
        })
        (root / "README.md").write_text("fixture\n", "utf-8")
        (root / "LICENSE").write_text("fixture license\n", "utf-8")

        write_json(root / "FL2VA/model_index.json", {
            "_class_name": "MiniMaxH3Pipeline",
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            },
        })
        write_json(root / "FL2VA/transformer/config.json", {
            "_class_name": "MiniMaxH3DiTModel",
            "hidden_size": 5376,
            "num_layers": 50,
            "token_refiner_num_layers": 2,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "patch_size": [1, 2, 2],
            "text_dim": 5120,
        })
        write_json(root / "FL2VA/text_encoder/config.json", {
            "text_config": {
                "model_type": "qwen3_vl_text",
                "dtype": "bfloat16",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
            }
        })
        write_json(root / "FL2VA/tokenizer/tokenizer_config.json", {
            "eos_token": control_tokens.kImEnd,
            "pad_token": control_tokens.kEndOfText,
            "additional_special_tokens": list(control_tokens.H3_SPECIAL_TOKENS),
        })
        write_json(root / "FL2VA/audio_vae/metadata.json", {
            "metadata": {
                "kwargs": {"sample_rate": 32000, "vae_latent_channels": 32}
            }
        })

        for relative, value in {
            "FL2VA/processor/chat_template.json": {},
            "FL2VA/processor/preprocessor_config.json": {},
            "FL2VA/processor/tokenizer_config.json": {},
            "FL2VA/processor/tokenizer.json": {},
            "FL2VA/processor/video_preprocessor_config.json": {},
            "FL2VA/processor/vocab.json": {},
            "FL2VA/tokenizer/tokenizer.json": {},
            "FL2VA/tokenizer/vocab.json": {},
            "FL2VA/video_vae/config.json": {},
            "FL2VA/video_vae/source/config.json": {},
            "FL2VA/audio_vae/config.json": {},
        }.items():
            write_json(root / relative, value)
        (root / "FL2VA/tokenizer/merges.txt").write_text("fixture\n", "utf-8")

        text_shard = "model-00001-of-00001.safetensors"
        transformer_shard = "model-00001-of-00001.safetensors"
        write_safetensors(
            root / "FL2VA/text_encoder" / text_shard,
            {"model.embed_tokens.weight": ("BF16", [2, 2])},
        )
        write_json(
            root / "FL2VA/text_encoder/model.safetensors.index.json",
            {"weight_map": {"model.embed_tokens.weight": text_shard}},
        )
        write_safetensors(
            root / "FL2VA/transformer" / transformer_shard,
            {
                "blocks.0.attn.to_q.weight": ("BF16", [2, 2]),
                "blocks.0.norm.weight": ("F32", [2]),
            },
        )
        write_json(
            root / "FL2VA/transformer/model.safetensors.index.json",
            {
                "weight_map": {
                    "blocks.0.attn.to_q.weight": transformer_shard,
                    "blocks.0.norm.weight": transformer_shard,
                }
            },
        )
        write_safetensors(
            root / "FL2VA/video_vae/source/model.safetensors",
            {"decoder.weight": ("F32", [2, 2])},
        )
        write_safetensors(
            root / "FL2VA/audio_vae/model.safetensors",
            {"decoder.weight": ("F32", [2, 2])},
        )
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        metadata_root = self.root / ".cache/huggingface/download"
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or ".cache" in path.relative_to(self.root).parts:
                continue
            relative = path.relative_to(self.root).as_posix()
            metadata = metadata_root / f"{relative}.metadata"
            metadata.parent.mkdir(parents=True, exist_ok=True)
            blob = hashlib.sha256(relative.encode("utf-8")).hexdigest()
            metadata.write_text(
                f"{h3_manifest.REVISION}\n{blob}\n0\n", "utf-8"
            )


class H3ManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "MiniMax-H3"
        self.fixture = H3ManifestFixture(self.root)
        self.license_hash = hashlib.sha256(
            (self.root / "LICENSE").read_bytes()
        ).hexdigest()
        self.license_patch = mock.patch.object(
            h3_manifest, "LICENSE_SHA256", self.license_hash
        )
        self.license_patch.start()

    def tearDown(self):
        self.license_patch.stop()
        self.temp.cleanup()

    def test_manifest_is_deterministic_and_declares_t2va_only(self):
        first = h3_manifest.build_manifest(self.root)
        second = h3_manifest.build_manifest(self.root)
        self.assertEqual(
            h3_manifest._canonical_bytes(first),
            h3_manifest._canonical_bytes(second),
        )
        self.assertEqual(first["model_kind"], "minimax-h3-fl2va-bf16")
        self.assertEqual(
            first["contract"]["runtime_capabilities"],
            ["text-to-audio-video"],
        )
        self.assertEqual(first["tensor_count"], 5)
        self.assertEqual(first["file_count"], len(first["files"]))

        manifest_path = Path(self.temp.name) / "manifest.json"
        h3_manifest.write_manifest(manifest_path, first)
        verified = h3_manifest.verify_manifest(self.root, manifest_path)
        self.assertEqual(verified, first)

    def test_wrong_revision_fails_before_inventory(self):
        metadata = (
            self.root
            / ".cache/huggingface/download/FL2VA/model_index.json.metadata"
        )
        lines = metadata.read_text("utf-8").splitlines()
        metadata.write_text(f"wrong-revision\n{lines[1]}\n0\n", "utf-8")
        with self.assertRaisesRegex(h3_manifest.H3ManifestError, "revision"):
            h3_manifest.build_manifest(self.root)

    def test_missing_shard_fails(self):
        (
            self.root
            / "FL2VA/text_encoder/model-00001-of-00001.safetensors"
        ).unlink()
        with self.assertRaisesRegex(h3_manifest.H3ManifestError, "shard missing"):
            h3_manifest.build_manifest(self.root)

    def test_unreferenced_tensor_fails(self):
        shard = (
            self.root
            / "FL2VA/text_encoder/model-00001-of-00001.safetensors"
        )
        write_safetensors(
            shard,
            {
                "model.embed_tokens.weight": ("BF16", [2, 2]),
                "unexpected.weight": ("BF16", [2, 2]),
            },
        )
        self.fixture.refresh_metadata()
        with self.assertRaisesRegex(
            h3_manifest.H3ManifestError, "absent from index"
        ):
            h3_manifest.build_manifest(self.root)

    def test_duplicate_index_key_fails(self):
        index = self.root / "FL2VA/text_encoder/model.safetensors.index.json"
        index.write_text(
            '{"weight_map":{"duplicate":"a.safetensors",'
            '"duplicate":"b.safetensors"}}',
            "utf-8",
        )
        self.fixture.refresh_metadata()
        with self.assertRaisesRegex(h3_manifest.H3ManifestError, "duplicate"):
            h3_manifest.build_manifest(self.root)

    def test_unsupported_dtype_fails(self):
        shard = (
            self.root
            / "FL2VA/text_encoder/model-00001-of-00001.safetensors"
        )
        write_safetensors(
            shard, {"model.embed_tokens.weight": ("F32", [2, 2])}
        )
        self.fixture.refresh_metadata()
        with self.assertRaisesRegex(
            h3_manifest.H3ManifestError, "unsupported dtype"
        ):
            h3_manifest.build_manifest(self.root)

    def test_ref2va_fails(self):
        (self.root / "Ref2VA").mkdir()
        with self.assertRaisesRegex(h3_manifest.H3ManifestError, "Ref2VA"):
            h3_manifest.build_manifest(self.root)

    def test_overlapping_payload_fails(self):
        path = Path(self.temp.name) / "overlap.safetensors"
        header = {
            "a": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4],
            },
            "b": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [2, 6],
            },
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(6))
        with self.assertRaisesRegex(safetensors.SafetensorsError, "overlapping"):
            safetensors.validate_file(path)


if __name__ == "__main__":
    unittest.main()
