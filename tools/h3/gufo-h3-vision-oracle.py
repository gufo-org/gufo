#!/usr/bin/env python3
"""Capture the MiniMax H3 Qwen3-VL vision tower oracle for FL2VA conditioning.

The engine port (``src/models/minimax_h3/vision_tower.hip``) must reproduce
``transformers`` ``Qwen3VLVisionModel`` (the stack shipped inside
``FL2VA/text_encoder`` as ``model.visual.*``) within the shared relative
gates. The input contract is the shipped ``FL2VA/processor`` image
preprocessor; the emitted pixel_values additionally validate the engine's
merge-block patch ordering, row layout ``((c*2+t)*16+y)*16+x``, and
``(x/255 - 0.5) / 0.5`` normalization because the test feeds the engine the
raw canvas pixels, not this file.

Fixture layout (schema ``gufo.minimax-h3-vision-tower-golden.v1``):
  video-vision-pixels.f32      processor pixel_values flattened (f32)
  video-vision-output.f32      merger output (f32)
  video-vision-deep-0.f32      deepstack tap 1
  video-vision-deep-1.f32      deepstack tap 2
  video-vision-deep-2.f32      deepstack tap 3

``--output`` is the fixture directory; the manifest is written there as
``vision-tower-oracle-v1.json`` and records ``--image`` relative to it. The
input keyframe is the VAE-encoder oracle's canvas, shared rather than copied.

Usage (oracle venv, CPU):

  python tools/h3/gufo-h3-vision-oracle.py \
      --model-root ~/models/MiniMax-H3 \
      --image tests/fixtures/minimax_h3/video-vae-encoder-keyframe-512.png \
      --output tests/fixtures/minimax_h3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy
import torch
from PIL import Image

SCHEMA = "gufo.minimax-h3-vision-tower-golden.v1"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True,
                        help="canvas-sized PNG keyframe to encode")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha(t: torch.Tensor) -> tuple[str, int]:
    contiguous = t.detach().cpu().to(torch.float32).contiguous()
    return sha256_bytes(contiguous.numpy().tobytes()), contiguous.numel() * 4


def write_f32(path: Path, tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().cpu().to(torch.float32).contiguous()
    payload = contiguous.numpy().tobytes()
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "dtype": "F32",
        "shape": list(contiguous.shape),
        "sha256": sha256_bytes(payload),
    }


def load_visual_weights(model_root: Path) -> dict[str, torch.Tensor]:
    component = model_root / "FL2VA" / "text_encoder"
    index = json.loads((component / "model.safetensors.index.json").read_text())
    wanted = {k: v for k, v in index["weight_map"].items()
              if k.startswith("model.visual.")}
    tensors: dict[str, torch.Tensor] = {}
    from safetensors.torch import load_file

    for shard in sorted(set(wanted.values())):
        blob = load_file(component / shard)
        for name, tensor in blob.items():
            if name in wanted:
                tensors[name.removeprefix("model.visual.")] = tensor
    missing = set(wanted) - {f"model.visual.{k}" for k in tensors}
    if missing:
        raise RuntimeError(f"visual tensors missing from shards: {sorted(missing)[:5]}")
    return tensors


def main() -> int:
    args = parse_args()
    torch.set_grad_enabled(False)
    model_root = args.model_root.expanduser().resolve()

    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers import AutoImageProcessor

    text_config = json.loads(
        (model_root / "FL2VA" / "text_encoder" / "config.json").read_text())
    vision_config = Qwen3VLVisionConfig(**text_config["vision_config"])

    processor = AutoImageProcessor.from_pretrained(
        model_root / "FL2VA" / "processor")
    image = Image.open(args.image).convert("RGB")
    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded["pixel_values"]
    grid_thw = encoded["image_grid_thw"]

    with torch.device("cpu"):
        vision = Qwen3VLVisionModel(vision_config).to(torch.bfloat16).eval()
    tensors = load_visual_weights(model_root)
    result = vision.load_state_dict(tensors, strict=False)
    unexpected = [k for k in result.unexpected_keys]
    if unexpected:
        raise RuntimeError(f"unexpected visual keys: {unexpected[:8]}")
    missing = [k for k in result.missing_keys
               if "weight" in k and "norm" not in k.split(".")[-2:]]
    # rotary inv_freq buffers are computed; tolerate only those absences.
    hard_missing = [k for k in result.missing_keys
                    if not k.endswith("rotary_pos_emb.inv_freq")
                    and "linear_fc" not in k]
    if hard_missing:
        raise RuntimeError(f"missing visual weights: {hard_missing[:8]}")

    outputs = vision(
        hidden_states=pixel_values.to(torch.bfloat16), grid_thw=grid_thw)

    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "video-vision-pixels.f32": write_f32(out_dir / "video-vision-pixels.f32",
                                             pixel_values),
        "video-vision-output.f32": write_f32(out_dir / "video-vision-output.f32",
                                             outputs.pooler_output),
        "video-vision-deep-0.f32": write_f32(
            out_dir / "video-vision-deep-0.f32", outputs.deepstack_features[0]),
        "video-vision-deep-1.f32": write_f32(
            out_dir / "video-vision-deep-1.f32", outputs.deepstack_features[1]),
        "video-vision-deep-2.f32": write_f32(
            out_dir / "video-vision-deep-2.f32", outputs.deepstack_features[2]),
    }
    manifest = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "reference": "transformers Qwen3VLVisionModel @ "
                     + __import__("transformers").__version__,
        "image": {"path": Path(os.path.relpath(args.image.resolve(),
                                               out_dir)).as_posix(),
                  "sha256": sha256_bytes(args.image.read_bytes()),
                  "width": image.width, "height": image.height},
        "grid_thw": [int(x) for x in grid_thw.flatten().tolist()],
        "geometry": {
            "rows": int(pixel_values.shape[0]),
            "columns": int(pixel_values.shape[1]),
            "merged_rows": int(outputs.pooler_output.shape[0]),
            "out_hidden_size": int(vision_config.out_hidden_size),
        },
        "arithmetic": {
            "model_dtype": "bfloat16",
            "pixel_values": "f32 upstream (uint8/255 - 0.5) / 0.5, merge-block "
                            "row order, layout ((c*2+t)*16+y)*16+x",
            "attention": "full block-diagonal (single image)",
            "deepstack": "mergers after blocks 8/16/24, postshuffle norm",
        },
        # Both sides are BF16. Against an FP32 run of this tower the oracle
        # itself reaches relative_max 2.46 on the merger (relative_l2 0.044),
        # so the max bound must leave room for two independent BF16 paths.
        "quality_bounds": {"relative_l2": 0.05, "relative_max": 2.0},
        "files": files,
    }
    (out_dir / "vision-tower-oracle-v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("wrote", out_dir / "vision-tower-oracle-v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
