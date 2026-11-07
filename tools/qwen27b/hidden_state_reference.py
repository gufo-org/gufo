#!/usr/bin/env python3
"""Write an independent reference for TextModelRunner::EncodeHiddenStates.

Runs an unquantized Hugging Face Qwen3.5/3.8 checkpoint with transformers on
fixed, MOSS-Image-framed prompts and records per token:

- the last decoder layer's output (the residual stream before the final norm),
- the same rows after the checkpoint's final norm, and
- the final norm's effective per-channel scale and epsilon.

tests/models/qwen/hip/text_hidden_states_test.cpp (GUFO_QWEN_HIDDEN_REFERENCE)
compares Gufo's Qwen adapter with the first and, after applying the norm once
itself, with the second. Token IDs are stored, so both sides encode identical
inputs. No download is implicit: pass local directories. The default float32
CPU run needs about four bytes of RAM per parameter; --device cuda (ROCm or
CUDA) loads the weights straight onto the GPU, normally with --dtype bfloat16,
without staging them in host RAM. Recorded rows are float32 either way.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MAGIC = b"GQHS"
VERSION = 1
# MOSS-Image framing; enable_thinking=True leaves the think block open.
FRAME = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
PROMPTS = ("A red fox reading a newspaper on a park bench, watercolor.", "")


def text_backbone(model):
    """The decoder owning embed_tokens, layers and the final norm."""
    for name, module in model.named_modules():
        if all(hasattr(module, attr) for attr in ("embed_tokens", "layers", "norm")):
            return name, module
    raise SystemExit("no decoder with embed_tokens, layers and norm found")


def norm_epsilon(norm) -> float:
    for attr in ("variance_epsilon", "eps"):
        value = getattr(norm, attr, None)
        if value is not None:
            return float(value)
    raise SystemExit("cannot read the final norm epsilon")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path,
                        help="Hugging Face model directory (for MOSS-Image: text_encoder)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path,
                        help="tokenizer directory (default: the checkpoint)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu",
                        help="torch device, e.g. cuda (also ROCm) to load the weights "
                             "directly onto the GPU (default: cpu); needs accelerate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    # A device map streams each tensor straight to its device instead of
    # materializing the whole model in host RAM first.
    placement = ({} if torch.device(args.device).type == "cpu" else
                 {"low_cpu_mem_usage": True, "device_map": {"": args.device}})
    model = AutoModel.from_pretrained(args.checkpoint, torch_dtype=getattr(torch, args.dtype),
                                      **placement)
    model.eval()
    name, backbone = text_backbone(model)
    norm = backbone.norm
    epsilon = norm_epsilon(norm)
    width = backbone.embed_tokens.embedding_dim
    # For x = 1 an RMS norm returns scale / sqrt(1 + eps), whatever its weight
    # parameterization (plain or zero-centered).
    ones = torch.ones(1, 1, width, dtype=norm.weight.dtype, device=norm.weight.device)
    scale = (norm(ones).float() * math.sqrt(1.0 + epsilon)).reshape(width).cpu()

    captured = []
    hook = backbone.layers[-1].register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            output[0] if isinstance(output, tuple) else output))
    cases = []
    for prompt in PROMPTS:
        ids = tokenizer(FRAME.format(prompt), add_special_tokens=False,
                        return_tensors="pt").input_ids
        captured.clear()
        backbone(input_ids=ids.to(backbone.embed_tokens.weight.device), use_cache=False)
        if len(captured) != 1:
            raise SystemExit(f"expected one last-layer output, got {len(captured)}")
        pre = captured[0]
        cases.append((ids.reshape(-1).numpy().astype("<u4"),
                      pre.float().reshape(-1, width).cpu().numpy().astype("<f4"),
                      norm(pre).float().reshape(-1, width).cpu().numpy().astype("<f4")))
    hook.remove()

    with args.output.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<IIIf", VERSION, len(cases), width, epsilon))
        out.write(scale.numpy().astype("<f4").tobytes())
        for ids, pre, normalized in cases:
            out.write(struct.pack("<I", ids.size))
            out.write(ids.tobytes())
            out.write(np.ascontiguousarray(pre).tobytes())
            out.write(np.ascontiguousarray(normalized).tobytes())
    print(f"{args.output}: decoder={name or '<root>'} width={width} "
          f"epsilon={epsilon} cases={[int(ids.size) for ids, _, _ in cases]}")


if __name__ == "__main__":
    main()
