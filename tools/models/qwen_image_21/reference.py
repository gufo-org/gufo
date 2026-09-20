#!/usr/bin/env python3
"""Compare native boundaries with the pinned official Diffusers pipeline.

Offline model inference only. No remote inference/API calls. See the model's
UPSTREAM.md for the source and model revisions used for qualification.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="A red cube on a white table.")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--teacher-force", action="store_true",
                        help="Replay each official block with Gufo's saved block inputs")
    args = parser.parse_args()
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    records = {"reference": {"torch": torch.__version__,
                             "transformers": version("transformers"),
                             "diffusers": version("diffusers"),
                             "model_revision": Path(args.model).name,
                             "teacher_forced": args.teacher_force}}
    step = [0]

    def replace(name, tensor, channel_first=False):
        path = args.native / f"{name}.f32"
        if not args.teacher_force or not path.exists():
            return tensor
        array = np.fromfile(path, dtype=np.float32)
        value = torch.from_numpy(array.copy())
        if channel_first:
            height, width = tensor.shape[-2:]
            value = value.reshape(tensor.shape[0], height, width, tensor.shape[1])
            value = value.permute(0, 3, 1, 2)
            if tensor.ndim == 5:
                value = value.unsqueeze(2)
        else:
            value = value.reshape(tensor.shape)
        return value.to(tensor)

    def input_hook(name, channel_first=False, compare_input=False):
        def hook(module, inputs, kwargs):
            key = name()
            value = inputs[0] if inputs else kwargs["hidden_states"]
            if compare_input:
                compare(key, value, channel_first)
            value = replace(key, value, channel_first)
            if inputs:
                return (value, *inputs[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = value
            return inputs, kwargs
        return hook

    def compare(name, value, channel_first=False):
        source = args.native / f"{name}.f32"
        if not source.exists():
            return
        if isinstance(value, (tuple, list)):
            value = value[0]
        if channel_first:
            if value.ndim == 5:
                value = value[:, :, 0]
            value = value.permute(0, 2, 3, 1)
        actual = value.detach().float().cpu().numpy().reshape(-1)
        native = np.fromfile(source, dtype=np.float32)
        actual.tofile(args.output / f"{name}.f32")
        if actual.size != native.size:
            records[name] = {"shape_mismatch": [native.size, actual.size]}
        else:
            a, b = actual.astype(np.float64), native.astype(np.float64)
            diff = a - b
            records[name] = {
                "elements": a.size,
                "max_abs": float(np.max(np.abs(diff))),
                "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-30)),
                "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
                "exact_fraction": float(np.mean(a == b)),
            }
        print(name, json.dumps(records[name]), flush=True)
        (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")

    lm = pipe.text_encoder.model.language_model
    # The next layer's input includes vision replacement / DeepStack additions;
    # the embed_tokens and decoder-layer output hooks do not.
    for i, layer in enumerate(lm.layers):
        layer.register_forward_pre_hook(input_hook(
            lambda i=i: "text.embedding" if i == 0 else f"text.block.{i - 1}",
            compare_input=True,
        ), with_kwargs=True)
    lm.layers[-1].register_forward_hook(
        lambda module, inputs, out: compare("text.block.35", out))
    visual = pipe.text_encoder.model.visual
    for i, block in enumerate(visual.blocks):
        block.register_forward_pre_hook(input_hook(
            lambda i=i: "vision.embedding" if i == 0 else f"vision.block.{i - 1}",
            compare_input=(i == 0),
        ), with_kwargs=True)
        block.register_forward_hook(lambda module, inputs, out, i=i: compare(f"vision.block.{i}", out))
    visual.merger.register_forward_hook(lambda module, inputs, out: compare("vision.output", out))
    for i, layer in enumerate(pipe.transformer.transformer_blocks):
        def dit_input(module, inputs, kwargs, i=i):
            kwargs = dict(kwargs)
            name = f"dit.{step[0]}.input" if i == 0 else f"dit.{step[0]}.block.{i - 1}"
            kwargs["hidden_states"] = replace(name, kwargs["hidden_states"])
            kwargs["modulation"] = replace(f"dit.{step[0]}.modulation", kwargs["modulation"])
            return inputs, kwargs
        layer.register_forward_pre_hook(dit_input, with_kwargs=True)
        layer.register_forward_hook(
            lambda module, inputs, out, i=i: compare(f"dit.{step[0]}.block.{i}", out)
        )
    target_rows = (args.size // 16) ** 2
    pipe.transformer.proj_out.register_forward_hook(
        lambda module, inputs, out: compare(f"dit.{step[0]}.output", out[:, -target_rows:])
    )
    for i, block in enumerate(pipe.vae.decoder.up_blocks):
        block.register_forward_pre_hook(input_hook(
            lambda i=i: "vae.decoder.mid" if i == 0 else f"vae.decoder.block.{i - 1}",
            channel_first=True,
        ), with_kwargs=True)
        block.register_forward_hook(
            lambda module, inputs, out, i=i: compare(f"vae.decoder.block.{i}", out, True)
        )
    for i, block in enumerate(pipe.vae.encoder.down_blocks):
        block.register_forward_pre_hook(input_hook(
            lambda i=i: "vae.encoder.input" if i == 0 else f"vae.encoder.block.{i - 1}",
            channel_first=True, compare_input=(i == 0),
        ), with_kwargs=True)
        block.register_forward_hook(
            lambda module, inputs, out, i=i: compare(f"vae.encoder.block.{i}", out, True)
        )
    pipe.vae.decoder.conv_out.register_forward_hook(
        lambda module, inputs, out: compare("vae.decoded", out, True)
    )

    def callback(pipeline, index, timestep, values):
        compare(f"latent.{index}", values["latents"])
        step[0] = index + 1
        return values

    noise = np.fromfile(args.native / "noise.f32", dtype=np.float32).reshape(1, target_rows, 64)
    latent = torch.from_numpy(noise).to(device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if args.image:
        kwargs["image"] = Image.open(args.image)
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt, width=args.size, height=args.size,
            num_inference_steps=args.steps, latents=latent,
            callback_on_step_end=callback, **kwargs
        )
    result.images[0].save(args.output / "image.png")
    compare("sigmas", pipe.scheduler.sigmas)
    tokenizer_ids = pipe.processor.tokenizer.encode(args.prompt)
    native_ids = [int(x) for x in (args.native / "tokens.txt").read_text().split()]
    records["tokenizer"] = {"exact": tokenizer_ids == native_ids}
    records["system_drop"] = pipe._drop_idx
    failures = []
    for name, record in records.items():
        if not isinstance(record, dict):
            continue
        if "shape_mismatch" in record:
            failures.append(f"{name}: shape mismatch")
        if "relative_l2" in record and not np.isfinite(record["relative_l2"]):
            failures.append(f"{name}: nonfinite result")
        # A matched BF16 block must stay within a 1% relative-L2 envelope.
        # Full trajectories are reported separately, without pretending that
        # this local rounding tolerance proves general perceptual quality.
        if args.teacher_force and ".block." in name and record.get("relative_l2", 0) > 0.01:
            failures.append(f"{name}: matched-block relative L2 exceeds 0.01")
    if tokenizer_ids != native_ids:
        failures.append("tokenizer mismatch")
    if records.get("sigmas", {}).get("max_abs", 1) != 0:
        failures.append("flow schedule mismatch")
    records["validation"] = {"passed": not failures, "failures": failures}
    (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
