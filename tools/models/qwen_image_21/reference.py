#!/usr/bin/env python3
"""Compare native boundaries with the pinned official Diffusers pipeline.

Offline model inference only. No remote inference/API calls. See the model's
UPSTREAM.md for the source and model revisions used for qualification.
"""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as pipeline_module
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
    parser.add_argument("--skip-decode", action="store_true",
                        help="Compare denoising latents without running the VAE decoder")
    parser.add_argument("--negative-prompt",
                        help="True CFG: Diffusers negative_prompt (needs --guidance > 1)")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="Diffusers true_cfg_scale")
    parser.add_argument("--init-image", type=Path,
                        help="Image to image start image (already at the output size)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="ComfyUI KSampler denoise: last steps+1 sigmas of int(steps/strength)")
    args = parser.parse_args()
    cfg = args.guidance > 1 and args.negative_prompt is not None
    img2img = args.init_image is not None and args.strength <= 0.9999
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    records = {"reference": {"torch": torch.__version__,
                             "transformers": version("transformers"),
                             "diffusers": version("diffusers"),
                             "model_revision": Path(args.model).name,
                             "teacher_forced": args.teacher_force,
                             "decoded": not args.skip_decode,
                             "guidance": args.guidance if cfg else 1.0,
                             "negative_prompt": args.negative_prompt if cfg else None,
                             "strength": args.strength if img2img else 1.0}}
    step = [0]
    # Boundary names of the negative (unconditional) pass carry this prefix,
    # as in Gufo's observer.
    branch = [""]

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
            key = branch[0] + name()
            value = inputs[0] if inputs else kwargs["hidden_states"]
            if compare_input:
                compare(key, value, channel_first, prefixed=False)
            value = replace(key, value, channel_first)
            if inputs:
                return (value, *inputs[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = value
            return inputs, kwargs
        return hook

    def compare(name, value, channel_first=False, prefixed=True):
        if prefixed and not name.startswith(("latent.", "sigmas", "init.", "noise")):
            name = branch[0] + name
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
    encode_prompt = pipe.encode_prompt
    encodes = [0]

    def encode_prompt_branch(*inputs, **kwargs):
        branch[0] = "negative." if encodes[0] else ""
        encodes[0] += 1
        try:
            return encode_prompt(*inputs, **kwargs)
        finally:
            branch[0] = ""

    pipe.encode_prompt = encode_prompt_branch
    transformer_calls = [0]

    def transformer_branch(module, inputs, kwargs):
        branch[0] = "negative." if cfg and transformer_calls[0] % 2 else ""
        transformer_calls[0] += 1

    pipe.transformer.register_forward_pre_hook(transformer_branch, with_kwargs=True)
    scheduler_step = pipe.scheduler.step

    def guided_step(model_output, *inputs, **kwargs):
        if cfg:
            compare(f"dit.{step[0]}.guided", model_output, prefixed=False)
        return scheduler_step(model_output, *inputs, **kwargs)

    pipe.scheduler.step = guided_step
    encode_vae_image = pipe._encode_vae_image

    def compare_encoded_image(*inputs, **kwargs):
        value = encode_vae_image(*inputs, **kwargs)
        compare("vae.encoded", value, True)
        return value

    pipe._encode_vae_image = compare_encoded_image
    for i, layer in enumerate(pipe.transformer.transformer_blocks):
        def dit_input(module, inputs, kwargs, i=i):
            kwargs = dict(kwargs)
            name = f"dit.{step[0]}.input" if i == 0 else f"dit.{step[0]}.block.{i - 1}"
            if i == 0:
                compare(name, kwargs["hidden_states"])
                compare(f"dit.{step[0]}.modulation", kwargs["modulation"])
            kwargs["hidden_states"] = replace(branch[0] + name, kwargs["hidden_states"])
            kwargs["modulation"] = replace(f"{branch[0]}dit.{step[0]}.modulation", kwargs["modulation"])
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
    records["reference"]["initial_noise_sha256"] = hashlib.sha256(noise.tobytes()).hexdigest()
    latent = torch.from_numpy(noise).to(device="cuda", dtype=torch.bfloat16)
    kwargs = {}
    if args.image:
        kwargs["image"] = Image.open(args.image)
    if cfg:
        kwargs["negative_prompt"] = args.negative_prompt
        kwargs["true_cfg_scale"] = args.guidance
    schedule_steps = args.steps
    if img2img:
        # ComfyUI KSampler denoise: schedule int(steps / strength) sigmas and
        # run the last `steps`; start at sigma * noise + (1 - sigma) * image.
        schedule_steps = int(args.steps / args.strength)
        retrieve_timesteps = pipeline_module.retrieve_timesteps

        def truncated_timesteps(scheduler, *inputs, **kw):
            timesteps, count = retrieve_timesteps(scheduler, *inputs, **kw)
            skip = count - args.steps
            scheduler.timesteps = scheduler.timesteps[skip:]
            scheduler.sigmas = scheduler.sigmas[skip:]
            return scheduler.timesteps, args.steps

        pipeline_module.retrieve_timesteps = truncated_timesteps
        probe = copy.deepcopy(pipe.scheduler)
        mu = pipeline_module.calculate_shift(
            target_rows, probe.config.get("base_image_seq_len", 256),
            probe.config.get("max_image_seq_len", 4096),
            probe.config.get("base_shift", 0.5), probe.config.get("max_shift", 1.15))
        truncated_timesteps(probe, schedule_steps, "cuda",
                            sigmas=np.linspace(1.0, 1 / schedule_steps, schedule_steps), mu=mu)
        sigma = probe.sigmas[0].float()
        with Image.open(args.init_image) as source:
            start = source.convert("RGBA")
        if start.size != (args.size, args.size):
            start = start.resize((args.size, args.size), Image.LANCZOS)
        pixels = pipe.image_processor.preprocess(start, width=args.size, height=args.size)
        with torch.inference_mode():
            encoded = encode_vae_image(pixels.unsqueeze(2).to("cuda", torch.bfloat16), None)
        side = args.size // 16
        packed = pipe._pack_latents(encoded, 1, encoded.shape[1], side, side)
        compare("init.vae.encoded", packed)
        latent = (sigma * latent.float() + (1 - sigma) * packed.float()).to(torch.bfloat16)
        compare("init.latent", latent)
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt, width=args.size, height=args.size,
            num_inference_steps=schedule_steps, latents=latent,
            output_type="latent" if args.skip_decode else "pil",
            callback_on_step_end=callback, **kwargs
        )
    if not args.skip_decode:
        result.images[0].save(args.output / "image.png")
        native_image = args.native / "image.png"
        if native_image.exists():
            a = np.asarray(result.images[0].convert("RGBA"), dtype=np.float64)
            with Image.open(native_image) as image:
                b = np.asarray(image.convert("RGBA"), dtype=np.float64)
            if a.shape != b.shape:
                records["png"] = {"shape_mismatch": [list(b.shape), list(a.shape)]}
            else:
                mse = float(np.mean((a - b) ** 2))
                records["png"] = {"exact": mse == 0,
                                  "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None}
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
        matched = (".block." in name or name == "vae.encoded"
                   or (name.startswith("dit.") and name.endswith((".input", ".modulation"))))
        if args.teacher_force and matched and record.get("relative_l2", 0) > 0.01:
            failures.append(f"{name}: matched-block relative L2 exceeds 0.01")
        # One guided denoiser step and the start latent must agree closely.
        if (name == "dit.0.guided" or name.startswith("init.")) and record.get("cosine", 1) < 0.999:
            failures.append(f"{name}: cosine below 0.999")
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
