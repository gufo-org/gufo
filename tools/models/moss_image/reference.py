#!/usr/bin/env python3
"""Compare native MOSS-Image stages with the checkpoint's PyTorch pipeline.

Offline model inference only. No remote inference/API calls. See
src/models/moss_image/UPSTREAM.md for the pinned checkpoint and libraries.

Run the probe first; this script replays its initial noise (noise.f32) through
the pinned pipeline components on the CPU and reports cosine similarity and
relative L2 per stage:

  moss_image_probe MODEL_DIR TEXT_ENCODER.gguf NATIVE_DIR PROMPT 256 4
  python3 reference.py --model MODEL_DIR --native NATIVE_DIR --output OUT_DIR \
      --prompt PROMPT --size 256 --steps 4

The checkpoint's pipeline_moss_image.py is imported from MODEL_DIR; review it
at the pinned revision before running.
"""

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="A red fox sitting in fresh snow, golden hour light")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    sys.path.insert(0, str(args.model))
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer
    from pipeline_moss_image import (MOSSImagePipeline, MOSSImageScheduler,
                                     MOSSImageTransformer)

    start = time.time()
    transformer = MOSSImageTransformer.from_pretrained(
        args.model / "transformer", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model / "tokenizer")
    text_encoder = AutoModel.from_pretrained(
        args.model / "text_encoder", dtype=torch.bfloat16).eval()
    vae = AutoencoderKLFlux2.from_pretrained(
        args.model / "vae", torch_dtype=torch.float32).eval()
    pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), text_encoder,
                             tokenizer, vae)
    side = args.size // 16
    records = {"reference": {"torch": torch.__version__,
                             "transformers": version("transformers"),
                             "diffusers": version("diffusers"),
                             "model": args.model.name,
                             "prompt": args.prompt, "size": args.size,
                             "steps": args.steps},
               "stages": {}}

    def native(name, cols):
        path = args.native / f"{name}.f32"
        return np.fromfile(path, dtype=np.float32).reshape(-1, cols) if path.exists() else None

    def rows(tensor):
        # [1, C, H, W] -> [H*W, C]; [L, C] stays.
        value = tensor.detach().float().cpu().numpy()
        return value.reshape(value.shape[1], -1).T if value.ndim == 4 else value

    def compare(name, reference, cols):
        ours = native(name, cols)
        if ours is None:
            return
        a, b = ours.astype(np.float64).ravel(), rows(reference).astype(np.float64).ravel()
        if a.shape != b.shape:
            raise SystemExit(f"{name}: native {ours.shape} vs reference {rows(reference).shape}")
        record = {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
                  "relative_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)),
                  "max_abs": float(np.abs(a - b).max())}
        records["stages"][name] = record
        print(f"{name:24s} cos {record['cosine']:.6f}  rel_l2 {record['relative_l2']:.4f}  "
              f"max_abs {record['max_abs']:.4f}", flush=True)

    with torch.no_grad():
        positive, positive_mask = pipe.encode_prompt(args.prompt)
        negative, negative_mask = pipe.encode_prompt("")
        compare("text.positive", positive[0], 4096)
        compare("text.negative", negative[0], 4096)
        noise = native("noise", 128)
        if noise is None or noise.shape[0] != side * side:
            raise SystemExit("run the probe at the same size first (noise.f32)")
        latents = torch.from_numpy(noise.T.reshape(1, 128, side, side).copy())
        scheduler = MOSSImageScheduler()
        scheduler.set_timesteps(args.steps)
        compare("sigmas", scheduler.sigmas.float().reshape(1, -1), len(scheduler.sigmas))
        for step, timestep in enumerate(scheduler.timesteps):
            cond = transformer(latents, timestep, positive, positive_mask).sample.float()
            uncond = transformer(latents, timestep, negative, negative_mask,
                                 skip_layers=[22]).sample.float()
            compare(f"dit.{step}.cond.output", cond, 128)
            compare(f"dit.{step}.uncond.output", uncond, 128)
            latents = scheduler.step(uncond + 3.0 * (cond - uncond), timestep, latents,
                                     return_dict=False)[0]
            compare(f"latent.{step}", latents, 128)
        z = latents.to(vae.dtype)
        bn = vae.bn
        z = (z * (bn.running_var + vae.config.batch_norm_eps).sqrt().view(1, -1, 1, 1)
             + bn.running_mean.view(1, -1, 1, 1))
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(
            b, c // 4, 2 * h, 2 * w)
        decoded = vae.decode(z, return_dict=False)[0]
        compare("vae.decoded", decoded, 3)
        pixels = ((decoded.clamp(-1, 1)[0].permute(1, 2, 0).numpy() + 1) * 127.5).round()
        from PIL import Image
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(args.output / "reference.png")
    records["reference"]["seconds"] = time.time() - start
    (args.output / "comparison.json").write_text(json.dumps(records, indent=1) + "\n")


if __name__ == "__main__":
    main()
