# MOSS-Image reference contract

Native implementation of the MOSS-Image-SFT text-to-image graph. Python is used
only for offline evaluation; production loads the transformer and VAE
safetensors directly and the text encoder from a GGUF conversion.

| Source | Pinned revision |
| --- | --- |
| [MOSS-Image-SFT checkpoint](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/tree/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9) | `f972bd4b4b9e3852133cc5da954d87bc32cd8fb9`, including `pipeline_moss_image.py` |
| [Diffusers](https://github.com/huggingface/diffusers/tree/v0.42.0) | `v0.42.0`; `AutoencoderKLFlux2` |
| [Transformers](https://github.com/huggingface/transformers/tree/v5.20.2) | `v5.20.2`; `Qwen3_5Model` text encoder |

The checkpoint ships its own pipeline (`pipeline_moss_image.py`, loaded by
Diffusers as a custom pipeline):

- `MOSSImagePipeline.encode_prompt`: the prompt as one user turn of the Qwen
  chat template with `add_generation_prompt=True` and `enable_thinking=True`
  (the `<think>` block is left open); `last_hidden_state` **after** the final
  RMSNorm; an empty prompt for the negative branch.
- `MOSSImageTransformer`: 42 single-stream blocks over the joint image+text
  sequence (image rows first), dense non-causal attention, per-head RMSNorm on
  Q/K, four-axis rotate-half RoPE (r, f, h, w) with 16 bands per axis, a shared
  six-part block modulation and the "situ" gate
  `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)`. Text tokens pass through a
  two-block `_TextProjection` first.
- Classifier-free guidance 3.0; the negative branch skips block 22
  (`uncond_skip_layers=[22]`). Default 50 steps.
- `MOSSImageScheduler`: shift 5 over the 1000-step grid, sigmas snapped to the
  training grid with `unique_consecutive`, Euler steps, terminal zero.
- `AutoencoderKLFlux2` decoder: latent BatchNorm denormalization
  (`running_var + 1e-4`), 2x2 unpatching to 32 channels, FP32 VAE.

Gufo follows these formulas with the BF16 operation boundaries of the PyTorch
reference (`-ffp-contract=off`, each BF16 operation rounded). The transformer
and text projection run on the Qwen-Image BF16 operators; dense attention uses
the MiniMax H3 128-dimension WMMA kernel with FP32 softmax and accumulation.
The VAE decoder runs in BF16 with FP32/FP64 GroupNorm statistics; the reference
decodes in FP32.

The text encoder (`Qwen3_5Model`, 9B, hybrid Gated DeltaNet/attention) runs on
Gufo's Qwen engine from a GGUF conversion of `text_encoder/` with prompt
hidden-state capture; only its final RMSNorm is read from the safetensors.

Gufo's RNG differs from PyTorch's. Compare shared initial noise, not numeric
seeds: the probe writes `noise.f32`, and `--noise` replays a reference draw.
Different BF16 GEMM/attention reductions need not yield identical images;
compare stage outputs as well as the complete trajectory.

Diffusers, Transformers and the MOSS-Image pipeline code and weights are
Apache-2.0 licensed. The weights are downloaded separately and are not part of
Gufo's MIT-licensed source.
