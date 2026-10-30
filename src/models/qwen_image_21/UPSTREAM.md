# Qwen-Image-2.1 reference contract

Native implementation of the BF16 inference graph. Python is used only for
offline evaluation; production loads safetensors directly.

| Source | Pinned revision |
| --- | --- |
| [Qwen checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | `b3179ad355be050328e483a9dfdd9e60cd62adfa` |
| [Official model repository](https://github.com/QwenLM/Qwen-Image-2.1/tree/fb7ae1d1f9611cd91524d03c53c5246b36ac8577) | `fb7ae1d1f9611cd91524d03c53c5246b36ac8577` |
| [Diffusers](https://github.com/huggingface/diffusers/tree/80c7ed262aeffbeb43ef13ae04baeb9b84515a69) | `80c7ed262aeffbeb43ef13ae04baeb9b84515a69` |
| [Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | Transformers `v5.17.0`; FP32 vision-position interpolation and the pipeline's pre-final-norm hook |
| [llama-swap routes](https://github.com/mostlygeek/llama-swap/blob/96e6f94c018b51ae9cfc420b3166cfcace5bf97a/internal/server/server.go) | `96e6f94c018b51ae9cfc420b3166cfcace5bf97a` |
| [OpenAI image generation](https://github.com/openai/openai-python/blob/main/src/openai/types/image_generate_params.py) / [editing schema](https://github.com/openai/openai-python/blob/main/src/openai/types/image_edit_params.py) | Reviewed 2026-09-20; inference tests use local servers only |

Relevant files under the pinned Diffusers `src/diffusers/`:

- `pipelines/qwenimage21/pipeline_qwenimage21.py`: raw prompt templates,
  system-token trimming, reference resizing, alpha compositing, latent packing,
  default 40 steps and guidance scale 1.
- `models/transformers/transformer_qwenimage21.py`: single-stream DiT,
  block-causal masking, adjacent complex RoPE pairs, shared modulation,
  zero-centered text RMSNorm and per-layer condition K/V caches.
- `models/autoencoders/autoencoder_kl_qwenimage21.py`: spatial image VAE,
  channel RMSNorm, residual down/up shuffles, deterministic posterior mean.
- `schedulers/scheduling_flow_match_euler_discrete.py`: dynamic exponential
  shift, terminal sigma 0.02, terminal zero, BF16 velocity-product rounding.

The text encoder output is **before final RMSNorm**. Images use the trained
Qwen3-VL spatial order, interleaved mRoPE and all three DeepStack taps. Condition
images retain RGBA for the VAE and are composited over white for Qwen3-VL.
Condition tokens use timestep zero; only their K/V is retained after the first
denoising step. Every request starts with separate caches and RNG.
The raw system prompt is `Comprehend and analyze the provided prompt.`; its
14-token prefix is removed from the encoder output, as in the pinned pipeline.
The repository README recommends 2048² output, while this Diffusers revision
defaults `output_resolution` to 1024. Gufo follows the pipeline default;
explicit `size` selects larger outputs without changing reference-image resizing.
Convolution bias follows the pinned PyTorch/MIOpen BF16 path: round the product
before adding bias. Linear projections retain their single final rounding.

The reference runtime must record its actual Torch/Transformers versions.
Different BF16 GEMM/attention reductions need not yield identical full images;
compare matched block inputs as well as the complete trajectory. Do not change
weights, remove reference images, reduce precision or skip diffusion steps to
pass a speed gate.

Guidance and image to image (Gufo extensions, `gufo.*` request fields):
`guidance` above 1 is the pipeline's `true_cfg_scale`: each step also runs the
negative prompt (same reference images) and
combines `negative + guidance * (positive - negative)` with BF16 rounding after
every tensor operation. Unlike the pipeline, which ignores the scale without a
negative prompt, an absent negative prompt encodes like an empty prompt, as a
ComfyUI KSampler with an empty negative conditioning would. `strength` below 1
follows ComfyUI KSampler `denoise` (the pinned Diffusers revision has no
Qwen-Image-2.1 image-to-image pipeline): `steps` steps over the last
`steps + 1` sigmas of an `int(steps / strength)` schedule, starting from
`sigma * noise + (1 - sigma) * latent` with the posterior-mean VAE latent of the
start image resized to the output size. Defaults (1 and 1) keep the unguided
full-schedule path unchanged. The negative prompt reuses the positive prompt's
condition K/V for its leading reference images: block-causal rows never see
later tokens, so rows before the first differing prompt token are identical;
only its own text rows get a cache. Requests above 6144 condition rows (more
than one 1 MP reference) fill the condition caches before the first step in
4096-row chunks; condition rows never attend to target rows and use the
timestep-zero modulation, so this is the joint first step split by rows (not
checked for bit identity with the joint step).

Gufo's RNG differs from PyTorch's. Compare shared initial noise, not numeric
seeds alone. Edit RNG seeds also bind the ordered, resized RGBA references:
reusing a generation seed must not replay the noise that generated the source.
This preserves repeatability for identical requests, including successive edits,
without changing the denoising formulas.

Diffusers and Transformers source is Apache-2.0 licensed. The separately
downloaded weights use the
[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE),
which limits the granted use to non-commercial purposes; commercial use requires
a separate license. Gufo's MIT license does not relicense those weights.
