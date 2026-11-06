# Qwen-Image-2.1 quality

**Generation and editing pass the retained independent checks.** The reference
is the [official Diffusers pipeline](../../../src/models/qwen_image_21/UPSTREAM.md),
with original BF16 safetensors, identical prompts/reference images and shared
initial noise. [Measured evidence](artifacts/qualification.json).

| Complete image, 40 steps | Final latent cosine | Decoded-image relative L2 | PNG PSNR |
| --- | ---: | ---: | ---: |
| Generate a red cube, 256×256 | 0.999912 | 0.008965 | 48.33 dB |
| Edit the cube to blue, 256×256 | 0.999933 | 0.002979 | 56.58 dB |
| Edit a teapot to blue and add a steaming cup, 1024×1024 | 0.999975 | 0.006881 | 51.85 dB |

Images were checked for the requested color/object and scene preservation.
Higher cosine/PSNR and lower relative L2 mean closer numerical agreement;
they are not prompt-adherence or aesthetic scores.

| Check | Result |
| --- | --- |
| 137 blocks with identical inputs | Worst relative L2 0.003054; all pass the 0.01 limit |
| Tokenizer and flow schedule | Exact |
| Native attention versus FP64 | Maximum relative L2 1.64e-6 over 512–8230 keys |
| Edit noise | Stable seed/pixel hashing; reference order, dimensions and alpha covered; generation seeds unchanged |
| Server | Generation/edit seed replay, independent `n=2`/concurrent seeds, size changes and disconnect recovery pass |
| Client compatibility | Two-reference multipart editing and generation pass through llama-swap, including aliases and non-square output |

References use the official 1024² conditioning area. **Limits:** these three
complete trajectories are not a broad typography/composition/perceptual corpus.
BF16 reduction order prevents bit-identical images across runtimes. Edit seeds
bind reference pixels to avoid reusing the noise that generated a source image;
cross-runtime comparisons must share actual noise, not just a numeric seed.
Latest full-resolution edit qualification: October 2, 2026.

## Reproduce

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen_image_21_test qwen_image_21_kernel_test qwen_image_21_probe
nix develop -c ./build/gpu-test/qwen_image_21_kernel_test
nix develop -c ./build/gpu-test/qwen_image_21_probe "$MODEL_DIR" /tmp/native-image \
  'Change the cube to blue.' 256 2 /tmp/reference.png
nix develop -c env MIOPEN_FIND_MODE=2 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  python3 tools/models/qwen_image_21/reference.py --model "$MODEL_DIR" \
  --native /tmp/native-image --output /tmp/official-image --teacher-force \
  --prompt 'Change the cube to blue.' --size 256 --steps 2 --image /tmp/reference.png
```

Teacher forcing checks conditioning inputs before replacing them and isolates
block arithmetic. For complete trajectories, use 40 steps and `--trajectory`
on the probe, then omit `--teacher-force`. `--skip-decode` gives a faster latent
comparison while investigating. Raw tensors/images stay outside Git.

## Qwen-Image-2.1-Turbo

Reference: Diffusers `aaf57d03a3edfbd9de457d2872417a92f149a91d`, which reads the
checkpoint's `sample_sigmas`, with shared initial noise at 256×256 and 8 steps.

| Check | Result |
| --- | --- |
| Schedule and tokenizer | Exact |
| Matched-input blocks | Generation: 297, worst relative L2 0.0073; editing: 346, worst 0.0048 (limit 0.01) |
| Generation trajectory | Final-latent cosine 0.99995; PNG 45.5 dB |
| Single-reference edit trajectory | Final-latent cosine 0.99991; PNG 50.2 dB |
| Original-plus-mask edit trajectory | Final-latent cosine 0.999829; PNG 40.6 dB |
| Server | Default and explicit 8 steps, seed/edit/mask replay, independent `n=2`/concurrent seeds, size changes, transparency, disconnect recovery and invalid-step rejection pass |

Use the commands above with 8 steps, `--model-revision`/`--diffusers-revision`
for the record, and `check_server.py --steps 8`. The pinned Diffusers revision
needs `huggingface_hub>=1.32`. Fixtures:
[original](artifacts/turbo-editing-input.png) and [mask](artifacts/turbo-editing-mask.png).
These checks compare against the Turbo reference, not the base model's quality.

## ROCm 10.1 migration

Focused cross-compiler checks, matched controls and remaining limits are in
[ROCM.md](ROCM.md). The original reference fixtures and tolerances above are
unchanged; migration measurements use separately recorded controls.
