# MOSS-Image quality

**Text-to-image matches the PyTorch reference closely at every stage boundary.**
The reference is the checkpoint's own pipeline
([pinned sources](../../../src/models/moss_image/UPSTREAM.md)) on the CPU with
the original BF16 weights and FP32 VAE, the same prompt and shared initial
noise (`moss_image_probe --noise`). 256×256, 4 steps, guidance 3.0,
`A red fox sitting in fresh snow, golden hour light`:

| Stage | Cosine | Relative L2 | Max abs |
| --- | ---: | ---: | ---: |
| Text encoder, prompt (21 tokens) | 0.999874 | 0.0159 | 0.5625 |
| Text encoder, empty prompt (10 tokens) | 0.999814 | 0.0193 | 0.5625 |
| Transformer step 0, positive | 0.999969 | 0.0079 | 0.0469 |
| Transformer step 0, negative (block 22 skipped) | 0.999981 | 0.0062 | 0.0312 |
| Latent after step 1 / 2 | 0.999996 / 0.999845 | 0.0027 / 0.0176 | 0.013 / 0.117 |
| Latent after step 3 / 4 | 0.991719 / 0.991756 | 0.1292 / 0.1289 | 1.09 / 1.11 |
| Decoded image (before clamping) | 0.994125 | 0.1083 | 0.896 |

Final PNG against the reference PNG: 31.6 dB PSNR; the images show the same
scene. Single-step outputs agree to 0.99997 or better; the four-step trajectory
amplifies BF16 reduction differences (the third step is the largest, sigma
0.717 to 0.024). Higher cosine/PSNR and lower relative L2 mean closer
numerical agreement; they are not prompt-adherence or aesthetic scores.

| Check | Result |
| --- | --- |
| Flow schedule | Bit-exact against the reference sigmas (4 steps) |
| Prompt framing | Same token counts as the reference chat template (21 and 10); encoder outputs as above |
| Operators | `moss_image_kernel_test`: modulation and gated residual exact against BF16-rounded host arithmetic; situ gate within one BF16 ulp; GroupNorm within BF16 rounding of an FP64 reference |
| Seeded replay | Same PNG bytes for the same seed across server restarts and builds with the same kernels |
| API | Edits and transparent backgrounds rejected with 400; `n`, seeds and sizes as for Qwen-Image |

**Limits:** one short trajectory at 256² is not a broad composition,
typography or perceptual corpus. Gufo's RNG differs from PyTorch's, so numeric
seeds do not reproduce reference images; compare shared noise. The VAE decodes
in BF16 where the reference uses FP32. Measured on Windows; Linux is TODO.

## Reproduce

```sh
cmake --build --preset gpu-test \
  --target moss_image_test moss_image_kernel_test moss_image_probe
./build/gpu-test/moss_image_kernel_test
./build/gpu-test/moss_image_probe "$MODEL_DIR" moss-image-text-encoder-bf16.gguf \
  /tmp/native-moss 'A red fox sitting in fresh snow, golden hour light' 256 4
python3 tools/models/moss_image/reference.py --model "$MODEL_DIR" \
  --native /tmp/native-moss --output /tmp/reference-moss --size 256 --steps 4
```

`reference.py` replays the probe's `noise.f32` and writes `comparison.json`.
The table above was measured with an equivalent earlier script that drew the
noise in NumPy and replayed it with `--noise`.
