# Qwen-Image-2.1 evaluation

Qualified against the pinned official Diffusers pipeline, Torch 2.12.0 and
Transformers 5.17.0, using the original safetensors and shared initial noise.
The checks cover the numerical graph and two complete images; they do not
establish broad typography, compositional or perceptual benchmark scores.

Maintained checks:

- `qwen_image_21_test`: RGBA PNG round trip, Pillow-compatible resize,
  deterministic flow schedule, request bounds, JSON generation and multipart
  editing schema. With `MODEL_DIR`, also checks six official tokenizer goldens
  covering Unicode, whitespace, code and special tokens.
- `qwen_image_21_probe`: native model boundaries and generated PNG, using the
  actual safetensors. Keep dumps outside Git.
- `qwen_image_21_kernel_test`: independent FP64 checks for normalization,
  projections, convolution and masked/unmasked attention; exact packed layouts,
  padding, down/upscaling and allocation tails. Exhaustively checks BF16 SiLU
  and checks conversion boundaries, including nonfinite values.
- `tools/models/qwen_image_21/reference.py`: independent official Diffusers
  comparison using identical initial noise; `--teacher-force` supplies saved
  native inputs to each reference block to separate local errors from drift
  accumulated through previous blocks.

Matched block inputs: 256×256 output, one reference image, two steps. The
reference is resized to the official 1024² conditioning area, so this also
exercises 4K-token image projections and condition caching.

| Boundary | Minimum cosine | Maximum relative L2 |
| --- | ---: | ---: |
| Qwen3-VL vision blocks | 0.99999903 | 0.001393 |
| Qwen3-VL text blocks | 0.99999534 | 0.003054 |
| First-step DiT blocks | 0.99999977 | 0.000676 |
| Cached-step DiT blocks | 0.99999841 | 0.001786 |
| VAE encoder blocks | 0.99999831 | 0.001838 |
| VAE decoder blocks | 0.99999703 | 0.002438 |
| Flow sigmas | Exact | 0 |

All 137 matched blocks pass the 1% relative-L2 gate. Prompt token IDs and the
VAE's first convolution match exactly; vision patch embeddings have relative
L2 error 0.000042.

Complete 256×256, 40-step trajectories, without teacher forcing:

| Task | Final latent cosine | Decoded RGBA relative L2 | PNG PSNR | PNG MAE (0–255) |
| --- | ---: | ---: | ---: | ---: |
| “A red cube on a white table.” | 0.99991216 | 0.008965 | 48.33 dB | 0.379 |
| “Change the cube to blue.” | 0.99993344 | 0.002979 | 56.58 dB | 0.123 |

Both images were visually checked for the requested object/color and preserved
scene. Different BF16 reductions prevent bit-identical images across runtimes.
Metrics are retained in [artifacts/qualification.json](artifacts/qualification.json);
large tensor dumps remain outside Git.
The native fused attention keeps probabilities and value accumulation in FP32.
Its standalone FP32 output differs from an FP64 oracle by at most 0.00000164
relative L2 across the measured 512–8230-key shapes. Online softmax changes
rounding; the independent block and trajectory checks above include masked
attention, fused feed-forward projections and native convolution. Subsequent
normalization/RoPE fusion, exhaustive BF16 SiLU substitution and convolution
input reuse preserve all 155 saved tensors and RGBA pixels in the edit replay.
PNG compression changes encoded bytes, with exact decoded-pixel round trips.

Serving checks cover generation/edit PNG replay, independent `n=2`/concurrent
seeds, replay after changing image size, and disconnect recovery (0.32 s). OpenAI Python SDK 2.41.1 also passed JSON generation and
two-reference multipart editing through the pinned llama-swap, including model
aliases, non-square output and opaque alpha. All requests used localhost.

The 2026-09-21 component-prefetch check preserves the exact generated PNG.
Generation and editing also pass repeated/concurrent seeds, image-size
replacement and disconnect recovery on the production binary.

Focused commands:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test \
  --target qwen_image_21_test qwen_image_21_kernel_test qwen_image_21_probe -j 8
./build/gpu-test/qwen_image_21_test
./build/gpu-test/qwen_image_21_test MODEL_DIR
./build/gpu-test/qwen_image_21_kernel_test
./build/gpu-test/qwen_image_21_probe MODEL_DIR /tmp/native-image \
  'Change the cube to blue.' 256 2 /tmp/reference.png
python tools/models/qwen_image_21/reference.py --model MODEL_DIR \
  --native /tmp/native-image --output /tmp/official-image --teacher-force \
  --prompt 'Change the cube to blue.' --size 256 --steps 2 --image /tmp/reference.png
```

Use the pinned [upstream sources](../../../src/models/qwen_image_21/UPSTREAM.md)
in a development reference environment. No hosted inference service is needed.
Before retaining an optimization, compare the same weights, prompt, references,
seed, dimensions and step count; rerun affected operator checks and independent
end-to-end generation/editing checks.
