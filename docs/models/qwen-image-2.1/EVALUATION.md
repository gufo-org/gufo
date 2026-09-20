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
- `qwen_image_21_kernel_test`: exact head-layout checks with ragged inputs,
  exact convolution/packed layouts (padding, down/upscaling and tails), and independent
  FP64 oracles for normalization, dense projections and masked/unmasked fused attention.
  Checks allocation boundaries and seeded replay without loading weights.
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
| First-step DiT blocks | 0.99999976 | 0.000690 |
| Cached-step DiT blocks | 0.99999883 | 0.001530 |
| VAE encoder blocks | 0.99999830 | 0.001841 |
| VAE decoder blocks | 0.99999711 | 0.002406 |
| Flow sigmas | Exact | 0 |

All 137 matched blocks pass the 1% relative-L2 gate. Prompt token IDs and the
VAE's first convolution match exactly; vision patch embeddings have relative
L2 error 0.000042.

Complete 256×256, 40-step trajectories, without teacher forcing:

| Task | Final latent cosine | Decoded RGBA relative L2 | PNG PSNR | PNG MAE (0–255) |
| --- | ---: | ---: | ---: | ---: |
| “A red cube on a white table.” | 0.99991216 | 0.008970 | 48.33 dB | 0.380 |
| “Change the cube to blue.” | 0.99993108 | 0.003307 | 55.95 dB | 0.133 |

Both images were visually checked for the requested object/color and preserved
scene. Different BF16 reductions prevent bit-identical images across runtimes.
See the [visual comparison](artifacts/comparison.png).
Metrics are retained in [artifacts/qualification.json](artifacts/qualification.json);
large tensor dumps remain outside Git.
The native fused attention keeps probabilities and value accumulation in FP32.
Its standalone FP32 output differs from an FP64 oracle by at most 0.00000164
relative L2 across the measured 512–8230-key shapes. Online softmax changes
rounding; the independent block and trajectory checks above include masked
attention and the native BF16 projection kernel. The subsequent packing-layout
and wave-normalization optimizations, plus two-way QK-loop unrolling, preserve
all 156 saved model boundaries and the PNG exactly; normalization keeps the
original reduction order.

Serving checks cover generation/edit PNG replay, independent `n=2`/concurrent
seeds, replay after changing image size, and disconnect recovery (0.35 s). OpenAI Python SDK 2.41.1 also passed JSON generation and
two-reference multipart editing through the pinned llama-swap, including model
aliases, non-square output and opaque alpha. All requests used localhost.

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
