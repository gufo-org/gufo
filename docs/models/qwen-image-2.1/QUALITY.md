# Qwen-Image-2.1 quality

**Generation and editing pass the retained independent checks.** The reference
is the [official Diffusers pipeline](../../../src/models/qwen_image_21/UPSTREAM.md),
with original BF16 safetensors, identical prompts/reference images and shared
initial noise. [Measured evidence](artifacts/qualification.json).

| Complete 256×256 image, 40 steps | Final latent cosine | Decoded-image relative L2 | PNG PSNR |
| --- | ---: | ---: | ---: |
| Generate a red cube on a white table | 0.999912 | 0.008965 | 48.33 dB |
| Edit the cube to blue | 0.999933 | 0.002979 | 56.58 dB |

Both images were checked for the requested color/object and scene preservation.
Higher cosine/PSNR and lower relative L2 mean closer numerical agreement;
they are not prompt-adherence or aesthetic scores.

| Check | Result |
| --- | --- |
| 137 blocks with identical inputs | Worst relative L2 0.003054; all pass the 0.01 limit |
| Tokenizer and flow schedule | Exact |
| Native attention versus FP64 | Maximum relative L2 1.64e-6 over 512–8230 keys |
| Optimized edit replay | All 155 saved tensors and decoded pixels unchanged |
| Server | Generation/edit seed replay, independent `n=2`/concurrent seeds, size changes and disconnect recovery pass |
| Client compatibility | Two-reference multipart editing and generation pass through llama-swap, including aliases and non-square output |

The block fixture produces 256×256 output with a reference resized to the
official 1024² conditioning area. **Limits:** independent complete trajectories
cover two images, not a broad typography/composition/perceptual corpus or
full-resolution quality sweep. BF16 reduction order prevents bit-identical
images across runtimes. Latest loader/state replay: September 21, 2026.

## Reproduce

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen_image_21_test qwen_image_21_kernel_test qwen_image_21_probe
nix develop -c ./build/gpu-test/qwen_image_21_kernel_test
nix develop -c ./build/gpu-test/qwen_image_21_probe "$MODEL_DIR" /tmp/native-image \
  'Change the cube to blue.' 256 2 /tmp/reference.png
nix develop -c python3 tools/models/qwen_image_21/reference.py --model "$MODEL_DIR" \
  --native /tmp/native-image --output /tmp/official-image --teacher-force \
  --prompt 'Change the cube to blue.' --size 256 --steps 2 --image /tmp/reference.png
```

Teacher forcing isolates each block from earlier drift; also check complete
generation/edit trajectories after arithmetic changes. The pinned versions and
source hashes are in the evidence above. Raw tensors/images stay outside Git;
no hosted inference service is needed.
