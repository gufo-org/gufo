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

## ROCm 10.1 migration

Focused cross-compiler checks, matched controls and remaining limits are in
[ROCM.md](ROCM.md). The original reference fixtures and tolerances above are
unchanged; migration measurements use separately recorded controls.

## LoRA adapters

Base generation without adapters is the qualified path above and must stay
pixel-identical to a build without adapter support. With adapters, the merged
BF16 weights are compared with an independent FP64 numpy reference of the
[ComfyUI semantics](../../../src/models/qwen_image_21/UPSTREAM.md#lora-adapters),
for every patched tensor of one published file per layout, at scale 0.8.

**Gates per tensor:** at most 1 BF16 ULP from the correctly rounded FP64
result, for at most 0.1% of the elements differing; relative L2 at most 0.01;
a non-zero update; unpatched tensors bit-identical to the checkpoint. Where
the update cancels the weight (`|W + ΔW|` far below `|W|`) or the rank-r product
cancels itself, any FP32 merge, including ComfyUI's, can be several BF16 ULPs of
the tiny result away; such elements must instead be within 2⁻⁸ BF16 ULP at the
scale of `|W|` plus the summed magnitudes of the `B·A` terms.
Their count is reported separately.

| Layout | File (revision) | Max ULP / cancellations | Mismatch | Relative L2 |
| --- | --- | ---: | ---: | ---: |
| PEFT, fused `gate_up`, F16 | `Danrisi/samsung_qwen2.1` (`299e8dbe`) | 11968 / 226 | 1.0e-5 | 1.6e-3 |
| DiffSynth `.default` | `prithivMLmods/Qwen-Image-2.1-Voxel-Style` (`b4ccfe93`) | 102 / 134 | 9.1e-6 | 1.6e-3 |
| PEFT `transformer.`, non-block targets | `Viggle/Qwen-Image-2.1-viggle-turbo` (`009a44a8`) | 56 / 27 | 8.6e-6 | 1.6e-3 |
| PEFT, metadata alpha | `PrunaAI/P-Q-Image-2.1` (`b479282e`) | 11462 / 30 | 8.1e-6 | 1.4e-3 |
| PEFT `diffusion_model.transformer.` | `xingewh/...-r256_comfy` (`53d988bf`) | 54 / 11 † | 2.5e-5 | 1.7e-3 |
| `base_model.model.` `.default`, F32 | `orwelian84/UltraTex-Qwen-Image-2.1` (`3afb07ac`) | 272 / 162 † | 1.9e-5 | 1.7e-3 |
| Bare `.lora_A` | `Haverbex/Qwen-Image-2.1-UltraFast` (`a99bedb7`) | 4 / 4 | 4.2e-6 | 6.7e-4 |
| Bare `.lora_down`, full weights stripped | `alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs` (`f7545234`) | 11341 / 106 | 3.4e-5 | 1.6e-3 |
| Diffusers `.lora.down.weight`, F32 | `lightx2v/Qwen-Image-2.1-Lightning` (`6b0a1176`) | 11902 / 327 | 1.4e-5 | 1.7e-3 |
| `.lora_down.weight` + `.alpha` | `t8star/Qwen-Image-2.1-viggle-turbo-4step-r64-comfy` (`a425c6ee`) | 54 / 10 † | 2.5e-5 | 1.7e-3 |
| Kohya + `.alpha` | `liueeeeaa/qwen-image-21-...-comfy-fused` (`80d38cd8`) | 12328 / 248 † | 2.4e-5 | 1.7e-3 |
| `.diff` with variable ranks | `nynxz/Qwen-Image-2.1` (`011b97bb`) | 11400 / 32 | 1.0e-5 | 1.6e-3 |
| DoRA, fused `gate_up` | `e-n-v-y/Qwen-Image-2.1-Fix` (`5b2c7be6`) | 12114 / 479 | 1.3e-5 | 1.7e-3 |
| PEFT + `.alpha` | `e-n-v-y/Qwen-Image-2.1-Fix-v2.0` (`045f6828`) | 23882 / 162 | 1.5e-5 | 1.7e-3 |

All gates pass on gfx1151 for every compared tensor. † Files over 600 MB: blocks 0, 15
and 31 plus every non-block tensor, with the unpatched ones bit-identical. This scale is never
below the earlier `max(|W|, |ΔW|)`, under which every file passed (worst 2⁻⁸,
Samsung, where one rank-16 product cancels and a sequential FP32 merge gives the
same value). Re-checked under it, Samsung, Lightning, `.diff` and DoRA stay
below 3e-5 BF16 ULP. Samsung at scale -0.5 and DoRA at scale 1.0 also pass.

| Check | Result |
| --- | --- |
| No adapters: probe trajectory, 1024² generation and edit versus main | Byte-identical: all 59 probe outputs (trajectories, tokens, PNGs); HTTP denoising 93.43-93.52 s versus 93.41-93.57 s on main |
| Switching A → B → A → A+B → none: rebuilt weights bit-identical, only changed targets rebuilt | Pass for 3 pairs (low-rank, DoRA, `.diff` and non-block targets): 0 mismatches |
| Seeded replay with the same `loras`; base → adapter → base images | Identical PNGs for base → Voxel → base, Voxel → Samsung → Voxel and a repeated request |
| Matched-seed 1024² image with and without a style adapter | Genshin TCG: cel-shaded 2D art versus a painterly render; phone-photo: handheld snapshot look; Voxel edit: block geometry |
| Rejected at startup: Qwen-Image Edit-2509, training auxiliary, Fun-Acc | Exit 1; each message names the first offending key |

### Reproduce

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen_image_21_lora_test qwen_image_21_kernel_test qwen_image_21_lora_dump
nix develop -c ./build/gpu-test/qwen_image_21_lora_dump "$MODEL_DIR" /tmp/lora-dump \
  --lora style="$LORA" --scale 0.8
nix shell --impure --expr 'with (builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux;
  python3.withPackages (p: [p.numpy])' -c python3 tools/models/qwen_image_21/lora_reference.py \
  --model "$MODEL_DIR" --lora "$LORA" --scale 0.8 --dump /tmp/lora-dump > lora.json
nix develop -c ./build/gpu-test/qwen_image_21_lora_dump "$MODEL_DIR" \
  --switch-check a="$LORA_A" b="$LORA_B"
```

The dump writes every patched tensor as FP32 (several GB for large adapters);
`--tensor SUBSTRING` limits it to matching tensors, including unpatched ones,
which the reference requires to equal the checkpoint. `lora_reference.py
--derive-strip OUT` copies a file without its full replacement weights.
