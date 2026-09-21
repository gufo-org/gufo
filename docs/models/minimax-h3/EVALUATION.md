# MiniMax H3 evaluation

The authoritative contract and component payload identities live in:

- [quality contract](../../../tests/fixtures/minimax_h3/quality-contract-v1.json)
- [DiT block oracle](../../../tests/fixtures/minimax_h3/dit-block0-oracle-v1.json)
- [denoiser oracle](../../../tests/fixtures/minimax_h3/denoiser-oracle-v1.json)
- [VisualVAE oracle](../../../tests/fixtures/minimax_h3/video-vae-oracle-v1.json)
- [AudioVAE oracle](../../../tests/fixtures/minimax_h3/audio-vae-oracle-v1.json)

A changed prompt, seed, shape, schedule, block/reuse policy, metric, or ceiling
requires a new contract version. Never loosen a threshold to admit a
candidate. Numerical comparisons require matching shapes and zero non-finite
values.

| Boundary | Frozen strict ceiling |
| --- | --- |
| BF16 DiT block | relative max and relative L2 below `1e-2` |
| Qwen layer-50 output | relative max below `0.1`, relative L2 below `0.05` |
| 256x256x22 video velocity | relative max below `0.05`, relative L2 below `0.04` |
| 256x256x22 audio velocity | relative max and relative L2 below `0.05` |
| VisualVAE | relative max and relative L2 below `0.05` |
| AudioVAE waveform | max absolute below `1e-3`, relative L2 below `0.05` |
| AudioVAE STFT magnitudes | relative max below `0.1`, relative L2 below `0.08` |

Use the smallest independent oracle covering the change:

| Changed boundary | Validation |
| --- | --- |
| Loader/tokenizer/API | malformed inventory, exact token IDs, lifecycle and protocol tests |
| Sampler or schedule | analytic host/HIP layout, shifted grids, seeded noise, Euler trajectory, reuse ranking |
| DiT operator/block | analytic primitives, native attention tails/replay against FP64 and one frozen 528-row block; qualify the affected long shape separately |
| Cross-block denoiser | one complete forward against the frozen conditioning and input latents |
| VisualVAE | one selected-frame tile against its F32 teacher |
| AudioVAE | one stereo waveform and deterministic STFT comparison |

Normal development does not run complete denoising schedules or generate full
videos. A selected-frame smoke may follow a change crossing component
boundaries; full generation and delivery-quality studies are explicit release
work. A playable MP4 or recognizable frame alone cannot qualify an
optimization.

Dependency-light schema/corruption checks and offline ML checks are separate:

```sh
nix build .#checks.x86_64-linux.h3-quality
nix build .#checks.x86_64-linux.h3-ml-quality

nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target minimax_h3_dit_hip_test
nix develop -c ctest --preset gpu-full \
  -R '^minimax_h3_dit_analytic$' --output-on-failure
```

Real-checkpoint tests use operator-owned artifacts and explicit
`GUFO_H3_MODEL_ROOT` plus component oracle paths. For a cross-block change:

```sh
GUFO_H3_MODEL_ROOT=/var/llms/huggingface/MiniMax-H3 \
GUFO_H3_DENOISER_GOLDEN=/var/llms/huggingface/strix-h3-oracles/denoiser-256x256x22-forward-v2 \
  nix develop -c ./build/gpu-test/minimax_h3_denoiser_hip_test --oracle-forward
```

Teacher capture tools are model-scoped under `tools/gufo/h3_*_golden.py`
(prompt, DiT, denoiser, video VAE, and audio VAE). The prompt teacher loads
only the selected checkpoint layers in Transformers; component teachers
execute independent ROCm PyTorch formulas. The denoiser teacher was also
checked against the pinned converted Diffusers transformer. Identical
integer seeds do not equate Gufo's PCG/Box-Muller noise with PyTorch's RNG:
cross-runtime comparisons inject the same retained noise tensors.

External payloads stay outside Git in content-addressed directories. Each
manifest binds the contract, model/source and oracle revisions, dtypes,
conditioning hash, shapes, generation parameters, and every payload hash.
Paths, role coverage, metrics, and hashes are validated before comparison:

```sh
nix develop -c python3 tools/h3/gufo-h3-quality.py verify \
  --artifact /var/llms/h3-oracles/sha256/<sha256>
```

The following retained measurements document component evidence:

| Boundary | Relative L2 | Relative max |
| --- | ---: | ---: |
| prompt layers 1 and 50, six-token fox prompt | `0` | `0` |
| 528-row block output, rechecked 2026-09-21 | `0.00461035` | `0.00483092` |
| 256x256x22 video velocity, native short attention | `0.0193639` | `0.0458577` |
| 256x256x22 audio velocity, native short attention | `0.0112591` | `0.0216204` |
| 512x512x22 refined text | `0.00469088` | `0.00167411` |
| 512x512x22 block-0 modulation | `0.0022895` | `0.00478469` |
| 512x512x22 video velocity | `0.0132707` | `0.0154553` |
| 512x512x22 audio velocity | `0.00731056` | `0.0149873` |
| selected VisualVAE frames | `8.14968e-7` | `2.24925e-6` |
| AudioVAE waveform | `1.08936e-5` | max absolute `6.03162e-5` |
| AudioVAE STFT | `3.62714e-6` | `5.8276e-6` |

### Native attention qualification

On 2026-09-21, the native HIP replacement was evaluated against independent formulas and
checkpoint teachers. The frozen ceilings above are unchanged.

- Short sequences now use native 128-key softmax tiles. They retain contiguous
  eight-key lane reductions, BF16 local probabilities and explicit FP32 FMAs.
  At 528/1872/4096 rows with all 56 heads, outputs are byte-identical to the
  removed CK implementation. The frozen block and complete-forward teacher
  errors above are unchanged; this preserves accuracy without claiming an
  improvement. CK's MIT attribution remains; its headers are no longer needed.

- The maintained analytic test compares dense attention with FP64 softmax/PV
  at 1/8/9, 31/32/33, 63/64/65, 127/128/129, 528 and 4096/4097 rows,
  including uniform and peaked scores. Relative L2 stays below `0.004` and
  relative max below `0.01`; all outputs are finite and repeated launches
  are byte-identical. Input allocations end at the final head's tail, and
  an output guard checks stores beyond the valid extent.
- A separate 56-head control at 4097/7136/37716 rows gives FP64-relative L2
  `0.0021471` / `0.0023339` / `0.0023794` on first/middle/last queries in
  the first and last heads.
- A real-weight 7136-row block, using the profiling input and independent
  FP32 PyTorch teacher, gives output relative L2 `0.00387690`.
  Relative max is `0.0104167`, the same pre-existing outlier as the removed
  implementation. This additional stress case therefore exceeds the frozen
  528-row fixture's 1% max ceiling; it demonstrates no added error on this
  case, not a newly passed strict block gate. The frozen 528-row gate passes.
- A short-path probability-refinement experiment improved one block but
  worsened the complete-forward teacher result. It was rejected; its
  arithmetic is absent from the retained kernel.

Raw tensors and traces remain outside Git. The compact
[qualification record](artifacts/native-attention.json) contains identities,
metrics and measurement scope. No full denoising schedule or video was generated.

### Full-resolution kernel qualification

The fused QKV RMSNorm/RoPE, packed-value attention, native BF16 FFN down
projection and fused SwiGLU/packing retain the same model arithmetic.
On 2026-09-21:

- At 37,716 rows, all **202,761,216 BF16 elements** of real-weight block 0
  are byte-identical across the baseline and final implementation.
- Full-size randomized controls also match every BF16 output: Q/K/V
  (270,348,288 elements each), attention (270,348,288), and FFN down
  (202,761,216). Independent FP64 sample relative L2 remains `0.00237943`
  for attention and `0.00257995` for the projection.
- The frozen 528-row block and complete 50-block forward retain exactly
  the teacher errors in the table above; no thresholds changed.
- Maintained analytic checks cover nonunit Q/K norm weights, partial rotary
  dimensions, both output layouts, attention tails/replay and guarded packing,
  and an independently solvable FFN projection with a partial 129-row tile.
  Fused SwiGLU/packing matches all 65,536 BF16 gate encodings and a partial
  row tile, including the original rounding and nonfinite behavior.
  The existing three-block reuse smoke also passes.

Packing changes storage order only. Original FFN weights are released after
packing, and SwiGLU writes packed activations directly. These checks establish
no added error at the measured boundaries; full-video perceptual evaluation
remains separate. Identities and measurement scopes are in the
[compact record](artifacts/full-resolution-kernels.json).

Component agreement does not establish end-to-end semantic quality. Pre-audit
multi-step latents and checkerboard video captures are invalid promotion
evidence. Fast/aggressive delivery quality must be compared with the audited
exact route. Delivery reports bind sibling parameter documents and measure
latent error, per-frame/temporal SSIM and pinned LPIPS, stereo waveform/STFT
error, durations/A/V sync, and human prompt-adherence review.

## Upstream authority

```text
repository: antirez/h3.c
revision:   8974cc055ea9c02fcd14cc27dfda3e1027c05153
license:    MIT
copyright:  Copyright (c) 2026 Salvatore Sanfilippo
```

The pinned source supplies the initial independent implementation contract for:

- safetensors tensor names and component boundaries;
- tokenizer and Qwen3-VL layer-50 prompt encoding;
- packed text/video/audio sequence layout;
- sigma schedule, RNG, Euler integration, and core reuse;
- the 50-block Omni Transformer;
- VisualVAE and AudioVAE decoding;
- selected-frame development and audiovisual mux behavior;
- exact, fast, and aggressive performance references.

The released Hugging Face Diffusers implementation is the authoritative
quality-parity reference for the model's Python pipeline:

```text
repository: huggingface/diffusers
revision:   fe15005a333d1270b490e4885a6ea13b66b1092a
```

A line-by-line text-only audit covers the pipeline, transformer, scheduler,
noise construction, VisualVAE, and AudioVAE. Where h3.c and Diffusers differ,
the native exact path follows Diffusers when the difference changes model
arithmetic or decoder context.

Gufo follows Diffusers for the 50-point/49-forward exact schedule, F32 Euler
operation order, timestep SiLU precision, MM-RoPE and fixed 256-pixel VisualVAE
tiles. Fast/aggressive borrow h3.c ranking/reuse but are approximate Gufo presets.
Gufo's PCG/Box-Muller stream differs from PyTorch; compare injected frozen noise,
not equal integer seeds. F32 decoders are compared numerically with upstream
FP16-autocast output. Unsupported conditioning is rejected. Exact mode uses all
50 blocks; no token reduction or block thinning applies.

## Adaptation provenance

Copyright (c) 2026 Salvatore Sanfilippo, MIT, applies to h3.c adaptations.
Metal/MPSGraph/Objective-C plumbing is not imported. Source headers and
[third-party notices](../../../THIRD_PARTY_NOTICES.md) retain applicable notices.

| Upstream h3.c boundary | Gufo destination / method |
| --- | --- |
| `h3_tokenizer.m`, `h3_text_encoder.c` | `tokenizer.*`, `prompt_encoder.*`, `prompt_encoder_ops.hip.hpp`: C++/HIP translation, strict JSON/ICU, streamed device weights and cancellation. |
| `h3_host.*`, packing/reuse helpers, Euler shader | `sampling.*`: checked layout, PCG/noise/reuse; independent HIP F32 update follows Diffusers arithmetic. |
| `h3_dit.c`, DiT shaders | `dit.*`, `dit_ops.hip.hpp`: operation-boundary HIP translation, model-owned buffers and independent analytic/block oracles. |
| `h3_dit_schedule.c`, core/final heads | `denoiser.*`: independent C++/HIP orchestration, streamed AdaLN, resident core, F32 velocities and cancellation. |
| `h3_video_vae.c`, released VisualVAE | `video_vae.*`, `video_vae_ops.hip.hpp`: F32 fixed tiles/chunks, selected-frame pruning, independent HIP ops. |
| `h3_audio_vae.c` | `audio_vae.*`, `audio_vae_ops.hip.hpp`: decoder-only F32 convolution, SnakeBeta, stereo and independent HIP ops. |
| `h3_ffmpeg.c`, mux tests | `media.*`: bounded pipes, cancellation, Nix FFmpeg and FFprobe validation. |
| `h3.c` generation flow/examples | `generation.*`, CLI: pinned preset contracts, explicit outputs and phase telemetry. |
| Released tokenizer/block/denoiser/VAE tests | Focused MiniMax tests and `tools/gufo/h3_*_golden.py`; independent external tensors and malformed-input coverage. |

All destinations are under `src/models/minimax_h3` unless named otherwise.
Upstream h3.c attributes Morton/INT8 TensorOps scheduling to ccv (Copyright
2010 Liu Liu, BSD-3-Clause). Those implementations are not imported here;
any future recognizable adaptation must carry its BSD notice.
