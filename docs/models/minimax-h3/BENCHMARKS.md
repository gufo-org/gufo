# MiniMax H3 benchmarks and memory

Nix release on gfx1151. Current qualified end-to-end timing matrix is **TODO**;
component traces and extrapolations are not complete-generation measurements.

2026-09-21 loading control: metadata-only HTTP readiness **0.17 s**.
The first four real prompt-encoder layers over six tokens take **1.48 s**,
including **0.83 s** summed batch loading and **0.21 s** GPU execution;
loading overlaps compute. Peak retained device memory is **1.88 GiB**.
Output is byte-identical to the previous loader. No full video was generated.

| Preset | Single request wall | Peak resident memory | Delivery quality |
| --- | ---: | ---: | --- |
| exact, 512x512 | TODO | TODO | Component evidence only |
| exact-1344x768 | TODO | TODO | Component evidence only |
| fast | TODO | TODO | Approximate; full comparison TODO |
| aggressive | TODO | TODO | Approximate; full comparison TODO |
| dev, selected frames | TODO | TODO | Bounded diagnostic only |

One worker executes jobs, with one queued job. There is no parallel C>1 model
execution; throughput/queue latency measurements are **TODO**.

## Native DiT components — 2026-09-21

gfx1151, ROCm 7.2.3, BF16 inputs/output, 56 heads of width 128. Kernel
measurements use the production compiler flags; real-weight block controls
use the optimized diagnostic executable. These are component measurements,
not complete-generation latency.

| Sequence rows | Native attention, GPU ms | Real-weight block, GPU ms | Block wall ms |
| --- | ---: | ---: | ---: |
| 528 | 0.63 | TODO | TODO |
| 1,872 | 6.02 | TODO | TODO |
| 4,096 | 27.33 | TODO | TODO |
| 4,097 | 14.20 | TODO | TODO |
| 7,136 | 39.50 | 312.85 | 326.31 |
| 37,716 | 1,087.53 | 1,761.13 | 1,823.90 |

Short attention averages ten timed launches after warm-up; the current long
attention controls time one launch after warm-up, including value packing.
Block controls use one process run per shape. Loading is excluded.
Full-resolution QKV projection includes normalization/RoPE and packed V;
FFN up includes SwiGLU. AdaLN and attention write directly in the next
projection's packed layout. Packing replaces original weights during loading.

**91.00 s is the time for one denoiser evaluation, not a complete video.**
The exact preset needs **49 evaluations**, followed by video/audio decoding.
Complete-generation latency remains **TODO**.

This single evaluation runs all 50 transformer blocks at **1344×768×124**,
with six text rows and zero initial latents. Setup/loading is excluded.
It does not run the denoising schedule or VAE.
Peak retained memory is **40.01 GiB**, including **2.93 GiB** scratch and
retained BLAS workspace. Full-resolution transformer blocks use only native
projections and allocate no BLAS handles.

Attention remains the largest cost, followed by projections; the GPU is
continuously busy during the transformer blocks. Block activation storage is
**2.90 GiB**. No complete-video speedup is established.
See [quality evidence](QUALITY.md#full-resolution-kernel-qualification),
the [current measurement record](artifacts/full-resolution-kernels.json), and
the [short-attention record](artifacts/native-attention.json).

The production Nix runtime closure is **3.82 GiB** with no Triton/AOTriton,
Composable Kernel or MIOpen dependencies.
This is package storage, not model RAM or GPU memory.

## Memory and phase ownership

| Component inventory | Tensor bytes | GiB |
| --- | ---: | ---: |
| Qwen prompt encoder | 66,714,780,128 | 62.13 |
| AdaLN/time precompute | 26,142,079,488 | 24.35 |
| DiT core and heads | 40,138,350,656 | 37.38 |
| VisualVAE | 10,415,484,128 | 9.70 |
| AudioVAE | 605,306,340 | 0.56 |

The phase owner releases streams, mappings, HIP registrations, weights, and
arenas on completion, failure, or cancellation. Component inventories and
phase peaks must not be added as though all were resident together.

1. **Prompt encoding:** the private tokenizer uses NFC, byte BPE, and the
   checkpoint's added-token rules. It inserts no chat template or vision
   framing. The encoder captures the first 50 Qwen3-VL layers before the final
   norm. It releases the embedding after lookup and streams two layers at a
   time, overlapping transfer with compute.
2. **Denoiser setup:** timestep arithmetic stays F32 through SiLU. Each large
   AdaLN projection is loaded, used to precompute schedule constants, and
   released. Core loading is bounded to two workers. Construction validates
   BF16 GEMM plans and releases temporary validation/staging buffers.
3. **Denoising:** retained core blocks stay resident. One BF16 device buffer
   carries hidden states in place on one ordered stream; scratch reuses dead
   activations. Final normalization writes BF16-rounded values directly into
   F32 projection scratch.
   F32 sample and velocity buffers implement Diffusers' two-stage Euler
   operation order. Reuse retains the previous video/audio velocities.
   Cancellation is checked at block and step boundaries.
4. **VisualVAE:** fixed 256-pixel spatial tiles and seven-latent/22-frame
   temporal chunks preserve decoder context. Selected-frame decoding skips
   unrelated chunks and readback while retaining required model work.
5. **AudioVAE and output:** F32 decoder convolution uses bounded im2col/GEMM;
   left/right channels remain independent. Unused audio-encoder weights are
   excluded. FFmpeg receives bounded RGB/PCM streams and produces H.264/AAC
   with 24-fps video and stereo 32-kHz audio.

Telemetry separates inspection, tokenization, loading, AdaLN precompute,
fresh denoiser forwards, sampler time, VAE work, media composition, faults,
swap, and memory peaks. `first_preview_ms` measures availability of the
requested frame set, not a streamed first frame.

## Measure

Use the [CLI examples](README.md) and `--profile` to record inspection, prompt,
loading, AdaLN precompute, fresh forwards, sampler, VAE, mux, faults/swap and
memory peaks. Record model/engine, seed, geometry, schedule and input hashes.
`first_preview_ms` is the requested frame set, not streamed first-frame latency.

`tools/gufo/h3_profile.py` handles telemetry; full generation requires explicit
`--allow-full-generation`. Routine iteration uses one block/forward or selected
frames with independent [quality gates](QUALITY.md), not full videos.
