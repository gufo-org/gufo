# ROCm 10.1 migration

October 24, 2026. BF16 Qwen-Image-2.1 on Strix Halo `gfx1151`.
Focused numerical, production timing and image API checks are complete.

## Configuration

The control uses the pinned ROCm 7.2.3 Nix toolchain, HIP Clang 22 and GCC
15.3. ROCm 10.1.0 was first checked with HIP Clang 24/GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`, then built through `nix build`
with cached Clang 24/GCC 16.2 and the matching repository ROCr patch.
Source `65bf6798` includes main `67d1f6ac`; this model's kernels are unchanged
from the independently checked `6d43e786`. Builds and inference run separately
on the same machine with the same BF16 checkpoint. Default Nix remains 7.2.3;
`nix develop` supplies the development toolchain.

## Migration steps

| Step | Status |
| --- | --- |
| Production builds on 7.2.3 and 10.1 | Complete |
| Independent attention/projection, BF16 conversion and layout checks | Pass on both toolchains; includes 64-channel output and ragged rows |
| Compiler register and spill comparison | New output kernel uses 30/31 VGPRs, zero scratch/spills on both compilers |
| Matched generation/edit trajectories with unchanged noise | Complete, 40 steps each; scores below |
| Warm HTTP performance, seeded replay and request isolation | Pass; per-phase timings, n=2, concurrent seeds, resizing and disconnect recovery |

Existing independent-reference limits remain in [QUALITY.md](QUALITY.md).
Two-step runs measure execution speed; complete 40-step trajectories are
checked separately. [BENCHMARKS.md](BENCHMARKS.md) retains its published
toolchain and numbers.

## Required change and quality

The first generation difference occurs after all 32 transformer blocks: the
64-channel output projection selected different BLAS arithmetic. It now uses
the existing packed BF16/FP32 WMMA projection with small channel tiles. Weight
packing replaces the original buffer; no extra persistent weight copy or
reduced precision is introduced. Tests include 1/31/32/33/256/4096 rows.

In a real-weight 16,384-value projection, mismatches against the independently
rounded FP64 result fall from 52 (main/7.2.3) and 50 (main/10.1) to 49 on either
candidate. Relative L2 improves slightly, from 0.001665077 to 0.001665071.

Complete 40-step, 256² trajectories use identical noise and the same reference
image. Lower decoded-image relative L2 means closer numerical agreement:

| Case | Main / 7.2.3 | Main / 10.1 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: | ---: |
| Generate red cube | 0.013058 | 0.019063 | 0.010939 | 0.010939 |
| Edit cube to blue | 0.004091 | 0.003909 | 0.003762 | 0.003858 |

The container comparison's entire generation trajectory and PNG match byte for
byte across the two compilers. Editing retains small cross-compiler differences; both candidates
improve against the independent reference. Tokenization, schedule and shared
noise agree, and the requested colors/object are preserved on visual review.
These two images do not establish broad perceptual or typography quality.

The independent reference is Diffusers `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
with PyTorch 2.12.0 and Transformers 5.17.0. These are fresh matched controls,
not replacements for the historical reference measurements in `QUALITY.md`.

## Cached Nix qualification

The Nix Clang 24/GCC 16.2 build passes the same operator checks. GCC 16's
`std::normal_distribution<float>` changes the seeded noise sequence; a
standalone CPU reproduction confirms this independently of HIP. The complete
40-step reference replays therefore use the Nix build's captured noise.
Decoded-image relative L2 is 0.025231 for generation and 0.014601 for editing;
tokenization and all 41 schedule values match exactly. These are different
inputs from the preceding table, so their scores are not a matched regression
comparison. Both outputs preserve the requested cube and colors on review.
No quality limit or frozen fixture changes.

A final production-API control on unchanged main uses the same GCC 16 noise
and the same reference PNGs. Generation PSNR improves from 39.50 to 39.85 dB;
editing improves from 43.64 to 44.31 dB. These matched checks support the
numerical improvement on both inputs; they do not establish general perceptual
quality.

## Focused performance and serving

Production HTTP generation, 1024², seed 42, two steps. Each entry is one
complete request in seconds; prompt encoding, denoising and VAE phases are
also compared individually against the 5% and 3 ms gate.

| Request | Main / 7.2.3 | Main / 10.1 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: | ---: |
| First request, model files cached | 8.078 | 8.109 | 8.046 | 8.151 |
| Warm request 1 | 5.247 | 5.254 | 5.275 | 5.291 |
| Warm request 2 | 5.253 | 5.290 | 5.267 | 5.315 |

No matched request/phase crosses the gate. The first 7.2.3 candidate request
immediately after DeepSeek took **14.299 s**, including 7.827 s in the
unchanged prompt phase. A fresh-process retry after the image weights had
been used took 8.046 s, with a 1.789 s prompt phase, matching the original
cached-file control. The earlier sample is retained; different file-cache
history must not be treated as a kernel comparison.

In the container comparison, seeded 1024² output repeats exactly within and
across both candidates.
The 10.1 API check also passes independent seeds with `n=2`, concurrent
requests, a size change and disconnect recovery (0.304 s). These bounded
two-step requests measure execution and isolation, not final image quality.

The rebased cached Nix build also passes all request/phase timing comparisons:

| Request | Candidate / 7.2.3 | Candidate / Nix 10.1 |
| --- | ---: | ---: |
| First request, model files cached | 8.050 s | 8.118 s |
| Warm request 1 | 5.233 s | 5.261 s |
| Warm request 2 | 5.243 s | 5.302 s |

Nix seeded replay, `n=2`, concurrent seeds, resizing and disconnect recovery
(0.302 s) pass. Cross-toolchain PNGs differ because of the noise sequence
described above; each build repeats its own output exactly.
