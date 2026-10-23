# ROCm 10.1 migration

October 23, 2026. BF16 Qwen-Image-2.1 on Strix Halo `gfx1151`.
Numerical migration checks are complete; production timing checks are in progress.

## Configuration

The control uses the pinned ROCm 7.2.3 Nix toolchain, HIP Clang 22 and GCC
15.3. The candidate uses ROCm 10.1.0, HIP Clang 24 and GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`. Both start from main `ade4f4d7`
and use the repository's matching ROCr queue-memory patch. Builds and
inference run separately on the same machine with the same BF16 checkpoint.

The optional Nix entry points are `nix build .#rocm10` and
`nix develop .#rocm10`; the measurements here qualify the container build.
The default Nix toolchain remains 7.2.3.

## Migration steps

| Step | Status |
| --- | --- |
| Production builds on 7.2.3 and 10.1 | Complete |
| Independent attention/projection, BF16 conversion and layout checks | Pass on both toolchains; includes 64-channel output and ragged rows |
| Compiler register and spill comparison | No new image-kernel spill increase in the initial compiler comparison |
| Matched generation/edit trajectories with unchanged noise | Complete, 40 steps each; scores below |
| Warm HTTP performance, seeded replay and request isolation | In progress |

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

Generation's entire trajectory and PNG now match byte for byte across the two
compilers. Editing retains small cross-compiler differences; both candidates
improve against the independent reference. Tokenization, schedule and shared
noise agree, and the requested colors/object are preserved on visual review.
These two images do not establish broad perceptual or typography quality.

The independent reference is Diffusers `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
with PyTorch 2.12.0 and Transformers 5.17.0. These are fresh matched controls,
not replacements for the historical reference measurements in `QUALITY.md`.
