# ROCm 10.1 migration

October 24, 2026. BF16 FL2VA on Strix Halo `gfx1151`.
Operator and single-block migration checks complete; full-video quality remains
unqualified.

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
| Independent prompt, DiT, sampling and audio primitive checks | Pass on both toolchains |
| Compiler register and spill comparison | No new H3 spill increase |
| One real-weight block at 1,872 and 37,716 rows | Complete; native large block exact, short BLAS result scored below |
| Matched block outputs and execution time | 7.2.3 output bytes preserved; results below |

This qualification is deliberately bounded to operators and individual
blocks. It does not run the full denoising schedule, decode a video or
establish full-video perceptual quality. Existing independent-reference
limits remain in [QUALITY.md](QUALITY.md); full-generation performance
remains unqualified in [BENCHMARKS.md](BENCHMARKS.md).

## Required changes

The 7.2.3 rocBLAS solution IDs for the 1,872-row block are unavailable in
10.1. Session construction now validates the current library default when a
pinned ID fails. The existing two-execution byte comparison remains mandatory;
the selected plan is shared by subsequent blocks. Successful 7.2.3 plans are
unchanged.

Explicit RoPE FMA operands preserve the established rounding under Clang 24.
This also fixes a near-zero cancellation failure in the analytic test. The
37,716-row native block now matches all 202,761,216 BF16 baseline values exactly
across both toolchains. Its intermediate precision and attention algorithm
are unchanged.

## Bounded measurements

One real-weight block, identical captured inputs; GPU time excludes weight
loading. These are component measurements, not video-generation timings.

| Rows | Main / 7.2.3 | Candidate / 7.2.3 | Candidate / 10.1 |
| ---: | ---: | ---: | ---: |
| 1,872 | 60.36 ms | 57.50 ms | 56.13 ms |
| 37,716 | 1,748.56 ms | 1,744.36 ms | 1,740.72 ms |

The cached Nix build takes 60.64 ms for the short block and 1,768.55 ms for
the large block. A matched short-block repeat gives 62.44 ms on 7.2.3 and
58.78 ms on Nix 10.1, resolving the initial borderline timing flag. Every
large-block output still matches the qualified 7.2.3 bytes exactly. The Nix
short block reproduces the container's independent-reference score below.

The short block changes BF16 outputs when its BLAS algorithm changes. Against
one independent Diffusers block with identical input bytes, relative L2 is
0.010605 on main/7.2.3 and 0.010602 on candidate/10.1; relative maximum error
is 0.010526 on both. This is **not a pass against the existing 0.01 absolute
ceiling**: the stress input exceeds it on the control too. The migration adds
no measured error in this case. The independent source is Diffusers
`80c7ed262aeffbeb43ef13ae04baeb9b84515a69`, PyTorch 2.12.0; it loads only block 0.
The original frozen fixtures and limits are unchanged.

For bounded reproduction, build `minimax_h3_dit_hip_test` and run `--analytic`,
`--profile-1872`, and `--profile-37716-fused` with `GUFO_H3_MODEL_ROOT` set.
`GUFO_H3_DIT_PROFILE_OUTPUT` retains the block output and its exact input bytes
for independent replay. Do not interpret an unset-model skip as qualification.
