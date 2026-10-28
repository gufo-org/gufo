# ROCm 10.1 migration

October 24, 2026, Strix Halo `gfx1151`. Q4_K_XL/Q8_K_XL targets and the Q4_K_M
DFlash2 draft use the pinned [model identities](artifacts/model-identities.json).
ROCm 7.2.3 uses Nix with Clang 22/GCC 15.3. ROCm 10.1.0 was first checked with
Clang 24/GCC 15.2 in `rocm/dev-ubuntu-26.04:10.1.0-full`, then built through
`nix build` with the cached Clang 24/GCC 16.2 toolchain and the matching
repository ROCr patch. Source `65bf6798` includes main `67d1f6ac`; this model's kernels are unchanged
from the independently checked `6d43e786`.
Builds and inference run separately.

The cached Nix build passes the affected numerical and serving contracts.
Canonical timing results are below. The separate calibration prefill is
slower across toolchains, but its matched Nix main/candidate control passes.

## Accuracy and state

Different output from ROCm 7.2.3 does not itself establish a quality regression.
Accuracy uses independent operator formulas and the existing numerical limits.
Same-runtime comparisons separately check batching, speculation and cache
consistency. Original-target parity remains limited as described in
[QUALITY.md](QUALITY.md).

| Check | Result |
| --- | --- |
| Q4/Q8 target prefill and selected final-row execution | Pass on 10.1, including 2,048 rows and a d8K continuation |
| Target full-logit fingerprints, short histories | 54/54 rows identical across compilers for each quantization |
| Quantized FP16 projections, WMMA attention, Q8 GEMV and W8A8 operators | Independent reference checks pass on 10.1 |
| Sampling and speculative verification | 216 policies pass; Q4 full replay and Q8 concurrent C2–C8 checks pass |
| Default sampling, thinking, images, cancellation and RAM/disk cache | Focused Q4 DFlash2 functional checks pass correctness |
| Fused SSM FP16 output | Initial 10.1 failure; explicit first square/add contraction fixes the focused operator check |

Large FP16-prefill logit vectors and one 16K greedy completion differ between
compilers. AR and DFlash2 agree within each compiler in the measured histories.
The SSM failure was a separate same-runtime inconsistency: one of 589,824
FP16 outputs crossed a rounding boundary. An independent FP64 formula favored
the separate calculation for that value. The fix makes the fused calculation's
first square/add explicit, retaining the existing reduction and state formats.
Original exact checks remain; the test also evaluates the independent formula.

## Production timing

Canonical HTTP recipe, greedy, thinking off, C1, pp2048/tg128, context capacity
32K. Depth is the cached prefix. Each cell is prefill / decode tok/s.

| Mode / depth | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| AR / 0 | 638.56 / 12.38 | 620.04 / 11.99 |
| AR / 16K | 559.00 / 11.72 | 537.14 / 11.38 |
| DFlash2 / 0 | 596.40 / 27.04 | 570.19 / 25.92 |
| DFlash2 / 16K | 512.95 / 22.97 | 490.24 / 23.44 |

Every canonical cell stays within the 5% and 3 ms timing gate. The earlier
10.1 16K prefill samples were 523.85 tok/s AR and 477.30 tok/s DFlash2.
The retained WMMA specialization bounds register pressure under Clang 24;
packed attention avoids additional spills, and ragged tails retain their
arithmetic. Q4/Q8 attention and complete DFlash2 checks pass.

A separate 2,055-token calibration prefill took 3,077.94/3,236.59 ms in AR and
3,319.59/3,521.53 ms in DFlash2 (7.2.3/Nix 10.1), exceeding that gate. Profiles
attribute most of this difference to quantized target GEMMs, with an additional
BF16 draft feature-projection cost. GPU work accounts for the delay. Diagnostic
profiles and rejected scheduling/tile experiments are retained separately;
none substitutes for the unprofiled timing samples.

The matched Nix 10.1 control on unchanged main `67d1f6ac` processes the same
2,055-token calibration in 3,240.25 ms AR / 3,543.41 ms DFlash2, versus
3,272.14 / 3,526.33 ms on the candidate. Every measured prefill, decode,
first-token and maximum-token-gap phase passes the gate. DFlash2's maximum
gap is 142.79 ms on main and 143.35 ms on the candidate: the earlier
cross-toolchain gap is not introduced by this change.

The Nix functional run passes default sampling, thinking, image input,
continuation, cancellation and RAM/disk restart, with 1,620 proposed and 869
accepted drafts, matching the prior corrected 10.1 run. The final matched
sampling-only main/candidate runs also agree on outputs and all 783 proposed /
416 accepted drafts. Their raw timing report retains 1,396 passes and 16
inconclusive measurements; tolerances are unchanged.

A focused reverse-order repeat resolves the first-request flag: prefill is
354.17/352.59 ms and decode 126.31/126.79 ms (main/candidate), with matching
first-token/gap timing. Remaining concurrent flags include swapped admission
order and a width-one versus width-two replay. The already measured width-two
thinking cohort takes 309.82/212.25 ms on main and 310.49/212.91 ms on the
candidate for its two requests. Different physical cohorts remain unqualified;
raw flags are preserved rather than averaged or relabeled as passes. Earlier
cross-run cache timings likewise do not establish unconditional latency parity.

Raw results and profiles remain under
`gufo-models-rocm101-20261023/port-models` on the development host.
[BENCHMARKS.md](BENCHMARKS.md) retains its published measurements.
