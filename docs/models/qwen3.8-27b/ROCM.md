# ROCm 10.1 migration

October 24, 2026, Strix Halo `gfx1151`. Q4_K_XL/Q8_K_XL targets and the Q4_K_M
DFlash2 draft use the pinned [model identities](artifacts/model-identities.json).
ROCm 7.2.3 uses Nix with Clang 22/GCC 15.3; 10.1.0 uses Clang 24/GCC 15.2 in
`rocm/dev-ubuntu-26.04:10.1.0-full` with the matching repository ROCr patch.
Builds and inference run separately. These results qualify the container
toolchain, not the optional Nix ROCm10 package.

Migration qualification is in progress. Long-context prefill is slower in the
initial 10.1 control; final performance and sampled serving checks are pending.

## Accuracy and state

Different output from ROCm 7.2.3 does not itself establish a quality regression.
Accuracy uses independent operator formulas and the existing numerical limits.
Same-runtime comparisons separately check batching, speculation and cache
consistency. Original-target parity remains limited as described in
[QUALITY.md](QUALITY.md).

| Check | Initial result |
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

## Initial production timing

Canonical HTTP recipe, greedy, thinking off, C1, pp2048/tg128, context capacity
32K. Depth is the cached prefix. Each cell is prefill / decode tok/s.

| Mode / depth | ROCm 7.2.3 | ROCm 10.1 |
| --- | ---: | ---: |
| AR / 0 | 641.66 / 12.37 | 616.58 / 11.99 |
| AR / 16K | 558.83 / 11.71 | 523.85 / 11.38 |
| DFlash2 / 0 | 596.39 / 27.04 | 570.98 / 25.92 |
| DFlash2 / 16K | 516.08 / 22.97 | 477.30 / 22.15 |

The 16K prefill differences exceed the 5% and 3 ms gate. They remain unresolved
in this initial control; no unconditional performance-parity claim is made.
Compiler resource reports identify additional register pressure and spills;
profiles must establish which specializations affect the measured requests.
Profiled runs are diagnostic, not replacement benchmark numbers.

Raw results and profiles remain under
`gufo-models-rocm101-20261023/port-models` on the development host.
[BENCHMARKS.md](BENCHMARKS.md) retains its published measurements.
