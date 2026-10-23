# ROCm 10.1 migration

October 23, 2026. Flash 0731 mixed IQ2/Q2/Q8 target with its DSpark sidecar
on Strix Halo `gfx1151`. Migration qualification is in progress.

## Configuration

The control uses the pinned ROCm 7.2.3 Nix toolchain, HIP Clang 22 and GCC
15.3. The candidate uses ROCm 10.1.0, HIP Clang 24 and GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`. Both start from main `ade4f4d7`
and use the repository's matching ROCr queue-memory patch. Builds and
inference run separately on the same machine with identical target/sidecar
weights, prompts, sampling settings and cache history.

The optional Nix entry points are `nix build .#rocm10` and
`nix develop .#rocm10`; the measurements here qualify the container build.
The default Nix toolchain remains 7.2.3.

## Migration steps

| Step | Status |
| --- | --- |
| Production builds on 7.2.3 and 10.1 | Complete |
| Independent projection/attention/indexer and sampling checks | Pass on both toolchains; existing FP64 tolerances retained |
| Compiler register and spill comparison | IQ2 tile uses fewer VGPRs; one existing MMQ specialization spills more |
| AR/DSpark sparse-indexer boundary checks | Pass on both toolchains: DSpark C1 and AR C2, exact full logits |
| Default sampling, replay and request state | In progress |
| Matched prefill/decode performance and numerical quality | In progress |

Compiler regression checks do not resolve the existing target-model parity
limits in [QUALITY.md](QUALITY.md). [BENCHMARKS.md](BENCHMARKS.md) retains its
published toolchain and reference measurements.

## Required changes

LLVM 24 reassociated the IQ2 SwiGLU products differently in scalar and tiled
expert kernels. A shared arithmetic helper preserves the existing rounded
product/reciprocal sequence. The indexer head sum and decode attention's
row-by-row PV accumulation also make their established order explicit.
No global fast-math flags, model precision or quality tolerances change.

The test references freeze the historical accumulation order too; otherwise
the new compiler changes both the implementation and its exact-match oracle.
Separate FP64 operator formulas remain independent and pass unchanged.

The initial compiler comparison found no general spill increase. The IQ2
tile-eight kernel drops from 148 to 105 VGPRs and the direct indexer from 21
to 17. One MMQ specialization grows from 312 to 348 scratch bytes; complete
request timings, rather than register counts alone, determine performance.

## Numerical and state checks

On ROCm 7.2.3, all recorded continuation scores and choices are identical to
main. The independent official continuation set uses the same 100 prompts
and 2,313 scored tokens:

| Metric | Main / 7.2.3 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: |
| Token-weighted NLL, lower is better | 0.400592 | 0.400592 | 0.401110 |
| First-token matches | 56/100 | 56/100 | 57/100 |
| Greedy token matches | 1,998/2,313 | 1,998/2,313 | 1,991/2,313 |
| Mean matching greedy prefix | 5.28 | 5.28 | 5.59 |
| Additional smoke token matches | 14/14 | 14/14 | 14/14 |

Cross-compiler logits are not bit-identical. The small NLL change does not
show a clear directional quality loss on this sample: the paired per-prompt
mean delta is -0.000221, with a 95% bootstrap interval of
[-0.007233, 0.006612] (20,000 resamples). This does not establish general
quality equivalence or resolve the target's existing upstream-parity limits.

At the ratio-four sparse-indexer boundary, both toolchains pass DSpark C1
and AR C2 with **0/128 token mismatches and zero maximum full-logit error**.
The 2,035-token prompt crosses the 512-entry compressed-cache boundary during
generation; the sidecar is loaded and actual drafts are exercised.
