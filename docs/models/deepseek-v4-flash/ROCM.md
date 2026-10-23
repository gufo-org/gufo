# ROCm 10.1 migration

October 23, 2026. Flash 0731 mixed IQ2/Q2/Q8 target with its DSpark sidecar
on Strix Halo `gfx1151`. Focused checks are complete; the short-request
timing exception below prevents an unconditional performance-parity claim.

## Configuration

The control uses the pinned ROCm 7.2.3 Nix toolchain, HIP Clang 22 and GCC
15.3. The candidate uses ROCm 10.1.0, HIP Clang 24 and GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`. Both start from main `ade4f4d7`
and use the repository's matching ROCr queue-memory patch. Builds and
inference run separately on the same machine with identical target/sidecar
weights, prompts, sampling settings and cache history.

The final code is rebased onto `d4eaaad4` (Flash-Next-only changes). Both
production binaries rebuild successfully; these models' HIP object hashes
remain unchanged across the rebase.

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
| Default sampling, replay and request state | Pass: `sampling-defaults` and `state-edges`, including actual C2 execution |
| Matched prefill/decode performance and numerical quality | Six pp2048/tg128 cells and official continuation scores recorded below |

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

The final production binary passes default/explicit sampling comparisons,
thinking on/off, seeded replay, cached sampler isolation, concurrent sessions,
Chat Completions/Responses, tool-call stops and failed-request recovery.
The focused functional run records 1,181 proposed and 827 accepted draft
tokens. This is candidate correctness evidence; the matched benchmark
controls below provide the timing comparison.

## Focused performance

Canonical HTTP benchmark, greedy/thinking off, pp2048/tg128, one warmed
sample per cell. Depth is the cached prefix. Each cell is **prefill / decode
tok/s**, with identical checkpoint, prompts and server capacity:

| Mode / depth | Main / 7.2.3 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: |
| AR / 0 | 485.02 / 16.31 | 484.58 / 16.31 | 490.38 / 16.26 |
| AR / 16K | 356.92 / 15.52 | 355.72 / 15.51 | 360.50 / 15.49 |
| DSpark prose / 0 | 477.16 / 18.25 | 482.64 / 18.25 | 487.47 / 18.66 |
| DSpark prose / 16K | 356.90 / 15.66 | 356.78 / 15.64 | 360.15 / 15.47 |
| DSpark repetition / 0 | 483.33 / 26.64 | 483.52 / 26.62 | 488.89 / 26.47 |
| DSpark repetition / 16K | 353.74 / 24.04 | 353.80 / 24.02 | 356.88 / 23.56 |

The 7.2.3 candidate preserves every recorded completion, token count and draft
count; no request/phase crosses the 5% and 3 ms timing gate. The 10.1
candidate has 0.9–2.2% faster prefill and -2.0% to +2.3% decode changes in
these tg128 cells. Repetitive output matches across compilers; ordinary prose
and draft acceptance can differ with the target's numerical changes.

The untimed synthetic-text warm-up exposed a **short-request regression**.
A focused fresh-process replay of that exact 2,051-token prompt confirms:

| Build | Total request | Decode phase | Accepted drafts / 16 output tokens |
| --- | ---: | ---: | ---: |
| Main / 7.2.3 | 5.162 s | 642 ms | 10 |
| Main / 10.1 | 5.273 s | 811 ms | 8 |
| Candidate / 10.1 | 5.511 s | 1,072 ms | 1 |

The three builds produce different wording. Correcting the arithmetic changes
the target trajectory and reduces draft acceptance in this case; the decode
phase exceeds the 5% and 3 ms gate. The eight-token prefix-setup request also
accepts one fewer draft on 10.1, adding about 37 ms of decode time while its
total request becomes faster. These observations are retained separately from
the tg128 table and are not dismissed as timing noise. No arithmetic fix was
reverted or quality tolerance relaxed to recover a faster token sequence.
