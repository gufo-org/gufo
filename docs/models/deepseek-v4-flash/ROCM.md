# ROCm 10.1 migration

October 23–24, 2026. Flash 0731 mixed IQ2/Q2/Q8 target with its DSpark sidecar
on Strix Halo `gfx1151`. The measured candidate passes the original independent
long-context limits and preserves the 7.2.3 reference vectors. Cross-compiler
logits and generated text are not identical. A subsequently confirmed
same-compiler timing regression remains under investigation before publication.

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
| Independent quantization/router/projection/attention/indexer checks | Pass on both toolchains; existing tolerances retained |
| Compiler register and spill comparison | IQ2 tile uses fewer VGPRs; one existing MMQ specialization spills more |
| AR/DSpark sparse-indexer boundary checks | Pass on both toolchains: DSpark C1 and AR C2, exact full logits |
| Default sampling, replay and request state | Pass: `sampling-defaults` and `state-edges`, including actual C2 execution |
| Matched prefill/decode performance and numerical quality | Six pp2048/tg128 cells and official continuation scores recorded below |
| Independent long-context logits with narrow prefill calls | Original RMSE/cosine/maximum-error limits pass at 4K and 12K on both toolchains |

Compiler regression checks do not resolve the existing target-model parity
limits in [QUALITY.md](QUALITY.md). [BENCHMARKS.md](BENCHMARKS.md) retains its
published toolchain and reference measurements.

## Required changes

LLVM 24 reassociated the IQ2 SwiGLU products differently in scalar and tiled
expert kernels. A shared arithmetic helper preserves the existing rounded
product/reciprocal sequence. The indexer head sum and decode attention's
row-by-row PV accumulation also make their established order explicit.
The MMQ Q8 quantizer now makes the established fused rounding decision
explicit. Clang 24 rounded the product first, turning values just below a
half-integer into ties: two captured activations changed from 33/-38 to 34/-39.
An independent FP64 test covers all three MMQ scale layouts, zero rows and
nearby floating-point values.

On Clang 24 and later, HC/Sinkhorn normalizations retain source order, and
router scores evaluate the existing scalar transcendental formula in FP64
before the final FP32 rounding. Applying these changes to Clang 22 too worsened
its long-context result, so its established arithmetic is preserved.
No global fast-math flags, tensor precision or quality tolerances change.

The test references freeze the historical accumulation order too; otherwise
the new compiler changes both the implementation and its exact-match oracle.
Separate FP64 operator formulas remain independent and pass unchanged.

The initial compiler comparison found no general spill increase. The IQ2
tile-eight kernel drops from 148 to 105 VGPRs and the direct indexer from 21
to 17. One MMQ specialization grows from 312 to 348 scratch bytes; complete
request timings, rather than register counts alone, determine performance.
The final router remains at 52 VGPRs with no scratch or spills on either
compiler; the three MMQ quantizers use fewer VGPRs.

## Numerical and state checks

The earlier #483 review exposed quality failures with 32-token prefill calls
that short official continuations missed. This branch does not import its
rolled MMQ K-loop change. The new `reference --prefill-only --narrow` check
covers the same two points using identical token IDs and independent
`antirez/ds4` revision `6289c516273979173abbc062209a81dd3706b804`.
Final results, with the original limits unchanged:

| Prefix / metric, 32-token calls | Candidate / 7.2.3 | Candidate / 10.1 | Limit |
| --- | ---: | ---: | ---: |
| 4K / full-logit RMSE | 0.816229 | 1.023443 | ≤1.12 |
| 4K / cosine similarity | 0.986692 | 0.979055 | ≥0.979 |
| 4K / maximum absolute logit error | 4.588997 | 4.286486 | ≤5.0 |
| 12K / full-logit RMSE | 0.827573 | 0.393417 | ≤1.12 |
| 12K / cosine similarity | 0.984899 | 0.996594 | ≥0.979 |
| 12K / maximum absolute logit error | 3.619398 | 2.072149 | ≤5.0 |

Both final 7.2.3 vectors are byte-identical to the original control. The final
10.1 vectors also reproduce the preceding corrected build's results.
There is one measurement per final binary, not two executions of the complete
reference harness. Intermediate failures are retained outside Git: the initial
10.1 candidate exceeded the 4K maximum-error limit at 5.184738; Q8 rounding
alone still failed at both depths. HC/router corrections were required too.

The official continuation set uses the same 100 prompts and 2,313 scored
tokens. The final 10.1 binary is compared with the recorded 7.2.3 main control:

| Metric | Main / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: |
| Token-weighted NLL, lower is better | 0.400592 | 0.402618 |
| First-token matches | 56/100 | 59/100 |
| Greedy token matches | 1,998/2,313 | 1,993/2,313 |
| Mean matching greedy prefix | 5.28 | 5.45 |
| Additional smoke token matches | 14/14 | 14/14 |

NLL increases by 0.002025. A paired per-prompt bootstrap gives a 95% interval
of [-0.006285, 0.010277] for this token-weighted difference (20,000 resamples,
seed 20261024). This sample does not establish general quality equivalence.
The initial 7.2.3 candidate repeated the control scores exactly; the final
7.2.3 checks above qualify the subsequently changed MMQ path.

At the ratio-four sparse-indexer boundary, both toolchains pass DSpark C1
and AR C2 with **0/128 token mismatches and zero maximum full-logit error**.
The 2,035-token prompt crosses the 512-entry compressed-cache boundary during
generation; the sidecar is loaded and actual drafts are exercised.

The final production binary passes default/explicit sampling comparisons,
thinking on/off, seeded replay, cached sampler isolation, concurrent sessions,
Chat Completions/Responses, tool-call stops and failed-request recovery.
The focused functional run records 1,118 proposed and 849 accepted draft
tokens. This is candidate correctness evidence; the matched benchmark
controls below provide the timing comparison.

## Focused performance

Canonical HTTP benchmark, greedy/thinking off, pp2048/tg128, one warmed
sample per cell. Depth is the cached prefix. Each cell is **prefill / decode
tok/s**, with identical checkpoint, prompts and server capacity:

| Mode / depth | Main / 7.2.3 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: |
| AR / 0 | 485.02 / 16.31 | 484.38 / 16.30 | 488.04 / 16.09 |
| AR / 16K | 356.92 / 15.52 | 356.21 / 15.51 | 359.62 / 15.35 |
| DSpark prose / 0 | 477.16 / 18.25 | 482.10 / 18.24 | 486.02 / 21.46 |
| DSpark prose / 16K | 356.90 / 15.66 | 355.65 / 15.63 | 359.43 / 15.34 |
| DSpark repetition / 0 | 483.33 / 26.64 | 483.81 / 25.87 | 487.69 / 26.29 |
| DSpark repetition / 16K | 353.74 / 24.04 | 353.45 / 24.01 | 356.96 / 23.41 |

The 7.2.3 candidate preserves every recorded completion, token count and draft
count; none of its 20 requests crosses the 5% and 3 ms prefill/decode gate.
The 10.1 candidate has 0.6–1.9% faster prefill and -2.6% to +17.6% decode
changes in these tg128 cells, all within the slowdown gate. Repetitive output
matches across compilers; ordinary prose and draft acceptance differ, so the
17.6% result is not an isolated kernel speedup.

The first candidate regressed a short synthetic request from 8 to 1 accepted
drafts on 10.1, increasing decode time from 811 to 1,072 ms. The final
corrections restore 8 accepted drafts. A matched fresh-process control uses
the exact 2,051-token prompt and 16 output tokens:

| Build | Prefill | Decode | Total request | Accepted drafts |
| --- | ---: | ---: | ---: | ---: |
| Main / 7.2.3 | 4,507.8 ms | 641.5 ms | 5.184 s | 10 |
| Candidate / 7.2.3 | 4,474.6 ms | 649.1 ms | 5.160 s | 10 |
| Main / 10.1 | 4,424.9 ms | 812.3 ms | 5.267 s | 8 |
| Candidate / 10.1 | 4,416.5 ms | 820.5 ms | 5.268 s | 8 |

No phase in this matched repeat crosses the gate. The first final-candidate
screen was slower on both toolchains: 995/984 ms decode on 7.2.3/10.1.
Those samples remain retained and inconclusive; their cause is unproven.
The pre-existing cross-compiler difference in wording and acceptance also
remains. These short requests are separate from the tg128 table.

The eight-token 16K prefix setup likewise accepts one draft on both unchanged
10.1 main and the candidate, taking about 602/606 ms respectively, versus
559 ms and two accepted drafts on 7.2.3.

However, the unchanged-10.1 control exposed a separate regression on the
first 16K calibration continuation. A capture replay confirms identical
request messages and two valid, different summaries. Accepted drafts drop
from 85 to 8; decode rises from 5,293.9 to 8,350.5 ms (24.18 to 15.33 tok/s),
while prefill changes from 5,843.4 to 5,862.8 ms. This is a real same-compiler
timing increase, even though the candidate remains close to the qualified
7.2.3 control on that prompt. It is not a kernel stall or an invalid-output
speedup. A compiler-specific scalar-order trial passed the same numerical
limits but did not fix this case (15.47 tok/s) and slowed d0 prose to
16.99 tok/s; it was rejected. The measured candidate remains the retained
implementation, with this timing exception unresolved.
