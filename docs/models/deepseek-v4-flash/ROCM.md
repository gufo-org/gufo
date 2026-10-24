# ROCm 10.1 migration

October 23–24, 2026. Flash 0731 mixed IQ2/Q2/Q8 target with its DSpark sidecar
on Strix Halo `gfx1151`. The measured candidate passes the original independent
long-context limits and preserves the 7.2.3 reference vectors. Cross-compiler
logits and generated text are not identical. A subsequently confirmed
same-compiler timing regression remains under investigation before publication.
The latest local changes improve resumed prefill and recover useful DSpark
drafting, but do not yet close that timing exception.

## Configuration

The control uses the pinned ROCm 7.2.3 Nix toolchain, HIP Clang 22 and GCC
15.3. The candidate uses ROCm 10.1.0, HIP Clang 24 and GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`. Both start from main `ade4f4d7`
and use the repository's matching ROCr queue-memory patch. Builds and
inference run separately on the same machine with identical target/sidecar
weights, prompts, sampling settings and cache history.

The matched measurements use source based on `d4eaaad4` (Flash-Next-only
changes); both production builds succeeded with unchanged model HIP objects
across that rebase. Main `8b52b42c` subsequently changes the optional Nix ROCm
10 input and development instructions, with no inference-source or default
7.2.3 toolchain changes. The container results do not qualify that new Nix
package independently.

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

Resumed prefill now groups compressor stores up to the next pool/shift
boundary. Pooling, normalization, RoPE, quantization and emitted row counts
retain their original order; speculative verification still captures every
intermediate rollback state. A new operator test checks ratios 4/128,
unaligned starts, ragged tails, exact state contents and byte-identical
compressed outputs. Spans crossing a pool boundary are rejected.

C1 drafting projects only the requested vocabulary rows, while retaining the
complete noncausal support transformer. Its backoff policy now counts the
guaranteed target token and evaluates recent completed-token cost. An
eight-cycle window avoids repeatedly penalizing a request for its first poor
drafts; clearly unprofitable probes can stop after four cycles. Retries reset
the cost window. Decisions use token counts, not runtime clocks; acceptance,
sampling, the 64-token retry interval and the concurrent policy are unchanged.
Measured draft/verify/commit costs for tails 1–5 were approximately
145/165/185/210/230 hundredths of an AR step at d0/d16K.

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
10.1 vectors also reproduce the preceding corrected build's results. The latest
compressor optimization repeats all four vectors byte-for-byte, in one run per
compiler; the independent teacher vectors are reused. Intermediate failures
are retained outside Git: the initial
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

The preceding numerical build passed `sampling-defaults` and `state-edges`,
including tool-call stops and failed-request recovery. The latest production
binary passes `sampling-defaults`, `long-context` and `cache`: 142 recorded
requests, thinking on/off, seeded replay, cached sampler isolation, Chat
Completions/Responses, cancellation, continuation and restoration after a
server restart. Actual C2 execution is observed, with 1,175 proposed and 785
accepted drafts across the run. The latest 10.1 native boundary replay also
passes both DSpark C1 and AR C2 with zero token/logit differences.
These are candidate correctness checks; matched benchmark controls provide
the timing comparison.

With a fresh fixture-owned disk cache, startup takes 65.2 seconds, including
50.3 seconds to fingerprint target/sidecar artifacts. Restart takes 15.9
seconds and reuses those identities. This existing cold identity work runs
before readiness and is separate from request-time prefill/decode.

## Focused performance

Canonical HTTP benchmark, greedy/thinking off, pp2048/tg128, one warmed
sample per cell. Depth is the cached prefix. Each cell is **prefill / decode
tok/s**, with identical checkpoint, prompts and server capacity:

| Mode / depth | Main / 7.2.3 | Candidate / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: | ---: |
| AR / 0 | 485.02 / 16.31 | 484.27 / 16.30 | 487.79 / 16.08 |
| AR / 16K | 356.92 / 15.52 | 377.89 / 15.52 | 378.69 / 15.32 |
| DSpark prose / 0 | 477.16 / 18.25 | 482.55 / 22.92 | 486.05 / 21.56 |
| DSpark prose / 16K | 356.90 / 15.66 | 377.61 / 15.31 | 379.65 / 16.05 |
| DSpark repetition / 0 | 483.33 / 26.64 | 483.06 / 26.69 | 487.04 / 26.38 |
| DSpark repetition / 16K | 353.74 / 24.04 | 373.62 / 24.06 | 376.50 / 23.46 |

The latest resumed-prefill change improves 16K prefill by about 5–6% relative
to the preceding candidate on each compiler. AR decode remains within the
5% timing margin. Recent-cost backoff improves 7.2.3 d0 prose generation from
18.25 to 22.92 tok/s, with identical output; draft counts intentionally change.
Repetitive output matches across compilers. Ordinary prose and draft
acceptance differ, so these cross-compiler decode rates do not isolate kernel
speed. The calibration requests and first-request checks below remain part of
qualification; passing the six table cells alone is insufficient.
The 7.2.3 phase screen covers all 20 benchmark/setup requests with unchanged
prompt/prefill/output token counts. Its only flag is the short first-request
decode sample discussed below; the matched repeat does not reproduce it.

The first candidate regressed a short synthetic request from 8 to 1 accepted
drafts on 10.1, increasing decode time from 811 to 1,072 ms. The final
corrections restore 8 accepted drafts. A matched fresh-process control uses
the exact 2,051-token prompt and 16 output tokens:

| Build | Prefill | Decode | Total request | Accepted drafts |
| --- | ---: | ---: | ---: | ---: |
| Main / 7.2.3 | 4,540.7 ms | 641.1 ms | 5.223 s | 10 |
| Candidate / 7.2.3 | 4,489.6 ms | 638.3 ms | 5.169 s | 10 |
| Main / 10.1 | 4,404.0 ms | 812.8 ms | 5.254 s | 8 |
| Candidate / 10.1 | 4,410.6 ms | 815.3 ms | 5.263 s | 8 |

No phase in this matched repeat crosses the gate. Worst token intervals are
128.6→127.4 ms on 7.2.3 and 224.6→223.9 ms on 10.1; snapshot time also
does not regress. The first final-candidate
screen was slower on both toolchains: 995/984 ms decode on 7.2.3/10.1.
The latest benchmark also contains short-request samples of 725/865 ms.
Those samples remain retained and inconclusive; the matched repeat does not
reproduce them, and their cause is unproven.
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

The store batching and recent-cost policy recover the same corrected summary
at **19.52 tok/s**: 69 accepted drafts from 141 proposals, 6,556.3 ms decode
and 5,669.5 ms prefill. The preceding numerical candidate needed 8,350.5 ms
decode and 5,862.8 ms prefill. The latest worst token interval is 173.4 ms.
The remaining difference from unchanged 10.1 main is 1,262.5 ms decode
(19.3% fewer tok/s), so this is not a no-regression result.

A separate unchanged-main AR control uses identical messages and generates
a different summary from unchanged-main DSpark, already at the first token.
AR also restores one fewer cached token. This establishes that the divergence
precedes draft verification; it does not establish a verifier defect or make
the faster DSpark result invalid.

The retained profiler trace shows 93.2% GPU-busy time after the first token of
the slow corrected continuation. Most of the delay is additional target
decoding after premature backoff. Before the first token it records 126,790
compressor-store launches, motivating the bounded grouping above. That trace
does not prove that every individual scheduling pause has been eliminated.
