# ROCm 10.1 migration

October 24, 2026. Flash 0731 mixed IQ2/Q2/Q8 target and DSpark on Strix Halo
`gfx1151`. Independent numerical checks pass their original limits, but the
migration is **not ready to claim no performance regression**: one ROCm 10.1
continuation still trails its original 24.18 tok/s result. Both versions produce
valid, different summaries.

## Configuration

The control uses pinned ROCm 7.2.3, HIP Clang 22 and GCC 15.3 through Nix.
ROCm 10.1.0 was first checked with HIP Clang 24/GCC 15.2 inside
`rocm/dev-ubuntu-26.04:10.1.0-full`, then built through `nix build`
with cached Clang 24/GCC 16.2 and the matching repository ROCr patch.
Source `65bf6798` includes main `67d1f6ac`; this model's kernels are unchanged
from the independently checked `6d43e786`. Builds and inference run separately.
Model revisions, prompts, sampling, cache history and hardware stay matched.
Default Nix still uses 7.2.3.

## Migration steps

| Step | Status |
| --- | --- |
| Production builds on 7.2.3 and 10.1 | Restored policy and final kernels pass both |
| Quantization/router/projection/attention/indexer checks | Original limits pass on both toolchains |
| 4K/12K narrow-prefill logits against an independent teacher | Pass on both toolchains; values below |
| Sparse-indexer boundary replay | DSpark C1 and AR C2: zero token/full-logit differences |
| Default sampling, cache, cancellation and restart | Final 142-request check passes correctness; timing limitations below |
| Same-compiler performance | Flagged continuation remains slower than original 10.1 main |

## Numerical corrections

ROCm 7.2.3 output is a regression control, not the accuracy oracle for 10.1.
Cross-compiler byte or token differences alone do not fail qualification.
Accuracy uses the independent FP64 operators and pinned target teacher below.
Exact replay checks instead protect same-runtime batching, speculation and
cache consistency. They cannot establish target-model accuracy.

LLVM 24 reassociated scalar and tiled IQ2 SwiGLU products differently. The
shared helper preserves their established rounded product/reciprocal sequence.
Indexer head sums and attention PV accumulation likewise retain their original
order. The MMQ Q8 quantizer explicitly preserves the fused rounding decision:
captured values changed from 33/-38 to 34/-39 when the compiler rounded the
product before deciding a tie. Independent FP64 checks cover all three scale
layouts, zero rows and nearby floating-point values.

HC/Sinkhorn order and FP64 router transcendental evaluation are restricted to
Clang 24 and later. Applying those corrections to Clang 22 worsened its
long-context result, so its arithmetic remains unchanged. Global fast-math,
tensor precision and quality tolerances do not change. Historical GPU references
freeze their accumulation order; separate FP64 formulas remain independent.

The original 4K/12K narrow-prefill failure from #483 is covered without importing
its rolled MMQ K-loop. The teacher is independent `antirez/ds4` revision
`6289c516273979173abbc062209a81dd3706b804`, using identical token IDs and
32-token prefill calls. The corrected build records:

| Prefix / metric, 32-token calls | Candidate / 7.2.3 | Candidate / 10.1 | Limit |
| --- | ---: | ---: | ---: |
| 4K / full-logit RMSE | 0.816229 | 1.023443 | ≤1.12 |
| 4K / cosine similarity | 0.986692 | 0.979055 | ≥0.979 |
| 4K / maximum absolute logit error | 4.588997 | 4.286486 | ≤5.0 |
| 12K / full-logit RMSE | 0.827573 | 0.393417 | ≤1.12 |
| 12K / cosine similarity | 0.984899 | 0.996594 | ≥0.979 |
| 12K / maximum absolute logit error | 3.619398 | 2.072149 | ≤5.0 |

The initial 10.1 candidate exceeded the independent 4K maximum-error limit
(5.184738 versus 5.0); its 12K RMSE was 0.619958. The retained corrections
reduce those errors to 4.286486 and 0.393417 respectively. They do not improve
every metric: 4K RMSE increases from 0.977081 to 1.023443, still within its
original limit. The final 10.1 result is substantially closer to the teacher
than 7.2.3 at 12K, without reproducing 7.2.3's values.

All four vectors repeated byte-for-byte after compressor-store batching. Both
7.2.3 vectors also match the original 7.2.3 control. Intermediate failures are
retained outside Git: Q8 rounding alone was insufficient; HC/router corrections
were required too. These checks do not resolve the pre-existing target-model
parity limits in [QUALITY.md](QUALITY.md).

The official continuation set contains 100 prompts and 2,313 scored tokens:

| Metric | Main / 7.2.3 | Candidate / 10.1 |
| --- | ---: | ---: |
| Token-weighted NLL, lower is better | 0.400592 | 0.402618 |
| First-token matches | 56/100 | 59/100 |
| Greedy token matches | 1,998/2,313 | 1,993/2,313 |
| Mean matching greedy prefix | 5.28 | 5.45 |
| Additional smoke token matches | 14/14 | 14/14 |

The token-weighted NLL delta is +0.002025, with paired 95% bootstrap interval
[-0.006285, 0.010277] (20,000 resamples, seed 20261024). This sample does not
establish general quality equivalence. Cross-compiler logits and text differ.

The cached Nix 10.1 rerun reproduces both narrow-prefill vectors and every
score in the official continuation/smoke report from the corrected container
build. Its focused functional run also preserves all 1,175 proposed and 785
accepted drafts.

## Execution optimizations

The completed-token-cost backoff counts the guaranteed target anchor, uses an eight-cycle recent
window and resets that window after retries. Clearly unprofitable starts can
still back off after four cycles. Acceptance rules, sampling and the 64-token
retry interval stay unchanged.

- Group resumed compressor stores only up to the next pool/shift boundary.
  Pooling, normalization, RoPE, quantization and speculative rollback stay in
  their original order. Independent state tests cover ratios 4/128 and tails.
- Project only requested C1 vocabulary rows, preserving the complete noncausal
  support transformer and all selected logits.
- Decode IQ2 signed weight bytes once and reuse them across up to four token
  rows. Integer sums and the final FP32 FMA remain identical.
- Keep sequential greedy Markov choices on the GPU and read the completed
  chain once. A new bounded operator test checks the independent FP64 formula,
  scalar replay, tie ordering and invalid ranges.
- Prefetch 16 attention value rows before consuming them in the original
  FP32 FMA order. Existing independent attention tests now cover 49 empty,
  short, wrapped, masked and indexed cases, including partial groups.

The kernel/operator checks pass exactly on both 7.2.3 and 10.1. Focused attention measurements
use rotating KV buffers; gains vary by shape, so they do not substitute for
complete-request timing. Coarse support allocations and asynchronous tensor
copies produced no measurable request gain and were discarded.

## Focused performance

Canonical HTTP history: thinking off, greedy, C1, 128 output tokens; the short
warmup has 16. The 16K continuation restores 16,297 cached tokens and processes
2,045 more. The restored recent-cost policy is held constant in this final
ROCm 10.1 comparison against `6ff84eee`. Messages, output bytes, draft counts
and cache work match:

| Request / phase | Before kernel changes | Final candidate |
| --- | ---: | ---: |
| Warmup prefill | 4,438.7 ms | 4,459.4 ms |
| Warmup generation | 19.60 tok/s | 19.99 tok/s |
| d0 prefill | 4,208.3 ms | 4,185.5 ms |
| d0 generation | 21.55 tok/s | 22.18 tok/s |
| First 16K continuation prefill | 5,561.9 ms | 5,562.4 ms |
| First 16K continuation generation | 19.52 tok/s | 20.13 tok/s |

Both production builds pass. The original ROCm 10.1 continuation remains a
separate, slower comparison: it reached 24.18 tok/s with 85 accepted drafts;
the corrected target with main's backoff reached 15.33 with eight accepted.
Restoring recent-cost backoff and adding the kernel changes reaches 20.13
with 69 accepted drafts. The old and corrected summaries are both valid.
The final rebased Nix build repeats this first continuation at 20.12 tok/s
with the same 69 accepted drafts; d0 remains 22.18 tok/s. No further policy
tuning is included. [BENCHMARKS.md](BENCHMARKS.md) retains
its published measurements; this was a focused repeat, not a new full sweep.

The earlier corrected-build profile was 93.2% GPU-busy during this slow decode.
Large totals inside blocking HIP calls overlap GPU execution and must not be
added to kernel time as separate waiting overhead. Prefill contained 126,790
compressor-store launches, motivating bounded grouping. These traces do not
prove every possible scheduling pause is eliminated.

The final build passes 142 functional requests: default sampling, seed replay,
thinking, Chat/Responses, cancellation, continuation and disk restart, including
actual C2 execution and 1,175 proposed/785 accepted drafts. Startup/restart take
14.8/15.4 s with the fixture fingerprint cache warm.

Timing qualification remains limited. The automatic comparison rejects changed
parent loader-path and lockfile metadata. A separate audit verifies that both
server wrappers set the same loader path and that only the unused optional
Nix ROCm10 input changed; both binaries use the retained container toolchain.
That audited comparison passes 2,002 measurements, with 26 flags around cache
snapshots/restores, concurrent admission and token gaps. Outputs and cache work
match. Raw reports and flags remain unchanged; this is not a blanket latency
pass.
