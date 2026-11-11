# Flash-Next staging comparison

The November 11, 2026 update from HRX `d8345774e3` to `8ebcdb6b70` adds
NPU conversion correctness and packed-weight documentation. It establishes no
repeatable Flash-Next GPU throughput gain.

## Matched native measurements

Same Gufo `9612db2e5` executable, Flash-Next UD-Q4_K_XL, autoregressive,
depth zero, seed 1. Each process runs `gufo bench -p 512,2048 -n 128 -r 3
--validate-prefill 512`. Model loading is excluded. Each process starts after
cooling to at most 38 C and 60 seconds idle on the Performance power profile,
with no other model or compiler running. Both variants use the pinned Nix
ROCm 10.1 toolchain. This is native benchmark throughput, not HTTP latency.

All values below are tok/s, mean ± standard deviation over three repetitions.
The native CLI emits aggregate statistics, not individual repetition timings.
Every emitted result and process boundary is retained in the
[comparison artifact](flash-next-staging.json); original benchmark output is
also retained in the [raw logs](flash-next-staging/).

| Pair | Phase | Baseline | Candidate | Candidate change |
| --- | --- | ---: | ---: | ---: |
| A | pp512 | 1255.71 ± 8.76 | 1168.64 ± 22.59 | -6.93% |
| A | pp2048 | 1824.78 ± 1.59 | 1821.75 ± 10.70 | -0.17% |
| A | tg128 | 30.09 ± 0.16 | 29.80 ± 0.90 | -0.96% |
| B | pp512 | 1272.40 ± 0.74 | 1269.78 ± 2.06 | -0.21% |
| B | pp2048 | 1829.32 ± 0.60 | 1832.49 ± 3.62 | +0.17% |
| B | tg128 | 30.34 ± 0.17 | 30.35 ± 0.16 | +0.03% |

The first candidate pp512 result was 6.93% slower and triggered the second
alternating pair. The repeat measured -0.21% at pp512, +0.17% at pp2048 and
+0.03% at tg128. The first pp512 slowdown did not reproduce; its timing evidence
remains visibly inconclusive. Neither pair demonstrates a repeatable gain.
The two pairs are not combined into one percentage.

## Correctness and runtime identity

All four processes pass the existing 512-token prefill check with the same
reported values: finite outputs, scalar winner rank 1, RMSE 0.520053,
cosine 0.991351, maximum error 3.81343. This compares batched prefill against
one-token-at-a-time GPU execution; it is not independent FP32 model validation
or a full-logit bitwise comparison.

A separate correctness-only run (`-p 1 -n 128 -r 1 --seed 1 --verbose`)
produces the same 128 generated token IDs under both runtimes, SHA-256
`4f097b22bb50fbf9121ab9de8897f03e65fb693ce66ade838e1b52671ed21984`. Both exit successfully
and report no drafts; these additional timings are excluded from the comparison.

Both Nix HIP libraries are byte-for-byte identical (3,041,320 bytes), SHA-256
`a51b02a3074840c1aca88d08394e1ae231d8f8e1f936ca7c57979fea9c8c3838`.
The shared Gufo executable's SHA-256 is
`5be51583d589d1df3a007898b22b12edac323659888f8e16706287b4364c5ca1`.
Loader traces confirm each wrapper loads its intended HRX library. The generic
`ROCm (HIP)` label in native benchmark output does not identify the actual
runtime library. Both wrappers pass `diagnose`; the standard updated
`.#gufo-hrx` package builds. The dependency output hash remains unchanged.

The NPU sign fix cannot improve Flash-Next execution through the identical GPU
library. Its demonstrated benefit remains correct F16/FP8 signs on the NPU
and the opportunity to remove LFM3's sign-repair workaround. The packed-weight
reference changes no executable behavior. No upstream performance PR is
justified by these Flash-Next measurements.
