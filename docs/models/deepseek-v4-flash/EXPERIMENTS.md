# DeepSeek V4 Flash experiments

| Experiment | Decision / evidence |
| --- | --- |
| Explicit IQ2/Q8 rounding, HC/router and attention/indexer arithmetic | Retained for ROCm 10.1; original 4K/12K narrow-prefill limits pass, 7.2.3 reference vectors remain exact, and C1/C2 sparse-boundary replay passes. [Migration checks](ROCM.md). |
| Bounded compressor stores for resumed prefill | Retained locally: group only up to the next pool/shift boundary, preserving per-row counts and speculative rollback. Exact operator state/output checks pass on 7.2.3/10.1; 16K HTTP prefill improves about 5–6%. |
| C1 vocabulary head cap and device-side Markov chain | Retained locally: preserve the complete support transformer and exact sequential choices, skip unused vocabulary rows and read the completed chain once. FP64 oracle, scalar replay, bounds and end-to-end token/draft counts pass. |
| Reuse packed IQ2 weights across short token tiles | Retained locally: exact integer sums and final FMA; rotating-weight operator probes improve 6–15%, with smaller request-level gains. Exact operator checks pass on 7.2.3 and 10.1; focused final HTTP controls retain text and draft counts. |
| Prefetch attention values in groups of 16 | Retained locally: same row-by-row FMA; 49 exact GPU/independent FP64 cases pass on 10.1. With restored recent-cost backoff, d0 generation improves 21.55→22.18 tok/s and the first 16K continuation 19.52→20.13; the old 24.18 trajectory remains faster. |
| Recent completed-token-cost backoff | Restored: the earlier configuration reached 19.52 tok/s on the corrected continuation with identical text. The final combination passes both builds and 142 functional requests, and reaches 20.13 tok/s. Cache/admission timing flags remain inconclusive. |
| Later confidence/width policy trials | Discarded; only the previously tested recent-cost backoff is restored. |
| Coarse DSpark allocation and asynchronous tensor copies | Rejected: no measurable complete-request improvement. |
| Smaller Q8 launch groups on ROCm 10.1 | Rejected: rotating 16 weight matrices preserves exact outputs, but most gains are only 0–3%; four-row groups slow the 8192-wide scalar/HC kernels by 8–10%. Existing loads already compile to vector memory operations. |
| Exact partial top-k | Retained for wide prefill through 32768 compressed keys; exact ties and bounded fallback. |
| Indexer query packing and shared scratch | Retained; unchanged F16 bytes, scores and full-logit controls through 64K; no persistent KV precision change. |
| Batched speculative projections/attention | Retained; private session state and scalar-equivalent replay. |
| Projection/mHC fusion | Retained only where independent HC/Sinkhorn and model guards pass. |
| Compressed KV and use-sized scratch | Retained; capacity does not allocate a filled context. |
| Extra F16 HC rounding | Rejected: fails official formula oracle and worsens probability comparisons. |
| FP32 compressor/router | Rejected: local arithmetic improvements fail end-to-end continuation/trajectory controls. |
| Paired IQ2 gate/up | Rejected (2026-09-19): exact forms slower; smaller tiles spill and fail exactness. |
| Transposed sparse values | Rejected (2026-09-19): no retained end-to-end gain. |
| Constant pair indices in the 8-row IQ2 gate/up tiles | Retained (2026-10-08): runtime indices kept 144–176 B of per-thread arrays in scratch with clang 22 and 23. The explicit scale product and FMA keep every output bit-identical to the scalar LUT kernel; the tile kernel is 9–10% faster and pp8–pp24 1.7–2.0% faster in isolated PR measurements. |
| Launch bound for the register-cached plain RMS norm | Retained (2026-10-08): the default 1,024-thread bound limited it to 192 VGPRs, and clang 23 spilled one; bit-identical. |

Next: resolve the [target arithmetic gaps](QUALITY.md) before claiming parity;
refresh only affected cells in [benchmarks](BENCHMARKS.md). Raw profiles and
abandoned implementations belong outside the working tree, with history in Git.
