# DeepSeek V4 Flash experiments

| Experiment | Decision / evidence |
| --- | --- |
| Explicit IQ2/Q8 rounding, HC/router and attention/indexer arithmetic | Retained for ROCm 10.1; original 4K/12K narrow-prefill limits pass, 7.2.3 reference vectors remain exact, and C1/C2 sparse-boundary replay passes. [Migration checks](ROCM.md). |
| Bounded compressor stores for resumed prefill | Retained locally: group only up to the next pool/shift boundary, preserving per-row counts and speculative rollback. Exact operator state/output checks pass on 7.2.3/10.1; 16K HTTP prefill improves about 5–6%. |
| C1 draft head cap and recent completed-token cost | Retained locally: keep the full support transformer, omit unused vocabulary rows and reset the backoff window on retries. Shallow 7.2.3 prose improves 18.25→22.92 tok/s; the flagged 10.1 continuation improves 15.33→19.52 tok/s with identical corrected output, but still trails its older 24.18 tok/s trajectory. |
| Four-cycle C1 cost window without hysteresis | Rejected: shallow prose fell 21.55→20.30 tok/s. Eight cycles retain useful drafting through short poor runs; sustained unprofitable requests still back off. |
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
