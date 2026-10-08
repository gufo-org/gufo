# DeepSeek V4 Flash experiments

| Experiment | Decision / evidence |
| --- | --- |
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
