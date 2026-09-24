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

Next: resolve the [target arithmetic gaps](QUALITY.md) before claiming parity;
refresh only affected cells in [benchmarks](BENCHMARKS.md). Raw profiles and
abandoned implementations belong outside the working tree, with history in Git.
