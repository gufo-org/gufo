# Qwen3.8 27B experiments

| Experiment | Decision / qualification |
| --- | --- |
| Quantized verification row groups | Retained per shape; scalar FP32 bits, full target logits and private acceptance/RNG must match. |
| Shared DFlash2 body/context injection | Retained across requests; independent attention, convolution, history and selector state. |
| Partial verification after rejection | Retained with the complete original proposal and unchanged consumed-prefix feedback. |
| BF16 draft gate/up reuse and tiled argmax | Retained; complete head, finite filtering and lowest-ID ties; no extra persistent buffer. |
| Long-context KV packing | Retained in idle FFN scratch with bounded head groups; exact attention output/log-sum-exp, cache bytes unchanged. |
| BF16 target projection reduction | Fixed per-row FP32 order retained for chunk/cache/continued-image equivalence. |
| Register-cached vision softmax | Retained; byte-identical embeddings with the shared Q4/Q8 projector, unchanged reduction order and memory allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical full embeddings, lower latency and 24 MiB less attention scratch. |
| Alternate tiles/waves/pipeline depths (2026-09-19) | Rejected: no release throughput improvement. |
| Dynamic verification chunk/controller alternatives | No new default retained; seeded private-acceptance policy remains. |
| Parallel mapped-weight reads and larger DFlash packing chunks | Retained: faster cold startup, unchanged encoded weights. |
| Compact active/saved recurrence and valid-prefix KV | Retained: unused attention-layer state and future KV rows excluded; Q4/Q8 sampling, rollback, image and disk replay pass. |
| Verification queries grouped by KV partition | Retained: unchanged arithmetic and storage; eight-token attention 2.95× faster at 32K and 3.66× at 64K. Matched d32K C1 HTTP TG improves 15.7% on Q4_K_XL and 20.2% on Q8_K_XL with Q4 DFlash2; shallow TG and PP remain comparable. |
| Split-K two-position KV prefetch and DPP score reductions | Rejected: small single-row component gains did not consistently help eight-row verification. |
| Shallow attention paired KV loads and DPP reduction | Retained: byte-exact; 12.6% less attention time in the complete AR profile. Matched d0 pp2048/tg128 controls improve TG by 1.4–2.6% across Q4/Q8 AR and Q4 DFlash2, with unchanged outputs and acceptance; PP remains comparable. No extra allocation. |
| Wider shallow pipeline, removed broadcast and separate score/value passes | Rejected: worse cold-KV or eight-row costs than the retained two-position pipeline. |
| Additional GEMV format specialization | Not promoted: Q8 cold components unchanged; Q6 gains at most 3% in the cold component test. |

Current focus: **Q4_K_XL, AR, C1**. Profile and push this configuration first;
do not repeat every candidate on other targets, speculative modes or concurrency
levels. Keep the work on one PR branch with incremental, reviewable commits.

Overall target: match llama.cpp generation speed, aiming for a further 10%,
on Q4_K_XL and Q8_K_XL with AR and **Q4_K_M DFlash2** at C1/C2/C4/C6/C8.
Cover both shallow (d0–d16K) and long-context (d32K–d128K) token generation:
screen d0 and d32K, then expand where needed to isolate or qualify the change.
Preserve prompt-processing speed, greedy AR/speculative agreement and sampled
replay. Check cross-engine differences against each engine's AR output.

Iterate with one affected Q4 AR C1 shape and one control. Do not refresh the
full benchmark sweep during exploration. Repeat only to resolve noise or a
failure. Qualify the other configurations after the focused optimization phase,
or earlier only when a specific correctness concern requires it. Remote GPU time
is limited.

Published workload numbers live only in [benchmarks](BENCHMARKS.md); source and
model qualification live in [evaluation](EVALUATION.md).
