# Qwen3-ASR experiments

| Experiment | Decision / evidence |
| --- | --- |
| Fused LDS-staged decode GEMV | Retained; fixed reductions, official IDs unchanged, near measured DRAM ceiling. |
| QKV/gate-up dispatch grouping | Retained; shares inputs/weights without changing each row's dot product. |
| Audio output stays on GPU | Retained; host receives only finite-status scalar; exact transcript. |
| Overlapped audio/text loading | Retained; independent model-owned resources. |
| Model-specific hipBLASLt selection | Rank 6 retained for strongest prefill-boundary agreement, even though ranks 7/8 were slightly faster. Requalify after library changes. |
| Mapped decoder weights | Rejected: lower loading latency but 58% slower resident requests. |
| Official 104-token audio window | Rejected: encoder quality failure and nonfinite long fixture; parity gap remains open. |
| Fused SwiGLU / non-temporal weight loads | Rejected: slower complete request. |
| Handwritten BF16 prefill WMMA | Rejected: slower than hipBLASLt on these shapes. |
| Matrix-core attention port | Not implemented: small fraction of measured request; quantify benefit first. |
| Strict two-pass argmax | Retained; exact ties only, nonfinite rows fail, 49/49 official IDs. Removes the biased 0.25-logit window and a reduction pass. |
| Explicit eager text-attention boundaries | Retained; independent BF16 formulas and matched upstream backend checks. Oracle metadata now records the actual nested implementations. |
| Production route cleanup | Retained; removes six environment switches, unused library attention and scratch; BLAS execution errors now fail the request. |
| Finish normalization within one wave | Retained; fewer barriers with the same sum tree and unchanged logits/tokens. No separate request-throughput gain claimed. |
| Native BF16 convolutions | Retained; LDS im2col and lossless weight packing replace MIOpen. Preserve the teacher's rotated K accumulation and BF16 rounding; exact raw outputs, unchanged encoder errors, 8.2% faster encoder. |

Next: cold-weight GEMM plan selection, launch overhead and long-form quality.
No precision reduction is qualified by these experiments.
