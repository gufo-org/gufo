# Inkling 2 Mini experiments

Rejected or neutral changes, measured November 22–26, 2026 on Strix Halo
against the build they modified. Every variant is exact unless noted. Kept
changes are in Git history; current speeds are in [benchmarks](BENCHMARKS.md).

## Prefill

| Variant | Result |
| --- | --- |
| 128-token tiles in the 4-wave dense GEMM, or from 64-token ubatches | slower (occupancy; pp64/pp128 −7%); 128-token tiles kept from 256 tokens |
| MoE gate/up: double-buffered or prefetched weight tiles, register rings, 8 waves per SIMD, 256-token items, padding tiles computed | 2 to 30% slower or unchanged: the WMMA + LDS loop is the bound (5.4 ms of 6.0 with loads removed) |
| VGPR caps, straight-line WMMA bodies, 16×128 tiles per wave | equal or slower except where kept (down and K = 2304 dense) |
| Five attention projections, or q and gate, in one launch | pp2048 equal or −0.8% |
| Controller gate kernels with other tiles, float4 LDS or register prefetch | slower: `k_gate_p64` already moves its bytes at the DRAM ceiling |
| Attention-residual mix from one row read, or later query tiles first | slower (fewer active waves) |
| Prefill attention with register-prefetched K/V tiles | spills; pp32768 −5% |
| 32-query attention blocks | pp32768 −11% |
| 128-query blocks on window visits too | equal within noise |
| Accumulator rescale skipped in window and near attention steps too, as in the far tiles | equal within noise |
| Message mixes as their own side-stream launch | pp512 −1.5%; folded into the message mix from 1,024-token ubatches |
| Router and controller gates with smaller token tiles, or blocks reordered so jobs sharing x run together (Nov 25) | equal or up to 2.5× slower: the gates are LDS-bandwidth bound, not DRAM or occupancy bound |
| Gate GEMMs with wave-uniform scalar weights, one token per lane (Nov 25) | 2 to 10× slower |
| Controller apply + norm below 512-token ubatches (Nov 25) | pp64 −1%: too few blocks; the separate kernels are kept there |
| MoE items of ≤64 tokens in a lean kernel beside the full items (Nov 25) | gate/up −2.4% only: small items are bound by their expert weight reads and barely overlap |
| MoE gate/up with activations staged in LDS, a two-deep weight prefetch, or 4 waves of 32 tokens × 128 rows (Nov 25) | full items up to +2.6%, overall 2 to 19% slower |
| MoE gate/up with each wave dequantising its own weight tile in registers, no LDS or barriers (Nov 26) | 15 to 23 TFLOPS against 29: the wave then loads four times the activation tiles per WMMA, and the weight rows no longer arrive ahead of their use |
| MoE gate/up waves of 64 tokens × 64 rows, or 64 tokens × 32 rows (Nov 26) | 64 × 64 fills all 256 VGPRs (full items 32.8 TFLOPS against 35.6); 64 × 32 doubles the activation loads (29.2) |
| MoE gate/up with 32-k LDS slabs (10 KiB per block), another LDS row stride, activations loaded per 16 k, or the `iterative-ilp`, `max-ilp` and `iterative-minreg` schedulers (Nov 26) | 3 to 9% slower or equal; a 64-element row stride (no padding) is 2.4 times slower |
| Expert items of at most 16 tokens in a kernel of their own, with or without LDS (Nov 26) | equal to 27% slower: an item's cost is its 18 LDS loads of the expert's weights, whatever its tokens |

## Decode and verification

| Variant | Result |
| --- | --- |
| Quantising each activation once (`k_quant`) instead of per GEMV block | tg −3%: about 240 more launches per token |
| A block's first weight rows loaded before it quantises its input, in the expert gate/up and down GEMVs (Nov 26) | tg −1.6% (22.03 against 21.68 ms per token): the decode GEMVs already read their weights at 197 to 238 GB/s, so the prologue is not their bound |
| Two row iterations per wave, or 16 waves per block, in the expert GEMVs | equal or −2% |
| Expert weights held in registers, or MoE inputs read from L2 | 2% slower per token |
| Predictive expert prefetch (60% top-1 reuse between tokens) | 0.5% slower: the reads compete with latency-bound kernels |
| Mix variants (prologue in LDS, all loads first) | slower |
| MoE gate/up and down in one launch | slower from 4 rows |
| Grouped expert kernels: more pairs per wave or block, other wave counts, LDS staging, DPP lane moves, VGPR caps | 5 to 140% slower; kept for 6+ rows of one session and 8 sessions |
| Dense int8 WMMA from 8 rows for one-input projections or the output projection | within noise |
| Verification attention with the split reduction in its last block | +0.3 ms per step |
| Controller gates of 2 to 8 tokens per block | verification +0.1 to +0.3 ms |

## DSpark

| Variant | Result |
| --- | --- |
| The full 200,058-row LM head for proposals | −9% (same acceptance as the 65,536-row head) |
| A 32,768-row proposal head | acceptance 70% → 49% |
| 4-bit drafter weights | propose −5%, unpacking eats the saved bandwidth |
| Fixed confidence gates instead of the width policy | 76.0 vs 75.0 tok/s on a replay; the gap to the oracle (89.6) is the drafter's accuracy |
| Copy drafts for greedy requests | ±2%, kept for sampled requests only |
| Coupled proposals from a window of the cumulative sum | sampled −7% |
| Back-off tolerance 0 for a lone request | greedy −4% |
| Verification MoE grouped by expert at 4 rows | slower: L2 already dedups shared experts |
| No drafter injection at eight sessions (not exact for later proposals) | +0.6%, within noise |

## Host

| Variant | Result |
| --- | --- |
| Sampler exps split evenly over threads | no faster: the slowest wake-up is paid per draw; chunks taken as threads wake are kept |
| Every sampled row on its own thread | slower below four rows |

## Numerics

| Variant | Result |
| --- | --- |
| Router and controller decode copies in F16 (the extension's choice) | KL 1.416e-3 vs 1.451e-3 with exact BF16, within noise; BF16 kept |
| Compiler contraction in the fused MoE tail, or unrolled `ss += v * v` | not byte-identical; written with explicit FMAs |

## Not exact, not taken

Each would need its own quality qualification.

| Idea | Expected effect |
| --- | --- |
| Relative-position bias by WMMA (Toeplitz band, as the Loom package does) | removes most attention VALU work |
| F16 MoE slot rows instead of F32 | about 45 ms less per pp2048 |
| Rejection sampling of sampled proposals | more accepted drafts at T > 0; output distributed as AR's, not equal to it |
| Expert layers 20–23 as NVFP4, expert pruning, sparse attention from 32K (mikealanni/gufo) | faster decode/prefill at KL ~1e-3 |
