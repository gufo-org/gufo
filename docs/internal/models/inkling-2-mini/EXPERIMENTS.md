# Inkling 2 Mini experiments

Rejected or neutral changes, measured November 22–23, 2026 on Strix Halo
against the build they modified. Every variant was exact unless noted.
[Benchmarks](BENCHMARKS.md) lists the kept changes.

## Prefill (pp2048)

| Variant | Result |
| --- | --- |
| 128-token tiles in the original 4-wave `k_gemm4` | 1,443 vs 1,658 tokens/s: 4 waves per block lose occupancy |
| Double-buffered weight tiles in the 128-token gate/up kernel | gate/up 309 → 351 ms: the second buffer costs occupancy |
| VGPR cap of 192 (`__launch_bounds__(256, 8)`) on the 128-token kernels | 1,895 vs 1,912 tokens/s: gate/up and down slightly faster, dense slower |
| The five attention projections in one launch | 1,900 vs 1,903 tokens/s |
| 32-token tiles in the controller gate kernel (twice the blocks) | 2,029 vs 2,037 tokens/s |
| 128-token tiles from 64-token ubatches | pp64 454 → 422, pp128 743 → 698 tokens/s; kept from 256 |
| Fast `k_mix` for the prefill mixes instead of `mix_k` | 1,718 tokens/s, but not byte-identical (different reduction order); replaced by an unrolled `mix_k` |
| Pointers of `mix_k`'s states in LDS | no change: the by-value kernel argument was not the cost |
| Float4-LDS, register-prefetched gate kernel (`k_gate_v2`) for the three controller jobs and the router | controllers 1.39 → 1.70 ms, router 243 → 296 µs: `k_gate_p64` already moves its ~340 MB at the DRAM ceiling; kept only for the skinny and single-job gates |
| The same with 32- or 128-token tiles | 1.91 / 1.73 ms: more blocks re-read the 1.2 MB F32 weights, fewer hide less latency |
| 16-token tiles for the single-job FFN controller gate | 622 → 952 µs |
| 256-token MoE items (16 waves sharing one dequantised weight tile) | gate/up 6.03 → 7.66 ms, down 3.88 → 4.59 ms |
| Straight-line WMMA body in the gate/up kernel | 6.03 → 6.88 ms (kept for down and the K = 2304 dense GEMMs, where it wins); with a 192-VGPR cap 6.65 ms; the cap alone changes nothing |
| Gate/up with 16 tokens x 128 rows per wave (activations loaded once per block instead of twice) | 6.54 ms, 6.66 ms with a one-step activation prefetch (248 VGPRs): the doubled LDS weight reads cost more than the saved global loads |
| 32-channel controller apply (half the LDS) | 1.38 → 1.36 ms, within noise; removing both `tanhf` saves only 10% |
| Next layer's messages mixed before the layer (beside its own input mixes) | no change: both are bandwidth-bound; moved beside the gate/up GEMM |
| Messages' inverse RMS as its own side-stream launch | pp2048 +0.2%, pp512 −1.5%; folded into the message mix and limited to ubatches of 1,024+ tokens |

## Prefill, round 3 (November 23)

| Variant | Result |
| --- | --- |
| MoE gate/up: weights fetched two k-steps ahead (two register sets) | 6.12 → 6.38 ms per visit: the weight latency was already hidden |
| MoE gate/up: activation operands through a three-step register ring (load for step q + 2 before step q's WMMAs) | 6.41 ms |
| MoE gate/up: B operands of the next 16-k step loaded during this step's WMMAs, wave-uniform live flags | 6.42 ms |
| MoE gate/up with 4 waves of 64 tokens x 64 channels (16 WMMAs per 8 operand loads) | 7.47 ms |
| Diagnostics (not exact, measurement only): no per-step dequantisation; no activation streaming; neither and no weight loads | 5.42 / 5.46 / 5.35 ms: the WMMA + LDS loop itself is the bound |
| Attention: later query tiles dispatched first (all KV heads of a tile together, or per head) | 62.3 → 67.4 / 67.8 ms |
| Attention-residual mix: each state's inverse RMS and query dots from one read of the row (same orders) | 44.6 → 73.9 ms: only n of the 8 waves work |
| q and gate projections in one launch (a token tile's q row tiles, then its gate row tiles) | pp2048 2,215 → 2,198 tokens/s |

## DSpark

| Variant | Result |
| --- | --- |
| The full 200,058-row LM head for proposals | depth 3, context 64: 62.4 vs 68.2 tokens/s with the 65,536-row head, same acceptance counts |
| A 32,768-row proposal head | context 64: 49% vs 70% of the proposals accepted, 56.4 vs 68.2 tokens/s |
| Back-off after any cycle that emitted less than a plain step | sampled requests drafted in about 70 of 300 cycles; judged over eight cycles against the measured plain steps instead |
| Verification MoE grouped by expert (one block row per distinct expert; all its pairs at once, or one after another) | 4 rows: 43.4 / 39.1 vs 37.8 ms: the pair-per-block grid already reads a shared expert once through L2 |

## Decode (tg128 after 64 tokens)

| Variant | Result |
| --- | --- |
| Quantising each activation vector once (`k_quant`) instead of in every GEMV block | 45.5 vs 46.9 tokens/s: GEMV times unchanged (the per-block quantisation hides behind the weight stream), about 240 extra launches per token |
| Two row iterations per wave in the expert GEMVs | 45.1 vs 46.0 tokens/s |
| 16 waves per block in every weight GEMV | 47.04 vs 47.09 tokens/s |
| More lanes per row in MoE down (its 18 NVFP4 units split unevenly over 8 lanes) | not tried: changes the reduction tree |
| MoE expert weights loaded into registers before the block quantizes its input (overlapping the two) | 22.43 vs 21.86 ms per token: the held units cost occupancy |
| MoE inputs quantized once per token into global memory, expert blocks reading them from L2 instead of LDS | 22.30 vs 21.9 ms per token; the expert GEMVs already stream about 219 GB/s (8 x 3 MB in 109 us) |
| DSpark block weights in 4 bits (the Loom package's choice; drafter only, the output cannot change) | propose 4.35 vs 4.59 ms at depth 7, accepted 1,084 vs 1,089 of the coding replays' proposals: the unpacking eats most of the saved bandwidth |
| Our width policy against fixed confidence gates (replay of 1,694 coding cycles, measured costs) | policy 75.0 tok/s, best gate (0.85) 76.0, per-cycle oracle 89.6: the gap is the drafter's accuracy, not the width choice |

## Numerics

| Variant | Result |
| --- | --- |
| Router and loop-controller decode copies in F16 (the extension's choice) | changes 100% / 99% of the values; KL against llama.cpp CPU 1.416e-3 vs 1.451e-3 with exact BF16, the same within noise; kept BF16 |
| Fused MoE tail with the compiler free to contract `cur + (x + y) * scale` | not byte-identical (one FMA instead of two roundings); contraction turned off |

## Ideas from another Inkling port (mikealanni/gufo, 2026-11-23)

That port's plain decode is 29.7 ms per step (here 22) and its 2K prefill 1,692
tokens/s (here 2,210); its notes say greedy speculation can differ from plain
decoding near ties. Of its retained ideas, these are exact and were taken or
were already here:

| Idea | Here |
| --- | --- |
| Confidence-gated draft width up to 7 | taken: verification up to 8 rows (coding +5% greedy) |
| Proposal vocabulary ranked by `markov_w1` row norms | already (65,536 rows, Q8_0) |
| Device argmax per verify row, lazy logits | already |
| Packed few-row attention per KV group | already for global visits (`attn_decode_rows`) |
| Skip drafter work after the first weak slot | already (the width policy's early stop) |

Not exact, so not taken: expert layers 20 to 23 converted from Q8_0 to NVFP4 at
load (KL 0.0013), expert pruning for few-row steps, block-sparse attention
from 32K, int8 decode activations shared across projections, NPU block-float
projections in prefill. Larger prefill batches (4,096) change the chunking the
byte comparison is defined on.

The multi-row projection spill noted then is fixed (the tiny-matrix token
loop, not the unit loop, held the registers).

## Not exact, not taken

Ideas that would change the arithmetic and therefore the logits; each needs
its own quality qualification before it could replace the exact path.

| Idea | Expected effect |
| --- | --- |
| Relative-position bias by WMMA (a 16 x 16 x 16 product of the R rows with a Toeplitz band of the bias table, as the Loom package does) instead of eight `fdot2` per score | removes most attention VALU work; changes the bias's summation order |
| A fast prefill `k_mix` for the message passes (one pass over the states) | measured 1,718 vs 1,658 tokens/s in the first round; different reduction order |
| F16 (or BF16) MoE slot rows instead of F32 `pairs` | halves the 302 MB written and read per visit by the down GEMM and the MoE tail (about 45 ms per pp2048) |
| Inverse RMS of the controller inputs computed by the mixes that produce them, in their own reduction order | drops the remaining half of the controller gates' prologue reads |
| Rejection sampling of sampled requests' proposals (the drafter's distribution as q) | more accepted proposals at temperature > 0; the output is then distributed as AR's, not equal to it |

## Round 5 (November 23)

| Variant | Result |
| --- | --- |
| Attention-residual mix with the states its prologue does not write processed first and the prologue's outputs read from LDS | 10.6 → 12.8 µs per launch: the mix is not bound by load latency |
| The mix's weighted sum with all states' loads issued first (unrolled to 8) | decode 21.67 → 21.75 ms |
| The norm's unrolled loads with `ss += v * v` | not byte-identical: that form was not contracted to the FMAs the rolled loop produced; written with explicit FMAs it is (kept) |
| MoE gate/up and down in one launch (down blocks wait on per-assignment counters) | 1 row 21.65 → 21.60 ms, 4 rows 41.73 → 41.90, 8 rows 66.1 → 67.3 |
| Three-input projections of 6 to 8 rows in one launch instead of two | 8 rows 55.9 → 59.0 ms (register pressure) |
| Drafter GEMVs of up to 4,096 rows with one row per wave (more blocks) | propose 3.47 → 3.93 ms: every block stages the 8 input rows |
| LM head pruned for greedy rows (a 4-bit proxy head, exact rows only for the rows whose bound reaches the best) | not built: a rigorous per-row bound (Cauchy–Schwarz on the proxy's residual) is about 10% of the logit scale, which leaves thousands of candidates; the sessions' cached logits would also need lazy materialisation |

Not exact or outside the model, not taken:

| Idea | Expected effect |
| --- | --- |
| Host sampler: the temperature-only distribution's exps on parallel threads (the same values, then the same sequential sum) | about 1 ms per sampled token at C = 1 (core sampling code, all models) |
| Rejection sampling of sampled requests' proposals | more accepted proposals at T > 0; the output is distributed as AR's, not equal to it |
