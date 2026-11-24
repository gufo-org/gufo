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

## Round 6 (November 23)

| Variant | Result |
| --- | --- |
| Copy drafts for greedy requests too | rewrite 83.8 vs 84.2 tok/s (the drafter keeps 359 of 434 proposals there already), a function written from scratch 77.8 vs 81.4 (7 of 18 copied proposals kept); kept for sampled requests only |
| Coupled proposals from a window of the drafter's cumulative sum (the token covering most of [u - d, u + d] instead of the one holding u; confidence = its share of the window) | d = 0.03: six sampled requests 57.8 vs 62.4 tok/s (1,047 of 2,175 proposals kept against 1,057 of 1,554: the same tokens accepted, a worse confidence); d = 0.01 with the old confidence: 62.4, 1,061 of 1,561. The misses are not at interval edges |
| Sampler exps split in equal parts over 4 or 8 threads (waiting for all of them) | 1.1 ms per draw in the server, as slow as before: the threads sleep between steps, and the slowest wake-up is paid on every draw; replaced by chunks taken as the threads wake |
| The chunk cursor on the caller's stack beside the captured variables | 0.60 vs 0.50 ms: every chunk taken invalidates the line the other threads read |
| Every sampled row on its own persistent thread | DSpark verification, 2.5 rows: 1.25 ms against 1.0 ms one after another on the calling thread; kept for four or more rows |
| The previous token's first experts read into the cache while the MoE input is mixed and routed (one-row steps; per visit a token keeps 3.5 of the previous token's 8 experts, and the previous first, second and third choices are chosen again 60%, 53% and 49% of the time) | two experts (three on controller visits): 20.73 to 20.76 ms per token against 20.62 to 20.64; one (two): 20.74 to 20.93. The guess is right often enough, but the reads compete with the latency-bound kernels they run beside |
| Back-off tolerance 0 for a lone request (as for batches) | C = 1 greedy 70.2 vs 72.9 tok/s: eight-cycle rates are noisy, and a back-off skips drafting that pays on average; a lone request keeps 3% |

Analysed, not built:

| Idea | Why not |
| --- | --- |
| The gate projection on the side stream beside the attention kernel | The attention window (19 us) already carries the output projection's prefetch; real work there moves the same bytes into the same window |
| Fewer launches (2.0 us of gap each) | Folding 144 launches into their neighbours in round 5 saved 0.09 ms: the fused kernels wait on counters |
| Selecting sampled rows on the device | The host sampler's doubles (glibc `exp`, sum in token order) cannot be reproduced bit for bit by a device kernel |
| A smaller drafter block for batched cycles (the drafter computes eight rows per request even when one proposal fits) | Changes the proposals (the block is non-causal), so it needs an acceptance study; about 2 ms of a 62 ms cycle at four requests |
| More than eight rows a step (drafts at five to eight requests) | Marginal rows get cheaper as the rows' experts overlap (about 2 ms a row past 16), but every row kernel, the slot tables and the conv captures are eight wide |


## Round 8 (November 23)

Grouped expert kernels, measured on a standalone replay of a 64-row step's
real routing (64 experts per visit, NVFP4 gate/up and down per visit; the
kept version is 1.37 / 0.73 ms):

| Variant | Result |
| --- | --- |
| Four or eight row pairs per wave (more rows per block) | 64-row step 244 → 305 / 329 ms: fewer, longer blocks |
| Work items of four pairs of one expert (big experts spread over blocks) | 215 → 225 ms: the extra weight reads and decodes cost more than the imbalance |
| Eight pairs per pass (registers) | 2.71 / 0.96 ms; fixed eight with masked pairs 3.29 ms (spills) |
| A fixed four-pair pass instead of one per pair count (smaller code) | 1.55 vs 1.45 ms: the padded pairs cost more than the code size |
| Gate and up rows of a channel in the same lanes (half the input loads) | 256 VGPRs and spills, 1.80 ms |
| The expert's first eight pairs staged in LDS once per block, the rest from global memory | 1.52 ms |
| VGPR caps for 10 or 12 waves per SIMD | 2.34 / 1.93 ms (spills) |
| 8, 12 or 24 waves per block | 1.48, 1.46, 1.45 ms against 1.37 with 16 |
| Lane moves by DPP and `permlane16` instead of LDS permutes | no faster (1.376 ms) and not exact as written; dropped |
| Grouped kernels from 3 rows | DSpark verification of 4 rows 35.9 → 36.5 ms, batched decode of 3 to 8 sessions 1 to 2% slower; kept for 8 rows of one session and wider |
| Dense int8 WMMA from 8 rows | 8-row verification 53.4 → 53.2 ms, eight-session decode 64.0 → 64.1 ms: within noise, not taken |

Prefill attention at long context (exact variants only):

| Variant | Result |
| --- | --- |
| The next key tile's K, V and bias rows loaded into registers while this one is computed | 243 VGPRs and spills, 5 waves per SIMD: a 12-wave block no longer fits a CU; pp32768 1,892 → 1,791 tokens/s |
| 32-query blocks (two per CU) | not byte-identical (the window visits' key tiles start at the block's first query) and pp32768 1,682 tokens/s |

## Round 10 (November 24)

Measured on standalone replays of the prefill GEMMs (2,048 tokens, 186
items), the 8-row GEMVs (weights rotated past the cache) and the grouped
expert kernels (real routing), then in `inkling2_bench`:

| Variant | Result |
| --- | --- |
| MoE gate/up with double-buffered weight tiles (one barrier per k-step; three blocks per WGP still fit) | 5.90 → 6.02 ms |
| MoE gate/up at 8 waves per SIMD (`amdgpu_waves_per_eu`) | no change |
| MoE gate/up computing padding tiles too (no per-tile branches) | 5.9 → 7.0 ms |
| MoE down and dense GEMMs storing whole 256-byte token rows through LDS | no change: the 151 MB of F32 pair rows cost their DRAM write time either way (0.6 ms of 3.7 when skipped) |
| Prefill MoE grids sized to the used units | no change (blocks already loop over units) |
| SITU's `u / 25` as a corrected product and `1 / x` as `v_rcp` plus a Newton step (both equal IEEE division on all 2^32 inputs, with guards) | no faster |
| SITU with a runtime check for the constants | the gain is lost: both paths are compiled |
| One-input projections of 8 rows through int8 WMMA | verification +0.2 ms |
| The 8-row output projection (K = 3072): rows per wave and waves per block, or WMMA | no faster |
| Grouped expert kernels from 4 or 5 rows of one session, or for 6 sessions | equal or slower (6 sessions −1.5%) |
| Controller gates of 2, 4 or 8 tokens per block (weight row in registers) | verification +0.1 to +0.3 ms: the per-token blocks' parallelism matters more than the L2-resident weight re-reads |
| Verification attention with the split reduction folded into its last block | +0.3 ms per step (the reverse of the kept decode change) |
| Controller apply and its RMS norm as two launches | no change |
| No drafter context injection at eight sessions (upper bound, not exact for later proposals) | +0.6% over HTTP, within noise: not worth deferring the injection |
| Prefetching the next K/V tile in the decode attention (global visits at 32K run at about 83% of DRAM) | not built: the kernels already hold 219 to 255 VGPRs |
