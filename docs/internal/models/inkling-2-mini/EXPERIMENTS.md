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

## Decode (tg128 after 64 tokens)

| Variant | Result |
| --- | --- |
| Quantising each activation vector once (`k_quant`) instead of in every GEMV block | 45.5 vs 46.9 tokens/s: GEMV times unchanged (the per-block quantisation hides behind the weight stream), about 240 extra launches per token |
| Two row iterations per wave in the expert GEMVs | 45.1 vs 46.0 tokens/s |
| 16 waves per block in every weight GEMV | 47.04 vs 47.09 tokens/s |
| More lanes per row in MoE down (its 18 NVFP4 units split unevenly over 8 lanes) | not tried: changes the reduction tree |

## Numerics

| Variant | Result |
| --- | --- |
| Router and loop-controller decode copies in F16 (the extension's choice) | changes 100% / 99% of the values; KL against llama.cpp CPU 1.416e-3 vs 1.451e-3 with exact BF16, the same within noise; kept BF16 |
| Fused MoE tail with the compiler free to contract `cur + (x + y) * scale` | not byte-identical (one FMA instead of two roundings); contraction turned off |

## Not exact, not taken

Ideas that would change the arithmetic and therefore the logits; each needs
its own quality qualification before it could replace the exact path.

| Idea | Expected effect |
| --- | --- |
| Relative-position bias by WMMA (a 16 x 16 x 16 product of the R rows with a Toeplitz band of the bias table, as the Loom package does) instead of eight `fdot2` per score | removes most attention VALU work; changes the bias's summation order |
| A fast prefill `k_mix` for the message passes (one pass over the states) | measured 1,718 vs 1,658 tokens/s in the first round; different reduction order |
| F16 (or BF16) MoE slot rows instead of F32 `pairs` | halves the 302 MB written and read per visit by the down GEMM and the MoE tail (about 45 ms per pp2048) |
| Inverse RMS of the controller inputs computed by the mixes that produce them, in their own reduction order | drops the remaining half of the controller gates' prologue reads |
