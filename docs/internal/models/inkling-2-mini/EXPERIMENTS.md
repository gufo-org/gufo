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
