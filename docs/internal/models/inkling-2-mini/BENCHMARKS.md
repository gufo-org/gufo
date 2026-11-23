# Inkling 2 Mini benchmarks

Strix Halo (Ryzen AI MAX+ 395, gfx1151, 128 GB), ROCm 10.1, November 23,
2026. Measured ceilings on this host (`tools/bench/gfx1151_peak.hip`): 240.5
GB/s DRAM read, 54.8 TFLOPS F16 WMMA.

## End to end

`gufo bench -m Inkling-2-Mini-UD-Q4_K_XL.gguf -p N -n 128 -r 2` against
llama.cpp `94f3af3d0` `llama-bench -fa 1` (HIP, `Mainline_Support` copy of the
same weights).

| Workload | Gufo | llama.cpp HIP |
| --- | --- | --- |
| pp512 | 1,483 tokens/s | not measured |
| pp2048 | 2,041 tokens/s | 383.5 tokens/s |
| pp8192 | 1,960 tokens/s | not measured |
| tg128 after 64 tokens | 47.0 tokens/s | 24.3 tokens/s (tg128) |
| tg128 after 2,048 tokens | 43.9 tokens/s | |
| tg128 after 8,192 tokens | 42.0 tokens/s | |

Batched decode (`inkling2_bench --ctx 8192 --batch 2,4`, aggregate):

| Sessions | 64-token context | 4,096-token context |
| --- | --- | --- |
| 1 | 46.1 | 43.5 |
| 2 | 68.3 | 61.7 |
| 4 | 93.4 | 80.2 |

The imported extension measured 44.7 / 86.2 tokens/s (1 / 4 sessions, short
context) and 1,366 tokens/s at a 4K prompt on Gufo 0.35.0; on this build its
kernels gave 1,395 tokens/s at pp2048 and 45.7 tokens/s decode.

## Prefill changes (all byte-identical)

pp2048 progression, each step measured with the previous one in place:

| Change | pp2048 |
| --- | --- |
| Imported extension, ported | 1,395 |
| Loop-controller gate and apply re-tiled to 64 tokens x 64 rows per block (k-order kept) | 1,566 |
| Controller apply grid with a token tile's channel blocks adjacent | 1,658 |
| Residual and message-bank copies replaced by buffer swaps | ~1,665 |
| 128-token MoE gate/up kernel: 8 waves of 32 tokens x (32 gate + 32 up) | 1,697 |
| Activations straight from global memory into the WMMA A operand | 1,859 |
| Same 128 x 128 kernel for MoE down and the dense projections | 1,903 |
| MoE tail fused (slot sum, post conv, residual add) | 1,912 |
| WMMA skipped on 16-token tiles past an item's end | 2,037 |
| 128-token tiles from 256-token ubatches (was 1,024; pp512 1,378 → 1,483) | 2,041 |

The generic attention-residual mix (`mix_k`) dropped from 72 to 46 ms per
pp2048 by unrolling its loops, which issues loads ahead without reordering the
additions.

Rejected variants are in [Experiments](EXPERIMENTS.md).

## Decode changes (all byte-identical)

tg128 after 64 tokens, alternating A/B on one build each:

| Change | tg128 |
| --- | --- |
| Imported extension kernels | 45.7 |
| One session: expert GEMV grid with an expert's row blocks adjacent | 46.0 |
| 8 waves per block in every weight GEMV (rows per block only) | 46.8 |
| A pass's 12 message mixes in one launch (they share the pass's bank) | 47.0 |

At 47 tokens/s a step takes 21.3 ms: about 16.8 ms of weight GEMVs at 87–98%
of the DRAM ceiling (the LM head reads 492 MB in 2.1 ms), about 3.4 ms of small
kernels and under 1 ms of gaps. Rejected variants are in [Experiments](EXPERIMENTS.md).
