# Gemma 4 31B kernel audit, 2026-10-21

Production `build/release` at `9aaf291a`, UD-Q4_K_XL target, Q8_0 Unsloth MTP
drafter, greedy, C1, gfx1151 (Strix Halo). Ceilings from
`tools/bench/gfx1151_peak` on the same host and session. Phases profiled with
`tools/prof/prof.py` (rocprofv3 kernel trace); decode windows taken between
consecutive SoftcapKernel launches, MTP between consecutive head launches.

## Ceilings measured

| Resource | Measured |
| --- | ---: |
| DRAM read | 239.66 GB/s |
| DRAM write | 220.55 GB/s |
| DRAM copy | 209.89 GB/s |
| WMMA fp16 / bf16 | 55.50 TFLOP/s |
| WMMA int8 | 57.13 TOPS |
| VALU fp32 FMA | 26.59 TFLOP/s |
| LDS read | 6736 GB/s |

## Throughput (native `gufo bench`, this build)

| Phase | d0 | d16384 | d65536 |
| --- | ---: | ---: | ---: |
| AR pp2048 (tok/s) | 580.10 | 474.44 | 309.96 |
| AR tg (tok/s) | 11.61 | 10.79 | 9.51 |
| MTP pp2048 (tok/s, repetitive prompt) | 559.11 | 455.06 | - |
| MTP tg (tok/s, repetitive prompt) | 31.41 | 71.17 | - |

MTP d0: 40 cycles, 113 drafted, 87 accepted, 35 lookup copies for 128 tokens;
3.2 tokens per cycle, 105.8 ms host span per cycle.

## AR decode, one token (660 launches, 84.5 ms GPU, 87.1 ms span)

| Component | launches | ms | share | bytes | effective |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFN gate\|up (Q4_K x54, Q5_K x6) | 60 | 34.1 | 40.3% | 7.98 GB | 234 GB/s |
| FFN down (Q6_K x26, Q4_K x34) | 60 | 20.7 | 24.5% | 4.76 GB | 230 GB/s |
| attn q\|k, fused per layer (54 x 12288 + 10 x 18432 rows) | 60 | 10.8 | 12.8% | 2.47 GB | 229 GB/s |
| attn output | 60 | 8.0 | 9.5% | 1.73 GB | 216 GB/s |
| attn v (50 x 4096, 10 x 2048 rows) | 60 | 4.1 | 4.9% | 0.89 GB | 217 GB/s |
| vocabulary head (Q5_K, 262144 rows) | 1 | 4.1 | 4.8% | 0.97 GB | 239 GB/s |
| attention, norms, QkvPost, GeGLU, merge, argmax | 47 | 2.7 | 3.2% | - | - |
| launch gaps | - | 2.5 | 2.9% | - | - |

Aggregate 18.80 GB in 84.5 ms of GPU time = 222.5 GB/s, 92.9% of the measured
read peak. Per-launch overhead against a pure stream (0.5-15 us) accounts for
most of the 6 ms gap; every projection sits at 216-239 GB/s.

## Prefill, 2048-token chunk at d0 (3.566 s GPU, 290 projection launches)

| Stage | ms | share | rate |
| --- | ---: | ---: | ---: |
| Projections (RoutedHalfKQuantKernel, dense path, 112-token tiles) | 3245.7 | 91.0% | 37.0 TFLOP/s = 66.7% of WMMA peak |
| Global + sliding attention (WMMA) | 150.3 | 4.2% | - |
| QkvPost, norms, GeGLU, quantize | 169.8 | 4.8% | 185-203 GB/s |

## MTP decode, one verification cycle (37 forwards / 40 cycles; 98.3 ms GPU, 105.8 ms host span)

| Component | ms/cycle | share |
| --- | ---: | ---: |
| Target verification projections (WMMA GEMV) | 93.0 | 87.5% |
| Target vocabulary head | 4.3 | 4.4% |
| Drafter (2.8-3.2 steps) | 6.4 | 6.5% |
| Verification attention (5 rows average) | 2.2 | 2.3% |
| Norms, QkvPost, GeGLU, merge | 2.6 | 2.7% |
| launch gaps | 3.1 | 3.2% |

Verification projections move 17.83 GB per forward in 93.0 ms = 191.7 GB/s,
14% behind the AR GEMV on the same bytes. Per matrix (MTP vs AR): gate|up 216
vs 233, down 209 vs 230, o 200 vs 216, q\|k 198 vs 229, k/v 208 vs 217,
head 225 vs 239.

## Decode depth cost is KV read traffic

Model: t = (18.80 GB weights + KV bytes) / 239.66 GB/s + ~7.5 ms fixed.

| Depth | KV bytes/token | bytes floor | measured | effective |
| ---: | ---: | ---: | ---: | ---: |
| 0 | ~0.01 GB | 78.4 ms | 86.1 ms | 218 GB/s |
| 16,384 | 1.62 GB | 85.2 ms | 92.7 ms | 204 GB/s |
| 65,536 | 4.08 GB | 95.4 ms | 105.2 ms | 180 GB/s |

Sliding rings are 0.80 GB/token once the window is full (50 layers x 16 MB);
global layers add 50 KiB per cached token (0.82 GB at 16K, 3.28 GB at 64K).

## Instrumentation notes

- `prof.py` short names keep the kernel template arguments, so one
  `KQuantGemvKernel<(GemvFormat)N, ...>` row holds every projection of that
  format; per-shape attribution needs the raw database
  (`rocpd_kernel_dispatch` joined to `rocpd_info_kernel_symbol`), where
  `grid_size_x / workgroup_size_x` gives the block count.
- `gufo bench -d N` re-prefills the depth context for more than one pass, so
  splitting a deep profile at a single idle-gap threshold yields several
  16K-token passes; kernel tables must be restricted to one pass.
- Raw outputs: `ar-d0.txt`, `mtp-d0.txt`, `mtp-trace.jsonl` in this directory.
## Verification-width dispatch experiments (2026-10-21)

Two dispatch variants for the MTP verification projections, each bit-exact per
row and width-invariant, both rejected. Reverted to the shipped dispatch;
`gufo bench` MTP tg128 d0 is back to 31.39/31.39/31.38/31.44 (baseline
31.39/31.44).

| Variant | Change | Projection effect | End to end |
| --- | --- | --- | --- |
| Fused q/k | One 12288-row walk, the layout autoregressive decode already uses, instead of the split q and k launches | The fused launch fills 96 eight-wave blocks of 32 rows where the pair fills 128 four-wave plus 32 eight-wave: 206.4 us for 37.16 MB (180 GB/s) against 188.2 us (197 GB/s) for the pair | MTP tg128 d0 31.40/31.43 -> 31.22/31.20 tok/s (ABBA); AR tg128 unchanged at 11.62 |
| Split-K 2 and 4 on short rows | Splits for o (5376x8192), q (8192x5376) and k/v (4096x5376) | o 146.3 -> 156.1 us (+6.7%), k 64.6 -> 67.7, v 90.7 -> 92.8, q 123.6 -> 127.2 (119.2 at four); each pays a reduce pass | +8,400 launches per 128 tokens, decode-pass GPU 3930.8 -> 3990.1 / 3993.1 ms |

Both experiments point the same way: per-launch bandwidth in the verification
forward tracks block fill (head, 8192 blocks: 225 GB/s; gate/up, 1344: 216;
q, 128: 200; o, 84: 169), and neither more nor fewer blocks helps once the
reduce/launch overhead is counted. The remaining gap to the AR GEMV (233
GB/s on gate/up) is the walk: the 2026-10-19 load-only probes cap at 209-212
GB/s at this launch shape for every load pattern tried.
## Prefill projections, 2026-10-21

Standalone harness `tools/bench/gemma4_prefill_bench.hip` builds the production
translation unit (`moe_prefill.hip.cpp`) with the model's flags and times
`LaunchDenseHalfGemm` on the 2048-token chunk's shapes, rotating weight copies
beyond the MALL. It reproduces the profiled chunk within 7% (3039 ms against
3246 ms) and iterates in about a minute.

| Shape (2048 rows) | ms | TFLOP/s |
| --- | ---: | ---: |
| gate\|up Q4_K 43008x5376 | 23.68 | 40.0 |
| gate\|up Q5_K 43008x5376 | 25.31 | 37.4 |
| down Q4_K 5376x21504 | 11.65 | 40.7 |
| down Q6_K 5376x21504 | 12.74 | 37.2 |
| q\|k 12288x5376 | 6.73 | 40.2 |
| o 5376x8192 | 4.57 | 39.5 |
| v Q6_K 4096x5376 | 2.48 | 36.3 |
| predicted chunk | 3039 | 39.4 |

Drop-work probes on gate/up (23.77 ms baseline): no weight decode 21.46 ms
(-9.7%), no barriers 23.33 ms (-1.8%), no WMMAs 10.66 ms. Token tiles 96 and 64
measure 23.96 and 39.94 ms, so 112 is the optimum. Kernel resources: 181 VGPRs,
0 spills, 30464 B LDS, 8 waves/SIMD (one block per SIMD, four per WGP).

The generated loop issues 56 WMMAs, 117 LDS reads, 62 waits and 192 decode VALU
per stage; 28 of the WMMAs are preceded by an `s_waitcnt lgkmcnt` 2-5
instructions after their LDS load. Prefetching those fragments into registers
costs exactly the occupancy that hides them (205 VGPRs, 7 waves/SIMD, +3.8%),
and making the super-block half a template argument (to fold the scale gather)
spills 187 bytes per thread.

## Follow-up qualification

The standalone build helper omitted `-ffast-math -fno-finite-math-only` even
though the production translation unit uses them. The helper now supplies
both; historical measurements above retain their original provenance.
Fresh matched-flag controls reach 39–40 TFLOP/s. Independent WMMA scheduling,
compact XOR-swizzled 128-token tiles and iterative-ILP scheduling did not
produce a qualified improvement. See the [follow-up experiment](../../EXPERIMENTS.md)
and [all measurements and candidate patches](../../artifacts/prefill-followup-2026-10-21.json).
The follow-up totals are weighted standalone kernel times, not full inference
measurements; production dispatch and published serving rates remain unchanged.
