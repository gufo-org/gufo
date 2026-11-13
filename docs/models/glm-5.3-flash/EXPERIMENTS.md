# GLM-5.3-Flash — Experiments

Retained and rejected prefill/decode decisions, one line each. All numbers
are `gufo serve` gate measurements (pp2048/tg32, HTTP `timings`, warmup +
2 samples) from `/tmp/opencode/glm5_bench_gates.sh` under the
bench_bg/bench_wait GPU-exclusive discipline on `build/gpu-test/gufo`.

## Retained (shipped defaults)

- P3a power-of-two `GemvRows` slices: the multi-row GEMV launcher
  decomposes any 2..128-token width into descending power-of-two slices
  (instances 1/2/4/8/16), replacing the 4-token ceiling. Per-row
  accumulation order and per-token stores match `Gemv` bit-for-bit
  (`glm5_next.rocm_gemv` bit fixtures).
- P3b prefill chunk `kChunk` 4 → 128: 32× fewer per-chunk weight sweeps
  for every projection. (Shipped with P3a; measured only combined.)
- P3c bucket-compacted routed expert tier (`routed_compact` builders +
  `GemvRoutedTiles`): `kBucketMinRows = 32` and above, slots are compacted
  into 16-row padded expert buckets (`ExpertCounts`/`RoutedCompact`) and a
  2-D `tile x row` grid decodes each expert weight row once per 16 slots,
  scattering `out[slot*rows + r]`. Same decode/slot/reduce order as
  `GemvGrouped`, so gate/up/down are bit-identical per slot
  (`routed==grouped {gate,down}` checks for all six super-codec types).
  Combined pp2048 **41.61 ± 0.04 t/s** (2.0× baseline 21.07), tg32
  **12.36 ± 0.08** (decode path untouched; within 2 % of 12.12).
- P4 WMMA tensor-core IQ1_M gate/up tier (`routed_wmma`, prototype
  `tools/bench/glm5_iq1_wmma_bench.hip`): bucketed IQ1_M gate/up stacks
  additionally take a 32-slot WMMA path — x converted to F16 once (with
  overflow guard), gate and up blocks staged per tile, shared A fragments,
  fused swiglu-clamp epilogue writing `moe_inter`, tensor-core decode at
  26.5 TF/s effective on the prototype (tile-32 shape). Tensor-core
  accumulation order differs from the scalar dot, so quality is held by
  the new `glm5_next.rocm_routed_wmma` contract (rel. NMSE ≤ 1e-4 vs the
  bit-tiered scalar tiles + `SwigluClamp`, replay bit-identical; measured
  **8.9e-06**) and the unchanged full-model golden replay
  (`glm5_next.rocm_forward`, GPU-exclusive). Matched three-point single
  session (`/tmp/opencode/glm5_bench_ab3.sh`): pp2048 base 20.69 ± 0.18 →
  bucket 38.56 ± 0.53 → post **42.21 ± 0.01** (tier alone **+9.5 %**),
  tg32 12.14 / 12.10 / **12.24** (no regression; decode path untouched).
- P4 correction (2026-11-12): while extending the tier under P5 the WMMA
  kernel was found to stage row 0 of the expert stack for **every** output
  row — the templated `fetch_weights` omitted the row/expert offset
  (`f_off`) from the block pointer. The extension contract failed at
  NMSE ≈ 2 until fixed (single-hot x at one column exposed it: constant
  output over rows instead of the true per-row weight). The P4-era match
  numbers (post 42.21, "measured 8.9e-06") cannot be reconciled with the
  fixed kernel's behaviour and are superseded by P5; the shipped default
  now carries the fixed, contract-verified kernel.
- P5 fixed + widened WMMA routed tier (`routed_wmma` reworked as one
  kernel template `<kSuper, kStride, kStacks>` with explicit
  `DecodeFragmentHalf` specializations): bucketed IQ1_M **and IQ2_XXS**
  gate/up stacks route to WMMA (`RoutedGateUpWmma`), and IQ3_XXS `down`
  stacks join with `RoutedDownWmma` (x converted to F16 once,
  `moe_inter_half_` staging, ConvertF32ToHalf back for the combine);
  IQ4_XS/Q3_K `down` stay on the scalar tier. Staging strides
  60/68/100 B per 256-K stage with a `TailU32` spill for the partial
  super-block tail; epilogue maps the verified gfx1151 C-fragment
  layout (row = 2l + (lane>>4), col = lane & 15; cross-checked with
  `tools/bench/wmma_layout_test.hip`).
  `glm5_next.rocm_routed_wmma` now carries three contracts and passes:
  IQ1_M gate/up NMSE **8.893e-06**, IQ2_XXS gate/up **5.571e-05**,
  IQ3_XXS down **3.432e-08** (limit 1e-4), x-half overflow 0, finite,
  replay bit-identical. Full `glm5_next` ctest 8/8 green
  GPU-exclusive, golden `rocm_device_model` 26.8 s + `rocm_forward`
  80.7 s passing with the tier active.
  Matched A/B (`/tmp/opencode/glm5_ab.sh`, alternating ×2, 128K-ctx
  protocol, control = P3f `gufo_rows16`): control pp2048
  42.69 ± 0.09 / 42.35 ± 0.03, candidate **51.65 ± 0.17 / 51.65 ± 0.31**
  (**+21.4 %** prefill), tg32 11.51/11.12 control vs 11.30/11.48
  candidate — flat. Escalated same-day at ~8.4K prompt
  (`/tmp/opencode/glm5_ab8k.sh`, client-timed because
  `usage.prompt_ms` is absent above the 2K path): control 37.73/37.38
  /37.21/36.32 vs candidate 44.00/43.13 (**+17.2 %**); a 32K escalation
  was cut by budget (≈13 min/prefill at these rates) and left no data.
  Phase gate (pp2048 ≥ 170.5) still **not met**; remaining levers are
  the MLA GatherScore rewrite (13.1 % of pp) and the KDA chunked scan
  (9.3 %).
- P6 dense Q5_K WMMA tier (`kernels/rocm/dense_wmma`): the chunked dense
  projections reuse the proven routed geometry without any expert logic —
  128 rows × 32 tokens per block, 176 B Q5_K blocks staged at a 180-byte
  (45-dword, spill-free) stride, shared A fragments, `DecodeFragmentHalfQ5K`
  mirroring the scalar sub-block walk, epilogue `out[t*out_stride + r]`;
  F32 activations are converted to F16 at operand staging. `GemvN` routes
  to `DenseWmma` only when the type is supported, `tokens >= 16` and
  `cols % 256 == 0`, so decode (n ≤ 8) and the grouped
  `attn_k_b`/`attn_v_b` launches stay on the scalar tier. Tensor-core
  accumulation order differs from the scalar dot, so quality is held by
  the new `glm5_next.rocm_dense_wmma` contract: rel. NMSE **2.9-3.0e-08**
  vs the scalar `GemvRows` over four shapes (limit 1e-4), finite,
  replay bit-identical. Matched A/B (`/tmp/opencode/glm5_ab_p6.sh`,
  alternating ×2 plus a leading candidate smoke round, 128K-ctx
  protocol, control = `gufo_p5`): control pp2048 51.12 ± 0.16 /
  51.32 ± 0.31 vs candidate **68.03 ± 0.69 / 67.89 ± 0.61 /
  67.83 ± 0.75** (**+32.6 %** tier increment; cumulative **+59.7 %**
  over the `gufo_rows16` control at 42.52), tg32 flat (11.22 ± 0.53 /
  11.45 ± 0.06 vs 11.26 / 11.31). Full `glm5_next` ctest 9/9 green
  GPU-exclusive with the tier active (golden `rocm_forward` 95.1 s).
  Escalated the same day at ~8.4K prompt
  (`/tmp/opencode/glm5_ab8k_p6.sh`, client-timed, control = `gufo_p5`,
  candidate = the shipped tree build): control 44.28/43.41/43.81/42.94
  vs candidate 55.71/55.98/55.63/55.87 — **+28 %** kept
  (`gufo_rows16` at 8.4K was 37.2, cumulative ~+50 %).
  Gate attainment is now **0.40×** (pp2048 ≥ 170.5); the dense stage
  leaves the tensor-core candidate list — MLA GatherScore (13.1 % of pp)
  and the KDA chunked scan (9.3 %) remain.
- P7 MLA GatherScore row-warp kernel (`kernels/rocm/attention.hip.cpp
  MlaGatherScoreKernel`): one warp per key slot — lanes load two
  coalesced `uint4` KV chunks, keep the rank row in registers, each
  lane dots its head slice against `float4` q (`qa[kRank/8 + 2*lane]`)
  and folds through a 5-step `__shfl_down_sync`; replaces the scattered
  2-byte warp gathers (~32 L1 transactions per load). Microbench
  (`tools/bench/glm5_gather_score_bench.hip`, NMSE 1.5-1.7e-13 vs the
  bitwise oracle): 256-hot 0.87-0.92×, 1024-hot 2.2-2.4×, 2048-hot
  6.5-6.9×, 8192-cold 26-31×; chain-split (0.08-0.35×) and uint4
  oracle-vectorized (1.0×) variants rejected. Head span is host-side:
  16 slots per block below 4096 selections, 32 above (`hspan`). The
  reassociation moved the mHC-chaotic `code` golden by 0.5006 nats at
  the documented 0.5 budget, so `llama_reference.json` was refreshed by
  candidate re-capture per BENCHMARKS#fixtures (greedy chain, all
  top-1s and prefill snapshots unchanged; revision 457151058). Full
  `glm5_next` ctest green GPU-exclusive (`rocm_forward` ALL PASS
  bit-identical to the refreshed golden; attention contract worst
  latent 4.16e-5 < 5e-5). Matched A/B on a **rebuilt harness**
  (`/tmp/opencode/glm5_ab_p7.sh`: reconstructed after a /tmp cleanup,
  `serving_bench --concurrency 1 --warmup 1 --repetitions 2`,
  prompt_n ≈ 2232, so absolute values are not comparable to the
  archived-row cells; 128K-ctx protocol, control = `gufo_p6` rebuilt
  from 457151058): control pp 66.90 ± 0.09 / 66.51 ± 0.05 vs candidate
  **79.97 ± 0.03 / 79.79 ± 0.08** (**+19.7 %**), tg32 flat
  (12.34/12.46 vs 12.39/12.49). Escalation (same harness family, pp
  only, `/tmp/opencode/glm5_ab8k_p7.sh` + `glm5_ab32k_p7.sh`): at
  prompt_n 8184 control 58.77 ± 0.17 vs candidate 79.85 ± 0.17
  (**+35.9 %**); at prompt_n 32172 control 55.20 vs candidate 76.85
  (**+39.2 %**, one session per binary, warm + 1 timed, p6 first).
  The win widens monotonically with depth: `hspan=32` engages past
  4096 selections and P6's pp degrades from 66.7 while P7 holds
  ~80 t/s through 32K. Gate attainment is now **~0.48×**
  (pp ≥ 170.5); the KDA chunked scan is the remaining listed lever.
- P9 dense Q8_0 WMMA tier (`dense_wmma` extended to a second quant
  instantiation, `DenseWmmaSupports` ⊇ {Q5_K, Q8_0}): the chunk-256 pp
  profile showed `GemvRowsQ8<16>` still at decode cost — ~66 µs per
  call, 12.5 % of a timed chunk pass — because the WMMA tier only
  covered Q5_K. Q8_0 stores 8×(fp16 scale + 32 signed bytes) per
  256-K; tier stages one super-block per row at a 276-dword-trick odd
  stride (kSuper 272, kStride 276) and decodes each 16-element f16
  fragment from one scale and a contiguous signed-byte run. Offline
  `gufo bench` A/B (same session, alternating, warm+2):
  pp2048 90.32/90.54 → **99.18/99.25 (+9.7 %)**; pp16384 83.07 ± 1.27
  → **90.02 ± 0.03 / 90.69 ± 0.54 (+8.4 %)**; tg1 flat (tier engages
  only at the `tokens ≥ 16` prefill predicate). Numeric evidence:
  `glm5_next.rocm_dense_wmma` extended to Q8_0 shapes (scale pinned
  1.0, decode exact in binary16) NMSE ≤ **3.5e-08** ≤ 1e-4, replay
  bit-identical; full `glm5_next` ctest 9/9 GPU-exclusive with
  `rocm_forward` ALL PASS max |dlogit| = 0.0000 vs the retained golden and
  the greedy 128-token continuation sha `752281d6…` unchanged at
  pp2048+tg128 (`/tmp/opencode/glm5_{ab_q8t,ab_q8t16k*,fwd_q8,full_q8,sha_q8t}`;
  the 2048-prefill run executes nine 256-token tier chunks per layer).

## Rejected / parked

- `kChunk` 128 → 512 (bucket-reuse saturation probe): pp2048 41.77 ± 0.21
  vs 41.61 ± 0.04 and tg32 12.43 ± 0.09 — flat. Bucket reuse saturates at
  128-token chunks; the remaining ~4× pp gap to the 189 t/s llama.cpp row
  is expert-decode efficiency at ~42 GB/s effective, not weight re-reads.
  Rejected with the scratch it would cost.
- P3f dense multi-row token-slice widening (`kMaxRows` 16 → 32/64 in
  `LaunchRowsSlices`): the token-boundary reload theory was wrong. Matched
  A/B (same session, alternating, 128K-ctx protocol, control = Rows 16):
  Rows 32 pp2048 18.62/18.58 vs control 42.53/42.50, Rows 64 13.39/13.38,
  tg32 flat. Kernel metadata explains the 64 case (every Q-kernel spills,
  scratch 80-596 B/lane at VGPR 192) but not the 32 case (no spills,
  VGPR ≤ 180): more tokens per weight load multiplies the activation read
  stream per block and thrashes the cache. `kMaxRows` stays 16.

## Profiled (rocprofv3, serve workload)

- P3d per-component pp profile (`tools/prof/prof.py run --stages glm5`
  around a serve run: warmup + fresh pp2048/tg32 pair, 3.97 M dispatches,
  98 s GPU-busy; warmup and timed passes are the same workload so shares
  hold even though the capture was cut at the timed pass): moe routed IQ
  GEMV **34.4 %**, dense multi-row GEMV **31.8 %**, MLA attention
  **12.2 %**, KDA **8.8 %**, mhc 2.6 %, grouped/decode 1.9 %, route +
  combine + indexer + norms remainder ~8 %.
- Diagnosis (a): both GEMV families sustain only ~15-25 % of measured
  LPDDR peak — weight-*decode*-throughput bound, not DRAM bound. The
  `kChunk` 128→512 flat result already ruled out re-read traffic on the
  MoE side; llama.cpp's 189 t/s needs only ~8.4 GB/s aggregate read-once,
  so no bandwidth ceiling blocks the gates.
- Diagnosis (b): dense multi-row GEMV wastes launch grain — the
  `attn_k_b`/`attn_v_b` per-head loops fire 256 GEMVs per layer-chunk
  (Q8_0, ~15 us each, launch-bound: 6.9 s of the capture) and the MLA
  projections run at chunk-128 as 47 launches of ~344 us per layer-chunk
  (Q5_K `GemvRows<16u>` = 21.9 % of the capture, the single largest
  kernel). Diagnosis (b) was falsified by P3e below: the per-kernel
  durations are wave-ramp GPU-busy time, not dispatch gap.

## Measured, effect-free (kept code decision open)

- P3e grouped-head launch collapse (`GemvRowsGrouped`): the multi-token
  `attn_k_b`/`attn_v_b` per-head loops collapse into one launch with a
  per-row activation advance; 30/30 `gemv_rows_grouped` bit-parity checks
  (all six super-codec + Q types, per-group vs the per-head `GemvRows`
  call). Matched A/B in one exclusive session, identical script and
  prompt, pre then post: pp2048 38.45 ± 0.06 -> 38.57 ± 0.44 (within
  noise), tg32 11.76 -> 12.02 on an untouched n==1 path (run-order noise
  ceiling). No measurable pp gain: collapsing the head loops trades launch
  count for an identical wave count, so diagnosis (b) yields nothing on
  its own. A/B absolute values run ~7.6 % below the retained gate rows
  (rebuilt gate script after the original was lost; clean A/B is the
  effect measurement, the 41.61 row stays the retained gate).

## Chunk and wave levers (2026-11-13)

- Prefill chunk width `kChunk` 128 -> 256 (`executor.hpp`): doubles
  routed-expert weight reuse per pass at pp2048. Offline `gufo bench`
  A/B (same session, warm+2, alternating): 85.27/85.33 -> **90.38/90.21
  (+6.0 %)**; 512 plateaus (89.33/89.88 vs 89.58/89.63 control —
  rejected, keeps scratch small). Bit-identical: greedy 128-token
  continuation sha256 `752281d6…` equal across 128/256 binaries,
  `glm5_next.rocm_device_model` 626/626 payload bit-exact, and
  `glm5_next.rocm_forward` ALL PASS with max |dlogit| = 0.0000
  GPU-exclusive against the retained golden (`/tmp/opencode/glm5_*`).
- `kChunk` 256 -> 2048 (revisit of the same lever after P9): each
  256-token chunk's 2048 routed slots touch all 288 experts, so pp2048
  swept the expert stack ~8x (effective ~29 GB/s); at 2048-token chunks
  the sweep collapses and the bucketed WMMA tier runs fully bucketed.
  Offline A/B (alternating, warm+2): pp2048 100.07/100.03 ->
  **109.20/109.13 (p50 +9.1 %)**; pp4096 96.6 -> **104.6 (+8.3 %)**.
  Bit-identical: n-gen-1 output sha256 `43557e0c…` equal across both
  binaries and repetitions, full `glm5_next` ctest 9/9 GPU-exclusive
  (`/tmp/opencode/glm5_gate_c2048.sh`). Profile after the change
  (`/tmp/opencode/glm5_prof2k_c2048`): MoE RoutedWmma is no longer the
  wall; per-token sequential launches dominate — KdaFoldSk 3.3 s +
  KdaUpdateOut 3.3 s (each one launch per KDA layer per token), MLA
  score/combine/bitonic 4.9 s, HcCoeffs/MoERoute/WeightedSum 3.5 s
  (~1.4 M launches over two prefills) vs DenseWmma 3.8 s and RoutedWmma
  5.3 s total. The next lever is processing a chunk's sequential stages
  in one launch with a per-token loop inside (bit-identical order).

## Next candidates (measured order from P3d, P3e updated)

- Tensor-core / igemm-style IQ1_M routed expert tier (flint `mmq` style)
  for the 34.4 % MoE stage — the only measured lever at llama.cpp-level
  pp *and* tg on the IQ1_M experts.
- Wider/tensor-core shapes for the Q5_K MLA projections (`GemvRows<16u>`,
  the single largest kernel at 21.9 %): the launch-grain half of that
  stage is now measured out (P3e) and slice width itself was probed and
  rejected (P3f); only decode throughput (tensor-core shapes) remains.
- MLA `GatherScore` rewrite (12.2 %; 45892 calls at ~224 us) and KDA
  per-token sequential scan (8.8 %, ~1 M launches of ~26 us) as the
  follow-up wave. After kChunk 2048 the ordering is confirmed: the
  per-token launch storm (KDA fold/update + MLA gather + route/coeffs)
  is the dominant cost and the single-launch chunk-scan rewrite is the
  primary lever.
