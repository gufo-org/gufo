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
  follow-up wave.
