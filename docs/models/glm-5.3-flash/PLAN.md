# GLM 5.3 Flash — implementation plan

Track one phase at a time; promote a phase only with the stated gate.
All GPU work follows `AGENTS.md`: exclusive GPU through
`tools/bench/gpu_exclusive.sh` / `bench_bg.sh` + `bench_wait.sh` (both in one
tool call), iteration at ≤16K context first, `nix develop` presets for test
builds, and `python3 tools/ci/check-format.py` before any C++ commit.

## Phase 0 — Reference baseline (no Gufo code)

Objective: hardware-relevant numbers and behavior ground truth before we
write a kernel.

1. Run the llama.cpp reference checkout (which already supports
   `glm5-next`, including IQ1_M via MMVQ) on Strix Halo.
   `llama-bench -ngl 99` at pp2048/tg32 and pp512/tg128; record per-phase
   timings; note KV sizes actually allocated (latent MLA cache, indexer
   kpool, KDA state) and whether MTP (`nextn`) is wired on the GGUF path.
2. Capture parity fixtures: fixed-prompt logits dumps (first 32 rows,
   fp32) at temp 0 for 17 continuation histories ≤16K, token IDs, and the
   chat template render (reasoning-effort blocks) — reference-only files
   under `artifacts/` keep independent of Gufo framing.
3. Gate: reference serves the shard set end-to-end; artifacts recorded.

## Phase 1 — Staging engine bring-up (`staging/models/glm5_next`)

Objective: load the real IQ1_M shards and produce correct CPU logits.

1. Engine scaffold per existing model dirs (`config`, `weights`,
   `engine`, `reference`, `cpu_ops`, `extension.cmake/json`).
2. Loader: split-aware (metadata shard holds zero tensor entries; tables
   live in data shards), mixed-dtype map incl. IQ1_M/IQ2_XXS/IQ3_XXS block
   types via `core::GgmlType` + `src/core/quant/ggml_dequant.cpp`
   (`DequantizeIQ1_M` exists since b20b5bd4).
3. Scalar `reference.cpp`: full block graph — 3-dense-then-hybrid layer
   schedule, NoPE MLA (q_a/kv_a_mqa/dec), DSA top-2048 gather with the
   compressor-APE kpool, KDA recurrence (gate lower bound −5, conv4,
   per-head 128), mHC sinkhorn-20 four-stream mix, sigmoid top-8 ×2.5
   scale with renorm and swiglu clamp 10 (exact clamp boundaries —
   preserve F32 boundaries like the qwen-image BF16 contracts), shared
   expert, nextn MTP deferred.
4. Control tokens: use the shared Qwen/GLM control-token constants where
   shapes match; eot 154827/eom 154829 semantics verified against the
   reference dumps, not assumed.
5. Gate: fp32 logits match Phase-0 fixture rows within the recorded
   reference-vs-reference tolerance on CPU, all 12 MLA blocks exercising
   indexer boundaries (`context` sweeps at 2047/2048/2049 to force
   top-k edges).

## Phase 2 — GPU kernels (staging, then promote)

Port, do not invent (see [../../COMMON_OPTIMIZATION_STRIX_HALO.md](../../COMMON_OPTIMIZATION_STRIX_HALO.md)):

| Component | Source kernel family | GLM-specific change |
| --- | --- | --- |
| KDA recurrence | Flash-Next GDN kernel (register-resident state, DPP reduce) | gate formulation (`g` lower-bounded at −5), 32×128 heads, conv4 weights in f32; same "named vectors, zero scratch" rule |
| MLA decode | deepseek_v4_flash MLA + flash-decoding split-KV | latent KV 576/token mirror (ckv 512 + kpe 64) F16; NoPE skips RoPE stage |
| DSA sparse gather | Flash-Next indexer kernels | compressor APE + kpool 4; `index_share_mtp` reuse for the MTP block |
| mHC combine | deepseek_v4_flash HC combine (fused HC + mixer launch) | 4 streams, sinkhorn 20, eps 1e-6; keep exact F32 reduce order |
| Routed MoE | qwen36_a3b/gemma4 bucket-compaction WMMA MoE | **new**: IQ1_M/IQ2_XXS/IQ3_XXS B-operand dequant after LDS read → F16 fragments → wmma_f32_16x16x16_f16_w32; swiglu clamp as fused epilogue |
| Attention/shared dense GEMM | existing Q8_0→F16 WMMA tiles | unchanged shapes per config (4096×1536/512 etc.) |
| Decode GEMVs | core gemv + MMVQ IQ ports (`vec_dot_iq1_m_q8_1`) | retune tiles for gfx1151 (upstream tables, not tuned) |

Gate per row: operator checks pass bit-identical-or-within-budget, and the
Phase-1 CPU parity fixtures still hold after each kernel switch.

## Phase 3 — IQ1_M decode economics

Objective: approach the ~90–95 tok/s ceiling from the README's traffic math.

Status 2026-11-10: serve registration landed; gate benchmarked
([BENCHMARKS.md](BENCHMARKS.md), [EXPERIMENTS.md](EXPERIMENTS.md)). First
phase-3 prefill steps shipped — wide power-of-two `GemvRows` slices,
`kChunk = 128`, bucket-compacted `GemvRoutedTiles` expert tier — doubling
prefill to pp2048 41.61 t/s (0.22×), tg32 12.36 t/s (0.74×); gate not met.
`kChunk = 512` was flat, so the remaining pp gap is IQ-expert decode
efficiency, not re-reads. The per-component prefill profile (see
EXPERIMENTS.md P3d) measured pp GPU time as moe routed IQ GEMV 34 %,
dense multi-row GEMV 32 %, MLA 12 %, KDA 9 % — both GEMV families run at
~15-25 % of measured LPDDR peak (weight-decode bound, never DRAM bound).
P3e collapsed the per-head `attn_k_b`/`attn_v_b` loops into one grouped
`GemvRows` launch (bit-identical parity) and measured pp-neutral in a
matched A/B: launch grain was wave ramp, not dispatch gap, so only the
decode-throughput half of the dense stage counts. P5 (routed IQ WMMA,
+21.4 %) and P6 (dense Q5_K WMMA, +32.6 %, cumulative +59.7 % over
`gufo_rows16`) took pp2048 to 67.9 t/s — gate attainment 0.40× — and
both GEMV families at the center of the profile now run on the tensor
cores. Items below are in measured-share order.

1. (Shipped P5 2026-11-12, +21.4 % pp2048 matched) Tensor-core/WMMA tier
   for the stacked IQ experts (Phase-2 mapping row "Routed MoE" — IQ
   dequant after LDS read → F16 fragments →
   `wmma_f32_16x16x16_f16_w32`, flint `mmq` style): the 34 % MoE stage
   and the measured llama.cpp-level lever for pp *and* tg.
2. (Shipped P6 2026-11-12, +32.6 % pp2048 matched — `kernels/rocm/
   dense_wmma`, gated at `GemvN` on Q5_K with `tokens >= 16` and
   `cols % 256 == 0`; contract `glm5_next.rocm_dense_wmma` NMSE
   ~3e-08, ctest 9/9) Wider/tensor-core shapes for the Q5_K MLA
   projections (`GemvRows<16u>`,
   single largest kernel at 22 % of the capture): launch grain measured
   out by P3e; decode throughput only. Upstream references
   (survey 2026-11-12): qwen38 `DenseF16GEMMKernel`
   (`src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp:7680`)
   decodes Q8 codes to F16 once per LDS stage
   (`CodesToHalves` :5416) and wmma-accumulates with F16 acts; the
   mid-width int8 `W8A8BlockedWmmaGEMMKernel` (:5056, `w8a8_wave64.hip`)
   is the alternative; ds4 adds token-tile `sharedx` staging and a
   `wmma_4w` tier at n_tok≥256
   (`src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_q8.hip.hpp:56,127`)
   under a per-shape bit policy
   (`ds4_rocm_matmul.hip.hpp:949-1013`). GLM-5 prefill chunks (128 tok)
   sit inside ds4's igemm window (n_tok≥32); decode rows 1-8 keep the
   scalar tier. The glm5 `RoutedWmmaKernel` already carries the
   proven gfx1151 decode→F16→wmma + C-fragment epilogue machinery.
3. MLA `GatherScore` rewrite (12 %) and KDA per-token sequential scan
   (9 %, ~1 M launches of ~26 us) as the follow-up wave. References:
   ds4 scores 16 heads inside one wave's registers with
   `permlanex16`/DPP butterfly reduction and 32 keys/block
   (`ds4_rocm_indexer_score.hip.hpp:49-104`), prefill
   `indexer_scores_wmma128_kernel` over packed F16 q/k in 128-key tiles
   (:151); qwen38 compacts selection before scoring (`SelectMarkKernel`
   `kernels.hip.cpp:2860`, `CompactSelectionKernel` :3058). For KDA,
   qwen38 `GdnPrefillKernel` (:1984) runs 4-token windows, two rows per
   wave, eight waves, double-buffered LDS with windows-ahead prefetch —
   the template for collapsing the 128 sequential Conv/EGate/FoldSk/
   UpdateOut rounds (`executor.cpp:415-460`). Both rewrites change
   accumulation order; plan for the P5-style NMSE contract + golden
   re-approval instead of bit-exact parity.
4. Retune MMVQ for IQ1_M/I-quant decode rows on gfx1151 (N, nrow, wave
   count — the upstream 6/5–6 tables are legacy defaults).
5. lm_head Q4_K GEMV at the 231 GB/s reference shape; output is 154,880
   rows.
6. Verify byte-exact logits (temp 0) after each fusion; ABBA + 55 °C
   cooldown protocol, 16K-first, then 65K/100K confirmation.

Gate: pp2048 ≥ 0.9× and tg32 ≥ 1.0× the Phase-0 llama.gl baseline with
fixtures byte-identical (or the recorded oracle budget re-approved).

## Phase 4 — Long context, speculation, multimodal

1. 1M-context objectives: latent KV 13.8 KB/token + indexer pool math in
   [KV-CACHE.md](../../KV-CACHE.md) style; confirm indexer build cost at
   128K/1M against `a·n + b·n²` fits.
2. MTP nextn (1 layer, shared indexer) wired like Flash-Next adaptive
   policy; confidence-gated chains only after k=1 greedy acceptance measured.
3. Vision: `glm5v` projector + 24-layer ViT as a follow-up (qwen-image
   pattern), out of scope for the text gates.
4. Then add [BENCHMARKS.md](BENCHMARKS.md)/[QUALITY.md](QUALITY.md),
   functional tests in `tests/functional/`, and ctest goldens; select the
   affected profiles only.

## Risks / open questions

- Resolved: `block_count = 46` = 45 trunk blocks + nextn at index 45; the
  12th `head_count_kv = 1` flag is the nextn layer, so the trunk is exactly
  34 KDA + 11 DSA (il = 3, 7, …, 43). `attention.indexer.types` is absent →
  all trunk DSA blocks run a full indexer; only nextn shares
  (`index_share_mtp = true`).
- NoPE MLA: check the llama.cpp branch actually skips rope application;
  a mis-set `rope.dimension_count` would be a silent quality break.
- `expert_gating_func = 2` and scale 2.5/norm semantics — transcribe from
  llama.cpp `llama-arch.cpp`/graph code, fixture-verify renorm on skewed
  routing.
- Unsloth dynamic quant mixes IQ1_M/IQ2_XXS/IQ3_XXS per-tensor: the bucket
  kernel must dispatch per expert tensor, not per block.
- 96 GB weights leave modest headroom on Strix Halo at 1M context: budget
  cache sizes at boot and refuse over-allocation explicitly.
