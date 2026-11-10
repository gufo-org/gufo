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

Status 2026-11-10: serve registration landed and gate benchmarked
([BENCHMARKS.md](BENCHMARKS.md) gufo column): pp2048 21.07 t/s (0.11×),
tg32 12.12 t/s (0.72×) — gate not met. Beyond the decode items below, close
the structural prefill gap: the executor prefill reuses the decode
`kChunk = 4` GEMV, re-reading all weights per 4 tokens; a wide-batch
prefill path is required for the pp2048 gate.

1. Retune MMVQ for IQ1_M/I-quant decode rows on gfx1151 (N, nrow, wave
   count — the upstream 6/5–6 tables are legacy defaults).
2. `GemvMulti` over the 8 routed experts of a block (one activation x,
   heterogeneous quant types per tensor) + fused shared-expert pairing;
   target ≥1 launch per expert-tensor currently 3.
3. lm_head Q4_K GEMV at the 231 GB/s reference shape; output is 154,880
   rows.
4. Verify byte-exact logits (temp 0) after each fusion; ABBA + 55 °C
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
