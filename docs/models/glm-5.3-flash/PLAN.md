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
cores.

Status 2026-11-14: pp2048 **184.5 t/s** (now **217.0** with A5+A3+
A2-decode shipped), pp16384 **147.0** (**165.8**),
exact default (sha `43557e0c…`, ctest 9/9). A fresh `rocprofv3` capture of
the shipped tree at pp2048 (`/tmp/opencode/prof_mla2`) shows steady-state
idle down to **2.2 %** — the launch storm is gone after P8 (KDA
single-launch scan), row-batched elementwise and the token-batched
indexer/MLA launches; every remaining ms is kernel throughput. Per-pass
GPU time (11.7 s total): routed IQ WMMA 2.7 s (23 %), dense Q5_K WMMA
2.2 s (19 %), `MlaGatherScore`+`Combine` 1.7 s (15 %), KDA
`KdaFoldUpdateChunk` 1.4 s (12 %), `GemvRowsQ8_0<16>` 0.8 s (7 %,
11.6 k calls of ~70 us), bitonic indexer sort 0.68 s (6 %, 22.6 k
single-block launches), `GemvRowsF32` 0.25 s, norm/mhc/grouped-GEMV/
converts ~0.7 s. The 600+ target with the exact default is out of reach of
any remaining single lever (even halving every stage lands ~310 t/s), so
the ladder splits: A = exact-default items toward ~280-310, B = opt-in
approximate fast track from which 600 is reachable (the reference 634 t/s
sat on a different quant mix plus W8A8 and a capped selection window — see
EXPERIMENTS.md notes on `mikealanni/fork-enhancements`).

Shipped since: P7 warp-per-key `MlaGatherScoreKernel` (+19.7 % pp,
golden re-approved at revision 457151058), single-launch chunked KDA
(+21.7 %), row-batched HC/MoE elementwise (+9.7 %), Q6_K dense WMMA tier
(+8.3 %), token-batched indexer/MLA launches (+3.0 %). Remaining
levers, exact-default ladder (A) then opt-in fast track (B):

**A. Exact-default levers (measured-share order, ladder to ~280-310)**

1. (Closed 2026-11-14: the token-widening form ran +31 % per dispatch at
   VGPR 256 with 23 spills - see EXPERIMENTS P3h. Weight re-reads
   halved while the kernel got 1.75x slower: routed IQ WMMA is
   compute-bound on the IQ decode VALU - its mix already shows 4.3 k
   VALU per 64 `v_wmma` - and every tiling form that re-reads fewer
   weight bytes pays it back in `kActPer`/accumulator VGPRs at the hard
   256 limit. The split-K occupancy sub-idea inherits the same
   verdict: no VALU saving.) Routed IQ WMMA headroom (2.7 s/pass).
2. (Shipped 2026-11-14 in part: the dword-trick decode +1.2-1.7 %
   pp2048 / +2.8 % pp16384 — see EXPERIMENTS "Q5_K wmma fragment dword
   decode"; the nibble→F16 pair tables and all wider-staging variants
   are rejected, decoding instruction count is not the wall) Dense
   Q5_K WMMA fragment decode (2.2 s/pass): `DenseWmmaKernel<176,
   180>` averages 5.3 ms/call at `blocks=32x64` and is VALU-bound on the
   Q5_K block decode (sign-extraction chains), not on tensor-core math.
 3. (Shipped 2026-11-14, +11.3 % pp2048, pp16384 +5.8 %..10.6 % —
    pp2048 216.0, see EXPERIMENTS "KDA fold + update chunk") KDA
    fold/update scan: the sub-chunk (a,b)/Blelloch premise below was
    wrong — `sk[j]` couples every row of a column, so the per-element
    fold is not scan-composable. The bit-exact win was the other axis:
    one thread per state column with the column in `float sc[128]`
    registers (zero spills), ~20 ms -> ~2 ms per launch, zero shared
    memory. Nothing serial across tokens remains that a scan could fix
    cheaper.
4. MLA score+combine rewrite (1.7 s/pass): combine averages 40 ms at
   `blocks=64x2048 wg=512`; score 37 ms at `256x4x2048`. Both are now
   single launches per layer-chunk but still VALU gathers; WMMA-tiled
   scoring (ds4 `indexer_scores_wmma128` pattern, packed F16 q/k over
   64-key tiles) and a wave64 combine over rank-slice splits. Changes
   accumulation order → P7-style NMSE contract + golden re-approval.
   Target ~1.5x (−0.55 s).
5. (Shipped 2026-11-13, +4.4 % pp2048 — pp2048 193.2, see EXPERIMENTS
   "Single-list top-k shrink") Indexer bitonic sort batching (0.68
   s/pass, 22.6 k launches of one 1024-thread block): the single-list
   case now runs its `k2`-sized network; what remains is batching
   tokens per launch for the deep-context multi-merge path (bucket by
   `k2`, merging rounds gated per-token). Bit-exact per token by
   construction.
6. `GemvRowsQ8_0<16>` (0.8 s/pass at 11.6 k calls) and `GemvRowsF32`
   (0.25 s, 13.6 k calls): tensors outside the dense-WMMA predicate
   (`cols % 256`); admit them to the tier (tier change → sha re-capture
   with recorded diff) or row-batch the call sites (token-batch pattern
   of P8). Target −0.5 s.

**B. Opt-in approximate fast track (from which 600+ is plausible)**

7. W8A8 blocked-INT8 tier for dense + routed operands (qwen38
   `W8A8BlockedWmmaGEMMKernel` family): halves weight bytes and ~2x
   tensor-core rate vs the F16 path; acts quantized per 32-group.
   Opt-in flag + own quality contract (`tests/functional/`, no default
   sha change). Expect −3…−4 s/pass across stages 1-2-6.
8. Online-softmax single-pass attention replacing DSA
   score→topk→gather→combine (the `mikealanni` WMA-attend shape): one
   WMMA pass over F16 packed K/V with running softmax; kills stages
   4+5 (~2.4 s → ~0.4 s) and the selection materialization. Approximate
   by design (fixed window or score floor for the sparsity decision);
   opt-in, `pi_agent.py` replay as quality gate.
9. Capped selection window / k-pool budget at long context (their 634
   point pinned it): quality tradeoff, opt-in, long-context regressions
   per the functional-test rules.
10. DFlash2-style speculative decoding (design at
    `mikealanni/fork-enhancements` commit `fc9b1a33`; model has native
    MTP nextn with `index_share_mtp`): TG-side, pp-neutral — pair with
    the pp ladder, not instead of it.

Order of attack: A5 (bit-exact, cheapest, shrinks the last per-token
loop) → A3 → A2 (decode shipped, staging variants rejected) → A1
(rejected 2026-11-14, compute-bound) → A6 (current) → A4, then promote
B behind one fast-profile flag with its own gate row in BENCHMARKS.md.

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
