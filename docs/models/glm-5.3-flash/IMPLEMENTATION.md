# GLM 5.3 Flash — engine implementation design

Design record for `staging/models/glm5_next`. Tensor names and shapes are
GGUF-measured from `GLM-5.3-Flash-UD-IQ1_M`; formula semantics are
transcribed verbatim from the reference llama.cpp tree (c49edf096):
`src/models/glm5-next.cpp`, `src/models/delta-net-base.cpp`,
`src/llama-graph.cpp`, and verified against the wt semantic anchor
`tests/fixtures/glm_next/wt_reference_seq_5c8854fd8.json` plus float64
stage simulation — never assumed.

## Engine layout

```
staging/models/glm5_next/
  config.{hpp,cpp}         # kv keys incl. kda.*, hyper_connection.*, indexer.*, swiglu_clamp_*
  weights.{hpp,cpp}        # split-aware loader, mixed quant dtype map
  kernels/rocm/
    executor.{hpp,cpp}     # block scheduling, caches, forward/generate, greedy sampling
    device_model.{hpp,cpp} # resident tensor map + per-stage forward (dump/debug axis)
    gemv.hip.cpp iq_tables.hip.hpp   # decode GEMV + IQ lookup tables (gen_iq_tables.py)
    ops.hip.cpp            # rmsnorm(+conv), sinkhorn mHC combine, MoE swiglu-clamp
    attention.hip.cpp      # KDA recurrence, MLA decode, DSA indexer + compressor APE
  CMakeLists.txt
```

## Verified hyperparameters (GGUF `glm5-next.*`, re-read from the shards)

- `block_count = 46` → **45 trunk blocks (il 0–44) + 1 nextn block stored at
  index 45** (`nextn_predict_layers = 1`); reference maps n_layer=45 to
  `LLM_TYPE_320B_A18B`.
- `attention.head_count_kv` array (46 entries): `0` → KDA block, `1` →
  MLA+DSA block. Trunk DSA blocks: il = 3, 7, …, 43 (**11**); KDA: the other
  **34**. Index 45 (nextn) = 1 (full DSA layer).
- `attention.{q_lora_rank=1536, kv_lora_rank=512}`; MLA q-heads 64 × 256
  (NoPE — `rope.dimension_count = 0`); `layer_norm_rms_epsilon = 1e-5`,
  indexer LayerNorm eps = 1e-6 (`layer_norm_epsilon`).
- KDA: `ssm.conv_kernel = 4`, `kda.head_dim = 128`, **64 heads**
  (`ssm_q/k/v` are [8192, 4096] → d_inner 8192 = 64 × 128),
  `kda.gate_lower_bound = -5`.
- Indexer: `head_count = 32`, `key_length = 128`, `top_k = 2048`,
  `kpool = 4`, `index_share_mtp = true`. The optional
  `attention.indexer.types` array is **absent** → default all-ones, i.e.
  every DSA block computes its full indexer; only the nextn block shares
  (`index_share_mtp`).
- mHC: `hyper_connection.{count=4, sinkhorn_iterations=20, epsilon=1e-6}`.
- MoE: `expert_count=288, expert_used_count=8, expert_shared_count=1,
  expert_feed_forward_length=2048, expert_shared_feed_forward_length=2048,
  expert_gating_func=2 (sigmoid), expert_weights_scale=2.5,
  expert_weights_norm=true, swiglu_clamp_exp=10, swiglu_clamp_shexp=10`
  (scalar keys; blocks 0–2 use dense `ffn_{gate,up,down}` 12288).
- `vocab_size = 154880`.

## Weight dtype map (UD mix — measured from all tensors)

Routed experts (3-D, one tensor per block):

| tensor | types seen in trunk | count |
| --- | --- | --- |
| `ffn_gate_exps` / `ffn_up_exps` | IQ2_XXS (21 blk) / IQ1_M (21 blk) / Q2_K (nextn 1) | 43 |
| `ffn_down_exps` | IQ3_XXS (39 blk) / IQ4_XS (il 11, 12, 44) / Q3_K (nextn) | 43 |

Per-block expert type pattern is fixed at load time: gate/up quant type is
identical for gate and up of a block; il 11/12/44 use IQ4_XS down. The kernel
dispatch is **per expert-tensor**, not per block. Non-expert types:
KDA `ssm_{q,k,v}`/`attn_output`/`attn_q_a` Q5_K, `hc_*_fn`/`ssm_beta`/
`ssm_{f,g}_{a,b}` Q8_0, conv/ssm_a/norms F32; MLA `attn_{k,v}_b`,
`attn_kv_a_mqa` Q8_0, `attn_q_b` Q8_0 or F32, `attn_q_a`/`attn_output`
Q5_K/Q4_K per block; indexer weights Q8_0/F32; dense ffn gate/up Q5_K,
down Q6_K; shared expert `*_shexp` Q5_K gate/up + Q6_K down. Total file
≈ 90.9 GB across 3 shards (~1412 tensors).

## Forward (per trunk block, exactly as reference)

1. **mHC pre-mix** (hc_mult = 4, sinkhorn iters 20, eps 1e-6; streams are
   [4 × 4096] F32, `hc_{attn,ffn}_fn` [16384, 24]):
   `mixes = hc_fn · rms_norm(streams, eps 1e-5)`; slices per slice-role:
   `pre = sigmoid(mixes₀ · s₀ + base₀) + 1e-6`;
   `post = sigmoid(mixes₁ · s₁ + base₁) · 2`;
   `comb = affine(mixes₂ · s₂ + base₂)` viewed [4][4][nt]. Sinkhorn over
   `comb`: row-softmax over dst (dim of 4), eps, normalize columns (+eps),
   then 19 more (row normalize +eps, column normalize +eps). Block input =
   `Σ_src streams_src ⊙ pre_src`; residual write-back =
   `streams·post_dst + Σ_src res_src · comb[dst][src]`.
2. **KDA blocks** (34; formula transcribed from `delta-net-base.cpp` +
   the glm5 branch):
   `q/k/v = {ssm_q, ssm_k, ssm_v}·x` (each 8192) → concat causal conv1d
   kernel 4 with per-stream `ssm_conv1d_{q,k,v}.weight` [4, 8192] → silu →
   reshape [128, 64].
   Gate: `g1 = ssm_f_b(ssm_f_a(x)) + ssm_dt.bias` (8192 ch) reshaped
   [128, 64·heads]; per head A = `ssm_a[h]` (= −exp(A_log));
   `g = −5·sigmoid(−(g1·A))` clamped at the lower bound −5 branch exactly as
   the reference (kda gate_lower_bound = −5). `beta = sigmoid(ssm_beta·x)`
   per head. Q/K L2-normalize (eps 1e-6, "match FLA"), then q scaled 1/√128.
   Recurrence per head (autoregressive form; fused variant is equivalent):
   `state←state·exp(g)`; `sk = Σ_i state_i·k_i`; `d = beta·(v−sk)`;
   `state_i += k_i·d`; `o = Σ_i state_i·q_i`. State fp32 [128, 128, 64] →
   **4.194 MB/layer, 142.6 MB per sequence** over 34 KDA layers.
   Output: `g2 = ssm_g_b(ssm_g_a(x))`;
   `y = rms_norm(o, ssm_norm[128]) ⊙ sigmoid(g2)` → `attn_output` [8192→4096].
3. **MLA+DSA blocks** (11 + nextn; NoPE, rope dimension 0):
   `q = attn_q_b(rms_norm(attn_q_a·x))` → 1536 → 16384 = 64 heads × 256;
   `kv_latent = rms_norm(attn_kv_a_mqa·x)` (512). Query absorbed per head:
   `q_abs[h] = attn_k_b[h]·q[h]` (256 → 512), softmax scale
   `1/√256 = 1/√n_embd_head_k_mla` (the reference `kq_scale`, folded into
   the absorbed q rows).
   **Cache row = the 512 latent only** (single KV head; `attn_v_b[h]`
   [512→256] is a weight applied inside the attention over the gathered
   latent rows, V is never cached separately). Output 64 × 256 = 16384 →
   `attn_output` → 4096.
   Indexer (per DSA block, full): `iq = indexer.attn_q_b(lm_head-side qr
   · rms of [1536])` → 4096 = 32 × 128;
   `ik = LN(indexer.attn_k·x)` with `k_norm` w+b eps 1e-6;
   `gate = indexer_compressor_gate·x` (128). Cached rows pack
   `[ik | gate]` 384-wide (+ F32 mirror pool). Pooling (kpool 4):
   `probs = softmax(indexer_compressor_ape[128, 4] + pooled-gate-part, dim
   over 4)`, `k_pooled = Σ probs·k_pool`;
   `score = Σ_heads relu(iq_h · k_pooled) · w_h` with `w = indexer.proj·x`
   [4096→32] · 1/√(128·32); select top `2048/4 = 512` pool groups
   (`n_top_pool`, no tail select: `kpool_select_tail` key absent → false),
   expand each pool to its 4 token slots → 2048 token slots as the additive
   mask for the main MLA attention (−INF outside selection + causal).
4. **MoE**: router `ffn_gate_inp` → sigmoid → + `exp_probs_b.bias` →
   top-8/288 → renormalize → ×2.5; `fused_experts` semantics;
   SwiGLU clamp is exactly `gate'=min(gate,10)` then
   `out = gate'·sigmoid(gate')·clamp(up,−10,10)` (gate clamped from above
   only) for routed (`swiglu_clamp_exp`) and shared expert
   (`swiglu_clamp_shexp`), both = 10. Blocks 0–2: dense `ffn_{gate,up,down}`
   trio (same clamp values).
5. **mHC post-mix** around each sub-layer (same builder, `hc_ffn_*`), and
   per token: `head = Σ4 streams / 4` (hc_mean) → `output_norm` → lm_head.

## MTP / nextn block (index 45)

Own full DSA+MoE layer, plus `nextn.{enorm,hnorm}` [4096] F32 RMS,
`eh_proj` [8192→4096] Q8_0 and `shared_head_norm` [4096]. Standard single
residual (no mHC on this block). Input = concat `enorm(hidden) ‖ hnorm(emb)`
→ eh_proj → layer → shared lm_head after shared_head_norm.
`index_share_mtp = true`: the draft reuses the last block's indexer
selection; the block ships its own indexer tensors.

## Cache sizing (per token / per sequence)

| Cache | Size | Notes |
| --- | --- | --- |
| MLA latent | 512 × 2 B × 12 blocks = 12.3 KB/tok | latent-only rows, F16 mirror (+F32 canonical if KV policy keeps it) |
| Indexer pool | 384 (F32) per input token ÷ 4 (pooled rows) ≈ 0.38 KB/tok/block | 11 full pools |
| KDA state | **4.194 MB/layer × 34 = 142.6 MB per sequence** | fp32 [128,128,64]; dominates context-memory growth |
| lm_head | 154,880 × 4096, F32 | decode GEMV ≈ 2.5 GB/token |

## IQ-expert MoE kernel (the genuinely new code)

- Bucket per expert tensor: gate/up is **IQ2_XXS or IQ1_M** (21 blocks
  each), down is **IQ3_XXS or IQ4_XS**; the nextn block's Q2_K/Q3_K trio
  may stay on the generic path (speculative decode only). Dispatch on the
  bucket's quant type.
- Path per common-doc §2: global IQ block load → LDS → dequant to F16
  → `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` → swiglu-clamp epilogue
  in-register. No dp4a inner loops.
- IQ dequant tables (`iq2xxs_grid`, `iq1m_sgnmap`, `iq3xxs_grid`) fit
  constant memory. IQ4_XS is grid-free (16-bit codes → `iq4_k_data`) and
  uses the magic-number `0x6400|q` trick.
- Decode (batch 1): one-wave-per-row GEMV (`vec_dot_iq*_q8_1` semantics)
  with multi-weight fusion over the top-8 trios (the 8 experts share x);
  x is quantized to Q8 once per (block, token). MMVQ tile tables retuned
  on gfx1151 (PLAN phase 3); IQ2 dequant is the heaviest read path.

## Quality gate (live fixtures)

The strict gate is the gufo **self-regression golden**
`tests/fixtures/glm_next/llama_reference.json` (schema
`gufo.glm_next.llama_reference.v2`): 5 case histories, top-32 logits after
prefill and after the 24-token greedy chain.
`rocm_forward_test` replays it with exact top-1/greedy checks plus a
0.5-nat budget, and additionally runs per-operator device checks against
analytic and CPU-simulated references (mHC sinkhorn vs wt's fused CPU
reference, indexer pooling/top-k selection, KDA recurrence state, IQ GEMV
vs dequant oracle).

Cross-engine semantics use `wt_reference_seq_5c8854fd8.json` (llama.cpp
per-position capture via `tools/glm5_parity/llm_probe`) as an anchor, not
as a gate: gufo's logits are faithful to float64 where wt's GPU kernels
deviate (wt lm_head alone 4.65e-3 relative), and mHC stream drift makes
strictly tighter budgets chaotic, so wt-parity is assessed by
distributional sanity (shared top-32 overlaps, flip positions at coin-toss
margins) instead of a logit budget. Fixture refresh contract, capture modes
and reproduction commands: [BENCHMARKS.md](BENCHMARKS.md#fixtures). Each
kernel promotion additionally requires matched ABBA timings (5% / 3 ms
margins per phase).

## Resolved / remaining unknowns

- ~~Indexer per-layer full/sharing array~~ → absent in GGUF → all trunk DSA
  layers full; nextn shares per `index_share_mtp`.
- ~~KDA 32 heads~~ → **64 heads, d_inner 8192**.
- ~~latent+kpe 576 cache row~~ → **latent 512 only** (NoPE).
- MTP acceptance policy and draft token plumbing follow the generic
  nextn/MTP path once Phase 4 enables it (draft head is present in-file).
- Remaining to verify during implementation: exact tie-breaking order in
  the indexer `argsort` (reference uses a stable `ggml_argsort` desc), and
  numerical budget per op against fp32 fixtures.
