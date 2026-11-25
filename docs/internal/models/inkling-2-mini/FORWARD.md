# B. Inkling 2 Mini forward-pass spec (llama.cpp `inkling2_loop`), for a Gufo / gfx1151 re-implementation

Status: read-only research, 2026-11-21. Nothing posted anywhere.
Primary source: llama.cpp at **e412c6f** (contains merged #32694 `29b52d3`, #32657 `a1f4222`, #32429, #32217, #32381, #32430). All `file:line` cites are to that checkout unless prefixed `HF:` (transformers PR #51050, open) or `PR#`.
Marking: **[F]** = fact read from code/headers/PRs; **[I]** = my inference; **[U]** = unverified.

Cross-check done: I compared the llama.cpp graph line by line against the Hugging Face reference in **transformers PR #51050 "Add Inkling2 Mini"** (open, author guarin; adds `inkling2_loop` modeling: `modular_inkling2_loop.py` + base `modeling_inkling2.py` on main). The math matches in every item listed below; differences are only dtype handling (section 5.4). This HF PR is the best available ground-truth description. It dequantizes the NVFP4/MXFP8 experts to bf16 on load [F: PR#51050 comment by guarin].

---------------------------------------------------------------------------------------------------

## 0. Headline architecture [F]

| item | value | source |
|---|---|---|
| physical layers | 24 (`num_hidden_layers`) | off-config.json |
| layer applications ("visits") per token | 48 = prelude L0-5 (6) + core L6-17 x3 passes (36) + postlude L18-23 (6) | `inkling2-loop.cpp:78-114` |
| hidden | 2304; 24 Q heads, 8 KV heads, head_dim 128 (Q width 3072, KV width 1024) | config |
| global (full) attention layers | 5, 11, 17, 23; the other 20 are local SWA (window 1024) | `local_layer_ids` |
| position encoding | **no RoPE**; content-conditioned relative-position bias (d_rel 16, extent 1024) + log-scaled q/r on global layers | `inkling2.cpp:250-262, 340-372` |
| MoE | every layer: 128 routed experts, top-8, inter 1152, sigmoid router + selection bias, route_scale 8, no shared expert, softcapped SwiGLU ("SITU") | `inkling2.cpp:396-412` |
| attention residual | block AttnRes, 6 layers/block (queries per layer, softmax over earlier block outputs) | `inkling2.cpp:178-193` |
| short convs | kernel 4 on K, V, attention output, FFN output (every layer); global layers add input-dependent taps on K,V | `inkling2.cpp:195-238` |
| loop | `core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=false` | config, `inkling2-loop.cpp:7-27` |
| vocab | 201024 padded rows in embed/lm_head; tokenizer 200058; logits cropped after softcap | `inkling2.cpp:425-446` |
| context | 1,048,576 | config |

The pass index has **no parameter of its own** [F]: passes 2 and 3 use identical shared weights and identical per-layer message/controller weights. The pass only changes (a) which cache slot (own KV + own conv state) the visit uses, (b) whether messages exist (pass 1 has none), (c) the message source (the immediately preceding pass only).

---------------------------------------------------------------------------------------------------

## 1. Tensors

Names: HF checkpoint name -> GGUF name (`gguf-py/gguf/tensor_mapping.py`, `conversion/inkling2.py`). Shapes: HF `[out,in]`; GGUF ggml `ne` = (in, out). All sizes verified from the safetensors header of `thinkingmachines/Inkling-2-Mini` rev ed40e7d (`model-00001-of-00001.safetensors` 19,526,019,168 B; `mtp.safetensors` 635,801,416 B) and from GGUF headers (ggml-org IQ4_NL, unsloth Q4_K_XL `Mainline_Support`, unsloth dspark Q8_0).

### 1.1 Global tensors (checkpoint prefix `model.llm.`)

| HF name | GGUF | shape | official dtype | ggml-org IQ4_NL | unsloth UD-Q4_K_XL (mainline layout) | params |
|---|---|---|---|---|---|---|
| `embed.weight` | `token_embd.weight` | [201024, 2304] | BF16 | Q8_0 | Q8_0 | 463,159,296 |
| `embed_norm.weight` | `token_embd_norm.weight` | [2304] | BF16 | F32 | F32 | 2,304 |
| `unembed.weight` (untied) | `output.weight` | [201024, 2304] | BF16 | Q8_0 | Q8_0 | 463,159,296 |
| `norm.weight` | `output_norm.weight` | [2304] | BF16 | F32 | F32 | 2,304 |
| `attn_res.output_query` | `output_res_query.weight` | [2304] | F32 | F32 | F32 | 2,304 |
| `attn_res.attn_queries.{i}` (23) | `blk.{i+1}.attn_res_query.weight` | [2304] SWA layers; [3,2304] global layers (4,10,16,22 -> layers 5,11,17,23) | F32 | F32 | F32 | 71,424 |
| `attn_res.mlp_queries.{i}` (24) | `blk.{i}.ffn_res_query.weight` | [2304] | F32 | F32 | F32 | 55,296 |
| `middle_cycle_cross_pass_queries`, `..._kv_queries` | `blk.{6..17}.loop_msg_query.weight` | each [24,2304] -> per core layer [4,2304] = rows (q,k,v,ffn) | F32 | F32 | F32 | 110,592 |

Query-row packing [F, `conversion/inkling2.py:327-329`]: `query_i = stack(q[2i], kv[2i], kv[2i+1], q[2i+1])` = (Q-input msg, K, V, FFN). Checkpoint `attn_queries.{i}` feeds layer i+1 (`conversion/inkling2.py:192`), so layer 0 has no attention-residual query (input is the normed embedding itself).

### 1.2 Per physical layer (24 layers)

Attention (`inkling2.cpp:106-149`):

| HF | GGUF | shape (HF) | official | ggml-org IQ4_NL | params/layer |
|---|---|---|---|---|---|
| `attn.wq_du` | `attn_q` | [3072,2304] | BF16 | Q8_0 | 7,077,888 |
| `attn.wk_dv` | `attn_k` | [1024,2304] | BF16 | Q8_0 | 2,359,296 |
| `attn.wv_dv` | `attn_v` | [1024,2304] | BF16 | Q8_0 | 2,359,296 |
| `attn.wo_ud` | `attn_output` | [2304,3072] | BF16 | Q8_0 | 7,077,888 |
| `attn.gate_attn` | `attn_gate` | [3072,2304] | BF16 | Q8_0 | 7,077,888 |
| `attn.wr_du` (rel states, 16 x 24 heads) | `attn_rel` | [384,2304] | BF16 | Q8_0 | 884,736 |
| `attn.attn_sink_proj` | `attn_sink_proj` | [24,2304] | BF16 | Q8_0 (BF16 in unsloth) | 55,296 |
| `attn.q_l2_clip.scale` | `attn_q_clip` | [24,128] | BF16 | F32 | 3,072 |
| `attn.rel_l2_clip.scale` | `attn_rel_clip` | [24,16] | BF16 | F32 | 384 |
| `attn.rel_logits_proj.proj` | `attn_rel_bias` (transposed on convert) | [16,1024] (HF) -> ne (16,1024) | BF16 | F32 | 16,384 (same 1024 extent on SWA layers: extent = `n_swa`=1024) |
| SWA layers (20): `attn_norm.weight` | `attn_norm` | [2304] | BF16 | F32 | 2,304 |
| global layers (4): `q_input_norm`,`k_input_norm`,`v_input_norm` | `attn_{q,k,v}_inp_norm` | [2304] x3 | BF16 | F32 | 6,912 |
| `attn.k_sconv.weight` / `v_sconv.weight` (SWA layers) | `attn_k_conv`,`attn_v_conv` | [1024,1,4] -> ne (4,1024) | BF16 | F32 | 4,096 each |
| global layers: `k_sconv.static_w` [4,1024] (flip + transpose on convert) | `attn_k_conv` ne (4,1024) | | BF16 | F32 | 4,096 each |
| global layers: `k_sconv.weight_proj`, `v_sconv.weight_proj` | `attn_k_conv_proj`,`attn_v_conv_proj` | [32,2304] (= 4 taps x 8 kv heads) | BF16 | Q8_0 (BF16 unsloth) | 73,728 each |
| `attn_sconv.weight` | `attn_post_conv` | [2304,1,4] | BF16 | F32 | 9,216 |
| `mlp_sconv.weight` | `ffn_post_conv` | [2304,1,4] | BF16 | F32 | 9,216 |

Attention params/layer = 26,892,288 (matrices) + ~19.8K small = 26,912,128; x24 = 645,891,072 [F, header sum].

MoE (`inkling2.cpp:150-157`):

| HF | GGUF | shape | official | ggml-org IQ4_NL | unsloth Q4_K_XL |
|---|---|---|---|---|---|
| `mlp_norm.weight` | `ffn_norm` | [2304] | BF16 | F32 | F32 |
| `mlp.gate.weight` (router) | `ffn_gate_inp` | [128,2304] | BF16 | F32 | F32 |
| `mlp.gate.bias` (selection bias) | `exp_probs_b.bias` | [128] | F32 | F32 | F32 |
| `mlp.experts.w13_weight` | `ffn_gate_up_exps` ne (2304, 2304, 128) | [128, 2304, 2304]: rows interleaved gate/up (even=gate, odd=up) | L0-19 **NVFP4** (u8 packed, [128,2304,1152]) ; L20-23 **MXFP8** (E4M3 [128,2304,2304]) | L0-19 IQ4_NL, L20-23 Q8_0 | L0-19 NVFP4 + `.scale` [128]; L20-23 Q8_0 |
| `mlp.experts.w2_weight` | `ffn_down_exps` ne (1152, 2304, 128) | [128,2304,1152] | NVFP4 [128,2304,576] / MXFP8 | same split | same split |
| `..._weight_scale` | folded by converter | NVFP4: E4M3 [128,2304,144] (w13) / [128,2304,72] (w2), block 1x16; MXFP8: UE8M0 u8 [128,2304,72]/[.,.,36], block 32 | | | |
| `..._weight_amax` (NVFP4 only) | -> `.scale` in NVFP4-preserving GGUFs | F32 [128] per expert | | | `ffn_*_exps.scale` F32 [128] |
| `..._weight_outer_scale_denominator` | | F32 [1] = **2688.0** (= 6 x 448) both w13 and w2 | | | |

Per-layer expert params 1,019,215,872 (w13 679,477,248 + w2 339,738,624); per expert 7,962,624 (= 3 x 2304 x 1152). x24 = 24,461,180,928.

NVFP4 dequant rule [F, `conversion/inkling2.py:33-65` and header values]: `w = e2m1(nibble) * e4m3_block_scale(16 elems) * (amax[expert] / 2688.0)`; element order: low nibble = even column. MXFP8: `w = e4m3(byte) * 2^(ue8m0-127)` per 32-element block ([32] x [1] block along the input dim; `hf_quant_config` claims block [32,32] but stored shapes say [1,32], the converter warns and uses stored shapes, `conversion/inkling2.py:159-160`). Layers **0-19 NVFP4, 20-23 MXFP8** [F].

Observed amax sample (layer 0): w13 0.121, 0.156, 0.199 ...; w2 0.242, 0.204 ... [F].

Quantization config also states the *activations* are quantized in the native pipeline: NVFP4 `per_token` amax, block 16, E4M3 scales, rule `fouroversix_mae`; MXFP8 block 32 UE8M0 [F, `hf_quant_config.json`]. llama.cpp and the HF reference are weight-only; native serving (vLLM/SGLang) presumably W4A4 on the expert GEMMs [U]. Consequence: no engine here reproduces the activation-quant noise; bit-level agreement with native serving is not a target.

### 1.3 Loop (core layers 6-17 only, 12 layers) (`inkling2-loop.cpp:42-67`)

Per core layer, 4 controllers (q, k, v, ffn; HF names `controllers.{attn,k,v,mlp}.*`):

| GGUF | shape (GGUF ne) | official | ggml-org | params |
|---|---|---|---|---|
| `loop_{q,k,v,ffn}_ctrl_down` | (4608, 64) | BF16 [64,4608] | **F32** (converter forces F32, `conversion/inkling2.py:309-313`) | 294,912 each |
| `loop_{q,k,v,ffn}_ctrl_gain` | (64, 2304) | BF16 [2304,64] | F32 | 147,456 each |
| `loop_{q,k,v,ffn}_ctrl_forget` | (64, 2304) | BF16 | F32 | 147,456 each |
| `loop_msg_query` | (2304, 4) | F32 | F32 | 9,216 |

Loop total 28,422,144 params; stored in F32 in GGUFs = 113.7 MB; read twice per token (passes 2,3) = 227 MB/token [F, I for the byte count].

### 1.4 Parameter budget (official checkpoint, logical params; header-derived)

| component | params | note |
|---|---|---|
| embeddings (embed + unembed) | 926,318,592 | 2 x 201024 x 2304; only the unembed is a GEMM at decode |
| attention (24 layers) | 645,891,072 | q,o,gate 7.08M each, k,v 2.36M, rel 0.88M, sinks/clips/rel-bias small |
| MoE experts | 24,461,180,928 | 128 x 7.96M x 24 |
| router | 7,080,960 | incl. bias |
| loop controllers + queries | 28,422,144 | |
| attention-residual queries | 129,024 | |
| short conv kernels/dyn-conv | 1,228,800 | |
| norms etc. | 133,632 | |
| **text total** | **26,070,385,152** | |
| vision tower + projector | 318,256,128 | 23 ViT layers (12.58M each = 289.4M) + patch/pos embed + proj_in 16.8M + proj_out 9.4M |
| **total (checkpoint)** | **26,388,641,280** | README says 26.4B total [F] |
| MTP drafter (separate file) | 317,866,000 | 3 layers x 87.29M + markov head 55.98M + small |

README: "3.4B active; each extra pass adds ~1.0B; 5.4B effective active" [F]. My count: 24 layers once = 24 x 90.9M = 2.18B + lm_head 0.463B = 2.64B (3.1B with the embedding table counted as the README apparently does); one core pass = 12 x 90.9M = 1.09B (+0.028B controllers) -> **4.88 G MAC/token with 3 passes (+0.46B if counting embed = 5.4B)** [I, consistent with README].

### 1.5 GGUF artifacts (sizes from `dl-list.tsv`)

- ggml-org `Inkling-2-Mini-{BF16 52.2 GB, Q8_0 27.8 GB, IQ4_NL 17.6 GB}.gguf` + `mmproj-Q8_0` 0.35 GB. Arch `inkling2_loop`, 24 physical blocks (704 tensors in IQ4_NL). Experts were **dequantized to F32 and re-quantized** (`conversion/inkling2.py:116-183`: `LazyChunkedTensor` f32); there is **no NVFP4-preserving path in the mainline Inkling converter** (the generic `_repack_nvfp4` in `conversion/base.py:1311-1437` is bypassed because `dequant_model` replaces the expert tensors). IQ4_NL layout: experts IQ4_NL L0-19 / Q8_0 L20-23; attention, lm_head, embed Q8_0; everything small F32 [F, header].
- unsloth `Mainline_Support/*.gguf_file` (744 tensors in Q4_K_XL): **native `NVFP4` ggml type for L0-19 experts + per-expert F32 `.scale` [128]**, Q8_0 for L20-23 experts and dense matrices, BF16 for `attn_sink_proj` and `attn_*_conv_proj`. Needs #32704 on mainline (section 5).
- unsloth `dspark-Inkling-2-Mini-Q8_0.gguf` (0.40 GB; BF16 variant 0.65 GB): 54 tensors, arch `dflash`.
- Q8_0 GGUF header was not parsed (the file size 27.8 GB equals ~all-Q8_0 [I]).

---------------------------------------------------------------------------------------------------

## 2. Forward pass

Notation: `x` = token hidden `[2304]`; `RMS(v)` = `v / sqrt(mean(v^2)+1e-6)` (no weight unless stated); `RMSw(v;w)` = RMS then times `w`. All math below is F32 in llama.cpp (HF keeps fp32 islands but casts the layer output to bf16, see 5.4).

### 2.1 Token in
```
x0 = RMSw(embed[token]; token_embd_norm)        inkling2.cpp:263-268 (norm only if the input is token ids; image-embedding rows are NOT normed)
pos = token position; tau = 1 + 0.1*ln(max(1,(pos+1)/8192))    inkling2.cpp:259-261  (used only by global layers)
blocks = []                                      list of finished block outputs
```

### 2.2 Layer visit `build_layer(il_slot, ip_phys, cur, blocks, messages?)` (`inkling2.cpp:273-421`)

Let `inpSA = cur`. `is_swa = layer ip in local set`.

**A. attention inputs**
1. If `ip>0`: `mixed = AttnRes(blocks, cur, attn_res_query[ip])` (2.3). SWA layer: 1 row -> `in_q=in_k=in_v=mixed[0]`. Global layer: 3 rows -> `in_q,in_k,in_v = mixed[0..2]`. If `ip==0`: all three = `cur`.
2. Pass >=2 only (messages present): `in_j = control(in_j, msg_j, ctrl_j)` for j in (q,k,v) (2.5), including SWA layers (their three inputs diverge here).
3. Norms: SWA: `in_* = RMSw(in_*; attn_norm)` (one shared weight; computed once if no messages, 3x if messages). Global: `in_q=RMSw(.;q_inp_norm)`, `in_k=RMSw(.;k_inp_norm)`, `in_v=RMSw(.;v_inp_norm)`.

**B. projections (all from the post-norm inputs)**
```
Q = Wq in_q            [3072]  -> view [24 heads,128]
K = Wk in_k            [1024]  -> conv_k  -> view [8,128]
V = Wv in_v            [1024]  -> conv_v  -> view [8,128]
R = Wr in_q            [384]   -> view [24 heads,16]
gate = Wg in_q         [3072]
sink = Ws in_q         [24]
```
Short conv with state (2.6) on K and V: SWA layers use static taps; global layers add dynamic taps `weight_proj(in_k)`, `weight_proj(in_v)` (`inkling2.cpp:328-331`).

**C. normalizations / clips** (`clip()` at `inkling2.cpp:164-176`)
```
Q  = L2clip( Q * attn_q_clip[head,chan], thr=22.627 )        per-head, ||.|| + 1e-6, factor=min(1,thr/(||.||+1e-6))
K  = L2clip( K, thr=22.627 )                                 no scale vector
V  = RMS(V)                                                  per kv-head over 128, no weight, eps 1e-6
R  = L2clip( R * attn_rel_clip[head,chan], thr=8.0 )
global layers only:  Q *= tau;  R *= tau                     (K, V not scaled)
```
**D. scores** for query t, head h, key j (distance d = pos_t - pos_j >= 0; GQA: kv head = h/3):
```
s_j = (Q_h . K_kv(h),j) / sqrt(128)  +  b_j      b_j = ( d < extent(1024) ) ? ( R_h . B[:,d] ) : 0     B = attn_rel_bias (16 x 1024)
mask: causal; SWA layers additionally j with d >= 1024 masked (-inf) (hparams.h:572, is_masked_swa STANDARD: p1-p0 >= n_swa)
softmax over { s_j (unmasked) } U { sink logit  t_h = Ws in_q  } ; the sink contributes probability mass but has zero value
o_h = sum_j p_j V_j                                              (output dim 128 per head, 3072 total)
o = o * sigmoid(gate)                                            elementwise (3072)
attn_out = conv_post( Wo o )                                     static conv on the 2304 output (2.6)
```
llama.cpp realizes the sink as: `table = (B_pad^T R) - t` ([1025, heads, tokens]; row 1024 = 0 - t), gathered with `rel_pos` index `min(d,1024)`; the mask/bias tensor is added on top and a zero sink logit is passed to attention (`inkling2.cpp:340-373`; `llama-graph.cpp:4433-4513`). Equivalent to the HF form `softmax([s, t])[:-1]` by shift invariance [F, HF modeling_inkling2.py eager_attention_forward]. Beyond distance 1024 on global layers the bias is **0, not clamped** (HF masks with `distance>=rel_extent -> 0`) [F].

For a high-performance kernel [I]: the 16-dim per-(q,h) vector `R` and the 1024x16 table `B` give `P[h,q,d] = R_hq . B[:,d]` (1025 floats per head/query, the `ggml_mul_mat` at `inkling2.cpp:357`); per key it is one table lookup by distance. A fused flash-attention kernel with (a) lookup bias, (b) sink key, (c) SWA/causal mask, (d) GQA 3:1 avoids the dense `n_head x n_ub x n_kv` mask llama.cpp materializes (section 5.2).

**E. residual / block bookkeeping**
```
if ip % 6 == 0:  blocks.push_back(inpSA)       # block boundary: previous partial sum is frozen, cur restarts at attn_out
else:            cur = inpSA + attn_out
ffn_inp = cur
```
Boundary layers: 0, 6, 12, 18 [F, `inkling2.cpp:380-384`]. Note layer 0 pushes the embedding; layer 6 pushes the sum of layers 0-5.

**F. MoE input**
```
h = AttnRes(blocks, ffn_inp, ffn_res_query[ip])            -> [2304]
pass>=2: h = control(h, msg_ffn, ctrl_ffn)
h = RMSw(h; ffn_norm)
```
**G. MoE** (`build_moe_ffn`, `llama-graph.cpp:2374-2790`; HF `Inkling2TopKRouter`)
```
logits  = Wr_router h                 F32 accumulation forced (expert_router_f32)       [128]
score   = sigmoid(logits)
idx     = top8( score + exp_probs_b )                   # bias only for selection
w       = score[idx] / (sum(score[idx]) + 1e-20) * 8.0  # route_scale 8
for e in idx:  gu = W13[e] h (2304 -> 2*1152; first 1152 = gate g, next 1152 = up u);  (NVFP4: times amax[e]/2688)
               a = 4*tanh(g/4) * sigmoid(g) * 25*tanh(u/25)         # "SITU": gate_softcap 4, up_softcap 25; sigmoid sees RAW g
               y_e = W2[e] a   (1152 -> 2304)
moe = sum_e w_e * y_e                    # F32 accumulate
ffn_out = conv_post_ffn( moe )           # static conv (2.6)
cur = ffn_inp + ffn_out                  # layer output
```
**H. output of the model** (`inkling2.cpp:423-449`)
```
y = AttnRes(blocks, cur, output_res_query)   (1 row)   ->  RMSw(y; output_norm)    # for the last token only when only logits are needed
logits = 30 * tanh( (lm_head y) / 30 )       # F32 accumulate; lm_head rows 0..200057 are used, padded rows dropped after softcap
```

### 2.3 Attention-residual mix (`mix()`, `inkling2.cpp:178-193`; HF `Inkling2BlockAttentionResidual`)
States = {all finished `blocks`} U {current partial `cur`}; `S_j = RMS(state_j)` (unweighted, eps 1e-6, F32). For each query row q: `p = softmax_j( q . S_j / sqrt(2304) )` (no mask, no bias), result `= sum_j p_j S_j` (the **normalized** states, `attn_res_normalize_values=true`). 2-7 states per mix (layer 0: 1; the count grows with the block index). Per token only, no persistent state. Cost per mix: states x 2304 x (queries+1) MACs: negligible.

### 2.4 Loop schedule (`inkling2-loop.cpp:73-114`; HF `Inkling2LoopTextModel.forward`)
```
cur = x0
for ip in 0..5:   cur = layer(ip, slot=ip)                       # prelude, blocks=[emb] after ip=0 pushes, etc.
prelude_cur = cur;  prelude_blocks = blocks (= [emb])             # B0 is pushed later by layer 6 itself
pass 0: for ip in 6..17:  cur = layer(ip, slot=ip)                # no messages
for pass in 1,2:
    bank     = blocks_of_previous_pass[ len(prelude_blocks)+1 : ]        # = the block outputs produced by the previous pass's FIRST core block only,
    bank_out = cur_of_previous_pass_end                                  #   plus the previous pass's final partial output (the 2nd core block's partial)
    cur = prelude_cur;  blocks = prelude_blocks                          # each pass restarts from the PRELUDE output, not from the previous pass output
    for ip in 6..17:
        m = AttnRes([bank..., bank_out], loop_msg_query[ip] 4 rows)      # 4 messages (q,k,v,ffn); softmax over 2 states
        cur = layer(ip, slot = 24 + (pass-1)*12 + (ip-6), messages=m)
for ip in 18..23: cur = layer(ip, slot=ip)                        # blocks come from the LAST pass
out = final mix + norm + lm_head
```
With 6-layer blocks and a 12-layer core, each pass produces two core-block states; the bank always holds exactly 2 states (block 6-11 output B1 and the end partial of 12-17); pass 3 sees only pass 2 (not pass 1) [F]. **Every pass is a separate stream over the whole batch: no token-to-token dependency across passes except through that pass's own KV/conv state**, so prefill can run layer-visit by layer-visit over a whole ubatch. [I]

### 2.5 Message controller (`control()`, `inkling2.cpp:240-248`; HF `Inkling2LoopMessageGate`)
```
g      = concat( RMS(cur), RMS(msg) )                       # [4608]
g      = Down g                                             # 4608 -> 64
gain   = tanh( Gain g ),  forget = tanh( Forget g )         # 64 -> 2304 each
cur'   = (1 - forget) * cur + gain * msg
```
Applied at the three attention inputs (before the input norms) and at the FFN input (before `ffn_norm`). 4 controllers x (4608x64 + 2 x 64x2304) = 2.36M MACs/layer-visit.

### 2.6 Short convolution (`convolve()`, `inkling2.cpp:195-238`; HF `Inkling2ShortConvolution`, `Inkling2DynamicShortConvolution`)
Per channel causal depthwise, kernel 4, taps ordered oldest-first in GGUF (tap 3 = current token). Zero history at sequence start.
```
static (SWA K,V; all attn-out; all ffn-out):   y_t = ( x_t + sum_{i=0..3} w[i,c] * x_{t-3+i,c} ) / sqrt(5)
dynamic (global-layer K,V):                    y_t = ( x_t + sum_{lag} (w_static[c,lag] + dyn[h(c),lag]) * x_{t-lag,c} ) / 3      # dyn = weight_proj(in_k or in_v), [4 taps x 8 kv heads], 128 channels share a tap
```
Conv input for K/V is the raw projection output **before** L2-clip/RMS. State = last 3 inputs per channel.

### 2.7 One token (decode) vs prefill batch
- Decode: 48 visits sequentially; per visit: AttnRes (tiny), 3-4 projections (q 7.1M, k/v 2.4M each, r 0.9M, gate 7.1M, o 7.1M), attention over that slot's cache, router, 8 expert GEMVs, convs. Pass>=2 adds message mixes + 4 controllers.
- Prefill: the same per layer-visit over the ubatch; conv uses `past` state + batch; attention bias gathers by distance; MoE sorts tokens by expert.
- Position input: log-scale `tau` uses the absolute position; the bias uses cell-position differences (`llama-kv-cache.cpp:2010-2023`: `distance = pos_q - pos_cell`, empty cell or `>= extent` -> index 1024 -> zero bias row).

### 2.8 Vision (mmproj; `tools/mtmd/models/inkling2v.cpp`, `conversion/inkling2.py:332-386`, HF `Inkling2Vision*`)
Patch 20x20x3 = 1200 -> Linear(1200->1024) per patch (4 patches of a 2x2 merge group stored adjacent); + learned 32x32 position table bilinear-resized (align_corners False); + CLS token (`class_embedding + cls_position_embedding`) per image; `ln_pre` LayerNorm; 23 ViT layers: LN -> MHA (16 heads x 64, biases, **2D axial RoPE theta 10000**, no sinks) with layer-scale `ls_1` -> LN -> MLP 1024-4096 GELU(exact) with `ls_2`; drop CLS; 2x2 spatial merge (-> 4096) -> `proj_in` 4096x4096 (no bias) -> GELU(erf) -> `proj_out` 4096->2304 (no bias). Image tokens = (W/40)(H/40), 1..4096 per image; preprocess: no resize unless > pixel limit; pad to multiple of 40 with value -1; mean/std 0.5. Image embeddings enter the text model **un-normed** (no `embed_norm`), use consecutive positions, causal attention [F/I: positions per `mtmd` generic path, not separately verified]. Template wraps `<|content_image|><|unused_200054|><|end_message|>` [F]. Mixed token+embedding ubatches are unsupported for this arch (`llm_arch_supports_mixed_batch` false, `llama-arch.cpp:1559-1561`).

---------------------------------------------------------------------------------------------------

## 3. Persistent per-sequence state

llama.cpp memory = `llama_memory_hybrid_iswa` (`llama-model.cpp:3364-3366,3395-3414`): KV cache (iswa = base cache for global slots + SWA cache for local slots) + recurrent memory (conv state) with filters "all 48 slots" for both. The slot count is `n_layer_all = 24 + 2*12 = 48` (`inkling2-loop.cpp:24-30`): **every visit (incl. repeated core layers) has its own KV and its own conv state**. Nothing else persists: attention-residual blocks, messages and bank are per-token, per-pass transients.

| state | slots | per token per slot | bounded? |
|---|---|---|---|
| global KV (layers 5, 11x3, 17x3, 23) | **8** | K 1024 + V 1024 values = 2048 elems: **4096 B f16 / 2176 B q8_0** | grows with context |
| local SWA KV (20 layers; the 12-layer core contributes 10 SWA layers x3) | **40** | same 2048 elems | window 1024 (+ ubatch headroom: size = pad256(1024*n_seq + n_ubatch), `llama-kv-cache-iswa.cpp:73`; default ub 512 -> 1536 cells) |
| conv state (K 1024, V 1024, attn-out 2304, ffn-out 2304 = 6656 channels x 3 inputs, F32) | 48 | n/a: **79,872 B per slot per sequence** = 3,833,856 B per sequence | constant |

Global KV bytes (8 slots; f16 = 32,768 B/token = 32 KiB; q8_0 = 17,408 B/token = 17 KiB):

| context | f16 | q8_0 |
|---|---|---|
| 1,024 | 33.6 MB (0.031 GiB) | 17.8 MB (0.017 GiB) |
| 32,768 | 1.074 GB (1.00 GiB) | 0.570 GB (0.53 GiB) |
| 200,000 | 6.55 GB (6.10 GiB) | 3.48 GB (3.24 GiB) |
| 204,800 | 6.71 GB (6.25 GiB) | 3.57 GB (3.32 GiB) |
| 1,048,576 | 34.36 GB (32.0 GiB) | 18.25 GB (17.0 GiB) |

SWA KV (40 slots x 4096 B x 1024) = 167.8 MB f16 / 89.1 MB q8_0 steady state (251.7 MB f16 with the default 1536-cell ring); the conv state is 3.83 MB. For comparison a vanilla 24-layer, 8-kv-head, 128-dim GQA model at full attention would need 24 slots: Inkling's mix is 8 full + 40 windowed slots [I].

Notes for the design [I]:
- The per-pass KV could not be shared between passes (K/V are computed from different inputs each pass); co-location of a slot's KV with its shared weights is a multi-GPU concern only (issue #32718).
- Speculative verify needs conv-state rollback; llama.cpp implements it with `n_rs_seq` snapshots (`llama-graph.cpp:4274-4330`, `llama-memory-recurrent.cpp:166+`). The SWA ring must retain `window + n_draft` cells for tail-rollback to be exact; the ubatch headroom in `size_swa` provides it in llama.cpp.
- Prompt-cache slot save/restore at `-c 1M` crashed on Vulkan (comment by jbornschein under #32694: `GGML_ASSERT(tensors[i]->data != NULL)` in `state_seq_get_data`); not reproduced by me.

---------------------------------------------------------------------------------------------------

## 4. MTP / DSpark drafter

**What it is** [F]: the checkpoint ships `mtp.safetensors` + `mtp_config` = a **DSpark-style block drafter** (3 dense layers, block size 8, recurrent "markov" head), not a next-token MTP head. unsloth: "`--mtp` / `draft-mtp` do not apply". The loop arch forbids nextn layers (`hparams.n_layer_nextn` check, `inkling2-loop.cpp:20`).

**Status**: NOT in mainline e412c6f. PR #32218 "Inkling 2 DSpark support" (pwilkin) is OPEN (approved by ngxson once, review by ruixiang63 asks for simplification; pwilkin "cleaned up" 2026-11-21 00:50Z). Works in unsloth's llama.cpp fork (prebuilt b12588-mix-c0c3de8) with `dspark-Inkling-2-Mini-{Q8_0,BF16}.gguf` and `--spec-type draft-dspark --spec-draft-n-max 3` (clamped to 7). One user (RTX 5090, Q4_K_XL + dspark, 2k t/s prefill / 80 t/s decode) reports it runs. Open bug: `--spec-draft-sampling probabilistic` stalls (linear scan over a 200K-entry q; jcsq6 on #32218).

**Drafter weights** (HF `mtp.safetensors`, GGUF arch `dflash`, 54 tensors): all BF16:
- `fuse_weight` [4,4] -> `dspark_fuse` F32: rows 0-2 = per-draft-layer softmax mixing weights over the 4 tapped target features, row 3 = mixing for the recurrent head.
- `mask_embed` [2304] -> `dspark_mask_embd`.
- per layer x3: `input_layernorm`, `self_attn.{q_proj [3072,2304], k_proj, v_proj [1024,2304], o_proj [2304,3072], q_norm, k_norm [128]}`, **context projections** `k_proj_ctx`, `v_proj_ctx` [1024,2304] + `k_norm_ctx` [128], `dflare_norm` [2304] (RMS on the fused feature), `post_attention_layernorm`, dense SwiGLU MLP 2304-9216-2304. Params 87.29M/layer, 261.9M total.
- markov head: `markov_w1` [201024,256] (token -> 256 embedding), `rnn_proj` [768, 5120]+bias (-> `joint_proj`), `hidden_out` [2304,256]+bias, `dflare_norm`. 56.0M.
- Layer attention: 24 heads, 8 KV, head_dim 128, q_norm/k_norm per head, **RoPE** (theta 1e6; `mtp_config.rope_theta`; pairing style not checked [U]), `local_layer_ids [0,2]` SWA 1024, layer 1 global. No bias.

**Target features** (`target_layer_ids = [5,23,41,48]`): indices into the 48 visits. In PR #32218 `res->t_layer_inp[visit]` stores the hidden state **entering** visit i (`inkling2-loop.cpp` diff: `res->t_layer_inp[visit++] = cur` before each `build_layer`), and index 48 = the final post-`output_norm` hidden (`inkling2.cpp` diff: `res->t_layer_inp[n_layer] = cur` after the final norm). So features = input of visit 5 (physical L5, pass 1), input of visit 23 (**physical L11, pass 2**: 18 + 5), input of visit 41 (**physical L17, pass 3**: 30 + 11), and the final normed output. The converter requires `target_ids[-1] == 48 = total depth` (`conversion` diff in #32218). [F for the code convention; whether HF's own `mtp` reference uses the identical convention (input vs output of the visit) was not checked: **[U]**].

**Draft step** (PR#32218 `dspark.cpp:graph_inkling2`, `speculative.cpp`):
1. Context injection (per committed/verified target token): `ctx = RMSw( sum_k softmax(fuse[l])_k * feat_k ; dflare_norm_l )`, `K_ctx = RoPE(RMSw(Wk_ctx ctx; k_norm_ctx))`, `V_ctx = Wv_ctx ctx` written to the draft KV cache for each of the 3 layers (draft KV = 4096 B f16 per layer-token; 2 SWA layers bounded, 1 global layer grows 4 KiB/token).
2. Draft block of 8 rows at positions n..n+7: row 0 = anchor = last accepted token embedding `RMSw(target_embed[id]; token_embd_norm)` (uses the **target's** embedding table), rows 1-7 = `mask_embed`. 3 transformer layers, attention **non-causal within the block** plus the injected context; "later mask rows affect earlier predictions" (the full 8 rows are always evaluated even for `n_max=3`).
3. `base_i = lm_head_target( RMSw(h_i; mtp_norm) )` for all 8 rows (target lm_head, shared). Recurrent chain for i = 1..7 (sequential): `joint = [state(256), markov_w1[prev_token] (256), h_i(2304), ctx_feat(2304)]`; `rnn_proj` -> (gate, cand, latent) of 256 each; `state = cand + sigmoid(gate)*(state - cand)`; `latent' = tanh(latent)`; `corr = hidden_out(latent') [2304]`; `logits_i = 30*tanh( (base_i + lm_head_target(corr))/30 )`; `prev = argmax(logits_i)` (first-maximum tie rule). Anchor row 0 is not a prediction slot (`sample_from_anchor=false`), so up to **7 draft tokens** per step; unsloth defaults to 3.
4. No confidence head (`has_confidence_head=false`; `p_min` must be 0).
5. Verification: ordinary batch verify in the target; target conv state needs rollback of rejected tokens.

Cost note [I]: each draft step reads the target `lm_head` **8 times** (once batched for base_i, 7 sequential times for the chained `corr`): 8 x 0.49 GB at Q8_0 = 3.9 GB, plus the 3 dense layers (0.28 GB Q8_0) - roughly the same bandwidth as one full target decode token (4.0 GB at IQ4_NL). The chain is inherently sequential (`prev` feeds the next markov embedding). Acceptance rates are not reported anywhere I could access [U]; Unsloth only states Q8_0 vs BF16 drafter accept alike and that longer drafts "spend more compute on rejected tokens".

---------------------------------------------------------------------------------------------------

## 5. Correctness: bugs, PR state, oracle

### 5.1 PR / issue ledger (all read-only; treated as untrusted)

| ref | state | what | relevance |
|---|---|---|---|
| #32217 | merged 11-14 | text+vision base (`inkling2` arch), CUDA FA per-head mask changes (`fattn-*.cuh`), mtmd | foundation |
| #32381 / #32294 | merged | chat parser for Inkling 1+2 (`common/parsers/inkling.cpp`, template: roles + `<|content_text|>`, `<|content_thinking|>`, `<|content_invoke_tool_json|>`, `<|end_message|>`, `<|content_model_end_sampling|>`) | serving only |
| #32429 | merged 11-19 | `llm_graph_input_attn_rel_pos`: relative-position attention input | defines the dense per-head mask path |
| #32430 | merged | mmproj conversion skips dequant for bf16 vision | |
| #32657 | merged 11-20 | loads `ffn_gate_up_exps.scale` (exposed as `ffn_up_exps_s`) | NVFP4 per-expert scale |
| **#32694** | merged 11-20 22:01Z (`29b52d3`) | **adds `inkling2_loop`** (`inkling2-loop.cpp`) by pcuenca; reviewed offline by ngxson, CISC | the code under study |
| **#32695** | OPEN | danielhanchen's alternative "looped arch" | see 5.2 |
| **#32704** | OPEN | one-line fix: re-share loop slots after the generic `.scale` pass | NVFP4-scale GGUFs only |
| **#32669** | OPEN | ngxson: auto-apply `.scale` tensors in `build_lora_mm/_mm_id` by weight pointer | see 5.3 |
| #32218 | OPEN | DSpark for Inkling 2 | section 4 |
| #32268 | OPEN | dh follow-ups to #32217 (many items already merged elsewhere) | stale |
| #32707 | OPEN | prefill "abominably slow" 5090; Vulkan 9070 XT 20 t/s tg, 340-400 t/s pp (FA/ubatch irrelevant); 12,402 nodes / 226 splits | perf |
| #32710 | OPEN | **CUDA perplexity = NaN** (`-fa on`, or `-fa off` with n_seq>1, or `-ngl 0` with a CUDA device); CPU and `-fa off -ub 512 n_seq 1` fine; plus a >~7K-token prompt -> repeated-token garbage on CUDA (Mumtasim) | CUDA backend op bug [I: not model-graph logic since CPU is correct] |
| #32718 | OPEN | multi-GPU layer split puts loop slots' KV/conv on a different device than the shared weights (14/36 slots); co-locating: +17.5% pp, +32.5% tg on 2x T4 | multi-GPU only |

#### 5.2 What #32695 does differently from merged #32694 [F, diff read]
Same math (I diffed `control()`, query packing and the bank logic: identical, incl. SWA q/k/v norm duplication). Differences are **structural/packaging**:
- GGUF layout: #32695 keeps arch `inkling2`, `block_count = 48` (one logical layer per visit) + `num_loops`, `loop.core_start/end`, `loop.gate_rank` hparams in `llama_hparams` (`loop_layer()`, `loop_visit()`), generic changes in `llama-model.cpp`, `llama-context.cpp` (graph_max_nodes x n_pass), `llama-model-saver.cpp`. #32694 instead adds a new arch `inkling2_loop` (24 physical blocks in the file; cache slots expanded at load time; `n_layer_all = 48`).
- Cross-pass queries: #32695 stores `loop_res_query` / `loop_kv_res_query` as two model-level tensors; #32694 repacks into per-layer `blk.N.loop_msg_query` [2304,4].
- Unsloth's *root* GGUFs target the #32695-like layout (user report: "`blk.24.attn_q.weight` not found", `block_count=48` with 24 blocks); unsloth's `Mainline_Support/*.gguf_file` re-lay them out for #32694. Mainline chose #32694. **#32695 is effectively superseded; do not use it.**

#### 5.3 #32669 and #32704 [F]
- **Bug fixed by #32704**: `inkling2_loop::load_arch_tensors` copies each physical `llama_layer` into its loop slots **before** the generic `.scale` pass attaches `ffn_{up,down}_exps_s`; slots 24-47 (passes 2 and 3) therefore run NVFP4 experts with **no per-expert scale**. Effect: PPL **130.43 vs 7.653** on unsloth NVFP4-scale GGUFs. ggml quants (IQ4_NL/Q8_0/BF16, no `.scale`) are unaffected. The fix re-copies the slots after the scale pass (6 added lines in `llama-model.cpp`).
- **#32669** (ngxson, 39 files): moves scale application into `build_lora_mm/_mm_id` through `model.w_scales[weight_ptr]`, removes the explicit `*_s` arguments (including Inkling's call at `inkling2.cpp:400-410`), moves the F32 lm_head accumulate to `prec_policy.set_acc(output, F32)`, and moves the router-F32 to `prec_policy` for `expert_router_f32`. Validated with 36 archs on dummy GGUFs (Inkling-2-**Small** only, not the loop model); bit-identical when scales = 1; fixes unrelated NaN archs. Since the scale map is keyed by weight pointer and the loop slots share the pointer, it would also make #32704 moot [I]; but it was **not tested on `inkling2_loop`** and its own "follow-ups" note "shared weights ... can't represent per-logical-layer scales" (for other archs). Unmerged.

#### 5.4 Is the merged code numerically correct?
Evidence that it is **believed correct**:
1. Graph matches the HF reference (#51050) element by element (message gate fp32, block bank = previous pass only, restart from prelude, per-pass cache copies, softcapped SwiGLU, router rule, conv scales `1/sqrt(5)` and `1/3`, clip rule, bias zero beyond extent, sink semantic, `tau`, final cap) [F, I read both].
2. unsloth: UD-Q8_K_XL gives PPL **7.653** on mainline+#32704 vs **7.651** on their own build, KLD 0.0006 [F, `Mainline_Support/README.md`]; the number 7.65 is their yardstick for "correct".
3. ggml-org IQ4_NL (CUDA `-fa off`) PPL 5.34-6.59 per chunk, 6.76 over 40 chunks wikitext-2 ctx 512; CPU agrees within 0.5% per chunk; GSM8K 10/10 [F, issue #32710]. Reviewers pcuenca (author), ngxson, CISC; test-llama-archs covers the arch.
Known differences vs HF/native [F/I]: (a) HF casts each layer output to bf16 (`(residual + hidden).to(dtype)`), llama.cpp keeps the residual in F32 - llama.cpp is arguably closer to an fp32 truth; (b) both are weight-only quant vs native W4A4 (activation quant) [U for native]; (c) router bias/gate: HF keeps gate weight fp32; GGUF F32 as well; (d) ggml experts are re-quantized from F32 dequant, with IQ4_NL (not NVFP4) rounding; (e) the L20-23 MXFP8 layers are Q8_0 in ggml-org files (different block structure than MXFP8: [F] values pass through F32 then Q8_0, so small requant error).
Evidence of **problems** (none in graph logic): CUDA NaN/garbage (#32710), prefill/mem performance (#32707), `-c 1M` slot-save crash, multi-GPU placement (#32718), `special_eos_id` warning. No numerics bug in the graph is known to me.

### 5.5 Recommended correctness oracle
1. **Ground-truth logits**: HF transformers PR #51050 branch (`Inkling2LoopForCausalLM`, bf16 weights dequantized on load, or fp32 for the islands) on CPU or a CUDA box, small prompts (<=1K tokens, plus one >1024-token prompt to exercise global-layer bias extent, SWA eviction, and `tau` > 1 needs pos > 8192: use >8.2K tokens for one test). Compare final logits/KL and, per layer-visit, hidden states.
2. **Engine oracle**: **llama.cpp `e412c6f` (mainline, no extra PRs), CPU backend (`-dev none`), ggml-org `Inkling-2-Mini-BF16.gguf` (52.2 GB; dequantized-then-bf16) and/or `-Q8_0.gguf`**. Use ggml-org files: no `.scale` tensors, so #32704 is not needed. Avoid CUDA for oracle duty (NaN issue). Vulkan on Strix Halo is reported working (jbornschein) but not verified by me.
3. If you want NVFP4-native (`.scale`) GGUFs as an oracle for *native-format* kernels: e412c6f **+ #32704** (and nothing else), unsloth `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (renamed .gguf). Sanity check PPL ~7.65.
4. Do not stack #32695, #32669 (untested on loop), or #32218 (DSpark) into the oracle; use #32218 only as the drafter oracle (unsloth fork b12588 is the working reference).

---------------------------------------------------------------------------------------------------

## 6. Compute and bytes

Definitions: MAC per layer-visit = attention matrices 26,892,288 + 8 experts 63,700,992 + router 294,912 = **90.89M MAC**. Per token: 48 visits = 4.363 G MAC; + controllers 24 pass-visits x 2.359M = 0.057 G; + lm_head 0.463 G (decode and last prefill token only) => **4.88 G MAC = 9.76 GFLOP per decoded token** (weights), shares: experts 62.6%, attention matrices 26.4%, lm_head 9.5%, controllers 1.2%, router 0.3%. Prefill per token (no lm_head): **8.84 GFLOP** + attention-over-context below.

Attention FLOPs (QK + PV, 24 heads x 128): 12,288 FLOP per (query, key, layer-slot). Global 8 slots: 98.3 kFLOP per context key per query -> per decoded token: 1K 0.10 GFLOP; 32K **3.2 GFLOP**; 200K **20 GFLOP** (2x all weight FLOPs); 1M **103 GFLOP**. SWA 40 slots x 1024 keys = 0.50 GFLOP per token, constant. Prefill of N tokens: global attention total = 98.3 kFLOP x N^2/2 (32K tokens: avg 1.6 GFLOP/token on top of 8.8). Relative-bias cost if done by table lookup: 2 x 16 x P table build per (q,head) = 1025 x 16 MAC = 16 kMAC per (q,head) per layer, i.e. 0.4 MMAC per query-visit per layer-slot x 48: ~19 MMAC/token, negligible; if recomputed per key it is 16 MAC/key vs 256 for QK (+6%) [I].

### 6.1 Weight bytes read per decoded token (batch 1, no cache reuse; every visit draws its own 8 experts; assumes 8 distinct experts per visit; layers 0-19 = 44 visits, layers 20-23 = 4 visits)

| config | experts | attn matrices | lm_head | loop ctrl (F32) | router (F32) | **total** |
|---|---|---|---|---|---|---|
| ggml-org IQ4_NL (4.5 bpw exp L0-19, Q8_0 exp L20-23, Q8_0 attn/lm_head, F32 small) | 1.847 GB (46%) | 1.372 (34%) | 0.492 (12%) | 0.226 (5.7%) | 0.057 (1.4%) | **3.99 GB** |
| ggml-org Q8_0 (8.5 bpw everywhere) | 3.249 (60%) | 1.372 (25%) | 0.492 (9%) | 0.226 | 0.057 | **5.40 GB** |
| native-like: NVFP4 0.5625 B/param experts (L0-19), MXFP8 1.031 B (L20-23), BF16 dense (2 B) | 1.839 | 2.582 | 0.926 | 0.226 | 0.057 | **5.63 GB** |
| hypothetical: 4.5-bpw experts everywhere, Q8_0 dense | 1.720 | 1.372 | 0.492 | 0.226 | 0.057 | 3.87 GB |
| hypothetical: 4.5 bpw everything (dense too), loop ctrl still F32 | 1.720 | 0.726 | 0.261 | 0.226 | 0.057 | 2.99 GB |

Observations [I]: (1) at 4-bit experts the dense/attention path (34-47%) and lm_head matter as much as the experts, so a 4-bit-only-experts engine is bounded near ~4 GB/token; keeping controllers/router in F32 costs 0.28 GB/token (7%) and quantizing them to Q8/BF16 would save 0.16-0.21 GB/token (check KL first); (2) going from 4-bit to 8-bit experts adds +1.4 GB (+35%) per token; (3) because the expert set is re-drawn each pass, repeated visits cannot reuse each other's expert pages unless routing overlaps; no measurement exists.
KV traffic per decoded token (global slots only, f16): 32 KiB x context: **1.07 GB at 32K, 6.55 GB at 200K, 34.4 GB at 1M**; q8_0 roughly halves these. Break-even where KV reads equal the 4.0 GB weight reads: ~122K tokens (f16) / ~230K (q8_0). SWA slots add 0.17 GB (f16) per token. Decode becomes KV-bound beyond ~100K context, so global-layer attention kernels (and KV quantization) dominate there.

### 6.2 Prefill [I]
Per ubatch of U tokens every expert is touched (U x 8 / 128 tokens per expert on average: U=512 -> 32), so weight bytes per ubatch ~ whole model per visit: core layers are read **3 times** per ubatch. Per-ubatch traffic ~ total weights + 2 x core weights = 17.6 + 2 x 7.2 = ~32 GB (IQ4_NL), i.e. ~62 MB/token at U=512 (31 MB at U=1024). Compute per token 8.84 GFLOP + attention. FP16/bf16 matrix throughput and the dense `n_head x U x n_kv` mask path are the observed limits: reported pp 340-400 t/s (RX 9070 XT Vulkan), ~200-400 t/s CUDA mid-range GPUs, 2k t/s with unsloth build on 5090 (#32707, comments) vs ~800 t/s for dense Qwen3.8-27B on the 9070 XT at 8K depth. The mask materialization is `2 x n_head x U x n_kv x 4 B` (U=1024, n_kv=89,630 -> 8.8 GB F32 + F16 copy, reported compute buffers 16-38 GiB, `llama-graph.cpp:4496-4510`): a fused kernel removes this entirely.

### 6.3 What dominates, and what changes with quantization
- Short context decode, 4-bit experts: experts 46% / attention matrices 34% / lm_head 12% / loop+router 7% of bytes; MoE FLOPs 63%.
- 8-bit: experts 60%, so expert GEMV efficiency dominates; at BF16 dense the dense path (attn+lm_head) is 62% of bytes.
- Long context (>100K): global-layer KV reads dominate everything; SWA is constant.
- Prefill: expert GEMMs (63% of FLOPs) with the 3x core re-read; global-attention FLOPs per prefill token (98 kFLOP x position) equal the 8.8 GFLOP weight FLOPs at position ~90K; the average over an N-token prefill crosses at N ~ 180K [I].

---------------------------------------------------------------------------------------------------

## 7. Implementation traps checklist (for a new kernel set)

1. Expert layout: HF w13 rows interleaved (gate even / up odd); GGUF is [gate block | up block]; NVFP4 scale = amax[e]/2688; E2M1 nibble order low=even column.
2. Conv taps: GGUF oldest-first; HF `static_w[0]` = current (flip on convert); dynamic taps lag 0 = current; divisor sqrt(5) static vs 3 dynamic; conv inputs are pre-clip K and V.
3. `tau` multiplies Q and R **after** L2 clip, global layers only; K/V unscaled.
4. Bias beyond distance 1024: **zero** (not clamped), including on global layers; table row 1024 = 0 (then minus sink).
5. SWA mask is `p_q - p_k >= 1024` masked => 1024 keys including self.
6. Sink logit from `in_q` (post-norm Q input), shared 24 heads, one logit per (token, head).
7. SWA layers with messages: three controller outputs differ => three `attn_norm` evaluations; K and V inputs differ from Q's.
8. R, gate, sink use `in_q`; K conv dyn-taps use `in_k`; V conv dyn-taps use `in_v`.
9. Block boundary at ip%6==0 uses `ip` (physical), identical in every pass; `blocks` reset to the prelude's `[emb]` at each pass start; message bank = previous pass's `blocks[2:]` + its final partial.
10. Router in F32; weights = **unbiased** sigmoid score of the selected experts, renormalized then x8.
11. Final logits: F32 lm_head accumulate, softcap 30, drop padded rows 200058..201023 **after** the softcap.
12. Embedding norm only for token ids (not image embeddings).
13. Slot-level state: 48 KV/conv slots; do not share across passes; SWA ring needs window + draft headroom.
14. Unsupported in llama.cpp today: mixed token+embd ubatches, `-sm tensor`; CUDA NaN issues; dense bias mask is the perf wall.

## 8. Source index
- `src/models/inkling2.cpp` (460 lines), `src/models/inkling2-loop.cpp` (113), `src/models/models.h:2489-2569`, `src/llama-graph.cpp:1404-1480,2374-2790,4274-4330,4433-4513`, `src/llama-kv-cache.cpp:2010-2023`, `src/llama-kv-cache-iswa.cpp:73`, `src/llama-hparams.cpp:208-217`, `src/llama-hparams.h:563-575`, `src/llama-model.cpp:3364-3414`, `src/llama-quant.cpp:343-352`, `src/llama-arch.cpp:1482-1608`, `conversion/inkling2.py`, `gguf-py/gguf/tensor_mapping.py:69-131,274-603,871-877`, `tools/mtmd/models/inkling2v.cpp`, `tools/mtmd/mtmd-image.cpp:2149-2185`, `common/parsers/inkling.cpp`.
- PRs/issues: ggml-org/llama.cpp #32694 #32695 #32704 #32669 #32218 #32268 #32217 #32429 #32381 #32430 #32657 #32707 #32710 #32718; huggingface/transformers #51050.
- Working data (safetensors headers, GGUF tensor tables, PR and issue dumps) was kept in a scratch directory and is not published.
