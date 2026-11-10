# GLM 5.3 Flash (`glm5-next`)

Hybrid KDA/DSA Mixture-of-Experts text/vision model on gfx1151, from Z.ai
(formerly the anonymous "Ox Alpha"). 320B total / 18B active parameters; the
measured IQ1_M staging target is 97.57 GB over 1412 tensors in three shards
plus an F16 projector. The staging engine `staging/models/glm5_next` loads
those shards and runs the full graph on gfx1151 behind a strict
self-regression fixture (see [BENCHMARKS.md](BENCHMARKS.md#fixtures));
`gufo serve` registration and decode tuning are still open. Track
progress in [PLAN.md](PLAN.md) and the engine design in
[IMPLEMENTATION.md](IMPLEMENTATION.md). The shared technique catalogue for
kernel work is [../../COMMON_OPTIMIZATION_STRIX_HALO.md](../../COMMON_OPTIMIZATION_STRIX_HALO.md).

This model combines almost every component Gufo already ships: the
DeepSeek-V4 family engine (MLA, mHC, DSA indexer, MoE) and the Flash-Next
engine (KDA/GDN recurrence, sparse-attention indexing, MTP speculation).

## Load and run (planned; loader not implemented)

```sh
MODEL=/home/praburaja/projects/llm/models/gguf/GLM-5.3-Flash-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf
MMPROJ=/home/praburaja/projects/llm/models/gguf/GLM-5.3-Flash-IQ1_M/mmproj-F16.gguf
./result/bin/gufo serve llm --model "$MODEL" --context 32768   # after phase 1
```

The metadata lives entirely in shard 00001 (which contains no tensor table
entries); tensor tables live in the data shards (742 + 670 entries,
`split.tensors.count = 1412`). The loader must concatenate local tables
rather than trusting the first header (llama.cpp handles this layout).

## Architecture facts (GGUF-measured, cross-checked against the release notes)

| Group | Values |
| --- | --- |
| Layers | `block_count = 46` = **45 trunk blocks (il 0–44) + 1 nextn block at index 45** (`nextn_predict_layers = 1`); blocks 0–2 use dense FFN (`feed_forward_length = 12288`); every block carries mHC tensors |
| Attention pattern | 3 × KDA linear-attention : 1 × MLA sparse-attention (per-block `attention.head_count_kv`): trunk DSA blocks il = 3, 7, …, 43 (**11**), **34 KDA**; nextn (index 45) has its own full DSA+MoE layer |
| KDA (linear) | **64 heads**, `kda.head_dim = 128` (`ssm_{q,k,v}` are [8192, 4096], d_inner 8192), `kda.gate_lower_bound = -5.0`, causal conv kernel 4; per-head fp32 state 128×128 (4.194 MB/layer, 142.6 MB/sequence); tensors `ssm_conv1d_{q,k,v}`, `ssm_beta`, `ssm_dt.bias`, `ssm_a`, `ssm_{g,f}_{a,b}`, `ssm_norm` |
| MLA (sparse) | `q_lora_rank = 1536`, `kv_lora_rank = 512`, 64 q-heads × 256 (`key_length_mla = value_length_mla = 256`), **NoPE** (`rope.dimension_count = 0`, norms only); **cache row = the 512 latent only** (`attn_{k,v}_b` [512, 256, 64] per-head weights are applied around the latent, no rope/V stored); tensors `attn_{q_a,q_b,kv_a_mqa,k,v}_{norm,weight}` |
| DSA indexer | 32 heads × 128 key, `top_k = 2048`, `kpool = 4`, `index_share_mtp = true`; tensors `indexer.*`, `indexer_compressor_{ape,gate}` (absolute-position compressor on pooled keys) |
| MoE | 288 experts, sigmoid gating top-8 (`expert_weights_scale = 2.5`, `expert_weights_norm = true`), 1 shared expert, both FFN widths 2048, `swiglu_clamp_{exp,shexp} = 10` on all blocks |
| mHC | `hyper_connection.count = 4` streams, sinkhorn iterations 20, epsilon 1e-6; per-block `hc_{attn,ffn}_{base,fn,scale}` |
| Speculation | `nextn_predict_layers = 1` (nextn MTP; shares the indexer) |
| Sizes | `context_length = 1,048,576`, `embedding_length = 4096`, `feed_forward_length = 12288`, RMS eps 1e-5 |
| Tokenizer | gpt2 BPE, `pre = glm4`, vocab 154,880, 321,649 merges; eos/pad/unk 154820, bos 154822, eot 154827 (template `[gMASK]<sop>`), eom 154829; chat template carries a `reasoning_effort ∈ {low, high, max}` system block and `clear_thinking` |
| Vision | `mmproj-F16.gguf` (`clip`, projector `glm5v`): 24-layer ViT, d 1024, FFN 4096, patch 14, 448², silu + swiglu clamp 10, merge 2, projection 4096 |
| Quantization | Unsloth dynamic (imatrix), `general.file_type = 31`, per-block expert types: routed `ffn_{gate,up}_exps` **IQ2_XXS (21 blocks) or IQ1_M (21)** + nextn Q2_K; `ffn_down_exps` **IQ3_XXS (39) or IQ4_XS (il 11, 12, 44)** + nextn Q3_K — expert tensors ≈ 93.4 GB of the 97.57 GB; KDA/MLA projections mostly Q5_K/Q8_0, dense + shared FFN Q5_K/Q6_K, embed & lm_head Q4_K, norms F32 |

Decode traffic (computed from the tensor table): ~2.53 GB/token of expert
weights (288-slice × top-8 over 43 MoE blocks) plus ≈0.1 GB shared/attention
state — the streaming ceiling at 240 GB/s is ~90–95 tok/s before attention
and bookkeeping. IQ1_M at 1.75 bpw makes decode overhead-bound, so launch
fusion and GEMV shaping (common-doc §2) dominate, and dp4a inner loops stay
banned (55× slower per MAC on gfx1151).

## Reference availability

- llama.cpp (`/home/praburaja/projects/llm/llama.cpp/llama.cpp` @ c49edf096)
  implements `LLM_ARCH_GLM5_NEXT`: KDA KV keys (`kda.head_dim`,
  `kda.safe_gate`, `kda.gate_lower_bound`), mHC tensor family
  (`hc_attn_base`, …), indexer-compressor tensors, quant guards
  (`llama-quant.cpp:343,489`), graph wiring with the DeepSeek-V4/DFLASH
  mHC branches. Use it for parity fixtures and first-run baselines.
- Release docs: Z.ai model card (320B-A18B, native multimodal), Raschka
  architecture notes (Aug 2026: Kimi-Linear-style 3:1 KDA:MLA/DSA, GLM-5.2
  MoE scale-down, DeepSeek-V4-style four-stream mHC). Native FP8 checkpoint
  is upstream; the IQ1_M UD build is the gguf-side staging artifact.

## Status

- [x] Phase 0 — reference-run baseline on Strix Halo (llama.cpp @ c49edf096) —
  pp2048 189.44 ± 5.11 Mtok/s, tg32 16.80 ± 0.02 t/s
  ([BENCHMARKS.md](BENCHMARKS.md)); sequential wt captures archived as the
  semantic anchor `tests/fixtures/glm_next/wt_reference_seq_5c8854fd8.json`
- [x] Phase 1 — `staging/models/glm5_next` split-aware loader + full-graph
  executor (the CPU scalar parity path is superseded by the wt anchor plus
  independent float64 simulation, see [BENCHMARKS.md](BENCHMARKS.md#fixtures))
- [x] Phase 2 — ROCm kernels (GEMV, ops incl. mHC sinkhorn, KDA/MLA+DSA
  attention, device model) — `glm5_next_*` ctest suite 5/5 against the gufo
  self-regression golden `llama_reference.json`, replay bit-for-bit; MTP
  speculation still wired in phase 4
- [ ] Phase 3 — IQ1_M decode GEMV/MMVQ tuning and fusion, serve registration
- [ ] Phase 4 — long-context objectives, quality and benchmark docs

[BENCHMARKS.md](BENCHMARKS.md) carries the llama.cpp reference timings and
the fixture provenance contract. No [QUALITY.md](QUALITY.md) yet: a
missing-model skip is not a quality pass.
