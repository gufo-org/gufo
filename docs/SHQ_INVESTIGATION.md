# SHQ-T16 on Strix Halo: DeepSeek V4 Flash and Qwen3.8-27B

Status: investigation, 2026-09-13. Numbers marked *measured* come from this
host (Strix Halo `gfx1151`, 128 GB) and the artifacts under `/persist/models`;
numbers marked *derived* are arithmetic on those measurements; everything else
cites a source. This document does not change any contract; it evaluates
whether and where `SHQ-T16` (see [QUANTIZATION.md](QUANTIZATION.md)) pays for
the two production models and what it would take.

## 1. Summary

1. **Decode on Strix Halo is a bytes-per-token problem.** Measured DRAM read
   ceiling is 241 GB/s; the Qwen3.8-27B Q8 decode already sustains 209 GB/s and
   the DS4 Flash decode sustains about 169 GB/s (derived). Every format decision
   therefore reduces to: how many bytes does one token read, and does the kernel
   stay at the memory roofline while decoding them. Bit width and layout matter;
   the quantization *algorithm* matters only through quality.
2. **DS4 Flash reads 9.6 GB per token, and 81% of it is not the experts.** The
   current artifact keeps every dense tensor at Q8_0 or F16 (attention 5.2 GB,
   shared expert 1.15 GB, indexer/compressors 0.62 GB, output head 0.56 GB) while
   the six routed experts contribute 1.8 GB. Moving the dense set to a 4-bit
   tile format cuts the per-token read to 5.6 GB (-41%), a decode ceiling of
   30 tok/s at today's kernel efficiency against 18 tok/s now. Measured
   (section 6.4, llama.cpp KL against the shipped artifact on WikiText-2):
   4-bit attention costs mean KL 0.092 and +6.3% perplexity, 5-bit costs KL
   0.049 at baseline perplexity, 6-bit KL 0.035 at +1.0%. The MLA projections
   are the sensitive tensors; a 5- or 6-bit dense tier (+18–29% decode) is
   the defensible recipe, and 4-bit is not.
3. **DS4 Flash experts cannot be SHQ4.** The official checkpoint stores expert
   weights as QAT MXFP4 (4.25 bpw); at that width the experts alone are 137 GiB.
   On 128 GB the routed experts must stay at or below ~3.0 bpw, practically
   2.2–2.6 bpw. SHQ as specified (4/6/8) has no tier there. Measured on the
   pinned 0731 checkpoint, the shipped IQ2_XXS/Q2_K experts sit at 35% / 28%
   RMS error against the official MXFP4 values, every 2-bit scheme lands at
   30–35%, and the third bit halves the error to 16–18%. A 2-bit tile tier
   without a codebook (32.6%) plus a 3-bit tier on `w2` (17.6%) fits in the
   same 78 GiB as today; that is the DS4 expert plan worth testing.
4. **Qwen3.8-27B gains modestly on decode and more on prefill.** UD-Q4_K_XL is
   5.14 bpw; a pure SHQ4 U4Z G64 artifact is 4.31 bpw, worth +16–19% decode
   throughput at equal kernel efficiency (11.6 → 13.5–13.8 tok/s, derived). The
   measured Q4 prefill deficit (474 vs 557 tok/s for Q8) is 79% the K-quant
   minimum correction; SHQ4's per-64 zero correction folds into one activation
   sum per group, and the S4 variant removes it entirely. A register-resident
   issue-rate model (section 6.3) puts the SHQ4-G64 W4A8 inner loop at 45 TOPS
   against the Q8_0 W8A8 loop's 41, so SHQ4 prefill should land at or above
   the Q8 rate instead of 15% below it.
5. **The GPU/NPU shared-layout rationale is not supported by this host's
   measurements for the main model.** An AIE program streams 47 GB/s and reaches
   ~11 TOPS only with the per-group epilogue removed (0.94 TOPS with it); every
   main-model offload route measured net zero or negative. The one-resident-copy
   property is valuable for a *companion* model on the NPU (ASR, a small chat
   model), where a +23% main-model gain was measured, not for splitting DS4 or
   Qwen3.8-27B. The XDNA2 native `bfp16` block format (8 values, one shared
   exponent) suggests an epilogue-free NPU tier worth one microbenchmark.
6. **The layout decodes at the roofline; the quantizer is the gap.** A
   scratchpad GEMV on the contract layout streams SHQ8 at 84–95% and SHQ4 at
   80–92% of the 241 GB/s ceiling on the Qwen FFN shapes (section 6.3), so
   the byte projections above are realistic; a lane-contiguous group order
   is worth another 3–9% and should be decided before the contract freezes.
   On
   Qwen3.5-0.8B the SHQ4 candidate reconstructs tensors at ~10% relative MAE
   where unsloth's Q4_K sits at ~7%; at production scale (section 4.3) SHQ4
   range search is 9.9% against Q4_K's 8.0% on the same tensors of both
   models. At whole-model level (section 4.2) every uncalibrated uniform
   4-bit class costs Qwen3.8-27B mean KL 0.044–0.051 while the imatrix-tuned
   mixed UD-Q4_K_XL costs 0.017: calibration and recipe, not block format,
   are the 2.7x. Closing that needs error feedback (GPTQ-style) or
   per-block scale search of the kind llama.cpp's K-quants already do; the
   T16 layout does not constrain either.

## 2. What SHQ-T16 is today

Contract recap (normative text lives in [QUANTIZATION.md](QUANTIZATION.md)):

| Tier | Codes | Scale | Zero | Group | bpw | Status |
| --- | --- | --- | --- | --- | ---: | --- |
| SHQ4-T16-G64-U4Z | UINT4 | BF16 per (lane, K64) | UINT4 | 64 | 4.3125 | quantizer, conformance vectors, CPU W4A8 reference, NPU program `qwen_aie2p_w4a8` |
| SHQ4-T16-G64-S4 | INT4 | BF16 | none | 64 | 4.25 | same |
| SHQ6-T16-G64 | INT6 (3 B / 4 w) | BF16 | none | 64 | 6.56 | quantizer, conformance |
| SHQ8-T16-G64 | INT8 | BF16 | none | 64 | 8.25 | quantizer, conformance, AIE2P W8A8 reference (#99) |

Physical unit: one `K16 x N16` microtile, 128 bytes for 4-bit, 128-byte
aligned, output-major within 16 lanes; scales and zeros in separate SoA planes.
GGUF carries the tiers as ggml types 1000–1002 (`src/core/gguf_reader.hpp`).

What does **not** exist: no HIP kernel consumes an SHQ tensor. In
`src/models/qwen/gemm_route.hpp` the three SHQ types return `{.quantized =
true}` with no `cpu_direct`, `hip_decode_direct` or `hip_prefill_direct`
route, so a Qwen artifact with SHQ tensors is rejected by the GPU runtime. The
production Qwen path decodes GGUF K-quants in-kernel (Q8_0 W8A8 WMMA prefill;
Q4_K/Q5_K/Q6_K/IQ4_XS decode GEMV and small-batch exact kernels). DS4 has its
own imported IQ2_XXS/Q2_K/Q8_0 kernels and no SHQ path. The only hardware
consumer of an SHQ layout so far is the XDNA2 W4A8 program for the Qwen MTP
projection.

The Qwen3.5-0.8B evidence ([benchmarks/qwen3.5-0.8b](../benchmarks/qwen3.5-0.8b/README.md)):
uniform SHQ4 with imatrix scale search reaches matched-token KL 0.117 against
the bf16 teacher; the mixed recipe `shq6_mirror` (ffn_down, embed and
linear-attention projections at SHQ6) reaches KL 0.050 at 550 MB against
unsloth Q4_K_M's 533 MB. Per-tensor reconstruction error is 10.1–10.7% relative
MAE for SHQ4 versus 6.7–7.1% for Q4_K in the same tensors.

## 3. The hardware, as measured here

| Resource | Value | Source |
| --- | --- | --- |
| DRAM read / write / copy | 241 / 220 / 209 GB/s | `tools/bench/gfx1151_peak` (benchmarks/qwen3.8-27b) |
| Practical decode ceiling reached | 209 GB/s (Qwen Q8 decode), 87% of read | measured |
| WMMA 16x16x16 INT8 | 55.07 TOPS | measured |
| WMMA 16x16x16 BF16 | 55.05 TFLOPS (same rate as INT8) | measured |
| VALU FP32 FMA | 27.08 TFLOPS | measured |
| WMMA operand formats | f16, bf16, iu8, iu4 in; f32/f16/bf16/i32 out; 16x16 tiles only | [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/) |
| Matrix core | none separate: WMMA issues on the SIMD32 VALU, epilogue VALU ops subtract from matrix throughput | benchmarks/qwen3.8-27b |
| hipBLASLt BF16 GEMM 17408x5120x2048 | 25.75 TFLOPS (47%) | measured |
| XDNA2 array | 4 x 8 AIE2P tiles, 64 KB L1 per tile, 512 KB L2 per column, shim DMA per column | [XDNA2 overview](https://www.emergentmind.com/topics/amd-xdna-2-npu) |
| XDNA2 peak | 50 TOPS INT8, ~25 TFLOPS BF16; native int8/int16/bf16 and `bfp16ebs8` (blocks of 8, one shared 8-bit exponent, 1 sign + 7 mantissa bits per value, 9 bpw; an 8x8 BFP16 MAC at 512 MAC/cycle per tile) | same; [AMD Quark BFP16](https://quark.docs.amd.com/release-0.9/pytorch/tutorial_bfp16.html) |
| XDNA2 measured streaming | 47 GB/s per AIE program (~5.9 GB/s per column) | benchmarks/qwen3.8-27b |
| XDNA2 measured arithmetic | ~11 TOPS W8A8 without per-group epilogue, 0.94 TOPS with it | same |
| AIE2P mixed matmul | `mmul_8_4`: INT8 x INT4, microtile M4 K16 N16 | [aie_api mmul_8_4](https://github.com/Xilinx/aie_api/blob/main/include/aie_api/detail/aie2p/mmul_8_4.hpp) |
| Unified memory | 128 GB LPDDR5X; DS4 C1 workload holds 90.7 GiB with an 80.8 GiB model | benchmarks/deepseek-v4-flash |

Consequences that every format proposal has to respect:

- **Decode is bandwidth-bound.** A single-token step reads every dense weight
  once. Throughput is `effective GB/s / bytes per token`. The kernel work is to
  keep the read coalesced and outstanding; the format work is to minimize bytes
  and keep dequantization off the critical path.
- **Prefill is issue-bound on the VALU.** With no separate matrix core, every
  per-group scale multiply, zero correction or lookup competes with WMMA for
  the same issue slots. The Q8_0 W8A8 kernel plateaus at 58% of the WMMA
  ceiling and the ablation that removes the per-32 activation scale epilogue
  reaches 64%; the K-quant minimum correction costs a further 8.6% of `pp2048`.
  A format with fewer, cheaper epilogue terms per K element is a prefill
  format.
- **`iu4` WMMA exists but is W4A4.** Using it needs 4-bit activations, which is
  a rotation-plus-calibration problem (QuaRot/SpinQuant class), not a layout
  problem. The realistic 4-bit prefill path on gfx1151 is W4A8: unpack nibbles
  to bytes in registers and issue `iu8` WMMA, which is exactly the arithmetic
  the NPU contract already specifies (`int32 dot - z * sum(a)`, FP32 group
  partials). The same INT32 sums on both devices is what makes an exact GPU/NPU
  split possible at all.
- **The NPU's problem is data movement and epilogue, not peak TOPS.** 47 GB/s
  is 20% of the GPU's device path, and the group-scale epilogue divides its
  arithmetic by ten. A format whose scale is applied once per larger block, or
  whose block exponent the hardware applies natively (`bfp16`), is the only
  kind of format that changes the NPU picture.

## 4. Model anatomy and byte budgets

### 4.1 DeepSeek V4 Flash

Architecture ([DeepSeek-V4 report](https://arxiv.org/html/2606.19348v1),
`runtime/model_data_internal.h`):

| Item | Value |
| --- | --- |
| Layers / hidden | 43 / 4096, vocabulary 129,280 |
| Residual | mHC, `n_hc = 4` streams, 20 Sinkhorn iterations; `hc_attn_fn`/`hc_ffn_fn` tensors [16384, 24] per layer |
| Attention | 64 query heads, head dim 512 (one latent KV head), partial RoPE 64, query LoRA rank 1024, grouped output projection g=8, d_g=1024 (`attn_output_a` 4096→8192, `attn_output_b` 8192→4096) |
| Attention kinds | layers 0–1 sliding window 128; then CSA (compress 4:1, indexer 64 heads x 128, top-k 512) interleaved with HCA (compress 128:1, dense) |
| KV entry | 512 FP8 + 64 BF16 RoPE dims (official mixed storage); compressed 4:1 or 128:1 |
| MoE | 256 routed + 1 shared expert per layer, 6 active, expert FF 2048, hash routing in the first 3 MoE layers, `Sqrt(Softplus)` affinity, weight scale 1.5 |
| Active parameters | 13B (paper); derived here 13.3B |
| Official precision | expert weights **MXFP4** from quantization-aware training (E2M1 codes, E8M0 scale per 32); indexer QK path FP4; everything else FP8 |
| MTP | depth 1; the DSpark support artifact carries 3 MTP-style layers (5.58 GiB, same expert quant mix) |

Official checkpoint layout (`model.safetensors.index.json` and shard headers
of `deepseek-ai/DeepSeek-V4-Flash`, read 2026-09-13; 159.6 GB total):

| Tensor | Storage | Scale |
| --- | --- | --- |
| `layers.N.ffn.experts.E.w1/w2/w3.weight` | `I8` holding two E2M1 nibbles per byte, e.g. w1 `[2048, 2048]` for a 2048x4096 matrix | `F8_E8M0 [2048, 128]`: one exponent per 32 K elements (MXFP4) |
| `layers.N.attn.wq_a/wq_b/wkv/wo_a/wo_b.weight`, shared experts, `head` | `F8_E4M3` | `F8_E8M0` per 128x128 block (wq_b `[256, 8]` for 32768x1024) |
| `layers.N.attn.indexer.wq_b.weight` | `F8_E4M3` | per-block E8M0 |
| compressors, norms, `hc_*`, router | BF16/FP32 | none |

So the official experts are 4.25 bpw and the official dense set is 8.0625 bpw
(FP8 plus a 1/16384 scale overhead), and both are block-scaled with
power-of-two exponents. The local artifact is a re-quantization of these.

Local artifact `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
(measured with `gguf-py`): 80.76 GiB, 284.33 B parameters, 2.44 bpw.

| Tensor family | Type | Size | Params |
| --- | --- | ---: | ---: |
| `ffn_gate_exps`, `ffn_up_exps` | IQ2_XXS (2.06 bpw) | 44.34 GiB | 184.7 B |
| `ffn_down_exps` | Q2_K (2.62 bpw) | 28.22 GiB | 92.3 B |
| `attn_q_b`, `attn_output_a`, `attn_output_b`, `attn_q_a`, `attn_kv`, shared experts, `output` | Q8_0 | 6.15 GiB | 6.21 B |
| `token_embd`, indexer `attn_q_b`, compressors, `ffn_gate_inp`, `hc_*` | F16 | 2.04 GiB | 1.10 B |

Per-token weight traffic (derived from the tensor table; routed experts
weighted 6/256, embedding excluded):

| Family | GB per token | Active params | bpw |
| --- | ---: | ---: | ---: |
| Attention (`q_a`, `q_b`, `kv`, `output_a/b`, indexer `q_b`) | 5.24 | 4.78 B | 8.8 |
| Routed experts (6 of 256) | 1.83 | 6.49 B | 2.25 |
| Shared expert | 1.15 | 1.08 B | 8.5 |
| Indexer + compressors | 0.62 | 0.31 B | 16 |
| Output head | 0.56 | 0.53 B | 8.5 |
| Router + mHC | 0.16 | 0.08 B | 16 |
| **Total** | **9.57** | **13.3 B** | |

At the measured 17.7 tok/s (C1, depth 0, AR) this is 169 GB/s, 70% of the read
ceiling; the ceiling at 241 GB/s is 25 tok/s. The routed experts are 19% of the
bytes. The quantization recipe that made the model fit (experts at 2.1–2.6 bpw)
left the dense 81% at 8–16 bits.

The production decode profile agrees with the byte accounting
([official-kernel-review.md](../benchmarks/deepseek-v4-flash/official-kernel-review.md),
4K depth, 16 generated tokens, 1,017.81 ms of GPU time, 63.6 ms per token):
ordinary Q8 projections 167.4 ms, Q8 projections with mHC expansion 158.8 ms,
grouped Q8 output projection 107.9 ms, MoE gate/up 124.2 ms. The Q8
projection families are 51% of decode GPU time and, at 6.9 GB per token over
~32 ms, run at about 215 GB/s, i.e. already at the read ceiling; the expert
GEMV runs at roughly 155 GB/s. Halving the bytes of a kernel that is at the
memory roofline halves its time; that is what a 4- or 6-bit dense set does
here, and it is why the "same efficiency" column of the scenario table below
is a fair projection for the projection kernels.

Scenarios (derived; "at 170 GB/s" keeps today's kernel efficiency, "at 241"
is the roofline):

| Recipe | GB/token | tok/s @170 | tok/s @241 |
| --- | ---: | ---: | ---: |
| Current (dense Q8/F16, routed 2.25) | 9.40 | 18.1 | 25.6 |
| Attention + shared expert at SHQ6 | 7.98 | 21.3 | 30.2 |
| Attention + shared expert at SHQ4 | 6.34 | 26.8 | 38.0 |
| … plus indexer/compressors/head at SHQ8 | 6.02 | 28.2 | 40.0 |
| All dense at SHQ4, routed 2.25 | 5.61 | 30.3 | 43.0 |
| All dense at SHQ4, routed 3.0 | 6.21 | 27.4 | 38.8 |
| All dense at SHQ4, routed 4.25 (exact MXFP4) | 7.23 | 23.5 | 33.3 |

Resident size by routed-expert width (dense set at SHQ4 ≈ 4.7 GiB including
the F16 embedding):

| Routed bpw | Experts | Model |
| ---: | ---: | ---: |
| 2.06 (IQ2_XXS) | 66.4 GiB | 71.0 GiB |
| 2.25 (today's mix) | 72.6 GiB | 77.1 GiB |
| 2.40 | 77.4 GiB | 81.9 GiB |
| 2.62 (Q2_K) | 84.7 GiB | 89.2 GiB |
| 3.00 | 96.8 GiB | 101.3 GiB |
| 3.25 | 104.8 GiB | 109.4 GiB |
| 2.42 (`w1`/`w3` at 2.09, `w2` at 3.09; section 6.2) | 78.1 GiB | 82.6 GiB |
| 4.25 (MXFP4, lossless vs official) | 137.1 GiB | 141.6 GiB |

With ~10 GiB of working set at C1 and more at C8 and 262K context, the routed
experts are capped near 3.0 bpw on this machine and comfortable at 2.4–2.6.
The consequence for SHQ is direct: the expert tier must be a 2–3 bit format,
and the official weights are already 4-bit codes with a power-of-two block
exponent, which is the natural starting point for that tier (section 6.2).

Where the bytes go is also where the quality risk sits. DeepSeek's own
deployment keeps every non-expert tensor in FP8 and reached its published
quality with experts in MXFP4 *by training for it*. Re-quantizing experts to
2.25 bpw is the largest quality loss already accepted by the current artifact;
taking attention from 8 to 4 bits is a second, untrained loss on tensors the
authors kept at 8 bits. The 736-choice DSpark replay and the antirez
comparison in the DS4 quality contract are the instruments to measure it, and
must run before any kernel work on a 4-bit dense set.

### 4.2 Qwen3.8-27B

Architecture ([model card](https://huggingface.co/Qwen/Qwen3.8-27B), GGUF
metadata):

| Item | Value |
| --- | --- |
| Layers / hidden | 64 (+1 MTP block in GGUF) / 5120, vocabulary 248,320 |
| Layout | 16 x (3 x Gated DeltaNet → FFN, 1 x Gated Attention → FFN): 48 linear-attention layers, 16 full-attention layers |
| Gated DeltaNet | 48 V heads, 16 QK heads, head dim 128; `ssm_qkv` 5120→10240, `ssm_gate` 5120→6144, `ssm_out` 6144→5120, conv1d 4 x 10240, `alpha`/`beta` 5120→48 |
| Gated attention | 24 Q heads, 4 KV heads, head dim 256, RoPE 64; `attn_q` 5120→12288, `attn_k`/`v` 5120→1024, `attn_gate` 5120→6144, `attn_output` 6144→5120 |
| FFN | dense SwiGLU, intermediate 17,408 (`ffn_gate`/`up` 5120→17408, `ffn_down` 17408→5120) |
| MTP | `nextn.eh_proj` 10240→5120 (#130) |
| Dense parameters | 27.32 B; every weight is read every token |

Local artifact `Qwen3.8-27B-UD-Q4_K_XL.gguf` (measured): 16.34 GiB, 5.14 bpw,
seven formats mixed per tensor: Q5_K 42% of elements (ffn_down/up, ssm_out,
attn_gate), IQ4_XS 22% (ffn_gate/up), Q4_K 20% (embed, attn_qkv), Q6_K 13%
(output, attn_output), the rest scattered. Measured on this host
(benchmarks/qwen3.8-27b): `tg128` 11.59 tok/s at depth 0 and 10.90 at 16K;
`pp2048` 474 tok/s against 557 for the Q8_K_XL artifact; DFlash-2 speculative
decode up to 33.5 tok/s at 77% acceptance on chat-framed 512-token
generations. Validation against the Q8 artifact: identical top-1 over 1024
tokens, 4.7x the logit RMSE.

Byte budgets (derived, 27.32 B parameters, decode at 209 GB/s):

| Artifact | bpw | Size | tok/s ceiling |
| --- | ---: | ---: | ---: |
| UD-Q8_K_XL | 8.5 | 27.0 GiB | 7.2 (measured 7.15) |
| UD-Q4_K_XL | 5.14 | 16.3 GiB | 11.9 (measured 11.6) |
| SHQ recipe 5.2 bpw (mirror of unsloth's tiers) | 5.2 | 16.5 GiB | 11.8 |
| SHQ recipe ~4.5 bpw (ffn_down, embed, linear-attn at SHQ6, rest SHQ4) | 4.5 | 14.3 GiB | 13.6 |
| Pure SHQ4 U4Z G64 | 4.31 | 13.7 GiB | 14.2 |

The measured artifacts already run at 97–98% of their byte ceiling, so for
this model SHQ can only buy what its bpw buys: a 4.5 bpw recipe is +17% decode
over UD-Q4_K_XL if quality holds, which the Qwen3.5-0.8B evidence says needs
the SHQ6 upcasts. The quality bar is concrete: measured here with
`llama-perplexity --kl-divergence` on WikiText-2 (24 x 2048 tokens),
UD-Q4_K_XL against UD-Q8_K_L is mean KL 0.017, 99% KL 0.11, top-1 agreement
94.7%, perplexity 6.12 vs 6.15. An SHQ recipe at 4.5 bpw has to land at or
under that with 12% fewer bytes, which is a quantizer problem (section 4.3)
before it is a kernel problem.

To separate format from quantizer, the same Q8_K_L source was requantized
with `llama-quantize --pure` (every linear tensor one type, no imatrix,
`output` Q6_K, embedding Q4_K) and scored the same way:

| Qwen3.8-27B artifact | bpw class | Size | Mean KL vs Q8 | Top-1 | PPL ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| UD-Q4_K_XL (unsloth: imatrix, per-tensor mix of Q4_K/Q5_K/Q6_K/IQ4_XS) | 5.14 | 16.34 GiB | **0.017** | 94.7% | 0.996 |
| Q4_1 pure (affine per-32, FP16 scale and min: the SHQ4 U4Z class) | 5.0 | 16.08 GiB | 0.044 | 91.7% | 1.013 |
| Q4_K pure (K-quant super-blocks, no imatrix) | 4.5 | 14.64 GiB | 0.046 | 91.7% | 1.000 |
| Q4_0 pure (symmetric per-32: the SHQ4 S4 class) | 4.5 | 14.64 GiB | 0.051 | 91.1% | 1.019 |

Every uniform 4-bit recipe without calibration lands at KL 0.044–0.051 and
91–92% top-1, whether the block structure is affine, symmetric or K-quant:
the format is worth at most 0.5 bpw between them (Q4_K matches Q4_1 at 10%
fewer bytes). The 2.7x lower KL of the shipped artifact comes from the
imatrix and the per-tensor mixture, not from the block format. For SHQ this
is the whole story: SHQ4-G64 U4Z is the Q4_1 class at 4.31 bpw, and the
quantizer (imatrix plus error feedback) and the recipe (SHQ6 on the
sensitive tensors) are what would carry it from 0.05 to 0.02; the tile
layout neither helps nor hurts that. The larger Qwen lever is prefill
(section 6.1).

### 4.3 Reconstruction study on the production artifacts

To move the Qwen3.5-0.8B quantizer comparison to production scale, each
tensor below was dequantized from its highest-precision local artifact
(Qwen: UD-Q8_K_L, whose attention and FFN tensors are Q8_0 or Q6_K; DS4: the
Q8_0 dense tensors of the shipped artifact), re-quantized with
`tools/gufo/shq.py` (range scale search, no imatrix), and scored as relative
mean absolute error against that reference. For Qwen the same tensor from the
UD-Q4_K_XL artifact (unsloth's imatrix-tuned K-quant) is scored against the
same reference. Reference precision limits the absolute numbers, not the
ranking. Script: `scratchpad/recon.py`, 2026-09-13.

| Tensor | Shape | max/rms | SHQ4 U4Z G64 | SHQ4 S4 G64 | SHQ4 U4Z G32 | SHQ6 G64 | unsloth tensor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen `blk.3.attn_q` | 12288x5120 | 19 | 9.92% | 11.70% | 8.70% | 2.37% | Q5_K 4.26% |
| qwen `blk.3.attn_k` | 1024x5120 | 13 | 10.14% | 12.04% | 8.84% | 2.43% | Q4_K 8.14% |
| qwen `blk.3.attn_output` | 5120x6144 | 90 | 9.99% | 11.74% | 8.78% | 2.66% | Q6_K 2.08% |
| qwen `blk.10.attn_qkv` (DeltaNet in-proj) | 10240x5120 | 23 | 9.97% | 11.73% | 8.76% | 2.65% | Q4_K 7.99% |
| qwen `blk.10.ssm_out` | 5120x6144 | 65 | 9.87% | 11.60% | 8.69% | 2.36% | Q5_K 4.33% |
| qwen `blk.20.ffn_gate` | 17408x5120 | 23 | 9.84% | 11.53% | 8.68% | 2.61% | Q4_K 7.83% |
| qwen `blk.20.ffn_down` | 5120x17408 | 31 | 9.87% | 11.59% | 8.68% | 2.35% | Q5_K 4.32% |
| qwen `blk.40.ffn_up` | 17408x5120 | 16 | 9.96% | 11.70% | 8.76% | 2.65% | Q5_K 3.97% |
| ds4 `blk.5.attn_q_b` | 32768x1024 | 14 | 9.88% | 11.60% | 8.75% | 2.68% | (Q8_0 reference) |
| ds4 `blk.5.attn_output_a` | 8192x4096 | 9 | 10.02% | 11.77% | 8.80% | 2.71% | |
| ds4 `blk.5.attn_output_b` | 4096x8192 | 7 | 9.86% | 11.60% | 8.73% | 2.72% | |
| ds4 `blk.5.attn_q_a` | 1024x4096 | 6 | 9.86% | 11.60% | 8.73% | 2.74% | |
| ds4 `blk.5.attn_kv` | 512x4096 | 9 | 9.87% | 11.60% | 8.71% | 2.73% | |
| ds4 `blk.5.ffn_gate_shexp` | 2048x4096 | 30 | 9.93% | 11.64% | 8.76% | 2.64% | |
| ds4 `blk.20.attn_q_b` | 32768x1024 | 9 | 9.87% | 11.58% | 8.72% | 2.69% | |
| ds4 `blk.20.ffn_down_shexp` | 4096x2048 | 82 | 10.95% | 13.31% | 9.36% | 3.06% | |
| ds4 `output` | 129280x4096 | 23 | 9.77% | 11.45% | 8.63% | 2.59% | |

Readings:

- SHQ4 U4Z G64 with range search sits at 9.8–10.0% on every tensor of both
  models; the number is a property of uniform 4-bit rounding on near-Gaussian
  weights, not of the model. unsloth's Q4_K, at 4.5 bpw with imatrix-tuned
  scales and mins, is 7.8–8.1% on the same tensors. SHQ4 G32 (4.625 bpw)
  reaches 8.7% by group size alone. The gap is the quantizer (section 8, E6),
  and it is the same gap on DS4's MLA projections as on Qwen's FFN.
- S4 costs 1.7 points over U4Z everywhere; the zero point is worth keeping on
  the decode path and dropping only where the prefill epilogue matters more.
- SHQ6 (6.56 bpw) is 2.4–2.7%, between Q5_K (4.3%) and Q6_K (2.1%): a real
  tier for the tensors that cannot take 4 bits.
- DS4's MLA projections have low outlier ratios (max/rms 6–14) and quantize
  like any other dense tensor; the shared-expert `ffn_down` (max/rms 82) is
  the only DS4 dense tensor that is visibly harder, which matches the
  Qwen3.5-0.8B finding that `ffn_down` wants the upcast tier.
- Against the official FP8 values of the 0731 checkpoint (`layers.5.attn.wq_b`,
  `wo_b`, `shared_experts.w1`, fetched by HTTP range), the shipped Q8_0
  tensors are 0.55% RMS from the source, SHQ4 U4Z G64 is 9.2% RMS / 9.9%
  MAE, G32 8.2% / 8.7%, and SHQ6 2.5% / 2.8%: the same figures as against the
  Q8_0 reference, so the Q8_0-referenced rows above are trustworthy for DS4.

## 5. What the field does, and what transfers

Formats, with the numbers that matter on this hardware:

| Family | bpw | Structure | Decode cost on a VALU-bound GPU | Relevance |
| --- | ---: | --- | --- | --- |
| GGUF Q4_K / Q5_K / Q6_K ([ggml-common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h)) | 4.5 / 5.5 / 6.56 | super-block 256, 8 sub-blocks of 32 with 6-bit scale and min packed into 12 bytes, FP16 `d`/`dmin` | scale/min unpack per 32 plus a min correction per (row, block); measured 79% of the Q4 prefill deficit here | the incumbent for Qwen; what SHQ must beat |
| GGUF Q2_K | 2.625 | super-block 256, 16 sub-blocks of 16 with 4-bit scale and min | cheap integer unpack | DS4 `ffn_down_exps` today |
| GGUF IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S | 2.06 / 2.31 / 2.5 / 3.06 / 3.44 | E8-lattice codebooks (256–1024 entries of 8 weights), sign bits and shifts, per-256 FP16 scale | codebook gather per 8 weights; the slow family in llama.cpp | DS4 `ffn_gate/up_exps` today |
| GGUF IQ4_XS / IQ4_NL | 4.25 / 4.5 | non-linear 16-entry LUT, 6-bit sub-scales per 32 | one `V_PERM_B32` LUT per 4 nibbles (already in the Qwen kernels) | best 4-bit quality per bpw in GGUF |
| ik_llama IQ2_KS / IQ2_K / IQ3_K / IQ4_KS ([discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)) | 2.19 / 2.375 / 3.44 / 4.25 | no codebook: 2x4 or 2x8 LUT selected by one bit per block, blocks of 16–32, per-row scale | integer only, 3–4x faster than trellis on CPU | the design model for a 2-bit tile tier |
| ik_llama IQ*_KT, QTIP, EXL3 ([QTIP](https://arxiv.org/abs/2406.11235), [exllamav3](https://github.com/turboderp-org/exllamav3)) | 2.0–4.0 | trellis-coded, incoherence (Hadamard) preprocessing, no codebook storage | decode regenerates weights with an LCG per element; compute-bound; ik reports lower quality than expected on real weights | best published quality at 2–3 bpw; poor fit for a VALU-bound decode |
| MXFP4 / NVFP4 ([OCP MX](https://arxiv.org/abs/2310.10537)) | 4.25 / 4.5 | E2M1 codes, E8M0 scale per 32 (MX) or UE4M3 per 16 with a tensor scale (NV) | `ldexpf` or LUT decode; fused MXFP4 WMMA GEMM on RDNA4 reaches 53% of FP16 peak ([guide](https://github.com/JohnTDI-cpu/rdna4-wmma-guide)) | DS4's official expert format; gpt-oss's format; 51–55 tok/s for gpt-oss-120b on Strix Halo in llama.cpp ([strix-halo-llm-perf](https://github.com/visorcraft/strix-halo-llm-perf)) |
| AWQ / GPTQ g64–g128 affine, MLX affine, Marlin | 4.25–4.5 | UINT4, per-group FP16 scale and zero (MLX stores scale and bias), pre-permuted for tensor-core fragments | one scale FMA and one zero term per group | **SHQ4-T16-U4Z is this class**: same arithmetic, a 16x16 tile order chosen for WMMA/AIE fragments |
| NPU2 Q4_1 g32 with BF16 activations ([Gemma3 on Ryzen AI NPU](https://arxiv.org/html/2602.06063v1)) | 4.5 | int4, per-32 scale and min converted to BF16, blocks of 32x256 | dequantize on the AIE into BF16, BF16 matmul | shows the NPU path other teams chose: BF16 arithmetic with pre-converted offsets, not INT8 x INT4 |

Two practical observations from the community numbers on this machine class:
llama.cpp on Strix Halo moves a 5.1 B-active MXFP4 MoE at 51–55 tok/s, i.e.
~160–170 GB/s effective, the same efficiency band as gufo's DS4 decode; and
ROCm and Vulkan backends trade the lead on decode by ±16% depending on
release. Nothing in the field reaches the 241 GB/s ceiling on decode; ~210 is
the practical ceiling gufo already reaches for Qwen.

What transfers to SHQ and what does not:

- Transfers: affine 4-bit with per-group scale and zero is the standard and the
  arithmetic is known good (AWQ/GPTQ/MLX). A fixed 16x16 tile that is one WMMA
  operand is what Marlin does for tensor cores and what the RDNA4 MXFP4 kernel
  does with `TILE_K = 32`. The NPU-side W4A8 with an activation-sum zero
  correction is the same trick AWQ kernels use.
- Does not transfer: codebooks and trellises. They win on paper at 2–3 bpw but
  turn a bandwidth-bound decode into a gather- or ALU-bound one on a GPU with
  no spare issue slots; the ik_llama data point (integer LUT quants at 3–4x the
  trellis speed with comparable quality) is the relevant one.
- Transfers with a caveat: MXFP4's E8M0 block exponent costs one `ldexp` per
  32 weights and is lossless against the official DS4 experts, but at 4.25 bpw
  it does not fit. Its structure (8 magnitudes, sign, per-32 exponent) is the
  right thing to *derive* a 2–3 bit expert tier from.

## 6. How SHQ could work on each model

### 6.1 Qwen3.8-27B

**Tensor map.** Bulk `ffn_gate`/`ffn_up`, `ssm_qkv`, `ssm_gate`, `attn_q`,
`attn_gate`, `attn_k`/`v` at SHQ4-G64; `ffn_down`, `ssm_out`, `attn_output`,
`output` and `token_embd` at SHQ6 (the unsloth choice and the Qwen3.5-0.8B
result agree on `ffn_down`); norms, conv1d, `alpha`/`beta` at BF16/F32. That is
the 4.5 bpw recipe above. The MTP `eh_proj` (#130) is the first tensor to
port because it is one shape, batch one, and has a CPU oracle.

**Decode GEMV.** One wave32 owns one 16-lane output tile and streams its
`[k_group]` sequence: per K64 group 512 B of codes, 32 B of BF16 scales, 8 B of
zeros, all contiguous and 128-byte aligned, which is one coalesced 552-byte
burst per group per tile. Each lane holds its output channel; the activation
vector (BF16, 5120 or 17408 elements) is read once per workgroup through LDS
and reused by every tile the workgroup owns. Arithmetic per weight: nibble
extract, `(q - z)` as an integer, one FMA against the BF16 activation with the
group scale applied once per 64 products. Compared with the Q4_K kernel there
is no 6-bit scale/min unpack and no per-32 `dmin` term. The bytes are 4.31 per
weight instead of 4.5; at 209 GB/s the whole model reads in 70 ms.

**Prefill and speculative verification.** W4A8: the activation block is
quantized once per stage to INT8 with one BF16 scale per (row, K64), the SHQ
K16xN16 microtile is unpacked to 256 INT8 bytes in VGPRs and fed to
`wmma_i32_16x16x16_iu8`. Because one microtile *is* one B fragment, there is
no permutation, no LDS transpose and no scale interpolation. Per K64 the
epilogue is one `v_cvt_f32_i32`, one activation-scale multiply and one
weight-scale multiply per accumulator element, plus the zero correction
`z[n,g] * sum(a[r,g])` where `sum(a)` is produced once with the activation
quantization. That is the same per-element cost the Q8_0 kernel already pays
per 32 elements, paid per 64, so the epilogue share that holds W8A8 at 58% of
the WMMA ceiling halves. With S4 tensors the zero term disappears. The
measured Q4 prefill deficit (474 vs 557 tok/s) is 79% the K-quant minimum
correction; the expected SHQ4 prefill is therefore at or above the Q8 rate,
not below it. The DFlash-2 verifier is a separate problem: it must reproduce
the decode GEMV bit for bit in FP32 and is VALU-bound at 32% of peak on the
marginal row; SHQ4 changes its bytes, not its issue count.

**Exactness contract.** Decode uses BF16 activation x dequantized weight in
FP32; prefill uses W4A8 with INT32 group sums. These differ in rounding, so a
drafted token verified by the W4A8 path cannot be accepted against a decode
step that used the BF16 path. The existing Qwen rule (verifier reproduces the
decode arithmetic exactly) still applies; the choice is which arithmetic both
use. The NPU contract already fixes W4A8 with FP32 group partials in (block,
group) order; adopting the same on the GPU for the exact small-batch kernel
keeps GPU, NPU and CPU oracle bit-identical, at the cost of quantizing the
single decode row to INT8 as well (an extra rounding on the activation, not on
the weights).

**NPU.** The `qwen_aie2p_w4a8` program consumes the SHQ4 tile today. The
measurements say a main-model offload does not pay: 47 GB/s streaming, 63 µs
dispatch, and either a decode that is bandwidth-saturated by the GPU or a
prefill where the NPU's usable arithmetic is ~3%. The place where the shared
layout is worth having is a companion model (the measured +23% when a
Qwen3.5-4B or ASR workload leaves the GPU); those models are the ones to
convert to SHQ first for the NPU.

### 6.2 DeepSeek V4 Flash

**Dense set (the 81%).** Candidates, in order of bytes saved per unit of
quality risk:

| Tensor | Params | Today | Proposed | Bytes/token saved |
| --- | ---: | --- | --- | ---: |
| `attn_q_b` (1024→32768) | 1.44 B | Q8_0 | SHQ4-G64 or SHQ6 | 0.75 / 0.35 GB |
| `attn_output_a`, `attn_output_b` | 2.89 B | Q8_0 | SHQ4-G64 or SHQ6 | 1.51 / 0.70 GB |
| shared expert gate/up/down | 1.08 B | Q8_0 | SHQ4-G64 | 0.57 GB |
| `output` head | 0.53 B | Q8_0 | SHQ6 or SHQ8 | 0.13 / 0.02 GB |
| indexer `attn_q_b`, compressors | 0.49 B | F16 | SHQ8 (indexer QK is FP4 in the official model; 8 bits is safe) | 0.47 GB |
| `attn_q_a`, `attn_kv` (the MLA latent projections) | 0.27 B | Q8_0 | keep Q8/SHQ8 | 0 |
| `ffn_gate_inp`, `hc_*` | 0.08 B | F16 | keep | 0 |

The risky ones are `attn_q_b` and the grouped output projections: they are
the absorbed-MLA path where a 4-bit error is applied 64 heads wide before a
softmax over a 512-dimensional latent. Section 6.4 measures it: attention at
Q4_K alone costs mean KL 0.076 against the shipped artifact, the shared
experts and head add 0.016; Q5_K on the same set costs 0.049 at baseline
perplexity and Q6_K 0.035. The quality gate remains the existing
qualification (pinned target trajectory ≥116/128 top-1, rank sum ≤142; 736
exact DSpark replay choices; antirez continuation comparison). The recipe
this supports is a 5- or 6-bit tier on the dense set (7.4–8.1 GB/token,
+18–29% decode), not SHQ4 (6.5 GB/token, +48%, KL 0.09).

The cheapest way to answer the quality question does not need SHQ at all:
requantize the same GGUF with llama.cpp's `Q4_K`/`IQ4_XS`/`Q6_K` on exactly
those tensors (with the same imatrix) and run the DS4 quality contract. If
4-bit attention fails there, SHQ4 attention would fail too, since the
quantizers are of the same class; if it passes, the SHQ kernels are what turn
the byte saving into speed.

**Routed experts (the 19%, and all of the memory).** The official weights
are MXFP4: each value is `sign x m x 2^e_block` with `m in {0, 0.5, 1, 1.5, 2,
3, 4, 6}` and one E8M0 exponent per 32 consecutive K. To see what a 2–3 bit
tile tier could do, five expert matrices of the pinned 0731 checkpoint
(`layers.5.ffn.experts.{0,7}.w1`, `layers.20.ffn.experts.3.w1`,
`layers.5.ffn.experts.0.w2`, `layers.20.ffn.experts.3.w2`) were fetched by
HTTP range from `deepseek-ai/DeepSeek-V4-Flash-0731` (revision `7872f01`, the
one the DS4 quality contract pins), decoded, and compared with (a) the same
experts in the shipped GGUF and (b) simple re-quantizers that keep the 32-wide
block structure. Script: `scratchpad/fp4_study_0731.py`. All five tensors
gave the same numbers to within 0.3 points:

| Observation | Value |
| --- | --- |
| Code entropy | 3.86 bits of 4: the E2M1 alphabet is used almost uniformly; there is no entropy-coding gain to take |
| Magnitude use | 0: 13%, 0.5: 13%, 1: 22%, 1.5: 9%, 2: 18%, 3: 12%, 4: 9%, 6: 4% |
| Block exponent range | 4 distinct values (2^-8 … 2^-5) in every tensor: an E8M0 byte per 32 carries 2 bits of information, so a per-tensor base plus a 2-bit field per block is enough (0.0625 bpw instead of 0.25) |

Relative RMS error of the re-quantized experts against the official MXFP4
values (lower is better; the shipped GGUF rows are the current artifact):

| Re-quantization | bpw | `w1`/`w3` (gate/up) | `w2` (down) |
| --- | ---: | ---: | ---: |
| Shipped GGUF: IQ2_XXS (gate/up), Q2_K (down), imatrix | 2.06 / 2.62 | **35.4%** | **27.6–28.4%** |
| 2-bit uniform symmetric, per-32 scale | ~2.1 | 34.4% | 34.4% |
| 2-bit, two fixed 4-entry tables + 1 select bit per 32 (IQ2_KS construction) | ~2.1 | 32.6% | 32.6% |
| 2-bit, per-block optimal 4-level LUT (lower bound for any per-32 4-level scheme) | n/a | 30.2% | 30.2% |
| 2-bit uniform asymmetric, per-16 scale and min (Q2_K construction, no imatrix) | ~2.6 | 33.7% | 33.7% |
| 3-bit uniform symmetric, per-32 scale | ~3.1 | 17.6% | 17.6% |
| 3-bit, per-block optimal 8-level LUT | n/a | 15.8% | 15.8% |

Readings:

- Two bits is two bits. Every 2-bit scheme lands at 30–35% RMS error; the
  E8-lattice codebook of IQ2_XXS (35.4%) does not beat a two-table LUT
  (32.6%) on these QAT'd weights and costs a codebook gather per 8 weights
  to decode. The shipped Q2_K on `w2` (27.6–28.4%) is the best 2-bit result
  here because of its per-16 scale-and-min with imatrix search, and it is
  also the widest at 2.62 bpw.
- The step that halves the error is the third bit: 17.6% uniform, 15.8%
  with an 8-entry LUT. On this machine that bit is affordable on `w2` only
  (see the memory table): `w2` at ~3.1 bpw with `w1`/`w3` at ~2.1 bpw
  averages 2.42 bpw, 78 GiB of experts, the same footprint as today.
- Unweighted error is the wrong final metric; the shipped quantizers are
  imatrix-weighted and the comparison that matters is the DS4 quality
  contract. These numbers bound what a tile tier can do, they do not
  qualify it.

A tile tier in that spirit, "SHQ3-MX" for `w2` and "SHQ2-MX" for `w1`/`w3`:

```text
codes:    2 or 3 bits per weight; one K16 x N16 microtile = 64 or 96 bytes
levels:   a 4- or 8-entry signed LUT per (lane, K32), chosen from 2 per-tensor
          tables by 1 select bit (values from the E2M1 set, imatrix-fitted)
exponent: 2-bit field per (lane, K32) over a per-tensor base  -> 0.0625 bpw
select:   1 bit per (lane, K32)                                -> 0.03125 bpw
total:    2.09 bpw (SHQ2-MX) or 3.09 bpw (SHQ3-MX)
```

The decode GEMV reads 64- or 96-byte microtiles, expands codes through a LUT
held in two VGPRs, and applies the block exponent with one `ldexp` per 32
products; there is no codebook gather (the IQ2_XXS cost) and no super-block
scale unpack (the Q2_K cost). Whether the imatrix-fitted version of this
reaches the shipped artifact's quality is experiment E4; the prior from the
table is "equal or better error at equal bytes, and a materially cheaper
kernel".

**MoE execution shape.** Per token per layer, six experts x three matrices of
4096x2048 (or 2048x4096) are read once each, plus the shared expert. In the
tile layout each expert is `[n_tile][k_group]` contiguous, so a workgroup
assigned one (expert, n_tile range) streams a contiguous run and never
gathers across experts; the router output only selects which six runs to
issue. For batched prefill the natural kernel is a grouped GEMM: rows are
sorted by expert, each expert's row slab multiplies its own tile sequence
with `iu8` WMMA after code expansion, and the block exponent is applied in the
FP32 epilogue per K32 — one `ldexp`, not a scale multiply, so the epilogue is
cheaper than the Q8_0 one that caps Qwen prefill. The DSpark support artifact
uses the same expert format and gets the same kernels.

**What SHQ does not change for DS4.** The KV cache is already the official
FP8 + BF16-RoPE mix and is compressed 4:1 (CSA) or 128:1 (HCA); it is not a
weight format problem. The hc residual streams, router and norms are 0.16
GB/token and stay F16/F32. The DSpark verifier path reads the same dense
tensors as decode, so its cost falls with them.

### 6.3 Measured: SHQ tile decode GEMV on gfx1151

A throwaway HIP microbenchmark (scratchpad only, 2026-09-13) packs random
Gaussian matrices into the contract layouts, replicates each into a ≥ 1 GiB
arena so timed passes stream cold memory (the 32 MB MALL otherwise inflates a
single 48 MB matrix to 400+ GB/s), checks every result against a CPU oracle
(max relative error ≤ 5e-4, FP32 summation order), and reports effective
bytes per second over the whole plane set. Kernel shape: wave32 per N16 tile,
lanes 0–15 on even K64 groups and 16–31 on odd groups, `U` groups loaded
before any is consumed, BF16 activation staged in LDS, one shuffle to combine
the halves. Best variant per shape:

| Shape | SHQ4 contract order (8 B loads) | SHQ4 lane-contiguous group (2 x 16 B loads) | SHQ8 (4 x 16 B loads) | Pure streaming read |
| --- | ---: | ---: | ---: | ---: |
| Qwen `ffn_gate` 17408x5120 | 194 GB/s (80%) | 202 GB/s (84%) | 222 GB/s (92%) | 242 GB/s |
| Qwen `ffn_down` 5120x17408 | 211 (88%) | 221 (92%) | 228 (95%) | 243 |
| Qwen `attn_qkv` 10240x5120 | 177 (74%) | 194 (81%) | 218 (90%) | 241 |
| DS4 `attn_q_b` 32768x1024 | 152 (63%) | 157 (65%) | 190 (79%) | 235 |
| DS4 `attn_output_a` 8192x4096 | 172 (71%) | 184 (76%) | 214 (89%) | 229 |
| DS4 `attn_output_b` 4096x8192 | 182 (76%) | 189 (78%) | 209 (87%) | 234 |
| DS4 shared `ffn_gate` 2048x4096 | 145 (60%) | 154 (64%) | 203 (84%) | 179 |

Readings:

- The tile layout streams at the roofline. SHQ8 with 16-byte lane loads
  reaches 84–95% of the 241 GB/s ceiling on every shape, the same band as
  the production Qwen Q8 decode (209 GB/s). Nothing in the layout costs
  bandwidth.
- The 4-bit kernel trails by 5–15 points on the large shapes because the
  contract's microtile order (`[k16_subtile][output_lane][8 B]`) gives a
  lane 8 contiguous bytes per subtile, i.e. four 8-byte loads per group.
  Reordering a K64 group as `[output_lane][k16_subtile][8 B]` makes a lane's
  64 codes 32 contiguous bytes (two 16-byte loads) and is worth +3 to +9%
  on the same kernel. For the WMMA prefill path a lane still fetches its
  16-k fragment column as one 8-byte load in either order; the reordered
  form spreads one fragment's 16 loads over four cache lines instead of one,
  which the other three subtiles of the same group then reuse. This is a
  contract-level decision to take before the family is frozen, and the
  measurement says the decode side prefers the lane-contiguous group.
- Small-K and small-N shapes (DS4 `attn_q_b` with 16 groups per tile, the
  2048-row shared expert) are latency-bound at 60–65%: one wave per tile has
  too little work in flight. The remedy is the usual one (several waves per
  tile splitting K with an LDS or atomic reduction, or several small
  projections in one dispatch, as the production kernels do at depth), not a
  format change. The DS4 dense set is 44% these shapes by bytes, so the
  4-bit projection kernel for DS4 needs that split from the start.
- Arithmetic is not the limit at these rates: the 4-bit kernel executes
  about twice the VALU work per byte of the 8-bit one and still sits within
  10% of it on the FFN shapes.

Bottom line for the byte-budget projections in sections 4.1 and 4.2: a
first-cut SHQ4 GEMV runs at 80–92% of the read ceiling on the shapes that
carry most of Qwen's bytes, which is the "same efficiency as today"
assumption those tables use.

**Prefill issue-rate model.** A second scratchpad kernel
(`scratchpad/w4_unpack_wmma.hip`) runs register-resident `iu8` WMMA chains
in the shape of the Q8 GEMM inner loop and adds the two costs a 4-bit weight
brings: unpacking the B fragment from packed nibbles, and the FP32 group
epilogue at the group size the format dictates (8 cvt+fma triples per
accumulator, after every 2 WMMA for K32 scales or every 4 for K64):

| Inner loop | TOPS | vs W8A8 |
| --- | ---: | ---: |
| W8A8, no epilogue | 50.7 | — |
| W4A8, nibble order (k, k+8) per byte: 4 VALU ops per fragment | 48.9 | -4% |
| W4A8, contract order (even k low, odd k high): 4 ops + 2 `v_perm_b32` | 47.3 | -7% |
| W8A8 + epilogue per K32 (Q8_0's shape) | 41.4 | -18% |
| W8A8 + epilogue per K64 | 45.6 | -10% |
| W4A8 (k, k+8) + epilogue per K64 (SHQ4-G64's shape) | 45.3 | -11% |
| W4A8 (k, k+8) + epilogue per K32 | 41.1 | -19% |

Nibble unpack costs 4–7% of WMMA issue; the K64 group halves the epilogue
penalty from 18% to 10%; the SHQ4-G64 inner loop therefore issues at 45 TOPS
against the Q8_0 W8A8 loop's 41 TOPS, +10% in the compute-bound regime,
before counting the K-quant minimum correction that the production Q4 path
also pays. This is an issue-rate model, not a GEMM with LDS staging and
global loads, so it bounds the arithmetic side only. Two contract details
fall out of it: a `(k, k+8)` nibble interleave within a microtile row is
cheaper to unpack for WMMA than the even/odd order (and costs the GEMV
nothing), and G64 is the right default group for prefill on this GPU.

### 6.4 Measured: DS4 with a 4-bit dense set (E1, first pass)

The experiment artifact `DeepSeek-V4-Flash-attnQ4K-experiment.gguf` (78.1
GiB, not shipped) re-quantizes the shipped GGUF with the in-tree
`llama-quantize` (scratchpad build, `--allow-requantize`, no imatrix, which
handicaps `Q4_K` slightly relative to the imatrix-tuned experts): `attn_q_b`,
`attn_output_a`, `attn_output_b` and the three shared-expert projections at
`Q4_K` (4.5 bpw), `output` at `Q6_K`, `attn_q_a`/`attn_kv` kept at `Q8_0`,
experts and F16 tensors copied. It is a proxy for the quality class of a
4-bit tile format on the same tensors, not an SHQ artifact. Per-token weight
traffic falls from 9.57 to 6.63 GB (derived, -31%).

The first thing the experiment showed is structural: **the DS4 runtime is
type-locked to the shipped recipe.** `runtime/model_data.cpp` expects an exact
ggml type per tensor (`tensor_expect_layout`: Q8_0 for the dense projections,
IQ2_XXS/Q2_K for the experts, F16 for the rest) and rejects the artifact with
`ds4: tensor output.weight has type q6_k, expected q8_0`. There is no Q4_K,
Q6_K or IQ4_XS path in the imported kernels, so `ds4_quality_test` and the
serving stack cannot score a 4-bit dense set at all today. Any DS4 dense-set
change, SHQ or GGUF, starts with new decode and prefill kernels in the DS4
runtime; the quality question has to be answered outside gufo first.

Scoring therefore uses the in-tree `llama.cpp` (HIP build in the scratchpad,
`llama-perplexity` with `--kl-divergence`), which has a `deepseek4` graph and
kernels for every type involved. The baseline artifact's logits over
WikiText-2 (test split, 2048-token chunks) are the reference; the experiment
artifact is scored against them. Because the two artifacts share every
expert tensor byte for byte, the KL isolates the dense-set change:

| Dense-set recipe (experts unchanged) | bpw of the dense set | GB/token | PPL (WikiText-2, 24x2048) | Mean KL vs baseline | Top-1 agreement | 99% KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shipped: Q8_0 attention, shared experts, head | 8.5 | 9.57 | 5.863 | — | — | — |
| Attention Q6_K, shared Q6_K, head Q6_K (the SHQ6 class) | 6.56 | 8.13 | 5.922 (+1.0%) | 0.035 | 92.6% | 0.36 |
| Attention Q5_K, shared Q5_K, head Q6_K | 5.5 / 6.56 | 7.41 | 5.848 (-0.3%) | 0.049 | 91.3% | 0.48 |
| Attention Q4_K, shared and head kept Q8_0 | 4.5 / 8.5 | 7.41 | 6.174 (+5.3%) | 0.076 | 90.0% | 0.79 |
| Attention Q4_K, shared Q4_K, head Q6_K (the SHQ4 class) | 4.5 / 6.56 | 6.74 | 6.234 (+6.3%) | 0.092 | 88.6% | 0.90 |

Decode ceilings at today's 170 GB/s effective: 17.8 (shipped), 20.9 (Q6 class,
+18%), 22.9 (Q5 class, +29%), 25.2 (Q4 class, +42%) tok/s; SHQ4 U4Z at 4.31
bpw on the same set would be 6.46 GB/token, 26.3 tok/s.

Readings:

- **The MLA projections are the sensitive tensors.** With the shared experts
  and head left at Q8_0, attention alone at Q4_K costs KL 0.076 of the 0.092;
  the shared experts and head add 0.016. `attn_q_b` and the grouped output
  projections carry the loss, as section 6.2 predicted.
- **Four bits on attention is a real quality step, six is not free either.**
  Q4_K: +6.3% perplexity, 11.4% of top-1 choices move. Q6_K: +1.0%
  perplexity, still 7.4% of top-1 choices move and mean KL 0.035. The
  calibration point measured the same way on this host: the *whole*
  Qwen3.8-27B at UD-Q4_K_XL (5.14 bpw, imatrix-tuned) against UD-Q8_K_L
  scores mean KL 0.017, top-1 94.7%, perplexity equal (6.12 vs 6.15). DS4's
  dense set, one fifth of the active parameters, costs twice that KL at
  Q6_K and five times it at Q4_K: the absorbed MLA path (64 heads over a
  512-wide latent, applied through the mHC residual) is far more sensitive
  per bit than a dense transformer's projections, and these numbers are on
  top of the 2-bit expert loss the baseline already carries.
- **Q5_K is the interesting point.** Perplexity is at the baseline (the
  -0.3% is within the ±1.7% chunk noise) and KL is 0.049, for 7.41 GB/token
  and +29% decode ceiling. No imatrix was used in any of these
  requantizations; an imatrix-tuned 5-bit tier, or a 6-bit tier with
  outlier-aware scales, is the recipe to qualify rather than 4 bits. SHQ has
  no 5-bit tier; SHQ6 (6.56 bpw) lands on the Q6_K row, and a "SHQ5"
  (5.5 bpw, 5-bit codes, same tile order) would be the DS4-motivated addition
  to the family.
- **KL, not perplexity, is the gate.** Q5_K's perplexity matches the baseline
  while 8.7% of greedy choices differ; the DSpark acceptance and the 736-choice
  replay contract are top-1 properties, so a dense-set change will show up
  there before it shows up in perplexity.
- Scoring ran in llama.cpp because gufo's DS4 runtime is type-locked
  (above). The same measurement on the gufo runtime needs Q5/Q6 or SHQ5/SHQ6
  kernels first; that is now the first DS4-side deliverable of any SHQ plan.

### 6.5 Where the NPU fits

The in-repo measurements close the main-model question for now: no route
(single-projection offload, drafter relocation, weight-streaming split, prefill
split) was net positive, and the two hard limits are the 47 GB/s AIE stream
and the per-group epilogue that divides arithmetic by ~10. SHQ's shared-layout
value is real in exactly one place measured here: a companion model running
concurrently on the NPU (+23% to the main model versus running the companion
on the GPU). For that use the NPU consumes the same SHQ4/SHQ8 tiles as the GPU
runtime would, and the model is small enough that 47 GB/s is adequate.

One format experiment is specific to the NPU: XDNA2 natively supports
`bfp16ebs8` (blocks of 8 values sharing one 8-bit exponent, 1 sign + 7
mantissa bits each, 9 bits per weight) with a dedicated 8x8 BFP16 MAC at 512
MAC/cycle per tile. A weight tier stored in exactly that layout is consumed
with no scale epilogue at all: the exponent is applied inside the MAC, which is
the term that divides the NPU's measured W8A8 rate by ten (11 vs 0.94 TOPS).
It costs 9 bpw against SHQ8's 8.25, so it is a prefill/companion tier, not a
decode tier, and it is not a GPU format (gfx1151 has no BFP path; the GPU
would read it as INT8 plus a per-8 `ldexp`). A one-kernel microbenchmark
(section 8, E5) decides whether it earns a place in the family for the
companion-model use.

## 7. Advantages and costs, itemized

Advantages of an SHQ artifact over the GGUF types in use:

1. **Fewer bytes for the same class of quantizer.** 4.31 vs 4.5 (Q4_K) and 4.25
   (S4) vs 4.25 (IQ4_XS) bpw; on Qwen a 4.5 bpw recipe is 12% fewer bytes than
   UD-Q4_K_XL. Decode throughput follows bytes on this machine.
2. **A cheaper prefill epilogue.** One scale per 64, zero folded into an
   activation sum or absent, no super-block scale decode, no minimum
   correction. Measured Q4 prefill loses 15% to Q8 today, 79% of which is that
   correction.
3. **Tiles that are WMMA and AIE fragments.** K16xN16 is the `iu8` B operand
   and the `mmul_8_4` microtile; 128-byte alignment matches the cache line.
   No in-kernel permutation for either device.
4. **One numerical contract across CPU, GPU and NPU** (INT32 dots, FP32 group
   partials in fixed order), which is what makes exact GPU/NPU results and
   exact speculative verification possible.
5. **Per-tensor tiers with one kernel family.** SHQ4/6/8 share the tile order;
   the kernel differs only in read width. The DS4 dense set and the Qwen upcast
   set both need this.
6. **Ownership of the recipe.** Per-model tensor selection, imatrix, eval
   suite and manifest are in-repo; no dependence on a third party's quantization
   choices for the attention tensors of a 284 B model.

Costs and risks:

1. **No production GPU kernel exists.** The decode side is de-risked by the
   scratchpad microbenchmark (section 6.3: 80–92% of the read ceiling on the
   large shapes, exact against the oracle); the W4A8 prefill tile and the
   small-shape split are still projections until measured against the
   K-quant kernels that reach 58% of WMMA and 209 GB/s in production.
2. **Quantizer quality gap.** ~10% vs ~7% relative MAE at equal bpw against
   llama.cpp's Q4_K on Qwen3.5-0.8B. Needs error feedback (GPTQ/OBQ
   updates) or at least per-block joint scale/zero search; imatrix scale
   search alone left this gap.
3. **DS4 has no expert tier.** SHQ4/6/8 cannot hold the experts; a 2–3 bit
   tier must be designed, quantized from MXFP4 codes, validated against the
   DS4 contract, and given decode and grouped-prefill kernels. Until then SHQ
   on DS4 means a mixed artifact: SHQ dense tensors plus IQ2/Q2_K experts,
   which the loader and the imported kernels would have to accept together.
4. **Quality risk on MLA/CSA projections** is now measured (section 6.4):
   4 bits costs KL 0.092 against the shipped artifact, 6 bits 0.035, on
   tensors the official model keeps at FP8. The DS4 dense set needs a 5- or
   6-bit tier, which SHQ has only at 6.56 bpw today.
5. **Contract freeze.** SHQ is a candidate; artifacts may need regeneration.
   Converting an 80 GiB model is a multi-hour job per recipe.
6. **The NPU argument is weaker than the design assumed**; the shared layout
   pays for companions, not for the main model.

## 8. Recommended experiments, in order

Each has a measurable gate and none needs the next one to start.

- **E1. DS4 4-bit dense-set quality, GGUF-only (no SHQ).** Requantize the
  current artifact with `Q4_K`/`IQ4_XS` on `attn_q_b`, `attn_output_a/b`, the
  shared experts and `output`, `Q6_K` as the fallback tier, same imatrix. Run
  `tools/ds4/check.py all` and the antirez comparison. Gate: the quality
  contract passes. Cost: conversion time plus one qualification run. This is
  the go/no-go for the whole DS4 dense-set plan and needs no kernel.
  Done (section 6.4) for Q4_K, Q5_K and Q6_K dense sets with llama.cpp KL
  scoring, since gufo's DS4 runtime cannot load them. Result: 4-bit fails
  the quality bar, 5–6 bits is the recipe to qualify on the gufo contract
  once the runtime has kernels for it.
- **E2. SHQ4 decode GEMV on gfx1151 for `nextn.eh_proj` (#130).** One shape,
  batch one, CPU oracle. Gate: bytes/time within 5% of the Q4_K kernel's
  effective bandwidth on the same shape, exact against the oracle. This is the
  first SHQ GPU consumer and the template for every other tensor. A
  scratchpad microbenchmark (`scratchpad/shq_gemv_bench.hip`, not part of the
  tree) already answers the bandwidth half of the gate; see section 6.3.
- **E3. SHQ4 W4A8 prefill tile** on the Qwen FFN shape (17408x5120, M=2048),
  `iu8` WMMA with one K16xN16 microtile per B fragment. Gate: ≥ the Q8_0
  W8A8 rate (31.7 TOPS on that shape); U4Z and S4 variants both measured.
  The issue-rate model in section 6.3 (W4A8-G64 loop at 45 TOPS vs Q8_0's 41)
  says the gate is reachable; the full tile with LDS staging is what remains.
- **E4. Expert tier study.** The unweighted pre-study is done (section 6.2):
  2-bit tile tiers reach 30–33% RMS against IQ2_XXS's 35%, and 3 bits reach
  16–18%. Next: imatrix-weighted fitting of the two per-tensor LUTs (the DS4
  imatrix used for the shipped artifact, or one regenerated with the DS4
  calibration set), applied to one full layer's 256 experts, and the layer's
  output KL on the calibration set against the shipped IQ2_XXS/Q2_K layer.
  Gate: not worse than the shipped layer at equal bytes with `w2` at 3.09
  bpw. Then a decode GEMV microbenchmark of the tile tier against the
  imported IQ2_XXS kernel at the same bytes.
- **E5. NPU `bfp16` epilogue-free microbenchmark.** One AIE2P program
  consuming an INT8-mantissa, per-8-exponent block format natively. Gate:
  sustained TOPS on a 10240x5120 shape versus the measured 0.94 / 11 TOPS.
  Decides whether an NPU tier exists.
- **E6. Quantizer error feedback.** Add GPTQ-style column updates to
  `tools/gufo/shq.py` and re-run the Qwen3.5-0.8B retention table. Gate:
  SHQ4 relative MAE at or below Q4_K's on the same tensors. The
  whole-model calibration (section 4.2): uncalibrated uniform 4-bit of any
  block class costs KL 0.044–0.051 on Qwen3.8-27B, the calibrated mixed
  artifact 0.017; that 2.7x is the target of E6 plus the recipe, and it is
  what a 4.5 bpw SHQ artifact must reach to replace UD-Q4_K_XL.

E1 and E4 together determine what a DS4 SHQ artifact could be; E2 and E3 make
the Qwen numbers real; E5 and E6 are independent.

## 9. Experiment log (2026-09-13, this host)

Everything below ran from the session scratchpad; nothing was added to the
tree. Commands are recorded so the numbers can be reproduced.

| Experiment | Tool | Command shape | Section |
| --- | --- | --- | --- |
| Artifact tensor tables | `llama.cpp/gguf-py` `GGUFReader` | per-tensor type, size, shape; per-token byte accounting | 4.1, 4.2 |
| SHQ4/6 reconstruction on production tensors | `tools/gufo/shq.py` `quantize_shq4/6` + `gguf.quants.dequantize` | range search, no imatrix; unsloth tensors dequantized for comparison | 4.3 |
| Official DS4 0731 tensors | HTTP range reads of `model-000NN-of-00046.safetensors` from `deepseek-ai/DeepSeek-V4-Flash-0731` (header JSON, then `data_offsets`) | five expert matrices, `wq_b`, `wo_b`, shared `w1` | 4.1, 6.2 |
| Expert re-quantization bounds | numpy; per-32 uniform, per-16 asymmetric, per-block k-means LUT, two-table LUT | scored against the decoded MXFP4 values; shipped GGUF experts scored the same way | 6.2 |
| SHQ tile decode GEMV | HIP, `tools/bench/build.sh <scratchpad>/shq_gemv_bench.hip` | ≥1 GiB rotated arena, CPU oracle, SHQ4 contract order, lane-contiguous order, SHQ8 | 6.3 |
| W4A8 issue-rate model | HIP, `tools/bench/build.sh <scratchpad>/w4_unpack_wmma.hip` | register-resident `iu8` WMMA chains, nibble unpack variants, epilogue per K32/K64 | 6.3 |
| DS4 dense-set requantization | scratchpad build of in-tree `llama-quantize` with a one-line patch (skip the imatrix requirement for tensors that are copied, not quantized) | `--allow-requantize --tensor-type <family>=<type> ... --output-tensor-type q6_k --token-embedding-type f16 <in> <out> Q4_K_M 24`; experts and F16 tensors pinned to their current types | 6.4 |
| KL scoring | scratchpad HIP build of `llama-perplexity` | baseline: `-f wiki.test.raw -c 2048 --chunks 24 -ngl 99 -b 2048 -ub 2048 --kl-divergence-base base.bin`; candidates: `--kl-divergence --kl-divergence-base base.bin`; WikiText-2 raw test split from `ggml-org/ci` | 4.2, 6.4 |
| Qwen pure-format calibration | same `llama-quantize` (`--pure --output-tensor-type q6_k --token-embedding-type q4_k`, ftypes Q4_1 / Q4_0 / Q4_K_M) and `llama-perplexity` | against UD-Q8_K_L logits | 4.2 |

Experiment artifacts (`/persist/models/*-experiment.gguf`) were deleted
after scoring; the shipped artifacts are untouched.

## 10. References

- DeepSeek-V4 technical report: <https://arxiv.org/html/2606.19348v1>
  (architecture setup 4.2.1, hybrid attention 2.3, FP4 QAT 5.2.1).
- DeepSeek-V4-Flash model card: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>.
- Qwen3.8-27B model card: <https://huggingface.co/Qwen/Qwen3.8-27B>.
- AMD GPUOpen, WMMA on RDNA 3: <https://gpuopen.com/learn/wmma_on_rdna3/>.
- RDNA4 WMMA lane mapping and fused MXFP4 GEMM: <https://github.com/JohnTDI-cpu/rdna4-wmma-guide>.
- AIE2P `mmul_8_4`: <https://github.com/Xilinx/aie_api/blob/main/include/aie_api/detail/aie2p/mmul_8_4.hpp>.
- XDNA2 architecture summary: <https://www.emergentmind.com/topics/amd-xdna-2-npu>.
- Gemma3 on Ryzen AI NPU (Q4_1 g32, BF16 arithmetic): <https://arxiv.org/html/2602.06063v1>.
- ggml block layouts: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h>;
  i-quant design discussion: <https://github.com/ggml-org/llama.cpp/discussions/5063>.
- ik_llama IQ*_K quants: <https://github.com/ikawrakow/ik_llama.cpp/discussions/8>.
- QTIP: <https://arxiv.org/abs/2406.11235>; exllamav3 (EXL3): <https://github.com/turboderp-org/exllamav3>.
- OCP Microscaling formats (MXFP4): <https://arxiv.org/abs/2310.10537>.
- Strix Halo llama.cpp measurements: <https://github.com/visorcraft/strix-halo-llm-perf>,
  <https://kyuz0.github.io/amd-strix-halo-toolboxes/>.
- In-repo: [QUANTIZATION.md](QUANTIZATION.md), [GPU_BACKEND.md](https://github.com/gufo-org/gufo/blob/b129eda8aad0e577814fa92bb4c6ca28bd5996c0/docs/GPU_BACKEND.md),
  [NPU_BACKEND.md](https://github.com/gufo-org/gufo/blob/b129eda8aad0e577814fa92bb4c6ca28bd5996c0/docs/NPU_BACKEND.md), [benchmarks/qwen3.8-27b/README.md](../benchmarks/qwen3.8-27b/README.md),
  [benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md](../benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md),
  [benchmarks/deepseek-v4-flash/README.md](../benchmarks/deepseek-v4-flash/README.md).
