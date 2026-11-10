# Common optimizations for Strix Halo (gfx1151)

Distilled, cross-model record of what measurably works on the production
target (Linux x86-64, AMD Strix Halo `gfx1151`), what measurably fails, and
the measurement discipline that separates a real win from a profiler
artifact. It consolidates the per-model logs under [models/](models/),
the mechanism docs under [kernels/](kernels/), the rules in
[PERFORMANCE.md](PERFORMANCE.md) and the `optimize-kernel` skill. Per-model
documents keep the raw evidence; this file is the index of transferable
knowledge. Apply it to every new model (GLM-5.3-Flash and successors) before
inventing anything new.

## 1. Hardware facts that steer every decision

| Fact | Consequence |
| --- | --- |
| Wave32 execution (`__builtin_amdgcn_wmma_*_w32` family) | Design kernels for 32-lane waves; wave64 only won for batched Q4 gate/up with a separate 32-lane reduce — elsewhere it cost occupancy or divergence. |
| Measured streaming peak ~240 GB/s (`tools/bench/gfx1151_peak.hip`) | Decode-time traffic bound: tokens/s ceiling = peak bandwidth / bytes-per-token. Any path reading more bytes than the weights demands that much extra throughput. |
| 32 MB MALL (last-level cache) | A benchmark loop whose working set fits the MALL reports fake 400–860 GB/s. Size every microbenchmark like the model, and never compare across workset sizes. |
| Non-temporal (`cache_streaming`) loads always lose | They bypass the MALL, which helps even decode-sized streams. Measured >50% bandwidth loss, repeatedly, on every model that tried. |
| WMMA F16 ceiling is power/clock-limited: ~39–55 TFLOPS measured, LDS-fragment microbench 47–52, 59 TFLOPS burst | Compute-bound design should budget ~30–34 TFLOPS for kernels that also touch memory (what production kernels reach), not the marketing peak. |
| hipBLASLt F16 dense reaches only 19–26 TFLOPS on model shapes vs 30–34 TFLOPS for custom Q8→F16 WMMA tiles | Do not route dense model projections through hipBLASLt; the custom tile kernels win by 25–60%. |
| dp4a MMQ grouped path is ~55× slower per MAC than dense MMQ on gfx1151 | For grouped/routed quantized GEMM, dequantize to F16 *after* the LDS read and run WMMA; never keep integer dot products in the inner loop. |
| WMMA reads one A-operand half per lane (row `r` for even lanes' halves, `r+16` for odd) | Half-wave fragment loads + `v_permlanex16` exchange are free; exploit it when one operand is shared (GQA queries, paired gate/up). |
| UMA with the host; ~large VRAM held by a resident server | Benchmarks need exclusive GPU (`tools/bench/gpu_exclusive.sh`); lazy first-touch allocations show as ~100 ms stalls and must be warmed or pre-budgeted. |
| Private-memory (scratch) use is catastrophic in hot loops | Any array indexed non-statically, or scheduling across too many live values, spills VGPRs to scratch. Named vectors and fixed-index stages are mandatory; `tools/ci/kernel-resources.json` ratchets scratch. |

## 2. Techniques that work (cross-model, promoted)

### Memory-bound decode (GEMV)

- **One wave per output row, 4 rows per 128-thread block, unroll 4.** The
  base GEMV shape that reaches 231 GB/s (96% of roofline) on lm_head-sized
  matrices. Variants: 1–8 waves per block and 1–4 rows per lane were byte-
  identical but slower; the 4-row shape wins everywhere it was retested.
- **K-quant warp-dot kernels over the generic vec-dot path, plus split-K
  rows (9–15) mapped to a masked 16-row path.** Kernels with
  per-quantization-type warp dot products beat library dequant+dot.
- **Launch fusion:** `GemvMulti` (one x, many weights) and
  `GemvGroupedPair` (gate+up sharing x) remove ~140 launches per decode
  step; at a ~7–10 µs launch floor this is several percent of decode
  directly. Small F16 tiles at 128 columns and fixed-index codebook stages
  (no per-lane scratch) underpin them.
- **Fused epilogues must be bit-exact** (same sum tree as the separate
  kernels) or the byte-identical promotion gate fails.
- **Flash decoding (split-KV) attention for long context decode queries:**
  attention core 1.28 ms → 0.090 ms on a 35K-token cache; distribute tiles
  across splits with a small carry block, keep ≥4 resident blocks/CU.

### Compute-bound prefill (GEMM/attention)

- **Custom Q8_0→F16 WMMA tiles** (`f32_16x16x16_f16_w32`, BK2 pipeline
  stages, swizzled LDS, 11–12 KB stages per WGP) at 30–34 TFLOPS; 0x6400|q
  magic F16 construction from bytes (no unpack+convert chains).
- **Dequant after LDS read, fragments per-warp owned,** for quantized B
  operands; wave-owned code fragments (no weight LDS) where the codebook is
  small.
- **WMMA flash-attention prefill with an FP16 KV mirror.** Separate K and
  V^T LDS buffers so tile j+1's QK overlaps tile j's PV: two barriers per
  16-key tile instead of four. Half-wave fragment loads with
  `v_permlanex16` are exact but flat-to-slower here; 32-key tiles spill.
- **Bucket compaction + tile map for routed MoE** (expert id in the low 16
  bits of the map entry): turns ragged routing into dense 128-token tiles;
  fixed tile heights beat ratio-based dispatch (paired gate/up 128-token
  tiles, down 64-token tiles at ≥1,024-row batches).
- **Masked ragged tails / exact one-token tails:** a prompt a few tokens
  past a full resident grid should not launch a near-empty second wave
  (was +15.5 ms at pp2051). Wave-lanes-form-the-dot tails are bit-identical.
- **Deferred residual FFN in registers** (−126 MB traffic per layer chunk)
  and skipping F32 side-copies a downstream consumer reads as F16 only
  (router, gated experts) — remove the wider store, not the narrow ones.
- **Batch small projections (router rows, alpha/beta) into tiled Q8 GEMMs**
  once the batch exceeds ~1,024 rows; below that, keep GEMV.

### Speculation and scheduling (decode throughput)

- **MTP / prompt-lookup draft trees:** k=2 MTP was the measured sweet spot
  (60.8 tps); wide prompt-lookup verify (copy-based) 124 → 188 tps;
  confidence-gated draft chains +11%; bounded draft width (64 rows) keeps
  verify cost honest. Verify against the exact target logits tree — any
  approximation shows up as acceptance divergence.
- **Side-stream overlap** (shared expert, Z projections) is byte-exact and
  helped chunk scans, but on ROCm 7.2 side streams were disabled; overlap
  is not the general escape hatch — most co-running kernels slow each other
  (measured, twice).
- **Prefill chunk pipelining:** a chunk starts reading the next chunk's
  rows once its own arrived (bounded reader pool); avoids GPU idle at chunk
  boundaries without extra memory.

### Linear/hybrid recurrence kernels

- **Register-resident recurrence state** (kRows×kCols accumulator, named
  float4 windows kept in registers, never a non-statically indexed array);
  DPP butterfly reductions. A single 112 B private-memory spill reloaded
  every 4 tokens cost ~4% of a pp2048 chunk — hunting scratch is often the
  cheapest win.
- Exact order-preserving convolutions/state updates with FP32 operands and
  the original product/FMA order beat "faster" F16 rewrites that broke
  greedy decisions.

### Quantized-tensor decode (I-quant precedent)

- Core `ggml_dequant` + per-model MMVQ ports support the full block set
  including IQ1_S/IQ1_M (`vec_dot_iq1_m_q8_1`, N=6/nrow=5–6 tables) since
  `b20b5bd4`. IQ1_M decode is feasible today; optimizing it follows the
  GEMV rules in §2 and the F16-WMMA rule (dequant-then-WMMA) for batches.
- IQ1_M specifics vs K-quants: sub-1-bit weight reads are *below* the 240
  GB/s streaming ceiling's design point, so decode is even more latency/
  overhead-bound — fusion, launch reduction and codebook stage caching
  dominate; a 55× dp4a penalty still forbids integer inner loops.

### Cross-cutting small wins (promoted more than once)

- Fewer barriers with an unchanged sum tree; fixed FP32 reductions where
  the oracle is FP32/FP64.
- Removing an intermediate store/reload while preserving the F32 boundary.
- Excluding unused tensors at load (cold-start weight traffic without extra
  persistent copies).
- Bounded scheduling barriers (explicit `asm` fences) only where the
  compiler demonstrably reorders; compiler-version-sensitive (Clang 22 vs
  24 spill differently — pin versions when comparing).

## 3. Techniques that failed (do not retry)

| Idea | Why it died |
| --- | --- |
| hipBLASLt for dense model F16 | 19–26 TFLOPS vs 30–34 TFLOPS custom WMMA on real shapes. |
| dp4a/integer inner loops for grouped quantized GEMM | ~55× slower per MAC on gfx1151. |
| Non-temporal/streaming cache loads | MALL bypass always net-negative, any workset. |
| F16 WMMA chunked DeltaNet/linear-attention | ~500× error growth, flipped greedy decisions; recurrence accuracy is not negotiable. |
| BF16 residual streams | NLL regression; F32 boundaries must stay where the reference has them. |
| Approximate routing/selectors | Byte-parity break; parity of *selection*, not just values, is the gate. |
| vec4 int8 GEMV global loads | Slower than retained vec loads on this fabric. |
| 128/256-row down-projection tiles, 2-wave-group and wave64 GEMV variants, iterative-ILP reschedules | Repeatedly neutral or slower; the retained tile/wave shapes held across five architectures. |
| 2-stage-ahead LDS prefetch, fragment-barrier removal, duplicate accumulators | VGPR spills / occupancy collapse; explicit resources check catches these. |
| Microbench-packed operand transposes (e.g. packed-V) | 1.15× on the isolated kernel, +0.6–0.9% end-to-end — global KV bandwidth was the actual bound. Microbench wins must re-appear in the full path before promotion. |
| Kernel-overlap "free lunch" | Exact output, real overlap in trace, net slower: co-running kernels contend. |
| Library attention/MIOpen where a bespoke kernel existed | Bespoke LDS staging + lossless packing beat it (ASR conv), and removed six env switches. |

## 4. Numerical contract (how to promote without breaking parity)

1. Scalar oracle first, with a recorded error budget; FP64 controls for
   reductions and penalties; fast-math is off in hot exact paths — explicit
   FMA-order / `asm volatile` guards where contraction would reorder
   (fast-math reassociation and tie-rounding have burned every model).
2. Fused/reordered routes must be **bit-identical** to the replaced path on
   fixed histories (all logits rows, greedy hashes) before any speed claim.
3. Keep the standard speed benchmark and numerical quality tests in
   `tests/functional/`; a missing-model skip is not a pass.
4. `tools/ci/kernel-resources.json` ratchets scratch/registers; run the ISA
   mix (`tools/prof/isa_mix.py`) but remember it counts epilogue
   occurrences — a "perfect" mix that spills is still a loss.

## 5. Measurement contract (the pitfalls that fake wins)

- First profile pass runs up to 10× slow (instruction upload); time the
  measured passes only, and check profiler scope artifacts (a missing
  `Mark("sample")` once made lm_head look pathologically slow).
- Profiler idle % separates launch-bound from kernel-bound before chasing
  any kernel.
- Power/thermal: 55 °C cooldowns between rounds, ABBA alternation, full
  history reuse only when revision/toolchain/harness/model/settings/cache
  history all match; timings are per-request/per-phase at 5% + 3 ms, never
  averaged.
- Fit `a·n + b·n²` to separate linear (GEMV) from quadratic (attention)
  regressions with depth; roofline calibrate with `gfx1151_peak.hip` in the
  same session as the measurement.
- Iterate candidates at ≤16K context; escalate only real wins to
  32K/65K/100K. Long-context runs hide short-context regressions and
  amplify long-context-only gaps.

## 6. Open levers (carried across models)

- Share the KV tile across the full GQA query group in prefill attention
  (4× redundant KV reads today, qwen3.6 log; the dominant depth lever).
- hipGraph-style capture for decode launch storms (launch floor is still
  several % of small-batch decode after fusion).
- Async logits D2H overlap with sampling prep.
- IQ1_M-specific decode GEMV tuning: the shared MMVQ port is upstream-
  translated, not gfx1151-tuned; fusion + codebook stage caching should be
  re-verified per model shape (GLM-5.3-Flash plan).

## 7. Where the evidence lives

- [kernels/README.md](kernels/README.md) and subdirs: mechanism-level
  kernel design (GEMV tiers, WMMA tiles, fused ops, routed MoE).
- [models/qwen3.8-flash-next/EXPERIMENTS.md](models/qwen3.8-flash-next/EXPERIMENTS.md):
  the largest retained/rejected A/B log (hybrid attention + MTP + HC mix).
- [models/qwen3.6-35B-A3B/](models/qwen3.6-35B-A3B/): MoE prefill/decode
  journey (14 → 60 tps) and the open GQA-KV lever.
- [models/gemma4/](models/gemma4/), [models/ling3-flash/](models/ling3-flash/),
  [models/deepseek-v4-flash/](models/deepseek-v4-flash/),
  [models/flint/](models/flint/), [models/minimax-h3/](models/minimax-h3/):
  per-architecture transfers of the §2 catalog.
- Audio/image/TTS models (qwen3-asr, qwen3-tts, embeddinggemma-2,
  qwen-image-2.1): exactness-first variants of the same contract.
- [models/glm-5.3-flash/](models/glm-5.3-flash/): the next engine
  (`glm5-next`); applies this catalogue to IQ1_M hybrid-attention MoE.
- [PERFORMANCE.md](PERFORMANCE.md), [TESTING.md](TESTING.md),
  `optimize-kernel` skill: the rules this file summarizes.
