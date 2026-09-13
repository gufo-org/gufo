# Review of the SHQ investigation

Reviewed 2026-09-13–14 against `620f4ebf8fd6c516d8412c0bca92a7baf456796a`, draft PR #234. This is a review, not a change to the quantization or execution contract. **Do not use the investigation's current summary as a kernel roadmap.** Its strongest observations survive, but several proposed formats do not have the bytes or error claimed for them, and its quality and performance conclusions exceed the experiments.

## 1. Quantitative review

### Scope, evidence, and units

I read [SHQ_INVESTIGATION.md](SHQ_INVESTIGATION.md) in full before the normative [quantization](QUANTIZATION.md), [GPU](GPU_BACKEND.md), and [NPU](NPU_BACKEND.md) documents, then the measurement sources: [Qwen3.8](../benchmarks/qwen3.8-27b/README.md), [DS4](../benchmarks/deepseek-v4-flash/README.md), [official-kernel review](../benchmarks/deepseek-v4-flash/official-kernel-review.md), and [0.8B mixed precision](../benchmarks/qwen3.5-0.8b/MIXED_PRECISION.md). Older claims in those files are not automatically authoritative: where necessary I checked current allocation/kernel code, local GGUF headers, and the pinned official checkpoint headers.

New checks and retained original evidence are in [benchmarks/shq-review-2026-09-13](../benchmarks/shq-review-2026-09-13/README.md). Specifically: actual Python SHQ6 plane lengths, local tensor-inventory arithmetic, a paired chunk bootstrap of the original Q5/Q6 logs, and HTTP-range reads of official safetensors headers. **No new GPU throughput, model-quality, or NPU benchmark was run.** Archived scratchpad sources explain what the original experiments actually tested; they are not newly qualified kernels. The original logs report a dirty build tree, so their executable provenance is weaker than a pinned clean-build result.

Throughout, GB = 10^9 bytes; GiB = 2^30 bytes; bpw includes stated scale/zero metadata but not file headers, padding unless specified, or runtime caches. An inventory-based bandwidth quotient is **derived**, not measured DRAM traffic. “Same quality” requires an explicit reference, dataset, and acceptance threshold.

### 1.1 The format arithmetic is wrong in consequential places

> “SHQ6-T16-G64 … 6.56” and “SHQ6 (6.56 bpw) lands on the Q6_K row.”

Wrong. The actual `quantize_shq6` implementation in [tools/gufo/shq.py](../tools/gufo/shq.py) emits 768 code bytes and 32 scale bytes for 16×64 weights: `8(768+32)/1024 = 6.25 bpw`. G32 emits 768+64 = 832 bytes, or **6.5 bpw**. The isolated check reproduces both. The inherited 6.5625 label in the contract/recipe does not override the serialized bytes. Q6_K really is 6.5625 bpw, but uses a different scale hierarchy and 16-weight subgroups. Its KL row is not an SHQ6 measurement or quality prediction.

SHQ4 U4Z G64 = `4+16/64+4/64 = 4.3125`; S4 = 4.25; G32 variants = 4.625 and 4.5. SHQ8 G64 = 8.25. Those numbers are correct. The 0.8B measured file sizes do not become smaller merely because their tier labels were wrong.

> “a ‘SHQ5’ (5.5 bpw, 5-bit codes, same tile order)”

Five code bits plus BF16/G64 is **5.25 bpw**, not 5.5. A packed UINT5 zero adds 5/64, making **5.328125**. See §3.1 for a concrete layout. Five-and-a-half describes different metadata, such as a symmetric G32 scheme, not this G64 proposal.

> “all contiguous and 128-byte aligned, which is one coalesced 552-byte [burst]”

552 bytes is the correct *sum* of code, scale, and zero planes for a U4Z 16×64 group. Separate SoA planes are not one contiguous memory transaction. Also, tightly packed SHQ6 microtiles are 192 bytes apart; SHQ5 microtiles would be 160 bytes apart. Every microtile cannot be 128-byte aligned without padding. Specify alignment of plane/group bases and actual strides before computing bpw or claiming shared layouts.

The comparison table also merges unlike GGUF formats. Q4_K/Q5_K have 32-weight subgroups and minima; Q6_K has 16-weight subgroups, signed subgroup scales, and no minimum correction. IQ2_S is 2.5625, not 2.5 bpw. IQ4_NL has an FP16 scale per 32, whereas IQ4_XS has compressed subgroup scales. These distinctions affect both quality and unpack cost. [GGML block definitions](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h).

The Gemma NPU row's “Q4_1 g32 … 4.5” is likewise wrong for the cited scale-and-min representation: `4+(16+16)/32 = 5.0 bpw`. The published 32×256 packed block is 4096 code bytes plus 512 scale and 512 minimum bytes. It is useful evidence for that implementation, not a measured Strix Halo speedup. [Gemma3 NPU paper](https://arxiv.org/html/2602.06063v1).

### 1.2 DS4's 9.57 GB is a useful inventory, not a traffic measurement

> “DS4 Flash reads 9.6 GB per token, and 81% of it is not the experts.”

The arithmetic is substantially right **under a full-one-pass tensor-inventory model**:

| Component | Bytes or calculation | GB/token under that model |
| --- | ---: | ---: |
| Routed experts, all resident | exactly 72.5625 GiB | — |
| Active routed experts | `72.5625 × 2^30 × 6/256` | 1.826095104 |
| Q8_0 dense tensors | 6,598,885,376 bytes | 6.598885376 |
| F16 excluding embedding | 1,132,283,904 bytes | 1.132283904 |
| F32 | 1,845,596 bytes | 0.001845596 |
| I32 routing lookup, entire table | 9,308,160 bytes | 0.009308160 |
| Total excluding embedding | sum | **9.568418140** |

Thus `(9.5684−1.8261)/9.5684 = 80.915%`. This verifies the rounded 9.57/81%, not its literal “reads” wording. Reading the lookup *row*, not the whole table, and one embedding row gives **9.559118244 GB** in the accompanying audit.

There are larger workload-dependent omissions. Hash routing in the first three layers avoids their learned router matrices. The indexer query/projection path is conditional on more than 512 compressed entries: the shallow `tg128` benchmark does not execute about **0.3633 GB** of indexer weights per token. Consequently `17.7 × 9.57 ≈169 GB/s` is not measured bandwidth, and even its active-weight proxy overcounts that shallow workload; the adjusted proxy is roughly **163 GB/s**. Check dispatch conditions in [runtime/rocm_graph.cpp](../src/models/deepseek_v4_flash/runtime/rocm_graph.cpp), not just tensor names.

“Attention 5.2 GB” already includes the approximately 0.352 GB indexer query matrix. The separate “indexer/compressors 0.62 GB” bucket must exclude it or it is counted twice. Embeddings are row lookups, MTP-only tensors are not ordinary decode weights, and shared/batched weights can be reused. “A single-token step reads every dense weight once” is not generally true. KV, recurrent state, scratch, support-model work, and multiple passes are absent from this inventory.

> “Current … 9.40 … 18.1 … 25.6.”

This row is inconsistent with 9.57. Its quotient arithmetic is correct for 9.40, but the inventory drops roughly the router/mHC bucket. A scenario table must use one baseline and an explicit tensor manifest. Here is a reproducible replacement using the audit's 9.5591 convention, retaining embeddings, router/mHC, norms, biases, sinks and lookup tables; experts remain unchanged unless indicated. “Attention/shared” includes q_a, q_b, kv, output_a/b, shared gate/up/down and matching indexer q_b; “auxiliary” overrides indexer/compressor/head formats. These are **analysis-only byte ceilings**, not runtime predictions.

| Scenario | GB/token | tok/s at assumed 170 GB/s | tok/s at measured read ceiling 241 GB/s |
| --- | ---: | ---: | ---: |
| Current inventory | 9.5591 | 17.78 | 25.21 |
| Attention/shared SHQ6, actual 6.25 bpw | 7.7466 | 21.95 | 31.11 |
| Attention/shared SHQ4 U4Z | 6.3280 | 26.86 | 38.08 |
| Previous row with auxiliary SHQ8 | 6.0969 | 27.88 | 39.53 |
| All eligible dense SHQ4 | 5.5965 | 30.38 | 43.06 |
| Previous row, experts 3 bpw | 6.2052 | 27.40 | 38.84 |
| Previous row, experts 4.25 bpw | 7.2197 | 23.55 | 33.38 |

The auxiliary-SHQ8 row actually *widens* the indexer q_b relative to the previous row while narrowing other auxiliary tensors. This is why unlabelled family totals are insufficient to reproduce the investigation's 6.02-type scenarios. The approximately 5.6 GB all-dense result survives; equal effective bandwidth and quality do not follow from it.

The measured KL experiments use a **narrower** set: three large attention projections plus three shared-expert matrices, 5.410652160 billion weights; head always Q6_K; q_a/kv/indexer unchanged. For that exact set:

| Selected tier; head Q6_K | GB/token | 170 GB/s quotient | Gain over 9.5591 quotient |
| --- | ---: | ---: | ---: |
| Q4_K, 4.5 bpw | 6.7255 | 25.28 | 42.1% |
| Q5_K, 5.5 bpw | 7.4019 | 22.97 | 29.1% |
| Q6_K, 6.5625 bpw | 8.1205 | 20.93 | 17.7% |
| SHQ4 U4Z, 4.3125 bpw | 6.5987 | 25.76 | 44.9% |
| Proposed symmetric SHQ5, 5.25 bpw | 7.2328 | 23.50 | 32.2% |
| Actual SHQ6, 6.25 bpw | 7.9091 | 21.49 | 20.9% |

Using the investigation's full-lookup convention adds approximately 0.0093 GB and reproduces its rounded 6.74/7.41/8.13 KL-table rows. It does **not** reproduce the earlier “9.57 to 6.63” for the same Q4 recipe. Nor does “SHQ4 … on the same set … 6.46 GB/token, 26.3 tok/s” hold: it is approximately **6.60 GB, 25.8 tok/s** with the specified Q6_K head. Narrowing the head too is a different experiment.

The §6.2 per-family savings inherit the SHQ6 error: for q_b, `1.442840576e9*(8.5−6.25)/8 = **0.4058 GB**`, not 0.35; output_a/b save **0.8116 GB**, not 0.70; head saves **0.1489 GB**, not 0.13. SHQ8 would also save approximately **0.00845 GB** on q_a/kv rather than literally zero, although that is small. The corresponding approximately 0.75/1.51/0.57 GB SHQ4 savings are correctly rounded. None is a timing measurement.

### 1.3 Resident capacity and the proposed expert budget

> “the routed experts must stay at or below ~3.0 bpw”

There are exactly `43 × 256 × 3 × 4096 × 2048 = 277,025,390,592` routed weights. Hence expert residency is **32.25 GiB per bpw**:

| Expert bpw | Expert GiB | With illustrative 4.5064 GiB non-expert set |
| ---: | ---: | ---: |
| 2.0625 | 66.5156 | 71.0220 |
| 2.25, shipped weighted average | 72.5625 | 77.0689 |
| 2.4 | 77.4000 | 81.9064 |
| 2.427083, proposed exact 2/3-bit mixture | 78.2734 | 82.7799 |
| 2.625 | 84.6563 | 89.1627 |
| 3.0 | 96.7500 | 101.2564 |
| 3.25 | 104.8125 | 109.3189 |
| 4.25 | 137.0625 | 141.5689 |

The source's 66.4 and 78.1 values result from prematurely rounding 2.0625 to 2.06 and the proposed mixture to 2.42. Its text's approximately 4.7 GiB dense overhead and its table's approximately 4.5 GiB overhead also need a common manifest. Current target payload is **80.7594 GiB**, including **72.5625 GiB experts**, not 78 GiB experts.

> “fits in the same 78 GiB as today”

False. The proposed `(2.09375 × 2 + 3.09375)/3 = 2.4270833` costs **78.2734 GiB**, **5.7109 GiB / 7.87% more expert memory**. Calling it equal bytes materially misstates the comparison.

SHQ4 experts do not fit even without other consumers: 137.06 GiB is decisive. But 3.0 is an engineering budget, not a hardware bound. The bound is

`expert_bpw ≤ (usable_GiB − dense − support − KV − scratch − reserve) / 32.25`.

This host reports approximately **125.09 GiB** MemTotal. The cited 90.74 GiB workload includes approximately 80.76 GiB target plus 5.58 GiB support, leaving about **4.40 GiB** for the rest—not 10 GiB free-standing target scratch. A 3.25-bpw expert configuration with the illustrative smaller dense set, support and that short-workload remainder totals approximately **119.3 GiB** before reserve and additional context growth. It is tight, not arithmetically impossible. Conversely C8/deep contexts can make even 3.0 unsafe. A configured 262K context limit is not a measurement at 262K; the cited run's live context was much shorter.

### 1.4 Official checkpoint: mixed precision, but not the stated layout everywhere

> “159.6 GB total” and “the official dense set is 8.0625 bpw.”

The document mixes preview-checkpoint facts with its later pinned **0731** evidence. At revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, the official index reports **166,878,536,440 bytes (166.88 GB), 48 shards**, not the preview's approximately 159.6 GB/46-shard layout. The retained header checker reads that revision, not `main`. [Pinned official index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json).

MXFP4 expert codes with an E8M0 byte per 32 are correctly **4.25 bpw**. A dense FP8 tensor with one **E8M0 byte per 128×128 block** is instead `8+8/16384` = **8.00048828125 bpw**, excluding edge padding. Header examples confirm FP8 projection matrices and E8M0 block scales, including the shared expert and indexer wq_b. But **head.weight and embed.weight are BF16**, as is the sampled router gate. “Everything else FP8” or treating the head as an official FP8 tensor is false.

Indexer QK FP4 quantization describes an activation/operator path, not proof that all indexer weights use FP4 or that SHQ8 preserves its outputs. Equally, the official mixed-precision design is not a license to quantize every dense weight uniformly.

> “The KV cache is already the official FP8 + BF16-RoPE mix”

This confuses numerical rounding with storage. Current gufo DS4 compressed decode caches are float32 allocations containing rounded values, with additional FP16 mirrors used on a prefill route. `DS4_N_HEAD_DIM=512` includes the 64 rotary coordinates: there are **448 non-RoPE coordinates**, not 512+64. An ideal payload with FP8 non-RoPE and BF16 rotary entries is `448+2×64 = **576 bytes**`, plus scale metadata—not 640. Actual allocations and reads are much larger (§3.2). Do not infer a packed cache from an FP8-quantization function name.

### 1.5 Profile and microbenchmark interpretations

> “projection families are 51% of decode GPU time … 6.9 GB … roughly 155 GB/s [expert GEMV].”

The cited [profile JSON](../benchmarks/deepseek-v4-flash/official-kernel-review.json) totals 1017.808134 ms for 16 tokens. The three named Q8 families sum to **434.031955 ms = 42.64%**, not 51%. Including the additional paired-Q8 family yields **493.483466 ms = 48.49%**. With the actual 6.598885376 GB Q8 payload, `6.598885376/(493.483466/16/1000) = approximately **214 GB/s**`. The approximate bandwidth is defensible only after fixing the numerator and identifying all included kernels.

Routed gate/up plus down take **177.474106 ms / 16 = 11.0921 ms/token**. Their 1.826095104 GB gives approximately **164.6 GB/s**, not 155. Gate/up alone is approximately 143.8 GB/s, down approximately 213 GB/s. These remain inventory/time proxies, not memory-counter measurements. Halving a 48.49%-of-time component gives `1/(1−0.4849/2) = **1.32×**` whole-decode speed, not 2×. Different shapes cannot inherit one global bandwidth.

> “the measured Q4 prefill deficit … is 79% the K-quant minimum correction”

The benchmark's 79% attribution concerns the **extra instruction count**, not 79% of elapsed prefill loss. Instruction classes have different dependencies, overlap and memory effects. The 474 versus 557 throughput comparison is a **14.9%** deficit (or 17.5% gain in the reverse direction); the separate min-correction ablation is evidence for that ablated kernel/workload, not a timing decomposition into 79/21. The published 8.6% end-to-end ablation cannot simply be converted into that percentage attribution.

The basic local rooflines—241/220/209 GB/s read/write/copy, 55.07 TOPS INT8, 55.05 TFLOPS BF16, 27.08 TFLOPS FP32, 25.75 TFLOPS hipBLASLt—are correctly transcribed. `25.75/55.05 = 46.8%`; `209/241 = 86.7%`. They do not establish format-independent model throughput.

> “SHQ8 … reaches 84–95% … on every shape.”

The table itself says **190/241 = 78.8%** for DS4 q_b. More seriously, the archived `Gemv8Split` packs `[tile][group][lane][64]`, not the contract's `[tile][group][subtile][lane][16]`. It is **not a measurement of contract-order SHQ8**. For Qwen FFNs, SHQ4 contract order reaches **80.5–87.6%**, while **83.8–91.7%** belongs to the reordered layout. The summary combines them into “contract … 80–92%.”

> “within 10% of it on the FFN shapes” and “another 3–9%”

For ffn_gate, `194/222−1 = −12.6%`, outside 10%. Reordering gains range **3.3–9.6%** across listed shapes; the maximum is `194/177−1`. DS4 SHQ4 results are only 60–76% in contract order. A “latency-bound” diagnosis and split-K remedy are plausible, but occupancy/issue/stall or controlled kernel ablations are needed to establish the cause. A shared-expert kernel apparently exceeding the separate streaming comparison also warns against treating unlike microbenchmarks as identical ceilings.

The GEMV check uses a sequential CPU accumulation versus a different GPU reduction and a relative-error denominator with a floor; it is not bit-exact verification of the normative oracle. Its mismatch printing does not enforce a failing process exit. Report actual tolerance, input distribution and output errors. A fast approximate scratchpad GEMV does not establish exact runtime compatibility.

> “45 TOPS against … 41, so SHQ4 prefill should land at or above the Q8 rate”

The table's register-loop rates and their relative percentages are arithmetically reasonable, but the model is incomplete. Archived `w4_unpack_wmma.hip` uses constant scales `1.0000001f` and `0.9999999f`: their FP32 product rounds to **1.0**, allowing the intended runtime scale product to fold away. It omits the U4Z zero correction, activation quantization, realistic scale/data loads, and LDS/global-memory pipeline. Its timing path does not provide a useful checked output. Inspect generated ISA and retain a correctness-producing variant before interpreting instruction costs.

The unpack path also contains byte permutations; “no permutation” is not established by saying weights have the same logical tile order. WMMA fragment mapping, signedness, and physical packing must match. **The experiment supports an issue-rate hypothesis, not a prefill throughput forecast or exact shared GPU/NPU arithmetic.** Integer dots can agree while FP32 group ordering and activation quantization differ.

### 1.6 Expert re-quantization: the advertised formats were not tested

> “All five tensors gave the same numbers to within 0.3 points.”

The sample is three w1 matrices and two w2 matrices, not measured w3 coverage. Shipped IQ2 w1 RMS errors are approximately 35.38–35.44%; shipped Q2 w2 errors approximately **27.59–28.35%**, a **0.76-point** spread. Distinguish within-family synthetic results from the shipped rows, and do not generalize a few experts in two layers to the entire checkpoint.

> “per-block optimal 4-level LUT (lower bound for any per-32 4-level scheme)”

The archived script runs eight Lloyd iterations from quantile initialization. This is a **feasible local solution, not a certified global optimum or lower bound**. Even a globally optimal scalar four-level fit would not bound a vector codebook such as IQ2_XXS or a differently grouped format. “Every 2-bit scheme lands at 30–35%” is contradicted by the same table's 27.6–28.4% Q2_K row.

The most important defect is the association between **measured error and claimed bpw**. The uniform and two-table experiments optimize a scale as `amax × f` over floating-point grids. Those scales are generally **not powers of two** and cannot be represented by the proposed original E8M0 exponent compressed to two bits. Therefore **32.6% at 2.09375 bpw and 17.6% at 3.09375 bpw were not demonstrated**. Storing FP16 scales per 32 instead costs 0.5 bpw: 2.5/3.5 before a selector, not 2.09/3.09. Alternatively restrict the fitter to the proposed decoder's actual exponent/LUT choices and remeasure. This can change both error and memory feasibility.

> “there is no entropy-coding gain to take”

Entropy 3.864 versus four code bits leaves approximately **0.136 bits/weight**, or **3.4% of code bytes**, before coding overhead. Small is not zero; correlations were not tested. Four distinct exponents in five tensors do not establish a two-bit range for every tensor. Scan every expert's scales, retain an overflow strategy and count its metadata.

> “equal or better error at equal bytes, and a materially cheaper kernel”

Neither half is established. Equal bytes is false (§1.3); fitted scales do not implement the proposed format; and unweighted RMS is not imatrix-weighted loss. Applying imatrix weighting to both candidates can reverse the ranking. A register LUT may be cheaper than a memory codebook gather, but no complete routed kernel was benchmarked. `ldexp` is not automatically cheaper than multiplication, and W*A8 still needs the activation scale in its epilogue. The proposed tier is a **decode-cost hypothesis**, not an IQ2_XXS quality improvement or a demonstrated bound on achievable quality.

The cited IK discussion does not justify grouping every IQ*_K format under one layout: its examples include IQ2_KS 2.1875, IQ2_K 2.375 and IQ3_K 3.4375 with differing groups/scales; IQ4_KS's two tables have 16 entries, not four or eight. CPU trellis comparisons do not establish GPU inferiority on this target. [IK format discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/8).

### 1.7 KL measurements: genuine signals, overinterpreted reference

The retained DS4 logs support the rounded table values:

| Candidate | PPL | Change vs reused baseline 5.862845 | Mean KL | Top-1 agreement | p99 / maximum KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected Q6_K, head Q6_K | 5.922057 | +1.01% | 0.034843 | 92.616% | 0.355335 / 2.318457 |
| Selected Q5_K, head Q6_K | 5.848013 | −0.253% | 0.048935 | 91.349% | 0.483502 / 6.021235 |
| Selected Q4_K, head Q6_K | 6.234338 | +6.336% | 0.092348 | 88.551% | 0.902025 / 7.828456 |

Attention-only Q4_K with shared/head Q8 has approximately PPL 6.174, KL 0.076 and 90.0% agreement. Those are valid **llama.cpp requantization observations**, not gufo SHQ results. The head changes in the combined experiment, so the additional approximately 0.016 KL cannot be attributed uniquely to shared experts. KL is not additive across tensor sets; the difference is a conditional effect, not an independent loss component. The capture run's approximately 5.8806 baseline PPL also differs from the reused-logit baseline; run an A/A check and document teacher-logit serialization and scoring before treating sub-percent shifts as exact invariance.

> “WikiText-2 … 24x2048” and “within the ±1.7% chunk noise”

There are **49,152 input tokens**, but this scorer evaluates the second half of each chunk, excluding its last position: **24×1023 = 24,552 scored positions**. The printed PPL standard error is not a paired chunk-level confidence interval. Teacher probabilities are stored in the scorer's compressed representation and its KL accumulation uses a small-probability cutoff; call this the scorer's approximate full-vocabulary KL, with direction **D(shipped || candidate)**, rather than an exact mathematical full-logit oracle.

I reconstructed each chunk mean from the 24 cumulative rows and bootstrapped paired chunks 100,000 times, seed 731. This uses the rounded logged values; it is not a replacement for retained per-token arrays.

| Paired statistic | Mean | 95% chunk-bootstrap interval |
| --- | ---: | ---: |
| KL(Q5) − KL(Q6) | +0.01409 | [+0.01167, +0.01667] |
| NLL(Q5) − NLL(Q6) | −0.01258 | [−0.01994, −0.00558] |
| NLL(Q5) − NLL(base) | −0.00253 | [−0.00885, +0.00374] |
| NLL(Q6) − NLL(base) | +0.01005 | [+0.00446, +0.01561] |

**All 24 chunks favor Q6 on KL. These data are sufficient to separate Q5 and Q6 on this slice.** They favor Q5 on NLL, illustrating why NLL and fidelity are different objectives. They do not establish broad equivalence, tail safety or task quality: the chunks come from one WikiText slice and need not be independent samples of deployment traffic. Use document/task-level paired resampling over a broader suite, not 24,552 nominally independent tokens. [Reproducible paired calculation](../benchmarks/shq-review-2026-09-13/paired_stats.py).

> “Because … every expert tensor [is unchanged], the KL isolates the dense-set change.”

It isolates the **conditional intervention on this already-quantized artifact**, including interactions with its expert errors and its changed routing/activations. It does not measure loss against official FP8/MXFP4, nor permit adding this KL to an unknown expert KL. Preserve this baseline for a controlled regression experiment, but use official checkpoint logits/trajectories as the reference for total model fidelity. Both comparisons are needed; neither replaces the other.

> “the MLA projections … one fifth of the active parameters … far more sensitive per bit than a dense transformer's projections”

The selected 5.4107B matrices plus 0.5295B head are approximately **45% of roughly 13.27B active parameters**, not one fifth. The total dense set is roughly half. V4's shared-KV/compressed-attention path is not simply V3's absorbed MLA. Cross-model KL ratios confound architecture, teachers, quantizers, grouping, calibration and bit allocation. The attention-only ablation establishes sensitivity of that **set in this configuration**, not a causal architectural explanation or identification of a single guilty projection. Per-family and per-layer ablations are still missing.

> “KL, not perplexity, is the gate … the 736-choice replay contract [is a] top-1 [property].”

KL is informative but the gufo contract does not contain an automatic acceptance threshold at 0.035 or 0.049. Two top-1 properties must not be conflated. Official-reference trajectory fidelity constrains the changed target. **DSpark replay requires the verifier to reproduce ordinary autoregressive execution of the same candidate target**, with the prescribed frontier/tie semantics; it does not require every changed model to reproduce the old target's WikiText choices. Q5's 8.651% disagreement neither automatically fails that replay nor predicts a 91.349% draft acceptance rate. It does make “baseline perplexity” inadequate qualification. These artifacts were rejected by gufo's type checks, so the gufo gates were never run. “E1 done/fails the bar” is not a formal contract result unless the bar is separately defined.

### 1.8 Qwen calibration and the alleged quantizer-only explanation

The pinned UD-Q4_K_XL payload is **17,548,181,504 bytes = 16.343 GiB**, for 27.3207B counted weights/scalars, **5.1384 bpw**. The format inventory contains **eight quantized types plus F32**, not seven quantized types. Do not mix the older local Q4 snapshot with the pinned one. Q8_K_L used as KL teacher and Q8_K_XL throughput artifacts are also not interchangeable; a theoretical pure-8.5-bpw size is not either file's measured size.

> “Validation against the Q8 artifact: identical top-1 over 1024 tokens, 4.7x the logit RMSE.”

This overstates and misidentifies the test. The source compares prefill with the sequential reference for each artifact over the **complete final-token vocabulary of a 1024-token prompt**. Token ID 198 matches at that position; it is not 1024 matching greedy choices, nor a full cross-artifact trajectory comparison. `0.09512630/0.02007260 = 4.74` correctly compares the two reported execution-validation RMSEs, but is not Q4-versus-Q8 quantization RMSE. The separate KL run is the relevant cross-artifact experiment.

> “The measured artifacts already run at 97–98% of their byte ceiling.”

Stored-file bytes include the embedding and MTP-only block. Excluding `blk.64` and charging one embedding row gives approximately **16.4820 GB/main token**. At 11.59 tok/s the derived weight-bandwidth proxy is approximately **191.0 GB/s**, not 203/209. Against an assumed 209 GB/s it is **91.4%**, not 97–98%. Cache/state and actual memory counters remain separate.

> “SHQ recipe ~4.5 bpw (ffn_down, embed, linear-attn at SHQ6, rest SHQ4)”

This is not a counted recipe. Including the proposed output projections/head, approximately **38%** of weights are up-tiered. `0.38×6.25 +0.62×4.3125 ≈5.05 bpw`, or approximately **5.17** with the erroneous 6.5625 tier. The isolated inventory calculation gives approximately **16.06 GiB** and **15.993 GB/main token** using actual SHQ6, before preserving additional small sensitive tensors. That is only approximately **3% idealized decode gain**, not 17%. A 4.5-bpw G64 recipe can up-tier only `(4.5−4.3125)/(6.25−4.3125) = **9.68%**` of weights. Specify a smaller set or stop claiming that size.

The pure-format ratio `5.14/4.3125 = 1.192` correctly gives approximately 19.2% from average bpw, under equal traffic/efficiency and ignoring fixed consumers. A 4.5-bpw ratio gives **14.2%**, not 17%. These resident-average ratios also need adjustment for the actual decode-active set. In the shape list, full-attention `attn_q` includes both query and gate halves; a separate full-attention gate projection double-counts it. The separate `attn_gate` tensors belong to DeltaNet layers.

The original Qwen KL logs support these measurements against the Q8_K_L teacher:

| Artifact | Mean KL | Top-1 agreement | PPL ratio to teacher |
| --- | ---: | ---: | ---: |
| Mixed UD-Q4_K_XL | 0.016732 | about 94.73% | about 0.995 |
| Pure Q4_1 | 0.043991 | 91.671% | 1.012721 |
| Pure Q4_K | 0.046145 | 91.707% | 0.999622 |
| Pure Q4_0 | 0.050767 | 91.092% | 1.018976 |

> “calibration and recipe, not block format, are the 2.7x”

Unsupported causal exclusion. The ratios are **2.63, 2.76 and 3.03**, and the experiment simultaneously changes tensor bit allocation, quantizer/calibration, representation and sometimes source/requantization route. Q4_1 has FP16 scale plus floating minimum per 32 (5 bpw); SHQ4 has BF16 scale plus integer zero per 64 (4.3125 bpw). They are not identical representational classes. A matched-byte factorial experiment is required: common high-precision source, identical tensor allocation, format × quantizer × calibration, same evaluations. A pure permutation of unchanged quantized values has no quality effect; changing grouping, scale precision or zero representation can.

The reconstruction evidence does show a plausible quantizer opportunity, but relative MAE, RMS, imatrix-weighted error and logit KL are different metrics. The production reconstruction table has a Qwen Q4_K comparison; it does **not** establish Q4_K's same error on the DS4 tensors. “9.9% versus 8.0% … on the same tensors of both models” is unsupported. Individual rows also exceed the summary's tight ranges. Q5_K's lower error is partly a larger byte budget. The 0.8B mixed result (approximately 0.050 KL, 550 MB versus 533 MB) comes from a small matched-position evaluation/calibration and lacks a matched unsloth KL proving equal quality. It is a lead, not production qualification.

Requiring unweighted MAE to beat Q4_K, as a proposed experiment gate, is not a substitute for weighted/model quality: an effective imatrix fit can worsen unweighted MAE while improving inference. Conversely, failure of one uncalibrated Q4_K recipe does not prove every affine SHQ4 recipe fails.

### 1.9 NPU and external-performance claims

> “every main-model offload route measured net zero or negative”

The Qwen source includes a split-prefill **estimated approximately +1.3%** result, not uniformly zero/negative, and explicitly leaves W4A8 follow-up open. The approximately **47 GB/s** streaming and **11/0.94 = 11.7×** arithmetic/epilogue contrast are measurements of those AIE programs and shapes, not universal NPU ceilings. Dividing 47 by eight gives 5.9 GB/s/column; it is not eight independently measured shim limits. The BF16 approximately 25 TFLOPS peak needs an authoritative definition/measurement, not a secondary overview treated as a local roofline.

> “a +23% main-model gain was measured [for a companion model]”

`6.21/5.04−1 = 23.2%` is correct, but compares a real GPU companion workload with a **synthetic heavy NPU workload**, not the same complete companion ported and quality-qualified. Main-only 7.60 versus 6.21 still shows interference. It does not measure the benefit of SHQ's one-resident-copy property. No DS4 main-model NPU offload result was supplied; do not generalize one Qwen experiment into a platform-wide impossibility.

The BFP16 mathematical payload can be nine bpw (`8×8+8` bits for eight values), but that does not establish nine-bpw DMA/storage, exact layout, or epilogue-free *end-to-end* execution. Activation conversion, block packing, scale semantics and quality must be included. [AMD Quark BFP description](https://quark.docs.amd.com/release-0.9/pytorch/tutorial_bfp16.html).

The external 5.1B-active, 4.25-bpw, 51–55 tok/s example implies `5.1e9×4.25/8×(51…55) = **138–149 GB/s**`, not 160–170 without additional dense/metadata bytes. Pin the artifact and route before using it as a ceiling. A reported 40.8 TFLOPS/53% result on **gfx1201** is not measured gfx1151 performance. Neither CPU trellis results nor another GPU's LUT kernel establish that QTIP/EXL3-style alternatives are ALU-bound here; benchmark equal-quality/equal-byte candidates on this machine. [RDNA4 WMMA experiment](https://github.com/JohnTDI-cpu/rdna4-wmma-guide), [QTIP](https://arxiv.org/abs/2406.11235).

## 2. Re-evaluation of the six summary points

| Summary point | Verdict | Evidence-supported replacement |
| --- | --- | --- |
| 1. Decode is a bytes/token problem; every format decision reduces to bytes and roofline | **Supports a weaker version.** | Single-row weight projections are strongly bandwidth-sensitive. Routing, shape latency, KV/state, speculative execution, activation conversion and concurrent scheduling also matter. Inventory-derived 169 GB/s is not a hardware measurement. A decode-only roofline is not the entire format decision. |
| 2. DS4 is 81% dense traffic; 5–6 bits is defensible and four is not | **Supports the inventory and a weaker quality conclusion.** | The 81% one-pass weight inventory is reproducible. Uncalibrated Q4_K on the tested set has substantial conditional fidelity loss. Q5_K and Q6_K are sensible **qualification candidates**, not qualified recipes; SHQ6 is neither Q6_K's bpw nor its numeric representation. Official-reference quality, gufo kernels and speculative gates remain absent. |
| 3. SHQ4 experts do not fit; the proposed two/three-bit tile plan is equal-size and promising | **Supports the capacity rejection, not the advertised expert plan.** | 137.06 GiB of experts alone cannot fit. Approximately three bpw is a context-dependent budget. The proposed mixture is larger than the shipped experts and its error came from scales its format cannot store. It may reduce decoding work, but quality superiority to imatrix IQ2_XXS/Q2_K is unproven. |
| 4. Qwen gains modestly on decode and reaches or exceeds Q8 prefill | **Supports a weaker decode hypothesis; does not support the prefill forecast.** | A truly pure 4.3125-bpw artifact has a favorable byte budget if quality survives. The stated mixed recipe is approximately 5.05 bpw and buys only about 3% idealized main-token weight savings. The 79% timing attribution and incomplete WMMA loop cannot establish Q8-level prefill. |
| 5. Shared GPU/NPU layout is unsupported for the main model but valuable for companions | **Supports a weaker, workload-specific conclusion.** | Existing Qwen offloads do not establish a compelling production split. They do not rule out DS4, another format, batching, or a different partition. Synthetic companion contention supports further investigation, not a measured SHQ-layout benefit. BFP16 deserves a complete pipeline experiment. |
| 6. The layout is at the roofline; quantizer/recipe, not format, explain the gap | **Does not support the categorical conclusion.** | Some scratchpad GEMVs are promising, but SHQ8 used another order and DS4 SHQ4 shapes are far below the ceiling. Calibration/allocation probably matter, yet the experiments do not isolate them from representation. Both layout and quantizer remain open engineering questions. |

The decision-relevant distinction is **qualification versus ranking**. The paired WikiText result ranks Q6 above Q5 on teacher fidelity, and Q5 above Q6 on NLL on this slice. It does not make either safe for the official-reference or DSpark contract. Nor does an 8.7% changed greedy-choice fraction alone answer whether the exact verifier implementation is correct. Keep four separate records: official-reference model quality, shipped-reference regression, same-candidate AR/verifier exactness, and measured draft acceptance/throughput.

## 3. Missing analysis and concrete measurements

### 3.1 A concrete SHQ5 proposal, not “Q5_K in tiles”

**Analysis/design proposal.** Keep a logical K16×N16 microtile. Five bits per weight require **160 code bytes**: either contiguous little-endian five-bit fields (ten bytes per lane's 16 values), or a **128-byte low-nibble plane plus a 32-byte high-bit plane**. The latter exposes the existing nibble unpack path and a separate bit extraction, avoiding awkward five-bit fields crossing machine words. Group four microtiles at K64: 640 code bytes plus 32 BF16 scale bytes = **672 bytes/1024 = 5.25 bpw**. Plane/group alignment and padding must be explicit; rounding every 160-byte tile to 256 bytes destroys the intended saving.

For a symmetric tier store signed two's-complement INT5, −16…15. With low nibble `lo` and high bit `hi`, reconstruct `q = lo − 16*hi`; equivalently sign-extend the combined five-bit field. For affine UINT5, reconstruct `w=s*(u−z)`, `z∈[0,31]`. Sixteen five-bit zeros per K64 tile group cost ten bytes, giving **682 bytes/1024 = 5.328125 bpw**; a byte-per-zero implementation costs 5.375. A four-bit zero field cannot represent the unrestricted UINT5 format. G32 symmetric would be 5.5 bpw, a separate precision/epilogue tradeoff.

On **iu8 WMMA**, unpack both operands to the required signed/unsigned byte fragments; five-bit weight codes do not have a native WMMA instruction. Relative to W4A8, add extraction of 16 high bits per lane fragment and combination/sign extension. For K64, accumulate four K16 integer dots, then apply dynamic activation and weight scales. The affine variant additionally subtracts `z*sum(a)` using the exact activation group. This must follow the normative overflow, rounding and FP32 accumulation order.

On **decode GEMV**, expand codes, accumulate the activation-weight dot per group, and apply BF16 scales. An affine implementation can reuse activation sums across output lanes if it measures better. Extra bitplane traffic and integer instructions compete with occupancy, load coalescing and reduction work; there is no defensible cycle count from the existing W4 loop.

**Recommendation:** start with symmetric SHQ5 as the simpler baseline, but do not select it solely from arithmetic. Fit symmetric and affine variants from the **same source and calibration**, compare weighted error and logits at matched total bytes, then benchmark complete kernels. Affine costs only 0.078125 additional bpw with packed zeros but adds a correction and may represent skewed groups better. Test G32 versus G64 and selective outlier handling too. A measured quality/throughput Pareto frontier, not the word “five-bit,” decides the tier.

### 3.2 KV cache, recurrent state and DSpark are missing byte consumers

**Analysis from current allocations, not measured DRAM traffic.** DS4 has 43 layers, 512-dimensional shared KV, a 128-row raw attention window, 21 CSA layers and 20 HCA layers. At context T, the current long-cache allocation includes:

`21*(floor(T/4)+2)*(512*4 + 512*2 + 128*4)` bytes

`+20*(floor(T/128)+2)*(512*4)` bytes,

plus raw rings, compressor state, workspaces and allocation headroom. The CSA terms are float32 compressed KV, its FP16 prefill mirror and float32 indexer keys. Leading growth is **19,136 bytes/context token**. Decode uses float32 compressed values; the mirror is not proof of FP16 decode storage. Allocation rounding and simultaneous old/new buffers during growth can raise peak memory beyond this formula. See [DS4 runtime allocation and dispatch](../src/models/deepseek_v4_flash/runtime/rocm_graph.cpp).

An optimistic **distinct-data, one-pass decode-read model** at T≥4096 is:

`43*128*2048 +21*512*2048 +21*(T/4)*512 +20*(T/128)*2048`

`= 33,292,288 + 3008*T bytes`.

The terms are the raw window, selected CSA KV, the growing indexer scan, and HCA KV. It is not actual DRAM traffic: QK/PV passes, query heads, cache reuse, score writes, sorting/top-k and kernel access patterns alter it. Multiplying every byte blindly by 64 heads is equally unjustified without measuring reuse.

| Context tokens | DS4 leading long-cache residency, GiB | DS4 distinct decode-read model, GB/token | Qwen FP16 attention KV residency, GiB |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.0730 | 0.0456 | 0.25 |
| 16,384 | 0.2920 | 0.0826 | 1 |
| 65,536 | 1.1680 | 0.2304 | 4 |
| 131,072 | 2.3359 | 0.4276 | 8 |
| 262,144 | 4.6719 | 0.8218 | 16 |

DS4 C8 can therefore require approximately **37.4 GiB** just for leading long-cache terms at 262K per sequence; target/support weights and other allocations are additional. The short-run 90.74 GiB observation cannot qualify that capacity.

Even this optimistic 0.822 GB/token at 262K reduces an all-dense byte-only ratio from `9.559/5.596 = 1.708×` to `(9.559+0.822)/(5.596+0.822) = **1.618×**`. Actual long-context attention may reduce the gain further. The weight-only projections are most credible at short contexts, but indexer activation changes the weight set around its threshold.

Qwen's current default attention cache policy is FP16: `16 full-attention layers × 2(K,V) ×4 KV heads ×256 dimensions ×2 bytes = **65,536 bytes/context token**`; FP32 doubles the last column. A single complete read has the same byte count, but actual traffic depends on GQA reuse and the attention implementation. The DeltaNet layers add approximately **144 MiB of live recurrent state** (`48×48×128×128×4`), roughly **302 MB/token** for one read/write pass, independently of context; allocation/checkpoint overhead is separate. At 262K, 16 GiB of unique KV reads rivals the approximately 16.48 GB active weights. A three-percent weight-only recipe improvement then becomes approximately **1.5%** in even the optimistic combined-byte model. See [Qwen cache policy](../src/models/qwen/hip/execution_policy.hpp) and current recurrent-state allocations rather than older dual-cache descriptions.

The DSpark support artifact occupies **5.5778 GiB / 5.9891 GB**, but it is also MoE: charging its entire file once per target token is wrong. Its six-of-256 expert plus dense one-row inventory is approximately **0.6807 GB**. This is a working upper inventory, not a measured marginal byte count: token-indexed tables are lookups, some projections/features can be reused, and shared target-head use must be counted at its actual dispatch sites. A full target-head pass, where required, adds approximately **0.5626 GB** per pass; do not silently charge it zero or once per emitted token. Its three-layer 128-row live KV window is only approximately **0.75 MiB**, apart from reserved capacity and feature/snapshot buffers.

For a speculative cycle, use

`bytes/emitted token = (target verifier + draft/support + head passes + KV/state + replay/capture) / emitted tokens`.

Measure the denominator and actual routing. Dense weights can be reused across verifier rows; routed expert unions can remain nearly linear. Under an explicitly hypothetical independent uniform router, m rows select an expected `256*(1−(250/256)^m)` unique experts per layer, approximately **44** for eight rows rather than 48; actual routing need not be uniform. The published approximately 20.5 versus 16.2 tok/s chat result and approximately 0.66 acceptance are more relevant than a synthetic maximum near 39.5 tok/s. Neither transfers unchanged after target re-quantization; C8 behavior must be retested.

**Measurement needed:** fixed prompts at 4K/16K/64K/128K/262K, C1 and capacity-feasible C8, AR and DSpark. Record peak resident/GTT memory, actual lengths, effective cache dtype, dispatched kernels, read/write counters where available, component times, expert unions, draft count, accepted and emitted tokens. A packed-KV optimization is a separate contract/quality change; it must not be assumed in an SHQ forecast.

### 3.3 Prefill, C8 and batched verification

**Analysis.** C8 means concurrent sequences, not necessarily an eight-row matrix. Decode may group eight rows; chunked prefill may have hundreds or thousands; speculative verification has its own physical width, padding and rejected rows. Dense weight reuse improves with rows, whereas routed expert GEMMs may still have few rows per expert. Compare equal prompt length, depth, batch policy and total work. The cited DS4 C8/depth results are not evidence for an eightfold gain over C1.

For a dense matrix with m reused rows and b bpw, ideal arithmetic intensity is approximately `16m/b` operations/byte. The measured 55 TOPS / 241 GB/s balance is approximately 228 ops/byte; the crude crossover is `m≈228b/16`, around **75 rows for symmetric SHQ5**, **89 for SHQ6**, **121 for Q8_0**. This is a roofline thought experiment excluding metadata/activations/epilogues, not an achieved crossover. Eight-row verification can remain weight-bandwidth-sensitive while large prefill becomes arithmetic/issue-sensitive.

The K64 epilogue amortizes scales over four WMMA instructions rather than two, but A8 quantization must generate scales and any sums at exactly that group size. Reusing a Q8_0 K32 activation buffer without conversion changes the arithmetic. Include absmax/reduction, rounding/clipping, sum computation, scratch writes, scale loads, conversion, affine corrections, accumulation order and synchronization in end-to-end timings. Calibrate on prefill/verifier activations too, not only teacher-forced single-row decode. DS4's cited approximately 36% quantized-matmul prefill share cannot inherit Qwen's much larger share or its forecast speedup.

**Measurement needed:** full checked W5/W6 kernels versus unchanged Q8_0 on q_b, grouped output_a, output_b, shared gate/up/down and head; m=1,2,3,4,5,6,8,16,128,512,2048, real strides and padded tails. Report dynamic-scale symmetric/affine cases separately; compile ISA to count actual unpack/permutation/epilogue work. Then measure matched C8 prefill and verifier wall time, not only a register loop. Include warmed and cold-cache runs, repeated-trial intervals and component attribution. Reject a route that wins a microbenchmark but loses the serving workload or numeric gate.

### 3.4 A qualification plan for a DS4 5/6-bit dense set

This is a proposed experiment sequence, **not a claim that these gates have passed**.

1. **Freeze references and manifests.** Pin the official 0731 revision, shipped target and support hashes, tokenizer, dataset token arrays, candidate per-tensor types/grouping, quantizer calibration and build revision. Keep experts unchanged for the dense-only experiment. Compare direct official→candidate quantization with shipped-Q8→candidate requantization; double quantization is a confound. Use a streamed/offline official teacher if full resident official inference cannot fit. Do not substitute shipped logits and rename them official quality.
2. **Qualify the representation independently.** Add standalone Python packing/oracle evidence first: SHQ5 signed limits, zeros if affine, exact plane lengths, scales, saturation, tails, misalignment and finite/nonfinite policies. Compare dequantized weights and grouped W*A8 results against the normative oracle. These tests can be tiny and do not need a 284B-model run. Error feedback/imatrix objectives must use representable scales, not an unstorable floating fit.
3. **Introduce only the needed DS4 routes.** Current layout checks reject even a Q6_K head. Add explicit GGUF/layout dispatch and direct C1 GEMVs for the selected three large attention and three shared-expert matrices plus head. Preserve grouped output_a layout, projection pairing/fusion, mHC broadcast/rounding, residual/norm boundaries and required FP32 accumulations. Add small-batch exact kernels for verifier physical widths and masked tails, plus chunked-prefill GEMM/grouped-output routes with a checked activation quantizer. Do not silently expand the whole model to FP16 or fall back through a different numeric path. The existing IQ2_XXS/Q2_K routed kernels can remain unchanged; new expert formats would separately require grouped routed gate/up/down kernels and reductions. Qwen's format support does not supply these DS4 routes automatically.
4. **Run the existing gufo numeric gates.** Against the pinned official reference, satisfy the contract's finite logits and RMSE ≤1.12, cosine ≥0.979, maximum error ≤5, plus the pinned trajectory **≥116/128 top-1**, rank sum **≤142**, worst rank **≤3**. Test complete trajectories as specified, not just the final token or a long random aggregate. Retain per-position ranks/margins and worst examples. Run the 100-case antirez suite with paired NLL/bootstrap reporting (seed 731), not only WikiText. The contract files remain authoritative if a gate is amended explicitly; this review does not redefine one.
5. **Run same-candidate execution consistency.** Ordinary AR versus DSpark verifier must pass the **736-choice** replay suite over its specified C1/C2/C4/C6/C8 cases and depths through 16K, including exact frontier and tie rules. Test padded physical widths and the actual deployment scheduler; arbitrary concurrency/batch invariance is not implied by passing a narrower replay schedule. Also compare prefill and decode continuations. An 8.7% change from the *old* target is a separate model-fidelity metric, not this test's expected failure count.
6. **Extend quality coverage before choosing a tier.** Preserve paired per-token logits/NLL and document IDs for diverse text, code, reasoning, multilingual and long-context retrieval; report mean, tails, top-1, margins and task results versus both references. Resample documents/tasks, not independent vocabulary entries or adjacent tokens. Evaluate Q5, Q6, SHQ5 symmetric/affine and SHQ6 with matched calibration and per-family ablations. Include the head separately. More bits do not guarantee better NLL, as the existing pair demonstrates.
7. **Measure the production outcome and keep a fallback.** Use release binaries from `nix build`; run canonical `nix build .#checks.x86_64-linux.pr` for an eventual implementation change. Report C1/C8 prefill, decode, time-to-first-token, inter-token latency, DSpark acceptance/emitted tokens, peak memory and long-context behavior. Compare equal-quality Pareto points, not only equal nominal bits. Keep the shipped route if any required gate fails or the end-to-end speed/memory tradeoff is unfavorable.

A Q5_K/Q6_K reference route can be a useful intermediate control, but it does not qualify SHQ5/6. Conversely SHQ cannot be rejected solely from a different format's uncalibrated KL. The minimum evidence for a “5–6-bit recommendation” is a candidate that passes these model and execution gates and improves a specified serving workload.

### 3.5 Rotations and iu4 WMMA

**Published evidence.** QuaRot uses function-preserving rotations to support low-bit weights, activations and KV; its reported Llama-2-70B W4A4/4-bit-cache result has ≤0.47 perplexity loss and retains approximately 99% of zero-shot performance. SpinQuant learns rotations because random choices can differ substantially in downstream quality. Neither result qualifies DS4 or Qwen's hybrid recurrent architecture. [QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406).

**Analysis.** Rotations are a plausible route to A4, but each insertion/fusion must preserve this model's residual/mHC structure, nonlinearities, router inputs, rotary attention relationships and DeltaNet state semantics. Q and K transformations must preserve their dot product; arbitrary rotations do not commute with RoPE or recurrence. Rotating already-MXFP4 expert values generally destroys the original small E2M1 alphabet and exponent structure; it does not recover pre-QAT weights. Count extra online transforms and recalibration against any unpack savings.

The measured **INT8 rate equals BF16** does **not** imply INT4 has the same rate. AMD's RDNA3 WMMA description gives a different iu4 operation rate, but it is not a measured gfx1151 ceiling. First benchmark native iu4 with correct packing, signedness, outputs and meaningful instruction chains on this GPU; do not assume 110 TOPS from a different architecture table. [AMD WMMA description](https://gpuopen.com/learn/wmma_on_rdna3/).

Even if realized iu4 throughput is only approximately 55 TOPS, W4A4 could save nibble→byte unpack and halve some activation/LDS traffic. Those gains may be erased by online rotations, activation packing and group epilogues. Single-token decode already reads four-bit weights on W4A8 and has little matrix reuse, so the strongest possible benefit is prefill/batched execution, not a second halving of its weight bytes. **Measurement needed:** dynamic complete W4A4 versus W4A8 and BF16 kernels, transformation overhead, and model gates at equal weight quality. Defer a production rotation project until this kernel-level opportunity is demonstrated.

### 3.6 Published attention-quantization evidence: relevant, not validation of these KLs

[SnapMLA](https://arxiv.org/html/2602.10718v1) studies FP8 MLA attention/cache on DeepSeek-V3.1 and LongCat. Its rotary-component analysis supports keeping RoPE in higher precision; results are task-dependent, not uniformly lossless. For example its DeepSeek table moves MMLU-Pro approximately 84.41→84.43, AIME25 87.92→85.42 and LiveCodeBench 73.46→72.74. This is evidence that attention quantization needs component-specific care, **not** a measurement of V4 Q4_K/Q5_K/Q6_K projection-weight KL.

The [DeepSeek-V3 report](https://arxiv.org/html/2412.19437v2) retains higher precision for selected sensitive operations including attention and gating while employing FP8 elsewhere. That supports mixed-precision caution, not an assertion that every attention projection weight requires a given bit width. The [V4 report](https://arxiv.org/html/2606.19348v1) describes its own compressed/shared-KV attention and FP4-aware choices; V3 MLA results cannot be transferred numerically to V4's different path.

I found **no published matched experiment validating the specific 0.076/0.092/0.049/0.035 KL figures or a universal V4 five-bit threshold**. These local logs are the evidence for those numbers. Published attention results are consistent with sensitivity being heterogeneous and with protecting rotary/cache/operator components; they neither confirm the investigation's causal “absorbed MLA” explanation nor contradict its conditional local KL observations.

The missing decisive experiment is the official-reference, component-ablated, representable-format qualification above. Until then the defensible roadmap is: correct the byte model; retain the shipped expert formats; compare properly specified SHQ5/6 dense candidates; qualify numeric and speculative behavior; and only then spend effort optimizing the winning production routes.
