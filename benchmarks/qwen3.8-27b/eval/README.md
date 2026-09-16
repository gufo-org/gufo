# Qwen27B quality and benchmark reference

Targets: UD-Q4_K_XL and UD-Q8_K_XL. DFlash2 drafts: Q4_K_M, Q8_0 and BF16.
**Adaptive is the default** in prompt, chat, bench and serving. Q4_K_M is the
recommended draft; a full comparison across context depths remains TODO.

## Maintained checks

Run on gfx1151 through Nix. Model-specific tests and tools live in
`tests/models/qwen27b` and `tools/qwen27b`; shared quantization tests remain in
`tests/models/qwen` and are selected by the same runner.

| Suite | Contract |
| --- | --- |
| `fast` | Sampling/verifier, CLI and HTTP parsing, quantization reference decoding and tool reporting. |
| `kernels` | Independent GEMM/decode controls; recurrence and replay; DFlash2 convolution, attention, top-k, selector and verifier distributions. |
| `model` | Target logits/features at widths 2–8 and matrix prefill; MTP committed-feature replay; DFlash2 loading, history and snapshots; executable sampling parity. |
| `serving` | Direct/served tokens, seeded replay, EOS, cache forks, persistent restore, concurrency and cancellation. |
| `reference` | Optional BF16 target comparison: KL, total variation, top-1, RMSE and NLL. This measures quantization differences, not original-checkpoint correctness. |

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py kernels
nix develop -c python3 tools/qwen27b/check.py model \
  --model "$MODEL" --mtp-model "$MTP" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
```

The correctness build retains optimization, symbols and assertions; measure
speed only with `result/bin/gufo`. Artifact variables `GUFO_QWEN27B_*_MODEL`
select test inputs. Load target/reference models sequentially.
Sampling-only checks load each model once; omit the draft for AR:

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" --sampling-only
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" "$DRAFT" --sampling-only
# Add --fixed to check the fixed controller.
```

For an optimization, first check the affected operator against independent
formulas or scalar decode. Then check model replay on each affected target and
draft precision. Compare full logits/features and token IDs, including cached
replay; retain the established tolerances. Run short warmed release timings
with matched artifacts and prompts, alternating binaries during experiments.
Profile separately. Broaden to depth/concurrency sweeps only when needed.

Generation changes must cover **C2/4/6/8 on Q4 and Q8**, preserving C1
performance. The target check reuses eight scalar oracles with different prompts
and prefix lengths. It compares complete logits and all five feature taps,
unequal chunks, combined widths 9–16/28/32/42/48/56/64, rotated coordinators and
accepted-prefix replay. It also appends chunks while the cohort shrinks through
C8/C4/C2/C1; recurrent replay must retain every committed chunk exactly.

Concurrent DFlash2 drafting shares projections across up to 64 rows. Attention,
convolution, selector predecessors, positions, RNG and controller state remain
private. The draft check requires byte-identical layer traces, complete logits,
selector candidates/probabilities and persistent payloads against isolated
execution, including ragged blocks and two rounds of feedback/context injection.
Shared context injection must also match scalar normalization, every K/V
projection and complete private KV snapshots. It reuses scalar scratch; unequal
chunks crossing ring boundaries must remain byte exact.
Both target and draft executables accept `--concurrency-only` for these checks.

Verification generates the original complete proposal before dividing target
work into chunks. Low-acceptance requests use two-row chunks when at least four
requests remain. Smaller groups keep full blocks. Rejected suffixes stop early;
proposal draws, target sampling, emitted tokens, counts and controller feedback
must match full verification. The generic verifier test checks C2/4/6/8,
including a fully accepting peer, penalties, random sampling, EOS, budget tails
and continuation.

Exact projections group adjacent sets of fourteen or sixteen rows in one launch.
Q4/Q5 use six output rows where measured; Q6 has wider FFN/SSM routes and adjacent
gate/up fusion. IQ4 FFN groups use sixteen lanes, keeping both original partial
sums independently before the unchanged reduction. Qualified IQ4 shapes also
use twelve/fourteen-row groups and paired token dots. The quantization test
compares their FP32 bits with scalar decode at widths 12/16/28/42, using
independent token inputs across twelve exponent levels.
Q3_K and IQ4_NL FFNs use wider native groups at fourteen/sixteen positions;
the same test checks complete scalar outputs and independent grouped inputs.
Q8 and BF16 projections reuse
cached weight rows across adjacent sixteen-token groups; every dot product
retains its scalar accumulation order. BF16 draft gate/up projections also use
this route at combined widths 32/48/64; SiLU arithmetic is unchanged.
Shared recurrence preserves
each sequence's computation order. Target-only tail rows join verification, and
existing FFN scratch holds combined logits when it fits.
Memory accounting includes target verification/sampling allocations, draft-owned
weights, draft KV state, scalar scratch and the lazy batch workspace. The draft
test compares allocated bytes with admission estimates before and after batching.

The **2026-09-15** concurrency refresh covers both targets, all three drafts,
all 23 sampling strategies under fixed/adaptive controllers, and serving
cache/fork/persistence/cancellation checks. Eight-token cold/cached and concurrent
runs must reproduce isolated IDs and acceptance counts. Deterministic speculation
must match AR; sampled runs reproduce within each configuration.

Release controls use the repetition prompt and a balanced code/JSON/prose corpus.
Track aggregate delivered throughput, whole-request latency, physical width,
output hashes and acceptance. Corpus throughput uses
`aggregate.output_tokens_per_second.overall`: total delivered tokens divided by
the sum of measured group spans. Per-group medians can misrepresent a mixed
workload. The depth sweep remains TODO.

For prefill changes, the existing target check also covers 128/257/2048-token
prefixes, repeated prefill and two-token scalar/verification replay. It prints
SHA-256 fingerprints of full logits and all five feature taps at every prompt
position, so two builds can be compared without storing logit dumps:

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/qwen27b_target_test \
  "$MODEL" --prefill-only
```

Large Q4 prefill (chunks of at least 1024 tokens) uses fully scaled packed
weights and FP16 activations with FP32 accumulation. Normalization writes FP16
directly; all 64 FFN gate/up pairs in the Q4 artifact share a kernel
that emits SwiGLU without an intermediate gate buffer. The mixed Q6/Q5 pair
uses the canonical fetch-stage decoder to preserve the established dot-product
rounding. Output projections add residuals directly in FP32
after completing the dot product; the SSM epilogue writes FP16. They retain the
separate FP32 producers' rounding boundaries, including halfway cases, and reuse
existing scratch allocations. Complete Q4/Q5 matrix tiles skip
tail checks; most formats write matrix results directly in the output layout.
Large attention K/V projections use the same 256×256 tile for Q4/Q5/Q6/Q8
weights. IQ3_S retains its faster LDS transpose. Q8 targets and short prefill
retain native integer WMMA.

Large IQ4_XS projections and Q5 projections with short reduction dimensions use
wider row groups to improve input reuse. Long Q5 reductions and mixed gate/up
pairs retain the smaller groups. The dot-product and epilogue order is unchanged.
IQ4_XS/IQ4_NL prefill looks up the integer codebook entries directly as exact
FP16 values, preserving FP32 scaling and eliminating signed-byte conversions.
Qualified IQ4 kernels interleave LDS reads with WMMA. Large Q5 projections and
Q4/Q5 gate/up pairs reuse Q5 block headers across four K64 iterations, avoiding
repeated metadata loads. Qualified IQ4_XS projections also reuse headers and
keep weights packed until the LDS commit, reducing live decoded values.
Q3/IQ4_NL mixed pairs retain immediate decoding. These changes preserve
arithmetic and use existing buffer allocations.

The shared quantization test checks all eight Q4 artifact formats against
independently decoded weights and FP64 dot products, including complete Q4/Q5
tiles, partial row/token tiles, small/medium IQ4 projections and 1024-row K/V
projections at width 1025.
It requires at least a 2× RMSE improvement over A8 and lower maximum error.
Paired gate/up, SwiGLU, normalization and in-place residual outputs must be
byte-identical to the separate FP32 producers and FP16 conversions. Mixed pairs
cover independent weight strides and row/token tails. A full normalization chunk
checks rare rounding ties; explicit FMAs preserve the first square's rounding
when the fixed-width loop unrolls. The existing test executable takes about
four seconds. The SSM test checks exact FP16 output and
unchanged recurrent state with FP32/BF16 storage. It also checks causal convolution
against an independent FP64 formula and exact final history, including nonzero
history and batches of 1, 2, 3 and 7 tokens. History advances in the next existing
kernel after convolution finishes reading it; no extra allocation, launch or test
executable is needed. The SSM test takes about 1.2 seconds.

Model precision qualification: two real 2048-token prefixes from
`docs/PERFORMANCE.md` and `src/models/qwen/hip/batched_decode.cpp`, with 32
full-vocabulary rows per prefix, sampled every 64 positions. The reference
streams each projection's independently decoded Q4 weights through FP32 GEMM.
Every row improves RMSE and KL over A8; aggregate RMSE/KL/total variation must
at least halve, and greedy agreement must not decrease. All 64 greedy choices
match the reference. Mean logit RMSE is **0.00179 / 0.00358** for the two texts,
16–31× lower than A8; all five feature taps also improve. This qualifies
execution of these quantized weights, not conversion or the original checkpoint.
The current kernels preserve all 64 captured logits and all ten complete
feature-tap tensors across both prefixes exactly relative to the qualified FP16
calculation. Repeat-prefill
and scalar/verification replay pass; Q8 fingerprints remain unchanged.

Existing short-prefix decode/draft qualification covers 102 target logit rows,
102 tapped feature rows, 96 C3 replay/cache rows, and logical context 262,144
with a short prefix. All 270 draft trace files across the three precisions
remain byte-exact. Trace mode exits before the state suite: loading, history,
snapshot and serving checks must also run when those paths change.

## Sampling and executable contract

Qwen AR and DFlash2 sample on the GPU. The CPU owns request history and RNG
state. Target processing is penalties → top-k → top-p → min-p → temperature;
`min-keep` is a candidate floor. Temperature zero uses the adjusted argmax.

Batched unadjusted argmax reduces vocabulary tiles in parallel, then selects the
winning token. It retains complete logits and reuses idle FFN scratch. The
maintained GPU sampling test covers tile boundaries, realistic vocabulary sizes,
C1/2/4/6/8, lowest-ID ties, nonfinite filtering and scratch bounds.

DFlash2 uses an anchor plus up to seven proposals. The trained top-16 selector
shares target temperature and reports its actual proposal distribution `q`.
There are no independent draft sampling controls. The verifier accepts token
`y` with probability `min(1, p(y)/q(y))`; rejection samples normalized
`max(p-q, 0)` over the full vocabulary. Only the consumed prefix is committed.
Greedy output must equal AR. Sampled AR/speculation consume different draws,
so equal seeds need not produce identical continuations across the two modes.
Repeated runs within one configuration must reproduce IDs, including cache hits.

Adaptive chooses the block length before drawing proposals using accepted-length
history and offline verification costs. It never uses live timing or the current
sample. Controller state persists within a request and resets for a new one.
`--draft-tokens` caps length; `--draft-policy fixed` selects the comparison policy.

| Entry point | Contract |
| --- | --- |
| `prompt`, `chat` | Shared target sampling and DFlash2; fixed/adaptive and draft cap. |
| `bench` | Qwen AR/MTP/DFlash2: greedy C1. DS4 additionally supports sampled benchmarks. |
| `serve`, `serve llm`, `gufo-server` | Startup draft/controller defaults; request target-sampling overrides through six HTTP adapters. |
| `eval` | Uses server draft configuration and sampling defaults. |
| Audio/video, diagnostics and probes | Do not run Qwen27B DFlash2; unsupported draft flags are rejected. |

Sampling/state refresh: **2026-09-15**. All
[23 named strategies](../../../tests/models/qwen27b/sampling_cases.hpp) pass on
Q4/Q8 AR and all 12 target/draft/controller combinations: Q4/Q8 targets,
Q4_K_M/Q8_0/BF16 drafts, fixed/adaptive. They cover temperatures, individual
filters, candidate floors, penalties/rewards, history windows and combinations.
Each case checks eight tokens and cold/cached replay. All six target/draft state
suites also pass seven-proposal controller, RNG and snapshot/reuse checks.

GPU checks cover 216 configurations / 864 AR quantiles, 141 acceptance/residual
controls and four full-vocabulary cases. The largest 24-bit RNG draw remains
exactly representable below one and accepts `p(y) = q(y)` on both GPU routes.
CPU reference replay covers 1,472 steps. No sampling implementation change was
needed.

HTTP: 15 configurations across six adapters on both targets, with AR and Q4_K_M
drafts under fixed/adaptive. Checks include startup defaults, request overrides,
seeded/unseeded execution, streaming and C2 state isolation. Prompt/two-turn chat
replay covers the same 15 configurations on Q4 AR and Q8/BF16 adaptive DFlash2.
This finite matrix does not establish arbitrary-context capability. Full
context/concurrency sweeps across all draft precisions remain TODO.

## Pinned original DFlash2 operators

Reference: `z-lab/dflash`, commit
`07ebd93db9f472af339b644bb70221ad8428328a`, unmodified `dflash/model.py`.
Source SHA-256: `f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e`.
Config SHA-256: `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`.

The runner uses PyTorch FP32 operations over independently decoded copies of the
same GGUF weights, including documented BF16 packing. Checks cover feature taps
5/19/33/47/61 after FFN residuals, anchor/mask embeddings, injection/norm/RoPE,
windowed noncausal attention, causal dynamic convolution and selector transition
scores. Each layer is checked both cumulatively and with captured layer inputs;
the head and conditional selector probabilities are also isolated.

One real 24-token Q4 target prefix, seven proposals, temperature 0.8: all three
drafts pass; all 21 candidate sets and random draws match.

| Draft | Worst stage relative RMSE | Full-logit max error | Proposal max total variation |
| --- | ---: | ---: | ---: |
| Q4_K_M | 5.83e-6 | 7.72e-5 | 1.60e-5 |
| Q8_0 | 5.74e-6 | 9.54e-5 | 1.25e-5 |
| BF16 | 6.16e-6 | 1.13e-4 | 7.01e-6 |

Gates: stage relative RMSE ≤1e-4; full-logit maximum error ≤1e-3; proposal total
variation ≤1e-4; isolated selector probability error ≤5e-6.
This validates packed-weight execution, not GGUF conversion or the original
full target checkpoint. Native BF16 execution need not be bit-identical.

```sh
gh api 'repos/z-lab/dflash/contents/dflash/model.py?ref=07ebd93db9f472af339b644bb70221ad8428328a' \
  -H 'Accept: application/vnd.github.raw' > /tmp/dflash-model.py
nix develop -c cmake --build --preset gpu-test --target qwen_dflash_gpu_test
nix develop -c build/gpu-test/tests/models/qwen27b/qwen_dflash_gpu_test \
  "$MODEL" "$DRAFT" --trace /tmp/dflash-trace
nix develop -c python3 tools/qwen27b/dflash_reference.py \
  --upstream /tmp/dflash-model.py --config "$ORIGINAL_DFLASH_CONFIG" \
  --target "$MODEL" --draft "$DRAFT" --trace /tmp/dflash-trace \
  --output /tmp/dflash-reference.json
```

Use a fresh trace directory for each draft. Keep generated logits, traces and
experiment JSON outside the repository. Committed logit fixtures should come
only from an independent full-quality PyTorch checkpoint, with provenance;
no such Qwen target fixture is currently bundled. Prompt corpus JSON files are
maintained test inputs. Historical experiment reports remain in Git history.

## Latest measurement provenance

The headline tables retain the latest qualified full matrix, with C1/C4
repetition refreshed below. The full matrix was measured on
**2026-09-16**, release SHA-256:
`5d809239f49e815bbc1111b7c2055f9a3d55da0b5a69b0beb23b1676e6b4ba30`.
It supplies concurrent Q4_K_M DFlash2, Q4_K_M C1 and C8 AR controls for both
targets. Q8_0/BF16 C1 draft controls use
`e1a1e5161fc07c1cb9c49cb1de9d56f2c94adb757017f3d0f033293ca159131c`;
AR C1/2/4/6 controls use
`26fd73c8389e7b371baf1bc6347ab9d18e78500c6a25f2dbf61f6be952a04e14`.
The full matrix uses one warmed pass, except Q8 C6 repetition (three).

The latest retained kernel release is
`570180f3ca44aa17ff1d5b6014d69e90ee9e04c9df4d797f7d6f61367df9f31d`.
It keeps the established verification chunk policy. Tiled argmax retains all
logits, finite filtering and lowest-ID ties without allocating another buffer.
BF16 draft gate/up reuses weights across sixteen-token groups without changing
SiLU arithmetic. Isolated actual-weight controls measured about 3.3% lower
head-plus-argmax latency at sixteen rows and 23% lower BF16 gate/up latency at
32 rows. These are operator results, not whole-model speedups.

On this release, Q4_K_M draft repetition measures **Q4 C1 62.08 / C4 93.62**
and **Q8 C1 48.85 / C4 89.60 aggregate tok/s**. Each cell uses three warmups
and three measured rounds, greedy tg64, adaptive, context capacity 4096.
Servers run sequentially. Completion hashes, output token counts, proposal
counts and acceptance counts match the control release for every request.
These measurements replace the corresponding headline cells. The remaining
performance refresh is **TODO**; runs affected by host activity are excluded.

The kernel refresh passes complete Q4/Q8 target concurrency checks, all six
target/draft concurrency checks, and the 23-strategy Q4/Q8 AR and Q4_K_M DFlash2
sampling checks under fixed/adaptive. Cold/cached and concurrent sampling retain
isolated IDs and acceptance counts. The restored chunk policy passes the
expanded C2/4/6/8 verifier fixture, including RNG, frontier logits, feedback,
continuation and fully accepting peers. No tolerance was relaxed.

DFlash2 measurements use adaptive, greedy tg64, context capacity 4096 and cached
prompts. The mixed workload repeats each maintained corpus prompt eight times,
giving 24 requests divisible by C1/2/4/6/8. Warm every unique prompt before
measurement: three groups at C1, two at C2, one at C4/6/8. Every measured request
must report a cache hit and zero prefill tokens. Repetition accepts every
proposal; mixed acceptance is 61.21% for Q4 and 49.13% for Q8. Keep individual
latency, physical width, output hashes and draft counts. Concurrent requests
must reproduce isolated output and acceptance counts. AR and the C1 draft
comparison use cached `prose_tides`, tg64 and context capacity 4096.
Fixed remains a comparison policy; adaptive is the default. Controller
comparisons and context-depth sweeps remain TODO.

Q4 release measured on 2026-09-14, SHA-256:
`0f6804912d5f9e97df7ffc1e5c49f5898b7444481006725d4c4b274fa2b0307a`.
The pp2048-only result, **610.56 tok/s**, averages 12 warmed samples in four
processes: **613.28 / 610.02 / 608.92 / 610.03 tok/s**, three repetitions each.
Use `gufo bench -p 2048 -n 0 -d 0 -c 1 -r 3 --verbose`; alternate release
binaries after sustained warmup when comparing implementations. Early transient
boosts are excluded from the headline. Q8's recorded 503.17 tok/s is from
the unchanged native wave64 release, SHA-256
`feec38298e8a84a8b9f5dfcf290b924aed597e9f2bd8643fa40b5bd154aa736f`.

Prompts live in [the adaptive corpus](../speculative-adaptive-corpus.json).
The repetition prompt is: "Output the word red exactly 1000 times, separated by
spaces. Do not add any other text." Keep its 100% acceptance control separate
from the balanced mixed workload.

The measured pp2048 chunk spends about **90%** of GPU time in FP16 quantized
GEMM, with **0.07%** dispatch idle time, excluding warmups and resets.
All 82 FP16 prefill kernels have zero
scratch spills; unused small SwiGLU instantiations are excluded from the build.
The profiler preserves anonymous-namespace and quantization names, so it reports
each kernel independently.

Retained: native wave64 for Q8/short prefill; packed-weight FP16 prefill with
fixed-width norm, matching/mixed gate/up fusion, in-place residuals and SSM output;
direct matrix stores, complete Q4/Q5 tiles, larger K/V tiles and wider row groups
for qualified projections; exact IQ4 half lookup, IQ4 instruction scheduling
and Q5/IQ4 header reuse with deferred IQ4 decoding.
Rejected: weight/activation repacking, alternative tile sizes, FP16 wave64,
two-stage LDS buffering, weight copies and alternative normalization reductions.
Dense BLAS, split-K for small projections, and convolution/KQ fusion did not
improve model prefill. Rejected experiments add no production paths.

Cached Q4 repetition/tg64 profiles are about 97% GPU-busy at C2/C4.
Quantized projection kernels occupy about 88% of measured GPU time,
excluding the warmup cohort.
A common Q5 projection takes about 0.30/0.58 ms at those widths,
and a paired projection with SwiGLU takes about 0.84/1.59 ms.
C1 already verifies seven or eight positions per block. C4 generally uses two
sixteen-position groups, repeating the dot products and weight decoding.
GPU-busy time alone does not establish a hardware throughput ceiling.
Retained: wider exact projection groups, native wave64 for measured shapes,
shared recurrence launches, batched drafting/context injection and target-only
tails, logit scratch reuse, fourteen/sixteen-row grouping, narrower IQ4 lane
groups, Q6 gate/up fusion, cached Q8/BF16 weight reuse and early termination of
rejected suffixes; qualified instruction scheduling for Q4/Q5 groups of sixteen
and distributed exact output reductions, fused matching gate/up activations and
qualified Q8 sixteen-position scheduling and native Q3_K/IQ4_NL FFN groups;
BF16 draft gate/up weight reuse and tiled argmax.
Rejected: extra Q8 staging, coefficient preconversion, paired-lane Q8 dots, wider
scalar tiles, direct global activation loads, compact fourteen-row staging,
predecoded weight caches, lossless half-coefficient FMAs and explicit paired-FP32
schedules. Mixed-format exact gate/up fusion improved its microbenchmark but
not serving throughput; alternative BF16 sixteen-position tiles did not help.
Lossless coefficient packing slowed the main microbenchmarks; DPP output
reductions did not improve serving. GPU target-feature handoff and combined
draft gate/up projections did not improve serving. Fused top-16 projection
selection was slower; fused argmax lost to the simpler tiled reduction.
A cost-based chunk planner did not qualify consistently across targets and was
removed; the established private-acceptance chunk policy remains.
No runtime switches select these experiments.

| Artifact | SHA-256 |
| --- | --- |
| Target UD-Q4_K_XL | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Target UD-Q8_K_XL | `af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377` |
| Draft Q4_K_M | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| Draft Q8_0 | `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |
| Draft BF16 | `26d47ca20ab07688327a63d912acad222d924eaaa92a980cc488de3c67e736bc` |

Current optimization focus: **Q4 and Q8 generation at C2, C4, C6 and C8,
with and without DFlash2**, preserving sampling correctness and **C1 performance**.

TODO: independent original-target/conversion and MTP qualification; optional
BF16 target comparison; refreshed pp2048/tg128 depths at all tracked widths
and all draft precisions. The 75-question capability comparison
requires confirmation before running.
