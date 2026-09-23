# Qwen3.8 27B evaluation

Targets **Q4_K_XL / Q8_K_XL**, DFlash2 draft **Q4_K_M**. Independent
original-target, GGUF conversion and native MTP parity remain **TODO**.
Agreement between Gufo paths is execution consistency, not proof of upstream
model accuracy. [Artifact identities](artifacts/model-identities.json).

## Current qualification

| Area | Retained evidence |
| --- | --- |
| Target execution | Full logits and all five feature taps match scalar execution at verification widths 2–8, including ragged 17–36-row cohorts, mixed FP16/FP32 KV, shrinking cohorts, snapshots and 8K + 1025-token continuation. Both target quants pass; grouped Q8 projections retain scalar FP32 bits and guarded output bounds. |
| RMSNorm | 90 maintained FP64 controls at absolute tolerance 1e-5, dimensions 5119/5120/5121, rows 1/7/33/64/65, three scales and optional weights; scalar/batched output is byte-identical. Explicit fused square accumulation prevents compiler contraction drift. |
| Target attention | Independent FP64 checks, FP16/FP32 KV and dispatch boundaries pass. Shared partition scales retain 128 byte-exact comparisons through 64K; the split-K threshold has maximum absolute error 1.17e-6 against FP64. Prefill causal tails retain byte-exact chunk/packing controls; maximum absolute error against original-input FP64 attention is 2.11e-4. |
| Draft attention / selector | 216 byte-exact attention controls include ring wrap and ragged blocks. Real-weight layer traces, complete logits, conditional probabilities, private RNG and persistent replay pass through C8. |
| Sampling | Maintained GPU sampler covers 216 AR policies, p/q acceptance and residual sampling, ties, nonfinite rows and tile boundaries. Seeded cold/cache replay retains IDs and proposal counts. |
| Weight placement | Complete 17,559,178,144-byte Q4 encoded-weight copy matches; read-only huge pages retain target arithmetic. Q8 full-target qualification also passes with this layout. |
| Serving | Q4/Q8 AR and DFlash2 pass live continuation, disk restoration, C4 generated-history forks, sampled/greedy transitions and cancellation inside a verified block. Q8 also retains all eight d32K generated-history forks. |

Current controls and binary/source identities:
[Q4 AR](artifacts/q4-ar-c1-pruning.json),
[Q4 DFlash2](artifacts/q4-dflash2-c1-focused.json),
[Q8 decoding and continuation](artifacts/q8-tg-focused.json).
No equality gate or tolerance was relaxed. Full-logit captures stay outside Git.

Generated history must retain **decode arithmetic** when a conversation resumes.
Qwen prefill and decode use different projection arithmetic: putting an unconsumed
output token into a new prefill chunk can change recurrent state. The server now
records published pending IDs separately from the greedy frontier and replays
those IDs with decode before prefilling the new suffix.

Cancellation can arrive while a verified block is being published. The runner
reuses its existing verification rollback, retains the completed frontier and
replays at most the latest block (nine tokens; one for AR). This adds no GPU copy
to normal decoding. The focused test covers first-token cancellation, longer
output, sampled replay and fresh-backend disk restore on both target quants.
A single rollback checkpoint does not cover a client lagging several complete
blocks. Disk payload/layout version 3 includes pending IDs and rejects stale
arithmetic/policy identities.

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" "$DRAFT" --continuation-only
```

The Q8 C1 HTTP controls retain all 128 AR tokens with DFlash2 at d0 and d32K,
including the same generated prefix history. The full Q8 target test and shared
runner/scheduler tests pass. The compact Q8 artifact records request hashes,
counts and validation log hashes.

Concurrent continuations freeze the live generated frontier before its first
branch mutates it. Peers restore that checkpoint instead of prefilling the
generated reply; the original prompt checkpoint remains available for branching.
This is budgeted and applies only to pools with multiple sessions. C1 retains
its copy-free live path. The 32K Q8 HTTP control reuses 32,764 tokens on all four
DFlash2 requests, and every 128-token continuation matches isolated AR.

Scheduled prefill also needs consistent attention at chunk boundaries. A
masked populated future value could change WMMA rounding compared with absent
padding. Each query's final partial tile now accumulates visible keys with FP32
FMA; complete tiles keep WMMA. At 8,193 + 2,059 tokens, Q8 full logits and all
five feature taps match whole-chunk execution exactly with 512/2,048-row
scheduling budgets. Independent FP64 attention and packed/unpacked operator
checks pass, including shallow packed chunks of 1,025/2,048 rows. Q4's distinct
large-prefill projection precision is outside that whole-model chunk equality
claim. The persistent arithmetic identity rejects
snapshots made before the attention fix.

## Adaptive decoding and concurrency

The controller chooses a block before sampling from private accepted-length
history and deterministic position. Full acceptance is censored: a saturated
history probes wider blocks. An independent geometric-distribution check verifies
zero expected feedback drift at every width 1–7 below saturation. Request timings
never enter sampled decisions; controller history resets for a new request.

Q4 greedy C2/C4/C6/C8 costs use complete measured cycles and context growth.
Private sampled/mixed cohorts match isolated proposal IDs, probabilities, RNG
and restored state. Focused shallow and d32K cohorts retain C1 AR output; perfect
acceptance is preserved. Evidence: [C2](artifacts/q4-c2-focused.json),
[C4](artifacts/q4-c4-focused.json), [C6](artifacts/q4-c6-focused.json),
[C8](artifacts/q4-c8-focused.json). The difficult Italian/Chinese C8 pair remains
faster under AR; these controls do not establish universal speculative profitability.

Q8 greedy cohorts of two through eight requests use one shared width chosen from
private acceptance histories and measured cycle costs. Odd cohorts use the next
measured capacity's estimate. Full-acceptance probes also cover growing cohorts
when two established histories support the probe and the remaining estimates
are neutral or fully accepted. Fixed, sampled and mixed cohorts retain private
choices; C1 keeps its existing policy. Real-weight checks cover layers, logits,
selector probabilities, RNG and persistent state at every cohort size C2–C8.

Current Q8_K_XL C8 controls reach **65.96 tok/s** on the nine-case mixed corpus
and **116.90 tok/s** on repetition, as sums of individual decode rates.
All 16 mixed completions and all eight repetitive completions match isolated
AR; repetition accepts all 880 proposed tokens. Matched llama.cpp controls reach
**63.02 / 86.53 tok/s** for mixed/repetition. Ragged projections group weight reads without
changing per-row arithmetic, and shared widths avoid excessive proposals when
the active cohort shrinks.

At d32K, C8 reaches **41.52 tok/s**: all eight requests reuse 32,764 tokens,
prefill 2,060 new tokens and match the isolated 128-token AR continuation.
The C1 kernel control retains its output and 292/78 proposal/acceptance counts
at **463.44 pp / 16.12 tg tok/s**.

C6's Italian/Chinese control varies with admission order: the retained candidate
measured **49.25–52.84 tok/s**, while unchanged-baseline controls ranged
**44.67–51.86**. The paired repeat was **52.84 versus 51.85**, with every
completion matching AR. This does not establish a uniform C6 speed gain.
Profiles, ablations, earlier C2/C4 controls and source identities are retained
in the [compact Q8 artifact](artifacts/q8-tg-focused.json); the full concurrency
and depth matrix is still pending.

## Meaning of Exact

`Exact` in the benchmark tables compares llama.cpp AR text hashes with Gufo C1
AR. It is **cross-engine agreement**, not an accuracy percentage. In the retained
September 21 corpus, each engine's C1 DFlash2 matches its own AR on all nine cases.
Cross-engine C1 agreement is 3/9 for Q4 and 4/9 for the historical Q8_K_L file.
Compared with its own C1, llama.cpp AR agrees on 7/10, 7/12, 8/12 and 10/16 Q4
requests at C2/C4/C6/C8; historical Q8_K_L agrees on 7/10, 3/12, 4/12 and 6/16.
The [hash audit](artifacts/q4-dflash2-c1-focused.json) records source identities.

Fresh pinned llama.cpp `68d9053a` controls also show AR/DFlash2 differences on a
Q4 prose prompt and on the Q8_K_XL d32K continuation. Each comparison uses the
same messages, model files and greedy settings within that engine; Gufo retains
AR/DFlash2 agreement. The Q8 d0 outputs match across all four modes. These
observations do not determine which arithmetic matches the original checkpoint.
Original-target qualification is still needed to resolve that question.

On the current Q8_K_XL nine-case corpus, cross-engine C1 AR agreement is 7/9.
At C8, llama.cpp preserves 10/16 of its own C1 AR completions with AR and 6/16
with DFlash2; Gufo preserves 16/16 with both. These are execution-consistency
checks, not evidence that either engine matches the unquantized checkpoint.

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
supported draft. Compare full logits/features and token IDs, including cached
replay; retain the established tolerances. Run short warmed release timings
with matched artifacts and prompts, alternating binaries during experiments.
Profile separately. Broaden to depth/concurrency sweeps only when needed.

Generation changes must cover **C2/4/6/8 on Q4 and Q8**, preserving C1
performance. The target check reuses eight scalar oracles with different prompts
and prefix lengths. It compares complete logits and all five feature taps,
unequal chunks, combined widths 9–16/28/32/42/48/56/64, rotated coordinators and
accepted-prefix replay. It also appends chunks while the cohort shrinks through
C8/C4/C2/C1; recurrent replay must retain every committed chunk exactly.

Batched target/draft checks compare full logits, all feature taps, layer traces,
selector probabilities, private KV/convolution state, RNG and controller feedback
against isolated execution. Cover ragged widths through C8, shrinking cohorts,
rejected suffixes, continuation, memory accounting and ring wrap. Check only the
affected operator first, then broaden to model/state qualification.

## Sampling and executable contract

Qwen AR and DFlash2 sample on the GPU. The CPU owns request history and RNG
state. Target processing is penalties → temperature → top-k → top-p → min-p;
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
history and offline verification costs, including context-dependent attention
cost. It never uses live timing or the current sample. Controller state persists
within a request and resets for a new one.
`--draft-tokens` caps length; `--draft-policy fixed` selects the comparison policy.

| Entry point | Contract |
| --- | --- |
| `prompt`, `chat` | Shared target sampling and DFlash2; fixed/adaptive and draft cap. |
| `bench` | Qwen AR/MTP/DFlash2: greedy C1. DS4 additionally supports sampled benchmarks. |
| `serve`, `serve llm`, `gufo-server` | Startup draft/controller defaults; request target-sampling overrides through six HTTP adapters. |
| `eval` | Uses server draft configuration and sampling defaults. |
| Audio/video, diagnostics and probes | Do not run Qwen27B DFlash2; unsupported draft flags are rejected. |

The retained matrix covers 23 named strategies on Q4/Q8 AR and Q4 DFlash2 under
fixed/adaptive controllers; 15 executable/HTTP configurations exercise six
adapters, request overrides, cold/cached replay, streaming, multi-turn EOS,
C2 isolation and cancellation. See
[sampling cases](../../../tests/models/qwen27b/sampling_cases.hpp).
Current target random draws retain 53 bits. GPU/CPU distribution comparisons
have numerical tolerances; this is not bit-identical upstream sampler parity.

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

One real 24-token Q4 target prefix, seven proposals, temperature 0.8:
all seven candidate sets and random draws match with the Q4_K_M draft.
Worst stage relative RMSE is **5.83e-6**, full-logit maximum error **7.72e-5**,
and proposal maximum total variation **1.60e-5**.

Gates: stage relative RMSE ≤1e-4; full-logit maximum error ≤1e-3; proposal total
variation ≤1e-4; isolated selector probability error ≤5e-6.
This validates packed-weight execution, not GGUF conversion or the original
full target checkpoint. Native BF16 execution need not be bit-identical.

```sh
nix develop -c python3 tools/qwen27b/dflash_reference.py --help
```

The reference runner verifies the pinned source/config and emits stage metrics;
keep generated tensors and full-logit captures outside Git. Do not add logits
from another quantized implementation as ground truth.

## Vision

Shared operators live in `src/models/qwen/vision`; each language runtime owns
its embedding, attention, speculative and snapshot integration.

Qualified: Q4/Q8 AR and DFlash2, Flash-Next AR/MTP, native 27B MTP CLI,
cold and sampled C4, HTTP C2/streaming, multiple images, and RAM/disk replay.
Controls also cover spatial shape recognition, a 2,304-token image crossing
prefill chunks, and Flash-Next image decoding/cache replay at 10,495 tokens.
Image throughput sweeps remain **TODO**.

Q8_K_XL uses FP32 activations and a fixed per-row reduction for BF16
projections. The image suite requires identical logits across prefill chunk
boundaries and identical greedy continuation text for cold, live and disk-restored
sessions, with AR and DFlash2. It also requires reuse of generated history.
`--disk-only` is a focused persistence check; qualification uses the full suite.

State qualification also checks interruption of greedy and sampled image
requests, immediate cached retry, replacement by another image/text request,
and seeded replay after a fresh backend restores disk state. GPU snapshots
contain only valid KV positions and recurrent layers; serialized snapshots
retain the same state plus their header and image layout. Rollback checks
require exact state bytes, including ragged recurrent-layer groups.

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen27b_vision_test qwen_vision_serving_test
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images \
  --probe build/gpu-test/tests/models/qwen27b/qwen27b_vision_test
build/gpu-test/tests/models/qwen27b/qwen_vision_serving_test \
  "$MODEL" "$DRAFT" /tmp/qwen-images
# Use "-" for AR only. Use --disk-only for a focused persistence check.
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images --http-url http://127.0.0.1:8080 \
  --model vision-test
nix develop -c python3 tools/qwen27b/vision_check.py \
  --directory /tmp/qwen-images --binary result/bin/gufo \
  --target "$MODEL" --draft "$DRAFT" --speculative dflash2
# For Flash-Next or native 27B MTP, use --speculative mtp.
```

Use a fresh server for the HTTP check. The native check covers cold C4,
greedy AR/speculative token IDs, proposal accounting, seeded sampled C4,
multiple images, exact prefix replay and disk restoration after restart.
The CPU check compares pixels against Pillow/Torchvision, including resizing,
EXIF orientation, palette, alpha, grayscale, gamma metadata and CMYK JPEG.
Run parser/cache/codec tests under the `cpu-sanitizer` preset when changing
input handling or persistence.

Flash-Next's maintained continuation fixture requires **identical full logits**
for bulk prefill, incremental prefill and restored cache state. Its router and
recurrent-gate projections keep one accumulation order when rows move between
chunks. Prompt projections and normalization retain their arithmetic even for
one-token tails. Prefill retains F16 SSM output activations, and attention
accumulates each query's final partial tile using only visible keys. Tests cover
unaligned boundaries, full-model continuations through 4096 tokens, and FP64 operator
controls.
This does not establish bit-identical output between prefill and token-at-a-time
kernels, which use different arithmetic routes.

For encoder arithmetic, compare the complete graph and isolated layers with
the original Transformers operators using the **same BF16 GGUF weights**:

```sh
build/gpu-test/tests/models/qwen27b/qwen27b_vision_test \
  "$MMPROJ" /tmp/qwen-images/shapes.png /tmp/qwen-trace 5120
# Flash-Next's output width is 2560.
nix develop -c python3 tools/qwen27b/vision_reference.py \
  --source "$TRANSFORMERS_CHECKOUT" --mmproj "$MMPROJ" \
  --image /tmp/qwen-images/shapes.png --trace /tmp/qwen-trace \
  --output /tmp/qwen-vision-reference.json
```

Transformers pin: `3713bd839e580d07e4b70f2c89e986cb3c0e8ddf`.
The tool verifies source hashes, exact pixels, every vision layer, isolated
layers 0/8/26, and output error against a full FP32 control. Both projectors
pass these checks on the focused fixtures. This qualifies operators and
bindings; it does not independently qualify GGUF conversion or the quantized
language model. Keep traces outside the repository.
When changing vision arithmetic or preprocessing, bump the image execution
version in `src/models/qwen/vision/prompt.cpp` to invalidate persisted prefixes.

Audited formulas include patch/merge order, learned position interpolation,
axial vision RoPE, normalization and GELU variants, language mRoPE, Flash
indexer positions, and shifted MTP inputs. Secondary implementation controls:
llama.cpp `18a04f09c24616898792bcfaa17f3550bdc78912` and
vLLM `63d9ad0a3a435cdf3a44495028b10f390a38f960`.
