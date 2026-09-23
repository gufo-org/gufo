# Qwen3.8 27B evaluation

Targets Q4/Q8; DFlash2 drafts Q4_K_M/Q8_0/BF16. Independent original-target,
conversion and native MTP parity remain **TODO**. Packed-weight operator
agreement is narrower evidence. [Artifact identities](artifacts/model-identities.json).

The current scalar and small-batch RMSNorm cache the 5120 input values and
weights in registers, retaining the descending reduction tree. An expanded
check caught a contraction difference in the earlier cached scalar path:
unrolling did not always retain the generic loop's fused square accumulation.
The cached paths now request that FMA explicitly and keep the runtime
normalization denominator. The earlier eight byte-identical controls were
insufficient to establish equivalence.

All 90 maintained batched controls pass the independent FP64 formula at
1e-5 absolute tolerance and require byte-identical scalar/batched results:
dimensions 5119/5120/5121, rows 1/7/33/64/65, three input scales and optional
weights. BF16 round-to-nearest-even and BF16-only output also pass. The
separate generic-versus-cached ablation has no differing values across
512 rows; the batched optimization preserves the original kernel in
112 FP32/BF16 controls.

C1 reuses the tiled argmax and idle FFN scratch. The maintained GPU sampler
check passes its 216 AR policies, speculative p/q/residual checks, ties,
nonfinite rows and tile boundaries. Snapshot sizing uses the vocabulary size
without downloading logits. Matched d0/d32K HTTP controls preserve all 128
greedy tokens and cached-prefix reuse; the repeated d0 output also matches.
Snapshot payload formats are unchanged. Cache compatibility includes the
corrected normalization arithmetic, excluding snapshots from older builds.

In a separate 16-token, depth-2048 profile, normalization falls from 23.99 to
6.54 ms and argmax from 2.53 to 0.12 ms; total kernel time falls from 1324.26 to
1304.37 ms. This explains approximately 1.24 ms saved per token.
[Measurements and scope](artifacts/q4-ar-c1-pruning.json).
Q4 DFlash2 qualification is included below; Q8 remains **TODO**.

The latest loader keeps the encoded weights unchanged in a read-only registered
anonymous mapping, with transparent huge pages. Sixteen bounded workers populate
and copy file chunks, then release their original mapping pages. A complete
17,559,178,144-byte copy check passes with both 4 and 16 workers. Target arithmetic,
sampling and snapshot formats are unchanged.

The Q4 d0/d32K controls retain all 128 greedy tokens; the deep request reuses
32,553 tokens and prefills 2,010. The maintained continuation smoke test passes
cold, exact-prompt and suffix reuse. Its visible-answer fixture now explicitly
disables thinking, matching the full cache suite and preserving its assertions.
The 16-token profile has the same 10,863 dispatches and 1,280.71 ms of kernel
time, saving approximately 1.48 ms/token over the preceding kernel cleanup.
Process RSS remains about 16.70 GiB. Warm readiness is 0.82–0.97 s; the matched
original mapping took 0.72 s. Anonymous weights replace the resident file mapping,
but the OS can retain an additional reclaimable file cache. This is a decode
speed/startup tradeoff, with no new quantization or original-checkpoint parity claim.
Q4 DFlash2 now passes the same d0/d32K controls with this memory layout.
Its warm readiness is 2.23 s with 17.80 GiB process RSS. Q8 qualification
remains **TODO**.

Attention reduction shares each partition exponential across the denominator
and both output vectors, retaining each sum's order. All 128 direct comparisons
are byte-identical through 64K, and the maintained independent FP64
FP16/FP32 attention checks pass through 32K.

C1 DFlash2 reuses the batched selector and finished FFN scratch, removing
15,616 bytes of separate partial buffers for the supported draft. Requested
greedy candidate outputs retain one-hot probabilities. Operator checks cover
C1/C2/C4/C6/C8, and the real-weight draft test retains exact layers, logits,
probabilities, private RNG, pending injection and RAM/persistent replay.
Allocation accounting matches the estimator.

The full target test exposed a second issue: when verification replay records
are missing, the fallback used prefill arithmetic to rebuild recurrent state.
The failure also reproduces with the original generic normalization kernels.
The fallback now reuses the recorded prefix and decodes only unrecorded rows.
The complete target check passes full-logit and feature equality across
verification widths 2–8, mixed-context FP16/FP32 KV, C1/C2/C4/C6/C8,
prefill/snapshot replay and an 8K prefix with a 1025-token continuation.
No equality gate or tolerance was relaxed.

The current release retains all 128 greedy tokens, acceptance counts and PP
in matched d0/d32K AR/DFlash2 controls. Separate 32-token HTTP requests at
temperature 0.8 and seed 47 combine top-k/top-p/min-p and repetition,
frequency and presence penalties. Each mode reproduces its output after
an exact 2033-token cache hit with zero prefill; DFlash2 also reproduces
all proposal and acceptance counts.

A separate four-cycle fixed-three-proposal profile attributes the speed gain
to normalization (6.72 to 1.99 ms) and attention reduction (0.99 to 0.59 ms).
Total kernel time falls from 415.77 to 411.06 ms; selector consolidation
removes 14 launches but is roughly speed-neutral. This profile explains the
mechanism; published throughput uses the adaptive HTTP controls.
[Current measurements and checks](artifacts/q4-dflash2-c1-focused.json).
These checks establish quantized execution consistency, not independent
original-checkpoint parity.

The 128-token split-K threshold is currently qualified on **Q4_K_XL C1 with
AR and Q4_K_M DFlash2**. Matched d0/d32K pp2048/tg128 controls retain all 128
greedy tokens and PP speed; DFlash2 matches AR at both depths.
Partitioning changes FP32 rounding: twelve controls at 128–4096 tokens and three
query amplitudes have lower maximum error and RMSE against an independent FP64
attention formula; the largest new absolute error is **1.17e-6**. Maintained
FP16/FP32 reference checks retain their tolerances and cover the 128-token
boundary. Q4 prefill/snapshot replay retains complete logits and features at
128, 257 and 2048 tokens, plus an 8K cached prefix with a 1025-token suffix.
This does not establish original-checkpoint parity.
Disk-cache identity now includes the partition threshold and count, preventing
restoration of snapshots computed with the old arithmetic. Broader target and
concurrency qualification follows the focused Q4 C1 phase.

Graph identity includes the target hidden-layer taps and their order. The Q4
64-token replay check changes that order after capture and verifies every
feature value against the reordered reference, with unchanged full logits.
The 128/257/2048-token and 8K-prefix replay fingerprints also remain unchanged.

Shared FP16 KV tiles preserve each head's scalar product/FMA order and softmax
sequence. Scalar AR uses 32 lanes per head; verification pairs two 16-lane
heads per wave. Each half keeps the original lane partials separate until their
original offset-16 addition, preserving the full reduction tree.

All 63 direct comparisons are byte-identical in both partial state and output:
widths 2–8, three input scales and contexts 512/2048/32765. Maintained FP64 checks
and scalar/batched comparisons through 64K pass, including dispatch boundaries
and scratch fallback. All 15 full-model logit/feature fingerprints are unchanged,
including verification and snapshot replay after an 8K prefix. Shared memory
remains 4 KiB per active block; persistent state and snapshot formats are unchanged.
Matched C1 pp2048/tg128 controls retain greedy AR/DFlash2 agreement at d0/d32K.
Current measurements:
[AR](artifacts/q4-ar-c1-focused.json),
[DFlash2](artifacts/q4-dflash2-c1-focused.json).

Draft attention prefetches 32 values and shares K/V reads across two query heads
for blocks of at least four rows. It preserves the original FP32 dot-product,
softmax and ascending-key value accumulation. All 216 full-output comparisons
with the previous kernel are byte-identical: widths 1–8, three input scales,
empty history, prefetch boundaries and ring wrap through 32K. The maintained
independent FP64 checks and draft/state tests pass, including unequal widths,
selector probabilities, private RNG and C2/C4/C6/C8. Persistent allocations
and cache formats are unchanged.

The Q4 adaptive controller includes measured attention-cost growth with context.
At saturation, its accepted-run estimate remains censored; it probes wider
profitable blocks instead of treating the configured cap as a rejection.
Completed blocks update that estimate in proportion to their accepted tokens.
An independent geometric-distribution check verifies zero expected feedback
drift at every width 1–7, away from the saturation cap; a fixed success
increment biased the estimate according to block width.
Sampled requests use private positions and acceptance history before drawing
proposals. Greedy Q4 cohorts of two, four, six or eight select their measured
costs; fixed mode, C1 and Q8 cost tables are unchanged.
The maintained draft check passes exact layers, logits, selector probabilities,
private RNG and persistent-state replay at C2/C4/C6/C8.
The current release reproduces all 32 sampled tokens and draft counts between
cold and cached requests at **65,138 prompt tokens**, temperature 0.8 and seed 1.
The cached replay prefills zero tokens. Changed draft widths can change sampled
sequences across builds; within-build replay remains deterministic.

Matched Q4 C1 pp2048/tg128 controls retain all three complete AR output hashes
at d0/d32K/d64K. The initial d32K PP reading was lower; the subsequent
baseline/candidate control agrees at 473.1 tok/s. Both candidate samples are
retained in the artifact.
The 128-token repetition controls retain the same output and 100% acceptance at
d0/d64K. Context-cost calibration, fixed-three-proposal controls and current
measurements are in the existing
[DFlash2 artifact](artifacts/q4-dflash2-c1-focused.json).
These controls cover Q4 C1; the focused C2 qualification follows.

The focused Q4 C2 release checks retain all six complete 128-token C1 AR
outputs: repetition, pangram/train and Italian/Chinese pairs. Fresh servers
use `cache_prompt: false`; every request prefills its 30–53 prompt tokens.
Paired greedy costs account for the projection jump above eight total rows.
Fixed-three controls improve low acceptance but lose on the higher-acceptance
pair, so adaptive remains the default. The maintained draft test additionally
compares sampled and mixed greedy/sampled pairs with isolated execution:
proposal lengths, IDs, probabilities, private RNG and restored state match.
The paired calibration applies only to two greedy drafters, including a
larger cohort that shrinks to two; sampled choices remain private.
The matched C2 d32K continuation also retains both 128-token C1 AR outputs:
each request reuses 32,552 tokens and prefills 2,011. New and previous
builds receive byte-identical messages, execute a physical two-request batch,
and retain matching prefill times. C1 d0 PP stays within 1% of the previous
control, with unchanged greedy output and DFlash2 acceptance counts.

The [C2 artifact](artifacts/q4-c2-focused.json) retains per-request timings,
fixed-width cost calibration, d32K continuation, C1 pp2048/tg128 controls and a separate
profile of four saturated decode cycles: 97.0% GPU-busy, with quantized
projections accounting for 87.6% of kernel time. Target verification takes
85.9%; draft generation and committed context injection account for the
remainder.

The batched selector retains exact private token chains and probabilities.
Ragged operator checks cover C2/C4/C6/C8, top-k 1/7/16, greedy and sampled
temperatures, tiny positive temperature and zero random draws. A separate
full-vocabulary comparison matches the original scalar kernel byte-for-byte,
including the refactored C1 path. The real-weight draft check retains exact
layers, logits, proposals, probabilities, private RNG and persistent replay.
Batch partial lists reuse finished FFN buffers; cache formats are unchanged.
The [focused C4 control](artifacts/q4-c4-focused.json) retains all four complete
C1 AR outputs and full acceptance. C1 pp2048/tg128 AR and DFlash2 retain output,
acceptance and speed.

The C4 cost table covers complete cycles at widths 1–7 and the projection jump
above sixteen verification rows. It applies only to four greedy Q4 drafters;
C1, C2, Q8, fixed mode and sampled choices remain unchanged. Sampled and mixed
C2/C4 cohorts match isolated lengths, token IDs, probabilities, RNG and restored
state. All twelve shallow and four d32K candidate completions match C1 AR.
The matched deep control reuses the same 32,552-token prefix and prefills 2,011
tokens per request, with comparable prefill times. C1 DFlash2 retains its output,
acceptance and pp2048/tg128 speed. The
[C4 artifact](artifacts/q4-c4-focused.json) records calibration, per-request
timings and hashes.

The C6 cost table likewise uses complete measured cycles for 12–48 verification
rows, while retaining the full-block probe at saturated acceptance. It changes
only six-request greedy Q4 decisions. C1/C2/C4, Q8, fixed mode and sampled
choices remain unchanged. The maintained model check passes sampled/mixed
C2/C4/C6 isolation, exact probabilities, private RNG and persistent-state replay.
New requests reset controller history before generating.
All 18 shallow and six d32K candidate completions match C1 AR; repetition
retains 100% acceptance. Deep requests reuse 32,552 tokens and prefill 2,011,
with comparable PP times. The [C6 artifact](artifacts/q4-c6-focused.json)
retains calibration, binary identities, individual timings and output hashes.
Broader depths remain pending.

The C8 table covers 16–64 verification rows. A comparison against the previous
controller retains all 1,511,622 decisions outside adaptive Q4 C8 across
histories, limits, budgets, positions, cohort sizes, Q8 and fixed mode. Maintained controller
invariants and the full-weight draft test pass, including independent sampled
and mixed C8 requests, selector probabilities, RNG and persistent replay.
All 24 shallow and both eight-request d32K candidate cohorts match C1 AR.
The second deep cohort checks a 7.48-second PP outlier; it does not recur, and
typical PP times remain comparable. Both cohorts are retained in the
[C8 artifact](artifacts/q4-c8-focused.json). The matched AR Italian/Chinese
control is still faster than DFlash2; these improvements do not establish
universal speculative profitability or original-checkpoint parity.

The fresh pinned llama.cpp `68d9053a` d0 controls use the same input messages,
Q4 target/draft, greedy sampling and context capacity. Its DFlash2 output differs
from its own AR output after “disjointed, repetitive, and”: AR continues with
“grammatically fragmented phrases,” DFlash2 with “syntactically broken phrases.”
Its default limit is three proposals; Gufo remains adaptive. Gufo's AR and
DFlash2 output hashes agree on this prompt. This isolates the observed mismatch
from Gufo's speculative acceptance, but does not establish which engine better
matches the original checkpoint. Counts, hashes and pinned binary identity are
retained in the [focused artifact](artifacts/q4-dflash2-c1-focused.json).
Its fresh d64K DFlash2 control also uses byte-identical request messages and
the same context capacity as Gufo. The reference's own d64K AR output has not
been measured, so its d64K output difference does not establish a speculative
verification mismatch.

The following results describe the qualification through `b509c070`, before
lowering the split-K threshold:

Grouping verification queries by KV partition retains the existing arithmetic.
The old/new FP16 kernels are byte-identical on 36 full-output cases through 64K
plus six partition-boundary controls. Maintained checks cover FP16/FP32,
1/3/8 query rows, scratch fallback and compact snapshot restoration. Matched
pp2048/tg128 C1 HTTP controls on Q4_K_XL and Q8_K_XL with Q4 DFlash2 retain output
hashes and accepted/proposed counts at d0 and d32K, with comparable PP speed.
Fresh AR requests match DFlash2 on all four 128-token outputs at those depths.
These focused controls do not refresh the benchmark sweep.

The shallow FP16 attention pipeline retains the original product/FMA order and
lane-zero sum tree. Scalar, batched and device-position paths match the previous
kernel byte-for-byte on 72 controls across amplitudes and ragged lengths.
Focused attention, graph and KV/snapshot tests pass. Q4_K_XL and Q8_K_XL retain
all four d0 pp2048/tg128 AR/DFlash2 output hashes and both acceptance counts.
Both targets also pass C2/C4/C6/C8 full-logit, feature and verification-replay
checks, including ragged cohorts and shrinking batches.
The full AR profile confirms lower attention time; the deep split-K path and
prefill kernels are unchanged.

The 2026-09-21 compact-state qualification preserves Q4/Q8 AR and DFlash2
tokens across all 23 sampling cases and C2/C4/C6/C8. Active recurrence and
rollback buffers contain only recurrent layers; snapshots retain only valid
KV rows. Operator checks cover FP16 production and head-major FP32 reference
snapshots, dirty unused tails and exact disk round trips. Matched production
pp2048/tg128 controls show no material speed regression.

Interrupted-chat qualification also passes on Q4_K_XL and **Q8_K_XL**, both
AR and Q4 DFlash2: reasoning/visible-text cancellation, preserved/removed
reasoning, greedy/seeded replay, a third turn, images and disk restart.
The native vision checks retain their cold-versus-live equality gate and
concurrent image/text isolation. The HTTP check requires exact replay with
the same history and verifies that `cache_prompt: false` bypasses reuse.
See [server check instructions](../../SERVER.md). Existing benchmark tables
were not refreshed by this cache qualification.

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
history and offline verification costs, including context-dependent Q4 attention
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

The retained matrix covers 23 named strategies on Q4/Q8 AR and every draft under
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
