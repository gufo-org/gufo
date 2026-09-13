# Qwen27B quality checks

Production targets: UD-Q4_K_XL and UD-Q8_K_XL. BF16 is an optional
quantization reference. A Gufo BF16 run is not independent proof that the
entire target model matches the original checkpoint.

## Maintained checks

Run on gfx1151 inside Nix. Model-specific tests live in
`tests/models/qwen27b`; shared quantization/operator controls remain in
`tests/models/qwen` and are selected by the same runner.

| Suite | Contract |
| --- | --- |
| `fast` | Sampling/verifier and HTTP parser regressions, executable option validation, NPU packing, GGUF reference decoding and strict result reporting. |
| `kernels` | Quantized/BF16 GEMM versus independent/decode controls; exact recurrent state and replay; DFlash convolution, windowed attention, full-vocabulary top-k, sampled selector and verifier distributions. |
| `model` | Target full-logit and tapped-feature replay at verification widths 2–8; MTP committed-feature alignment; DFlash loading, ring/snapshot/restore; prompt/chat/bench parity, seeded multi-turn replay and HTTP adapter sampling. |
| `serving` | Direct versus served tokens, seeded sampled replay, EOS, bounded prefill, cache forks, persistent restore, concurrency, cancellation and reclamation. |
| `reference` | Teacher-forced target versus optional BF16: KL, total variation, top-1 agreement, RMSE and NLL difference. Informational quantization measurements. |

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py kernels
nix develop -c python3 tools/qwen27b/check.py model \
  --model "$MODEL" --mtp-model "$MTP" --dflash-model "$DRAFT"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
# Optional; load the two targets sequentially, never simultaneously.
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
```

The GPU correctness preset uses optimized code with symbols; test assertions
remain enabled. Performance measurements always use Nix release binaries.
Artifact variables `GUFO_QWEN27B_*_MODEL` are test inputs, not execution switches.

The maintained executable test is `tests/models/qwen27b/cli_test.py`.
Use `--scope options` without models, or `--scope sampling` / `--scope http`
with artifact variables. `--backend ar` needs only the target;
`--backend dflash2` adds the draft; the default checks both. The 15 executable
cases cover individual controls and combinations across six HTTP adapters,
seeded replay, unseeded requests, streaming and C2 state isolation.
Terminal checks use four tokens per turn; HTTP checks use eight.

`sampling_cases.hpp` holds 23 named cases shared by CPU, GPU and model tests:
greedy; cold/normal/hot temperature; top-k, top-p and min-p independently;
each minimum-candidate floor; repetition penalty/reward; positive/negative
frequency and presence penalties; history disabled, one token and longer
windows; and combined filters. Seeds include 0, 73 and 808.

The GPU test checks 216 configurations (864 AR quantiles). Its 141 random
configurations also bracket each acceptance probability **p/q** and check the
normalized positive **p−q** residual CDF against the double-precision CPU
reference (2e-5 probability tolerance). Four additional controls use the full
248,320-token vocabulary. Exact nucleus cutoffs, cross-thread argmax ties,
zero random draws and underflow have regressions. CPU replay checks 1,472
steps against the reference, with evolving committed history.

For quick model qualification, load a target and optional draft once:

```sh
nix develop -c build/gpu-test/tests/models/qwen27b/inference_backend_gpu_test \
  "$MODEL" "$DRAFT" --sampling-only
# Omit "$DRAFT" for AR only; add --fixed for the fixed DFlash2 controller.
# Do not also set GUFO_QWEN27B_DFLASH_MODEL when passing a draft positional path.
```

Each named case checks four generated tokens, cold/cached replay for both
backends, first-token equality and actual drafting. Deterministic strategies
must match all AR IDs. Sampled strategies preserve the target distribution;
AR and DFlash2 consume different RNG sequences, so their full sampled
continuations need not match. This is a finite, explicit coverage matrix,
not a claim to test every numeric parameter combination or model capability.

The [same-artifact Vulkan comparison](llama-comparison.json) checks two raw
tg300 prompts against each engine's own AR. All six Gufo speculative runs
match every AR token and repeat deterministically. The pinned Laurent fork
matches JSON, but its two prose runs first diverge from its AR at token
138 and 121 (zero-based), and differ from each other. This does not identify
a controller bug or establish semantic quality loss; neither engine is an
independent original-checkpoint oracle.

## Executable and sampling contract

| Entry point | Qwen27B behavior |
| --- | --- |
| `prompt`, `chat` | Shared target sampling and GPU DFlash2 implementation; `fixed`/`adaptive` and draft length cap. Chat reads stdin and rejects prompt-only input/display arguments and raw framing. |
| `bench` | Greedy, C1, with AR/MTP/DFlash2; sampling flags are rejected. Qwen C>1 measurements use the HTTP benchmark. |
| `serve`, `serve llm`, `gufo-server` | DFlash2 and sampling defaults from startup; request sampling overrides across OpenAI chat/completions/responses, Anthropic messages, and llama completion/infill. `--cpu` is removed. |
| `eval` | Uses the connected server's draft configuration and sampling defaults; `--greedy` overrides temperature only. |
| `video`, `transcribe`/`asr`, audio/video serving, `diagnose`/`info`, `probe` | Do not run Qwen27B DFlash2; unsupported draft flags fail explicitly. |

Use `--speculative dflash2` in prompt, chat, bench and serving. `off` disables
speculation. Each behavior has one spelling.

The target supports temperature, top-k, top-p, min-p, min-keep, seed,
repeat penalty/window, frequency penalty and presence penalty. Ordering is
penalties → top-k → top-p → min-p → temperature. `min-keep` is a floor even
when top-k is 1. At temperature 0 the final choice is greedy.

Qwen AR and DFlash2 apply these operations on the GPU, including acceptance
and residual sampling. The CPU holds request history and RNG state; cached
host logits are uploaded before sampling. The shared CPU sampler supplies
reference/generic paths and is also used by DS4 serving.

DFlash2 shares the target temperature and samples its trained top-16 selector.
There are no independent draft temperature/filter/seed controls. Target
filters and penalties apply to **p**; verification uses the actual proposal
**q**. Unsupported draft sampling fields and unsupported alternative sampler
fields in HTTP requests return an error instead of being ignored.
This includes typical/tail-free/Mirostat/dynamic-temperature, XTC, DRY,
top-n-sigma, custom sampler ordering and logit bias; these are unsupported.
The block controller is configured when starting the server, not per request.
OpenAI chat and server defaults bound temperature to [0,2]; the native sampler
accepts any finite nonnegative temperature.

Use the existing model test's optional
`--acceptance-trace request.json output.json` mode to inspect block acceptance.
The request contains `prompt`, optional `raw`, and `max_tokens` (1–256).
It records proposed/emitted text, accepted counts and first corrections, and
requires the full continuation to match AR IDs. This diagnostic includes
extra snapshot/proposal work; never use its runtime as a speed measurement.

## Model/operator checks

`qwen_q4kxl_quant_ops_test` owns the fused/packed-SwiGLU versus separate
batched-verification contract: two distinct inputs, same/mixed formats,
partial row groups and full-size Q8/Q6 controls. All 70,686 outputs per route
must be finite and bit-identical.
It replaces the older same-format-only test in `qwen_kquant_gemv_ops_test`;
independent CPU decode/GEMV/GEMM controls remain.

The existing target test checks full logits and all five DFlash2 features
at verification widths 2–8. Its mixed-cache control uses three sessions with
32/64/128 capacities, rotates each as coordinator and checks nine full-logit
rows per cache precision. This exercises packed FFNs in independent requests.

The unused in-tree CPU DFlash forward pipeline and its duplicate operator
tests are removed. The pinned upstream runner owns the full reference.
All three draft artifacts pass loading and state tests. The optimized
implementation preserves all 270 trace files (90 per draft) byte for byte,
including intermediate layers, logits and sampled proposal distributions.
A long-injection control compares discarded-chunk execution with incremental
injection; the complete serialized history must remain byte-identical.

## Current evidence

- [Short-block Q5 projections](dflash2-prose-projections.json): medium
  projections at widths 5–6 reuse four rows, with unchanged arithmetic.
  Selected isolated gains are 3–13%; release prose improves 0.24%,
  repetition 0.21%, and pooled JSON decreases 0.13%. All 16 continuations,
  102 full-logit rows, 102 feature rows, C3 cache controls and 90 Q4 draft
  traces remain exact. The matched 9,450-call projection union includes
  unchanged FFN down projections and takes 2.14% less GPU time; whole prose
  GPU time falls 0.43%. Wider tiles, compiler barriers, full draft blocks and
  residual/norm fusion were rejected. A 256-token acceptance diagnostic
  remains AR-exact; sparse later-position observations do not justify a
  controller change.
- [Remaining mixed-format projections](dflash2-remaining-formats.json):
  selected Q6/IQ projections reuse activations across four output rows at
  widths 3–8, preserving the exact kernel arithmetic. Both quantized operator
  suites, 102 full-logit rows, 102 feature rows, 18 final-feature checks,
  C3 cache controls, 90 Q4 draft traces and all 16 measured AR continuations
  pass. The affected 1,341 JSON projection calls take 11.53% less GPU time,
  without private scratch or added allocations; total GPU time falls 0.41%.
  Paired release throughput improves JSON 0.53% and repetition 0.21%;
  pooled prose is flat (-0.02%). Narrow indexing alone and several shorter
  batch layouts regress and remain excluded. No maintained test/tool or
  execution option is added.
- [Contiguous FFN projections](dflash2-contiguous-ffn.json): already adjacent
  same-format gate/up tensors share one exact matrix launch at widths 3–8,
  followed by packed SwiGLU using existing scratch. Projection-plus-activation
  microbenchmarks improve 1.1–4.4%; short paired end-to-end controls improve
  JSON 0.30%, prose 0.28% and repetition 0.14%. All 102 verification logit
  rows, 102 feature rows, 18 final-feature checks, C3 cache controls,
  90 Q4 draft traces and 12 measured AR continuations remain exact.
  A pp2048/tg16 depth-4096 control also matches AR across the attention
  split-K threshold. The JSON trace confirms 1,554 fewer launches without
  added allocations; total traced GPU time does not improve, so throughput
  comes from the unprofiled paired runs. No new test executable, tool or
  execution option is added.
- [Consolidated feature transfers](dflash2-feature-transfer.json): target
  verification captures taps on the GPU and copies them to the host together,
  reusing the logits workspace without extra allocation or changed arithmetic.
  The existing target test now checks all five DFlash2 taps and the retained
  final row: 102 full-vocabulary rows, 102 feature rows and 18 final-row checks
  are bit-exact across Q4/Q8. Cache/context controls, 46 sampling cases and
  all 12 measured AR continuations pass. Short paired runs improve JSON 1.93%
  and repetition 1.45%; prose is nearly flat (+0.27%). The trace confirms
  129 fewer copy API calls over 43 rounds and identical model-kernel calls.
  API waits include preceding GPU work; they are not pure copy cost.
  Lossless Q6 head expansion preserves outputs but loses 14–26%, so it is
  rejected. A lower-barrier norm saves about 0.30µs per call in isolation;
  its projected total benefit is below 0.1%, so it remains unshipped. A
  single-wave norm is about 5.6× slower. Both preserve checked outputs.
  No new maintained test/tool or execution option is added.
- [Rejected matrix/loop probes](dflash2-matrix-probes.json): skipping the final
  tile barrier loses up to 2.5% while preserving every checked output. Native
  integer WMMA with four activation components slightly improves sampled FP64
  operator error, but takes 67–78% more time in the relevant paired controls.
  Coalesced weight staging and larger K splits do not close the gap; staged
  and direct outputs agree exactly. This does not qualify changed model
  numerics. The draft GGUF's block size is eight, so seven proposals plus the
  anchor is the correct cap. Production and maintained checks are unchanged.
- [Shared-memory synchronization](dflash2-synchronization.json): packed
  projections retain LDS completion, the workgroup barrier and compiler
  ordering while avoiding global cache invalidation. Arithmetic is unchanged.
  Both kernel suites, partial rows, all 51 Q4 full-logit rows, cache/context
  controls, 90 draft traces and 12 measured AR continuations pass.
  Short paired runs improve prose 0.71%, JSON 1.02% and repetition 1.22%.
  The same 18,318 affected projection calls take 1.12% less GPU time with
  zero scratch. Slower integer-residual, full-tile and precomputed-sum probes
  are rejected; no new maintained test, tool or execution option is added.
- [Exact wave reductions](dflash2-reductions.json): immediate XOR shuffles
  retain the scalar addition order. All 51 Q4 verifier/scalar full-logit rows,
  mixed-cache/context controls, 90 draft trace files and 12 measured AR token
  continuations remain exact. Both kernel suites and the partial-row probe
  pass. Short paired runs improve prose 1.0% and JSON 0.39%; repetition is
  flat. All 18,318 affected projection calls take 0.38% less GPU time with
  zero scratch. Q8 target and Q8/BF16 drafts contain none of the seven
  affected packed formats. The record also rejects slower lossless row
  packing and residual-precision WMMA; neither changes production numerics.
- [Measured controller costs](dflash2-controller-cost.json): Q4 uses an offline
  cost curve for each proposal count; Q8 retains its previous formula.
  All 20 measured greedy continuations and both depth controls match AR;
  state/snapshot tests and 23 sampling replays pass. JSON improves 4.7% and
  prose 2.8%. Three chat prompts are flat in aggregate, with code/reasoning
  losing 1.0–1.6%; the one depth-4096 sample loses 1.0%. Decisions precede
  proposals and never use live timing. Target/draft arithmetic is unchanged.
- [Q5 batch-eight FMA scheduling](dflash2-fma-order.json): alternating
  multiplication operands retains all 51 Q4 verifier/scalar logit rows,
  cache/context controls and all 90 draft trace files. Both kernel suites
  and the maintained Q5 partial-row/tile probe pass. All 16 release
  continuations and acceptance statistics match their AR references.
  JSON improves 0.35%, repetition 0.31%, and prose is effectively flat.
  The affected kernel union takes 0.93% less GPU time with zero scratch.
  Q8 target and Q8/BF16 draft artifacts contain no Q5 tensors.
  Further Q4 batch-eight/Q5 batch-seven scheduling changes pass the same
  quality checks but leave release throughput effectively flat, so they
  are reverted. Six additional Q5 FMA patterns also fail to beat the
  controls consistently; the same record includes both rejected experiments.
- [Mixed-format scalar projections](dflash2-mixed-formats.json): independently
  specializing gate/up formats retains all 51 Q4 verifier/scalar logit rows,
  the cache/context controls and all 90 Q4 draft trace files. The consolidated
  fused contract and two kernel suites pass. Depth-zero AR improves 0.5%;
  JSON/prose DFlash2 controls are unchanged within noise. The relevant fused
  kernel union takes 3.6% less GPU time; the four new variants have no spills.
  No Q8-target tensor pair selects a new specialization; its preceding
  full-model qualification remains below.
- [Two-token verification](dflash2-width2.json): two-token Q4/Q5/IQ4/Q6
  projections preserve 51 verifier/scalar full-logit rows on each target,
  mixed-capacity and logical-context controls, all 90 Q4 draft trace files,
  and every measured AR ID/acceptance statistic. Three kernel suites and
  partial-row/tile controls pass. One-proposal repetition improves 12.6%;
  adaptive JSON/prose improve 0.2–0.3%. The affected projection union takes
  14.3% less GPU time with zero scratch. The maintained suite now covers
  every width 2–8. [Compact Q4 staging](dflash2-q4-staging.json) covers the
  preceding widths 3–8 work. Q8/BF16 draft qualification remains in the
  [preceding Q5 pass](dflash2-middle-projections.json), which also documents
  the rejected controller trials. Earlier [batch-eight scheduling](dflash2-row-scheduling.json),
  [IQ4/vocabulary work](dflash2-head-iq4.json),
  [widths 4–6](dflash2-midbatch.json) and
  [scalar/row reuse](dflash2-scalar-row-reuse.json) remain documented.
- [Packed decoding and verification](dflash2-packed-decode.json): 40 retained
  projection cases cover widths 3–8 with finite, bit-exact output comparisons.
  Both target suites and all 270 draft trace files pass the scale/index update;
  the final Q5 batch-8 layout is rechecked on Q4 target logits and draft traces.
  All 24 new compiled kernel variants have zero private scratch. Paired Q4/Q4
  release probes retain every AR token ID and acceptance statistic, including
  TG300 JSON/prose and TG128 repetition; bounded 0/4K depth checks also match.
- [Sampling qualification](sampling-strategies.json): all 12 combinations of
  Q4/Q8 targets, Q4/Q8/BF16 drafts and fixed/adaptive controllers pass the
  23-case AR/DFlash2 replay check. HTTP passes AR on both targets, all six
  adaptive target/draft pairings, and fixed Q4 drafts on both targets, each
  with 15 configurations across six adapters. Final release checks cover
  options, a bounded Q4/Q4 benchmark replay, and 15 prompt/chat configurations
  for Q4 AR and Q8/BF16 adaptive DFlash2. These are correctness probes, not
  new speed measurements.
- Q4 and Q8: fixed 24/29/24-token prefixes, eight forced continuation tokens
  each, plus widths 2–7 from the first snapshot. Repeated prefill and snapshot
  continuation are byte-identical. All 51 batched-verifier logit rows per
  target equal scalar decode byte for byte; committing five rows across the
  replay ring's boundary also preserves
  subsequent logits. Independent sessions with 32/64-token cache capacities
  retain all logits and returned IDs in either batch order, with FP16 and
  FP32 caches. A 262,144-token logical cache with only a 24-token prefix
  matches a small cache for prefill, scalar decode and verification. Both
  cache-addressing regressions failed before their fixes.
  Prefill and scalar decode retain the same top-1 choice
  on these prefixes.
- MTP: feedback replay equals a fresh teacher-forced committed prefix. The
  regression fails when replay uses the newest target hidden row for the old
  proposal anchor.
- DFlash2: FP32 history is bounded by its 2,048-token attention window: 80 MiB
  at a 262,144-token logical context. Ring wrap, persistent restore and replay
  retain proposals and confidence values. Scratch ingestion is bounded to
  256 rows.
- DFlash2 operators: independent equations cover both convolution coefficient
  planes, attention window boundaries and partial-top-k partitions. Top-k
  1, 7 and 16 run through the full 248,320-token vocabulary, including poisoned
  scratch, zero random draws, zero-mass candidates and tiny temperatures.
- Release companion comparison: both targets × Q4/Q8/BF16 drafts × three
  chat prompts × two interleaved repetitions: all 72 baseline/candidate
  cases match all 128 target token IDs.
  See [the optimization record](dflash2-optimization.json). This is a development
  corpus, not a capability evaluation or proof across arbitrary contexts.
- Fixed-block depth sweep: both targets × three drafts × five depths; all
  30 speculative TG128 traces match their target AR IDs. pp2048 includes
  DFlash2 context preparation. One timed repetition after warmup; see
  [the depth record](dflash2-depths.json).
- `prompt` and `chat` share GPU/speculative setup. Their first-turn tokens
  match, and two chat turns reproduce AR with both draft backends. Verbose
  generation emits a SHA-256 over little-endian token IDs; the corpus rejects
  missing/incomplete traces and text-only matches.
- Removed an unused 170 MiB target scratch allocation per session. Inactive
  fusion/prefetch branches and their kernels/tests are deleted; production
  arithmetic and independent operator controls remain.

Use kernel ablations and short release probes during iteration. Run the
affected operator check first, then model replay on both target quants for a
retained change. Reserve full corpus/depth sweeps for qualification milestones;
label short measurements with their actual scope.
Compare token IDs and full logits, not just decoded text or acceptance.
Benchmarks retain independent target/draft prefix snapshots, separate from
verification rollback. Repeated cached-depth TG must reproduce AR token hashes;
DFlash2 prefill-only invocations must load and initialize the companion.
Check both streamed decode weights and reused context-injection weights:
cache hints can reverse their relative performance. Symmetric quantized
projections must round the scaled dot before adding it, matching scalar decode;
the existing exact GEMM checks guard this against compiler FMA contraction.
Verification and its persistent-cache identity use that decode-equivalent contract.

Use `tools/qwen27b/drafts.py` for matched companion comparisons; optional
`--draft-bf16` includes BF16. `--baseline-binary` interleaves release A/B runs
and rejects changed autoregressive token streams between releases. Do not rank acceptance across different
target-generated continuations.

## Original DFlash2 source

Pinned upstream: `z-lab/dflash` commit
`07ebd93db9f472af339b644bb70221ad8428328a`, `dflash/model.py`.
Its SHA-256 is
`f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e`.
The reference imports its unmodified `DFlash2DraftModel` and `CandidateSelector`.
PyTorch runs FP32 operations over independently decoded copies of the same
GGUF weights, including Gufo's documented BF16 weight packing.

| Operation | Source contract checked |
| --- | --- |
| Target taps | Layer outputs 5/19/33/47/61, after the FFN residual; normalize GGUF layer-input indices once on load. |
| Embeddings | Independently decode the target anchor and mask-token embedding rows. |
| Context injection | Shared feature projection/norm, per-layer K/V projection, per-head K norm, absolute-position RoPE. |
| Attention | All proposal keys visible; historical keys satisfy query minus key < window; 32 query heads, 8 KV heads, dimension 128. |
| Dynamic convolution | Causal zero padding; static and dynamic grouped coefficients; separate prepare/finish planes. |
| Selector | Unary top-k then predecessor × projected hidden × successor score; selected token becomes the next predecessor; report the probability actually sampled. |

The trace checks every layer twice: accumulated forward execution, then a
teacher-forced layer with Gufo's input. It also isolates the output head and
selector, compares conditional distributions by token ID, and checks that each
random draw selected the reported token and probability. Candidate order can
differ from upstream's unsorted top-k without changing the distribution.

One real 24-token prefix from the Q4 target, seven proposals at temperature
0.8; all three draft artifacts pass. All 21 candidate sets and random draws
match. Full [per-stage evidence](dflash-upstream.json):

| Draft | Worst stage relative RMSE | Full-logit max error | Proposal max total variation |
| --- | ---: | ---: | ---: |
| Q4_K_M | 5.83e-6 | 7.72e-5 | 1.60e-5 |
| Q8_0 | 5.74e-6 | 9.54e-5 | 1.25e-5 |
| BF16 | 6.16e-6 | 1.13e-4 | 7.01e-6 |

```sh
# Fetch this small source file explicitly; no model download is implicit.
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

Use a new trace directory and run artifacts sequentially. Gates are fixed:
stage relative RMSE ≤1e-4, full-logit maximum error ≤1e-3, proposal total
variation ≤1e-4, and isolated selector probability error ≤5e-6.
This checks execution over packed weights and captured target features.
It does not validate GGUF conversion against original safetensors, establish
original-target correctness, or promise native BF16/MLX bitwise equality.

## Token generation and verification policy

- DFlash2 uses one anchor plus up to seven proposals. Block length is selected
  before drawing tokens and bounded by the context and remaining output budget.
  `--draft-policy fixed|adaptive` is wired through prompt, chat, bench and serving.
  Adaptive is the default for the recommended Q4_K_M draft. It tracks an EMA
  of accepted lengths and maximizes expected emitted tokens per estimated
  verification cost. Full acceptance raises the estimate because the observed
  length is censored. Its state is saved within a request and reset for a new
  request, including prefix-cache hits. Decisions never use wall-clock timings
  or the current sampled proposal. `--draft-tokens` bounds either policy;
  nondefault `--min-draft-tokens` values remain unsupported for DFlash2.
  Unary top-16 candidates receive the predecessor/hidden/successor transition
  score; the temperature softmax is the proposal distribution `q`.
- The verifier uses the same target distribution `p` as AR, including committed
  repetition history. It accepts proposal `y` with probability `min(1,p(y)/q(y))`.
  Rejection samples normalized `max(p-q,0)` across the **whole target
  vocabulary**, including tokens outside the draft's top-16.
- Every candidate row is validated before target execution: dimensions, unique
  in-range IDs, finite nonnegative normalized probabilities, and positive mass
  for the proposed token. Verification restores rejected state and commits only
  the consumed prefix. EOS and output limits stop further commits.
- The sample-dependent confidence cutoff and its CLI/Nix wiring are removed:
  discarding a token based on its sampled probability changes `q`. Greedy
  diagnostics report the actual one-hot proposal distribution.
- Seeded runs must reproduce token IDs within a fixed Gufo configuration,
  including cached replay. AR and speculation consume different random draws,
  so equal seeds do not imply equal sampled continuations. Greedy runs must
  match AR exactly. Fresh, speculative and restored-frontier first tokens use
  the same GPU target sampler; saved logits are uploaded before sampling.
- Greedy requests with penalties also use drafting. Each row is compared with
  the target's penalty-adjusted argmax after earlier accepted tokens update
  tentative history. Rejection and EOS commit only that prefix; the RNG does
  not advance. [Release checks](dflash2-sampling.json) cover both targets and
  all three drafts; the serving regression checks AR equality and cached replay.

The GPU test enumerates acceptance branches and residual CDF intervals using
the GPU selector's probabilities, then reconstructs the target distribution
(tolerance 2e-6). It covers filtered and unfiltered targets, including mass
outside the candidate set. This is stronger than checking acceptance rate or
running a noisy frequency test.

Gufo applies penalties → top-k → top-p → min-p → temperature in both AR and
verification. Upstream applies temperature before top-p. Matching CLI values
therefore need not define the same target distribution; the preservation
contract here is Gufo AR's distribution.

Remaining: original checkpoint/conversion and MTP qualification, optional BF16
target comparison, and longer capability/depth coverage.

## Optimization evidence

Each record contains its own workload, artifact hashes and measurements.
Gains from different workloads must not be added together.

| Record | Retained result |
| --- | --- |
| [Draft operators](dflash2-optimization.json) | FP32 draft activations preserve the pinned packed-weight reference; exact projection geometry and bounded injection. |
| [Initial profile](draft-profile.json) | Target projections dominate; draft precision alone does not predict total speed. |
| [Verification](dflash2-verification.json) | Exact Q6 vocabulary/Q4 projections and batched embedding; no material total-speed gain. |
| [Recurrence/injection](dflash2-recurrence.json) | Register-resident FP32 recurrence, batched state-only replay, sixteen-row BF16 injection; C1 generation +2.7–3.6%. |
| [Projection follow-up](dflash2-projections.json) | Compact Q5 staging, two-tile IQ4 projection and BF16 grouping; Q4 +1.4–2.1%, Q8 within noise. |
| [Attention](dflash2-attention.json) | Exact batched attention/QK/cache writes and narrow SSM projections; C1 chat +3.2–3.5%. |
| [Rollback and cache addressing](dflash2-rollback.json) | Compact rollback, fused draft normalization/RoPE and Q4 convolution projections; fixes mixed-capacity and 32-bit cache offsets. |
| [Exact projections](dflash2-exact-gemm.json) | BF16 streaming with cached K/V injection; symmetric projections omit offset scratch while retaining scalar rounding; dead verification settings removed. |
| [Verification widths 3–8](dflash2-batch-widths.json) | Selected Q4/Q5/IQ4 staging tiles and Q8 batch-7 workgroups; Q4 batch-7 spills removed. Full target logits and all 270 draft trace files remain exact. |
| [Greedy sampling controls](dflash2-sampling.json) | Penalty-enabled requests retain DFlash2 and exact AR output; one C1 chat probe reaches 30.25–31.43 tok/s on Q4 and 25.89–27.03 on Q8 across the three drafts. |
| [Controllers](dflash2-controllers.json) | With Q4 draft, adaptive gains 3.3% on the pilot and 2.0% on three other prompts for the Q4 target; Q8 target is effectively unchanged. Q8/BF16 drafts can favor fixed. Accepted-length-only and positional predictors were slower overall. |
| [Sampling and executable wiring](dflash2-wiring.json) | Six target/draft HTTP pairings pass; fixed/adaptive terminal, cached benchmark and backend checks cover both targets. The sampling-floor regression fails before the fix; 576 GPU/reference comparisons pass afterward. Pure repetition reaches 100% acceptance across all six pairings. |

Rollback snapshots omit unused attention-layer rows: **202 → 151.5 MiB**
per speculative session, preserving every live state byte. Existing operator
checks cover FP32/BF16 storage, incomplete layer groups and repeated saves.
The cache fixes use each session's capacity and size_t arithmetic; the large
logical-context test exercises only 27 tokens and does not fill the cache.

Draft normalization/RoPE shares the target's existing fused kernel, including
cacheless in-place execution. The Q4 convolution coefficient projection splits
eight rows into two groups of four. Tests compare complete outputs with the
existing scalar/ungrouped controls. Final target replay and all 270 draft
trace files remain byte-identical.

Rejected experiments remain evidence, not production routes: symmetric-format
correction removal changed bits; 32-row BF16 injection, Q8 token grouping,
residual/RMSNorm fusion and Q5/IQ4 prefill epilogues failed performance checks.
The prefill epilogues won isolated probes but did not produce a consistent
model gain. Their implementations and switches are absent. Maintained GEMM,
attention and recurrence tools call production kernels; the duplicate standalone
W8A8 benchmark is removed. No new test executable is introduced.
