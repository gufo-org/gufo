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
| `model` | Target full-logit replay at verification widths 3–8; MTP committed-feature alignment; DFlash loading, ring/snapshot/restore; prompt/chat/bench parity, seeded multi-turn replay and HTTP adapter sampling. |
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

The unused in-tree CPU DFlash forward pipeline and its duplicate operator
tests are removed. The pinned upstream runner owns the full reference.
All three draft artifacts pass loading and state tests. The optimized
implementation preserves all 270 trace files (90 per draft) byte for byte,
including intermediate layers, logits and sampled proposal distributions.
A long-injection control compares discarded-chunk execution with incremental
injection; the complete serialized history must remain byte-identical.

## Current evidence

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
  each, plus widths 3–7 from the first snapshot. Repeated prefill and snapshot
  continuation are byte-identical. All 49 batched-verifier logit rows per
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
