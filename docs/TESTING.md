# Testing and Regression Control

Status: design draft, 2026-08-11

## Purpose

Strix Engine is math-heavy, stateful, and hardware-specific. A change can
compile, launch, produce plausible text, and benchmark faster while still
corrupting logits, KV state, routing decisions, or speculative commits.

The testing rule is:

> Correctness and declared quality budgets pass before performance results
> count.

The framework is influenced by the full-vocabulary oracles and session tests in
`DwarfStar`, the quantizer and backend gates in `ROCmFPX`, and the
CPU-oracle-first discipline in `hipEngine`.

## Reference Terminology

Use precise names in reports:

- Source checkpoint: the exact safetensors revision from which a model is
  converted.
- Full-quality teacher: the source checkpoint executed in its original
  BF16, FP16, or FP32 storage type using the pinned reference runtime.
- High-precision CPU oracle: an analytic implementation using FP32, FP64, or
  wider integer accumulation for small fixtures.
- Canonical backend: the currently promoted Strix implementation for a given
  model artifact and execution route.
- Candidate: the implementation or quantization under test.
- Golden artifact: a retained output produced by an identified oracle.

Do not call a BF16 source checkpoint "FP32". Reports must record the source
storage dtype and the accumulation contract separately.

## Regression Classes

Different changes require different comparison rules.

| Class | Comparison | Default requirement |
| --- | --- | --- |
| Refactor with unchanged arithmetic | Candidate vs canonical backend | Bit exact at declared boundaries |
| New kernel with identical operation order | Candidate vs CPU or canonical oracle | Bit exact where representable |
| Reassociated floating-point math | Candidate vs high-precision oracle | Declared `atol`/`rtol`, plus logit gate |
| Quantization change | Quantized candidate vs full-quality teacher | Distribution, perplexity, and task gates |
| Scheduler or batching change | Batched route vs isolated route | Same request-visible results and state |
| GPU/NPU routing change | New route vs unsplit route | Same route contract and committed state |
| Performance optimization | Candidate vs promoted baseline | Correctness and release quality floor, then measured end-to-end win |

Exactness is required when the implementation claims the same arithmetic
contract. A tolerance must not be introduced merely to admit a failing
candidate.

Quantized weights are intentionally approximate. They are not expected to
produce exact teacher logits. Their quality is evaluated using matched-token
probability distributions and downstream behavior.

## Oracle Hierarchy

Use the most independent practical oracle:

1. Hand-checkable analytic or high-precision CPU implementation.
2. Full-quality source checkpoint in a pinned external reference runtime.
3. Existing promoted Strix implementation with the same arithmetic contract.
4. Small committed golden fixture generated from one of the above.

A new HIP or AIE kernel must not be its own oracle. Cross-device agreement is
useful, but two backends agreeing does not prove that both are correct.

Every golden artifact records:

- Source model repository and immutable revision.
- Source file hashes.
- Tokenizer files and chat-template hash.
- Prompt-suite and token-stream hash.
- Oracle runtime and version.
- Storage and accumulation dtypes.
- Generation or teacher-forcing parameters.
- Artifact schema version.
- Command or structured invocation used to create it.

Updating a golden and changing the implementation in the same review requires
explicit justification. CI must never regenerate a golden automatically.

## Full-Logit Testing

### Why full logits

Matching generated text or the top token is insufficient. Large changes in the
probability tail can leave greedy output unchanged while damaging sampling,
reasoning, speculative acceptance, or later tokens.

For quality-sensitive tests, retain or recompute the complete vocabulary logit
vector at each selected position. Hosted APIs that expose only a small top-k
subset are diagnostic tools, not full-quality oracles.

### Matched token history

Quantization and backend comparisons must be teacher forced:

1. Tokenize once with the pinned tokenizer and template.
2. Execute the teacher and candidate with the same input token prefix.
3. Compare their complete next-token distributions.
4. Append the predetermined evaluation token, not each model's sampled token.
5. Repeat for every scored position.

This isolates intrinsic model error. Once two free-running generations choose
different tokens, later logit differences include trajectory divergence and
must not be reported as quantization-only error.

A separate free-running suite remains necessary for product behavior, but its
results are classified independently.

### Metrics

Given teacher logits `z_t` and candidate logits `z_c`:

```text
p_t = softmax(z_t)
p_c = softmax(z_c)
KL = sum_v p_t[v] * (log(p_t[v]) - log(p_c[v]))
NLL = -log(p_c[target_token])
```

Required aggregates:

- Mean, median, p95, p99, p99.9, and maximum KL divergence.
- Teacher-token NLL and corpus perplexity.
- Top-1 agreement.
- Top-5 and top-k set overlap.
- Candidate probability assigned to the teacher top-1 token.
- Rank displacement of the teacher top-1 token.
- Maximum and RMS difference of normalized log probabilities.
- Count of non-finite values.

Raw-logit maximum error is retained for debugging, but normalized log
probabilities are the quality contract because adding one constant to every
logit does not change the distribution.

For MoE models also record:

- Router-logit KL.
- Router top-k expert agreement and order agreement.
- Expert load distribution.
- First layer and token where selected experts differ.

For speculative systems also record:

- Proposed tokens and proposal probabilities.
- Acceptance count and accepted-length distribution.
- Rejected-token fallback behavior.
- Target verifier logits.
- Direct commit, replay, and fallback counts.
- Committed KV and recurrent-state agreement.

### Logit artifact format

Large logits are stored outside Git in a content-addressed artifact directory:

```text
<artifact-id>/
  manifest.json
  tokens.u32
  positions.json
  logits-00000.f32.zst
  logits-00001.f32.zst
  metrics.json
  checksums.sha256
```

Files are chunked by position so a failed position can be inspected without
loading the entire corpus. `manifest.json` identifies vocabulary order and
tokenizer hash. The test runner rejects mismatched vocabularies instead of
comparing arrays by position blindly.

Small hand-checkable tensors may be committed under `tests/fixtures/`. Full
model weights, raw profiler traces, and large logit dumps must not be committed.

## Quality Thresholds

Thresholds are defined per:

```text
(model revision, quantized artifact, context regime, capability suite)
```

There is no universal KL or top-1 threshold that is valid for every model.
Initial thresholds are established by measuring:

- The full-quality teacher against itself across supported reference routes.
- The first accepted quantized release.
- Repeat-run and machine-to-machine noise.
- Known higher- and lower-quality quantization candidates.

Each model implementation then declares release budgets for mean and tail KL,
perplexity delta, top-token agreement, router agreement, and task scores.
Capability-suite composition, grading, and task-gate policy are defined in
EVAL.md. A new
candidate must satisfy the absolute quality floor and must not regress the
currently published artifact beyond its declared budget.

Calibration prompts, quantizer-search prompts, and held-out evaluation prompts
must be disjoint. Tuning against the held-out suite invalidates it as a release
gate.

## Test Pyramid

### T0: Static and format tests

Run on every change:

- Manifest and schema validation.
- Tensor name, dtype, shape, offset, alignment, and checksum checks.
- Safetensors and Strix-container corruption rejection.
- Unsupported layout and architecture rejection.
- Deterministic model-kind and model-local dispatch resolution.

### T1: Quantization and CPU primitives

- Exhaustive four-bit decode values.
- Scale, zero-point, saturation, NaN, and infinity handling.
- Quantize/dequantize against hand-checkable fixtures.
- Weighted and unweighted scale-search behavior.
- Imatrix import and dimension validation.
- Repeated conversion produces byte-identical output.
- Wide CPU accumulation for integer paths.
- Odd dimensions, padding, and partial groups.

For a weighted quantizer, include a fixture where imatrix weighting changes the
selected scale or zero point and improves the weighted error.

### T2: Device kernel tests

Every HIP and AIE kernel is compared with the CPU oracle over:

- Minimum, typical, and boundary shapes.
- Non-power-of-two and padded dimensions.
- Single-row decode and multi-row prefill shapes.
- G32 and G64 quantization groups.
- Symmetric and asymmetric variants where supported.
- Empty, masked, tail, and cancellation cases where applicable.

Tests report the first mismatching index, shape, kernel variant, maximum error,
and non-finite count. A launch-only smoke test is not a numerical test.

### T3: Layer and state-boundary tests

Capture and compare selected boundaries:

- Embedding output.
- Normalization output.
- Q, K, V projections and rotary application.
- Attention output.
- FFN or expert output.
- Router logits and selected experts.
- Residual stream after each layer.
- Final normalized hidden state and LM-head logits.
- KV pages and model-specific recurrent state.

Exact contracts use byte comparisons. Reassociated or quantized contracts use
declared numerical metrics. The report identifies the first divergent layer
and component.

### T4: Full-model deterministic tests

Use a small multi-category prompt suite covering:

- Plain completion and chat templates.
- Code.
- Tool calls.
- JSON and structured output.
- Reasoning and mathematics.
- Multilingual text.
- Short and long contexts.
- Repeated-token and boundary-length stress cases.

Required route comparisons include:

- Prefill followed by decode vs token-serial recomputation.
- Isolated request vs continuously batched request.
- Original row order vs reversed row order.
- GPU-only vs NPU-only where both support the operation.
- GPU/NPU split vs unsplit execution.
- Direct engine call vs HTTP server.
- Eager dispatch vs graph replay.

### T5: Quantized-model quality

Run teacher-forced full-logit evaluation (BENCHMARKS.md), perplexity, and
capability suites (EVAL.md) against the source checkpoint. Report both absolute quality and delta from the
currently promoted quantization.

A quantization release cannot be promoted from reconstruction error alone.
Tensor MSE and layer similarity guide the search; full-model logits,
perplexity, and task behavior decide promotion.

### T6: Concurrency, lifecycle, and persistence

Exercise:

- One, two, and many simultaneous requests.
- Mixed prefill, decode, cancellation, and timeout in one scheduler step.
- Admission control under KV pressure.
- Prefix sharing with independent continuation.
- Cancellation during GPU, NPU, and split work.
- Session reset and slot reuse.
- KV snapshot save, load, corruption rejection, and model mismatch.
- Server disconnect while generation is in flight.

Every completed request is compared with the same request executed alone under
the same numerical route. Request IDs, row mappings, and KV ownership must not
leak between sessions.

### T7: End-to-end performance

Performance is tested only after the required T0-T6 gates pass.

Retain:

- Time to first token.
- Inter-token latency distribution.
- Prefill tokens per second.
- Decode tokens per second.
- Aggregate throughput at multiple concurrency levels.
- Per-request latency and fairness.
- GPU, NPU, CPU, memory, and synchronization utilization.
- Peak resident memory and KV capacity.
- Speculative acceptance and effective committed tokens per second.

Every performance artifact identifies the exact model, quantization, prompt
tokens, output tokens, concurrency, route, hardware, clocks, firmware, kernel
driver, ROCm version, compiler, build revision, warmup, and repetitions.

Microbenchmarks guide kernel work. They cannot promote a default route unless
the corresponding end-to-end workload also improves.

## Change Gate Matrix

| Change | Required minimum gate |
| --- | --- |
| Documentation only | Link and schema-example checks |
| Container or loader | T0, targeted T1, malformed input |
| Quantizer algorithm | T1, T3, T5 |
| Quantization recipe | T3, T5, memory and speed report |
| HIP or AIE kernel | T1, T2, affected T3/T4 route |
| GPU/NPU partitioning | T2, T4 split parity, T6 cancellation |
| KV layout or attention | T2, T3 KV boundaries, T4 long context, T6 snapshots |
| Scheduler | T4 isolated parity, T6 concurrency and fairness |
| Speculative decoding | T3 state, T4 logits, proposal/acceptance/commit suite |
| HTTP server | Direct-vs-HTTP T4 and protocol tests |
| Release | All applicable tiers plus T7 |

## Pull Request and Release Suites

### Pull request

- CPU-only T0 and T1.
- Targeted device kernels for touched components.
- Small deterministic full-model route suite.
- Direct-vs-server API smoke.
- No golden regeneration.

### Quantization candidate

- Deterministic reconversion from the pinned safetensors source.
- Tensor and layer error reports.
- Full-logit and perplexity suite.
- Capability and long-context suite.
- GPU-only, NPU-only, and shared-route validation.
- Model-card and artifact-manifest generation.

### Release

- Full HIP and XDNA kernel matrix on Strix Halo.
- Multi-request and mixed-route stress.
- Teacher-forced full-logit suite.
- Perplexity corpus.
- KV snapshot matrix.
- Single-user and concurrent-request performance guards.
- Full held-out quality suite.
- Extended concurrency and soak tests.
- Clean-machine install and model download.
- Published artifact checksum verification.
- API compatibility suite.

Release and quantization-candidate suites are started explicitly when preparing
an engine or model artifact. They are not run on an unattended schedule.

## Baseline and Artifact Management

Baselines are immutable records, not files overwritten by the latest run.

Each result is keyed by:

```text
model revision
quantization artifact hash
engine revision
backend route
hardware and software fingerprint
test-suite revision
```

Promotion creates a new baseline pointer after review. It does not delete the
previous result. A result produced with a different prompt set, wall-time
scope, token count, or route is not a valid performance denominator.

Reports use structured JSON as the source of truth. Markdown summaries are
generated views. Failed and rejected experiments may be retained so the same
idea is not repeatedly rediscovered.

## Determinism and Flakiness

- Pin random seeds, tokenizer, template, sampling configuration, and token
  streams.
- Disable sampling for numerical gates unless sampling itself is under test.
- Record deterministic and non-deterministic backend modes separately.
- Repeat a failure before classifying it as timing noise.
- Never average away a numerical mismatch.
- Quarantine is permitted only with an owner, failure record, and expiry.
- A hardware reset or driver failure is infrastructure failure, not a pass.

## Performance Method

- Compare baseline and candidate in the same session where practical.
- Alternate execution order to reduce clock bias.
- Warm both paths before measurement.
- Use the same prompt tokens, output count, KV state, batching, and route.
- Report median and tail latency, not only the best run.
- Retain raw per-repetition values.
- Measure single-request latency before aggregate throughput.
- Reject benchmark-only branches keyed to known prompts or token IDs.

An optimization is promoted only when its relevant end-to-end metric improves
without violating correctness, quality, latency, memory, or fairness budgets.

## Proposed Test Utilities

The initial command set should be:

```text
strix-test-fixtures   Run CPU and small deterministic fixtures
strix-test-kernel     Compare one HIP or AIE kernel with a CPU oracle
strix-capture         Capture layer boundaries or full logits
strix-compare         Compare exact, tolerant, or distribution artifacts
strix-quality         Run teacher-forced logits, perplexity, and task suites
                      (task suites defined in EVAL.md)
strix-stress          Run concurrency, cancellation, and lifecycle tests
strix bench    Produce correctness-linked performance artifacts
strix-report          Validate JSON artifacts and render summaries
```

All utilities support machine-readable JSON output and return nonzero status
when a declared gate fails. Proposed command interfaces are contracts for later
implementation; they do not imply that the tools already exist.

## Initial Repository Layout

```text
tests/
  fixtures/
    cpu/
    formats/
    api/
  models/
    <model>/
      prompts/
      expected/
      thresholds.json
  kernels/
  integration/
  concurrency/
  quality/
  performance/
tools/
  testing/
artifacts/
  .gitignore
```

Model implementations own their thresholds and model-specific suites. The shared
framework owns artifact schemas, comparison math, test discovery, hardware
fingerprinting, and report validation.

## Failure Workflow

When a regression is found:

1. Preserve the failing input and artifact metadata.
2. Minimize it to the earliest divergent layer, token, request, or state page.
3. Add a failing targeted test before changing the implementation where
   practical.
4. Fix the narrowest responsible component.
5. Run the targeted test, then the applicable gate matrix.
6. Retain the correctness and performance artifacts used for promotion.

If a kernel optimization fails logits, state, or quality gates, revert the
candidate behavior and keep the reproducer. A speedup is not partial credit for
incorrect inference.

## Canonical PR Test Command

The canonical PR test command composes all Milestone 0 gates:

```sh
nix build .#checks.x86_64-linux.pr
```

This single command executes:
1. Format validation with `clang-format` in dry-run error mode.
2. Static analysis with `clang-tidy` against the compilation database.
3. Dependency inventory consistency against `THIRD_PARTY_NOTICES.md`.
4. Documentation, local links, anchors, and fenced JSON syntax validation.
5. CTest test suite execution.

The five checks are independent Nix derivations, so Nix can build them in
parallel and reuse their cached results. Static analysis deduplicates the
compilation database and runs up to eight `clang-tidy` workers. The production
package hashes only production sources and builds with `BUILD_TESTING=OFF`;
CTest compiles and runs from a separate test derivation.

The production package remains a dependency of the PR gate. Its build and
install checks execute `strix --version` and `--help`.

## Hardware Test Tiers

The complete HIP/XRT suite is retained, but it is not the default inner-loop
command. Run the smallest tier that covers the ownership boundary changed:

| Change | Required hardware tests |
| --- | --- |
| Repository-owned code without model or accelerator changes | `ctest --preset hardware-fast` |
| DeepSeek graph, kernels, state, or server adapter | `ctest --preset hardware-fast` and `ctest --preset deepseek-gpu` |
| Qwen compute kernels | `ctest --preset hardware-fast` and `ctest --preset qwen-gpu-kernel-oracle` |
| XRT/AIE program or XDNA2 runtime | `ctest --preset hardware-fast` and `ctest --preset xdna2-programs` |
| Shared allocator, dispatch, toolchain, or cross-model runtime | `ctest --preset hardware-full` |

`hardware-fast` excludes tests labeled `slow` or `external-model`. The model presets
require their documented model environment variables. The comprehensive Qwen
kernel oracle and real-model DeepSeek tests remain mandatory when their owned
implementation changes, but unrelated model/device suites need not run on
every iteration.

Run the complete `hardware-full` preset before merging shared runtime or toolchain
changes, and periodically as a scheduled/manual retention gate. This preserves
cross-system coverage without charging every model-private edit for every
other model and accelerator.
