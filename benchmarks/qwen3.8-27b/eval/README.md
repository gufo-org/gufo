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
| `fast` | NPU packing, GGUF reference decoding and strict result reporting. |
| `kernels` | Quantized GEMM versus independent/decode controls; DFlash convolution, windowed attention, full-vocabulary top-k, sampled selector and verifier distributions. |
| `model` | Target full-logit replay; MTP committed-feature alignment; DFlash loading, ring/snapshot/restore; prompt and multi-turn GPU chat token-ID parity across AR/MTP/DFlash. |
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
The unused in-tree CPU DFlash forward pipeline and its duplicate operator
tests are removed. The pinned upstream runner owns the full reference.
All three draft artifacts pass loading and state tests after this removal;
all 90 Q4 trace files remain byte-identical to the qualified implementation.

## Current evidence

- Q4 and Q8: three fixed 24-token prefixes, four forced continuation tokens
  each. Repeated prefill and snapshot continuation are byte-identical. All
  twelve batched-verifier logit rows per target equal scalar decode byte for
  byte; rejected-prefix replay also preserves subsequent logits. Prefill and
  scalar decode retain the same top-1 choice on these prefixes.
- MTP: feedback replay equals a fresh teacher-forced committed prefix. The
  regression fails when replay uses the newest target hidden row for the old
  proposal anchor.
- DFlash2: FP32 history is bounded by its 2,048-token attention window: 80 MiB
  at a 262,144-token logical context. Ring wrap, persistent restore and replay
  retain proposals and confidence values. Scratch ingestion is bounded to
  256 rows.
- DFlash operators: independent equations cover both convolution coefficient
  planes, attention window boundaries and partial-top-k partitions. Top-k
  1, 7 and 16 run through the full 248,320-token vocabulary, including poisoned
  scratch, zero random draws, zero-mass candidates and tiny temperatures.
- Release companion comparison: both targets × Q4/Q8/BF16 drafts × three
  chat prompts: all 18 cases match all 128 target token IDs.
  See [the measurement record](draft-selection.json). This is a development
  corpus, not a capability evaluation or proof across arbitrary contexts.
- `prompt` and `chat` share GPU/speculative setup. Their first-turn tokens
  match, and two chat turns reproduce AR with both draft backends. Verbose
  generation emits a SHA-256 over little-endian token IDs; the corpus rejects
  missing/incomplete traces and text-only matches.
- Removed an unused 170 MiB target scratch allocation per session. Inactive
  fusion/prefetch branches and their kernels/tests are deleted; production
  arithmetic and independent operator controls remain.

Run the affected operator check first, then model replay on both target quants.
Require complete speculative corpus results before comparing release speed.
Compare token IDs and full logits, not just decoded text or acceptance.
Use `tools/qwen27b/drafts.py` for matched companion comparisons; optional
`--draft-bf16` adds the reference. Do not rank acceptance across different
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
  match AR exactly.

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

Experiment: removing symmetric-format activation corrections, including explicit
rounded multiply/add, failed bitwise GEMM verification. Rejected; the production
reduction remains unchanged.

Retained: FP32 activations for small BF16-weight draft GEMMs reduce the worst
BF16 forward-stage relative error from 5.25e-3 to 6.16e-6 against FP32 upstream
formulas. Short release probes retain exact greedy output; BF16 companion
rates improve from 23.60/22.43 to 24.43/22.97 tok/s on Q4/Q8 targets.

The [TG profile](draft-profile.json) separates executors by HIP stream.
Target work takes 90% of Q4/Q8 companion kernel time, primarily exact quantized
projections. Draft GPU work per step is 17.2 ms (Q4), 18.2 ms (Q8) and 26.1 ms
(BF16); target work is approximately 155 ms. These are one profiled C++ corpus
case per companion, with initial prefill excluded.
