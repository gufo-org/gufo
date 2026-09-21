# Qwen3.8 Flash-Next evaluation

State qualification (2026-09-21): AR/MTP image conversations passed greedy and
seeded cancellation/retry, concurrent image/text isolation, and disk reload into
a fresh backend. The direct snapshot test passed at 4,095 tokens, including
mid-decode save, ring wrap, pending predictor state and RNG replay. Snapshots
retain live KV, unpooled indexer rows, pooled keys, recurrence/PLE history and
the required MTP frontier; temporary verifier scratch is excluded.

The predictor uses one RMSNorm over all 10,240 hidden values, a shared
2,560-wide hidden projection per HC branch, and one embedding projection
broadcast to the four branches. It embeds shifted text token IDs; visual
information comes from the target hidden stream and mRoPE positions.

Source pins:

- [vLLM AMD predictor, 751f6807](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py):
  full-width normalization, separate projections, recursive pre-mixer carry
  and text-only inputs.
- [vLLM proposer, same revision](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/v1/spec_decode/llm_base_proposer.py):
  shifts token IDs while retaining target positions; disables external
  multimodal embeddings for draft classes that do not support them.
- [SGLang predictor, 993d1fcc](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py):
  the same normalization/fusion; also permits supplied multimodal embeddings.
  Gufo follows the text-only vLLM input path.
- Official checkpoint `Qwen/Qwen3.8-Flash-Next`,
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`: its safetensors index contains
  `mtp.fc_embedding`, `mtp.fc_hidden`, `mtp.pre_fc_norm_embedding` and
  `mtp.pre_fc_norm_hidden`.

The GGUF stores `[fc_embedding | fc_hidden]` in each projection row. Loading
splits those encoded rows without dequantization or requantization.
The target's original Q8 output head scores every vocabulary row. There is no
private Q4 head or fixed vocabulary subset. Greedy proposals use full-head
argmax. Sampled proposals use its exact top 64 logits and Gufo's bounded
proposal policy; this does **not** claim upstream draft-sampler equivalence.
Full-vocabulary target verification uses the same FP64 filtered distribution
as AR, with p/q acceptance and residual correction. Boundary tests cover
top-p/min-p support, acceptance, residual draws and seeded replay. Target RNG
draws retain 53 bits; compact proposal masses remain exact multiples of 2^-24.

## Maintained checks

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The audit uses scalar CPU operators and the original GGUF weights, independently
of device repacking and HIP kernels. Its reference library is built only by the
explicit probes, outside the production/default test build. It checks:

- Exact encoded weight bytes after splitting; adversarial unequal HC scales
  distinguish full-width RMSNorm from branch-wise normalization.
- Normalized hidden input, fusion, attention, MoE residual and head mixer,
  with independently computed attention caches and recursive input carry.
- Text and image-pad IDs, image mRoPE positions and continuation beyond the
  image. An attached image must not cause predictor-side vision encoding.
- Independent batched versus serial predictor bodies and heads, including
  recursive chains. Candidate IDs/scores match host sorting of the full Q8
  head; operator tests separately check Q8 dots against FP64.
- Full predictor versus headless catch-up, including FFN input/output and the
  next recursive step across the sparse-attention boundary. Use `--batch 2048`
  to cover 224/257/2047/2048 rows: the minimum final tile, a two-tile tail,
  and large ragged/aligned chunks. Candidates and recorded stages match exactly.
- Short batched catch-up at 9/10/16 total rows against independent full
  predictor execution, including the scalar tail. Full-head candidates and
  subsequent recursive stages must match exactly.

Stage comparisons supply the same recorded input to each CPU/GPU stage and
emulate Q8 activation/F16 cache storage on the CPU. This separates operation
errors from accumulated quantization and MoE routing changes. The scalar
`ReferenceModel::MtpStep` also supports uninterrupted float32 execution;
the stage audit is not an end-to-end float32 equivalence claim.

Norm tolerance is 2e-6 relative RMS; fusion/attention 0.002; MoE/head 5e-5.
The initial eight-state text/image audit passed, with maximum fusion/attention
relative RMS below 0.0008. Full-model session tests additionally cover
C2/C4/C6/C8, ragged budgets, sampled acceptance/rejection and cache restoration;
selector/attention operator tests cover sparse deep contexts.

Current optimization qualification (2026-09-20, shared-expert rows 2026-09-21):

- The shared expert's SwiGLU rows for its F16 down projection live in a
  private buffer, so the routed experts read the token rows the router
  narrowed instead of a second narrowing pass. Every GEMM input is byte for
  byte the same: greedy pp2048 output, the 4096-token chunk-boundary logits
  (`--prefill-only`) and the C2/C4/C6/C8 batch checks (`--batch-only`) are
  unchanged.
- Projection checks retain exact scalar/batch FP32 output through 64 input and
  640 expert-down rows, including mixed activation scales, duplicate/inactive
  experts, nonfinite scales and output guards. Ragged Q8 inputs end at the
  allocation boundary. Captured model inputs also check FP32 contraction.
- Batched Q4 gate/up uses wave64 with independent 32-lane reductions.
  Shared, disjoint and mixed routing retain exact outputs, including small
  unsaturated activations and captured model inputs. C1 keeps its vector path.
- Q8 matrix verification retains the vector path's four K8 partials, rounded
  products, FMA chain and reduction order. Complete FP32 outputs match through
  48 inputs for 12,289/65,537-row matrices and 32 inputs for the 2,561-row
  control, including partial input waves.
- Routed Q8 checks repeat identical activations across column minitiles,
  reordered expert slots and a ragged final tile. FP64 dots use unambiguous
  activation codes, isolating accumulation from quantizer tie semantics.
- Selector load scheduling preserves exact FP32 scores and top-k masks through
  d128K, including ties, tight strides and replay. Independent FP64 score
  error remains below 1e-5; full prefill logits match across chunk boundaries
  through 4096 tokens.
- Full `--batch-only` checks retain exact logits, tokens, acceptance, residual
  draws and RNG at C2/C4/C6/C8, with independent state, every 1–8-token rollback
  prefix, cancellation/recovery and image restoration. A frozen peer's
  snapshot is copied during decode graph capture; snapshot bytes and
  capture/replay logits remain exact.
- Compact GDN rollback saves one full state and exact FP32 keys, decay, beta
  and error values. A host FMA oracle and fresh-prefix GPU execution check
  every retained prefix, with guarded, independently addressed allocations.
  Scalar and batched output/state agree; original and optimized kernels also
  match byte for byte in a separate ablation. Recording intermediate values
  must preserve the original rounded-product and FMA contraction order.
- All ten fresh C1/C2/C4/C6/C8 repetitive/mixed serving cohorts match AR
  completion hashes with zero cache hits.
- Scheduler tests check that decode time covers measured model work even
  when asynchronous prompt capture overlaps other requests. AR and MTP time
  selection/execution directly, excluding capture and waits between steps.
- Vision softmax checks match every probability byte over 256–16384 patch
  rows. The 4096-patch attention specialization retains complete QK/PV FP32
  output. Full Flash-Next 256×256/1024×1024 and Qwen27B 1024×1024 embeddings
  match byte for byte; a 736×736 ragged control also matches.

**Open vision parity gap:** a 1024×1024 synthetic texture produces 6.47%
embedding relative L2 error against the pinned BF16 reference, above the 5%
gate. This also occurs before the retained byte-exact optimizations. Against
the FP32 control, native and upstream BF16 errors are 7.83% and 8.10%; the
discrepancy alone does not establish worse model quality. Cumulative rounding
needs investigation; keep the existing gate. Reproduce with RGB byte
`pixels[i] = (137*i + 53*(i//3072)) % 256` and the vision reference commands in
[Qwen27B evaluation](../qwen3.8-27b/EVALUATION.md#vision).

**Remaining limit:** these checks use converted GGUF weights. They do not
establish unquantized-checkpoint equivalence or detect every conversion error.
Pinned Transformers ignores the MTP weights, so its trunk forward is not an
MTP oracle.

## Session, prefill and serving checks

All eight greedy depth points and all fresh-server cohorts match AR. The
2176-token scalar/prefill control retains the same top-1 token (logit RMSE 0.18).

Tests/probes live in `tests/models/qwen38_flash_next/`. The
`qwen38_flash_next_tests` target contains 14 focused operator/configuration/I/O
checks. Build only affected targets during iteration:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_select_ops_test
nix develop -c ctest --test-dir build/gpu-test -R 'qwen38_flash_next[.]select_ops' \
  --no-tests=error --output-on-failure
```

- **Operators and loading:** independent scalar/FP64 references, exact FP32
  selector scores, near-tie ranking cases, ragged shapes, guards and replay.
  Malformed compression/mRoPE metadata, unsupported kernel geometry and
  incompatible MTP sidecars fail loading.
- **State and sampling:** `qwen38_flash_next_session_test --batch-only` compares
  tokens/full logits and sampled RNG/acceptance against isolated execution at
  C2/C4/C6/C8, including ragged budgets, reordered requests, bounded rollback,
  cancellation recovery and image attachments. `--sampling-only` covers 23
  AR/MTP configurations, penalties, residual correction and short budgets.
  `qwen38_flash_next_snapshot_test` checks persistence and continuation replay.
  AR sessions sharing an MTP-capable model allocate no predictor state;
  mixed-mode target logits match exactly and snapshots cannot cross modes.
- **Prefill and serving:** `--prefill-only` checks full logits across boundaries
  through 4096 tokens, including 1/8/9/32/33-token tails. Image checks cover
  AR/MTP, concurrency, RAM reuse and disk restoration. Official template and
  Unicode/NFC token goldens cover both Qwen models. HTTP tests require complete
  UTF-8 in every streamed JSON event and schema-correct tool arguments, including
  quoted closing markers and calls after unclosed reasoning.
- **Audits:** `qwen38_flash_next_gpu_probe --mtp-audit` checks original encoded
  weights, full-width normalization, predictor stages with independently
  computed caches, recursive carry and independent text/image batches against
  a scalar CPU oracle. `--cost-audit C` measures catch-up/proposal/verification
  costs: median of three complete warmed cycles, with two warm-ups per shape.
  `0` selects C1/C2/C4/C6/C8; `--depth N` restricts the default d0/d4K/d32K
  calibration for focused iteration. Neither audit stores logit fixtures.

The model tests/probes accept `--model "$MODEL" --mtp-model "$MTP"`; build the
session/snapshot tests with `qwen38_flash_next_model_tests`. Preserve arithmetic
and replay when optimizing layout/fusion. Arithmetic corrections additionally
need independent references and `gufo bench --validate-prefill N`. Measure with
Nix release binaries, matching artifacts, capacity, prompts and sampling;
profile separately using `tools/prof/prof.py run --stages qwen-flash -- ...`.

Sampled MTP and AR share the same FP64 target filtering, normalization and CDF.
Proposal acceptance and residual correction use that canonical distribution;
shared target RNG draws retain 53 bits. Greedy verification stays on the GPU.
Replay requires the same build, seed, request budget, configured capacity and sampling
settings; sampled MTP need not match AR's same-seed sequence. Greedy output must
remain independent of draft width and batching.

The pinned official image processor allows 64–16384 merged tokens; a 1024-token
cap changes resolution and output. Short prefill tails retain the arithmetic of
larger chunks. The artifact uses native RoPE without YaRN extension. Official
Transformers `c587bc884db2c2e31fc2b8102314656b17aa07b1` defines FP32 QSA but ignores
MTP weights; operator/export audits do not close original-model parity.
