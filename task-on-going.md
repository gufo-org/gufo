# HRX integration: current status and recovery plan

Last updated: 2026-08-28 (course-corrected to full-prompt blocked prefill)

## Baseline identity

- Parent revision: `8895a092a718` (`fedeizzo/hrx-integration`)
- Working-copy revision: `f6e352e6dc56` (dirty; experiments described below)
- Hardware fingerprint: `bb565d5eff3a9f23b4ac3f1ff03f66bebef651e57093cd558c4cded823358849`
- Platform: AMD Strix Halo, gfx1151 (20 CUs), XDNA2 (32 AIE tiles)
- Toolchain: ROCm 7.2.3, XRT 2.21.0
- Target model: `models/Qwen3.8-27B-Q8_0.gguf`
- DFlash2 model: `models/Qwen3.8-27B-DFlash2-Q8_0.gguf`
- The shell inherited a stale `LD_PRELOAD` pointing at an obsolete
  `result-hrx/lib/libamdhip64.so`. All measurements explicitly unset it.

## Performance status

There is now a measured HRX decode gain from changing weight placement, but the
experimental implementation is not yet suitable to land because it keeps the
mapped checkpoint and a second device-local copy alive.

| route | test | throughput |
|---|---:|---:|
| HIP reference, current release binary | pp128 | 399.59 t/s |
| HIP reference, current release binary | tg16 | 7.62 t/s |
| HRX native baseline | pp128 | 3.90 t/s |
| HRX native baseline | tg16 | 3.54 t/s |
| HRX with asynchronous operation snapshot | pp128 | 3.90 t/s |
| HRX with asynchronous operation snapshot | tg16 | 3.55 t/s |
| HRX with wave32 vocabulary projection | tg16 | 3.55 t/s |
| HRX with wave256 argmax | tg16 | 3.55 t/s |
| HRX device-local weights, first run | pp1 / tg16 | 4.33 / 4.35 t/s |
| HRX device-local weights, repeated run | pp1 / tg16 | 5.19 / 5.20 t/s |
| HRX production loader, mapped A/B arm | pp1 / tg16 | 3.59 / 3.60 t/s |
| HRX production loader, device-local default | pp1 / tg16 | 5.45 / 5.38 t/s |
| HRX int8 prefill, current release binary | pp8 | 25.30 t/s |

The small-kernel HRX variants are within measurement noise and are not wins.
Device-local weight placement is a large, reproducible signal: tg16 improves by
23-47% over the same-binary mapped baseline of 3.53 t/s. The spread between the
two copied-weight runs means more interleaved samples are still required before
reporting one headline number.

The latest same-binary A/B is +49.4% for tg16 (3.60 to 5.38 t/s). A separate
27B `gufo serve` process was resident during these measurements (most recently
about 6.6 GiB RSS), so final qualification must be repeated on an otherwise idle GPU.
The direction and magnitude have reproduced despite that contamination.

Quality validation of the asynchronous snapshot candidate passed at four prompt
and four decode positions: top-1 matched HIP, cosine similarity was 1.0, maximum
absolute logit error was at most 7.63e-6, and RMSE was at most 1.21e-6.

## What is implemented and verified

- The native HRX executor loads the Qwen Q8_0 model and runs prompt/decode
  end-to-end on gfx1151.
- The arena, state-management, layer dispatch, vocabulary projection, sampling,
  and transaction/rollback paths exist.
- State snapshots can now be enqueued on the same stream without an immediate
  host synchronization. Public `SaveState` retains its synchronous contract.
- State-copy dispatch sizes now use the actual buffer element count instead of
  always launching the maximum recurrent-state grid.
- The accidentally corrupted arena lifecycle assertions in
  `qwen_hrx_executor_test.cpp` were restored. The test now checks stream ordering
  around an enqueued snapshot and later restore.
- A release `nix build .#hrx` containing the current experimental tree succeeds.
- A HIP `rocprofv3` baseline was captured in
  `/tmp/gufo-prof-hip-baseline/hip-baseline_results.db`.
- A release `result-hrx-weightcopy` binary supports the temporary
  `GUFO_HRX_WEIGHT_MODE=copy` A/B route. It validates the memory-placement
  theory without changing kernel arithmetic.
- The production-shaped loader now defaults to device-local storage, owns HRX
  buffers through move-safe RAII regions, discards mmap-backed tensor references
  after native bindings are built, and lets the bench command release its GGUF
  reader/mapping. `GUFO_HRX_WEIGHT_MODE=mapped` remains the explicit A/B and
  low-memory fallback route.

## Experiments completed but not retained as performance wins

### Asynchronous transaction snapshot

This removes a host synchronization and avoids over-dispatching the smaller
convolution-state copy. Correctness passes, but pp128 remains 3.90 t/s and tg16
moves only from 3.54 to 3.55 t/s. It is useful infrastructure, not the primary
bottleneck.

### Two-row and wave32 vocabulary projection

The two-row route was neutral/slightly regressive and was removed. A wave32
variant compiles with no spills, 44 VGPRs, 56 SGPRs, and reported 100% occupancy,
but it does not improve the final stage or end-to-end throughput. The experiment
was removed from the current working copy after its rejection.

### Wave256 argmax

The device oracle passes and compilation reports 7 VGPRs, 8 SGPRs, 64 bytes of
LDS, no scratch/private memory, and reported 100% occupancy. End-to-end tg16 and
the final-stage duration are unchanged. Argmax is too small a fraction of token
time to matter. The experiment was removed from the current working copy after
its rejection.

## Profiling evidence

`rocprofv3` works for the HIP backend. A short pp1/tg2 capture contains 6,765
dispatches, 1,105.57 ms of summed GPU time, 1,270.70 ms wall span, and 165.13 ms
(13.0%) idle time in the GPU span.

HIP GPU time is dominated by:

| stage | calls | GPU time | share |
|---|---:|---:|---:|
| blocked W8A8 GEMM | 1,984 | 642.42 ms | 58.1% |
| GEMV | 583 | 327.18 ms | 29.6% |
| SSM other/input projections | 144 | 58.69 ms | 5.3% |
| runtime fill | 19 | 20.22 ms | 1.8% |

The main HIP kernels are fused/blocked multi-output kernels: blocked W8A8 WMMA
GEMMs, two-row fused SwiGLU GEMV, two-row Q8_K GEMV, fused SSM input projections,
and fused QKV projections. Sampling is only 0.1% of HIP GPU time, confirming why
the HRX argmax rewrite could not affect throughput materially.

`rocprofv3` cannot currently profile the native HRX executable. It aborts during
IREE AMDGPU device initialization, before model execution, in this stack:

```text
iree_hsa_executable_freeze
  -> hsa_executable_freeze
  -> ExecutableImpl::Freeze
  -> RegionMemory::Freeze
  -> GpuAgent::InvalidateCodeCaches
  -> AqlQueue::ExecutePM4
  -> abort in hsa_signal_wait_scacquire
```

The failed run is recorded under `/tmp/gufo-prof-hrx-baseline-2`. Until the
ROCr/rocprof/IREE interaction is fixed, HRX measurements use the executor's
`GUFO_HRX_TRACE_STAGES` device synchronization timing plus
`iree-benchmark-loom` profile replay/counters for isolated production-shaped
kernels. The baseline HRX token is about 280-283 ms: FFN is about 168-170 ms,
SSM about 69.5 ms, attention about 17.7 ms, and the final stage about 24.5 ms.

## The core problem

The HRX path is structurally a correctness executor, not yet a performant model
executor. It serializes a very large number of narrow, one-token operations and
does not use the policy, batching, fused projection, or Q8_K XL paths described
by the PR plan. Its approximately 3.9 pp128 result is expected because prompt
processing is literally a loop of 128 decode-like token executions; it is not
real prefill.

Specific gaps found in the current code:

- `ForwardPromptBatch` loops over tokens and calls the single-token executor.
- `--hrx-fusions` is parsed and stored, but executor dispatch selection never
  reads the policy.
- The native kernels do not match the fused, blocked HIP kernel topology that
  accounts for almost 90% of the HIP profile.
- A decode operation still copies roughly 202 MiB of recurrent state for
  rollback. Removing the immediate host wait did not remove the copy itself.
- Native matrix binding is strict Q8_0 only. Q8_K_XL mixed Q8_K/BF16 weights and
  the required conversion/binding logic are missing.
- There is no genuine batch arena or batch-shaped native primitive set.
- DFlash2-on-HRX GPU integration is not complete.
- DFlash2-on-NPU is still a stub: initialization and proposal generation return
  without executing an XDNA graph.
- No shared GPU/NPU BO ownership and synchronization contract exists for the
  DFlash2 handoff.
- The canonical PR check does not compile the HRX executor test: normal checks
  build without `ENGINE_ENABLE_HRX`, while `.#hrx` sets `BUILD_TESTING=OFF`.
  This allowed a syntactically corrupted HRX test body to remain unnoticed.

## Ranked theories

1. **Serial prefill is the pp bottleneck.** A token loop mathematically caps
   pp128 near decode throughput. A real `[M,K] x [K,N]` prefill route with batched
   state/attention primitives is required before prompt performance can approach
   HIP's roughly 400-500 t/s.
2. **Kernel topology, not argmax, controls decode.** HIP spends 87.7% of GPU time
   in blocked W8A8 GEMM and GEMV and relies on multi-output/fused kernels. HRX
   needs comparable fused projections and blocked/two-row kernels. Optimizing a
   0.1% sampling stage cannot move the result.
3. **Imported mapped weights are using an unfavorable memory path (validated).**
   HRX imports host-visible/device-visible file-backed GGUF mappings, whereas
   device-local HRX allocations reduce every major compute stage. The temporary
   `GUFO_HRX_WEIGHT_MODE=copy` experiment improves tg16 from 3.53 to 4.35-5.20
   t/s. It is an experiment only until the mapped backing can be released or a
   direct device-local loading path exists.
4. **Rollback state traffic can matter after compute improves.** The 202 MiB
   snapshot is currently hidden by much larger compute time. It should become a
   decode snapshot/journal policy rather than an unconditional full-state copy,
   but it is not the first-order bottleneck at 3.55 t/s.
5. **Q8_K XL is likely required for competitive memory-bandwidth behavior.** The
   current Q8_0-only route cannot exercise the target production quantization or
   reuse the HIP Q8_K fast-path structure.
6. **Launch overhead is secondary but significant.** HIP still shows 13% idle
   time inside its GPU span. HRX's many more narrow dispatches and stage
   synchronizations make fusion/batching important even after kernel arithmetic
   improves.

## What is missing / ordered implementation cards

The cards should be implemented and retained only when their correctness and
performance acceptance criteria pass.

1. **Finish the measurement manifest.** Record exact binaries, model hashes,
   commands, hardware fingerprint, pp/decode baselines, parity thresholds, and
   profiler limitations. Keep one immutable baseline result for every A/B.
2. **Land or reject transaction cleanup.** Keep the asynchronous snapshot only
   as a verified synchronization/launch cleanup; document that it has no current
   throughput gain. Replace unconditional full copies with a decode-aware policy
   only after state rollback semantics are covered by tests.
3. **Implement Q8_K XL model binding.** Support the actual Q8_K/BF16 tensor
   contract, validate shapes/strides/alignment at bind time, and add a small
   deterministic kernel oracle before an end-to-end benchmark.
4. **Implement decode fusion policy.** Make `--hrx-fusions` select real executor
   routes. Start with fused QKV, fused SSM input projections, and fused
   FFN/SwiGLU projection topology based on the HIP profile. Require parity plus a
   statistically credible tg improvement for each retained route.
5. **Add a batch arena and batch-native primitives.** Activations, residuals,
   norm/quantization, QKV/SSM/FFN intermediates, KV writes, and recurrent state
   updates must carry an explicit token dimension without per-token allocation or
   synchronization.
6. **Replace serial prompt execution with chunked prefill.** Run actual batched
   matrix operations and batch-aware attention/state updates. Validate multiple
   prompt lengths and boundary positions against HIP. The first meaningful pp
   gate is a large step above the ~3.9 t/s serial baseline; final qualification
   should target the current HIP range.
7. **Make the target/provider contract neutral.** Separate model execution
   policy from AMDGPU/XDNA provider selection so DFlash2 can share buffers and
   state transitions without backend-specific CLI behavior leaking into model
   code.
8. **Integrate DFlash2 on HRX GPU.** Port/route the already working HIP DFlash2
   stages through HRX, validate draft logits/tokens against HIP, and benchmark
   acceptance rate plus target-side cost.
9. **Define GPU/XDNA shared-buffer ownership.** Use XRT-importable BOs with
   explicit layout, lifetime, producer/consumer fences, cache visibility, and a
   fallback copy path for diagnosis. Do not hide copies behind the API.
10. **Build and execute XDNA2 artifacts.** Produce versioned DFlash2 drafter
    graphs for the real model shapes, load them through XRT, bind shared BOs, run
    proposal generation, and verify output against the GPU oracle.
11. **Qualify end-to-end speculative execution.** Measure pp, target decode,
    draft cost, accepted tokens/step, synchronization/copy overhead, effective
    accepted-token throughput, memory use, and parity/fallback behavior.
12. **Close the HRX test coverage gap.** Ensure a Nix check compiles and runs the
    HRX executor/arena tests, then run `nix build .#checks.x86_64-linux.pr` before
    landing.

## Immediate next action

### Device-local acceptance gate: passed

`nix build .#hrx` at commit `8895a092` produced `result-hrx-gate`. Running
`GUFO_HRX_WEIGHT_MODE=device-local gufo bench --qwen-backend hrx-native
--validate-hrx 8` completed the full HIP-to-device-local comparison that
previously failed with `RESOURCE_EXHAUSTED`. The 64 MiB bounded H2D chunking
removed the transient allocation failure. Result: all four prompt positions and
all eight decode positions match HIP top-1, cosine similarity 1.0, worst
max-absolute error 7.63e-6, worst RMSE 1.21e-6. Model load takes 152 s in
device-local mode.

### Carried forward from the loader work

Mapped HRX parity against HIP passed for four prompt and eight decode
positions (top-1 match, cosine 1.0, worst max-absolute error 7.63e-6, worst
RMSE 1.21e-6). GNU time reported a 50,083,012 KiB maximum RSS during
device-local loading versus 28,325,024 KiB mapped; that is a load-time peak
with the mapping still open, not post-unmap steady state, which still needs a
live sample. Stage-synchronized pp1/tg1 for mapped versus device-local:
attention 17.48 to 10.37 ms, SSM 69.44 to 45.67 ms, FFN 167.11 to 95.83 ms,
final 23.85 to 19.59 ms, whole token 279.79 to 173.43 ms.

### Interleaved weight-placement A/B

Three interleaved samples, same binary, `-p 1 -n 16`, with a 27B `gufo serve`
process resident (about 6.6 GiB RSS):

| sample | mapped pp1 / tg16 | device-local pp1 / tg16 |
|---|---:|---:|
| 1 | 3.59 / 2.83 | 4.18 / 4.17 |
| 2 | 1.05 / 3.57 | 4.23 / 4.24 |
| 3 | 3.50 / 3.50 | 4.12 / 4.19 |

Device-local tg16 mean is 4.20 t/s with a 0.04 t/s spread; mapped tg16 mean is
3.30 t/s with a 0.4 t/s spread. Device-local is both faster (+27% on means,
+20% on medians) and far more stable. The absolute values are below the
5.38 t/s recorded earlier for device-local, so the resident server still
contaminates level comparisons; only same-session A/B deltas should be quoted.

### Corrections to earlier assumptions

- The `Q8_K XL` checkpoint on this machine
  (`models/Qwen3.8-27B-UD-Q8_K_XL.gguf`, a symlink to the `Q8_K_L` file) does
  not contain Q8_K or BF16 tensors. Its 866 tensors are Q8_0 (438), F32 (360),
  Q6_K (57), and Q5_K (11), with 65 blocks and one MTP prediction layer. The
  binding card therefore means mixed Q5_K/Q6_K/Q8_0 support plus the extra
  block, not a Q8_K/BF16 contract.
- The four existing fused artifacts (`qwen_fused_swiglu_bf16`,
  `qwen_fused_rmsnorm_qkv_bf16`, `qwen_fused_down_residual_bf16`,
  `qwen_fused_rope_kv_cache_bf16`) take BF16 weight operands and cannot run
  against the Q8_0 checkpoint. That, not a missing policy lookup alone, is why
  `--hrx-fusions` had no route to select.
- `qwen_q8_0_vocab_gemv_k5120.loom` is byte-identical to
  `qwen_q8_0_gemv_k5120.loom` except for the row-capacity constant (248320 vs
  17408). Wide fused Q8_0 projections can reuse the vocabulary artifact with no
  new kernel.
- Checkpoint tensor adjacency (verified for every layer of the Q8_0 model):
  `ffn_gate` is immediately followed by `ffn_up` in all 64 layers, and
  `ssm_alpha` by `ssm_beta` in all 48 SSM layers. `attn_q`/`attn_k`/`attn_v`
  and `attn_qkv`/`attn_gate` are separated by other tensors, so fusing those
  would require a repack during the device-local copy.

### Chunked prefill (cards 5 and 6): first implementation

Prefill was a literal loop of single-token executions, so pp128 could never
exceed the decode rate. The first real batched route is now implemented.

Three new Loom artifacts decode each Q8_0 weight block once and contract it
against up to eight tokens, so the weight stream is amortized across the chunk
instead of being re-read per token:

| artifact | K | workgroup | VGPR / SGPR | spills | occupancy |
|---|---:|---:|---|---:|---:|
| `qwen_q8_0_gemm_k5120_t8` | 5120 | 160 | 84 / 44 | 0 | 93% |
| `qwen_q8_0_gemm_k6144_t8` | 6144 | 192 | 84 / 44 | 0 | 93% |
| `qwen_q8_0_gemm_k17408_t8` | 17408 | 544 | 84 / 44 | 0 | 81% |

Each takes `(rows, tokens)` scalars with token-major input and output. Short
chunks read zeroed padding rows and stores are guarded by the token count, so
no separate tail kernel is required. All three are optional manifest entries:
when they are absent the executor keeps the sequential route.

The arena gained token-major staging buffers
(`kBatchHidden`, `kBatchNormed`, `kBatchAttentionQGate`, `kBatchAttentionK`,
`kBatchAttentionV`, `kBatchContext`, `kBatchSsmQkv`, `kBatchSsmGate`,
`kBatchSsmAlphaBeta`, `kBatchFfnGateUp`, `kBatchFfnActivation`,
`kBatchProjected`), about 2 MiB in total for an eight-token chunk. The batched
residual stream ping-pongs between `kBatchHidden` and `kBatchNormed` exactly
like the single-token path.

`--hrx-fusions chunked-prefill` selects the route. Per layer it now runs:

- one RMSNorm per token (still per token; a batched norm is a later step),
- one batched GEMV-style projection per weight matrix for the whole chunk,
- the genuinely sequential work per token in order: split Q/gate, per-head
  norms, RoPE plus KV write at the token position, attention decode, and the
  SSM convolution and DeltaNet recurrence, which must observe the state left by
  the previous token,
- one batched output projection, and one residual add per token.

The last chunk's residual is copied back into the single-token `kHidden`
buffer so the final projection and any following decode step keep their
existing contract. Attention context and SSM recurrent output share one batch
buffer because both are exactly 6144 floats wide, which is also the K the
output projections use.

Weight traffic per prefill token drops by up to 8x; the per-token dispatch
count rises slightly for the sequential parts. Correctness is unchanged by
construction: projections never depend on recurrent state, so hoisting them out
of the token loop is order-preserving.

### Decode fusion A/B (two interleaved samples, device-local)

| fusions | pp128 | tg16 |
|---|---:|---:|
| none, sample 1 | 6.25 t/s | 5.44 t/s |
| q8, sample 1 | 6.26 t/s | 5.62 t/s |
| none, sample 2 | 6.27 t/s | 5.49 t/s |
| q8, sample 2 | 6.19 t/s | 5.35 t/s |

The three decode routes are inside run-to-run noise at this contention level
(mean tg16 5.47 unfused versus 5.49 fused). They are retained as verified
launch-count reductions with proven parity, not as a throughput claim. The
dominant decode term is the 28 GiB weight stream per token, so removing 240 of
about 1,170 dispatches cannot move it much.

A same-session HIP baseline was attempted in the same run but produced no
parseable output; it needs to be re-measured on an idle GPU.

## Prefill is the active work item

`pp128` at 6.2 t/s is not a tuning problem. The serial route executes one full
model pass per prompt token, so 128 tokens stream the entire 28 GiB checkpoint
128 times. At roughly 200 GB/s that fixes pp near the decode rate regardless of
kernel quality, which is exactly what the measurements show (pp128 6.2 versus
tg16 5.4). HIP instead runs one blocked W8A8 WMMA GEMM per projection over the
whole chunk, reading each weight once for all tokens and doing the arithmetic
on the matrix cores. That structural difference, not kernel tuning, is the
64x gap.

### Course correction: the eight-token model pass is the largest bottleneck

The T8 route improved the serial baseline, but it cannot approach HIP even with
a perfect inner kernel. `ForwardPromptBatch` slices PP128 into sixteen chunks
and `ForwardPromptChunk` takes each chunk through all 64 layers before starting
the next one. Every large projection consequently streams the approximately
28 GiB checkpoint sixteen times per PP128 operation instead of once.

The lower bound makes the problem unambiguous:

- PP128 with `int8-prefill` is 20.27 t/s, or about 6.32 s;
- sixteen checkpoint passes move roughly 448 GiB of weights;
- even the measured gfx1151 DRAM read ceiling of 241 GB/s puts repeated weight
  traffic alone at about 1.86 s, a maximum of only 68.8 t/s before all other
  work;
- using the approximately 190-209 GB/s sustained rates observed by the model
  and roofline tools lowers that structural ceiling to roughly 54-60 t/s;
- HIP PP128 at 399.59 t/s completes in about 0.32 s, which is only possible
  because its layer-major blocked W8A8 route reuses a weight tile across a much
  larger token tile.

This supersedes the previous immediate focus on tuning a K=5120/T=8 WMMA
kernel. FFN remains the largest measured stage (67.3%), but that is a symptom of
all projections using the narrow topology. The primary card is now to replace
the eight-token, chunk-major model traversal with a layer-major prefill route
whose projection tile is at least PP128-sized.

The production HIP implementation is the starting specification, not a design
to rediscover. Its retained topology is:

- tiled Q8_1 activations in fragment order (16 tokens x 32 K values per tile,
  with scales alongside the tile);
- a blocked W8A8 WMMA GEMM covering 128 output rows x 128 tokens;
- BK=2 staging, 256 threads / eight waves, zero spills, and eight resident waves
  per SIMD on gfx1151;
- row-major Q8_0 weights read directly, since load-time weight repacking was
  measured and rejected on HIP;
- fused quantization epilogues where they remove a material intermediate, only
  after the blocked projection is working.

Ordered implementation direction:

1. Raise the batch arena and native primitive contract from a hard-coded eight
   tokens to a production prefill tile (first PP128; tail support required).
2. Change prompt execution to keep the whole tile live and advance layer by
   layer. Recurrent attention/SSM work remains ordered within each layer; it
   must not force the projection operands back into eight-token model passes.
3. Port the existing HIP tiled-Q8_1 quantizer and 128x128 blocked W8A8 WMMA
   topology into the Loom/HRX artifact contract. Use its measured BK=2/eight-
   wave configuration rather than extending the one-row dot4i kernel.
4. Validate the isolated production shapes first: K=5120 with rows 17408/34816,
   K=17408 with rows 5120, and K=6144 with rows 5120. The gate is correctness
   plus a meaningful fraction of the measured 55.07 TOPS ceiling, not merely a
   win over another narrow HRX kernel.
5. Wire the blocked route for all Q8_0 prefill projections, then measure one
   stage-traced PP128 run. Only after projection topology is competitive should
   attention/SSM batching, fused epilogues, or DFlash2 handoff be optimized.

Acceptance gates for this macro card:

- PP128 performs one model weight pass (apart from deliberate tail tiling), not
  sixteen T8 passes;
- no per-token projection dispatches remain in the prefill path;
- the blocked-kernel oracle matches the documented W8A8 numerical envelope;
- PP128 improves by multiples, not measurement noise; an initial useful gate is
  above the 54-69 t/s structural ceiling of the T8 traversal, followed by
  convergence toward the current 399.59 t/s HIP result;
- decode remains on its separate GEMV path and must not regress.

Planned sequence, each gated on parity against HIP plus a measured pp number:

1. Chunk of 8 with the `_t8` artifacts: implemented. Weight traffic per
   prefill token drops 8x. See the parity-harness gap below: the batched route
   was not actually covered by `--validate-hrx` until the harness was fixed.
2. Widen the chunk to 16 or 32 tokens. A generator emits the `_tN` variants,
   and the T=16 K=5120 kernel already compiles at 129 VGPRs with no spills;
   occupancy drops from 93% to 62% because it becomes VGPR-limited, and the
   report says nine fewer registers would reach the next residency tier. Both
   widths need measuring rather than assuming: halved weight traffic against
   lower occupancy.
3. Batch the remaining per-token elementwise stages: implemented.
   `qwen_rmsnorm_batch_f32` (one workgroup per token row),
   `qwen_residual_add_batch_f32` (capacity 163840 elements), and
   `qwen_swiglu_pointwise_batch_f32`, which understands the interleaved
   gate/up chunk layout the fused projection produces. Each replaces one
   dispatch per token with one dispatch per chunk, removing roughly 5T
   dispatches per layer. This matters because after batching the projections
   the launch overhead of the remaining per-token work is comparable to the
   weight-stream time: about 4,500 dispatches per eight-token chunk against
   roughly 140 ms of unavoidable weight traffic.
4. Move the projections onto the int8 dot path (in progress, see below), and
   after that onto the matrix cores. Loom exposes `vector.mma` with matrix
   fragments, including a quantized fragment form that carries a block scale
   and an encoding schema, so a true WMMA GEMM is expressible.

### Parity harness gap: batched prefill was never validated

`--validate-hrx` drives the candidate through `ValidateHrxStep`, which calls
`ForwardToken` once per position. `ForwardPromptBatch` - and therefore every
chunked and int8 prefill stage - was never entered. Both "passing" runs
reported exactly the serial route's numbers (worst max-absolute error 7.63e-6,
cosine 1.0) because they measured the serial route.

The harness now runs an explicit `prefill-batch` phase first: it feeds the
whole validation prompt through `ForwardPromptBatch`, compares the resulting
logits against the HIP reference for the last prompt position, and only then
runs the existing per-token prompt and decode phases. Prefill parity numbers
below this line come from that phase.

### Measured: batching alone does not pay, int8 does

`--hrx-fusions chunked-prefill` measured pp128 5.53 t/s against 6.25 t/s for
the serial route: the eight-token chunk was slightly *slower* despite reading
each weight once for eight tokens. Isolated kernel benchmarks
(`iree-benchmark-loom`, rows=17408, K=5120, minimum of ten samples) explain
why:

| kernel | time | per token |
|---|---:|---:|
| `qwen_q8_0_gemv_k5120` (T=1) | 11.5 ms | 11.5 ms |
| `qwen_q8_0_gemm_k5120_t2` | 21.7 ms | 10.9 ms |
| `qwen_q8_0_gemm_k5120_t4` | 26.1 ms | 6.5 ms |
| `qwen_q8_0_gemm_k5120_t8` | 53.7 ms | 6.7 ms |
| `qwen_q8_0_gemm_i8_k5120_t8` | 22.4 ms | 2.8 ms |

Cost scales almost linearly with the token count, so the f32 route is bound by
per-lane instruction issue, not by the weight stream. Each lane was doing a
`vector.sitofp` per weight value and then a horizontal `vector.dotf` per token.

Loom exposes `vector.dot4i<s8s8>`, the hardware int8 dot product that HIP's
W8A8 path uses. Keeping both operands in int8 removes every conversion and does
four multiply-accumulates per instruction: the same eight-token chunk drops
from 53.7 ms to 22.4 ms, which is 4.1x better per token than the serial GEMV.
Registers fall from 84 to 65 VGPRs and occupancy stays at 93%.

The int8 prefill route therefore consists of:

- `qwen_activation_quantize_k{5120,6144,17408}`: quantizes a token-major f32
  chunk into int8 with one f32 scale per 32 values, one workgroup per token.
- `qwen_q8_0_gemm_i8_k{5120,6144,17408}_t8`: Q8_0 weights against those int8
  activations through `dot4i`, with the per-block scale product in f32.
- `--hrx-fusions int8-prefill`, which implies `chunked-prefill`.

Activations become int8 in prefill, exactly as in HIP's W8A8 prefill, so the
prefill parity envelope has to be re-derived rather than assumed equal to the
f32 route.

Measured on the fixed harness, `int8-prefill` against the HIP reference for a
four-token prompt: top-1 matches (157 against 157), cosine similarity
0.99986589, RMSE 0.05335, worst max-absolute error 0.32689. That fails the
existing envelope, which requires RMSE at most 1e-4 and cosine at least
0.999999 - thresholds derived from an f32 decode path. Quantizing activations
to int8 with one scale per 32 values cannot meet a 1e-4 RMSE bound, so the
route needs its own documented prefill envelope (top-1 agreement plus a cosine
floor around 0.999) rather than the decode envelope. Whether that is
acceptable is a modelling decision, not a bug: HIP's own prefill is W8A8, but
the reference logits used here come from HIP's per-token path.

### Prefill throughput, device-local weights

| route | pp128 | tg16 |
|---|---:|---:|
| serial (`none`) | 6.25 t/s | 5.44 t/s |
| `chunked-prefill` (f32 `_t8`) | 5.53 t/s | 5.29 t/s |
| `int8-prefill` | 20.27 t/s | 4.88 t/s |

int8 prefill is 3.2x the serial baseline. The remaining gap to the weight
roofline is still large: an eight-token chunk takes about 394 ms while the
28 GiB weight stream at the 190 GB/s the decode path already achieves would
take about 147 ms, so the projections run at roughly 70 GB/s effective.

Two kernel variants were tried and rejected on measurement:

- T=16 int8 (95 VGPRs, no spills, 93% occupancy) measured 198 ms against
  22.4 ms for T=8 on the same shape.
- A packed activation layout that replaces eight scalar loads per quarter with
  one `vector<8xi32>` load (48 VGPRs, 33 global loads instead of 81) measured
  119 ms against the same 22.4 ms.

Neither register pressure nor instruction count predicted the result, so the
next step was stage-level measurement inside the chunk rather than more kernel
variants. `GUFO_HRX_TRACE_STAGES` now also instruments the chunk path and
reports per-chunk attention, SSM, and FFN milliseconds.

### Int8-prefill stage profile after the rebase

The rebased working copy builds with `nix build .#hrx`. A release-binary run of
`int8-prefill` at pp8, with the separate 27B server still resident, measured
25.30 t/s and reported:

| stage | time | chunk share |
|---|---:|---:|
| attention | 21.15 ms | 7.2% |
| SSM | 75.40 ms | 25.5% |
| FFN | 198.89 ms | 67.3% |
| whole traced chunk | 295.43 ms | 100% |

The next optimization target is therefore the FFN Q8_0-by-int8 projections,
not another end-to-end policy variant. In particular, the current dot4i kernel
assigns one workgroup to one output row and reloads the same activation chunk
for every row. The active experiment is an isolated gfx11 WMMA tile that reuses
one activation tile across 16 output rows. It must first beat the existing
K=5120/T=8 kernel with an oracle and standalone benchmark before it is wired
into the executor.

That isolated gate now passes. The signed-int8 fragment mapping first passed a
16x16 exact device oracle. The full K=5120/T=8 kernel then passed a nonzero
production-layout oracle using the exact 34-byte Q8_0 blocks (f16 scale plus 32
signed bytes). It compiles to two WMMA instructions per Q8 block with 40 VGPRs,
24 SGPRs, 1 KiB LDS, no spills, and reported 100% occupancy. Same-session
`iree-benchmark-loom` results (ten measured samples, one hot input set):

| isolated K=5120, rows=17408 route | p50 | p90 |
|---|---:|---:|
| current dot4i T8 | 27.91 ms | 33.24 ms |
| raw-Q8_0 WMMA T8 | 13.92 ms | 14.07 ms |

The WMMA tile is 50.1% faster at p50 and has much lower variance. This clears
the isolated acceptance gate. Integration still requires a WMMA-specific
activation quantizer that zero-pads the physical token tile to 16 and emits
block-major scales, plus a default-off policy route; the existing dot4i layout
must remain unchanged for its fallback path.

## Structural fix landed: PP128 now makes one weight pass

`kHrxPrefillChunkTokens` is 128 and the prefill tile size is chosen by the
active route (`PrefillChunkTokens()`), so `--hrx-fusions blocked-prefill`
takes a 128-token prompt through the layers once instead of sixteen
eight-token passes. Every batch-native artifact was widened to the 128-token
tile: `qwen_rmsnorm_batch_f32` (token capacity 128),
`qwen_residual_add_batch_f32` (655360 elements),
`qwen_swiglu_pointwise_batch_f32` (2228224 elements), and both DeltaNet batch
kernels. The `_t8` dot4i artifacts keep their own eight-token bound and remain
the fallback when the blocked artifacts are absent.

Measured, device-local weights, release binary:

| route | pp128 |
|---|---:|
| serial (`none`) | 6.25 t/s |
| `chunked-prefill` (f32 `_t8`) | 5.53 t/s |
| `int8-prefill` (dot4i `_t8`) | 20.27 t/s |
| `blocked-prefill` (128x128 W8A8) | **211.72 t/s** |

That is 34x the serial baseline, 10.4x the previous best, and 53% of the
399.59 t/s HIP reference. PP128 now takes about 0.605 s, which is roughly
43 GB/s of effective weight streaming against the isolated kernel rates of
73-80 GB/s, so about 40% of the wall time is still outside the projections.

### Batch-native SSM front end and wave-level recurrence

With one weight pass in place the remaining prefill cost moved to the per-token
stages. Two further changes, both numerically exact reorganizations:

- `qwen_deltanet_prepare_batch_f32` and `qwen_ssm_conv_silu_batch_f32` replace
  128 preparation and 128 convolution dispatches per layer with one each. The
  convolution keeps its four-tap window in registers across the tile, so the
  rolling state is still read and written exactly once per layer and the
  per-token arithmetic is unchanged. Prefill logits were bit-identical before
  and after (cosine 0.99992108 both times).
- `qwen_deltanet_recurrence_batch_f32` now runs one wave per (value row, key
  vector) with each lane folding four key elements, so its two per-token
  reductions stay inside the wave. The artifact went from four waves with
  LDS-backed workgroup reductions to **zero barriers**, 23 VGPRs and 100%
  reported occupancy.

Stage profile for one 128-token chunk, device-local weights:

| stage | before batching | after SSM front end | after wave recurrence |
|---|---:|---:|---:|
| attention | 92.8 ms | 90.5 ms | 93.1 ms |
| SSM | 274.4 ms | 225.7 ms | 133.7 ms |
| FFN | 227.1 ms | 221.5 ms | 226.8 ms |
| chunk | 594.4 ms | 537.7 ms | 453.7 ms |

| route | pp128 |
|---|---:|
| `int8-prefill` (dot4i `_t8`) | 20.27 t/s |
| `blocked-prefill` | 211.72 t/s |
| + batched SSM front end | 229.65 t/s |
| + wave-level recurrence | **269.78 t/s** |

That is 43x the 6.25 t/s serial baseline and 68% of the 399.59 t/s HIP
reference. Prefill parity holds throughout: top-1 matches HIP and cosine
similarity is 0.9999 (the small changes between runs are float reassociation
in the reduction order, not a correctness change).

### Blocked-kernel ablations: what does not limit it

The blocked projection sits at roughly 70 GB/s of weight traffic and 16.9 TOPS
for rows=17408, K=5120, 128 tokens (1.347 ms device time). Five hypotheses were
tested and rejected by measurement:

| change | result |
|---|---|
| BK=2 staging (half the barriers) | 0.537 ms vs 0.536 ms - no effect |
| LDS double buffering (one barrier per block) | 1.314 ms vs 1.347 ms - 2% |
| 256 rows per workgroup (half the activation re-reads) | 1.414 ms - worse |
| contiguous weight addresses (coalescing probe) | 1.386 ms - no effect |
| epilogue scale multiplies removed | 1.362 ms - no effect |

So the kernel is not barrier-bound, not bandwidth-bound, not limited by
activation re-reads, not by weight-read coalescing, and not by epilogue VALU.
It responds only to latency hiding (register prefetch was worth 23% and wide
32-byte staging another 8%), which points at the LDS-to-fragment-to-MMA
dependency chain with eight waves per SIMD. The next kernel experiment should
increase independent MMA work in flight per wave rather than tune staging
further.

### Blocked 128x128 W8A8 projection: isolated gate passed

The blocked route from the macro card is implemented and validated in
isolation. `qwen_q8_0_gemm_i8_blocked_k{5120,6144,17408}_t128` each compute
128 output rows x 128 tokens per 256-thread workgroup, with eight wave32
subgroups sharing one staged K panel.

Three defects were found and fixed while bringing the first kernel up:

- the oracle's expected constant omitted the f16 block scale, so a correct
  kernel looked wrong (it produced 655360, the true value, against an expected
  327680);
- the epilogue read the staged per-row weight scales at `parity * 16` while the
  staging wrote them at `parity * 8 + pair`, so odd-parity rows read past their
  wave's scale block and came out exactly half-sized;
- activation scales were staged from the weight-row mapping (`tid / 2`), which
  only reaches token tiles 0-3, so every token from 64 upward was garbage.

Two topology experiments were then measured under the `dispatch_complete`
protocol on rows=5120, K=5120, 128 tokens:

| variant | device time |
|---|---:|
| baseline blocked (one K block per staging round) | 0.536 ms |
| BK=2 staging (half the barriers) | 0.537 ms |
| register prefetch of the next round | 0.413 ms |
| prefetch + BK=2 | 0.443 ms |
| prefetch + BK=4 | 0.758 ms |
| prefetch + wide 32-byte staging (retained) | 0.382 ms |

Barrier count is not the limit - BK=2 is a wash - but global load latency is:
issuing the next round's loads before the current round's matrix work is worth
23%, and moving whole 32-byte payloads per thread (threads 0-127 stage weight
rows, 128-255 stage token rows) adds another 8%.

The retained topology measures, at 128 tokens per dispatch:

| artifact | rows | device time |
|---|---:|---:|
| `..._blocked_k5120_t128` | 5120 | 0.385 ms |
| `..._blocked_k6144_t128` | 5120 | 0.498 ms |
| `..._blocked_k17408_t128` | 5120 | 1.247 ms |

Row counts no longer have to fill the 128-row macro tile: staging clamps to the
last live row and every epilogue store is guarded, so the 96-row SSM alpha/beta
projection uses the same artifact. `qwen_activation_quantize_blocked_k{5120,
6144,17408}` emit the matching fragment order, zeroing payload and scale for
physical tokens beyond the logical count.

At the measured per-shape rates one whole-checkpoint pass is roughly 330 ms,
which is approximately 390 t/s for PP128 before attention, SSM, and elementwise
work - the same order as the 399.59 t/s HIP result, and about 900 t/s if the
projections can be pushed to the DRAM roofline.

**The structural inefficiency is still in place and is the next commit.**
`ForwardPromptBatch` still slices PP128 into sixteen eight-token chunks and
walks all 64 layers per chunk, so the checkpoint is streamed sixteen times. No
kernel result can beat that; the executor must hold a 128-token tile live and
make one weight pass.

### Active WMMA integration direction

The first production route is deliberately K=5120 only and default-off behind
`--hrx-fusions wmma-prefill`. This flag implies `int8-prefill` and
`chunked-prefill`. K=5120 is first because the stage profile says FFN is 67% of
the chunk, and every FFN layer's wide gate/up projection consumes K=5120. It
also covers the attention/SSM input projections without introducing another
artifact shape.

The executor keeps two independent quantized operand contracts:

- dot4i keeps its existing physical T8, token-major payload and token-major
  scales and remains the fallback for all K values;
- WMMA uses a physical 16-token payload tile, zero-fills columns outside the
  logical 1-8 token chunk, and stores the eight live scales block-major. This
  lets one wave load a 16x16 signed-int8 RHS fragment directly and one output
  row lane load all eight scales in a single vector operation.

The K=5120 WMMA kernel reads the model's interleaved 34-byte Q8_0 blocks
directly; no weight repack or second checkpoint copy is introduced. One wave
computes 16 output rows by a physical 16-token tile. It performs two K=16 WMMA
operations per Q8 block, writes the raw i32 fragment to 1 KiB LDS, applies the
row-specific f16 weight scale and token-specific activation scales in f32, and
accumulates across 160 blocks. Logical output stores are guarded by `tokens`,
so short validation prompts do not expose the padded columns.

Current integration state:

- the production dynamic `(rows, tokens)` artifact passes its nonzero device
  oracle and compiles at 40 VGPRs with no spills;
- the separate padded/block-major K=5120 quantizer compiles;
- the manifest, loader, executor dispatch, policy parser, and policy unit test
  are wired;
- `nix build .#hrx` and `nix build .#checks.x86_64-linux.tests` pass;
- the combined HIP-reference/native-HRX run completed. Top-1 matched (157),
  cosine was 0.99989790, RMSE 0.04701518, mean absolute error 0.03675622, and
  maximum absolute error 0.25952494. This is slightly closer to HIP than the
  existing int8-prefill result, but still fails the current f32-derived
  envelope; the W8A8-specific envelope decision remains open.
- the same four-token run exposed a decisive performance regression:
  attention 44.25 ms, SSM 91.92 ms, FFN 934.72 ms, whole chunk 1070.89 ms.
  The integrated topology is therefore not an end-to-end win and must not be
  retained in its current form.

The isolated 13.92 ms result did not predict integrated performance because
the benchmark's `case_end_to_end` timing and hot synthetic operands do not
model 64 layers of distinct weights, while the candidate performs two LDS
barriers for every one of 160 Q8 blocks in every output-row tile. Those 320
barriers per workgroup dominate the real FFN path. The matrix arithmetic is
faster, but the result-fragment scale application topology is wrong.

A follow-up mapping experiment then derived the exact gfx1151 i32 accumulator
layout. The kernel was rewritten to keep the result in registers, removing all
320 per-workgroup barriers and all LDS. Its production-shaped oracle passes; it
compiles at 40 VGPRs, 36 SGPRs, zero LDS, and zero spills. Under the same
`dispatch_complete`, hot-input benchmark protocol it measures 4.15 ms p50
versus 3.14 ms for the existing dot4i T8 kernel. Thus even the barrier-free
single-wave WMMA topology is 32% slower than dot4i. Its remaining redundant
weight-scale loads could be reduced with subgroup broadcasts, but that is now
explicitly de-prioritized: optimizing a T8 kernel cannot break the 54-69 t/s
structural ceiling imposed by sixteen checkpoint passes at PP128.

The T8 WMMA card is therefore rejected as a production direction. Its fragment
mapping and oracle remain useful input to the new 128x128 blocked kernel, where
eight waves share staged weights, activations, and scales like the proven HIP
implementation.

Historical acceptance sequence for the rejected T8 route:

1. Full prefill-batch and decode parity against HIP, using a documented W8A8
   prefill envelope rather than the f32 decode envelope. Top-1/finite behavior
   passed; the envelope decision is still unresolved.
2. One stage-traced run to verify that FFN time moves in the same direction as
   the isolated 50% K=5120 result. **Failed:** FFN regressed to 934.72 ms for
   four tokens because of the per-block LDS/barrier topology.
3. The barrier-free rewrite failed its isolated gate (4.15 ms versus 3.14 ms),
   so the slow pp128 A/B is intentionally not run.
4. Remove the T8-only production policy/artifacts once the fragment-layout
   evidence has been transferred to the blocked PP128 implementation.

Superseded T8 follow-ups (kept only as rejected-card history):

1. **Add K=17408 WMMA for FFN down.** Gate/up and down are the two large FFN
   projections. Parameterize the proven tile rather than inventing another
   topology; K=17408 has 544 Q8 blocks and will expose whether the per-block LDS
   barriers become dominant.
2. **Add K=6144 WMMA for attention/SSM outputs.** This is lower priority because
   attention is only 7% and all SSM work is 25% of the traced chunk, but it
   completes projection coverage once FFN wins are established.
3. **Eliminate the per-block result LDS round trip (now the blocking item).** The current kernel stores
   each i32 WMMA result fragment to LDS so lanes can apply independent row and
   token scales. Deriving the gfx11 result-fragment lane mapping, or adding a
   target-supported fragment repack, could retain/scalefold the eight live
   values in registers and remove two workgroup barriers per Q8 block. This is
   required before matrix-core adoption can continue. If the result fragment
   cannot be scaled in registers, reject and remove the WMMA integration.
4. **Evaluate larger row tiles per workgroup.** Two or four waves sharing the
   same activation fragment could amortize activation/scales and reduce
   workgroup count, but must be gated on LDS usage, occupancy, and actual
   timing; the earlier T16 and packed-layout regressions show that static
   instruction/register reports are not sufficient.
5. **Avoid redundant activation quantization.** Quantization is already shared
   across the multiple projections that consume one normalized activation.
   Do not fuse quantization into one projection unless the other consumers can
   reuse the result; otherwise reduced launch count would duplicate work.
6. **Add FFN substage timing.** If end-to-end FFN improvement is smaller than
   predicted, separately time RMSNorm, quantize, gate/up, SwiGLU, re-quantize,
   down, and residual for one representative layer. This distinguishes a slow
   K=17408 down path from quantization or elementwise overhead without another
   speculative kernel rewrite.
7. **Use IREE/Loom final-batch profiling for kernel counters.** `rocprofv3`
   remains useful for HIP but has been unreliable on the HRX execution path.
   The standalone benchmark can capture device timestamps/counters around the
   exact candidate and baseline without a 27B model load.

### HRX test coverage gap closed

The canonical `tests` check now configures with `-DENGINE_ENABLE_HRX=ON` and
`-DHRX_ROOT`, so a hosted PR run compiles the HRX arena and executor tests; the
GPU-labelled tests are excluded from execution with `ctest -LE gpu` because the
sandbox has no gfx1151 device. No test that previously ran is skipped: only
`hrx_backend_test` and `qwen_hrx_executor_test` carry the `gpu` label.

Turning the gate on immediately exposed four latent breakages that the old
configuration could not see:

- `src/cli/bench/bench.cpp` defined `BenchStats`, `ComputeStats`,
  `MakeBenchmarkTokens`, and `MakeTestName` inside an `ENGINE_ENABLE_HIP`
  guard while the HRX route used them, so an HRX-without-HIP build never
  compiled. The helpers are now outside the guard.
- `qwen_hrx_executor_test.cpp` called a nonexistent `GetTestModelPath()` and
  `GgufReader::Open`. It now uses the `GUFO_HRX_Q8_MODEL` environment variable
  and `GgufReader::OpenFile`, matching the other cases in the same file.
- The same test used `QwenHrxExecutor::CurrentPosition()`, which did not exist.
  It is now a public read-only accessor.
- `bench_cli_test` compiles `bench.cpp` but lacked `GUFO_HRX_KERNEL_DIR`.

`nix build .#checks.x86_64-linux.tests` now passes.

### Decode fusion policy work in progress

`--hrx-fusions` now selects three real Q8_0 routes, each default-off and each
requiring no new kernel artifact:

- `ping-pong`: the residual stream alternates between the `kHidden` and
  `kNormed` arena buffers instead of copying the stage result back into
  `kHidden`. Each layer performs two swaps, so `kHidden` still owns the stream
  at layer boundaries, at the final stage, and across transactions. Removes 128
  copy dispatches per token.
- `ffn-gate-up`: one GEMV over the contiguous `ffn_gate`+`ffn_up` weight span
  (34816 rows, K=5120) writing a contiguous gate/up activation pair. Removes 64
  dispatches per token and streams one 190 MiB weight range per layer instead
  of two.
- `ssm-alpha-beta`: one GEMV over the contiguous `ssm_alpha`+`ssm_beta` span
  (96 rows). Removes 48 dispatches per token.

`q8` enables all three; `all` now includes them. The arena over-allocates
`kFfnGate` and `kSsmAlpha` to hold both halves; the unfused routes still use
the separate `kFfnUp` and `kSsmBeta` buffers. Fused weight bindings are built
only when the two tensors are provably adjacent inside one HRX buffer,
otherwise the route falls back to the unfused path.

A token currently issues roughly 1,170 dispatches (48 SSM layers x 11, 16
attention layers x 12, 64 FFN x 7, plus embedding and final). The three routes
together remove about 240 of them, or 20%.

#### Ordering defect found and fixed

The first fused build failed parity at prompt step 0 (max absolute error 3.68,
cosine 0.976). The cause was in the rewritten SSM stage: the alpha/beta GEMV
result was bound to a `const bool` initialized before the dispatch chain, so
those GEMVs were enqueued before the RMSNorm that produces their input. That
reordering corrupted every SSM layer in both the fused and unfused routes. The
alpha/beta dispatch is now a lambda invoked at its correct position inside the
chain.

After the fix, `--hrx-fusions q8` with device-local weights passes parity
against HIP for position zero, a four-token prompt, and four greedy decode
steps: all top-1 tokens match, cosine similarity 1.0, worst max-absolute error
7.63e-6, worst RMSE 9.6e-7 - the same envelope as the unfused route.

Remaining acceptance for these routes: an interleaved pp128/tg16 A/B against
`none` on the same binary.
