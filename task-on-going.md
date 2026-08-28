# HRX integration: current status and recovery plan

Last updated: 2026-08-28

## Baseline identity

- PR head: `c9fe1120776e5485a8842b1f0eed7effe97fbb48`
- Working-copy revision: `ebb6f1b8` (dirty; experiments described below)
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

The small-kernel HRX variants are within measurement noise and are not wins.
Device-local weight placement is a large, reproducible signal: tg16 improves by
23-47% over the same-binary mapped baseline of 3.53 t/s. The spread between the
two copied-weight runs means more interleaved samples are still required before
reporting one headline number.

The latest same-binary A/B is +49.4% for tg16 (3.60 to 5.38 t/s). A separate
27B `gufo serve` process was resident during these measurements and uses about
27.3 GiB RSS, so final qualification must be repeated on an otherwise idle GPU.
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
but it does not improve the final stage or end-to-end throughput. The default-off
wave32 experiment is still present in the working copy and should be removed
before landing unless a later controlled measurement shows a gain.

### Wave256 argmax

The device oracle passes and compilation reports 7 VGPRs, 8 SGPRs, 64 bytes of
LDS, no scratch/private memory, and reported 100% occupancy. End-to-end tg16 and
the final-stage duration are unchanged. Argmax is too small a fraction of token
time to matter. The default-off experiment is still present and should be removed
before landing unless later evidence justifies it.

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

The device-local A/B passed the performance screen. Stage-synchronized pp1/tg1
changed as follows:

| stage | mapped | device-local | reduction |
|---|---:|---:|---:|
| attention | 17.48 ms | 10.37 ms | 40.7% |
| SSM | 69.44 ms | 45.67 ms | 34.2% |
| FFN | 167.11 ms | 95.83 ms | 42.7% |
| final | 23.85 ms | 19.59 ms | 17.9% |
| whole token | 279.79 ms | 173.43 ms | 38.0% |

Next, inspect GGUF mapping ownership and implement a production device-local
loader that releases file-backed mappings after the copy (or otherwise proves
that steady-state residency is not duplicated). Add failure-safe RAII cleanup,
then run interleaved mapped/device-local samples and full logit parity. Remove
the neutral wave32 vocabulary and wave256 argmax experiments before beginning
the Q8_K XL card.

The production loader and bench-side mapping release are now implemented and a
release Nix build passes. GNU time reports a 50,083,012 KiB maximum RSS during
device-local loading versus 28,325,024 KiB mapped; this is a load-time peak and
does not measure post-unmap steady state. Live post-load residency still needs
to be sampled.

Mapped HRX parity passes against HIP for four prompt and eight decode positions:
all top-1 tokens match, cosine similarity is 1.0, worst max-absolute error is
7.63e-6, and worst RMSE is 1.21e-6.

An eight-step combined HIP-to-device-local-HRX validation failed during the
one-shot H2D transfer even after `hipDeviceReset`: IREE reported
`RESOURCE_EXHAUSTED` from `hsa_amd_memory_pool_allocate`. The destination buffer
had already allocated, so the failure was checkpoint-sized transient transfer
storage, amplified by the resident 27B server. The loader is being changed to
64 MiB bounded H2D chunks; direct device-local parity is the acceptance gate.
