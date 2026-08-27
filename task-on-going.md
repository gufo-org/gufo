# Native HRX Qwen3.8-27B MVP

## Bare-minimum MVP definition

The first MVP is intentionally limited to one exact production contract:

- Linux x86-64 on AMD Strix Halo (`gfx1151`).
- Qwen3.8-27B with 64 authoritative layers: 16 full-attention and 48 SSM.
- Strict direct Q8_0 embeddings and matrices with F32 norm/SSM vectors.
- One session, sequential tokens, greedy decoding, and runtime-bounded context.
- Native HRX artifacts for every model operation; no HIP or host-compute
  fallback.
- Entry point: `bench --qwen-backend hrx-native`.

The implementation is an MVP only after it compiles on `gfx1151` and produces
matching full-model token/logit results against the established CPU/HIP route.
Code presence or artifact readiness alone is not an E2E parity claim.

## Implemented, target validation pending

- Move-only HRX buffers, checked transfers, native fill/copy, arena reset, and
  recurrent-state snapshot/restore.
- Stable arena storage for hidden/scratch vectors, 16-layer KV cache, 48 SSM
  convolution states, 48-head DeltaNet recurrent state, logits, token, and
  position.
- Strict immutable model bindings with tensor type, shape, encoded-size, and
  nonzero-offset validation.
- Native Q8_0 embedding and fixed-K GEMVs for K=5120, K=6144, and K=17408.
- Dedicated K=5120 vocabulary GEMV for 248320 rows and deterministic argmax.
- True RMSNorm, residual add, copy, Q+gate split/sigmoid, SwiGLU, per-head
  normalization, causal attention decode, SSM convolution/SiLU, DeltaNet
  preparation/recurrence, and final normalization artifacts.
- Complete Q8_0 FFN composition.
- Full-attention composition with separate Q+gate/K/V projections, RoPE/KV
  update, causal 24Q/4KV GQA, output projection, and residual.
- Gated DeltaNet composition with QKV/gate/alpha/beta projections, four-tap
  convolution state, 16-key-head/48-value-head mapping, alpha decay, beta
  correction, recurrent-state mutation, output projection, and residual.
- `ForwardToken` composition across the authoritative 64-layer order, followed
  by final norm, vocabulary projection, argmax/readback when requested, and
  position advancement.
- `ForwardPromptBatch` as a simple sequential `ForwardToken` loop.
- Narrow fail-closed native bench route and focused contract/oracle tests.

## Required before calling it an MVP

1. Build the HRX package and compile every Loom artifact on `gfx1151`.
2. Fix all AOT/C++ compile errors as one batch.
3. Run focused primitive tests for Q8 signed decoding, non-unit scales,
   nonzero offsets, state mutation, reset, and snapshot/restore.
4. Compare staged attention, FFN, and DeltaNet outputs with the CPU/HIP semantic
   oracle.
5. Compare final logits/top-1 for position zero.
6. Compare a multi-token prompt plus greedy decode so KV and recurrent state
   evolution are exercised.
7. Run the canonical PR check and record the exact revision, artifact hashes,
   software versions, and numerical errors.

Any failure after partial state mutation currently requires `Reset` before a
retry.

## Explicitly deferred until after MVP parity

### Models and formats

- Mixed Q5_K, Q6_K, Q8_K, BF16, and F32 matrix routes.
- Load-time quantized-to-BF16 conversion.
- Other Qwen sizes, architectures, layer layouts, head counts, and dimensions.
- LoRA, adapters, multimodal inputs, and non-GGUF model sources.

### Execution features

- Batched sessions and continuous batching.
- Optimized prompt prefill; prompt evaluation remains sequential.
- Speculative decoding and draft models.
- Sampling beyond greedy argmax, including temperature, top-k, top-p, and
  repetition penalties.
- Transactional per-token rollback after a mid-dispatch failure.
- Concurrent use of one executor and multi-device execution.
- `prompt` and `serve` integration; they remain on the established backend.

### Performance

- Graph capture/replay of a complete token.
- Kernel fusion beyond the minimum existing artifacts.
- Persistent kernels, async overlap, multiple streams, and transfer overlap.
- Matrix tiling/vectorization tuning and architecture-specific scheduling.
- Scratch-memory aliasing, arena compaction, and reduced peak memory.
- Optimized long-context attention, paged KV cache, and KV quantization.
- Production throughput/latency targets and roofline-driven optimization.

### Robustness and productization

- General artifact ABI/version manifest and compatibility negotiation.
- Automatic artifact discovery beyond the configured/install directory.
- Recovery from individual dispatch failures without resetting the session.
- Long-context soak tests, fuzzing, fault injection, and multi-session stress.
- Stable public native-HRX API guarantees.
- User-facing diagnostics, telemetry, deployment guidance, and support policy.
- Removal of intentionally duplicated executor code or architectural cleanup.

### Broader validation

- Numerical tolerances across additional prompts and model files.
- Quality/regression suites beyond initial token parity.
- Performance comparison against optimized HIP.
- Full sanitizer, coverage, and flaky-test campaigns.
- Review and optimization of every kernel's generated ISA.

## First target commands

Use Nix only and ensure all new files are visible to Nix before building:

```sh
git add CMakeLists.txt src tests tools docs task-on-going.md
nix build .#hrx
nix build .#checks.x86_64-linux.pr
```

Then run the focused HRX tests from the produced check/package environment and
invoke the native bench route with a strict Q8_0 Qwen3.8-27B GGUF. Do not report
native E2E success until the multi-token full-model parity gate passes.
