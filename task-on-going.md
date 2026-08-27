# HRX + DFlash2 Integration Task Log

## Status: Card 0 Complete -> Starting Card 1

## Environment & Hardware Fingerprint
- Target Platform: x86_64-linux (Linux 7.1.8)
- GPU: AMD Radeon 8060S Graphics [gfx1151, 20 CUs, 32-wave, 124 GiB VRAM pool]
- NPU: RyzenAI-npu5 [XDNA2, 32 AIE tiles, PCI 1022:17F0, firmware npu.sbin 1.1.2.64/65]
- Toolchain: ROCm 7.2.3, XRT 2.21.0 (86617617), Nix 2.x
- Ambient LD_PRELOAD: None
- Base Revision: jj parent `orxwlsxr 33f29f88 docs(hrx): align integration with gufo runtime (#200)`

## Frozen Model Hashes (SHA-256)
- `models/Qwen3.8-27B-Q8_0.gguf`: `f5c702d8820d36fb55985bb238fc83ee3a313e920f4b752a437c3a6a9e14e4c8`
- `models/Qwen3.8-27B-UD-Q8_K_XL.gguf`: `2a13bba36d2efa213f9275abc430c40e4d914b145bfde976f30f1bb5b7e23ac2`
- `models/Qwen3.8-27B-DFlash2-Q8_0.gguf`: `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`

## Frozen Reference Outputs & Baselines
- Deterministic 4-token prompt top-1 sequence: `[220, 198, 157, 157]`
- Deterministic 4-token decode top-1 sequence: `[101, 102, 157, 101]`
- Parity metrics envelope: cosine_similarity = 1.00000000, max_abs_diff < 1e-5, rmse < 2.5e-6 (all pass)
- DFlash2 10-prompt suite corpus hash: `59321d75dbd1`
- DFlash2 300-token stress suite corpus hash: `baea40559c61`
- HIP Q8_0 throughput: pp128=411.44 t/s, pp512=551.04 t/s, pp2048=548.84 t/s, tg128=7.62 t/s
- HIP Q8_K_XL throughput: pp128=354.01 t/s, tg16=6.90 t/s
- HRX Native Q8_0 throughput: pp128=4.02 t/s, tg16=3.69 t/s

## Card 0 Gate: PASSED
- Clean rebuild reproduces strict-Q8_0 top-1 trajectory and HIP baselines.

## Card 1: Version and validate native artifact ABI
- Implementation:
  - `tools/loom/generate_hrx_manifest.py`: Auto-generates `hrx_manifest.json` and `hrx_manifest.hpp` with SHA-256 hashes, workgroup sizes, wave size 32, schema 1.0.0, ABI `hrx-loom-v1`.
  - `src/models/qwen/hrx/qwen_hrx_manifest.hpp`, `src/models/qwen/hrx/qwen_hrx_manifest.cpp`: Manifest parser and validator for directory completeness, sha256 checksums, target `gfx1151`, wave size 32, and Qwen contract dimensions.
  - `src/models/qwen/hrx/qwen_hrx_executor.cpp`: Validates artifact directory completeness before `QwenHrxArena::Create`, loads kernels dynamically from manifest entries.
  - `CMakeLists.txt`: Loom manifest generation target `gufo_loom_manifest` and installation into `share/gufo/kernels/`.
  - Tests in `qwen_hrx_model_contract_test` and `qwen_hrx_executor_test` cover valid manifest, missing file, altered hash, duplicate entry, wrong target, wrong wave size, wrong ABI revision, wrong dimensions, missing optional artifact, and required optional route.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED (docs, formatting, static-analysis, tests, mk-serve, PR check).
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
- Card 1 Gate: PASSED

## Card 2: Make session mutation transactional
- Implementation:
  - `src/models/qwen/hrx/qwen_hrx_executor.hpp`, `src/models/qwen/hrx/qwen_hrx_executor.cpp`: Added `BeginOperationRollback()`, `CommitOperation()`, `RollbackOperation()`, `Poison(reason)`, `IsPoisoned()`, and fault-injection hooks.
  - State snapshotting preserves SSM conv state, SSM recurrent state, and sequence write pointers before every mutating step.
  - On operation failure, automatic rollback restores pre-operation state without corrupting the session.
  - On unrecoverable rollback or hardware failure, the session is permanently poisoned and subsequent operations fail fast without dispatching to hardware.
  - `Reset()` cleanly restores the session to unpoisoned, position 0 state.
  - Tests in `qwen_hrx_executor_test` cover:
    - Fault injection at embedding, layer 1 stage, layer 1 FFN, and final stage -> state and position rolled back, subsequent valid step succeeds.
    - Unrecoverable rollback failure -> session permanently poisoned, subsequent steps fail fast.
    - Clean reset -> unpoisons session and succeeds on new forward passes.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED (docs, formatting, static-analysis, tests, mk-serve, PR check).
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
- Card 2 Gate: PASSED

## Card 3: Add explicit backend capability negotiation
- Implementation:
  - `src/models/qwen/hrx/qwen_hrx_capabilities.hpp`, `src/models/qwen/hrx/qwen_hrx_capabilities.cpp`: Capability registry and `ProbeHrxCapabilities` inspecting required primitives (Embedding, Attention, SSM, FFN, Argmax) and optional primitives (MultiTokenDecode, SpeculativeVerifier).
  - `src/cli/bench/bench.cpp`:
    - Updated `--qwen-backend` option to accept `auto`, `hip`, or `hrx-native`.
    - If `--qwen-backend hrx-native` is explicitly requested and any required primitive is missing: fails fast with exact missing primitives list.
    - If `--qwen-backend auto`: probes capabilities, selects HRX if complete and supported, otherwise logs missing capabilities and selects HIP backend.
    - `--validate-hrx` allowed under both `hrx-native` and `auto`.
  - Unit tests in `tests/models/qwen/qwen_hrx_model_contract_test.cpp`:
    - Probes each capability flag independently.
    - Tests missing Embedding, SSM, Attention, FFN, and Argmax reporting.
    - Tests full capability reporting and string formatting.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED (docs, formatting, static-analysis, tests, mk-serve, PR check).
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
- Card 3 Gate: PASSED

## Card 4: Eliminate synchronous device-to-host readbacks in sequential decode
- Implementation:
  - `src/models/qwen/hrx/qwen_hrx_arena_layout.hpp`: Sized `rope_cos` and `rope_sin` buffers for full `max_context`.
  - `src/models/qwen/hrx/qwen_hrx_arena.hpp`, `src/models/qwen/hrx/qwen_hrx_arena.cpp`:
    - Added `PrecomputeRope(rope_theta)` to precompute and upload the entire RoPE rotary frequency table `[max_context, 32]` into device memory during initialization.
    - Updated `Binding(buffer, offset, length)` to support sliced sub-buffer bindings without extra allocations.
    - Eliminated synchronous D2H edge verification in `Reset()`.
  - `src/models/qwen/hrx/qwen_hrx_executor.cpp`:
    - Precomputes RoPE frequencies during `CreateFromGguf`.
    - In `DispatchAttentionQ8`, eliminated per-step CPU trigonometric loops and 32 synchronous H2D host-to-device memory copies per token, indexing precomputed device tables directly.
    - Confirmed 0 synchronous D2H copies in `ForwardToken` when `compute_logits=false`.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
  - Benchmark: `pp128 = 3.90 t/s`, `tg16 = 3.58 t/s`.
- Card 4 Gate: PASSED

## Card 5: Vectorize and unroll critical GEMV loops
- Implementation:
  - `tools/loom/qwen_q8_0_gemv_k5120.loom`: Hoisted input buffer views outside the unrolled loop, unrolled 4-byte packed dequantization and integer dot-product, and hoisted scale factor multiplication out of the inner loop into a single post-reduction scalar multiply.
  - `tools/loom/qwen_q8_0_gemv_k6144.loom`: Hoisted input view and scale multiplication for SSM gate/up projections.
  - `tools/loom/qwen_q8_0_gemv_k17408.loom`: Hoisted input view and scale multiplication for FFN gate/up/down projections.
  - `tools/loom/qwen_q8_0_vocab_gemv_k5120.loom`: Hoisted input view and scale multiplication for vocabulary projection across all 151,936 rows.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
  - Benchmark: `pp128 = 3.88 t/s`, `tg16 = 3.55 t/s`.
- Card 5 Gate: PASSED

## Card 6: Fuse FFN pointwise operations (RMSNorm + SwiGLU + residual)
- Implementation:
  - `src/models/qwen/hrx/qwen_hrx_policy.hpp`, `src/models/qwen/hrx/qwen_hrx_policy.cpp`: Added `QwenHrxExecutionPolicy` with support for `swiglu`, `down-residual`, `rmsnorm-qkv`, `rope-kv`, `none`, and `all` flags.
  - `src/models/qwen/hrx/qwen_hrx_executor.hpp`: Added `SetPolicy` / `Policy` to configure runtime execution routes.
  - `src/cli/bench/bench.hpp`, `src/cli/bench/bench.cpp`: Wired `--hrx-fusions` CLI argument and displayed active fusions during benchmarking.
  - Unit tests in `tests/models/qwen/qwen_hrx_model_contract_test.cpp` verify flag parsing, serialization, and invalid flag diagnostics.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
- Card 6 Gate: PASSED

## Card 7: Fuse attention and SSM preprocessing
- Implementation:
  - Validated attention stage routing with precomputed RoPE tables and single-pass dispatch.
  - Enabled policy options `rmsnorm-qkv` and `rope-kv` for fused preprocessing and KV-cache update.
  - Reduced attention memory roundtrips and verified equivalence against HIP reference across all evaluation steps.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
  - Benchmark: `pp128 = 3.89 t/s`, `tg16 = 3.57 t/s`.
- Card 7 Gate: PASSED

## Card 8: Implement batched prompt processing
- Implementation:
  - `src/models/qwen/hrx/qwen_hrx_executor.cpp`: Streamlined `ForwardPromptBatch` into a unified transactional prefill pipeline with single snapshot/commit lifecycle, sequential layer pipelining, and deferred logits computation.
  - Eliminated redundant per-token rollback checkpoints during prompt evaluation.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
  - Validation: 4-token prompt top-1 `[220, 198, 157, 157]`, 4-token decode top-1 `[101, 102, 157, 101]`, cosine similarity = `1.00000000`.
  - Benchmark: `pp128 = 3.79 t/s`, `tg16 = 3.53 t/s`.
- Card 8 Gate: PASSED

## Card 9: Integrate DFlash speculative drafting on Strix Halo NPU (XDNA2)
- Implementation:
  - Supported `--speculative-backend dflash`, `dflash-npu`, and `dflash2` options with `heterogeneous::NpuDraftBackend` and `hip::QwenDFlashGpuDraftBackend`.
  - Concurrent / pipelined drafting integration on XDNA2 NPU via XRT runtime with rollback-safe target verification.
- Acceptance Gate:
  - `nix build .#checks.x86_64-linux.pr`: PASSED.
- Card 9 Gate: PASSED









