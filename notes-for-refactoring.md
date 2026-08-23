# Refactoring Notes — Qwen modules + fast iteration loop

> Review pass (verified against codebase) — factual corrections E1–E7 and design refinements folded in; see §8 Review findings.

Goal: structure the code so each architectural component (attention, SSM, FFN,
norm, RoPE, quantized GEMM, …) is a self-contained **module** with a
**fast end-to-end module test** (correctness vs CPU reference + timing).
Optimization iteration = hypothesis → change one module → run that module's
e2e test (sub-second) → keep or reject with evidence → full pipeline only
at the end via `strix-server bench`.

Explicitly not code changes yet — design + refactor plan only.

---

## 1. The iteration loop (target)

```
theory (hypothesis)
  → change ONE module (behind a policy toggle, default OFF — pattern exists)
  → ctest -R <module>            # correctness vs CPU reference + timing
  → keep / reject with evidence  # optimize-kernel skill contract
  → ... repeat ...
  → strix-server bench A/B       # full-pipeline gate, only at the end
```

What makes a loop fast, in order of cost:
1. **Build time** — the module change must recompile as little as possible.
   Small headers, one translation unit per module, per-module test binaries.
   (HIP compiles are the expensive part of `gpu-test` builds.)
2. **Test run time** — module e2e test must run in well under a second:
   synthetic weights, tiny shapes, 1–3 layers, no model file I/O, no server.
3. **Feedback clarity** — test output must answer exactly two questions:
   "is it still correct?" and "is it faster?" (numbers vs recorded baseline).

Existing assets this builds on (do not reinvent):
- Policy toggles in `src/core/hip/detail/qwen_attention_policy.hpp`:
  `ShouldFuseQKNormRoPEKvWrite`, `ShouldFuseResidualAddRMSNorm`,
  `ShouldFuseFFNSwiGLU`, `ShouldFuseSSMGateResidual`,
  `ShouldFuseRMSNormProjection`, `ShouldPrefetchNextLayer`. Module tests flip
  these to compare routes.
- `optimize-kernel` skill: baseline → one route behind a toggle → equivalence
  (CPU oracle vs GPU) → interleaved A/B bench → keep/reject with evidence.
- `src/bench/kernel_bench_main.cpp` (1071 lines): existing per-kernel
  micro-bench harness — level **below** module e2e; keep for kernel tuning.
- `src/testing/compare/logit_comparator.hpp`: existing numeric comparison
  (max/mean diff, RMSE, cosine, top-1) — reuse as the module correctness
  metric.
- `tests/kernels/` per-kernel tests — level **below** module e2e.

### Testing optimizations that span many modules

Blast radius has two shapes, each with a dedicated gate:
- **Shared-dependency** (change `quant::Gemm` dispatch, scratch-arena layout,
  norm/quant backend, plan cache → many consumers). Gate = the full L1 tier
  (`ctest -L module`), each module vs its CPU oracle. Because every L1 module
  test is a GPU-vs-CPU oracle (not "does it compile"), a shared change surfaces
  as *localized* module failures — the specific broken module lights up; you do
  not need a global test to locate it. The module dependency graph determines
  the impact set structurally, so don't enumerate: run the whole tier.
- **Cross-module coupling** (fusion or state handoff `fused_qknorm_rope_kv` /
  `ssm_residual_folded`). Gate = the L2 aggregate (whole synthetic model, GPU
  vs CPU generator) + the per-fusion fusion-route L1 tests.

Correctness composes: modules correct (L1 oracles) + fusion correct
(fusion-route) ⇒ pipeline correct (L2 cross-check). A many-module optimization
is NOT a bench run — it's `ctest -L module` + one L2, both seconds.
`strix-server bench` A/B stays phase-boundary only.

## 2. Module decomposition

Derived from the actual decode path (`QwenGpuExecutor::ForwardToken`
`src/core/hip/qwen_gpu_decode.cpp`, per-layer loop) and the CPU reference
(`ForwardLayer` = `src/models/qwen_forward.cpp:272` (96 lines);
`ForwardSSM` = `src/models/qwen_ssm.cpp:64` (177 lines)):

| Module | Boundary | State | Notes |
|---|---|---|---|
| `quant_gemm` | quantized GEMV/GEMM by `GgmlType` (F32, BF16, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q8_K) | none | Single dispatch point; today the switch is duplicated in `qwen_forward.cpp` (`QuantizedDot`, `DequantizeRow`) and re-implemented per kernel. Two distinct silent-failure sites: (a) `QwenTensorRef::Get` (`qwen_state.hpp`) handles `kQ8_K`/`kQ8_0`, so only `kQ5_K`/`kQ6_K` → `0.0F` (line 68); (b) the CPU GEMV dispatch real-handles only `Q3_K`/`Q4_K`/`Q6_K`, everything else → `0.0F` (lines 25/39). `QuantizedRowBytes` returning 0 is a further silent no-op guard in `QuantizedRow`/`TensorGEMV`. Unify must cover `Get()`, the GEMV dispatch, AND that guard. |
| `norm` | RMSNorm (attn norm, FFN norm, final norm) | none | Fused variants (residual+norm, norm+proj) are routes inside the module. |
| `rope` | RoPE (+ QK-norm fusion route) | none | |
| `attention` | QKV proj → RoPE → KV-cache write → score → out proj | `QwenKvCache` | Full-attention layers only. Fused `QKNormRoPEKvWrite` is a route. |
| `ssm` | in-proj → conv → DeltaNet recurrent update → gate/out | `QwenSsmCache` | Hybrid layers. `ShouldFuseSSMGateResidual` route lives here. |
| `ffn` | SwiGLU (gate/up/down) | none | `ShouldFuseFFNSwiGLU` route. |
| `residual` | residual add (standalone or fused) | none | Currently inlined in the loop; `ShouldFuseResidualAddRMSNorm` route. |
| `embed` / `unembed` | embedding lookup; final norm + lm_head | none | |
| `sample` | greedy argmax (sampling policy) | none | Trivial; keep as a module for completeness. |

Composition (the pipeline) is thin: `embed → per layer: [norm → (attn|ssm) →
residual → norm → ffn → residual] → unembed → sample`. The pipeline owns
`QwenModelWeights`, `QwenKvCache`, `QwenSsmCache`, scratch arena, stream;
modules are **stateless functions** called per layer.

### Module contract (proposed)

```cpp
// Context: everything a module needs from the runtime, one object.
struct ModuleCtx {
  hipStream_t stream;              // or cpu handle
  ScratchArena& arena;             // named-slice allocator (QwenScratchArena)
  BlasPlanCache& plans;            // hipBLASLt plan cache
};

// A module = a free function (or small class) over tensors + weights view.
void SsmForward(ModuleCtx& ctx, const SsmLayerView& layer,   // that module's weights only
                const Tensor& x, SsmState& state, Tensor& out);
void AttnForward(ModuleCtx& ctx, const AttnLayerView& layer,
                 const Tensor& x, KvCache& kv, std::size_t pos, Tensor& out);
```

A single raw `workspace` pointer cannot express the shared, named scratch
arena the decode path actually uses. `QwenScratchArena` is partitioned into
named sub-buffers (`d_hidden`, `d_normed`, `d_ssm_qkv`, `d_q`, `d_k`, `d_v`, …)
that are REUSED across module boundaries (QKV projection writes `d_ssm_qkv`,
then the SSM/attn stage consumes it). So `ModuleCtx` carries the arena (or an
arena-view with named-slice accessors) + `hipStream_t` + `BlasPlanCache&`;
modules request named slices and exchange `Tensor&` that alias arena slices
(rather than an opaque pointer). This is the real Data-Clump fix — a module
gets exactly the named slices it needs, and other modules keep their own.

Rules:
- A module owns its kernel selection internally. A hypothesis ("merge two
  BLAS calls", "fuse gate+residual") touches exactly one module file and is
  expressed as a route selected by the existing policy toggles.
- Each module exposes two backends behind the same signature:
  **CPU** (the existing `qwen_oracles` / `qwen_ssm.cpp` / `qwen_forward.cpp`
  code, re-homed) and **HIP**. The CPU backend is the oracle; the module e2e
  test cross-checks them. This also solves "oracles compiled into the
  production library" — they become the CPU backend of modules. Note the
  problem is NOT that they sit on the GPU hot path: `qwen_forward.cpp:125,131`
  (`ReferenceRMSNorm` inside `ForwardRMSNorm`) and `:144,149` (`ReferenceRoPE`
  inside `ForwardRoPE`) are calls internal to the CPU *reference* code; the
  GPU decode path is `src/core/hip/qwen_gpu_decode.cpp` (HIP kernels), which
  the executor never routes through `qwen::Reference*`. The CPU backend is an
  oracle/fallback for the L1 cross-check, not a production fast path — the
  executor never calls it on the hot path. Treat it as an oracle, not a
  supported fallback runtime (the `quant_gemm` CPU backend is a slow
  per-element dequant reference, fine for small-shape parity only).
- `LayerView` = the slice of `QwenLayerWeights` the module needs (fixes the
  current Data Clump: every call site drags the whole layer struct plus
  caches plus config plus arena).
- **Fusion-ownership rule (DECIDED).** "One hypothesis = one module file" breaks
  for two existing toggles that cross module boundaries: `opt-c010-rmsnorm-projection`
  (`qwen_gpu_decode.cpp:130`, fuses the pre-node norm INTO the QKV/SSM/FFN
  projection GEMVs — crosses norm↔gemm/attn/ffn) and `opt-c010-ssm-gate-residual`
  (`:196`, folds the post-SSM residual into the `ssm_out` GEMV — crosses
  ssm↔residual). `opt-c014-layer-prefetch` (`:88,105`) is layer-level, outside
  any module entirely.
  **DECIDED: Option B — composition/fusion layer.** The 4 cross-module fusions
  (`ShouldFuseQKNormRoPEKvWrite` norm↔rope↔attention,
  `ShouldFuseRMSNormProjection` norm↔quant_gemm/ffn,
  `ShouldFuseResidualAddRMSNorm` residual↔norm,
  `ShouldFuseSSMGateResidual`-decode ssm↔residual) are owned by the per-layer
  composition step (`ExecuteStep`), each behind its policy toggle. Modules stay
  pure single-stage stateless functions (`norm`, `rope`, `attention`, `ssm`,
  `ffn`, `residual`, `quant_gemm`) — L1-testable singly, unfused. Exception
  (within-module): `ShouldFuseSSMGateResidual`-prefill (SSM gate) lives inside
  the `ssm` module. State handoff (`fused_qknorm_rope_kv`, `ssm_residual_folded`)
  resolves in the composition layer, NOT in module signatures. Each fused kernel
  gets an L1 **"fusion-route" test**: fused kernel vs chained-unfused-module
  calls, numerics compare. "One hypothesis = one file" transfers to the
  composition layer: one toggle → one fused kernel → one fusion-route test.
  **Gotcha:** `ShouldFuseFFNSwiGLU` is prefill-only and REJECTED (37× regression)
  — not a live route. Decode's "fused FFN" (`LaunchFusedRMSNormSwiGLUGEMV`,
  decode:315) is driven by `fused_rmsnorm_proj` (`ShouldFuseRMSNormProjection`),
  NOT the SwiGLU toggle — do not wire FFN fusion-route test expectations to the
  wrong toggle.
- **Capture-transparency.** The executor also has graph-capture + SSM-state
  replay machinery (`replaying_ssm_state_`, `ReplaySsmState`,
  `CanReplaySsmPosition`, `DisableSsmReplayCapture`) that lives outside any
  module signature. Module functions must stay capturable: host-side policy
  decisions are made BEFORE capture; no host branching inside capture. Gate
  each decode-touching commit with the `STRIX_DISPATCH_TELEMETRY=1` capture
  smoke test (expect `miss_captured` then `hit`).

## 3. Test pyramid

| Level | What | Binary | Runtime target |
|---|---|---|---|
| L0 kernel | single kernel vs naive reference | existing `tests/kernels/*` | already per-kernel |
| **L1 module e2e** | **one module, synthetic weights, 1–3 layers, GPU vs CPU backend, correctness + timing** | **new `tests/modules/<name>_module_test.cpp`, one per module** | **< 1–2 s** |
| L2 model integration | whole synthetic model, GPU vs CPU generator | existing `tests/models/qwen_generator_test.cpp` pattern, extended to GPU | seconds |
| L3 pipeline | real model, real prompts | `strix-server bench` (existing protocol) | minutes |

> **L2-GPU is a REQUIRED Phase 3 deliverable** (moved from Phase 5 optional):
> whole synthetic model through the GPU executor vs CPU generator. It is the
> literal "many-module" gate — without it, cross-module HIP interaction is only
> caught at L3 `bench`.

L1 is the workhorse for the optimization loop. Per module test, concretely:
1. Deterministic synthetic inputs (fixed-seed RNG in a shared header).
2. Build module inputs from `ModelConfig` (small: hidden 128–512, 1–3
   layers, short context).
3. Run HIP module, run CPU module on identical inputs.
4. Assert with `CompareLogits`-style envelope (reuse
   `src/testing/compare/logit_comparator.hpp`; consider moving it out of
   `src/` into `tests/testing/` or a `strix_test` helper library).
5. Time the HIP module — but NOT through the full executor. See §6: timing
   uses a tight direct-launcher loop for that module's kernels only
   (plan cache warm, no prefetch), so the number is per-module, not the
   whole per-layer pipeline (which includes the prefetch side-stream and
   other layers).
6. Print `MODULE <name>: PASS  correctness=...  time=<us>  baseline=<us>
   delta=<+/->%`. Optional: compare against a recorded baseline file
   (`artifacts/baselines/<module>.json`) so "did we gain?" is a number.

Why one binary per module (not one big test file):
- `tests/kernels/qwen_gpu_ops_test.cpp` is 3833 lines in one TU with 9 local
  duplicated quant-block structs (Q8K, Q8_0, Q5_K, Q6_K) — it is slow to
  compile and slow to run,
  which is exactly what kills iteration speed. Per-module binaries compile
  independently and run independently.
- CMake already does per-file `add_executable`+`add_test`+LABELS — add
  `tests/modules/CMakeLists.txt` or a generated list; label them
  `module;<name>` so `ctest -L module` runs the whole L1 tier and
  `ctest -R ssm_module` runs one.

## 4. Structural enablers (refactors, risk-ascending)

These are what make §2/§3 possible. Order = dependency order.

1. **Hygiene + shared test header.**
   `tests/testing/` header: seeded RNG, tensor helpers, timing loop,
   baseline load/store. Remove `tests/tools/__pycache__/`, gitignore it.
2. **Single source of truth for quant block layouts.** `ggml_dequant.hpp`
   currently declares only FUNCTIONS (DequantizeQ4_K/Q5_K/Q6_K/Q3_K/Q8_K/Q8_0,
   DotProduct*, QuantizedRowBytes, Fp16ToFloat), not the struct layouts.
   Canonical block_q* layouts already exist INTERNALLY in
   `ggml_dequant.cpp:39-75`; HIP `static_assert`'d layouts already exist in
   `src/core/hip/qwen_gpu_quant_ops.hpp` (Q8_0=34, Q5K=176, Q6K=210, Q8K=292
   bytes). The real task is to RECONCILE the divergent definitions
   (`qwen_state.hpp` local 256-wide `Q8KBlock` vs HIP 292-byte vs
   `ggml_dequant.cpp` `block_q8_K`), expose one canonical set in the header
   with `static_assert`, then delete the local copies (12 verified in-scope:
   `qwen_state.hpp:42,55` = 2; `qwen_gpu_ops_test.cpp` = 9 incl Q5_K/Q6_K;
   `qwen_mtp_xdna2_eh_proj_test.cpp:31` = 1; `qwen_mtp_xdna2_rmsnorm_test.cpp`
   has 0, only `ErrorMetrics`). `static_assert` byte-size is NOT sufficient
   proof of field-order equivalence — add per-type parity tests (Phase 1.3).
3. **One quantized-GEMM dispatch.** `quant::Gemm(GgmlType, ...)` /
   `quant::Dequantize` routing through `ggml_dequant.hpp` (adds Q8_K, Q8_0,
   Q5_K); delete the per-file switches in `qwen_forward.cpp`
   (`QuantizedDot`/`DequantizeRow`) and make `QwenTensorRef::Get` fail
   loudly (UNREACHABLE/assert) instead of returning 0.0F for unsupported
   types — cover BOTH silent-failure sites (`Get()` + the GEMV dispatch) and
   the `QuantizedRowBytes`=0 guard. This becomes the `quant_gemm` module.
   **Sequence after the in-flight prefill GEMM work settles** (see §7):
   `qwen_gpu_prefill.cpp`/`qwen_gpu_prefill_ops.hip` were just rewritten
   (Phase B direct-quant prefill, pp128 92.71→2.61 tok/s) and this dispatch
   overlaps exactly that path.
4. **Re-home oracles as CPU module backends.** `qwen_oracles.{cpp,hpp}`
   (namespace `strix::models::qwen`) + the CPU forward bodies become the CPU
   backend implementations of `norm`/`rope`/`ssm`/`ffn` modules. The GPU
   decode path (`qwen_gpu_decode.cpp`) does NOT call `qwen::Reference*`; the
   point is to stop carrying the CPU reference code as part of the GPU-path
   library and instead treat it as a module's CPU backend/oracle.
5. **Extract module functions from the mega-functions.**
   `QwenGpuExecutor::ForwardToken` decode loop → per-layer call into
   `attn`/`ssm`/`ffn`/`norm`/`residual` module functions; CPU
   `ForwardLayer` (96 lines) / `ForwardSSM` (177 lines) split the same way.
   Behavior-identical first (characterization tests exist: the L1 tests run
   green before and after — this is the safety net the refactoring workflow
   requires).
6. **`QwenScratchArena` / `QwenKvCache` encapsulation.** Private spans +
   accessors; `QwenKvCache::AdvancePos` bounds-checks. Feeds the
   `ModuleCtx` object.
7. **Break the forward↔ssm include tangle** (`qwen_ssm.cpp` includes
   `qwen_forward.hpp` for `ForwardRMSNorm`; `qwen_forward.hpp` includes
   `qwen_ssm.hpp`). Disappears naturally once norm is its own module header.
8. **Move Qwen code to `src/models/qwen35/`** (match `minimax_h3/`,
   `deepseek_v4_flash/` convention) with a `modules/` subdirectory; update
   CMake (38 KB single file — add `src/models/qwen35/CMakeLists.txt` via
   `add_subdirectory` as part of this move).

## 5. Smell inventory (evidence for the refactors above)

Fowler-catalog mapping from `refactoring-patterns` skill:

| Smell | Where | Fix (named refactoring) |
|---|---|---|
| Long Method | `ForwardSSM` 177 lines, `ForwardLayer` 96 lines, `ForwardToken` decode mega-loop | Extract Method per module (§4.5) |
| Switch Statements | `QuantizedDot`/`DequantizeRow` on `GgmlType`; silent 0.0F — Q8_K/Q8_0/Q5_K in the GEMV dispatch, Q5_K/Q6_K in `Get()` | Replace Conditional with Strategy — single `quant::Gemm` dispatch covering both sites + the `QuantizedRowBytes` guard (§4.3) |
| Duplicate Code | 12 local quant-block struct copies (2 `qwen_state.hpp` + 9 `qwen_gpu_ops_test.cpp` incl Q5_K/Q6_K + 1 `qwen_mtp_xdna2_eh_proj_test.cpp`); oracles vs forward logic overlap | Extract to `ggml_dequant.hpp` (§4.2); oracles → CPU backend (§4.4) |
| Data Clumps | `(layer, kv_cache, ssm_cache, arena, config, pos)` passed everywhere | Parameter Object `ModuleCtx` + `LayerView` (§2) |
| Large Class / Divergent Change | `qwen_state.hpp` holds 6 unrelated types; `QwenScratchArena` one struct for every sub-stage's buffers | Extract Class → per-module state; arena → factory-built named spans (§4.6) |
| Inappropriate Intimacy | `qwen_ssm.cpp` → `qwen_forward.hpp`; CPU `qwen_forward.cpp` calls `qwen::Reference*` (these are CPU-reference-internal calls, NOT the GPU hot path) | Module split removes both (§4.4–4.7) |
| Primitive Obsession | `QwenTensorRef` = `void*` + enum + size, with inline dequant in `Get()` | Keep as a value type but move all dequant behind `quant::` (§4.3) |
| Feature Envy | Free `Forward*` functions operating entirely on `QwenTensorRef`/caches | Move Method into module objects (§2) |
| Misfiling | `qwen_oracle_test.cpp` is ~80% logit-comparator test; oracles tested under `tests/models/` for generic kernels; `src/testing/` lives in `src/` | Re-file with modules; `tests/modules/` + `tests/core/` |
| No test framework | every test = hand-rolled `main()`+`assert`; `assert` dies under `NDEBUG`; no shared fixtures | Adopt `cpp-testing` skill path: GoogleTest via FetchContent + `gtest_discover_tests` (or, minimal: shared `tests/testing/` header) |
| Speculative Generality | `QwenTensorRef::Get` supports types the CPU path never loads | Fold into §4.3 dispatch |

Test coverage gaps: every `src/models/qwen_*` file has a CPU test
(`qwen_forward/s sm/generator/oracle(s)_test`), but **L1 module e2e tests
against the HIP path do not exist as isolated modules** — today the GPU
counterparts are entangled in `qwen_gpu_ops_test.cpp` (one 3833-line TU)
and `qwen_mtp_gpu_test.cpp`. `tests/kernels/persistent_deltanet_test.cpp`
and `tests/kernels/qwen_attention_policy_test.cpp` are the closest
precursors and should be folded into `ssm_module`/`attention_module` tests.

## 6. Open questions (resolved)

- **GPU module-e2e: real executor vs direct launcher** → **Split it.**
  Correctness runs through the REAL executor on a 1-layer synthetic model
  (matches production launch structure: graph capture, plan cache, arena);
  timing runs through a tight direct-launcher loop for the module's kernels
  ONLY (plan cache warm, no prefetch). Resolves the §3 step-5 muddiness.
- **GoogleTest vs hand-rolled** → **Defer gtest to Phase 5.** Nix is the only
  build; gtest = a fetchable nix dep (churn + slower iteration), defeating
  the sub-second goal. Use the shared `tests/testing/` header + hand-rolled
  `main()`. gtest stays its own bookmark/project.
- **Baseline storage** → `artifacts/baselines/<module>.json`, **tracked in
  git**, keyed by a model-config hash. Recording gated behind an env flag
  (`STRIX_RECORD_BASELINE=1`) so normal runs don't churn the JSON. Owner =
  the module author at phase boundaries / on an accepted optimization
  (per optimize-kernel contract).
- **CPU `QwenGenerator` roadmap** → Keep alive ONLY as the CPU module
  backend + oracle. Not a supported production path (server uses the GPU
  executor). Do not over-invest.
- **NDEBUG / asserts live** → **RESOLVED.** All test presets
  (cpu-test, gpu-test, cpu-sanitizer, development) are
  `CMAKE_BUILD_TYPE=Debug` + `BUILD_TESTING=ON` (CMakePresets.json); no
  `NDEBUG` in CMakeLists.txt/CMakePresets.json/flake.nix; flake testCheck
  sets Debug (flake.nix:409). The existing hand-rolled `assert` tests ARE a
  valid safety net.
- **`QwenTensorRef` lifetime** → Production concern only (synthetic weights
  bypass the reader). Recommend `QwenModelWeights`/executor hold a keep-alive
  ref (e.g. `shared_ptr<GgufReader>`) and assert it outlives the executor in
  the ctor. Add to Phase 1 as correctness hardening.

## 7. Safe refactoring plan (jj, green-to-green)

VCS is **jj** (not git): one jj commit per step on a `refactor-qwen-modules`
bookmark; `jj restore` / `jj abandon` to roll back; `jj split` when a commit
accidentally mixes two concerns. The pre-refactor commit stays as the
rollback anchor until the bookmark lands.

### Safety gates (what proves "nothing broke")

| What could break | Gate | Cost | Run when |
|---|---|---|---|
| CPU numerics/build | `nix build .#checks.x86_64-linux.tests` (Debug CPU build, `BUILD_TESTING=ON`, full ctest) | low (ccache) | **every commit** |
| GPU numerics | `cmake --build --preset gpu-test` + `ctest --preset gpu-full` (also the `qwen-gpu-kernel-oracle` preset) | high (HIP) | commits touching `src/core/hip/` or GPU-visible APIs |
| GPU launch structure | `STRIX_DISPATCH_TELEMETRY=1 ... bench -p 16 -n 16` -> expect `miss_captured` then `hit` | low | any commit changing launch structure in decode |
| End-to-end quality | `strix-server bench --validate-prefill 1024 --n-prompt 1024 --n-gen 0` — envelope must not regress (current: rmse `0.02007260`, cosine `0.99997753`, top-1 `198`) | **very high** | phase boundaries only |
| End-to-end perf | interleaved A/B `strix-server bench` (`-p 2048 -n 128`, decode-only, prefill-only) | high | phase boundaries only |
| CPU memory bugs | `cpu-sanitizer` preset (ASAN) | medium | phases 1-2 (structural CPU moves) |

Rules:
- **Never refactor red.** Gates green before starting a step and after.
- One concern per commit. A commit touching both layout and dispatch is two commits.
- Red step -> `jj restore` the files, redo smaller. Do not debug forward on a broken intermediate state.
- Per-commit: tests. GPU ctest only when HIP code touched. Logits envelope + bench A/B **only at phase boundaries** (logits comparison is extremely slow - never per commit).
- Untouched during the whole refactor: `src/core/xdna2/` (NPU), `src/server/`, `src/tokenization/`, `minimax_h3/`, `deepseek_v4_flash/`.
- **Coordinate (do not expand): `qwen_gpu_prefill.cpp` / `qwen_gpu_prefill_ops.hip`** — just rewritten (Phase B direct-quant prefill, pp128 92.71→2.61 tok/s per `task-on-going.md`); §4.3 quant dispatch overlaps this path. Sequence §4.3 after the prefill work settles to avoid a merge conflict and to keep the refactor's perf gates from being confused by a known-regressed state.

### Phase 0 - Freeze the baseline (no code change)
> The tree is currently PERF-red (in-flight prefill pp128 regression). Phase 0
> will therefore freeze a known-regressed perf state. Decouple the refactor's
> perf gates from the in-flight prefill fix so the two efforts don't get
> confused; treat Phase 0 perf numbers as a snapshot of a transient state.
1. `jj new -m "qwen refactor baseline"`; record revision.
2. Run + record all gates: tests, gpu-full, capture smoke, bench envelope + headline numbers (per optimize-kernel skill: fingerprint, revision, raw samples). Store in `task-on-going.md`.
3. Verify asserts are live in test builds (no `NDEBUG` in `cpu-test`/`gpu-test` configure paths) - the existing hand-rolled tests are the safety net only if asserts are live.

### Phase 1 - Expand (purely additive; zero behavior change)
Only new files/symbols. Old code byte-identical -> cannot break behavior.
1. `tests/testing/` shared header: seeded RNG, tensor builders, timing loop (N repeats, warmup, median), baseline JSON load/store, `CompareLogits` wrapper.
2. Reconcile + expose one canonical block-layout set in `src/core/quant/ggml_dequant.hpp` (`Q8KBlock`, `Q8_0Block`, `BlockQ4K`, ...) with `static_assert` on sizes, drawn from the HIP static_assert'd source (`qwen_gpu_quant_ops.hpp`) and the internal `ggml_dequant.cpp` layouts. Local copies untouched.
3. New `quant::Gemm` / `quant::Dequantize` dispatch (Parallel Change - sits next to the old switches), plus new unit tests proving parity with the old `QuantizedDot`/`DequantizeRow` for every type the old code handles. These parity tests must compare DEQUANT OUTPUT (not just byte size) against the old path for every type the old code handles — this is the guard against a silent numerics change when unifying the divergent Q8K/Q8_0 layouts (see §4.2).
4. `src/models/qwen/modules/` headers: `ModuleCtx`, `LayerView`s, module function declarations (`SsmForward`, `AttnForward`, `FfnForward`, `NormForward`, ...). Nothing wired yet.
5. Module functions as **thin wrappers** around existing code (new function calls the old one; no caller changes). `ssm` first.
6. **Synthetic-weights builder** (no GgufReader): a helper to build `QwenModelWeights` for a small synthetic 1–3 layer model from in-memory tensors. The whole L1 tier depends on this seam (the executor normally builds weights from a `GgufReader`; there is no existing in-memory path). Also add the `QwenTensorRef` reader keep-alive (`shared_ptr<GgufReader>`) + ctor assert per §6.
Gate: tests green, zero test output changes.

### Phase 2 - Migrate (callers move, behavior identical)
1. Replace the 12 local block-struct copies with the canonical ones, one file per commit (static_asserts make it mechanical and provably safe).
2. Two-step quant cut-over:
   a. add debug counters/asserts logging which `GgmlType` reaches `QwenTensorRef::Get` vs `QuantizedDot`/`DequantizeRow` across the existing tests; run the suite; record which types hit each site (evidence for step b);
   b. switch the call sites to `quant::` dispatch; replace the silent `0.0F` fallback with loud failure (UNREACHABLE + comment citing the step-a evidence) — cover `Get()`, the GEMV dispatch, AND the `QuantizedRowBytes`=0 guard. Only deliberate behavior change in the whole refactor, and it fires only on paths that were already broken.
3. Extract module bodies out of `qwen_forward.cpp` / `qwen_ssm.cpp` into the module .cpp files, one module per commit; the forward<->ssm include tangle dissolves when `norm` gets its own header.
4. Re-home `qwen_oracles.{cpp,hpp}` as the CPU backends of the norm/rope/ssm modules. Still compiled into `strix_core`, no library change.
5. HIP side: `ForwardToken` per-layer loop calls module functions (which currently wrap the old inline bodies); policy toggles unchanged.
Gate: tests green every commit; GPU ctest for commits in 2.5; ASAN through 2.1-2.4; envelope + short bench (`-p 128 -n 16`) at phase end.

### Phase 3 - Wire the fast loop (the payoff)
1. L1 `ssm` module e2e test: synthetic 1-3 layer model, GPU vs CPU backend, `CompareLogits` envelope, timing (via the direct-launcher loop per §6) + baseline file under `artifacts/baselines/`. Target < 1-2 s.
2. Register with CTest labels `module;ssm` (pattern: `ctest -R ssm_module`).
3. **Sensitivity check**: make a deliberate micro-change behind a toggle (e.g. flip `ShouldFuseSSMGateResidual`), confirm the test output shows a real correctness and/or timing delta, then revert. A harness that cannot see an intentional change is not a harness. Also prove **many-module detection**: deliberately break a SHARED component (e.g. bump `quant::Gemm` scaling by `1e-3`) and confirm EVERY dependent module's L1 test fails, then revert. A harness sensitive only to a single module's change is insufficient — the shared-component break is the proof that global blast radius is detectable.
4. Replicate L1 tests to all modules (attn, ffn, norm, rope, quant_gemm, residual, embed/unembed, sample); re-file `persistent_deltanet_test` and `qwen_attention_policy_test` content into `ssm_module`/`attention_module`.
5. **L2-GPU integration test** (whole synthetic model through the GPU executor vs CPU generator) — the literal "many-module" gate. REQUIRED for the fast loop (moved from Phase 5 optional). Without it, cross-module HIP interaction is only caught at L3 `bench`, which is too slow to be a regular gate.
Gate: gpu-full green; module envelopes match Phase 0 baseline.

### Phase 4 - Organize (mechanical moves, lowest urgency)
1. `src/models/qwen35/` subdirectory + `add_subdirectory` in the root CMakeLists (keep flat until this phase - moving files is the riskiest "obviously safe" step; do it once, tests after).
2. `tests/modules/CMakeLists.txt` for the L1 tier; keep root-list per-test entries for the rest.
3. Re-file tests: `qwen_oracle_test.cpp` -> `tests/core/logit_comparator_test.cpp` (it is 80% logit-comparator coverage); oracles test follows the CPU backends. Delete `tests/tools/__pycache__/` (jj honors `.gitignore`).
4. Contract: delete the old local structs/switches now that nothing uses them (the Parallel Change cleanup half).
Gate: full flake check set (format, static-analysis, tests) + gpu-full.

### Phase 5 - Optional, separate project
- GoogleTest adoption per the cpp-testing skill (FetchContent + nix change) - its own bookmark, its own gates.
- ~~L2 GPU integration test~~ → moved to Phase 3 (required "many-module" gate).
- Full `strix-server bench` A/B per future optimization change (that is the optimize-kernel loop's job, not the refactor's).

### Why this order
- Additive before subtractive: phases 1-4.1 only ever add or move; the only deletion is in 4.4, after evidence exists that nothing references the old paths.
- Expensive gates are frequency-matched: cheap gate per commit, GPU gate per HIP commit, expensive gates per phase. Total baseline cost stays bounded while step count grows.
- `ssm` is the FEATURE canary for both the harness (3.1) and the extraction
  pattern (1.5/2.5): if one module goes through cleanly, the other eight are
  repetitions, not experiments. But ssm is also the HARDEST module (most
  cross-module fusion + capture/replay coupling). If the immediate goal is to
  de-risk the L1 harness cheaply, standardize the harness on the stateless
  **norm** module first; keep ssm as the feature canary.

## 8. Review findings

A review pass (verified against the codebase) corrected the plan's factual
anchors and folded in design refinements. Sources:
- `scout-plan-verify.md` — fact verification
- `reviewer-plan-refine.md` — design critique + refined plan

Corrected facts (E1–E7): `ForwardSSM` is in `src/models/qwen_ssm.cpp:64` (not
`qwen_forward.cpp`); `qwen_forward.cpp:125,131,144,149` are CPU-reference-internal
calls, NOT the GPU hot path (GPU decode is `qwen_gpu_decode.cpp`); the in-scope
block-struct count is 12 (not 13), and `qwen_gpu_quant_ops.hpp` is the best
canonical source, omitted before; `ggml_dequant.hpp` declares functions only
(layouts live in `ggml_dequant.cpp`); "silent 0.0F" is two distinct sites
(`Get()` vs the GEMV dispatch) plus a `QuantizedRowBytes` guard; NDEBUG/asserts
are live (resolved); `qwen_gpu_prefill.*` added to the coordinate list.

Design refinements folded in: `ModuleCtx` redefined to carry the named
scratch arena (a raw workspace pointer cannot express shared, cross-module
arena slices); a fusion-ownership rule added (two existing toggles cross
module boundaries); correctness=executor vs timing=direct-launcher split;
capture-transparency constraint; synthetic-weights builder added to Phase 1;
per-type dequant-parity tests to guard the divergent Q8K/Q8_0 layouts.

Two top blockers to resolve before Phase 0:
1. Several incorrect factual anchors that the smell inventory and phases rest
   on (now corrected here).
2. `ModuleCtx` (shared-arena) + undefined fusion ownership are unimplementable
   as originally written (now specified).
