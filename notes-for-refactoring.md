# Qwen architecture and fast experimentation roadmap

## Purpose

The Qwen implementation should make performance work on AMD Strix Halo a short,
auditable loop:

1. state a hypothesis;
2. select one execution route through an explicit policy;
3. rebuild only the affected Qwen component;
4. compare the candidate with a reference route;
5. measure it on `gfx1151`;
6. keep or reject it with enough route and configuration metadata to reproduce
   the result.

The production target is Linux x86-64 on Strix Halo. CPU implementations are
correctness oracles and test backends, not the supported inference path.

This document describes the architecture after the module-oriented refactor and
the remaining work. It is a living roadmap, not a record of individual local
benchmark runs.

## Current state

The refactor has established the following foundations:

- Qwen-specific code is consolidated under `src/models/qwen/`.
- CPU-oriented stage interfaces live under `src/models/qwen/modules/`.
- HIP runtime, composition, MTP, and kernels live under
  `src/models/qwen/hip/`.
- XDNA2 code lives under `src/models/qwen/xdna2/`.
- Large HIP translation units were split into concern-oriented kernel files.
- Quant block layouts are canonicalized in `src/core/quant/ggml_dequant.hpp`.
- Shared quantized GEMM/dequantization entry points exist in
  `src/core/quant/ggml_gemm.*`.
- `ExecuteDecodeStep` is the decode composition root in its own translation
  unit; `decode.cpp` owns token I/O and graph orchestration.
- HIP launch declarations are split by operation family under
  `src/models/qwen/hip/ops/`; `ops.hpp` remains a compatibility umbrella.
- GGUF weight extraction and tensor validation live in `weights.cpp`, separate
  from CPU cache and scratch state.
- GPU allocation ownership remains in `arena.cpp`, while recurrent snapshot and
  SSM replay behavior live in `ssm_replay.cpp`.
- Immutable `QwenExecutionPolicy` data selects production or experimental
  routes, with distinct decode/prefill SSM and FFN decisions.
- `ResolveQwenLayerRoute()` produces pure per-layer decision records with
  stable fingerprints; policy fingerprints are available in dispatch telemetry.
- CPU and HIP module contexts are distinct types rather than a nullable
  backend-tagged union.
- `QwenGpuArena` exposes typed non-owning scratch spans without changing the
  allocation addresses used by graph capture.
- Deterministic synthetic Qwen weights and shared comparison/timing helpers
  exist for module tests.
- Tests now mirror production under `tests/models/qwen/`, while preserving
  existing CTest executable names.
- Norm, FFN, SSM, fusion-route, quant parity, and partial HIP integration
  coverage exists.

The refactor is not complete merely because files were moved. Attention and SSM
still contain transitional composition seams, raw GPU arena pointers remain
public beside the typed view, and the isolated `prefill_chunk.cpp` composition
still owns a broad direct launch chain that should become stage plans. HIP
operation coverage is split by kernel family so each
experiment has a focused build and CTest target.

## Architectural rules

### Composition owns cross-stage fusion

A module represents one stage: norm, attention, SSM, FFN, residual, RoPE,
embedding, unembedding, sampling, or quantized projection. A fusion spanning
multiple stages belongs to the decode/prefill composition layer.

Examples owned by composition:

- RMSNorm + projection;
- residual add + RMSNorm;
- Q/K norm + RoPE + KV write;
- SSM output + residual;
- fused RMSNorm + SwiGLU projection.

Modules must keep an unfused reference route. A policy change must not require
editing a module's mathematical contract.

### Policies are data, not scattered compile-time decisions

`QwenExecutionPolicy` is immutable inside an executor and is resolved into a
pure `QwenLayerRoutePlan` before each layer launch chain. It carries independent
decode and prefill decisions, preserves current production defaults, and emits
a stable policy fingerprint through dispatch telemetry.

Remaining policy work is to validate incompatible combinations, include policy
identity in every graph-cache key, report route rejection reasons, and expose a
controlled same-binary A/B selection surface without reading mutable process
state during capture. A benchmark result without its resolved route IDs and
policy fingerprint is not reproducible evidence.

### Backends expose capabilities explicitly

The former backend-tagged `ModuleCtx` has been replaced by explicit capability
types:

- `CpuModuleContext` for stateless CPU stages;
- `CpuLayerContext` for CPU stages that require configuration, stable scratch,
  and layer/position metadata;
- `HipModuleContext` for HIP launches and their stream.

Reference members make incomplete stateful CPU contexts unrepresentable, while
C++ overload selection chooses CPU or HIP behavior without a runtime backend
tag. Future HIP module extraction can add narrow scratch and GEMM capabilities
to `HipModuleContext`; nullable fields must not return as backend selectors.

### Scratch aliases are typed and address-stable

HIP graph capture and replay depend on stable device addresses. Refactoring the
arena must not introduce allocation into module calls or change buffer
lifetimes.

`QwenGpuArena::GetScratchView()` now exposes non-owning typed spans over the
existing stable allocations, and decode module calls consume those spans. This
is an additive parallel change: raw pointers remain for kernels not yet migrated.
The next step is to split the broad view into `DecodeScratch`,
`AttentionScratch`, `SsmScratch`, and `FfnScratch`, document legal aliases and
live epochs, migrate remaining call sites, and only then make raw pointers
private.

### Dispatch has one source of truth

Tensor format support, packed row stride, physical byte size, and CPU/HIP GEMM
route selection must not be reimplemented in model call sites.

The canonical quant layer should own:

- supported format descriptors;
- elements and bytes per block;
- physical byte size and row stride;
- dequantization and dot-product dispatch;
- loud failure for unsupported or misaligned formats.

Model loading must validate formats by tensor role. Accepting a type globally is
unsafe when embeddings, norms, projections, and recurrent parameters have
different consumer capabilities.

### Decode and prefill share decisions, not executors

Decode and prefill need different kernels and performance strategies. They
should remain separate executors, but use a common pure route resolver:

```text
(policy, mode, tensor formats, shape, capabilities)
    -> DecodeLayerPlan or PrefillLayerPlan
    -> stable route IDs and rejection reasons
```

This prevents a policy from being wired only in decode or only in prefill while
preserving mode-specific implementations.

## Intended module topology

```text
embed
  -> repeated layer plan
       -> pre-norm
       -> attention or SSM
       -> residual
       -> FFN norm
       -> FFN
       -> residual
  -> final norm
  -> unembed
  -> sample
```

The composition root selects fused or unfused edges. Leaf module implementations
select kernels only within their own stage.

The remaining extraction work is:

1. split CPU and HIP backend implementations into explicit files/namespaces;
2. replace the SSM forwarding shim and remove `SsmLayerView::source`;
3. expose complete HIP attention and SSM module operations;
4. route decode and prefill through plans rather than direct launch chains;
5. consolidate GEMM selection across CPU, decode, prefill, and MTP;
6. make route selection pure and independently testable.

## Build boundaries

A kernel experiment should not require editing the repository root build list or
recompiling unrelated model sources. Qwen should own CMake registration through
subdirectories or helper functions, without source globbing.

Target seams should distinguish at least:

- Qwen CPU/reference modules;
- HIP kernel objects;
- HIP runtime/composition;
- MTP;
- XDNA2;
- test support and focused test executables.

The final `strix_core` interface can remain stable while internal object-library
boundaries reduce incremental HIP build churn.

## Test architecture

Tests mirror production ownership under `tests/models/qwen/`:

```text
tests/models/qwen/
├── cpu/             # model-level CPU/reference behavior
├── modules/         # extracted stage contracts
├── hip/             # policy, kernel, MTP, and integration coverage
├── mtp/             # CPU MTP reference behavior
├── tokenization/    # tokenizer and chat-template behavior
├── xdna2/           # XDNA2 MTP kernels
└── support/         # deterministic Qwen-only synthetic weights
```

`tests/models/qwen/README.md` documents focused CTest labels and current GPU
integration limitations. Generic quant and comparison helpers remain under
`tests/core/` and `tests/testing/`; ownership, not filename, determines
placement.

### Testing tiers

| Tier | Purpose | Expected use |
|---|---|---|
| CPU unit | Pure math, policy, parsing, and module oracle checks | Every change |
| HIP kernel | One kernel family against an independent CPU reference | Kernel work |
| Module | One complete stage through CPU and HIP backends | Module work |
| Integration | Production composition/executor across several stages | Cross-stage work |
| Quality | Real model, logits, and tokens | Phase/release gate |
| Experiment | Route equivalence and controlled timing evidence | Optimization loop |

The existing HIP module-pipeline test is useful but is not a complete synthetic
GPU decode. Until embedding, attention, SSM, unembedding, and sampling traverse
the production module seam, it must be named and documented as a partial module
pipeline rather than a whole-model L2 gate.

### Minimal test support, not a second framework

CTest remains the runner. A new macro-heavy registration framework is not
needed. Shared C++ support should provide always-on checks with source locations,
explicit skip handling, deterministic fixtures, comparison helpers, and HIP
RAII:

```cpp
Check(condition, "message");
CheckEq(actual, expected, "message");
CheckNear(actual, expected, tolerance, "message");
CheckSpanNear(actual, expected, tolerance, "message");
return RunTests({{"case name", TestFunction}, ...});
```

Checks must not disappear under `NDEBUG`. Hardware/model absence should map to
CTest skip code 77 only for tests explicitly allowed to skip.

A CMake helper should preserve current CTest names while adding consistent
labels such as:

- `qwen;cpu;unit`;
- `qwen;hip;kernel`;
- `qwen;hip;integration`;
- `qwen;xdna2`;
- `qwen;quality;external-model;slow`;
- `qwen;experiment;performance`.

## Experiment contract

Every optimization should provide:

1. hypothesis and affected route;
2. current and candidate route IDs;
3. deterministic correctness comparison;
4. capture/replay compatibility when applicable;
5. interleaved A/B measurements on production binaries;
6. device, revision, model, shape, and policy fingerprints;
7. explicit keep/reject decision.

Timing tests should report by default. They become gates only after variance and
thresholds are characterized on controlled Strix Halo hardware.

## Implementation sequence

1. Rebase and remove machine-specific run logs.
2. Make tensor-role validation and physical-size metadata correct.
3. Organize Qwen test support and mirror production ownership. **Done.**
4. Introduce stable route fingerprints and pure decision records without
   changing launches. **Done.**
5. Introduce immutable execution policy with current defaults. **Done;** graph
   cache identity still needs the policy fingerprint.
6. Add typed GPU scratch views while preserving addresses and aliases. **Broad
   view done;** narrower lifetime-specific views remain.
7. Split backend contexts. **Done;** physical CPU/HIP implementation-file
   separation remains.
8. Complete attention and SSM module extraction.
9. Consolidate GEMM dispatch by parallel change: CPU, decode, prefill, then MTP.
10. Keep decode/prefill on shared pure route resolution and grow thin
    mode-specific plans. **Initial layer plan and decode-step translation-unit
    seam done;** narrower attention/SSM/FFN composition remains.
11. Split Qwen CMake ownership and the monolithic GPU operations test. **GPU
    operation tests done;** source-level Qwen CMake ownership remains.
12. Enable runtime experiment overrides and same-binary A/B only after policy
    identity participates in capture caches.

Each step should be independently reviewable. On supported hardware, the
validation ladder is: focused CPU checks, focused HIP kernel/module checks,
synthetic integration, capture miss-to-hit smoke, real-model quality, then
production-binary A/B benchmarks.

## Current validation limitation

The continuation work was prepared on Apple Silicon, which is not a supported
build or execution target for this repository. Builds, tests, HIP/XDNA2 checks,
and model benchmarks must therefore be run later on Linux x86-64 Strix Halo.
Static review cannot establish numerical equivalence, graph-capture correctness,
or performance.
