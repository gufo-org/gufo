# Card 10: Publish and measure the public extension API

Status: **complete for in-card gates** (deferred functional harness pending post-Card-14 qualification window). RFC step and dependencies: **7; 09**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Forwarding headers can expose a large transitive core/cache closure. Directory names alone do not define a stable API; measure and govern the declarations actually reachable before promising compatibility.

### Original scope

Classify every model-used core header: stable/platform/experimental forwarding API, host wrapper or private. Publish headers and generated transitive `include/gufo/ext/SURFACE`, measure paths/declarations and coupling before committing stability. Enforce changes via version/CHANGES/ext-api gate, including forwarded src/cache headers. Test transitive closure and declaration changes; record the measured surface and any narrowing required.

## Implementation plan

1. Enumerate every core header used by model code and classify it as stable API, platform API, experimental forwarding, Host wrapper or private; record rationale and intended owner.
2. Publish forwarding/contract headers under include/gufo/ext, including platform leaves and experimental contracts, without accidentally exposing serving policy.
3. Generate the full transitive SURFACE using tools/ci/check-extension-api.py --write; measure paths, declarations and coupling, including forwarded src/core and cache declarations.
4. Extend version/CHANGES/ext-api diff tests for transitive additions/removals and forwarded declaration changes; narrow wrappers or explicitly document unresolved coupling before dependent include migrations.

## Definition of done

- [x] Every inventoried model-used header has a classification and public replacement or approved private removal plan.
- [x] Measured SURFACE closure is reproducible; forwarded source declarations are governed equally with public headers.
- [x] Public CMake/schema contracts and API version/changelog gates reject missing required policy changes; no unsupported stability claim is made from wrapper paths alone.

## Validation criteria

- Run python3 tools/ci/check-extension-api.py --write, inspect the resulting closure and then run the check without --write; output must be stable.
- Run python3 tests/tools/check_extension_api_test.py and focused extension schema/configure tests; add to-be-written fixtures for newly classified closure/declaration edge cases.
- Compile representative HIP-less and HIP public-header consumers, record declaration/path counts and confirm no serving/private headers leak beyond the explicitly published closure. Run formatting for changed C++ and both platform builds where affected.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Agent Sequence
- **Parent Commit**: `klwmvoxr a761679a`
- **Worker Run ID**: `6d5263a1-9f0d-46ef-a092-76804e8f1003`
- **Reviewer Run ID**: `922a7582-1ace-4ef7-b37c-e926d5df7c5b` (PASS, 0 findings)
- **Integrator Run ID**: `daeda205-c3f4-4d9f-a035-6467a64d8706`
- **Orchestrator**: Closure

### Implementation (Worker)

- **Agent**: Implementation Worker (`worker`), working copy `@` (`47a03291...` over parent `klwmvoxr a761679a`).
- **Inventory & Classification**: Produced [`header-classification.md`](../evidence/card10/header-classification.md). Classified every model-used header into stable (`include/gufo/ext/`), platform (`include/gufo/ext/platform/`), experimental (`include/gufo/ext/experimental/`), host wrapper (`host.hpp`), or private/internal (the 2 test-only ASR serving edges to be removed in Card 12).
- **Forwarding Headers**: Created 44 new forwarding headers:
  - 7 platform headers in `include/gufo/ext/platform/` (`fs.hpp`, `library.hpp`, `mapped_file.hpp`, `os_info.hpp`, `pages.hpp`, `prefault.hpp`, `publish.hpp`).
  - 14 experimental headers in `include/gufo/ext/experimental/` (`hip_utils.hpp`, `heap_placement.hpp`, `host_allocations.hpp`, `managed_alloc.hpp`, `snapshot_copy.hpp`, `snapshot_transfer.hpp`, `transfer_pool.hpp`, `weight_upload.hpp`, `dispatch_telemetry.hpp`, `gemv_dispatcher.hpp`, `hipblaslt_plan_database.hpp`, `hip_graph_decode_executor.hpp`, `ggml_dequant.hpp`, `ggml_gemm.hpp`).
  - 23 stable headers in `include/gufo/ext/` (`adapter.hpp`, `cache.hpp`, `ledger.hpp`, `types.hpp`, `cancellable_gate.hpp`, `crypto.hpp`, `sha256.hpp`, `gguf_reader.hpp`, `image.hpp`, `json.hpp`, `mapped_prefetch.hpp`, `model_config.hpp`, `reasoning.hpp`, `sampling.hpp`, `session_mode.hpp`, `utf8.hpp`, `draft_backend.hpp`, `prompt_lookup.hpp`, `speculative_verifier.hpp`, `continuation.hpp`, `device_probe.hpp`, `host_budget.hpp`, `text_contracts.hpp`).
- **Version & Governance**:
  - Bumped extension API version from `0.1.1` to `0.1.2` in `include/gufo/ext/version.hpp` (`kApiRevision = 2`, `kApiVersion = "0.1.2"`) and `cmake/GufoExtension.cmake` (`GUFO_EXTENSION_API_VERSION "0.1.2"`).
  - Documented `## 0.1.2` in `include/gufo/ext/CHANGES.md`.
- **Surface Measurement**:
  - Generated and verified `include/gufo/ext/SURFACE` (97 paths total: 49 public headers, 48 forwarded `src/` headers, 4,235 LoC, 31,365 tokens). Metrics documented in [`SURFACE-metrics.md`](../evidence/card10/SURFACE-metrics.md).
  - Confirmed zero serving leaks (`src/cli/`) and zero model include leaks.
- **Diff & Policy Tests**:
  - Extended `tests/tools/check_extension_api_test.py` covering transitive additions, transitive deletions, subdirectory declaration modifications, and closure resolution across `platform/` and `experimental/`. 7/7 tests passed.
  - Verified rejection when `ext-api` label is missing or version bump is absent.
- **Consumer Compilation**:
  - Updated `tests/extensions/ext_runtime_neutral_contracts_test.cpp` (compiled with stable and platform headers in standard C++20 without HIP).
  - Updated `tests/extensions/fixture/cpu.cpp` (compiled with representative stable and platform headers).
  - Updated `tests/extensions/fixture/hip.cpp` (compiled with representative experimental HIP headers under `ENGINE_ENABLE_HIP`).
  - Re-anchored `tests/extensions/runtime_test.cpp` and `tests/extensions/metadata_test.py` to API version `0.1.2`.
  - Built and passed all 7 CTest targets under both `build/cpu-test` and `build/gpu-test`.
- **Validation**:
  - All four isolation gates passed (`check-extension-isolation.py`).
  - Exact continuation identity goldens passed (3/3 in CPU and GPU, SHA256SUMS verified).
  - Formatting clean across 785 C++ files (`check-format.py`).
  - Docs check passed (`check_docs_test.py`).
  - Detailed build and test log retained in [`build-and-tests.txt`](../evidence/card10/build-and-tests.txt).
  - Deferred functional commands recorded in [`pending-commands.txt`](../evidence/card10/pending-commands.txt).
  - Implementation notes recorded in [`RESUME-NOTES.md`](../evidence/card10/RESUME-NOTES.md).
- Ready for independent review. Worker editing ownership is explicitly released.

### Review Findings and Dispositions

The [full independent review](../evidence/card10/review-findings.md) returned **PASS** with **0 findings** (clean conformance).

| Area | Review Verdict | Disposition & Proven Proof |
| --- | :---: | --- |
| 1. Scope & Cardinality | PASS | Exactly 44 forwarding headers created (7 platform, 14 experimental, 23 stable). Zero model files in `src/models/` or `src/families/` modified (deferred to Card 11). |
| 2. Forwarding Integrity | PASS | Every header strictly follows `#pragma once` + direct `#include "src/..."`. No inline code or macros. All 44 targets verified on disk. |
| 3. Serving Boundary & Isolation | PASS | Zero serving headers in `include/gufo/ext/` or `SURFACE`. `check-extension-isolation.py` passed all 4 checks (core boundary, serving boundary, target ownership, family graph). |
| 4. Governance & Versioning | PASS | Version bumped to `0.1.2` (`kApiRevision = 2`, `kApiVersion = "0.1.2"`) in `version.hpp` and `GufoExtension.cmake`. `CHANGES.md` updated. Policy gate rejects missing label (exit 1). |
| 5. SURFACE Reproducibility | PASS | 97 sorted unique paths. Programmatic closure matches exactly ($\emptyset$ diff). `check-extension-api.py --write` is idempotent. |
| 6. Unit Policy Tests | PASS | `check_extension_api_test.py` extended with 4 new test cases covering additions, deletions, subdirectory modifications, and closure resolution. 7/7 pass. |
| 7. Consumer Compilation | PASS | HIP-less (`fixture/cpu.cpp`, `ext_runtime_neutral_contracts_test.cpp`) and HIP (`fixture/hip.cpp`, `libalpha_hip.a`) compile cleanly. CTests pass 7/7 in CPU and GPU test builds. |
| 8. Continuation Identity | PASS | 3/3 passed in CPU and GPU (`qwen27b`, `qwen38_flash_next`, `deepseek_v4_flash`). 42/42 fixture checksums in `SHA256SUMS` match 100%. |
| 9. Repository Conventions | PASS | `RFC.md` remains byte-identical and uncommitted. No `<windows.h>` or `_WIN32` outside `src/core/platform/`. Formatting clean (785 files). Documentation clean (11 tests). |
| 10. Evidence Honesty | PASS | Deferred functional commands recorded in `pending-commands.txt` as PENDING per project decision. No unexecuted tests claimed as passes. |

### Validation and Acceptance Audit

The integration rerun ledger is retained in [`integration-reruns.txt`](../evidence/card10/integration-reruns.txt). The orchestrator closure transcript is retained in [`orchestrator-gates.txt`](../evidence/card10/orchestrator-gates.txt). The orchestrator executed the full pinned gate set after all edits:

1. `python3 tools/ci/check-extension-api.py`: **PASS** (97 paths, exit 0).
2. `python3 tools/ci/check-extension-api.py --base a761679a --labels '["ext-api"]'`: **PASS** (46 changed paths recognized, exit 0).
3. `python3 tools/ci/check-extension-api.py --base a761679a --labels '[]'`: **PASS** (Negative control: expected rejection with exit 1).
4. `python3 tests/tools/check_extension_api_test.py`: **PASS** (7/7 unit tests OK, exit 0).
5. `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership`: **PASS** (all 4 boundary and ownership gates pass, exit 0).
6. `python3 tests/tools/check_docs_test.py`: **PASS** (11/11 tests OK, exit 0).
7. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`: **PASS** (785 C++ files formatted, exit 0).
8. Bounded CPU CTest: **10/10 passed** (100%, exit 0), including all 7 extension suite tests and 3 continuation identity goldens.
9. Bounded GPU CTest: **10/10 passed** (100%, exit 0), including all 7 extension suite tests and 3 continuation identity goldens.
10. `nix build .#checks.x86_64-linux.pr`: **PASS** (exit 0).
11. `codex exec review`: **PASS** (0 critical, 0 major findings; exit 0).

#### Definition of Done Audit
- **DoD 1: Every inventoried model-used header has classification and public replacement or approved private removal plan.**
  - **Status**: **Proven**. Documented in [`header-classification.md`](../evidence/card10/header-classification.md). 41 unique `#include` paths classified: 23 stable, 7 platform, 14 experimental, 1 host wrapper, 2 private test-only ASR edges to be removed in Card 12. Automated scan confirms 0 unclassified headers.
- **DoD 2: Measured SURFACE closure is reproducible; forwarded source declarations are governed equally with public headers.**
  - **Status**: **Proven**. `include/gufo/ext/SURFACE` records 97 paths. Programmatic closure matches on-disk file with zero diff. Idempotence verified with `--write`. `tests/tools/check_extension_api_test.py` proves forwarded source declaration changes trigger policy gates identically to public header edits.
- **DoD 3: Public CMake/schema contracts and API version/changelog gates reject missing required policy changes; no unsupported stability claim is made from wrapper paths alone.**
  - **Status**: **Proven**. API version bumped to `0.1.2`, documented in `CHANGES.md` and enforced in `GufoExtension.cmake`. Policy gate rejects changes without `ext-api` label (exit 1). Experimental headers isolated under `include/gufo/ext/experimental/` with explicit no-stability warranties.

### Remaining Limitations

- **Deferred Functional Qualification**: Per the 2026-10-31 project decision, all end-to-end functional harness runs (`tests/functional/run.py` rows, `prompt_lookup.py`, standard speed benchmarks, and matched main-control timings) are deferred to the single post-Card-14 qualification window. Exact commands are preserved in [`pending-commands.txt`](../evidence/card10/pending-commands.txt).
- **Model Includes Migration**: Model and family code under `src/models/` and `src/families/` still reference internal `#include "src/..."` paths. Their migration to public `gufo/ext/` headers is the specific scope of Card 11.
- **ASR Test-Only Serving Includes**: `src/cli/serve/asr_service.hpp` and `src/cli/serve/audio_asr_api.hpp` remain referenced in `tests/models/qwen3_asr/` test harness, scheduled for removal in Card 12.
- **Windows Execution**: Forwarding headers and version bump are OS-neutral, but execution on native Windows remains pending.
- **RFC Integrity**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.
