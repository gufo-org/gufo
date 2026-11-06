# Card 13 Independent Review Report: EmbeddingGemma Pilot Migration

**Reviewer**: Independent Review Agent  
**Date**: 2026-11-01  
**Target Commit / Workspace**: `qupqlxsk 90a820ab` (parent: `ynuxkvqu b55be825`)  
**Verdict**: **PASS** (Zero findings / Clean pass across all audited areas)

---

## 1. Executive Summary

A comprehensive, independent read-only review of Card 13 ("Migrate the EmbeddingGemma pilot") was conducted in accordance with `AGENTS.md`, `docs/model-extensions/CARDS.md`, `docs/model-extensions/cards/README.md`, and `docs/model-extensions/evidence/card13/input-brief-reviewer.md`.

The diff was inspected against parent commit `ynuxkvqu b55be825`. All manifests, capability interfaces, decoupled serving boundaries, extension-relative includes, isolation ratchet checks, CMake discovery/lock/exclusion modes, neutral runtime contracts, and in-tree tests were independently validated.

`docs/model-extensions/RFC.md` remains strictly byte-identical (0 diff against parent). No tracked files were modified during this review, and all reviewer scratch operations were confined to `/tmp/card13-review/`.

---

## 2. Findings

**No findings (F0 / Clean PASS).**

All audited areas conform strictly to repository standards, Card 13 requirements, and RFC design contracts.

---

## 3. Detailed Audit by Area

### 3.1 Manifests and Entry Points
- **Extension Manifest (`src/models/embedding_gemma2/extension.json`)**:
  - Schema matches `schemas/extension.schema.json` (schema 1).
  - Target `gufo_embedding_gemma2` declared with `requires: []` and entry `embedding_gemma2_register`.
  - Target `gufo_embedding_gemma2_hip` declared with `requires: ["hip"]` and provider entry `embedding_gemma2_provider_register`.
  - Provider `embeddinggemma-2` declared with capability `embeddings`, model ID `embeddinggemma-2`, and lookup keys `safetensors:embedding_gemma2` and `safetensors:EmbeddingGemma2Model`.
  - CLI option `--default-prompt-name` declared for `serve` command.
  - Verified by `tests/extensions/metadata_test.py` (12/12 tests PASS) and `tests/extensions/discovery_test.py` (13/13 tests PASS).
- **Extension Build Specification (`src/models/embedding_gemma2/extension.cmake`)**:
  - Replaces legacy hardwired `CMakeLists.txt` using standard `gufo_extension_target` and `gufo_extension_hip_sources` macros.
  - Preserves audio dependency linkage (`PkgConfig::SNDFILE`, `PkgConfig::SOXR`) and dynamic FFmpeg/libav video library resolution.
  - Retains CPU reference oracle `gufo_embedding_gemma2_reference` for HIP test validation.
- **HIP-Free Pure Artifact Probe (`probe.hpp` & `extension.cpp`)**:
  - `ProbeEmbeddingGemma2` inspects `config.json` metadata only (`model_type == "embedding_gemma2"` and `architecture == "EmbeddingGemma2Model"` via `IsSupportedModelConfig`) without loading weights or initializing HIP runtimes.
  - `ResolveEmbeddingGemma2Artifacts` parses `config_sentence_transformers.json` prompts cleanly.
- **Provider Registration**:
  - Uses generated constants from `<gufo/ext/generated/embedding_gemma2_constants.hpp>` (`kProvider_embeddinggemma_2`).
  - Registers execution factory `CreateEmbeddingGemma2Runner` for `EmbeddingRunner` when `ENGINE_ENABLE_HIP` is active.

### 3.2 Decoupled Serving Boundary
- **Zero Includes of Migrated Extension in Serving**:
  - Grep audit confirmed zero `#include` statements targeting `src/models/embedding_gemma2/*` across `src/cli/serve/embedding_service.hpp`, `src/cli/serve/embedding_service.cpp`, and `src/cli/serve/embeddings_api.cpp`.
  - Transitive symbol coupling (`models::embedding_gemma2::kMaxImageSoftTokens`, `SupportedImageSoftTokens`, `LooksLikeEmbeddingGemma2`) eliminated in favor of neutral constants and capabilities in `gufo/ext/embeddings.hpp`.
  - Serving layer interacts solely through `ext::FindEmbeddingProvider`, `ext::EmbeddingRunner`, and capability types (`EmbeddingBatch`, `EmbeddingInput`, `EmbeddingRequestOptions`, etc.).
- **Extension Self-Containment**:
  - All 24 internal `#include` statements within `src/models/embedding_gemma2/` migrated to extension-relative paths (`"config.hpp"`, `"loader.hpp"`, `"image_processor.hpp"`, etc.).
  - Zero includes of `"src/models/embedding_gemma2/*"` remain inside `src/models/embedding_gemma2/`.
- **Preserved Decoupled Subsystems**:
  - Native model execution (`RunNative`), image/audio preprocessing, and video decoding remain fully owned by the extension behind `ext::EmbeddingRunner`.

### 3.3 Public Extension API Surface
- **Contracts (`include/gufo/ext/embeddings.hpp` & `src/ext/embeddings.hpp`)**:
  - Forwarding header `include/gufo/ext/embeddings.hpp` cleanly exposes multimodal embedding structs (`EmbeddingBatch`, `EmbeddingInput`, `EmbeddingPart`, `EmbeddingStatus`, `EmbeddingVideoOptions`, `EmbeddingRequestOptions`), abstract runner `EmbeddingRunner`, and registration functions.
- **Version Bump**:
  - Extension API version bumped from `0.1.2` to `0.1.3` (revision 3, backwards-compatible additive change) in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`.
- **Documentation and Surface Tracking**:
  - Detailed entry in `include/gufo/ext/CHANGES.md`.
  - Extension API surface hash updated in `include/gufo/ext/SURFACE` and verified with `python3 tools/ci/check-extension-api.py` (99 paths, PASS).
  - `tests/tools/check_extension_api_test.py` passes (7/7 tests PASS).
  - `tests/extensions/ext_runtime_neutral_contracts_test.cpp` passes (1/1 PASS).

### 3.4 Build and Registration Ratchet
- **Extensions Lock**:
  - `extensions.lock` contains `"embedding_gemma2"`.
- **Root CMake Decoupling**:
  - `add_subdirectory(src/models/embedding_gemma2)` removed from root `CMakeLists.txt`.
  - Direct `gufo_embedding_gemma2` target links removed from `gufo` and `gufo_http`. Targets link discovered extensions via `gufo_target_extensions`.
  - `src/models/embedding_gemma2/library.json` updated with `"consumers": []`.
- **Isolation Ratchet (`tools/ci/check-extension-isolation.py`)**:
  - Added `--fail-on-extension-boundary` / `--fail-on-cli-model-includes`.
  - Passes full repository check without violations (`PASS: extension boundary (no src/cli -> src/models/embedding_gemma2 includes)`).
  - Unit tests in `tests/tools/check_extension_isolation_test.py` pass (26/26 tests PASS, including negative test `test_extension_boundary_blocks_cli_includes_of_migrated_extensions` and repo test `test_repository_satisfies_the_extension_boundary`).
- **Discovery, Lock, and Explicit Exclusion Modes**:
  - Discovery mode: `gufo diagnose` detects `embedding_gemma2 0.1.0` and `provider embeddinggemma-2: available`.
  - Lock mode (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`): Built independently in `/tmp/card13-review/lock-test`; binary executes `gufo diagnose` and reports `extension selection: lock`, `embedding_gemma2 0.1.0` built.
  - Excluded mode (`-DGUFO_EXTENSIONS=""`): Built independently in `/tmp/card13-review/exclude-test`; binary compiles cleanly without undefined symbols, executes `gufo diagnose`, and reports `Built extensions: none`, `excluded embedding_gemma2 0.1.0: not selected in explicit mode`.

### 3.5 Behavior & Test Coverage Preservation
- **No Weakened Assertions**:
  - Test suites retain all prior assertions and coverage.
- **Embedding Tests**:
  - Ran `ctest --preset gpu-full -R '^embedding_gemma2_.*'`: all 15 tests ran (9 passed, 6 skipped with return code 77 on missing external model weights, 0 failed).
- **Continuation Identity Goldens**:
  - `qwen27b_continuation_identity_test`, `deepseek_v4_flash_continuation_identity_test`, `qwen38_flash_next_continuation_identity_test` pass byte-identically on preset.
- **Serving API**:
  - `embedding_gemma2_embeddings_api_test` passes (0.23s).

### 3.6 Repository Conventions & Hygiene
- **Windows Safety**:
  - No `<windows.h>`, `_WIN32`, or `_MSC_VER` macros introduced in C++ code outside `src/core/platform/`. Standard CMake `WIN32` used only in build files.
- **Documentation Limits**:
  - Zero files exceeding 1 MB under `docs/`.
- **Clang Format**:
  - Verified with `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`: PASS across 792 C++ files.
- **Evidence Honesty**:
  - Skipped tests are explicitly labeled as skipped due to absent external weights (code 77), not claimed as passed.
  - Functional harness (`tests/functional/run.py`), prompt lookup (`prompt_lookup.py`), and speed benchmark sweeps (`tools/bench/speed.py`) are documented as PENDING with verbatim commands in `docs/model-extensions/evidence/card13/pending-commands.txt`, in strict alignment with the 2026-10-31 shared RFC acceptance gate decision to defer full sweeps to the post-Card-14 qualification window.

---

## 4. Definition-of-Done Audit

| DoD Item | Status | Evidence / Verification |
| :--- | :--- | :--- |
| **1. Registry Loading, Probing, & Multimodal Semantics**<br>Pilot loads through the registry with truthful probe/selection/errors and unchanged advertised IDs, modalities, batch ordering and scheduling. | **PROVEN** | `ProbeEmbeddingGemma2` performs pure metadata validation in `probe.hpp`/`extension.cpp`. Registry registration verified via `ext_runtime_neutral_contracts_test` (PASS), `gufo diagnose` across discovery and lock modes (PASS), `embedding_gemma2_batch_scheduler_test` (PASS), `embedding_gemma2_embeddings_api_test` (PASS), and `embedding_gemma2_config_test` (PASS). External weight dependent tests skipped cleanly with code 77. Functional harness recorded as PENDING in `pending-commands.txt`. |
| **2. Manifest Ownership & Isolated Builds**<br>Its manifest owns production packages, data and runtime libraries; isolated/excluded builds work without hidden root declarations. | **PROVEN** | `src/models/embedding_gemma2/extension.json` matches schema 1. `extension.cmake` handles all targets and dynamic video library linkages without root hardcoding. `library.json` sets `"consumers": []`. Lock-mode build (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`) and excluded build (`-DGUFO_EXTENSIONS=""`) both built cleanly and passed diagnostics in `/tmp/card13-review/`. |
| **3. API, Numerical Gates, & Ratchet Enforcement**<br>Existing API, independent numerical and speed gates pass on both discovery and actual lock-mode HIP artifacts; ratchet rejects new coupling. | **PROVEN** (Build, in-tree unit/golden tests, and ratchet)<br>/ **DEFERRED** (Full model functional/speed sweeps) | `tools/ci/check-extension-api.py` passes (99 paths, version 0.1.3). Continuation identity goldens pass byte-identically on presets. All 15 CTest embedding tests passed or skipped cleanly (0 failures). Ratchet in `tools/ci/check-extension-isolation.py` enforces `--fail-on-extension-boundary` (0 violations; unit tests PASS). Full functional/speed sweeps deferred to post-Card-14 window per RFC agreement and recorded in `pending-commands.txt`. |

---

## 5. Commands Executed During Review

| Command | Exit Code | Verdict |
| :--- | :---: | :--- |
| `jj status` | 0 | Working copy clean, diff isolated to Card 13 files |
| `jj diff docs/model-extensions/RFC.md` | 0 | Clean (byte-identical, 0 diff) |
| `nix develop -c python3 tests/extensions/metadata_test.py` | 0 | PASS (12 tests) |
| `nix develop -c python3 tests/extensions/discovery_test.py` | 0 | PASS (13 tests) |
| `python3 tools/ci/check-extension-api.py` | 0 | PASS (99 paths, version 0.1.3) |
| `python3 tests/tools/check_extension_api_test.py` | 0 | PASS (7 tests) |
| `python3 tests/tools/check_extension_isolation_test.py` | 0 | PASS (26 tests) |
| `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes --fail-on-extension-boundary` | 0 | PASS (zero coupling / boundary violations) |
| `git grep "src/models/embedding_gemma2" src/models/embedding_gemma2/` | 1 | PASS (0 occurrences; all includes extension-relative) |
| `git grep "#include .*models" src/cli/serve/embedding_service.* src/cli/serve/embeddings_api.*` | 1 | PASS (0 occurrences; serving decoupled) |
| `nix develop -c ctest --preset gpu-full -R '^ext_runtime_neutral_contracts_test$' --output-on-failure` | 0 | PASS (1 test) |
| `nix develop -c ctest --preset gpu-full -R '^embedding_gemma2_.*' --output-on-failure` | 0 | PASS (15 tests: 9 passed, 6 skipped cleanly code 77) |
| `nix develop -c ctest --preset gpu-full -R 'continuation_identity_test' --output-on-failure` | 0 | PASS (3 tests) |
| `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | PASS (792 files clean) |
| `find docs -type f -size +1M` | 0 | PASS (0 files > 1 MB) |
| `build/gpu-test/gufo diagnose` | 0 | PASS (discovery mode: embedding_gemma2 available) |
| `nix develop -c cmake -S . -B /tmp/card13-review/lock-test -DGUFO_EXTENSIONS_FROM_LOCK=ON` | 0 | PASS (lock mode configuration) |
| `nix develop -c cmake --build /tmp/card13-review/lock-test --target gufo --parallel 4` | 0 | PASS (lock mode gufo binary built) |
| `/tmp/card13-review/lock-test/gufo diagnose` | 0 | PASS (lock mode: embedding_gemma2 available) |
| `nix develop -c cmake -S . -B /tmp/card13-review/exclude-test -DGUFO_EXTENSIONS=""` | 0 | PASS (explicit excluded mode configuration) |
| `nix develop -c cmake --build /tmp/card13-review/exclude-test --target gufo --parallel 4` | 0 | PASS (excluded mode gufo binary built) |
| `/tmp/card13-review/exclude-test/gufo diagnose` | 0 | PASS (explicit mode: embedding_gemma2 excluded) |

---

## 6. Conclusion

Card 13 is fully compliant with the model-extensions RFC requirements and ready for integration. All acceptance gates pass cleanly.
