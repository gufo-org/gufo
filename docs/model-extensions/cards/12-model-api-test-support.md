# Card 12: Add in-tree model API test support

Status: **integrated, ready for orchestrator closure**. RFC step and dependencies: **7b; 11**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Model API tests currently reach production CLI/serving internals. An in-tree fixture library can support them without making HTTP testing machinery part of the extension source API.

### Original scope

Add in-tree-only gufo_test_support for HTTP/serving fixtures, remove model-test production CLI includes. Public extension API excludes this support. Run all affected model API tests and report remaining coupling explicitly.

## Implementation plan

1. Inventory model API test includes of src/cli/ and identify reusable HTTP/server lifecycle and serving fixtures.
2. Add in-tree-only gufo_test_support wrapping current gufo_http and gufo_serving fixtures, with explicit test-only dependencies and build conditions.
3. Migrate model API tests to the fixture surface and remove production CLI includes; keep extension production libraries independent of support.
4. Document any remaining test coupling and keep support out of include/gufo/ext/SURFACE and installed extension contracts.

## Definition of done

- [x] Affected model API tests use test support, with no production CLI includes remaining at the migrated boundaries.
- [x] Test support is absent from production link/install and is not a public external-extension dependency.
- [x] Fixture lifecycle preserves request/error/cancellation assertions rather than weakening model API coverage.

## Validation criteria

- List affected API CTests with ctest --preset gpu-full -N and run every migrated model API test with its required weights; missing-model skips remain pending.
- Configure/build with BUILD_TESTING=OFF and inspect production link/install inputs to prove test support is excluded.
- Run isolation inventory for tests and inspect SURFACE to ensure support is private; retain explicit remaining coupling rather than a global clean claim.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Roles and Working Copy
- **Card**: Card 12 — Add in-tree model API test support
- **Parent commit**: `ytumsxmz 7752857d` ("refactor(extensions): migrate models to public extension headers")
- **Agent Sequence**:
  - Implementation Worker (`worker`): Run ID `02a253d3-11d5-42ec-ae8e-d89e20ef0716`
  - Independent Reviewer (`reviewer`): Run ID `397daa87-23c0-4244-acce-79507e4def2a` (verdict CLEAN PASS, F0)
  - Review Integrator (`integrator`): Run ID `fde70554-5ab1-4a6f-9e2f-b5012f550e0c`
  - Orchestrator (`orchestrator`): Closure
- **Working copy changes**:
  - `CMakeLists.txt`
  - `src/models/qwen3_asr/CMakeLists.txt`
  - `src/models/qwen3_asr/tests/audio_api_test.cpp`
  - `src/models/qwen_image_21/CMakeLists.txt`
  - `tests/models/embedding_gemma2/CMakeLists.txt`
  - `tests/models/embedding_gemma2/embeddings_api_test.cpp`
  - `tests/models/qwen3_tts/audio_api_test.cpp`
  - `tests/models/qwen_image_21/model_test.cpp`
  - `tests/support/model_api_test_support.hpp`
  - `tests/support/test_support.cpp`
  - `tools/ci/check-extension-isolation.py`
  - `tests/tools/check_extension_isolation_test.py`
  - `docs/model-extensions/cards/12-model-api-test-support.md`
  - `docs/model-extensions/CARDS.md`
  - `docs/model-extensions/cards/README.md`
  - `docs/model-extensions/evidence/card12/**`

---

### Scope Audit

The diff strictly matches the scope specified by RFC Step 7b and Card 12:
- **Created test support library**: `tests/support/model_api_test_support.hpp`, `tests/support/test_support.cpp`, and `gufo_test_support` STATIC library in root `CMakeLists.txt` under `if (BUILD_TESTING)`.
- **Migrated model API tests (4/4)**:
  1. `src/models/qwen3_asr/tests/audio_api_test.cpp`
  2. `tests/models/embedding_gemma2/embeddings_api_test.cpp`
  3. `tests/models/qwen3_tts/audio_api_test.cpp`
  4. `tests/models/qwen_image_21/model_test.cpp`
- **Removed production CLI source compilation**: Stripped production serving sources (`audio_asr_api.cpp`, `asr_service.cpp`, `embeddings_api.cpp`, `embedding_service.cpp`, `audio_tts_api.cpp`, `tts_service.cpp`) from test targets; linked `gufo_test_support`.
- **Ratchet serving boundary**: Removed `in_test_tree(path)` exemption from `serving_boundary_violations()` in `tools/ci/check-extension-isolation.py`. No files in `src/models/` may include `src/cli/`.
- **Untouched and preserved**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted. Public API surface `include/gufo/ext/SURFACE` remains 97 paths.

---

### Implementation Summary

1. **In-Tree Test Support Library (`gufo_test_support`)**:
   - Aggregates HTTP and serving fixtures: `HttpRequest`, `HttpResponse`, `AsrService`, `AudioAsrApi`, `AudioStream`, `TtsService`, `AudioTtsApi`, `EmbeddingService`, `EmbeddingsApi`, `ImageService`, `ImageApi`.
   - Exposes `gufo::test::ModelApiTestSupportVersion()` in namespace `gufo::test`.
   - Declared as `STATIC` library conditionally built only when `BUILD_TESTING` is `ON`.
   - Excluded from `include/gufo/ext/SURFACE` and public extension headers.

2. **Model API Test Migrations**:
   - `src/models/qwen3_asr/tests/audio_api_test.cpp`: Replaced `#include "src/cli/serve/asr_service.hpp"` and `#include "src/cli/serve/audio_asr_api.hpp"` with `#include "tests/support/model_api_test_support.hpp"`.
   - `src/models/qwen3_asr/CMakeLists.txt`: Removed `${PROJECT_SOURCE_DIR}/src/cli/serve/audio_asr_api.cpp` and `asr_service.cpp`; linked `gufo_test_support`.
   - `tests/models/embedding_gemma2/embeddings_api_test.cpp`: Replaced `#include "src/cli/serve/embeddings_api.hpp"` and `embedding_service.hpp` with `#include "tests/support/model_api_test_support.hpp"`.
   - `tests/models/embedding_gemma2/CMakeLists.txt`: Removed `${PROJECT_SOURCE_DIR}/src/cli/serve/embeddings_api.cpp` and `embedding_service.cpp`; linked `gufo_test_support`.
   - `tests/models/qwen3_tts/audio_api_test.cpp`: Replaced `#include "src/cli/serve/audio_stream.hpp"`, `audio_tts_api.hpp`, `tts_service.hpp` with `#include "tests/support/model_api_test_support.hpp"`.
   - `CMakeLists.txt`: Removed `src/cli/serve/audio_tts_api.cpp` and `tts_service.cpp` from `qwen3_tts_audio_api_test`; linked `gufo_test_support`.
   - `tests/models/qwen_image_21/model_test.cpp`: Replaced `#include "src/cli/serve/http_server.hpp"` and `image_api.hpp` with `#include "tests/support/model_api_test_support.hpp"`.
   - `src/models/qwen_image_21/CMakeLists.txt`: Replaced direct link `gufo_http` with `gufo_test_support` in `qwen_image_21_test`.

3. **Serving Boundary Enforcement**:
   - `tools/ci/check-extension-isolation.py`: Enforces 0 `src/cli/` includes from all layers in `BOUNDARY_LAYERS` (`src/models/`, `src/families/`, `src/cache/`, `src/ext/`, `include/gufo/ext/`).
   - `tests/tools/check_extension_isolation_test.py`: Adds negative unit test verifying that any `src/cli/` include in `src/models/` (production or test) fails the check.

---

### Review Findings and Dispositions

| Finding ID | Severity | Description | Disposition |
|---|---|---|---|
| **F0** | None | Independent reviewer reported **CLEAN PASS** with zero findings across all audited areas (scope, surface stability, production exclusion, assertion integrity, isolation ratchet, formatting). | **Accepted**: No code modifications required during integration. Acceptance audited and verified. |

---

### Definition of Done Audit

| DoD Item | Status | Verification Evidence / Reference |
|---|---|---|
| **DoD 1**: Affected model API tests use test support, with no production CLI includes remaining at the migrated boundaries. | **PROVEN** | `src/models/qwen3_asr/tests/audio_api_test.cpp`<br>`tests/models/embedding_gemma2/embeddings_api_test.cpp`<br>`tests/models/qwen3_tts/audio_api_test.cpp`<br>`tests/models/qwen_image_21/model_test.cpp`<br>All 4 include `tests/support/model_api_test_support.hpp`. Zero `src/cli/` includes remain in `src/models/`. Gate 4 (`check-extension-isolation.py`) passes all 5 boundary checks cleanly. |
| **DoD 2**: Test support is absent from production link/install and is not a public external-extension dependency. | **PROVEN** | `CMakeLists.txt`: `gufo_test_support` declared exclusively under `if (BUILD_TESTING)`. Verified via `cmake -B /tmp/... -DBUILD_TESTING=OFF` yielding 0 occurrences in build targets, rules, and install scripts. `python3 tools/ci/check-extension-api.py` confirms 97 paths in `SURFACE` unchanged with zero public header leaks. |
| **DoD 3**: Fixture lifecycle preserves request/error/cancellation assertions rather than weakening model API coverage. | **PROVEN** | Diffs confirm zero assertion alterations, removals, or relaxations. Tests execute full HTTP request/response lifecycles and error paths identically. Verified by 100% pass on CPU (Gate 1: 4/4 passed) and GPU (Gate 2: 4/4 passed). |

---

### Acceptance Gates and Integration Re-runs

All 10 verification gates were re-executed once by the integration agent; full logs and details are recorded in [`docs/model-extensions/evidence/card12/integration-reruns.txt`](../evidence/card12/integration-reruns.txt):

| Gate | Description | Command Executed | Exit Code | Verdict |
|---|---|---|---|---|
| **Gate 1** | Model API CTests on CPU | `nix develop -c ctest --test-dir build/cpu-test -R '^(qwen3_asr_audio_api_test\|qwen_image_21_test\|qwen3_tts_audio_api_test\|embedding_gemma2_embeddings_api_test)$' --output-on-failure` | 0 | **PASS** (4/4 passed, 0.19s) |
| **Gate 2** | Model API CTests on GPU | `nix develop -c ctest --test-dir build/gpu-test -R '^(qwen3_asr_audio_api_test\|qwen_image_21_test\|qwen3_tts_audio_api_test\|embedding_gemma2_embeddings_api_test)$' --output-on-failure` | 0 | **PASS** (4/4 passed, 0.42s) |
| **Gate 3** | Extension API Surface Gate | `python3 tools/ci/check-extension-api.py` | 0 | **PASS** (97 paths, SURFACE unchanged) |
| **Gate 4** | Extension Isolation & Boundary Gate | `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes` | 0 | **PASS** (All 5 boundary checks clean) |
| **Gate 5** | Isolation Unit Tests | `python3 tests/tools/check_extension_isolation_test.py` | 0 | **PASS** (24/24 unit tests OK) |
| **Gate 6** | Extension API Policy Unit Tests | `python3 tests/tools/check_extension_api_test.py` | 0 | **PASS** (7/7 unit tests OK) |
| **Gate 7** | Documentation Policy Unit Tests | `python3 tests/tools/check_docs_test.py` | 0 | **PASS** (11/11 unit tests OK) |
| **Gate 8** | Continuation Identity Goldens (CPU & GPU) | `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity_test' --output-on-failure`<br>`nix develop -c ctest --test-dir build/gpu-test -R 'continuation_identity_test' --output-on-failure` | 0<br>0 | **PASS** (3/3 CPU, 3/3 GPU passed) |
| **Gate 9** | Clang-Format Verification | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | **PASS** (787 C++ files formatted) |
| **Gate 10** | Production Exclusion Proof (`BUILD_TESTING=OFF`) | `nix develop -c cmake -B /tmp/card12-integration/btoff-test -DBUILD_TESTING=OFF`<br>`grep -ri "test_support" /tmp/card12-integration/btoff-test/` | 0 | **PASS** (0 matches in targets, rules, or install scripts) |

---

### Remaining Limitations and Deferred Work

1. **Deferred Functional Test Harness**: Per user decision (2026-10-31), functional harness suites (`tests/functional/run.py` rows, prompt lookup, and speed benchmarks) do not run during Card 12. All functional verification commands remain recorded in [`pending-commands.txt`](../evidence/card12/pending-commands.txt) and are deferred to the single consolidated qualification window following Card 14.
2. **Explicit Remaining Test Coupling**:
   - Text generation test runners (`tests/models/deepseek_v4_flash/`, `tests/models/qwen/`, `tests/models/qwen27b/`, `tests/models/qwen38_flash_next/`) include `src/cli/text_runner.hpp` or `src/cli/generation_stats.hpp` for CLI integration harness execution. These will be addressed in Cards 19–22 when shared text contracts and model extensions are isolated.
   - Core continuation identity tests (`tests/cli/continuation_identity_test.cpp`) test the `gufo` CLI binary end-to-end and intentionally include `src/cli/` headers.
   - All 4 model API test suites (ASR, TTS, Image, Embedding) have **zero** production CLI includes remaining.

---

### Evidence Files

- [`model-api-test-inventory.md`](../evidence/card12/model-api-test-inventory.md): Full inventory of test includes and target sources before and after migration.
- [`test-support-architecture.md`](../evidence/card12/test-support-architecture.md): Target architecture and specification for `gufo_test_support`.
- [`build-and-tests.txt`](../evidence/card12/build-and-tests.txt): Worker compilation and gate outputs.
- [`review-findings.md`](../evidence/card12/review-findings.md): Independent reviewer audit report (CLEAN PASS, F0).
- [`integration-reruns.txt`](../evidence/card12/integration-reruns.txt): Integration re-run commands, exit codes, and verdicts.
- [`orchestrator-gates.txt`](../evidence/card12/orchestrator-gates.txt): Orchestrator closure re-runs, Codex review output, and final gate verification.
- [`pending-commands.txt`](../evidence/card12/pending-commands.txt): Verbatim deferred functional harness and benchmark commands.
- [`RESUME-NOTES.md`](../evidence/card12/RESUME-NOTES.md): Handoff notes and ownership release.
- [`README.md`](../evidence/card12/README.md): Evidence directory index and provenance.
