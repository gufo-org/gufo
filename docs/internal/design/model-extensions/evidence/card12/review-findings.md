# Card 12 Independent Review Findings

Reviewer: Independent Review Agent  
Working Copy: `@` (`ynuxkvqu d99e0d7f`) over parent `ytumsxmz 7752857d` ("refactor(extensions): migrate models to public extension headers")  
Target Card: Card 12 — Add in-tree model API test support (RFC Step 7b)  
Verdict: **CLEAN PASS** (Zero findings / F0)

---

## 1. Executive Summary

An exhaustive independent read-only audit of the working copy changes for Card 12 was conducted. The diff strictly implements the Card 12 specification without scope under-run or over-reach:
1. **In-tree Test Support Target**: Created `gufo_test_support` STATIC library in root `CMakeLists.txt` conditionally scoped under `if (BUILD_TESTING)`, linking `gufo_http` and `gufo_serving`. It is backed by `tests/support/model_api_test_support.hpp` and `tests/support/test_support.cpp`.
2. **Model API Tests Migrated (4/4)**:
   - `src/models/qwen3_asr/tests/audio_api_test.cpp`
   - `tests/models/embedding_gemma2/embeddings_api_test.cpp`
   - `tests/models/qwen3_tts/audio_api_test.cpp`
   - `tests/models/qwen_image_21/model_test.cpp`
   All production CLI/serving header includes (`src/cli/serve/*.hpp`) were removed and replaced with `#include "tests/support/model_api_test_support.hpp"`. All compiled production serving sources were removed from test targets (`audio_asr_api.cpp`, `asr_service.cpp`, `embeddings_api.cpp`, `embedding_service.cpp`, `audio_tts_api.cpp`, `tts_service.cpp`), and direct `gufo_http` linkage in `qwen_image_21_test` was replaced with `gufo_test_support`.
3. **Serving Boundary Isolation Ratchet**: Removed the `in_test_tree(path)` exemption from `serving_boundary_violations()` in `tools/ci/check-extension-isolation.py`. Verified that `src/models/**` now has **0** `src/cli/` includes across both production and test files. Negative regression proof was added and verified in `tests/tools/check_extension_isolation_test.py`.
4. **Surface & Production Isolation**: Public extension API surface (`include/gufo/ext/SURFACE`, 97 paths) is untouched and unchanged. Under `BUILD_TESTING=OFF`, `gufo_test_support` is completely absent from the build graph, install manifests, and output artifacts.
5. **Assertion & Behavior Preservation**: Diff inspection confirms that zero test assertions were altered, weakened, or removed. All 4 migrated tests pass on both CPU and GPU presets. Continuation identity goldens pass byte-identically on both CPU and GPU presets.
6. **RFC Byte-Identity**: Verified that `docs/model-extensions/RFC.md` is strictly untouched and byte-identical (`jj diff --name-only` confirmed clean).
7. **No Deferred Work Claimed as Pass**: Functional harness runs remain properly marked as PENDING in `pending-commands.txt` deferred to post-Card-14 qualification window.

---

## 2. Review Findings (F1...Fn)

**No findings (F0).** The implementation is complete, strictly conforms to repository and RFC guidelines, and satisfies all validation criteria.

---

## 3. Explicit "No Finding" Statements for Audited Areas

- **Scope & Diff Size**: Clean. Diff touches exactly the 4 model API test suites, root and model `CMakeLists.txt`, `tests/support/`, `tools/ci/check-extension-isolation.py`, `tests/tools/check_extension_isolation_test.py`, and Card 12 documentation/evidence. No unrelated files modified.
- **Public API & Surface Stability**: Clean. `python3 tools/ci/check-extension-api.py` confirms 97 paths with zero modifications or header leaks.
- **Production Isolation & Install Rules**: Clean. Verified via CMake configure with `-DBUILD_TESTING=OFF`; 0 references to `test_support` exist in the build graph, makefiles, or install scripts.
- **Test Assertion Integrity**: Clean. Diffs in test files are strictly confined to header inclusion (`#include "tests/support/model_api_test_support.hpp"` replacing `src/cli/serve/*.hpp`). No test cases, inputs, assertions, or tolerances were altered.
- **Serving Boundary Ratchet & Negative Proof**: Clean. `serving_boundary_violations()` enforces zero `src/cli/` includes across `BOUNDARY_LAYERS` (`src/models/`, `src/families/`, `src/cache/`, `src/ext/`, `include/gufo/ext/`). `tests/tools/check_extension_isolation_test.py` validates rejection of test files in `src/models/` that include `src/cli/`.
- **Platform & Windows Compatibility**: Clean. No `<windows.h>`, `_WIN32`, or unportable constructs used in `tests/support/`. Standard CMake target definitions used.
- **Formatting & File Size**: Clean. `python3 tools/ci/check-format.py` passes cleanly on all 787 C++ files. Zero files > 1 MB in `docs/`.

---

## 4. Definition of Done (DoD) Audit Table

| DoD Item | Status | Verification Evidence / Path |
|---|---|---|
| **DoD 1**: Affected model API tests use test support, with no production CLI includes remaining at the migrated boundaries. | **PROVEN** | `src/models/qwen3_asr/tests/audio_api_test.cpp`<br>`tests/models/embedding_gemma2/embeddings_api_test.cpp`<br>`tests/models/qwen3_tts/audio_api_test.cpp`<br>`tests/models/qwen_image_21/model_test.cpp`<br>All 4 include `tests/support/model_api_test_support.hpp`. `grep -rn "src/cli" src/models/` yields 0 matches. Verified by Gate 4 (`check-extension-isolation.py`). |
| **DoD 2**: Test support is absent from production link/install and is not a public external-extension dependency. | **PROVEN** | `CMakeLists.txt`: `gufo_test_support` declared exclusively within `if (BUILD_TESTING)`. Verified via `cmake -B /tmp/card12-review/btoff-test -DBUILD_TESTING=OFF` yielding 0 occurrences in build targets, makefiles, and `cmake_install.cmake`. `python3 tools/ci/check-extension-api.py` confirms 97 paths in `SURFACE` unchanged. |
| **DoD 3**: Fixture lifecycle preserves request/error/cancellation assertions rather than weakening model API coverage. | **PROVEN** | Diffs show zero changes to assertion logic, request payloads, or error paths. Tests execute against identical HTTP handlers via `gufo_test_support`. Verified by 100% pass on CPU (Gate 1: 4/4 passed) and GPU (Gate 2: 4/4 passed). |

---

## 5. Reviewer Executed Commands & Validation Gates

All validation gates were independently re-executed by the reviewer with exact outputs verified:

| Gate | Description | Command Executed | Exit Code | Result |
|---|---|---|---|---|
| **Gate 1** | Model API CTests on CPU | `nix develop -c ctest --test-dir build/cpu-test -R '^(qwen3_asr_audio_api_test\|qwen_image_21_test\|qwen3_tts_audio_api_test\|embedding_gemma2_embeddings_api_test)$' --output-on-failure` | 0 | **PASS** (4/4 passed, 0.19s) |
| **Gate 2** | Model API CTests on GPU | `nix develop -c ctest --test-dir build/gpu-test -R '^(qwen3_asr_audio_api_test\|qwen_image_21_test\|qwen3_tts_audio_api_test\|embedding_gemma2_embeddings_api_test)$' --output-on-failure` | 0 | **PASS** (4/4 passed, 0.41s) |
| **Gate 3** | Extension API Surface Gate | `python3 tools/ci/check-extension-api.py` | 0 | **PASS** (97 paths, SURFACE unchanged) |
| **Gate 4** | Extension Isolation & Boundary Gate | `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes` | 0 | **PASS** (All 5 boundary checks clean) |
| **Gate 5** | Isolation Unit Tests | `python3 tests/tools/check_extension_isolation_test.py` | 0 | **PASS** (24/24 unit tests OK) |
| **Gate 6** | Extension API Policy Unit Tests | `python3 tests/tools/check_extension_api_test.py` | 0 | **PASS** (7/7 unit tests OK) |
| **Gate 7** | Docs Policy Unit Tests | `python3 tests/tools/check_docs_test.py` | 0 | **PASS** (11/11 unit tests OK) |
| **Gate 8** | Continuation Identity Goldens (CPU & GPU) | `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity_test'`<br>`nix develop -c ctest --test-dir build/gpu-test -R 'continuation_identity_test'` | 0<br>0 | **PASS** (3/3 CPU, 3/3 GPU passed) |
| **Gate 9** | Clang-Format Verification | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | **PASS** (787 C++ files formatted) |
| **Gate 10** | Production Exclusion Proof (`BUILD_TESTING=OFF`) | `nix develop -c cmake -B /tmp/card12-review/btoff-test -DBUILD_TESTING=OFF`<br>`grep -ri "test_support" /tmp/card12-review/btoff-test/` | 0 | **PASS** (0 matches in targets, rules, or install scripts) |

---

## 6. Conclusion & Handoff

Card 12 is fully validated and ready for integration. No tracked repository files were modified or committed during this review.
Review ownership is released to the integration agent / orchestrator.
