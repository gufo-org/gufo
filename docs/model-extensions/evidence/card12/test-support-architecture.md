# Test Support Architecture — Card 12

This document details the architectural design, build integration, isolation boundaries, and lifecycle guarantees of `gufo_test_support` added in Card 12 (RFC Step 7b).

## 1. Overview and Problem Statement

Prior to Card 12, model API test executables directly reached production CLI and HTTP serving headers located under `src/cli/serve/`:
- `src/models/qwen3_asr/tests/audio_api_test.cpp` included `src/cli/serve/asr_service.hpp` and `src/cli/serve/audio_asr_api.hpp`, while compiling `asr_service.cpp` and `audio_asr_api.cpp` directly in `src/models/qwen3_asr/CMakeLists.txt`.
- `tests/models/embedding_gemma2/embeddings_api_test.cpp` included `src/cli/serve/embeddings_api.hpp` and `src/cli/serve/embedding_service.hpp`, compiling `embeddings_api.cpp` and `embedding_service.cpp` in `tests/models/embedding_gemma2/CMakeLists.txt`.
- `tests/models/qwen3_tts/audio_api_test.cpp` included `src/cli/serve/audio_stream.hpp`, `src/cli/serve/audio_tts_api.hpp`, and `src/cli/serve/tts_service.hpp`, compiling `audio_tts_api.cpp` and `tts_service.cpp` in root `CMakeLists.txt`.
- `tests/models/qwen_image_21/model_test.cpp` included `src/cli/serve/http_server.hpp` and `src/cli/serve/image_api.hpp`, linking `gufo_http`.

This direct coupling violated the extension isolation boundary established in Card 09 (which required models, families, runtime, cache, and public headers to compile without reaching `src/cli/`). Furthermore, compiling serving `.cpp` sources directly inside test executables duplicated translation units and blurred ownership boundaries.

Card 12 introduces `gufo_test_support`, an in-tree test fixture library that encapsulates serving and HTTP fixtures for model API tests without exposing them as public extension APIs or leaking them into production builds.

---

## 2. CMake Target Specification

In root `CMakeLists.txt`, within the `if (BUILD_TESTING)` block:

```cmake
  # In-tree test support for model API tests (RFC Step 7b, Card 12).
  # Wraps HTTP and serving fixtures so tests stop reaching src/cli/ directly.
  # Excluded from production, external extensions, and installation.
  add_library(gufo_test_support STATIC tests/support/test_support.cpp)
  target_include_directories(gufo_test_support PUBLIC ${CMAKE_SOURCE_DIR})
  target_link_libraries(gufo_test_support PUBLIC gufo_http gufo_serving)
```

### Key Target Properties

| Property | Value | Rationale |
|---|---|---|
| **Target Name** | `gufo_test_support` | Canonical name specified in RFC Step 7b. |
| **Target Type** | `STATIC` | In-tree static library; no shared library ABI overhead. |
| **Condition** | `if (BUILD_TESTING)` | Only defined when tests are enabled. Completely absent in production. |
| **Sources** | `tests/support/test_support.cpp` | Dedicated translation unit providing `ModelApiTestSupportVersion()` marker. |
| **Dependencies** | `gufo_http`, `gufo_serving` | Links serving and HTTP implementations, eliminating redundant compilation of `.cpp` files in test targets. |
| **Includes** | `PUBLIC ${CMAKE_SOURCE_DIR}` | Allows test executables to access repository root headers via standard paths. |
| **Installation** | None | Excluded from `install()` rules and package distribution. |

---

## 3. Header and Include Conventions

### Header Path
- Header file: `tests/support/model_api_test_support.hpp`
- Inclusion syntax: `#include "tests/support/model_api_test_support.hpp"`

### Wrapped Headers
The header aggregates and wraps the internal serving headers required by model API tests:
- `src/cli/serve/asr_service.hpp`
- `src/cli/serve/audio_asr_api.hpp`
- `src/cli/serve/audio_stream.hpp`
- `src/cli/serve/audio_tts_api.hpp`
- `src/cli/serve/embedding_service.hpp`
- `src/cli/serve/embeddings_api.hpp`
- `src/cli/serve/http_server.hpp`
- `src/cli/serve/image_api.hpp`
- `src/cli/serve/tts_service.hpp`

### Test Support Namespace
Provides a version marker in `gufo::test`:
```cpp
namespace gufo::test {
const char* ModelApiTestSupportVersion();
}
```
All serving types remain in `gufo::server::*`, preserving exact types and method contracts in test fixtures without re-wrapping or losing precision.

---

## 4. Production and Extension API Isolation

### 4.1 Production Exclusion Proof (`BUILD_TESTING=OFF`)
When `BUILD_TESTING=OFF` (as in production releases and the shipped Nix package):
- `gufo_test_support` is never added to CMake's target dictionary.
- The build graph contains 0 rules, 0 objects, and 0 links for `gufo_test_support`.
- `cmake_install.cmake` contains 0 references to `test_support`.
- The production executable `gufo` never links or references test support symbols.

### 4.2 Public Extension API Surface
- `include/gufo/ext/SURFACE` lists exactly 97 public headers.
- `gufo_test_support` and `tests/support/**` are strictly internal to the repository's test suite and are never exported in `SURFACE` or installed under `include/gufo/ext/`.
- `python3 tools/ci/check-extension-api.py` confirms 97 paths with 0 changes.

---

## 5. Serving Boundary Isolation Ratchet

With `src/models/qwen3_asr/tests/audio_api_test.cpp` migrated to `tests/support/model_api_test_support.hpp`:
1. **Zero CLI includes in `src/models/**`**: Not a single file under `src/models/` (neither production code nor test code) includes `src/cli/**`.
2. **Elimination of test-tree exemption**:
   In `tools/ci/check-extension-isolation.py`, `serving_boundary_violations()` previously included `or in_test_tree(path)`. This exemption has been removed:
   ```python
   def serving_boundary_violations(report):
       violations = []
       for edge in report["includes"]:
           path = edge["path"]
           if not path.startswith(BOUNDARY_LAYERS):
               continue
           target = edge["resolved"] or posixpath.normpath(
               posixpath.join(posixpath.dirname(path), edge["include"]))
           if target.startswith("src/cli/"):
               violations.append(f"{path}:{edge['line']}: {edge['include']}")
       return sorted(violations)
   ```
3. **Negative proof in test suite**:
   In `tests/tools/check_extension_isolation_test.py`, `test_serving_boundary_blocks_cli_includes_from_production_layers()` now asserts that `src/models/qwen3_asr/tests/audio_api_test.cpp` including `src/cli/serve/http_server.hpp` is actively rejected as a violation.

---

## 6. Lifecycle Assertion Integrity

The migration strictly preserves all request, error, and cancellation assertions across all 4 tests:
- `qwen3_asr_audio_api_test`: Transcription request parsing, audio duration checking, WAV decoding, cancellation check token verification, and error formatting.
- `embedding_gemma2_embeddings_api_test`: Batching, multi-part prompt inputs, truncation with image preservation, JSON error responses, prompt template mapping, cancellation, and concurrency tests.
- `qwen3_tts_audio_api_test`: Streaming audio synthesis, chunking callbacks, cancellation check during generation, invalid waveform rejection, sample rate verification.
- `qwen_image_21_test`: Text-to-image generation requests, multipart image edit requests, parameter validation errors (dimensions, steps), 400 Bad Request and 404 Not Found status checks.

No test assertion was modified, weakened, or bypassed.
