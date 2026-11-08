> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Card 12: Add in-tree model API test support

You are the **implementation agent** (`worker`) for Card 12 of the model-extensions RFC in
`/home/mixer/gufo` (jj working copy `@`, parent commit `ytumsxmz 7752857d`). You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. Task-owned paths:
   - `CMakeLists.txt`
   - `src/models/qwen3_asr/CMakeLists.txt`
   - `src/models/qwen3_asr/tests/audio_api_test.cpp`
   - `src/models/qwen_image_21/CMakeLists.txt`
   - `tests/models/embedding_gemma2/CMakeLists.txt`
   - `tests/models/embedding_gemma2/embeddings_api_test.cpp`
   - `tests/models/qwen3_tts/audio_api_test.cpp`
   - `tests/models/qwen_image_21/model_test.cpp`
   - `tests/support/**` (including new `tests/support/model_api_test_support.hpp` or similar)
   - `tools/ci/check-extension-isolation.py`
   - `tests/tools/check_extension_isolation_test.py`
   - `docs/model-extensions/cards/12-model-api-test-support.md`
   - `docs/model-extensions/evidence/card12/**`
   - `docs/model-extensions/CARDS.md` (status updates)
4. **No functional harness.** `tests/functional/run.py`, `prompt_lookup.py`, speed
   benchmarks, and main-control matched timings are deferred by user decision (2026-10-31)
   to the single post-Card-14 qualification window. Do NOT run them or start a `gufo serve`
   server. Record deferred requirements as PENDING with their verbatim commands in
   `docs/model-extensions/evidence/card12/pending-commands.txt`.
5. Scratch goes in `/tmp/card12-*` only (logs, caches, temp files) and MUST be deleted
   when tests finish. No evidence file over 1 MB under `docs/`.
   No raw build logs, no per-row report trees under `docs/`.
6. Keep tool output small: redirect build and test logs to `/tmp/card12-logs/<name>.log`
   and inspect only using `tail -n 30` or grep. Never output entire build logs.
7. Host RAM: Before running any GPU build or test, check `free -g` and ensure at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
8. **Preserve unrelated work and user changes.** No behavior change.

## Objectives & Background

Card 12 implements RFC Step 7b:
"gufo_test_support: HTTP and serving fixtures for model API tests, so model tests stop including src/cli/ headers. It wraps today's serving and HTTP code and does not need the serving split."

Model API tests currently reach production CLI/serving internals directly:
- `src/models/qwen3_asr/tests/audio_api_test.cpp` includes `src/cli/serve/asr_service.hpp` and `src/cli/serve/audio_asr_api.hpp`, and compiles those `.cpp` files in `src/models/qwen3_asr/CMakeLists.txt`.
- `tests/models/embedding_gemma2/embeddings_api_test.cpp` includes `src/cli/serve/embeddings_api.hpp` and `src/cli/serve/embedding_service.hpp`, and compiles those `.cpp` files in `tests/models/embedding_gemma2/CMakeLists.txt`.
- `tests/models/qwen3_tts/audio_api_test.cpp` includes `src/cli/serve/audio_stream.hpp`, `src/cli/serve/audio_tts_api.hpp`, and `src/cli/serve/tts_service.hpp`, and compiles those `.cpp` files in root `CMakeLists.txt`.
- `tests/models/qwen_image_21/model_test.cpp` includes `src/cli/serve/http_server.hpp` and `src/cli/serve/image_api.hpp`, and links `gufo_http`.

An in-tree fixture library `gufo_test_support` can support them without making HTTP testing machinery part of the public extension source API.

## Step-by-Step Implementation Plan

### Step 1: Inventory Model API Test Includes of `src/cli/`
Catalog all 4 affected model API tests and their exact includes of `src/cli/`.
Document this inventory in `docs/model-extensions/evidence/card12/model-api-test-inventory.md`.

### Step 2: Add In-Tree-Only `gufo_test_support`
1. Create `tests/support/model_api_test_support.hpp` (or modular test support headers in `tests/support/`, e.g. wrapping HTTP server request/response and serving fixtures: `HttpRequest`, `HttpResponse`, `AsrService`, `AudioAsrApi`, `TtsService`, `AudioTtsApi`, `EmbeddingService`, `EmbeddingsApi`, `ImageService`, `ImageApi`, etc.).
2. In root `CMakeLists.txt`, within `if (BUILD_TESTING)`:
   Add `add_library(gufo_test_support STATIC tests/support/test_support.cpp)` (or header-only/interface if appropriate, but RFC says `gufo_test_support STATIC`). If static, provide `tests/support/test_support.cpp` with a version / registration marker.
   Link `gufo_http` and `gufo_serving` to `gufo_test_support`.
   Set `target_include_directories(gufo_test_support PUBLIC ${CMAKE_SOURCE_DIR})`.
   Ensure `gufo_test_support` is NEVER linked to `gufo`, `gufo_core`, `gufo_ext_runtime`, or installed.

### Step 3: Migrate Model API Tests to `gufo_test_support`
1. **ASR** (`src/models/qwen3_asr/tests/audio_api_test.cpp`):
   - Replace `#include "src/cli/serve/asr_service.hpp"` and `#include "src/cli/serve/audio_asr_api.hpp"` with `#include "tests/support/model_api_test_support.hpp"`.
   - In `src/models/qwen3_asr/CMakeLists.txt`:
     Remove `${PROJECT_SOURCE_DIR}/src/cli/serve/audio_asr_api.cpp` and `${PROJECT_SOURCE_DIR}/src/cli/serve/asr_service.cpp` from `qwen3_asr_audio_api_test`.
     Add `gufo_test_support` to `target_link_libraries(qwen3_asr_audio_api_test PRIVATE gufo_qwen3_asr gufo_test_support)`.
2. **EmbeddingGemma** (`tests/models/embedding_gemma2/embeddings_api_test.cpp`):
   - Replace `#include "src/cli/serve/embeddings_api.hpp"` and `#include "src/cli/serve/embedding_service.hpp"` with `#include "tests/support/model_api_test_support.hpp"`.
   - In `tests/models/embedding_gemma2/CMakeLists.txt`:
     Remove `${PROJECT_SOURCE_DIR}/src/cli/serve/embeddings_api.cpp` and `${PROJECT_SOURCE_DIR}/src/cli/serve/embedding_service.cpp` from `embedding_gemma2_embeddings_api_test`.
     Add `gufo_test_support` to `target_link_libraries`.
3. **TTS** (`tests/models/qwen3_tts/audio_api_test.cpp`):
   - Replace `#include "src/cli/serve/audio_stream.hpp"`, `audio_tts_api.hpp`, `tts_service.hpp` with `#include "tests/support/model_api_test_support.hpp"`.
   - In root `CMakeLists.txt`:
     Remove `src/cli/serve/audio_tts_api.cpp` and `src/cli/serve/tts_service.cpp` from `qwen3_tts_audio_api_test`.
     Add `gufo_test_support` to `target_link_libraries(qwen3_tts_audio_api_test PRIVATE gufo_qwen3_tts gufo_test_support)`.
4. **Image** (`tests/models/qwen_image_21/model_test.cpp`):
   - Replace `#include "src/cli/serve/http_server.hpp"` and `#include "src/cli/serve/image_api.hpp"` with `#include "tests/support/model_api_test_support.hpp"`.
   - In `src/models/qwen_image_21/CMakeLists.txt`:
     Replace `gufo_http` with `gufo_test_support` in `target_link_libraries(qwen_image_21_test PRIVATE gufo_qwen_image_21 gufo_test_support)`.

Compile each test target and verify all 4 tests pass!

### Step 4: Isolation Ratchet and Checks
1. In `tools/ci/check-extension-isolation.py`:
   - Tighten `--fail-on-serving-boundary`: Now that `src/models/qwen3_asr/tests/audio_api_test.cpp` no longer includes `src/cli/`, verify that `src/models/**` has 0 includes of `src/cli/**` (neither in production nor in tests).
   - In `serving_boundary_violations`: Remove the test-tree exemption for `src/models/` or verify that all files under `src/models/**` (including test files) are strictly forbidden from including `src/cli/`.
   - In `tests/tools/check_extension_isolation_test.py`: Update the test fixtures and verify with negative proof that any `src/models/**` test file including `src/cli/` is rejected.
2. Confirm target ownership: `python3 tools/ci/check-extension-isolation.py --fail-on-target-ownership` passes.
3. Confirm API surface: `python3 tools/ci/check-extension-api.py` passes (97 paths, unchanged SURFACE, zero test-support leaked).
4. Prove exclusion from production:
   Configure and build with `BUILD_TESTING=OFF` (e.g., in a scratch directory `/tmp/card12-btoff` or `build/release`) to prove `gufo_test_support` is excluded from production link/install.

### Step 5: Verification & Acceptance Gates
Run the full gate suite:
1. `nix develop -c ctest --test-dir build/cpu-test -R '^(qwen3_asr_audio_api_test|qwen_image_21_test|qwen3_tts_audio_api_test|embedding_gemma2_embeddings_api_test)$' --output-on-failure`
2. `nix develop -c ctest --test-dir build/gpu-test -R '^(qwen3_asr_audio_api_test|qwen_image_21_test|qwen3_tts_audio_api_test|embedding_gemma2_embeddings_api_test)$' --output-on-failure`
3. `python3 tools/ci/check-extension-api.py`
4. `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes`
5. `python3 tests/tools/check_extension_isolation_test.py`
6. `python3 tests/tools/check_extension_api_test.py`
7. `python3 tests/tools/check_docs_test.py`
8. Continuation identity goldens (CPU & GPU):
   `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity_test' --output-on-failure`
   `nix develop -c ctest --test-dir build/gpu-test -R 'continuation_identity_test' --output-on-failure`
9. Code format check: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
10. `BUILD_TESTING=OFF` build check.

### Step 6: Evidence Documentation
Create under `docs/model-extensions/evidence/card12/`:
- `model-api-test-inventory.md`: Complete before/after inventory of model API test includes and compiled sources, plus explicit documentation of remaining text runner and continuation identity test coupling.
- `test-support-architecture.md`: Specification of `gufo_test_support` target, include path conventions, production isolation, and absence from `SURFACE`.
- `build-and-tests.txt`: Exact commands, outputs, and exit codes for all builds and tests.
- `pending-commands.txt`: Verbatim list of deferred functional/benchmark commands per 2026-10-31 user decision.
- `RESUME-NOTES.md`: Summary of actions, files touched, scratch locations, and clean handoff.
- Update `docs/model-extensions/cards/12-model-api-test-support.md` checking off DoD items and drafting the initial record.

End with an explicit editing-ownership handoff. Do NOT commit.
