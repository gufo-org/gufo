# Card 11 Integration — Edits Complete, Editing Ownership Released

Integration agent (`dd111c9e-4200-4e2c-bc44-4c549b28b08d`, `codex-exec-writer`) has completed all integration tasks and releases editing ownership to the orchestrator.
No commits, pushes, servers, functional harness runs, benchmarks, or full sweeps were performed.
No active background handles.
`docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.

## Status & Summary

- **Card**: Card 11 — Migrate models to public extension headers
- **Working Copy**: `@` over parent commit `xppvypvw 8d8ef356` ("feat(extensions): publish and measure public extension api surface")
- **Review Findings**: Verdict **PASS** (1 Nit: F1 test reporting language).
- **Finding F1 Resolution**: Updated test summaries in `cards/11-public-header-migrations.md` and `RESUME-NOTES.md` to explicitly delineate executed passes from missing-asset skips (e.g., "2 passed, 6 skipped (missing model weights); 0 failures" rather than "8/8 pass (100%)").
- **Integration Gates**: All affected gates re-run ONCE with clean results recorded in `docs/model-extensions/evidence/card11/integration-reruns.txt`:
  * `python3 tools/ci/check-extension-api.py`: PASS (97 paths, exit 0)
  * `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes`: PASS (all 5 boundary and ownership checks pass, exit 0)
  * `python3 tests/tools/check_extension_isolation_test.py`: PASS (24/24 unit tests OK, exit 0)
  * `python3 tests/tools/check_extension_api_test.py`: PASS (7/7 unit tests OK, exit 0)
  * `python3 tests/tools/check_docs_test.py`: PASS (11/11 tests OK, exit 0)
  * `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`: PASS (785 C++ files formatted, exit 0)
  * Bounded CPU CTest (`extension_model_core_includes_gate` + 3 identity goldens): 4/4 passed (100%, exit 0)
  * Bounded GPU CTest (`extension_model_core_includes_gate` + 3 identity goldens): 4/4 passed (100%, exit 0)
- **Acceptance Audit**: All 3 DoD requirements proven and checked `[x]` in `cards/11-public-header-migrations.md`.
- **Card & Ledger Documentation**:
  * `docs/model-extensions/cards/11-public-header-migrations.md`: Updated status, agent sequence (Worker: `1c0adb86-f704-460b-9694-93aa312ac1cd`, Reviewer: `70a872ba-869f-42b1-80e6-b71880841633`, Integrator: `dd111c9e-4200-4e2c-bc44-4c549b28b08d`), review findings disposition table, validation and acceptance audit, and remaining limitations.
  * `docs/model-extensions/CARDS.md`: Marked Card 11 `complete` for in-card gates, updated active range through Card 11, and added `### Card 11` execution record summary.
  * `docs/model-extensions/cards/README.md`: Updated checkpoint with Card 11 completion summary.
  * `docs/model-extensions/evidence/card11/README.md`: Created provenance ledger.
  * `docs/model-extensions/evidence/card11/integration-reruns.txt`: Created exact rerun log.

## Changed Paths in Working Copy

1. **Root Build Configuration**:
   - `CMakeLists.txt` (added `${CMAKE_SOURCE_DIR}/include` at line 29; added `extension_model_core_includes_gate`)
2. **Model Headers & Sources (95 files modified, 152 includes migrated)**:
   - `src/models/embedding_gemma2/**` (9 files, 11 includes)
   - `src/models/qwen3_asr/**` (7 files, 11 includes)
   - `src/models/qwen3_tts/**` (6 files, 10 includes)
   - `src/models/qwen_image_21/**` (4 files, 7 includes)
   - `src/models/minimax_h3/**` (7 files, 11 includes)
   - `src/models/qwen/**` (31 files, 53 includes)
   - `src/models/qwen38_flash_next/**` (19 files, 29 includes)
   - `src/models/deepseek_v4_flash/**` (12 files, 20 includes)
3. **Tools & Isolation Enforcement**:
   - `tools/ci/check-extension-isolation.py` (added `--fail-on-model-core-includes`)
   - `tests/tools/check_extension_isolation_test.py` (added negative and positive boundary tests)
4. **Documentation & Evidence**:
   - `docs/model-extensions/CARDS.md`
   - `docs/model-extensions/cards/README.md`
   - `docs/model-extensions/cards/11-public-header-migrations.md`
   - `docs/model-extensions/evidence/card11/**`

## Scratch to Delete at Closure

- `/tmp/card11-logs/`
- `/tmp/card11-review/`
- `/tmp/card11-integration/`
- `/tmp/gufo-card11-isolation.json`

---

## Historical Worker Handoff

# Card 11 Worker Resume Notes

## Phase & Role
- **Card**: Card 11 — Migrate models to public extension headers
- **Agent Role**: Implementation Worker (`worker`)
- **Status**: Complete, handed off to independent reviewer
- **Working Copy**: `@` (parent commit `xppvypvw 8d8ef356`)
- **Constraints Maintained**:
  * No `jj commit`, `jj new`, `jj squash`, or `git commit`.
  * `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.
  * No functional harness runs (`tests/functional/run.py`, `prompt_lookup.py`, speed benchmarks deferred to post-Card-14 qualification window).
  * Scratch kept in `/tmp/card11-*`.
  * Host RAM checked before GPU builds/tests (>= 15 GiB headroom preserved; `llama-swap.service` kept running).

## Completed Implementation Tasks

1. **Root CMakeLists.txt Include Directory**:
   - Updated line 29 of `CMakeLists.txt` to:
     `include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)`
   - This ensures all model targets and dependent test executables locate `<gufo/ext/...>` headers.

2. **Mechanical Model Include Migrations (Strict Order)**:
   Migrated all eight model implementations from private `src/core/` and `src/cache/` includes to public `<gufo/ext/...>` headers:
   - **EmbeddingGemma** (`src/models/embedding_gemma2`): 11 includes across 9 files migrated. Compiled `gufo_embedding_gemma2`, verified `embedding_gemma2_*_test` on CPU (7 passed, 2 skipped due to missing model weights; 0 failures) and GPU (14 passed, 0 skipped; 0 failures).
   - **ASR** (`src/models/qwen3_asr`): 11 includes across 7 files migrated. Compiled `gufo_qwen3_asr`, verified `qwen3_asr_*_test` on GPU (2 passed, 6 skipped due to missing model weights; 0 failures) and CPU (2 passed, 3 skipped due to missing model weights; 0 failures). Preserved test-only serving edge in `tests/audio_api_test.cpp`.
   - **TTS** (`src/models/qwen3_tts`): 10 includes across 6 files migrated. Compiled `gufo_qwen3_tts`, verified `qwen3_tts_*_test` on GPU (4 passed, 8 skipped due to missing model weights; 0 failures).
   - **Image** (`src/models/qwen_image_21`): 7 includes across 4 files migrated. Compiled `gufo_qwen_image_21`, verified `qwen_image_21_*_test` on GPU (3 passed, 0 skipped; 0 failures) and `qwen_image_21_test` (1 passed, 0 skipped; 0 failures).
   - **H3** (`src/models/minimax_h3`): 11 includes across 7 files migrated. Compiled `gufo_minimax_h3`, verified `minimax_h3_*_test` on GPU (6 passed, 1 skipped due to missing model weights; 0 failures).
   - **Qwen 27B** (`src/models/qwen`): 53 includes across 31 files migrated. Compiled `gufo_qwen27b`, verified `qwen27b_*_test` (3 passed, 1 skipped due to missing model weights; 0 failures) and continuation identity golden (1 passed, 0 skipped; 0 failures).
   - **Flash-Next** (`src/models/qwen38_flash_next`): 29 includes across 19 files migrated. Compiled `gufo_qwen38_flash_next`, verified all 22 `qwen38_flash_next.*` tests (22 passed, 0 skipped; 0 failures) and continuation identity golden (1 passed, 0 skipped; 0 failures).
   - **DeepSeek** (`src/models/deepseek_v4_flash`): 20 includes across 12 files migrated. Compiled `gufo_deepseek_v4_flash`, verified `deepseek_v4_flash_continuation_identity_test` on GPU (1 passed, 0 skipped; 0 failures) and CPU (1 passed, 0 skipped; 0 failures).
   - Total migrated: 152 includes across 95 files.
   - Preserved own root-relative includes (`src/models/<self>/**`: 386 total) and family library includes (`src/families/qwen/**`: 65 total).

3. **Ratchet Isolation Enforced**:
   - Added `--fail-on-model-core-includes` to `tools/ci/check-extension-isolation.py` enforcing zero production `src/models/**` includes of `src/core/**` or `src/cache/**`.
   - Added unit test `test_model_core_boundary_blocks_core_and_cache_includes_from_production_models` in `tests/tools/check_extension_isolation_test.py` with negative proof asserting detection of synthetic violations, and `test_repository_satisfies_the_model_core_boundary`.
   - Added CTest gate `extension_model_core_includes_gate` to root `CMakeLists.txt` (passes in both `cpu-test` and `gpu-test`).

4. **All Gates Verified**:
   - Gate 1 (`check-extension-api.py`): PASS (SURFACE unchanged, 97 paths).
   - Gate 2 (`check-extension-isolation.py` with all 5 ratchet flags): PASS (0 boundary violations).
   - Gate 3 (`check_extension_isolation_test.py`): PASS (24/24 tests OK).
   - Gate 4 (`check_extension_api_test.py`): PASS (7/7 tests OK).
   - Gate 5 (Continuation identity goldens): PASS on both CPU and GPU presets.
   - Gate 6 (Code format check): PASS (785 C++ files formatted).
   - Gate 7 (Docs checks): PASS (11/11 tests OK).
   - Gate 8 (Windows portability audit): PASS (zero `<windows.h>` or `_WIN32` introduced).

5. **Evidence Produced**:
   - `docs/model-extensions/evidence/card11/model-include-inventory.md`
   - `docs/model-extensions/evidence/card11/isolation-ratchet.md`
   - `docs/model-extensions/evidence/card11/build-and-tests.txt`
   - `docs/model-extensions/evidence/card11/pending-commands.txt`
   - `docs/model-extensions/evidence/card11/RESUME-NOTES.md`
