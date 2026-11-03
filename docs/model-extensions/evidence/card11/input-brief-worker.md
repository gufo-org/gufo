> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Card 11: Migrate models to public extension headers

You are the **implementation agent** (`worker`) for Card 11 of the model-extensions RFC in
`/home/mixer/gufo` (jj working copy `@`, parent commit `xppvypvw 8d8ef356`). You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. Task-owned paths:
   - `CMakeLists.txt`
   - `src/models/**`
   - `tools/ci/check-extension-isolation.py`
   - `tests/tools/check_extension_isolation_test.py`
   - `docs/model-extensions/cards/11-public-header-migrations.md`
   - `docs/model-extensions/evidence/card11/**`
   - `docs/model-extensions/CARDS.md` (status updates)
4. **No functional harness.** `tests/functional/run.py`, `prompt_lookup.py`, speed
   benchmarks, and main-control matched timings are deferred by user decision (2026-10-31)
   to the single post-Card-14 qualification window. Do NOT run them or start a `gufo serve`
   server. Record deferred requirements as PENDING with their verbatim commands in
   `docs/model-extensions/evidence/card11/pending-commands.txt`.
5. Scratch goes in `/tmp/card11-*` only (logs, caches, temp files) and MUST be deleted
   when tests finish. No evidence file over 1 MB under `docs/`.
   No raw build logs, no per-row report trees under `docs/`.
6. Keep tool output small: redirect build and test logs to `/tmp/card11-logs/<name>.log`
   and inspect only using `tail -n 30` or grep. Never output entire build logs.
7. Host RAM: Before running any GPU build or test, check `free -g` and ensure at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
8. **Preserve unrelated work and user changes.** No behavior change.

## Objectives & Background

Card 11 implements the second half of RFC Step 7:
"Mechanical public-header include migrations one model at a time, preserving root-relative own includes temporarily: EmbeddingGemma, ASR, TTS, Image, H3, Qwen 27B, Flash-Next, DeepSeek. Record per-model Linux/Windows and affected unit checks; keep each rewrite reviewable and ratchet only migrated boundaries."

Card 10 published 44 forwarding headers in `include/gufo/ext/` (`gufo/ext/*.hpp`,
`gufo/ext/platform/*.hpp`, `gufo/ext/experimental/*.hpp`, and `gufo/ext/host.hpp`).
Card 11 migrates model sources in `src/models/` to consume these public headers instead of
private `src/core/` and `src/cache/` includes.

## Step-by-Step Implementation Plan

### Step 1: Include Directory in CMakeLists.txt
In root `CMakeLists.txt` line 29:
Change `include_directories(${CMAKE_SOURCE_DIR})` to:
`include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)`
This ensures all model libraries, tools, and tests can locate `<gufo/ext/...>` headers.

### Step 2: Mechanical Model Include Migrations (Strict Order)
Migrate the eight models in the exact order specified by the RFC:
1. **EmbeddingGemma** (`src/models/embedding_gemma2`)
2. **ASR** (`src/models/qwen3_asr`)
3. **TTS** (`src/models/qwen3_tts`)
4. **Image** (`src/models/qwen_image_21`)
5. **H3** (`src/models/minimax_h3`)
6. **Qwen 27B** (`src/models/qwen`)
7. **Flash-Next** (`src/models/qwen38_flash_next`)
8. **DeepSeek** (`src/models/deepseek_v4_flash`)

#### Mapping Rules:
- Replace `#include "src/core/..."` with `<gufo/ext/...>`:
  - `src/core/platform/<leaf>.hpp` -> `<gufo/ext/platform/<leaf>.hpp>`
  - `src/core/hip/<kernel-or-alloc>.hpp` -> `<gufo/ext/experimental/<kernel-or-alloc>.hpp>`
  - `src/core/hip/detail/<detail>.hpp` -> `<gufo/ext/experimental/<detail>.hpp>`
  - `src/core/quant/<quant>.hpp` -> `<gufo/ext/experimental/<quant>.hpp>`
  - `src/core/speculative/<spec>.hpp` -> `<gufo/ext/<spec>.hpp>`
  - `src/core/logging/logging.hpp` -> `<gufo/ext/host.hpp>`
  - `src/core/crypto/sha256.hpp` -> `<gufo/ext/sha256.hpp>`
  - `src/core/<core-header>.hpp` -> `<gufo/ext/<core-header>.hpp>` (`json.hpp`, `image.hpp`, `cancellable_gate.hpp`, `gguf_reader.hpp`, `mapped_prefetch.hpp`, `model_config.hpp`, `reasoning.hpp`, `sampling.hpp`, `session_mode.hpp`, `utf8.hpp`)
- Replace `#include "src/cache/..."` with `<gufo/ext/...>`:
  - `src/cache/adapter.hpp` -> `<gufo/ext/adapter.hpp>`
  - `src/cache/ledger.hpp` -> `<gufo/ext/ledger.hpp>`
  - `src/cache/types.hpp` -> `<gufo/ext/types.hpp>`
- **Preserve temporary root-relative own includes** (`#include "src/models/<self>/..."`) — these migrate in each model's dedicated extension migration card.
- **Preserve family includes** (`#include "src/families/qwen/..."`) — these remain valid family library dependencies.
- **Preserve test-only ASR serving includes** (`src/cli/serve/asr_service.hpp`, `src/cli/serve/audio_asr_api.hpp` in `src/models/qwen3_asr/tests/audio_api_test.cpp`) — these are explicitly documented in Card 10/11 for removal in Card 12 (`gufo_test_support`).

#### Per-Model Verification:
Compile each model and run its affected unit tests before proceeding to the next model:
- `EmbeddingGemma`: build `gufo_embedding_gemma2`, run `embedding_gemma2_*_test`
- `ASR`: build `gufo_qwen3_asr`, run `qwen3_asr_*_test`
- `TTS`: build `gufo_qwen3_tts`, run `qwen3_tts_*_test`
- `Image`: build `gufo_qwen_image_21`, run `qwen_image_21_*_test`
- `H3`: build `gufo_minimax_h3`, run `minimax_h3_*_test`
- `Qwen 27B`: build `gufo_qwen27b`, run `qwen27b_*_test` and `qwen27b_continuation_identity_test`
- `Flash-Next`: build `gufo_qwen38_flash_next`, run `qwen38_flash_next.*` tests and identity golden
- `DeepSeek`: build `gufo_deepseek_v4_flash`, run `deepseek_v4_flash_continuation_identity_test`

### Step 3: Ratchet Isolation Check
In `tools/ci/check-extension-isolation.py`:
- Add a ratchet gate: `--fail-on-model-core-includes`.
- It must verify that NO production file under `src/models/**` includes `src/core/**` or `src/cache/**`.
- Add test coverage in `tests/tools/check_extension_isolation_test.py` verifying the ratchet catches violations.
- Add CTest test `extension_model_core_includes_gate` in root `CMakeLists.txt`.

### Step 4: Verification & Acceptance Gates
Run the full gate suite:
1. `python3 tools/ci/check-extension-api.py` (confirm SURFACE unchanged, 97 paths).
2. `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes`.
3. `python3 tests/tools/check_extension_isolation_test.py`.
4. `python3 tests/tools/check_extension_api_test.py`.
5. Continuation identity goldens (Card 02 requirement):
   `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity_test' --output-on-failure`.
   `nix develop -c ctest --test-dir build/gpu-test -R 'continuation_identity_test' --output-on-failure`.
6. Code format check: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`.
7. Docs check: `python3 tests/tools/check_docs_test.py`.
8. Windows portability audit: confirm zero `<windows.h>` or `_WIN32` outside `src/core/platform/`.

### Step 5: Evidence Documentation
Create under `docs/model-extensions/evidence/card11/`:
- `model-include-inventory.md`: Complete before/after table of migrated includes per model, plus remaining intentional coupling (own root-relative, family, and the exempt ASR test edge).
- `isolation-ratchet.md`: Verification of the ratchet, per-model include edge counts before and after.
- `build-and-tests.txt`: Command log of compilations and tests.
- `pending-commands.txt`: Verbatim list of deferred functional/benchmark commands per 2026-10-31 user decision.
- `RESUME-NOTES.md`: Summary of actions, files touched, scratch locations, and clean handoff.
- Update `docs/model-extensions/cards/11-public-header-migrations.md` checking off DoD items and drafting the initial record.

End with an explicit editing-ownership handoff. Do NOT commit.
