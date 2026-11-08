> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Card 13: Migrate the EmbeddingGemma pilot

You are the **implementation agent** (`worker`) for Card 13 of the model-extensions RFC in
`/home/mixer/gufo` (jj working copy `@`, parent commit `ynuxkvqu b55be825`). You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. Task-owned paths:
   - `CMakeLists.txt`
   - `extensions.lock`
   - `src/models/embedding_gemma2/**` (including `extension.json`, `extension.cmake`, `extension.cpp`, headers, sources, and manifests)
   - `src/cli/serve/serve.cpp`
   - `src/cli/serve/http_server.hpp`, `src/cli/serve/http_server.cpp`
   - `src/cli/serve/embeddings_api.hpp`, `src/cli/serve/embeddings_api.cpp`
   - `src/cli/serve/embedding_service.hpp`, `src/cli/serve/embedding_service.cpp` (or moving service/provider implementation to model extension and exposing clean interface)
   - `include/gufo/ext/**` (if public extension interface or constants are updated)
   - `src/ext/**`
   - `tests/models/embedding_gemma2/**`
   - `tools/ci/check-extension-isolation.py`
   - `tests/tools/check_extension_isolation_test.py`
   - `docs/model-extensions/cards/13-embeddinggemma-pilot.md`
   - `docs/model-extensions/evidence/card13/**`
   - `docs/model-extensions/CARDS.md` (status updates)
4. **No functional harness.** `tests/functional/run.py`, `prompt_lookup.py`, speed
   benchmarks, and main-control matched timings are deferred by user decision (2026-10-31)
   to the single post-Card-14 qualification window. Do NOT run them or start a `gufo serve`
   server. Record deferred requirements as PENDING with their verbatim commands in
   `docs/model-extensions/evidence/card13/pending-commands.txt`.
5. Scratch goes in `/tmp/card13-*` only (logs, caches, temp files) and MUST be deleted
   when tests finish. No evidence file over 1 MB under `docs/`.
   No raw build logs, no per-row report trees under `docs/`.
6. Keep tool output small: redirect build and test logs to `/tmp/card13-logs/<name>.log`
   and inspect only using `tail -n 30` or grep. Never output entire build logs.
7. Host RAM: Before running any GPU build or test, check `free -g` and ensure at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
8. **Preserve unrelated work and user changes.** No behavior change.

## Objectives & Background

Card 13 implements RFC Step 8 for Embeddings:
"Migrate EmbeddingGemma pilot, extension-relative includes, metadata/options/pure probe/artifact resolution/provider/model IDs. Preserve extension batch scheduling and audio decoder ownership. Move model production dependency/data/runtime-library declarations into manifests. Existing API/numerical quality/speed gates, default and lock artifact, isolated build and fail-on-violation ratchet."

EmbeddingGemma 2 is the pilot extension and the first real provider. It exercises:
1. `src/models/embedding_gemma2/extension.json` metadata manifest (name: `embedding_gemma2`, provider: `embeddinggemma-2`, capability: `embeddings`, keys: `safetensors:embedding_gemma2`, `safetensors:EmbeddingGemma2Model`, model_ids: `["embeddinggemma-2"]`, canonical options: `--served-model-name`, `--default-prompt-name`).
2. `src/models/embedding_gemma2/extension.cmake` build manifest using `gufo_extension_target` and `gufo_extension_hip_sources`.
3. Extension-relative includes inside `src/models/embedding_gemma2/` (e.g. `"config.hpp"`, `"audio_decoder.hpp"`, `"loader.hpp"`).
4. Pure probe (HIP-free) verifying `config.json` model_type and architecture without loading weights or initializing HIP.
5. Artifact resolution: resolving model root, prompt mappings (`config_sentence_transformers.json`), and weights.
6. Decoupling core serving (`src/cli/serve/`): `http_server` and `embeddings_api` should no longer include concrete `src/models/embedding_gemma2/*` headers. `EmbeddingService` or an abstract embedding interface / provider bridge moves under the model or is accessed neutrally via the registry / capability contract.
7. Populating `extensions.lock` with `embedding_gemma2`.
8. Updating `tools/ci/check-extension-isolation.py` with a fail-on-violation ratchet ensuring `src/cli/` has 0 includes of `src/models/embedding_gemma2/`.
9. Testing both discovery mode and lock mode (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`), as well as excluded extension builds (`-DGUFO_EXTENSIONS=""`).

## Step-by-Step Implementation Plan

### Step 1: Inventory & Decouple Serving Boundary
- Inspect all references from `src/cli/` to `src/models/embedding_gemma2/` (currently 8 includes: 3 in `embedding_service.hpp`, 5 in `embedding_service.cpp`, and usage in `serve.cpp`).
- Move the provider implementation and model calls under the `embedding_gemma2` extension boundary.
- Ensure `src/cli/serve/` has zero `#include "src/models/embedding_gemma2/..."` includes.
- Switch internal includes in `src/models/embedding_gemma2/` to extension-relative own includes.

### Step 2: Declare Metadata & Extension Entry Point
- Create `src/models/embedding_gemma2/extension.json` conforming to `schemas/extension.schema.json`.
- Implement pure artifact probe (HIP-free) that checks `config.json` for `model_type == "embedding_gemma2"` and `architecture == "EmbeddingGemma2Model"`.
- Implement artifact resolution for prompt files and model directory.
- Define registration entry point `embedding_gemma2_register` and provider registration `embedding_gemma2_provider_register` using generated constants from `gufo/ext/generated/embedding_gemma2_constants.hpp`.

### Step 3: Manifest-driven Build in `extension.cmake`
- Create `src/models/embedding_gemma2/extension.cmake` defining `gufo_embedding_gemma2` (and registration target if separated) via `gufo_extension_target` and `gufo_extension_hip_sources`.
- Preserve FFmpeg/libav and libsndfile/soxr dependencies and video libraries loading behavior.
- Remove hardcoded `add_subdirectory(src/models/embedding_gemma2)` and direct links from root `CMakeLists.txt` (it is now built via `gufo_build_discovered_extensions()` and linked via `gufo_target_extensions(gufo ...)`).
- Update `src/models/embedding_gemma2/library.json` target ownership or claims as needed.

### Step 4: Lock Selection and Isolation Ratchet
- Add `"embedding_gemma2"` to `extensions.lock`.
- In `tools/ci/check-extension-isolation.py`, add fail-on-violation enforcement that no `src/cli/**` files include `src/models/embedding_gemma2/**`. Add a negative unit test in `tests/tools/check_extension_isolation_test.py`.
- If public extension API headers or `SURFACE` are affected, run `python3 tools/ci/check-extension-api.py --write` and bump version / update `CHANGES.md` if declarations changed.

### Step 5: Verification & Evidence
- Run the full suite of focused checks:
  1. Affected CTests on CPU and GPU (`embedding_gemma2_*`).
  2. Continuation identity goldens on CPU and GPU.
  3. `python3 tools/ci/check-extension-api.py`.
  4. `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes` (and any new ratchet flag).
  5. `python3 tests/tools/check_extension_isolation_test.py`.
  6. `python3 tests/tools/check_extension_api_test.py`.
  7. `python3 tests/extensions/metadata_test.py` and `discovery_test.py`.
  8. Lock-mode build verification (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`).
  9. Excluded-extension build verification (`-DGUFO_EXTENSIONS=""`).
  10. `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`.
- Write evidence files:
  - `pilot-inventory.md`
  - `build-and-tests.txt`
  - `pending-commands.txt`
  - `RESUME-NOTES.md` (with explicit editing ownership handoff).
- Update `docs/model-extensions/cards/13-embeddinggemma-pilot.md`.
