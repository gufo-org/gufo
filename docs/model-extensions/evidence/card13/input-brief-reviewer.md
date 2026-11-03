> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 13 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 13.
**Read-only**: do not edit, create, delete, or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed if your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions).
2. `docs/model-extensions/cards/13-embeddinggemma-pilot.md` (plan, DoD, validation).
3. RFC sections implementation-plan step 8 (embeddings pilot), "Layout", "Registration and selection", "Capability interfaces".
   `docs/model-extensions/RFC.md` must be byte-identical — verify with `jj diff --name-only`
   and report if it moved.
4. The actual change: `jj diff` against parent commit `ynuxkvqu b55be825` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card13-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card13/` — including
   `input-brief-worker.md` (the plan the worker was given), `RESUME-NOTES.md`,
   `pilot-inventory.md`, `build-and-tests.txt`, and `pending-commands.txt`.

## What to judge

- **Does the diff implement Card 13, not a smaller or larger thing?**
  * Manifests & Entry point:
    - `src/models/embedding_gemma2/extension.json` matches `schemas/extension.schema.json`.
    - Declares unconditional registration target `gufo_embedding_gemma2`, `gufo_embedding_gemma2_hip` (`requires: ["hip"]`), provider `embeddinggemma-2`, capability `embeddings`, keys `safetensors:embedding_gemma2` and `safetensors:EmbeddingGemma2Model`, option `--default-prompt-name`.
    - `src/models/embedding_gemma2/extension.cmake` replaces old `CMakeLists.txt` using `gufo_extension_target` and `gufo_extension_hip_sources`.
    - HIP-free pure artifact probe (`ProbeEmbeddingGemma2`) inspecting metadata only (`config.json`), without loading weights or calling HIP.
    - Provider registration using generated constants (`gufo/ext/generated/embedding_gemma2_constants.hpp`).
  * Decoupling serving boundary:
    - Serving headers (`src/cli/serve/embedding_service.hpp`, `embeddings_api.cpp`, `embedding_service.cpp`) contain zero includes of `src/models/embedding_gemma2/*`.
    - Serving layer interacts with EmbeddingGemma solely through neutral contracts (`include/gufo/ext/embeddings.hpp`, `ext::FindEmbeddingProvider`).
    - Internal includes in `src/models/embedding_gemma2/` are extension-relative (`"config.hpp"`, `"loader.hpp"`, etc.).
    - Ownership of FFmpeg/libav and libsndfile/libsoxr audio/video decoding logic preserved in the extension.
  * Public extension API surface:
    - `include/gufo/ext/embeddings.hpp` added as clean capability interface.
    - Version bumped to `0.1.3` in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`.
    - Entry added in `include/gufo/ext/CHANGES.md`.
    - `SURFACE` updated and verified with `python3 tools/ci/check-extension-api.py` (99 paths).
  * Build & Registration Ratchet:
    - `extensions.lock` contains `"embedding_gemma2"`.
    - Direct link `gufo_embedding_gemma2` removed from root `CMakeLists.txt` (`gufo`, `gufo_http`). Built via `gufo_build_discovered_extensions()`, linked via `gufo_target_extensions()`.
    - `tools/ci/check-extension-isolation.py` enforces no `src/cli/` includes of `src/models/embedding_gemma2/` (and unit tests pass).
    - Excluded extension builds (`-DGUFO_EXTENSIONS=""`) build cleanly without unresolved symbols.
    - Shipped lock-mode build (`-DGUFO_EXTENSIONS_FROM_LOCK=ON`) compiles and runs cleanly.
- **Behavior & Coverage Preservation:**
  * No test assertions weakened or removed.
  * All 15 embedding tests (`ctest -R '^embedding_gemma2_.*'`) pass or skip cleanly on missing external weights.
  * Continuation identity goldens pass byte-identically on CPU and GPU presets.
  * No behavior change or regression in embedding serving API.
- **Evidence Honesty:**
  * Every claim needs a command, exit code, and verdict line.
  * A skipped or asset-missing test, an unexecuted command, or an inconclusive measurement must NOT be presented as a pass.
  * Deferred functional work must appear as PENDING with its verbatim command in `pending-commands.txt`.
  * Check the 3 DoD items against retained evidence and say which are proven, partially proven, or unproven.
- **Repo Conventions:**
  * Windows-safe CMake and C++ (no `<windows.h>` or `_WIN32` outside `src/core/platform/`).
  * No file > 1 MB under `docs/`.
  * Scratch only in `/tmp/card13-*`.
  * Formatting passes (`tools/ci/check-format.py`).

## Output

Save findings to `docs/model-extensions/evidence/card13/review-findings.md`,
and return the same content in your final message. Number findings F1…Fn. For each:
**severity** (Critical / Major / Minor / Nit), **paths + line refs**, **what is wrong**,
**practical impact**, **suggested correction**, and **how to verify** (exact command).
If no findings, state clean PASS across all audited areas.
Then:
- A definition-of-done audit table (3 items: proven / partially proven / unproven, with the evidence path or missing proof),
- The list of commands actually run with exit codes (or exact gate commands to run),
- Explicit "no finding" statements for areas checked and found clean.
