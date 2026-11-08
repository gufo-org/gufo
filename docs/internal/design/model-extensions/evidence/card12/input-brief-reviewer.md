> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 12 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 12.
**Read-only**: do not edit, create, delete, or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed if your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions).
2. `docs/model-extensions/cards/12-model-api-test-support.md` (plan, DoD, validation).
3. RFC sections implementation-plan step 7b, "Tests, benchmarks, tools and documentation".
   `docs/model-extensions/RFC.md` must be byte-identical — verify with `jj diff --name-only`
   and report if it moved.
4. The actual change: `jj diff` against parent commit `ytumsxmz 7752857d` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card12-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card12/` — including
   `input-brief-worker.md` (the plan the worker was given), `RESUME-NOTES.md`,
   `model-api-test-inventory.md`, `test-support-architecture.md`, `build-and-tests.txt`, and
   `pending-commands.txt`.

## What to judge

- **Does the diff implement Card 12, not a smaller or larger thing?**
  * In-tree test support added: `gufo_test_support` declared in `CMakeLists.txt` under `if (BUILD_TESTING)` linking `gufo_http` and `gufo_serving`.
  * Fixtures aggregated in `tests/support/model_api_test_support.hpp` and backed by `tests/support/test_support.cpp`.
  * All 4 model API tests migrated:
    1. `src/models/qwen3_asr/tests/audio_api_test.cpp`
    2. `tests/models/embedding_gemma2/embeddings_api_test.cpp`
    3. `tests/models/qwen3_tts/audio_api_test.cpp`
    4. `tests/models/qwen_image_21/model_test.cpp`
  * Production CLI/serving includes (`src/cli/serve/*.hpp`) removed from the migrated tests and replaced with `#include "tests/support/model_api_test_support.hpp"`.
  * Production sources removed from test targets (`audio_asr_api.cpp`, `asr_service.cpp`, `embeddings_api.cpp`, `embedding_service.cpp`, `audio_tts_api.cpp`, `tts_service.cpp`).
  * Direct `gufo_http` link removed from `qwen_image_21_test` and replaced with `gufo_test_support`.
  * Isolation check `serving_boundary_violations()` in `tools/ci/check-extension-isolation.py` tightened by removing the `in_test_tree(path)` exemption; confirm that `src/models/**` now has 0 `src/cli/` includes across both production and test files.
  * Negative proof added in `tests/tools/check_extension_isolation_test.py`.
  * Remaining test coupling (DeepSeek, Qwen 27B, Flash-Next text runners and continuation identity goldens) explicitly cataloged in `model-api-test-inventory.md`.
- **Public API & Surface Integrity:**
  * `SURFACE` must remain clean and stable (`python3 tools/ci/check-extension-api.py`).
  * No test support headers leaked into `include/gufo/ext/` or `SURFACE`.
- **Production Isolation & Build Conditions:**
  * `gufo_test_support` must be strictly absent under `BUILD_TESTING=OFF`.
  * Not linked to `gufo`, `gufo_core`, `gufo_ext_runtime`, or any installed target.
- **Behavior & Coverage Preservation:**
  * No test assertions weakened or removed.
  * All 4 affected model API tests pass on both CPU and GPU presets.
  * Continuation identity goldens pass byte-identically on CPU and GPU presets.
- **Evidence Honesty:**
  * Every claim needs a command, exit code, and verdict line.
  * A skipped or asset-missing test, an unexecuted command, or an inconclusive measurement must NOT be presented as a pass.
  * Deferred functional work must appear as PENDING with its verbatim command.
  * Check the 3 DoD items against retained evidence and say which are proven, partially proven, or unproven.
- **Repo Conventions:**
  * Windows-safe CMake and C++ (no `<windows.h>` or `_WIN32` outside `src/core/platform/`).
  * No file > 1 MB under `docs/`.
  * Scratch only in `/tmp/card12-*`.
  * Formatting passes (`tools/ci/check-format.py`).

## Output

Save findings to `docs/model-extensions/evidence/card12/review-findings.md`,
and return the same content in your final message. Number findings F1…Fn. For each:
**severity** (Critical / Major / Minor / Nit), **paths + line refs**, **what is wrong**,
**practical impact**, **suggested correction**, and **how to verify** (exact command).
If no findings, state clean PASS across all audited areas.
Then:
- A definition-of-done audit table (3 items: proven / partially proven / unproven, with the evidence path or missing proof),
- The list of commands actually run with exit codes (or exact gate commands to run),
- Explicit "no finding" statements for areas checked and found clean.
