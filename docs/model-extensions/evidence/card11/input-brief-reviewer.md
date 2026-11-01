> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 11 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 11.
**Read-only**: do not edit, create, delete, or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed if your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions).
2. `docs/model-extensions/cards/11-public-header-migrations.md` (plan, DoD, validation).
3. RFC sections "Include scheme", "Host services", "Family libraries", "Interface stability policy",
   implementation-plan step 7. `docs/model-extensions/RFC.md` must be byte-identical — verify
   with `jj diff --name-only` and report if it moved.
4. The actual change: `jj diff` against parent commit `xppvypvw 8d8ef356` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card11-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card11/` — including
   `input-brief-worker.md` (the plan the worker was given), `RESUME-NOTES.md`,
   `model-include-inventory.md`, and `isolation-ratchet.md`.

## What to judge

- **Does the diff implement Card 11, not a smaller or larger thing?**
  * All 8 models migrated: EmbeddingGemma, ASR, TTS, Image, H3, Qwen 27B, Flash-Next, DeepSeek.
  * Only private `src/core/` and `src/cache/` includes replaced with `<gufo/ext/...>` public headers.
  * Root-relative own includes (`src/models/<self>/**`) preserved until each model's migration card.
  * Family library includes (`src/families/qwen/**`) preserved.
  * Test-only serving includes in ASR (`src/models/qwen3_asr/tests/audio_api_test.cpp`) preserved for Card 12.
  * Top-level `CMakeLists.txt` adds `${CMAKE_SOURCE_DIR}/include` to `include_directories` so `<gufo/ext/...>` resolves.
  * No arithmetic, provider routing, or target ownership redesign mixed into the include change.
  * Isolation ratcheted: `--fail-on-model-core-includes` added to `tools/ci/check-extension-isolation.py`.
  * Unit test added in `tests/tools/check_extension_isolation_test.py` with negative control.
  * CTest gate `extension_model_core_includes_gate` added in `CMakeLists.txt`.
- **Public API & Surface Integrity:**
  * `SURFACE` must remain clean and stable (`python3 tools/ci/check-extension-api.py`).
  * No new public headers or unexpected declarations introduced.
- **Isolation Ratchet & Boundary Enforcement:**
  * `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes` passes.
  * Confirm zero production model files include `src/core/` or `src/cache/`.
- **Behavior Preservation:**
  * No behavior change or regression.
  * Identity golden tests pass byte-exactly on CPU and GPU.
- **Evidence Honesty:**
  * Every claim needs a command, exit code, and verdict line.
  * A skipped or asset-missing test, an unexecuted command, or an inconclusive measurement must NOT be presented as a pass.
  * Deferred functional work must appear as PENDING with its verbatim command.
  * Check the 3 DoD items against retained evidence and say which are proven, partially proven, or unproven.
- **Repo Conventions:**
  * Windows-safe CMake and C++ (no `<windows.h>` or `_WIN32` outside `src/core/platform/`).
  * No file > 1 MB under `docs/`.
  * Scratch only in `/tmp/card11-*`.
  * Formatting passes (`tools/ci/check-format.py`).

## Output

Save findings to `docs/model-extensions/evidence/card11/review-findings.md`,
and return the same content in your final message. Number findings F1…Fn. For each:
**severity** (Critical / Major / Minor / Nit), **paths + line refs**, **what is wrong**,
**practical impact**, **suggested correction**, and **how to verify** (exact command).
If no findings, state clean PASS across all audited areas.
Then:
- A definition-of-done audit table (3 items: proven / partially proven / unproven, with the evidence path or missing proof),
- The list of commands actually run with exit codes (or exact gate commands to run),
- Explicit "no finding" statements for areas checked and found clean.
