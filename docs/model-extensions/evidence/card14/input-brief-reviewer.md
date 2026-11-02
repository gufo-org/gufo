> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 14 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 14.
**Read-only**: do not edit, create, delete, or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed if your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions).
2. `docs/model-extensions/cards/14-generated-infrastructure.md` (plan, DoD, validation).
3. RFC sections implementation-plan step 9, "Tests, benchmarks, tools and documentation",
   "Functional registries", "Documentation and notices".
   `docs/model-extensions/RFC.md` must be byte-identical — verify with `jj diff --name-only`
   and report if it moved.
4. The actual change: `jj diff` against parent commit `qupqlxsk 642af9d6` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card14-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card14/` — including
   `input-brief-worker.md` (the plan the worker was given), `RESUME-NOTES.md`,
   `build-and-tests.txt`, and `pending-commands.txt`.

## What to judge

- **Does the diff implement Card 14, not a smaller or larger thing?**
  * Model Metadata Manifests:
    - Complete, valid `extension.json` files exist for all remaining 7 models:
      `deepseek_v4_flash`, `minimax_h3`, `qwen`, `qwen38_flash_next`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`.
    - `src/models/embedding_gemma2/extension.json` expanded to declare packaging, docs, tests, fixtures, and labels.
    - All 8 manifests strictly validate against `schemas/extension.schema.json`.
    - Truthfulness: unmigrated models declare their targets/providers truthfully without pretending they already have factories or `extension.cmake`.
  * Truthful Discovery Handling:
    - In `cmake/GufoExtensionDiscovery.cmake`: models lacking `extension.cmake` are marked `GUFO_EXTENSION_METADATA_ONLY = TRUE` and omitted from `gufo_selected_extensions` in discovery mode (`*`) with reason `"metadata only"`.
    - Explicit selection (`-DGUFO_EXTENSIONS=...`) for unmigrated models fails because `extension.cmake` is missing.
    - In `tools/ci/check-extension-isolation.py`: unmigrated metadata-only extensions are not treated as migrated extensions during boundary checks.
  * Production Derivation Independence:
    - `.devops/nix/package.nix`: filters out test/functional/tools/fixtures under `src/`.
    - Verified derivation hash invariance: `nix eval --raw .#packages.x86_64-linux.default.drvPath` is identical before and after touching model test files.
  * Functional Fragments & Infrastructure Generator:
    - `tests/functional/devlab.json` split into `tests/functional/devlab.core.json` and model fragments (`src/models/*/functional/devlab.json`).
    - `tests/functional/impact.json` split into `tests/functional/impact.core.json` and model fragments (`src/models/*/functional/impact.json`).
    - Generator `tools/ci/generate-infrastructure.py` manages `cmake/Checks.cmake`, `tools/ci/check-format.py`, `tools/ci/check-docs.py`, `docs/models/README.md`, `tests/functional/devlab.json`, and `tests/functional/impact.json`.
    - Drift check `python3 tools/ci/generate-infrastructure.py --check` exits 0 with zero drift.
    - Conformance tests in `tests/tools/generate_infrastructure_test.py` test missing metadata, duplicate metadata, drift detection, and conditional metadata.
  * Verification Gates:
    - Bounded hosted CPU contract and check-pr targets: `nix develop -c cmake --build --preset cpu-test --target check-pr`.
    - Bounded PR checks in Nix: `nix build .#checks.x86_64-linux.pr`.
    - Clang-format clean: `tools/ci/check-format.py`.
    - Documentation check clean: `tools/ci/check-docs.py`.
    - Python tests clean: `generate_infrastructure_test.py`, `metadata_test.py`, `discovery_test.py`, `check_extension_isolation_test.py`.
    - Continuation identity goldens clean: `continuation_identity` tests pass on CPU/GPU.
- **Evidence Honesty**:
  * Every claim needs a command, exit code, and verdict line.
  * No functional harness runs executed (user decision 2026-10-31); all deferred commands recorded as PENDING in `pending-commands.txt`.
  * Check the 4 DoD items against retained evidence and say which are proven, partially proven, or unproven.
- **Repo Conventions**:
  * Windows-safe CMake and C++ (no `<windows.h>` or `_WIN32` outside `src/core/platform/`).
  * No file > 1 MB under `docs/`.
  * Scratch only in `/tmp/card14-*`.
  * Preserved unrelated work and pending user changes.

## Output

Save findings to `docs/model-extensions/evidence/card14/review-findings.md`,
and return the same content in your final message. Number findings F1…Fn. For each:
**severity** (Critical / Major / Minor / Nit), **paths + line refs**, **what is wrong**,
**practical impact**, **suggested correction**, and **how to verify** (exact command).
If no findings, state clean PASS across all audited areas.
Then:
- A definition-of-done audit table (4 items: proven / partially proven / unproven, with the evidence path or missing proof),
- The list of commands actually run with exit codes (or exact gate commands to run),
- Explicit "no finding" statements for areas checked and found clean.
