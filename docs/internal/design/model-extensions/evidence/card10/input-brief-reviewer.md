> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 10 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 10.
**Read-only**: do not edit, create, delete, or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed if your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions).
2. `docs/model-extensions/cards/10-public-api-surface.md` (plan, DoD, validation).
3. RFC sections "Include scheme", "Host services", "Family libraries", "Interface stability policy",
   implementation-plan step 7. `docs/model-extensions/RFC.md` must be byte-identical — verify
   with `jj diff --name-only` and report if it moved.
4. The actual change: `jj diff` against parent commit `klwmvoxr a761679a` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card10-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card10/` — including
   `input-brief-worker.md` (the plan the worker was given) and `RESUME-NOTES.md`.

## What to judge

- **Does the diff implement Card 10, not a smaller or larger thing?**
  * Core headers used by model code classified into stable, platform, experimental, host wrapper, or private with removal plan.
  * Public forwarding headers created under `include/gufo/ext/`, `include/gufo/ext/platform/`, `include/gufo/ext/experimental/`.
  * Forwarding headers must use `#pragma once` and forward directly to the underlying `src/` headers.
  * No serving headers (`src/cli/**` or serving policy) re-exported or leaked.
  * No model files under `src/models/` modified to consume public headers (that belongs to Card 11).
  * API version bumped to 0.1.2 in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake`.
  * Comprehensive entry for `0.1.2` in `include/gufo/ext/CHANGES.md`.
  * Transitive `include/gufo/ext/SURFACE` generated and measured. Output must be reproducible and stable.
  * `check_extension_api_test.py` extended to cover newly classified closure/declaration cases.
  * Representative HIP-less and HIP consumers compiled and tested.
- **Transitive closure integrity.**
  * Inspect `include/gufo/ext/SURFACE`. Are there any unexpected files or cycles?
  * Do any serving headers (`src/cli/**`) or model-internal headers leak into `SURFACE`?
  * Run `python3 tools/ci/check-extension-api.py` (without `--write`) — does it pass cleanly with exit 0?
- **Behavior preservation.**
  * No behavior change or regression in existing runtime / contracts.
  * Identity golden tests still pass byte-exactly.
- **Evidence honesty.**
  * Every claim needs a command, exit code, and verdict line.
  * A skipped or asset-missing test, an unexecuted command, or an inconclusive measurement must NOT be presented as a pass.
  * Deferred functional work must appear as PENDING with its verbatim command.
  * Check the 3 DoD items against retained evidence and say which are proven, partially proven, or unproven.
- **Repo conventions**:
  * Windows-safe CMake and C++ (no `<windows.h>` or `_WIN32` outside `src/core/platform/`).
  * No file > 1 MB under `docs/`.
  * Scratch only in `/tmp/card10-*`.
  * Formatting passes (`tools/ci/check-format.py`).

## Output

Write `/tmp/card10-review/findings.md` and save to `docs/model-extensions/evidence/card10/review-findings.md`,
and return the same content in your final message. Number findings F1…Fn. For each:
**severity** (Critical / Major / Minor / Nit), **paths + line refs**, **what is wrong**,
**practical impact**, **suggested correction**, and **how to verify** (exact command).
Then:
- A definition-of-done audit table (3 items: proven / partially proven / unproven, with the evidence path or missing proof),
- The list of commands actually ran with exit codes (or commands to run),
- Explicit "no finding" statements for areas checked and found clean.
