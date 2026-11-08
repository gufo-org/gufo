> Instructions given to the independent review agent (written before it ran).
> A brief, not a result.

# Reviewer brief — Card 09 independent read-only review

Repository: `/home/mixer/gufo` (jj). You are the **independent reviewer** for Card 09.
**Read-only**: do not edit, create, delete or move any tracked file; do not run
`jj commit/new/squash/abandon`; do not start servers; do not run the functional
harness. Read-only jj and build/test execution are allowed **if** your profile has a
shell — otherwise emit the exact gate commands for the integration agent instead of
assuming results (the orchestrator does not re-run the same gate twice).

## Inputs, in this order

1. `AGENTS.md` (repo rules), `docs/model-extensions/CARDS.md` (shared acceptance gates
   + the 2026-10-31 "no functional harness during cards 06–14" decision),
   `docs/model-extensions/cards/README.md` (workflow, validation conventions),
   `docs/model-extensions/target-ownership.md`.
2. `docs/model-extensions/cards/09-serving-runtime-boundary.md` (plan, DoD, validation).
3. RFC sections "The text-generation boundary today", "Continuation state per model",
   "Target graph", "Host services", implementation-plan step 6. `docs/model-extensions/RFC.md`
   must be byte-identical — verify with `jj diff --name-only` and report if it moved.
4. The actual change: `jj diff` against parent `uytorytn c03aaa23` (use
   `jj diff --name-only` first, then per-file diffs; redirect big outputs to
   `/tmp/card09-review/*.log` and read with `head`/`grep` — never dump a whole
   build log into your context).
5. `docs/model-extensions/evidence/card09/` — including
   `input-brief-worker.md` (the plan the worker was given) and `RESUME-NOTES.md`.

## What to judge

- **Does the diff implement the card, not a smaller thing?** Required: a real
  `gufo_serving` target owning the seven serving sources (no more per-executable
  re-compilation); `gufo_ext_runtime` extended with neutral continuation state/snapshot
  bases, host budget queries, neutral runner/chat/tool/tokenizer/reasoning types and a
  default HIP probe, free of concrete model types in the contracts it introduces;
  neutral backing/resource contracts in the cache owner plus `CreateHipComponentCacheResources`
  next to `gufo_cache_hip` with injected budget/log callbacks; policy (queueing,
  envelopes, retention/admission, identity policy) still core/serving-owned; new
  dependency/behavior gates.
- **Behavior preservation.** Look hard for semantic drift hidden in moves: changed
  defaults, reordered initialisation, lost `if (BUILD_TESTING) -UNDEBUG`, lost
  `ENGINE_ENABLE_HIP` paths, dropped retry/limits (2 GiB pre-readiness reserve,
  host-reserve pressure retries), changed `std::function` capture lifetimes, changed
  virtual dispatch through the new aliases, changed identity-byte construction inputs,
  silent fallbacks where a cast used to be checked.
- **Layering claims vs. reality.** `src/models/**`, `src/families/**`, `src/cache/**`,
  `src/ext/**` must contain no `src/cli/**` include. Do not trust the gate: grep it
  yourself. Check the new isolation mode actually fails on a violation and that the
  link-graph guard cannot be satisfied by an empty or vacuous graph.
- **Interim edges** must be explicitly listed, not implied (serving still holds the
  text runners and `continuation_identity.cpp` model lines; model test binaries still
  link `gufo_serving` until Card 12; `include/gufo/ext/` stays unpublished until
  Card 10, so `SURFACE` must be unchanged and no extension-API version bump).
- **Evidence honesty.** Every claim needs a command, exit code and verdict line.
  A skipped/asset-missing test, an unexecuted command, a "will run later" or an
  inconclusive measurement must NOT be presented as a pass. Deferred functional work
  must appear as PENDING with its verbatim command. Check the three DoD boxes against
  the retained evidence and say which are unproven.
- **Repo conventions**: single canonical option/target names, no aliases; Windows-safe
  CMake (no `_WIN32`/`windows.h` outside `src/core/platform/`); no new
  `docs/` bloat (>1 MB files, raw logs, caches); scratch only under `/tmp/card09-*`;
  moved-source flag/assertion/sanitizer parity documented or intentionally changed;
  Conventional Commit readiness (do not commit yourself).
- **Scope creep** into Card 10/12 territory (public header publication, runner-header
  split, `gufo_test_support`) is a finding, as is a silent narrowing of the card.

## Output

Write `/tmp/card09-review/findings.md` and return the same content in your final
message. Number findings F1…Fn. For each: **severity** (Critical / Major / Minor /
Nit), **paths + line refs**, **what is wrong**, **practical impact**, **suggested
correction**, and **how to verify** (exact command). Then:
- a definition-of-done audit table (3 items: proven / partially proven / unproven, with
  the evidence path or the missing proof),
- the list of commands you actually ran with exit codes (or the list of commands you
  could not run and want the integration agent to run),
- explicit "no finding" statements for areas you checked and found clean, so coverage
  is visible.

Report what you could not check and why. Do not soften a finding to be helpful, and do
not propose weakening an acceptance criterion.
