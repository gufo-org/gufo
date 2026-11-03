> Instructions given to the integration agent (written before it ran). A brief, not
> a result.

# Integration brief — Card 11 third-agent integration

Repository: `/home/mixer/gufo` (jj). You are the **third distinct agent** (`codex-exec-writer`)
for Card 11: you address findings, audit acceptance, re-run the affected
gates ONCE, ensure all evidence is curated and linked, and draft the execution record.
You do not push and you **never** run `jj commit`/`jj new`/`jj squash` — the orchestrator commits
after its own pinned re-run of the full gate set.

## Environment reality

- `nix develop -c <cmd>` works from `/home/mixer/gufo`; builds reuse `build/cpu-test` and `build/gpu-test`.
  Scratch and logs go in `/tmp/card11-integration/`.
- If a command fails or has permissions issues, record exact command and output. Static python checks
  run on the host interpreter.
- No functional harness, no servers, no speed benchmarks, no main-control timings
  (deferred by user decision to post-Card-14 qualification window).
- Keep tool output small: redirect logs, print `tail`/`grep` verdict lines only.

## Inputs

1. `docs/model-extensions/evidence/card11/RESUME-NOTES.md` and worker evidence in `docs/model-extensions/evidence/card11/`.
2. `docs/model-extensions/cards/11-public-header-migrations.md` (plan, DoD, validation criteria).
3. The review report: `docs/model-extensions/evidence/card11/review-findings.md` (verdict PASS, 1 Nit).

## Your job, in order

1. **Address findings**:
   - F1 (Nit / Advisory): In `docs/model-extensions/cards/11-public-header-migrations.md` and `docs/model-extensions/evidence/card11/RESUME-NOTES.md`, adjust test reporting language so that missing-asset skips are explicitly separated from executed passes (e.g., "2 passed, 6 skipped (missing model weights); 0 failures" rather than "8/8 pass (100%)"). Record this resolution in the findings disposition table.
2. **Re-run affected gates ONCE**:
   - `python3 tools/ci/check-extension-api.py`
   - `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes`
   - `python3 tests/tools/check_extension_isolation_test.py`
   - `python3 tests/tools/check_extension_api_test.py`
   - `python3 tests/tools/check_docs_test.py`
   - Format check: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (or host python3 tools/ci/check-format.py)
   - Bounded CTest selection in CPU and GPU:
     `nix develop -c ctest --test-dir build/cpu-test -R '^(extension_model_core_includes_gate|(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test)$' --output-on-failure`
     `nix develop -c ctest --test-dir build/gpu-test -R '^(extension_model_core_includes_gate|(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test)$' --output-on-failure`
   Record exact commands, exit codes, and verdicts in `docs/model-extensions/evidence/card11/integration-reruns.txt`.
3. **Acceptance audit**:
   - Audit each DoD item:
     * DoD 1: All eight models use published core/platform/experimental paths in place of classified private core paths. (PROVEN)
     * DoD 2: Every rewrite has per-model Linux/Windows and affected unit evidence; remaining own/family coupling is explicitly inventoried. (PROVEN for Linux build and affected unit tests, static Windows audit; native Windows binary build deferred to CI).
     * DoD 3: No new private dependency is hidden in the isolation baseline or accepted as a migration exception. (PROVEN)
   - Mark checkboxes `[x]` in `docs/model-extensions/cards/11-public-header-migrations.md`.
4. **Draft Execution Record in `docs/model-extensions/cards/11-public-header-migrations.md`**:
   - Document the agent sequence:
     * Parent commit: `xppvypvw 8d8ef356`
     * Worker run ID: `1c0adb86-f704-460b-9694-93aa312ac1cd`
     * Reviewer run ID: `70a872ba-869f-42b1-80e6-b71880841633` (PASS, 1 Nit)
     * Integrator run ID: your own conversation ID
     * Orchestrator: Closure
   - Write comprehensive Execution Record sections: Scope audit, Findings and dispositions, Validation and acceptance audit, and Remaining limitations.
5. **Update Ledgers**:
   - Update Card 11 status in `docs/model-extensions/CARDS.md` (mark complete for in-card gates; deferred functional stays pending).
   - Update `docs/model-extensions/cards/README.md` with Card 11 checkpoint summary.
   - Create `docs/model-extensions/evidence/card11/README.md` documenting evidence files and provenance.
6. **Update `RESUME-NOTES.md`**:
   - State that integration phase is finished and release editing ownership to the orchestrator.
