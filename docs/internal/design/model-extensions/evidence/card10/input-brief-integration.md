> Instructions given to the integration agent (written before it ran). A brief, not
> a result.

# Integration brief — Card 10 third-agent integration

Repository: `/home/mixer/gufo` (jj). You are the **third distinct agent** (`codex-exec-writer`)
for Card 10: you audit acceptance (even though the review found 0 defects), re-run the affected
gates ONCE, ensure all evidence is curated and linked, and draft the execution record.
You do not push and you **never** run `jj commit`/`jj new`/`jj squash` — the orchestrator commits
after its own pinned re-run of the full gate set.

## Environment reality

- `nix develop -c <cmd>` works from `/home/mixer/gufo`; builds reuse `build/cpu-test` and `build/gpu-test`.
  Scratch and logs go in `/tmp/card10-integration/`.
- If a command fails or has permissions issues, record exact command and output. Static python checks
  run on the host interpreter.
- No functional harness, no servers, no speed benchmarks, no main-control timings
  (deferred by user decision to post-Card-14 qualification window).
- Keep tool output small: redirect logs, print `tail`/`grep` verdict lines only.

## Inputs

1. `docs/model-extensions/evidence/card10/RESUME-NOTES.md` and worker evidence in `docs/model-extensions/evidence/card10/`.
2. `docs/model-extensions/cards/10-public-api-surface.md` (plan, DoD, validation criteria).
3. The review report: `docs/model-extensions/evidence/card10/review-findings.md` (verdict PASS, 0 findings).

## Your job, in order

1. **Audit findings**: Review report had 0 findings (clean PASS). Record this in the disposition table.
2. **Re-run affected gates ONCE**:
   - `python3 tools/ci/check-extension-api.py`
   - `python3 tools/ci/check-extension-api.py --base a761679a --labels '["ext-api"]'`
   - `python3 tests/tools/check_extension_api_test.py`
   - `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership`
   - `python3 tests/tools/check_docs_test.py`
   - Format check: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (or host python3 tools/ci/check-format.py)
   - Bounded CTest selection in CPU and GPU:
     `nix develop -c ctest --test-dir build/cpu-test -R '^(ext_runtime_neutral_contracts_test|extension_runtime_test|extension_role_test|extension_api_policy_test|extension_api_surface|extension_metadata_test|extension_discovery_test|(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test)$' --output-on-failure`
     `nix develop -c ctest --test-dir build/gpu-test -R '^(ext_runtime_neutral_contracts_test|extension_runtime_test|extension_role_test|extension_api_policy_test|extension_api_surface|extension_metadata_test|extension_discovery_test|(qwen27b|qwen38_flash_next|deepseek_v4_flash)_continuation_identity_test)$' --output-on-failure`
   Record exact commands, exit codes, and verdicts in `docs/model-extensions/evidence/card10/integration-reruns.txt`.
3. **Acceptance audit**:
   - Audit each DoD item:
     * DoD 1: Every inventoried model-used header has classification and public replacement or approved private removal plan.
     * DoD 2: Measured SURFACE closure is reproducible; forwarded source declarations are governed equally with public headers.
     * DoD 3: Public CMake/schema contracts and API version/changelog gates reject missing required policy changes; no unsupported stability claim is made from wrapper paths alone.
   - All 3 items are proven. Mark checkboxes `[x]` in `docs/model-extensions/cards/10-public-api-surface.md`.
4. **Draft Execution Record in `docs/model-extensions/cards/10-public-api-surface.md`**:
   - Document the agent sequence:
     * Parent commit: `klwmvoxr a761679a`
     * Worker run ID: `6d5263a1-9f0d-46ef-a092-76804e8f1003`
     * Reviewer run ID: `922a7582-1ace-4ef7-b37c-e926d5df7c5b` (PASS, 0 findings)
     * Integrator run ID: your own conversation ID
     * Orchestrator: closure
   - Write comprehensive Execution Record sections: Scope audit, Findings and dispositions, Validation and acceptance audit, and Remaining limitations.
5. **Update Ledgers**:
   - Update Card 10 status in `docs/model-extensions/CARDS.md` (mark complete for in-card gates; deferred functional stays pending).
   - Update `docs/model-extensions/cards/README.md` with Card 10 checkpoint.
   - Create `docs/model-extensions/evidence/card10/README.md` documenting evidence files and provenance.
6. **Update `RESUME-NOTES.md`**:
   - State that integration phase is finished and release editing ownership to the orchestrator.
