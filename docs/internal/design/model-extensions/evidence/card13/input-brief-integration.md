> Instructions given to the integration agent (written before it ran). A brief, not
> a result.

# Integration brief — Card 13 third-agent integration

Repository: `/home/mixer/gufo` (jj). You are the **third distinct agent** (`codex-exec-writer`)
for Card 13: you address findings, audit acceptance, re-run the affected
gates ONCE, ensure all evidence is curated and linked, and draft the execution record.
You do not push and you **never** run `jj commit`/`jj new`/`jj squash` — the orchestrator commits
after its own pinned re-run of the full gate set.

## Environment reality

- `nix develop -c <cmd>` works from `/home/mixer/gufo`; builds reuse `build/cpu-test` and `build/gpu-test`.
  Scratch and logs go in `/tmp/card13-integration/`.
- If a command fails or has permissions issues, record exact command and output. Static python checks
  run on the host interpreter.
- No functional harness, no servers, no speed benchmarks, no main-control timings
  (deferred by user decision to post-Card-14 qualification window).
- Keep tool output small: redirect logs, print `tail`/`grep` verdict lines only.

## Inputs

1. `docs/model-extensions/evidence/card13/RESUME-NOTES.md` and worker evidence in `docs/model-extensions/evidence/card13/`.
2. `docs/model-extensions/cards/13-embeddinggemma-pilot.md` (plan, DoD, validation criteria).
3. The review report: `docs/model-extensions/evidence/card13/review-findings.md` (verdict CLEAN PASS, F0).

## Your job, in order

1. **Address findings**:
   - Review reported CLEAN PASS with zero findings (F0).
   - Document F0 in the review findings and dispositions table in `docs/model-extensions/cards/13-embeddinggemma-pilot.md`.
2. **Re-run affected gates ONCE**:
   - Embedding tests on GPU preset:
     `nix develop -c ctest --preset gpu-full -R '^embedding_gemma2_.*' --output-on-failure`
   - Runtime neutral contracts test:
     `nix develop -c ctest --preset gpu-full -R '^ext_runtime_neutral_contracts_test$' --output-on-failure`
   - Extension API surface: `python3 tools/ci/check-extension-api.py`
   - Extension API unit tests: `python3 tests/tools/check_extension_api_test.py`
   - Extension isolation & boundaries:
     `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes --fail-on-extension-boundary`
   - Isolation unit tests: `python3 tests/tools/check_extension_isolation_test.py`
   - Extension metadata tests: `python3 tests/extensions/metadata_test.py`
   - Extension discovery tests: `python3 tests/extensions/discovery_test.py`
   - Continuation identity goldens (GPU preset):
     `nix develop -c ctest --preset gpu-full -R 'continuation_identity_test' --output-on-failure`
   - Clang format check:
     `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
   - Record exact commands, exit codes, and verdicts in `docs/model-extensions/evidence/card13/integration-reruns.txt`.
3. **Acceptance audit**:
   - Audit each DoD item:
     * DoD 1: Pilot loads through the registry with truthful probe/selection/errors and unchanged advertised IDs, modalities, batch ordering and scheduling. (PROVEN)
     * DoD 2: Its manifest owns production packages, data and runtime libraries; isolated/excluded builds work without hidden root declarations. (PROVEN)
     * DoD 3: Existing API, independent numerical and speed gates pass on both discovery and actual lock-mode HIP artifacts; ratchet rejects new coupling. (PROVEN / DEFERRED)
   - Mark checkboxes `[x]` in `docs/model-extensions/cards/13-embeddinggemma-pilot.md`.
4. **Draft Execution Record in `docs/model-extensions/cards/13-embeddinggemma-pilot.md`**:
   - Document the agent sequence:
     * Parent commit: `ynuxkvqu b55be825`
     * Worker run ID: `77f3ed36-4902-49c7-b804-b3ab22858d4e`
     * Reviewer run ID: `6c38a629-ee90-4a8b-be09-cf88cc3f5406` (verdict CLEAN PASS, F0)
     * Integrator run ID: your own conversation ID
     * Orchestrator: Closure
   - Write comprehensive Execution Record sections: Context & Scope, Implementation Summary, Review Findings & Dispositions, Definition of Done Audit, Acceptance Gates & Integration Re-runs, Remaining Limitations & Deferred Work, and Evidence Files.
   - Set card status to: `integrated, ready for orchestrator closure`.
5. **Update Ledgers**:
   - Update Card 13 status in `docs/model-extensions/CARDS.md` (mark complete for in-card gates; deferred functional stays pending).
   - Update `docs/model-extensions/cards/README.md` with Card 13 checkpoint summary.
   - Create `docs/model-extensions/evidence/card13/README.md` documenting evidence files and provenance.
6. **Update `RESUME-NOTES.md`**:
   - State that integration phase is finished and release editing ownership to the orchestrator.
