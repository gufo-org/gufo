> Instructions given to the integration agent (written before it ran). A brief, not
> a result.

# Integration brief — Card 12 third-agent integration

Repository: `/home/mixer/gufo` (jj). You are the **third distinct agent** (`codex-exec-writer`)
for Card 12: you address findings, audit acceptance, re-run the affected
gates ONCE, ensure all evidence is curated and linked, and draft the execution record.
You do not push and you **never** run `jj commit`/`jj new`/`jj squash` — the orchestrator commits
after its own pinned re-run of the full gate set.

## Environment reality

- `nix develop -c <cmd>` works from `/home/mixer/gufo`; builds reuse `build/cpu-test` and `build/gpu-test`.
  Scratch and logs go in `/tmp/card12-integration/`.
- If a command fails or has permissions issues, record exact command and output. Static python checks
  run on the host interpreter.
- No functional harness, no servers, no speed benchmarks, no main-control timings
  (deferred by user decision to post-Card-14 qualification window).
- Keep tool output small: redirect logs, print `tail`/`grep` verdict lines only.

## Inputs

1. `docs/model-extensions/evidence/card12/RESUME-NOTES.md` and worker evidence in `docs/model-extensions/evidence/card12/`.
2. `docs/model-extensions/cards/12-model-api-test-support.md` (plan, DoD, validation criteria).
3. The review report: `docs/model-extensions/evidence/card12/review-findings.md` (verdict CLEAN PASS, F0).

## Your job, in order

1. **Address findings**:
   - Review reported CLEAN PASS with zero findings (F0).
   - Document F0 in the review findings and dispositions table in `docs/model-extensions/cards/12-model-api-test-support.md`.
2. **Re-run affected gates ONCE**:
   - Model API CTests on CPU and GPU:
     `nix develop -c ctest --test-dir build/cpu-test -R '^(qwen3_asr_audio_api_test|qwen_image_21_test|qwen3_tts_audio_api_test|embedding_gemma2_embeddings_api_test)$' --output-on-failure`
     `nix develop -c ctest --test-dir build/gpu-test -R '^(qwen3_asr_audio_api_test|qwen_image_21_test|qwen3_tts_audio_api_test|embedding_gemma2_embeddings_api_test)$' --output-on-failure`
   - Extension API surface: `python3 tools/ci/check-extension-api.py`
   - Extension isolation & boundaries: `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership --fail-on-model-core-includes`
   - Isolation unit tests: `python3 tests/tools/check_extension_isolation_test.py`
   - Extension API unit tests: `python3 tests/tools/check_extension_api_test.py`
   - Documentation unit tests: `python3 tests/tools/check_docs_test.py`
   - Continuation identity goldens (CPU & GPU):
     `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity_test' --output-on-failure`
     `nix develop -c ctest --test-dir build/gpu-test -R 'continuation_identity_test' --output-on-failure`
   - Format check: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`
   Record exact commands, exit codes, and verdicts in `docs/model-extensions/evidence/card12/integration-reruns.txt`.
3. **Acceptance audit**:
   - Audit each DoD item:
     * DoD 1: Affected model API tests use test support, with no production CLI includes remaining at the migrated boundaries. (PROVEN)
     * DoD 2: Test support is absent from production link/install and is not a public external-extension dependency. (PROVEN)
     * DoD 3: Fixture lifecycle preserves request/error/cancellation assertions rather than weakening model API coverage. (PROVEN)
   - Mark checkboxes `[x]` in `docs/model-extensions/cards/12-model-api-test-support.md`.
4. **Draft Execution Record in `docs/model-extensions/cards/12-model-api-test-support.md`**:
   - Document the agent sequence:
     * Parent commit: `ytumsxmz 7752857d`
     * Worker run ID: `02a253d3-11d5-42ec-ae8e-d89e20ef0716`
     * Reviewer run ID: `397daa87-23c0-4244-acce-79507e4def2a` (verdict CLEAN PASS, F0)
     * Integrator run ID: your own conversation ID
     * Orchestrator: Closure
   - Write comprehensive Execution Record sections: Scope audit, Implementation summary, Findings and dispositions, Validation and acceptance audit, and Remaining limitations.
   - Set status to: `integrated, ready for orchestrator closure`.
5. **Update Ledgers**:
   - Update Card 12 status in `docs/model-extensions/CARDS.md` (mark complete for in-card gates; deferred functional stays pending).
   - Update `docs/model-extensions/cards/README.md` with Card 12 checkpoint summary.
   - Create `docs/model-extensions/evidence/card12/README.md` documenting evidence files and provenance.
6. **Update `RESUME-NOTES.md`**:
   - State that integration phase is finished and release editing ownership to the orchestrator.
