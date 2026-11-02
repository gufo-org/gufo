# Card 13 Evidence Index and Provenance

This directory contains the durable evidence artifacts for **Card 13: Migrate the EmbeddingGemma pilot** (RFC Step 8 embeddings).

## Provenance and Agent Sequence

- **Parent Commit**: `ynuxkvqu b55be825` ("model-extensions-2 | test(extensions): add in-tree model api test support and isolate serving fixtures")
- **Implementation Worker**: Agent `77f3ed36-4902-49c7-b804-b3ab22858d4e`
- **Independent Reviewer**: Agent `6c38a629-ee90-4a8b-be09-cf88cc3f5406` (Verdict: CLEAN PASS, F0)
- **Review Integrator**: Agent `b3930d21-4138-46f7-b64f-835f02539530`
- **Orchestrator**: Final closure

## Evidence Files

| File | Author / Role | Description |
|---|---|---|
| [`input-brief-worker.md`](input-brief-worker.md) | Orchestrator | Input task brief provided to the implementation worker agent. |
| [`input-brief-reviewer.md`](input-brief-reviewer.md) | Orchestrator | Input task brief provided to the independent review agent. |
| [`input-brief-integration.md`](input-brief-integration.md) | Orchestrator | Input task brief provided to the integration agent. |
| [`pilot-inventory.md`](pilot-inventory.md) | Worker | Initial inventory of serving includes and leaked symbols before migration. |
| [`build-and-tests.txt`](build-and-tests.txt) | Worker | Initial execution logs for build, diagnostics, and test suites during implementation. |
| [`pending-commands.txt`](pending-commands.txt) | Worker / Integrator | Verbatim commands for functional harness and speed benchmarks deferred to the post-Card-14 qualification window. |
| [`review-findings.md`](review-findings.md) | Reviewer | Independent read-only audit report recording CLEAN PASS with zero findings (F0) across all validation areas. |
| [`integration-reruns.txt`](integration-reruns.txt) | Integrator | Exact commands, exit codes, and verdicts for the single integration re-run of all 10 affected verification gates. |
| [`orchestrator-gates.txt`](orchestrator-gates.txt) | Orchestrator | Pinned gate set re-runs, Codex review resolutions (P1, P2), and final closure verification. |
| [`RESUME-NOTES.md`](RESUME-NOTES.md) | Worker / Integrator | Handoff state notes and release of editing ownership. |
| [`README.md`](README.md) | Integrator | This index file. |

## Gate Verification Summary

All 10 verification gates passed cleanly across worker, reviewer, and integration phases:

1. **Gate 1 (GPU Embedding CTests)**: 15 tests (9 passed, 6 skipped cleanly due to missing external weights with exit code 77, 0 failed; 4.00s).
2. **Gate 2 (Runtime Neutral Contracts Test)**: 1/1 passed (0.00s).
3. **Gate 3 (Extension API Surface Gate)**: 99 paths in `SURFACE`, version 0.1.3 (revision 3).
4. **Gate 4 (Extension API Policy Unit Tests)**: 7/7 tests passed (0.002s).
5. **Gate 5 (Extension Isolation & Boundaries)**: All 6 boundary checks clean (`core`, `serving`, `model-core`, `extension`, `target-ownership`, `family-graph`).
6. **Gate 6 (Isolation Ratchet Unit Tests)**: 26/26 tests passed (22.960s).
7. **Gate 7 (Extension Metadata Tests)**: 12/12 tests passed (10.415s).
8. **Gate 8 (Extension Discovery Tests)**: 13/13 tests passed (46.906s).
9. **Gate 9 (Continuation Identity Goldens)**: 3/3 passed on GPU preset (0.01s).
10. **Gate 10 (Clang Format)**: 792 C++ files clean.
