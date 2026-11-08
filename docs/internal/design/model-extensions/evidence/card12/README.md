# Card 12 Evidence Index and Provenance

This directory contains the durable evidence artifacts for **Card 12: Add in-tree model API test support** (RFC Step 7b).

## Provenance and Agent Sequence

- **Parent Commit**: `ytumsxmz 7752857d` ("refactor(extensions): migrate models to public extension headers")
- **Implementation Worker**: Agent `02a253d3-11d5-42ec-ae8e-d89e20ef0716`
- **Independent Reviewer**: Agent `397daa87-23c0-4244-acce-79507e4def2a` (Verdict: CLEAN PASS, F0)
- **Review Integrator**: Agent `fde70554-5ab1-4a6f-9e2f-b5012f550e0c`
- **Orchestrator**: Final closure

## Evidence Files

| File | Author / Role | Description |
|---|---|---|
| [`input-brief-worker.md`](input-brief-worker.md) | Orchestrator | Input task brief provided to the implementation worker agent. |
| [`input-brief-reviewer.md`](input-brief-reviewer.md) | Orchestrator | Input task brief provided to the independent review agent. |
| [`input-brief-integration.md`](input-brief-integration.md) | Orchestrator | Input task brief provided to the integration agent. |
| [`model-api-test-inventory.md`](model-api-test-inventory.md) | Worker | Comprehensive inventory of model API test includes and compiled sources before and after migration, documenting remaining text runner coupling. |
| [`test-support-architecture.md`](test-support-architecture.md) | Worker | Architecture specification for `gufo_test_support` STATIC library and header layout. |
| [`build-and-tests.txt`](build-and-tests.txt) | Worker | Initial execution logs for all 10 validation gates during implementation. |
| [`review-findings.md`](review-findings.md) | Reviewer | Independent read-only audit report recording CLEAN PASS with zero findings (F0) across all validation areas. |
| [`integration-reruns.txt`](integration-reruns.txt) | Integrator | Exact commands, exit codes, and verdicts for the single integration re-run of all 10 validation gates. |
| [`orchestrator-gates.txt`](orchestrator-gates.txt) | Orchestrator | Closure re-runs of the pinned gate set, Codex review output, and final verification transcript. |
| [`pending-commands.txt`](pending-commands.txt) | Worker / Integrator | Verbatim commands for functional harness and speed benchmarks deferred to the post-Card-14 qualification window. |
| [`RESUME-NOTES.md`](RESUME-NOTES.md) | Worker / Integrator | Handoff state notes and release of editing ownership. |
| [`README.md`](README.md) | Integrator | This index file. |

## Gate Verification Summary

All 10 verification gates passed cleanly across worker, reviewer, and integration phases:

1. **Gate 1 (CPU Model API CTests)**: 4/4 passed (0.19s).
2. **Gate 2 (GPU Model API CTests)**: 4/4 passed (0.42s).
3. **Gate 3 (Extension API Surface)**: 97 paths in `SURFACE`, zero modifications or leaks.
4. **Gate 4 (Extension Isolation & Boundaries)**: All 5 boundary checks clean (`core`, `serving`, `model-core`, `target-ownership`, `family-graph`).
5. **Gate 5 (Isolation Unit Tests)**: 24/24 unit tests passed.
6. **Gate 6 (Extension API Policy Unit Tests)**: 7/7 unit tests passed.
7. **Gate 7 (Documentation Policy Unit Tests)**: 11/11 unit tests passed.
8. **Gate 8 (Continuation Identity Goldens)**: 3/3 passed on CPU and GPU.
9. **Gate 9 (Clang Format)**: 787 C++ files formatted.
10. **Gate 10 (Production Exclusion Proof)**: Zero matches for `test_support` under `BUILD_TESTING=OFF`.
