# Card 14 Evidence Index and Provenance

This directory contains the durable evidence artifacts for **Card 14: Generate infrastructure from model metadata** (RFC Step 8 infrastructure generator and manifest completeness).

## Provenance and Agent Sequence

- **Parent Commit**: `qupqlxsk 642af9d6` ("model-extensions-3* | feat(extensions): migrate embeddinggemma pilot to isolated extension")
- **Implementation Worker**: Agent `dc55e98b-7ffd-4cfc-9c34-0bb24f0d008f`
- **Independent Reviewer**: Agent `22b8613a-2bd9-414f-8f05-f74b49d86492` (Verdict: PASS with 2 Minor / Nit findings)
- **Review Integrator**: Agent `57f173b6-4af4-4f84-ba0a-d5b175156e90`
- **Orchestrator**: Final closure

## Evidence Files

| File | Author / Role | Description |
|---|---|---|
| [`input-brief-worker.md`](input-brief-worker.md) | Orchestrator | Input task brief provided to the implementation worker agent. |
| [`input-brief-reviewer.md`](input-brief-reviewer.md) | Orchestrator | Input task brief provided to the independent review agent. |
| [`input-brief-integration.md`](input-brief-integration.md) | Orchestrator | Input task brief provided to the integration agent. |
| [`build-and-tests.txt`](build-and-tests.txt) | Worker | Initial execution logs for build, diagnostics, and test suites during implementation. |
| [`pending-commands.txt`](pending-commands.txt) | Worker / Integrator | Verbatim commands for functional harness and speed benchmarks deferred to the single post-Card-14 qualification window. |
| [`review-findings.md`](review-findings.md) | Reviewer | Independent read-only audit report recording PASS with 2 Minor/Nit findings (F1 dead code, F2 non-existent paths). |
| [`integration-reruns.txt`](integration-reruns.txt) | Integrator | Exact commands, exit codes, and verdicts for the single integration re-run of all 8 affected verification gates. |
| [`orchestrator-gates.txt`](orchestrator-gates.txt) | Orchestrator | Pinned gate set re-runs, Codex review finding P2 resolution, and final closure verification. |
| [`RESUME-NOTES.md`](RESUME-NOTES.md) | Worker / Integrator | Handoff state notes, review dispositions, and release of editing ownership. |
| [`README.md`](README.md) | Integrator | This index file. |

## Gate Verification Summary

All 8 affected verification gates passed cleanly across worker, reviewer, and integration phases:

1. **Gate 1 (Infrastructure Drift Check)**: `python3 tools/ci/generate-infrastructure.py --check` passed (exit 0, zero drift across all 6 managed configuration files).
2. **Gate 2 (Infrastructure Generator Unit Tests)**: `python3 -m unittest tests/tools/generate_infrastructure_test.py` passed (9/9 tests in 0.031s).
3. **Gate 3 (Extension Schema Validation)**: `nix develop -c python3 tests/extensions/metadata_test.py --schema` passed (all 8 model manifests conform to `schemas/extension.schema.json`).
4. **Gate 4 (Documentation Integrity Check)**: `python3 tools/ci/check-docs.py` passed (315 markdown files, 1591 links verified with zero broken links or invalid JSON blocks).
5. **Gate 5 (Clang-Format Code Formatting Check)**: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` passed (792 C++ files clean).
6. **Gate 6 (CPU-Test Check-PR Suite)**: `nix develop -c cmake --build --preset cpu-test --target check-pr` passed (96/96 tests passed, 100%, 139.82s).
7. **Gate 7 (Hosted PR Checks Suite - Nix Sandbox)**: `nix build .#checks.x86_64-linux.pr` passed (all 4 derivations `check-dependency-inventory`, `check-docs`, `check-format`, `check-pr-tests` built and passed).
8. **Gate 8 (Nix Derivation Hash Invariance)**: Formally proved that modifying model tests, fixtures, or functional fragments does not alter `nix eval --raw .#packages.x86_64-linux.default.drvPath` (`/nix/store/qnga21mdw4f1363jpnsb2139992ngvvn-gufo-0-unstable-642af9d-dirty.drv`).
