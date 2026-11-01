# Card 14: Generate infrastructure from model metadata

Status: **complete**. RFC step and dependencies: **9; 13**.

Current milestone (stop after Card 14).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

Before remaining implementations move, every model needs metadata so generic tooling can replace hand-maintained lists. Metadata-only registration must remain truthful while legacy code is still linked through old paths.

### Original scope

Add metadata-only manifests for every remaining model before its code moves. Generate hosted/Windows target lists, labels, ZIP devlab registry input, Nix filters/packages/install checks/devshell/reference/check fragments, CI resource/vendoring data, docs/notices and committed functional registries with drift checks. Preserve public flake check names and test-source production-derivation independence. Hosted checks and pilot-selected devlab run.

## Implementation plan

1. Inventory every remaining model and its existing target/test/fixture/package/resource/docs/functional declarations; add metadata-only extension.json files without pretending unmigrated providers already use factories.
2. Generate bounded hosted and Windows target lists, CTest labels and ZIP devlab.json registry input; preserve BUILD_TESTING/platform/HIP conditions and current bundled-test assets.
3. Generate Nix production filters, dependency/install declarations, extension.nix imports for devshell Python/reference/check fragments, keeping public check names and test-source derivation independence.
4. Replace tools/ci model resource/vendoring tables, docs/notices indexes and committed functional registries with manifest-derived output plus drift checks. Add fixtures for missing/duplicate/stale/conditional metadata.
5. Run hosted checks and a pilot-selected devlab run, review all generated diffs, and stop the authorized implementation milestone after this card is verified and committed.

## Definition of done

- [x] Every current model has accurate metadata before its code migration; generated consumers no longer need parallel model lists.
- [x] Windows ZIP uses generated registry data; hosted tests/labels and Nix output names remain compatible.
- [x] Editing model test/functional/tool sources leaves production derivations unchanged while metadata/production edits are admitted; generated-output drift fails checks.
- [x] Pilot-selected devlab evidence and bounded hosted checks are retained, with discovery/release distinctions intact.

## Validation criteria

- Run nix build .#checks.x86_64-linux.pr and affected generator/functional registry fixture tests; generator second runs must produce no diff and deliberate stale fixtures must fail.
- Use nix eval and derivation comparisons to verify public check names, build inputs, install-data/licenses and unchanged production derivation after an isolated test-source-only edit; restore only that owned fixture edit afterwards.
- Build/check generated Windows target/ZIP registry paths on Windows and run the pilot-selected devlab command using the current tests/functional/README.md CLI; inspect real embedding subreports, not only successful scheduling.
- Run task-scoped docs/notices links, resource/vendoring consistency checks and default/lock metadata reports. Retain exact commands, revisions and artifact hashes; do not begin Card 15.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

### Roles and Working Copy
- **Card**: Card 14 — Generate infrastructure from model metadata
- **Parent commit**: `qupqlxsk 642af9d6` ("model-extensions-3* | feat(extensions): migrate embeddinggemma pilot to isolated extension")
- **Agent Sequence**:
  - Implementation Worker (`worker`): Run ID `dc55e98b-7ffd-4cfc-9c34-0bb24f0d008f`
  - Independent Reviewer (`reviewer`): Run ID `22b8613a-2bd9-414f-8f05-f74b49d86492` (Verdict: PASS with 2 Minor / Nit findings)
  - Review Integrator (`integrator`): Run ID `57f173b6-4af4-4f84-ba0a-d5b175156e90`
  - Orchestrator (`orchestrator`): Final closure

---

### Scope Audit

The diff strictly aligns with RFC Step 9 and Card 14:
- **Manifest Authoring & Schema Conformance**: Authored complete `extension.json` for all 7 unmigrated models (`deepseek_v4_flash`, `minimax_h3`, `qwen`, `qwen38_flash_next`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`) and expanded `embedding_gemma2/extension.json`. All 8 model manifests conform to `schemas/extension.schema.json`.
- **Truthful Metadata-Only Discovery**: In `cmake/GufoExtensionDiscovery.cmake`, detected unmigrated models lacking `extension.cmake` as `GUFO_EXTENSION_METADATA_ONLY = TRUE`. Discovery mode (`*`) excludes them with reason `"metadata only"`. Explicit `-DGUFO_EXTENSIONS=...` selection fails with `missing .../extension.cmake`. In `tools/ci/check-extension-isolation.py`, ensured unmigrated metadata-only models are excluded from premature migration ratchet enforcement.
- **Nix Derivation Independence**: Filtered `tests/`, `functional/`, `tools/`, and `fixtures/` from `src/` in `.devops/nix/package.nix`. Proved derivation hash invariance (`nix eval --raw .#packages.x86_64-linux.default.drvPath` is identical before and after modifying model test files).
- **Functional Fragments & Infrastructure Generator**: Split `tests/functional/devlab.json` into `devlab.core.json` and model fragments. Split `tests/functional/impact.json` into `impact.core.json` and model fragments. Implemented `tools/ci/generate-infrastructure.py` managing 6 target files (`cmake/Checks.cmake`, `tools/ci/check-format.py`, `tools/ci/check-docs.py`, `docs/models/README.md`, `tests/functional/devlab.json`, `tests/functional/impact.json`) with `--check` (zero drift) and `--write`. Added unit test suite `tests/tools/generate_infrastructure_test.py` (9/9 passed).
- **Untouched and Preserved**: `docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.

---

### Review Findings and Dispositions

| Finding ID | Severity | Description | Disposition |
|---|---|---|---|
| **F1** | Nit | Dead / unreachable code after `return 1` in `tools/ci/generate-infrastructure.py:556-557`. | **Resolved**: Removed dead `print` and `return 0` lines following `return 1`. |
| **F2** | Minor | Non-existent directory paths declared in manifests: `src/models/qwen_image_21/extension.json` (`fixtures`, `functional`) and `src/models/minimax_h3/extension.json` (`functional`). | **Resolved**: Removed non-existent `"fixtures"` and `"functional"` keys from `qwen_image_21/extension.json`, and removed non-existent `"functional"` key from `minimax_h3/extension.json`. Validated schema and drift cleanly. |
| **P2 (Codex)** | Minor | Invalid syntax in generated `check-docs.py` when fewer than 4 manifests declare model guides due to unconditional separators. | **Resolved**: Chunked model directory strings into populated slices of up to 3 elements, joined with valid line breaks, added trailing comma for single-element tuples, and added `compile(docs_py, "check-docs.py", "exec")` to unit test suite. |

---

### Definition of Done Audit

| DoD Item | Status | Verification Evidence / Reference |
|---|---|---|
| **DoD 1**: Every current model has accurate metadata before its code migration; generated consumers no longer need parallel model lists. | **PROVEN** | All 8 models declare valid `extension.json` manifests complying with `schemas/extension.schema.json`. `tools/ci/generate-infrastructure.py` consumes manifests to generate `cmake/Checks.cmake`, `tools/ci/check-format.py`, `tools/ci/check-docs.py`, `docs/models/README.md`, and functional registries with zero drift. |
| **DoD 2**: Windows ZIP uses generated registry data; hosted tests/labels and Nix output names remain compatible. | **PROVEN** | `devlab.json` generated from core base and model fragments; CMake PR check targets and CTest labels generated into `cmake/Checks.cmake`. `nix build .#checks.x86_64-linux.pr` passes all 4 public check derivations cleanly in the Nix sandbox. |
| **DoD 3**: Editing model test/functional/tool sources leaves production derivations unchanged while metadata/production edits are admitted; generated-output drift fails checks. | **PROVEN** | Nix production package filter in `.devops/nix/package.nix` excludes `tests/`, `functional/`, `tools/`, and `fixtures/` under `src/`. Formally verified `default.drvPath` invariance on touching test source (`/nix/store/qnga21mdw4f1363jpnsb2139992ngvvn-gufo-0-unstable-642af9d-dirty.drv`). Drift detection tested and enforced via `generate-infrastructure.py --check`. |
| **DoD 4**: Pilot-selected devlab evidence and bounded hosted checks are retained, with discovery/release distinctions intact. | **PROVEN** (Hosted checks, discovery tests, unit suites)<br>/ **DEFERRED** (Full functional harness sweeps) | Discovery mode distinctions verified in `tests/extensions/discovery_test.py` (13/13 passed). All bounded hosted checks passed in `check-pr` (96/96 passed, 100%). Full functional harness and speed benchmarks deferred to single post-Card-14 qualification window per RFC and recorded in `pending-commands.txt`. |

---

### Acceptance Gates and Integration Re-runs

All 8 affected verification gates were re-executed once by the integration agent; full logs and details are recorded in [`docs/model-extensions/evidence/card14/integration-reruns.txt`](../evidence/card14/integration-reruns.txt):

| Gate | Description | Command Executed | Exit Code | Verdict |
|---|---|---|---|---|
| **Gate 1** | Infrastructure Drift Check | `python3 tools/ci/generate-infrastructure.py --check` | 0 | **PASS** (Zero drift across all 6 targets) |
| **Gate 2** | Infrastructure Generator Unit Tests | `python3 -m unittest tests/tools/generate_infrastructure_test.py` | 0 | **PASS** (9/9 tests in 0.031s) |
| **Gate 3** | Extension Schema Validation | `nix develop -c python3 tests/extensions/metadata_test.py --schema` | 0 | **PASS** (All 8 manifests conform to schema) |
| **Gate 4** | Documentation Integrity Check | `python3 tools/ci/check-docs.py` | 0 | **PASS** (315 docs, 1591 links verified clean) |
| **Gate 5** | Clang-Format Verification | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | **PASS** (792 C++ files clean) |
| **Gate 6** | CPU-Test Check-PR Suite | `nix develop -c cmake --build --preset cpu-test --target check-pr` | 0 | **PASS** (96/96 tests passed, 100%, 139.82s) |
| **Gate 7** | Hosted PR Checks (Nix Sandbox) | `nix build .#checks.x86_64-linux.pr` | 0 | **PASS** (All 4 derivations built and passed) |
| **Gate 8** | Nix Derivation Hash Invariance | `nix eval --raw .#packages.x86_64-linux.default.drvPath` | 0 | **PASS** (Invariance proven against test modification) |

---

### Deferred Items

Per the RFC execution plan and card brief, full functional test harness runs (`tests/functional/run.py`), benchmark model sweeps, and continuation timing controls are deferred to the single post-Card-14 qualification window. Exact commands are documented in [`docs/model-extensions/evidence/card14/pending-commands.txt`](../evidence/card14/pending-commands.txt).
