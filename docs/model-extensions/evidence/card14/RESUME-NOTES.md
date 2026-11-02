# Card 14 Resume Notes: Generate Infrastructure from Model Metadata

## Summary of Accomplished Work

Card 14 implements the infrastructure generator and manifest completeness step of the model extensions RFC:

1. **Manifest Authoring & Schema Conformance**:
   - Authored complete, valid `extension.json` manifests for all 7 unmigrated models (`deepseek_v4_flash`, `minimax_h3`, `qwen`, `qwen38_flash_next`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`) and expanded `embedding_gemma2/extension.json`.
   - Every manifest defines `schema: 1`, `name`, `version`, `provider`, `packages`, `runtime_libraries`, `production_paths`, `docs`, `tests`, `fixtures`, `bundled_tests`, and `functional` configuration.
   - All 8 manifests pass strict validation against `schemas/extension.schema.json` via `nix develop -c python3 tests/extensions/metadata_test.py --schema`.

2. **Truthful Metadata-Only Discovery**:
   - Updated `cmake/GufoExtensionDiscovery.cmake` to detect unmigrated models lacking `extension.cmake`, mark them `GUFO_EXTENSION_METADATA_ONLY = TRUE`, and cleanly omit them from `gufo_selected_extensions` in discovery mode (`*`) with exclusion reason `"metadata only"`.
   - Explicit selection of metadata-only extensions (e.g. `-DGUFO_EXTENSIONS=deepseek_v4_flash`) fails with `missing .../extension.cmake`.
   - Updated and verified `tests/extensions/discovery_test.py` (13/13 tests OK).
   - In `tools/ci/check-extension-isolation.py`, ensured unmigrated extensions lacking `extension.cmake` are not treated as migrated extensions during boundary checks, maintaining isolation ratchet invariants (27/27 tests OK).

3. **Production Derivation Independence**:
   - Filtered `tests/`, `functional/`, `tools/`, and `fixtures/` under `src/` in `.devops/nix/package.nix`.
   - Formally proved derivation hash invariance: `nix eval --raw .#packages.x86_64-linux.default.drvPath` is identical before and after modifying a model test file (`/nix/store/qnga21mdw4f1363jpnsb2139992ngvvn-gufo-0-unstable-642af9d-dirty.drv`).

4. **Functional Registry Fragments & Merged Registries**:
   - Split `tests/functional/devlab.json` into core base `tests/functional/devlab.core.json` and 7 per-extension fragments (`src/models/*/functional/devlab.json`).
   - Split `tests/functional/impact.json` into core base `tests/functional/impact.core.json` and 3 per-extension fragments (`src/models/*/functional/impact.json`).
   - Confirmed functional unit test suite passes: `PYTHONPATH=tests/functional python3 -m unittest tests/functional/test_impact.py tests/functional/test_devlab.py` (31/31 tests OK).

5. **Infrastructure Generator & Drift Checker Tool**:
   - Implemented `tools/ci/generate-infrastructure.py` managing 6 target files:
     * `cmake/Checks.cmake`: model PR targets and CTest overrides
     * `tools/ci/check-format.py`: vendored path exclusions
     * `tools/ci/check-docs.py`: required model documentation paths
     * `docs/models/README.md`: model directory table
     * `tests/functional/devlab.json`: merged DevLab test suite
     * `tests/functional/impact.json`: merged functional impact rules
   - Supports `--check` (default, exits 0 on sync, 1 on drift with unified diffs) and `--write`.
   - Drift check verified: `python3 tools/ci/generate-infrastructure.py --check` exits 0 with 0 drift.

6. **Unit Tests for Infrastructure Generator**:
   - Created `tests/tools/generate_infrastructure_test.py` covering:
     * Missing metadata (`name` missing, syntax errors)
     * Duplicate metadata (duplicate name, duplicate provider ID)
     * Drift detection (detecting modified content and exit status 1)
     * Conditional metadata (omission of optional sections like docs or tests)
     * Live repo sync
   - Registered `generate_infrastructure_test` in `CMakeLists.txt` and `cmake/Checks.cmake`.
   - Added `docs/models/README.md` to `flake.nix` `testSource.files` so all tests run cleanly in the Nix sandbox.

7. **Review Findings Dispositions**:
   - **F1 (Nit)**: Removed unreachable dead code after `return 1` in `tools/ci/generate-infrastructure.py:556-557`.
   - **F2 (Minor)**: Removed non-existent `"fixtures"` and `"functional"` directory paths from `src/models/qwen_image_21/extension.json`, and removed non-existent `"functional"` directory path from `src/models/minimax_h3/extension.json`.
   - Verified that `generate-infrastructure.py --check` passes with zero drift and `metadata_test.py --schema` passes with all 8 manifests valid. Fixed fenced json code block syntax in `review-findings.md` so `check-docs.py` passes 100%.

8. **Integration Verification Re-runs**:
   All 8 affected verification gates were re-executed once and recorded in `docs/model-extensions/evidence/card14/integration-reruns.txt`:
   - Gate 1: `python3 tools/ci/generate-infrastructure.py --check` (exit 0, PASS)
   - Gate 2: `python3 -m unittest tests/tools/generate_infrastructure_test.py` (exit 0, 9/9 PASS)
   - Gate 3: `nix develop -c python3 tests/extensions/metadata_test.py --schema` (exit 0, PASS)
   - Gate 4: `python3 tools/ci/check-docs.py` (exit 0, 315 docs / 1591 links PASS)
   - Gate 5: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (exit 0, 792 C++ files PASS)
   - Gate 6: `nix develop -c cmake --build --preset cpu-test --target check-pr` (exit 0, 96/96 100% PASS, 139.82s)
   - Gate 7: `nix build .#checks.x86_64-linux.pr` (exit 0, all 4 check derivations built and passed)
   - Gate 8: `nix eval --raw .#packages.x86_64-linux.default.drvPath` (exit 0, derivation hash invariance verified)

## Agent Sequence
- **Worker**: Agent `dc55e98b-7ffd-4cfc-9c34-0bb24f0d008f`
- **Reviewer**: Agent `22b8613a-2bd9-414f-8f05-f74b49d86492` (PASS with 2 Minor / Nit findings)
- **Integrator**: Agent `57f173b6-4af4-4f84-ba0a-d5b175156e90`
- **Orchestrator**: Final closure

## Handoff of Editing Ownership

The integration phase is complete. Card 14 is set to `integrated, ready for orchestrator closure`.
This concludes the authorized implementation milestone (Cards 01–14).
Full functional test harness runs and speed benchmark sweeps remain deferred to the consolidated post-Card-14 qualification window as recorded in `pending-commands.txt`.
No Git or Jujutsu commits have been made (`git` / `jj` working copy remains uncommitted).
`docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.
Editing ownership is released and returned to the parent caller / orchestrator (`96a2ae41-29cf-4272-bacb-3b48823c7bba`).
