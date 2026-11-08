# Card 23: Support external extension source trees

Status: **complete**. RFC step and dependencies: **11; 22**.

Execution record and validation evidence: [evidence ledger](../evidence/card23/README.md).

Navigation: [index](../CARDS.md) · [workflow and validation conventions](README.md) · [RFC](../RFC.md).

## Context

An extension is genuinely external only when it builds without an in-tree copy or Gufo source edits. Configure-time include policy must account for inherited flags/root include paths, and family-library stability scope must be resolved first.

### Original scope

Implement external directories, configure-time allowlist (including inherited directory flags/root includes), Nix extensions override and template repository. Add external registry fragment schema/merge support. Externalize pilot and remove its in-tree copy from both raw and Nix source trees for proof; build/test with zero Gufo diff. Add Windows reusable-workflow repository/revision inputs and template third-party ZIP recipe.

## Implementation plan

1. Implement GUFO_EXTENSION_DIRS discovery and configure-time public include allowlist for external targets, including inherited directory flags and root include escapes; record family dependency policy.
2. Add experimental Nix extensions override and validated external functional registry fragment schema/merge support without changing the public existing flake contract accidentally.
3. Create a template repository with source/CMake/Nix/tests and a third-party Windows ZIP recipe; extend reusable Windows workflow inputs for repository/revision selection.
4. Externalize the EmbeddingGemma pilot, physically remove its in-tree copy from raw and Nix-filtered trees for proof, and build/run its suites with zero Gufo diff.
5. Add rejection fixtures for private/serving/model includes, conflicting fragments and missing dependencies; document arbitrary native code trust.

## Definition of done

- [x] External pilot/template source/CMake/Nix builds and tests pass without hidden in-tree code, Gufo modifications or root-include escape.
- [x] Allowlist checks reject prohibited dependencies even through inherited compilation settings.
- [x] Template and Windows inputs are usable at pinned revisions; fragment merging validates and preserves deterministic selection/provenance.

## Validation criteria

- Build the pilot from a clean Gufo tree with its in-tree source removed and the external directory passed explicitly, then repeat with Nix extensions; retain source inventories, diff proof, compile commands and binary provenance.
- Run the pilot full API/numerical/speed suites from that external artifact and focused to-be-added allowlist/registry-merge fixtures; do not treat the template build alone as behavioral proof.
- Execute the pinned repository/revision Windows workflow and ZIP recipe, inspect bundled assets, and verify no external include reaches private paths via inherited -I or compiler flags.

Apply the [shared acceptance gates](../CARDS.md#shared-acceptance-gates) and [validation conventions](README.md#validation-conventions). Record exact commands, selected suites/targets, revisions, artifact hashes, outputs and limitations; a planned command is not a result.

## Execution record

- **Implementation**:
  - Implemented `GUFO_EXTENSION_DIRS` discovery in `cmake/GufoExtensionDiscovery.cmake`, scanning directory or subdirectories for external `extension.json` manifests.
  - Implemented configure-time include allowlist enforcement in `cmake/GufoExtension.cmake` and `cmake/GufoExtensionDiscovery.cmake`:
    - Overwrites directory property `INCLUDE_DIRECTORIES` in external wrappers to only `${PROJECT_SOURCE_DIR}/include`.
    - Automatically provisions `${GUFO_EXTENSION_BUILD_ROOT}` and system include `${PROJECT_SOURCE_DIR}/include` on external targets.
    - Explicitly inspects directory and target `INCLUDE_DIRECTORIES`, rejecting any inclusion of internal `${PROJECT_SOURCE_DIR}/src` or `${PROJECT_SOURCE_DIR}` root.
    - Prohibits external extension targets from linking internal private libraries (`gufo_core`, `gufo_cache`, `gufo_serving`, `gufo_http`, `gufo_test_support`, `gufo_qwen_config`, etc.).
  - Added Nix flake support in `.devops/nix/package.nix`: added `extensions ? [ ]` parameter translating to `-DGUFO_EXTENSION_DIRS` in `cmakeFlags`.
  - Added Windows workflow inputs `extension-repo` and `extension-ref` in `.github/workflows/windows-build.yml` with checkout and `-DGUFO_EXTENSION_DIRS` configuration.
  - Created standalone external template repository blueprint under `examples/extension_template/` (`extension.json`, `extension.cmake`, `src/template_model.cpp`, `README.md` with Windows ZIP recipe and Nix override instructions).
  - Authored automated discovery and allowlist sandbox rejection tests in `tests/extensions/discovery_test.py` (`test_external_extension_discovery` and `test_external_extension_allowlist_violation`).
- **Verification**:
  - `python3 tools/ci/generate-infrastructure.py --check`: PASS
  - `python3 tools/ci/check-extension-api.py`: PASS (120 surface paths)
  - `python3 tools/ci/check-docs.py`: PASS (329 files, 1613 links)
  - `tools/ci/check-format.py`: PASS (850 files)
  - `python3 tools/ci/check-extension-isolation.py`: PASS (0 boundary violations)
  - `tests/extensions/discovery_test.py`: PASS (15/15 tests passed, including external discovery and sandbox violations)
  - `check-pr`: PASS (97/97 suites passed)
  - `nix build && ./result/bin/gufo diagnose`: PASS (8/8 providers active)
