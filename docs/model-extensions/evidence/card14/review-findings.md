# Card 14 Independent Review Findings: Generate Infrastructure from Model Metadata

Reviewer: Independent Read-Only Review Agent  
Review Date: 2026-11-01  
Repository: `/home/mixer/gufo`  
Parent Revision: `qupqlxsk 642af9d6`  
Working Copy: `mpqrslmo de72a207` (uncommitted, read-only audit)  
Audit Verdict: **PASS with 2 Minor / Nit findings**

---

## 1. Executive Summary

Card 14 delivers the model metadata manifests and automated infrastructure generator for the Gufo model extension subsystem. All 8 models (`deepseek_v4_flash`, `embedding_gemma2`, `minimax_h3`, `qwen`, `qwen38_flash_next`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`) now have metadata manifests adhering to `schemas/extension.schema.json`. Unmigrated models are truthfully represented without `extension.cmake`, and `cmake/GufoExtensionDiscovery.cmake` correctly treats them as `GUFO_EXTENSION_METADATA_ONLY = TRUE`, cleanly excluding them from discovery builds (`*`) with reason `"metadata only"` while failing explicit selection. Production derivation hash invariance is verified under Nix: editing tests, functional fragments, fixtures, or tools under model directories does not invalidate the production package derivation. The infrastructure generator `tools/ci/generate-infrastructure.py` accurately generates and drift-checks 6 central configuration files (`cmake/Checks.cmake`, `tools/ci/check-format.py`, `tools/ci/check-docs.py`, `docs/models/README.md`, `tests/functional/devlab.json`, `tests/functional/impact.json`). All 13 local and hosted gates pass cleanly.

---

## 2. Findings

### Finding F1 (Nit): Dead / Unreachable Code in Infrastructure Generator CLI

- **Severity**: Nit
- **Paths & Line References**: [`tools/ci/generate-infrastructure.py:556-557`](file:///home/mixer/gufo/tools/ci/generate-infrastructure.py#L556-L557)
- **What is wrong**:
  At the end of `main()` in `tools/ci/generate-infrastructure.py`, after lines 553–554 print the drift error and return exit code 1 (`return 1`), lines 556–557 contain duplicate unreachable statements:
  ```python
  553:    print("\nFAIL: Generated infrastructure has drifted from model metadata. Run with --write to update.", file=sys.stderr)
  554:    return 1
  555:
  556:    print("PASS: Infrastructure files are up to date with model metadata.")
  557:    return 0
  ```
- **Practical impact**:
  Harmless at runtime because `return 1` terminates execution before these lines are reached. However, this is dead code left over from refactoring `main()`.
- **Suggested correction**:
  Delete lines 556–557 from [`tools/ci/generate-infrastructure.py`](file:///home/mixer/gufo/tools/ci/generate-infrastructure.py).
- **How to verify**:
  Inspect [`tools/ci/generate-infrastructure.py#L550-L560`](file:///home/mixer/gufo/tools/ci/generate-infrastructure.py#L550-L560) and confirm no unreachable statements exist after `return 1`.

---

### Finding F2 (Minor): Non-Existent Directory Paths Declared in Manifests

- **Severity**: Minor
- **Paths & Line References**:
  - [`src/models/qwen_image_21/extension.json:70-71`](file:///home/mixer/gufo/src/models/qwen_image_21/extension.json#L70-L71):
    ```json
    {
      "fixtures": "src/models/qwen_image_21/reference/",
      "functional": "functional/"
    }
    ```
  - [`src/models/minimax_h3/extension.json:105`](file:///home/mixer/gufo/src/models/minimax_h3/extension.json#L105):
    ```json
    {
      "functional": "functional/"
    }
    ```
- **What is wrong**:
  - Neither `src/models/qwen_image_21/reference/` nor `src/models/qwen_image_21/functional/` exists on disk.
  - `src/models/minimax_h3/functional/` does not exist on disk.
  - Both `qwen_image_21` and `minimax_h3` are explicitly defined as `outside-current-scope` for functional DevLab tests in `tests/functional/devlab.core.json`.
- **Practical impact**:
  `schemas/extension.schema.json` only requires `fixtures` and `functional` to be non-empty strings (`minLength: 1`) without checking filesystem existence; `generate-infrastructure.py` guards reading functional fragments with `.is_file()`. Thus, tests and builds currently pass. However, declaring non-existent directories in model metadata compromises strict metadata truthfulness.
- **Suggested correction**:
  Since `fixtures` and `functional` are optional fields in `extension.schema.json`:
  1. Remove `"fixtures"` and `"functional"` from [`src/models/qwen_image_21/extension.json`](file:///home/mixer/gufo/src/models/qwen_image_21/extension.json).
  2. Remove `"functional"` from [`src/models/minimax_h3/extension.json`](file:///home/mixer/gufo/src/models/minimax_h3/extension.json).
  (Or alternatively, create the respective directory structures if test fixtures are intended to be placed there prior to code migration).
- **How to verify**:
  ```sh
  python3 -c 'from pathlib import Path; assert not (Path("src/models/qwen_image_21/reference").exists())'
  ```

---

## 3. Explicit "No Finding" Clean Statements

The following audited areas were thoroughly inspected and found fully compliant:

1. **`RFC.md` Invariance**:
   - `docs/model-extensions/RFC.md` remains strictly byte-identical and untouched (`jj diff --name-only | grep RFC` produced no output).
2. **Schema Conformance**:
   - All 8 manifests (`deepseek_v4_flash`, `embedding_gemma2`, `minimax_h3`, `qwen`, `qwen38_flash_next`, `qwen3_asr`, `qwen3_tts`, `qwen_image_21`) strictly conform to `schemas/extension.schema.json` (`tests/extensions/metadata_test.py --schema` passed).
3. **Truthful CMake Discovery**:
   - `cmake/GufoExtensionDiscovery.cmake` properly checks `EXISTS "${root}/extension.cmake"`. Unmigrated models are assigned `GUFO_EXTENSION_METADATA_ONLY = TRUE`, excluded in discovery mode (`*`) with message `"excluded (metadata only)"`, and trigger `FATAL_ERROR "missing .../extension.cmake"` if explicitly requested. Verified in `tests/extensions/discovery_test.py` (13/13 passed).
4. **Extension Isolation Boundary Scanner**:
   - `tools/ci/check-extension-isolation.py` restricts migrated status strictly to models with existing `extension.cmake`, preventing unmigrated metadata-only models from premature migration ratchet enforcement. Verified in `tests/tools/check_extension_isolation_test.py` (27/27 passed).
5. **Nix Production Derivation Independence**:
   - `.devops/nix/package.nix` filters out `/tests/`, `/functional/`, `/tools/`, and `/fixtures/` under `src/`.
   - Verified that `nix eval --raw .#packages.x86_64-linux.default.drvPath` outputs `/nix/store/4nqkz6iivwjxjdx6kgw9jd1mw0y3bjf9-gufo-0-unstable-642af9d-dirty.drv` identically before and after modifying model test files.
   - Evaluated filter expressions in pure Nix: test/fixture/tool/functional paths evaluate to `false` (excluded) while production sources and `extension.json` evaluate to `true` (included).
6. **Functional Registries and Generator Conformance**:
   - `tests/functional/devlab.json` and `tests/functional/impact.json` are cleanly split into core bases (`devlab.core.json`, `impact.core.json`) and per-model fragments (`src/models/*/functional/`).
   - `python3 -m unittest tests/functional/test_impact.py tests/functional/test_devlab.py` passes 31/31 tests.
   - `python3 tools/ci/generate-infrastructure.py --check` reports zero drift across all 6 targets.
   - Conformance test suite `tests/tools/generate_infrastructure_test.py` passes 9/9 tests covering missing/duplicate metadata, drift detection, and conditional metadata omissions.
7. **Hosted & Local Verification Suites**:
   - `nix develop -c cmake --build --preset cpu-test --target check-pr`: 96/96 (100%) tests passed in 140.96s.
   - `nix build .#checks.x86_64-linux.pr`: All 4 hosted derivations (`check-dependency-inventory`, `check-docs`, `check-pr-tests`, `check-pr`) built and passed cleanly.
   - Continuation identity tests passed 3/3 (`qwen27b`, `deepseek_v4_flash`, `qwen38_flash_next`).
   - Clang-format CI check: 792 C++ files passed cleanly.
   - Documentation check: 313 markdown files and 1591 links verified with zero broken links.
8. **Repository Conventions & Safety**:
   - No `<windows.h>` or `_WIN32` checks introduced outside `src/core/platform/`.
   - No file under `docs/` exceeds 1 MB.
   - No uncommitted commits created (`jj` working copy remains clean of commits).
   - All reviewer scratch files kept in `/tmp/card14-review/`.
   - No GPU memory leaks or service disruptions (`free -g` confirmed 111 GiB available).

---

## 4. Definition of Done (DoD) Audit Table

| DoD Item | Status | Verification Evidence / Details |
|---|---|---|
| **1. Every current model has accurate metadata before its code migration; generated consumers no longer need parallel model lists.** | **Proven** (with minor F2 note) | All 8 models have valid manifests in `src/models/*/extension.json`. Manifests validate 100% against schema. Central files (`Checks.cmake`, `check-format.py`, `check-docs.py`, `docs/models/README.md`, `devlab.json`, `impact.json`) are generated from manifests without manual parallel lists. |
| **2. Windows ZIP uses generated registry data; hosted tests/labels and Nix output names remain compatible.** | **Proven** | Registry files (`devlab.json`, `impact.json`) generated deterministically. Hosted PR test targets in `cmake/Checks.cmake` match ctest overrides and run 96/96 tests. Public Nix check and output names are preserved in `flake.nix` and pass all 4 checks in `nix build .#checks.x86_64-linux.pr`. |
| **3. Editing model test/functional/tool sources leaves production derivations unchanged while metadata/production edits are admitted; generated-output drift fails checks.** | **Proven** | Nix derivation path `/nix/store/4nqkz6iivwjxjdx6kgw9jd1mw0y3bjf9-gufo-0-unstable-642af9d-dirty.drv` is identical before and after touching test sources. Filter expression verified in Nix. Drift detection verified in `generate_infrastructure_test.py` and live repo check exits 0. |
| **4. Pilot-selected devlab evidence and bounded hosted checks are retained, with discovery/release distinctions intact.** | **Proven / Deferred per policy** | Bounded hosted checks pass (`check-pr` 96/96, `checks.pr` 4/4). Discovery mode distinguishes metadata-only models from migrated pilot (`embedding_gemma2`). Functional harness runs deferred per project policy (2026-10-31) and documented in `pending-commands.txt`. |

---

## 5. Execution Record: Commands Executed by Independent Reviewer

| # | Command | Exit Code | Result Summary |
|---|---|---|---|
| 1 | `jj status` | 0 | Working copy clean of commits; expected tracked changes present |
| 2 | `jj diff --name-only \| grep RFC` | 0 | Confirmed `docs/model-extensions/RFC.md` is strictly untouched |
| 3 | `python3 tools/ci/generate-infrastructure.py --check` | 0 | PASS: Zero drift across all 6 managed infrastructure files |
| 4 | `nix develop -c python3 tests/extensions/metadata_test.py --schema` | 0 | PASS: All 8 extension manifests valid against schema |
| 5 | `nix develop -c python3 tests/extensions/metadata_test.py` | 0 | PASS: Ran 12 tests in 8.757s (OK) |
| 6 | `python3 -m unittest tests/tools/generate_infrastructure_test.py` | 0 | PASS: Ran 9 tests in 0.032s (OK) |
| 7 | `python3 tests/tools/check_extension_isolation_test.py` | 0 | PASS: Ran 27 tests in 23.144s (OK) |
| 8 | `PYTHONPATH=tests/functional python3 -m unittest tests/functional/test_impact.py tests/functional/test_devlab.py` | 0 | PASS: Ran 31 tests in 0.207s (OK) |
| 9 | `nix develop -c python3 tests/extensions/discovery_test.py` | 0 | PASS: Ran 13 tests in 52.958s (OK) |
| 10 | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | PASS: Checked 792 C++ files |
| 11 | `python3 tools/ci/check-docs.py` | 0 | PASS: Checked 313 markdown files, 1591 links |
| 12 | `nix develop -c ctest --test-dir build/cpu-test -R 'continuation_identity' --output-on-failure` | 0 | PASS: 3/3 tests passed (qwen27b, ds4, qwen38) |
| 13 | `nix eval --raw .#packages.x86_64-linux.default.drvPath` | 0 | Hash verified: `/nix/store/4nqkz6iivwjxjdx6kgw9jd1mw0y3bjf9-gufo-0-unstable-642af9d-dirty.drv` |
| 14 | `nix eval --impure --expr '<filter-expression>'` | 0 | Proved test/fixture/functional paths evaluate to `false` and production/manifest to `true` |
| 15 | `nix build .#checks.x86_64-linux.pr --no-link` | 0 | PASS: All 4 hosted check derivations built and passed |
| 16 | `nix develop -c cmake --build --preset cpu-test --target check-pr` | 0 | PASS: 96/96 (100%) tests passed in 140.96s |
| 17 | `free -g` | 0 | System memory healthy: 111 GiB available |
| 18 | `find docs -type f -size +1M` | 0 | Verified zero documentation files > 1 MB |
