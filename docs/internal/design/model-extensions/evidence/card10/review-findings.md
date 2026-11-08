# Card 10 Independent Review Findings

- **Card**: Card 10 — Publish and measure the public extension API
- **Reviewer Role**: Independent Reviewer (`reviewer`)
- **Working Copy**: `@` (`d14eb504...` over parent `klwmvoxr a761679a`)
- **Review Date**: 2026-10-31
- **Verdict**: **PASS** (Zero findings / clean conformance)

---

## Findings

No findings (0 defects identified).

All evaluated areas conform strictly to Card 10 specifications, RFC requirements, repository conventions, and acceptance gates.

---

## Explicit "No Finding" Statements

1. **Scope and Cardinality (No Finding)**:
   The diff implements Card 10 exactly without feature creep or premature implementation. Core headers used by model code are classified into stable (`include/gufo/ext/*.hpp`), platform (`include/gufo/ext/platform/*.hpp`), experimental (`include/gufo/ext/experimental/*.hpp`), host wrapper (`include/gufo/ext/host.hpp`), or private with approved removal plans (2 test-only ASR headers to be removed in Card 12). Exactly 44 forwarding headers were created. Zero model files under `src/models/` or `src/families/` were modified to consume public headers (properly deferred to Card 11).

2. **Forwarding Header Integrity (No Finding)**:
   Every newly added forwarding header strictly adheres to `#pragma once` followed by the direct `#include "src/..."` directive. No auxiliary macros, inline code, or serving headers are included. All forwarded targets exist on disk.

3. **Serving Boundary and Isolation (No Finding)**:
   Zero serving headers (`src/cli/**` or HTTP serving policies) leak into `include/gufo/ext/` or `include/gufo/ext/SURFACE`. `tools/ci/check-extension-isolation.py` passed all four isolation checks (core boundary, serving boundary, target ownership, and acyclic family graph).

4. **Extension API Governance & Versioning (No Finding)**:
   The extension API version is cleanly bumped from `0.1.1` to `0.1.2` (`kApiRevision = 2`, `kApiVersion = "0.1.2"`) in `include/gufo/ext/version.hpp` and `cmake/GufoExtension.cmake` (`GUFO_EXTENSION_API_VERSION "0.1.2"`). `include/gufo/ext/CHANGES.md` documents `## 0.1.2` with an itemized breakdown of stable, platform, experimental headers, and transitive surface governance. Diff gate `check-extension-api.py` enforces the `ext-api` PR label and rejects changes when the label is absent.

5. **Transitive SURFACE Closure & Reproducibility (No Finding)**:
   `include/gufo/ext/SURFACE` records 97 paths (49 public headers, 48 forwarded `src/` implementation headers). It is sorted, contains no duplicates, and was verified to match the programmatic closure of `tools/ci/check-extension-api.py` with 0 difference. Running `python3 tools/ci/check-extension-api.py --write` is idempotent and produces zero diff.

6. **Unit Policy Tests (No Finding)**:
   `tests/tools/check_extension_api_test.py` was extended to test transitive additions, transitive deletions, subdirectory declaration modifications across `platform/` and `experimental/`, and closure resolution. All 7 unit tests pass.

7. **HIP-less and HIP Consumer Compilation (No Finding)**:
   Representative HIP-less consumers (`ext_runtime_neutral_contracts_test.cpp`, `fixture/cpu.cpp`) and HIP consumers (`fixture/hip.cpp`, `libalpha_hip.a`) compiled cleanly without errors or warnings. CTest suites in both `build/cpu-test` (7/7 passed) and `build/gpu-test` (7/7 passed) succeeded.

8. **Behavior Preservation & Continuation Identity (No Finding)**:
   Existing runtime contracts and continuation behavior remain identical. CPU and GPU continuation identity tests (`qwen27b`, `qwen38_flash_next`, `deepseek_v4_flash`) passed 3/3 in both configurations, and `sha256sum -c docs/model-extensions/evidence/card02/SHA256SUMS` matched all 42 fixture checksums (100%).

9. **Repository Conventions & Portability (No Finding)**:
   `RFC.md` remains byte-identical and uncommitted. No `<windows.h>` or `_WIN32` preprocessor directives were introduced outside `src/core/platform/`. Code formatting passed across 785 C++ files (`tools/ci/check-format.py`). Documentation tests passed (`tests/tools/check_docs_test.py`). No file under `docs/` exceeds 1 MB. Scratch files remained confined to `/tmp/card10-*`.

10. **Evidence Honesty & Deferred Commands (No Finding)**:
    Deferred functional harness runs and benchmark commands are explicitly documented in `docs/model-extensions/evidence/card10/pending-commands.txt` as PENDING with verbatim commands, in accordance with the 2026-10-31 project decision. No skipped or unexecuted tests were misrepresented as passes.

---

## Definition of Done (DoD) Audit Table

| DoD Requirement | Status | Evidence Path & Proof |
| --- | --- | --- |
| **1. Every inventoried model-used header has a classification and public replacement or approved private removal plan.** | **Proven** | `docs/model-extensions/evidence/card10/header-classification.md` comprehensively inventories all 41 unique non-model/family `#include` paths in `src/models/` and `src/families/` into stable (23), platform (7), experimental (14), host wrapper (1), and private with Card 12 removal plan (2). Automated scan confirms 0 unclassified model-used headers. |
| **2. Measured SURFACE closure is reproducible; forwarded source declarations are governed equally with public headers.** | **Proven** | `include/gufo/ext/SURFACE` contains 97 sorted paths. Programmatic closure matches exactly with set difference $\emptyset$. Running `python3 tools/ci/check-extension-api.py --write` produces no changes. `tests/tools/check_extension_api_test.py` proves forwarded source declaration changes trigger policy requirements equally with public headers. |
| **3. Public CMake/schema contracts and API version/changelog gates reject missing required policy changes; no unsupported stability claim is made from wrapper paths alone.** | **Proven** | Version bumped to `0.1.2` in `version.hpp` and `GufoExtension.cmake`; `CHANGES.md` updated. Policy gate `check-extension-api.py --base a761679a --labels '["ext-api"]'` passes (exit 0) and rejects with exit 1 when labels are `[]`. Experimental headers are isolated under `include/gufo/ext/experimental/` with explicit no-stability warranties. |

---

## Executed Verification Commands Log

| # | Command | Exit Code | Verdict |
|---|---|:---:|:---:|
| 1 | `jj status` | 0 | Clean working copy |
| 2 | `jj diff --name-only` | 0 | `RFC.md` untouched, no model files modified |
| 3 | `jj diff cmake/GufoExtension.cmake include/gufo/ext/version.hpp include/gufo/ext/CHANGES.md` | 0 | Version bumped to 0.1.2, changelog updated |
| 4 | Forwarding header pattern validation (`#pragma once` + `#include "src/..."`) | 0 | 44/44 headers conform strictly |
| 5 | Forwarded include target existence check on disk | 0 | 44/44 targets exist |
| 6 | `include/gufo/ext/SURFACE` integrity check (97 paths, sorted, unique, 0 leaks) | 0 | PASS |
| 7 | `python3 tools/ci/check-extension-api.py` | 0 | PASS (97 paths) |
| 8 | `python3 tools/ci/check-extension-api.py --write` (idempotence verification) | 0 | PASS (0 diff) |
| 9 | `python3 tools/ci/check-extension-api.py --base a761679a --labels '["ext-api"]'` | 0 | PASS |
| 10 | `python3 tools/ci/check-extension-api.py --base a761679a --labels '[]'` | 1 | Expected rejection (missing label) |
| 11 | `python3 tests/tools/check_extension_api_test.py` | 0 | 7/7 tests OK |
| 12 | `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership` | 0 | 4/4 isolation checks PASS |
| 13 | `free -h` | 0 | 112 GiB available (>= 15 GiB required) |
| 14 | `nix develop -c cmake --build build/cpu-test --target ext_runtime_neutral_contracts_test extension_runtime_test extension_role_test` | 0 | Built successfully |
| 15 | `nix develop -c ctest --test-dir build/cpu-test -R '^(ext_runtime_neutral_contracts_test\|extension_runtime_test\|extension_role_test\|extension_api_policy_test\|extension_api_surface\|extension_metadata_test\|extension_discovery_test)$' --output-on-failure` | 0 | 7/7 tests passed (100%) |
| 16 | `nix develop -c cmake --build build/gpu-test --target ext_runtime_neutral_contracts_test extension_runtime_test extension_role_test` | 0 | Built successfully |
| 17 | `nix develop -c cmake --build build/gpu-test/tests/extensions/isolated-fixture --target alpha_hip --clean-first` | 0 | Recompiled `fixture/hip.cpp` with all 14 experimental HIP headers |
| 18 | `nix develop -c ctest --test-dir build/gpu-test -R '^(ext_runtime_neutral_contracts_test\|extension_runtime_test\|extension_role_test\|extension_api_policy_test\|extension_api_surface\|extension_metadata_test\|extension_discovery_test)$' --output-on-failure` | 0 | 7/7 tests passed (100%) |
| 19 | `nix develop -c ctest --test-dir build/cpu-test -R '^(qwen27b\|qwen38_flash_next\|deepseek_v4_flash)_continuation_identity_test$' --output-on-failure` | 0 | 3/3 passed (100%) |
| 20 | `nix develop -c ctest --test-dir build/gpu-test -R '^(qwen27b\|qwen38_flash_next\|deepseek_v4_flash)_continuation_identity_test$' --output-on-failure` | 0 | 3/3 passed (100%) |
| 21 | `sha256sum -c docs/model-extensions/evidence/card02/SHA256SUMS` | 0 | 42/42 fixtures OK |
| 22 | `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` | 0 | 785 C++ files formatted |
| 23 | `python3 tests/tools/check_docs_test.py` | 0 | 11/11 tests OK |
| 24 | Windows portability check (no `windows.h`/`_WIN32` outside `src/core/platform/`) | 0 | Clean |
| 25 | Documentation file size check (`find docs/ -type f -size +1M`) | 0 | 0 files > 1 MB |
| 26 | AST/regex check: all `#include` directives in `src/models/` & `src/families/` classified | 0 | 41/41 classified, 0 unclassified |

---

## Conclusion

Card 10 is fully verified, robust, and meets all criteria for approval. All DoD requirements are proven with concrete evidence. The change is ready for integration.
