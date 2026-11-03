# Card 10 Integration — Edits Complete, Editing Ownership Released

Integration agent (`daeda205-c3f4-4d9f-a035-6467a64d8706`, `codex-exec-writer`) has completed all integration tasks and releases editing ownership to the orchestrator.
No commits, pushes, servers, functional harness runs, benchmarks, or full sweeps were performed.
No active background handles.
`docs/model-extensions/RFC.md` remains strictly byte-identical and uncommitted.

## Status & Summary

- **Card**: Card 10 — Publish and measure the public extension API
- **Working Copy**: `@` over parent commit `klwmvoxr a761679a` ("model-extensions | chore(extensions): merge main and resolve integration conflicts")
- **Review Findings**: Verdict **PASS** (0 findings / clean conformance).
- **Integration Gates**: All affected gates re-run ONCE with clean results recorded in `docs/model-extensions/evidence/card10/integration-reruns.txt`:
  * `python3 tools/ci/check-extension-api.py`: PASS (97 paths, exit 0)
  * `python3 tools/ci/check-extension-api.py --base a761679a --labels '["ext-api"]'`: PASS (46 changed paths recognized, exit 0)
  * `python3 tools/ci/check-extension-api.py --base a761679a --labels '[]'`: PASS (Negative control: expected rejection with exit 1)
  * `python3 tests/tools/check_extension_api_test.py`: PASS (7/7 tests OK, exit 0)
  * `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership`: PASS (all 4 boundary and ownership gates pass, exit 0)
  * `python3 tests/tools/check_docs_test.py`: PASS (11/11 tests OK, exit 0)
  * `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`: PASS (785 C++ files formatted, exit 0)
  * Bounded CPU CTest: 10/10 passed (100%, exit 0), including all 7 extension suite tests and 3 continuation identity goldens
  * Bounded GPU CTest: 10/10 passed (100%, exit 0), including all 7 extension suite tests and 3 continuation identity goldens
- **Acceptance Audit**: All 3 DoD requirements proven and checked `[x]` in `cards/10-public-api-surface.md`.
- **Card & Ledger Documentation**:
  * `docs/model-extensions/cards/10-public-api-surface.md`: Updated status, agent sequence (Worker: `6d5263a1-9f0d-46ef-a092-76804e8f1003`, Reviewer: `922a7582-1ace-4ef7-b37c-e926d5df7c5b`, Integrator: `daeda205-c3f4-4d9f-a035-6467a64d8706`), review findings disposition table, validation and acceptance audit, and remaining limitations.
  * `docs/model-extensions/CARDS.md`: Marked Card 10 `complete` for in-card gates, updated active range through Card 10, and added `### Card 10` execution record summary.
  * `docs/model-extensions/cards/README.md`: Updated checkpoint with Card 10 completion summary.
  * `docs/model-extensions/evidence/card10/README.md`: Created provenance ledger.
  * `docs/model-extensions/evidence/card10/integration-reruns.txt`: Created exact rerun log.

## Changed Paths in Working Copy

1. **Build & Extension Governance**:
   - `cmake/GufoExtension.cmake` (version 0.1.2)
   - `include/gufo/ext/version.hpp` (API revision 2, version 0.1.2)
   - `include/gufo/ext/CHANGES.md` (changelog for 0.1.2)
   - `include/gufo/ext/SURFACE` (measured transitive surface of 97 paths)
2. **Forwarding Headers (44 new)**:
   - 7 platform headers: `include/gufo/ext/platform/{fs,library,mapped_file,os_info,pages,prefault,publish}.hpp`
   - 14 experimental headers: `include/gufo/ext/experimental/{dispatch_telemetry,gemv_dispatcher,ggml_dequant,ggml_gemm,heap_placement,hip_graph_decode_executor,hip_utils,hipblaslt_plan_database,host_allocations,managed_alloc,snapshot_copy,snapshot_transfer,transfer_pool,weight_upload}.hpp`
   - 23 stable headers: `include/gufo/ext/{adapter,cache,cancellable_gate,continuation,crypto,device_probe,draft_backend,gguf_reader,host_budget,image,json,ledger,mapped_prefetch,model_config,prompt_lookup,reasoning,sampling,session_mode,sha256,speculative_verifier,text_contracts,types,utf8}.hpp`
3. **Tests**:
   - `tests/extensions/ext_runtime_neutral_contracts_test.cpp`
   - `tests/extensions/fixture/cpu.cpp`
   - `tests/extensions/fixture/hip.cpp`
   - `tests/extensions/metadata_test.py`
   - `tests/extensions/runtime_test.cpp`
   - `tests/tools/check_extension_api_test.py`
4. **Documentation & Evidence**:
   - `docs/model-extensions/CARDS.md`
   - `docs/model-extensions/cards/README.md`
   - `docs/model-extensions/cards/10-public-api-surface.md`
   - `docs/model-extensions/evidence/card10/RESUME-NOTES.md`
   - `docs/model-extensions/evidence/card10/README.md`
   - `docs/model-extensions/evidence/card10/integration-reruns.txt`
   - `docs/model-extensions/evidence/card10/SURFACE-metrics.md`
   - `docs/model-extensions/evidence/card10/header-classification.md`
   - `docs/model-extensions/evidence/card10/build-and-tests.txt`
   - `docs/model-extensions/evidence/card10/pending-commands.txt`
   - `docs/model-extensions/evidence/card10/review-findings.md`
   - `docs/model-extensions/evidence/card10/input-brief-worker.md`
   - `docs/model-extensions/evidence/card10/input-brief-reviewer.md`
   - `docs/model-extensions/evidence/card10/input-brief-integration.md`

## Scratch to Delete at Closure

- `/tmp/card10-logs/`
- `/tmp/card10-review/`
- `/tmp/card10-integration/`

---

## Historical Worker Handoff

# Card 10 Worker Resume Notes

## Phase & Role
- **Card**: Card 10 — Publish and measure the public extension API
- **Agent Role**: Implementation Worker (`worker`)
- **Status**: Complete, ready for independent review
- **Working Copy**: `@` (commit hash `47a03291...` over parent `klwmvoxr a761679a`)

## Completed Tasks

1. **Header Classification & Inventory**:
   - Inventoried all model-used `#include` paths in `src/models/` and `src/families/` into `src/core/`, `src/cache/`, `src/cli/`, and `src/ext/`.
   - Produced `docs/model-extensions/evidence/card10/header-classification.md` classifying all 45+ model-used headers into:
     * Stable public API (`include/gufo/ext/*.hpp`)
     * Platform public API (`include/gufo/ext/platform/*.hpp`)
     * Experimental forwarding (`include/gufo/ext/experimental/*.hpp`)
     * Host wrapper (`include/gufo/ext/host.hpp`)
     * Private/internal with approved removal plan (`src/cli/serve/asr_service.hpp`, `src/cli/serve/audio_asr_api.hpp` in qwen3_asr tests, removed in Card 12).

2. **Published Forwarding Headers**:
   - Created 7 platform forwarding headers in `include/gufo/ext/platform/` (`fs.hpp`, `library.hpp`, `mapped_file.hpp`, `os_info.hpp`, `pages.hpp`, `prefault.hpp`, `publish.hpp`).
   - Created 14 experimental forwarding headers in `include/gufo/ext/experimental/` (`hip_utils.hpp`, `heap_placement.hpp`, `host_allocations.hpp`, `managed_alloc.hpp`, `snapshot_copy.hpp`, `snapshot_transfer.hpp`, `transfer_pool.hpp`, `weight_upload.hpp`, `dispatch_telemetry.hpp`, `gemv_dispatcher.hpp`, `hipblaslt_plan_database.hpp`, `hip_graph_decode_executor.hpp`, `ggml_dequant.hpp`, `ggml_gemm.hpp`).
   - Created 23 stable forwarding headers in `include/gufo/ext/` (`adapter.hpp`, `cache.hpp`, `ledger.hpp`, `types.hpp`, `cancellable_gate.hpp`, `crypto.hpp`, `sha256.hpp`, `gguf_reader.hpp`, `image.hpp`, `json.hpp`, `mapped_prefetch.hpp`, `model_config.hpp`, `reasoning.hpp`, `sampling.hpp`, `session_mode.hpp`, `utf8.hpp`, `draft_backend.hpp`, `prompt_lookup.hpp`, `speculative_verifier.hpp`, `continuation.hpp`, `device_probe.hpp`, `host_budget.hpp`, `text_contracts.hpp`).

3. **Extension API Version Bump & Changelog**:
   - Bumped `kApiRevision = 2;` and `kApiVersion = "0.1.2"` in `include/gufo/ext/version.hpp`.
   - Updated `set(GUFO_EXTENSION_API_VERSION "0.1.2")` in `cmake/GufoExtension.cmake`.
   - Added `## 0.1.2` entry in `include/gufo/ext/CHANGES.md`.

4. **Generated & Measured `SURFACE` Closure**:
   - Generated `include/gufo/ext/SURFACE` with `python3 tools/ci/check-extension-api.py --write`.
   - Measured and documented 97 paths (49 public headers + 48 forwarded `src/` headers, 4,235 LoC, 31,365 tokens) in `docs/model-extensions/evidence/card10/SURFACE-metrics.md`.
   - Confirmed zero serving leaks (`src/cli/`) and zero model include leaks.

5. **Policy Diff Tests & Tooling Extension**:
   - Extended `tests/tools/check_extension_api_test.py` with tests for:
     * Transitive additions to `SURFACE` requiring policy compliance
     * Transitive deletions from `SURFACE` requiring policy compliance
     * Forwarded declaration modifications across subdirectories (`platform/`, `experimental/`)
     * Transitive closure resolution across subdirectories
   - All 7 tests pass in `check_extension_api_test.py`.

6. **Extension Consumer Tests**:
   - Updated `tests/extensions/ext_runtime_neutral_contracts_test.cpp` to include the public stable and platform headers and test runtime version constants.
   - Updated `tests/extensions/fixture/cpu.cpp` to include representative public stable/platform headers.
   - Updated `tests/extensions/fixture/hip.cpp` to include representative experimental HIP headers under `ENGINE_ENABLE_HIP`.
   - Updated `tests/extensions/runtime_test.cpp` and `tests/extensions/metadata_test.py` to match the `0.1.2` API version.
   - Built and passed 7/7 CTest suite in both `build/cpu-test` and `build/gpu-test`.

7. **Validation & Acceptance Gates**:
   - `python3 tools/ci/check-extension-api.py` (PASS, 97 paths)
   - `python3 tools/ci/check-extension-api.py --base a761679a --labels '["ext-api"]'` (PASS)
   - `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership` (PASS)
   - Exact identity goldens in CPU and GPU: 3/3 passed (100%), SHA256SUMS verified.
   - Formatting: `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py` (PASS, 785 C++ files).
   - Documentation: `python3 tests/tools/check_docs_test.py` (PASS, 11 tests).

8. **Scratch & Cleanup**:
   - Build logs recorded under `/tmp/card10-logs/` (`build-cpu-test.log`, `ctest-cpu.log`, `ctest-cpu-rerun.log`, `build-gpu-test.log`, `ctest-gpu.log`).
   - No temporary scratch files created under `docs/` or repository root.
   - `RFC.md` remains byte-identical and uncommitted.
   - No commits made (`jj commit`, `git commit` not invoked).
