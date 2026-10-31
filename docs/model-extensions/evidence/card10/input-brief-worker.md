> Instructions given to the implementation agent, written before it ran. This is a
> brief, not a result. Results go in the other files of this directory.

# Worker brief — Card 10: Publish and measure the public extension API

You are the **implementation agent** (`worker`) for Card 10 of the model-extensions RFC in
`/home/mixer/gufo` (jj working copy `@`, parent commit `klwmvoxr a761679a`). You have
**sole editing ownership** of the shared worktree while you run.

## Absolute constraints

1. **Never run `jj commit`, `jj new`, `jj squash`, `git commit`, or anything that
   changes revision state.** The orchestrator commits. Read-only commands (`jj st`,
   `jj diff`, `jj log`) are permitted.
2. `docs/model-extensions/RFC.md` must stay **byte-identical and uncommitted**.
   Do not read or write `HANDOFF.md` (deleted; its decisions now live in `CARDS.md` and `cards/README.md`).
3. Task-owned paths:
   - `include/gufo/ext/**` (new and existing headers, `SURFACE`, `version.hpp`, `CHANGES.md`)
   - `cmake/GufoExtension.cmake`
   - `src/ext/**` (if any contract definitions need alignment or exports)
   - `tests/extensions/**`
   - `tests/tools/**`
   - `tools/ci/check-extension-api.py` (if adjustment is needed)
   - `docs/model-extensions/cards/10-public-api-surface.md`
   - `docs/model-extensions/evidence/card10/**`
   - `docs/model-extensions/CARDS.md` (status updates if appropriate)
   - Do NOT edit model files in `src/models/` to migrate their includes in this card;
     Card 11 is dedicated to migrating model includes mechanically.
4. **No functional harness.** `tests/functional/run.py`, `prompt_lookup.py`, speed
   benchmarks, and main-control matched timings are deferred by user decision (2026-10-31)
   to the single post-Card-14 qualification window. Do NOT run them or start a `gufo serve`
   server. Record deferred requirements as PENDING with their verbatim commands in
   `docs/model-extensions/evidence/card10/pending-commands.txt`.
5. Scratch goes in `/tmp/card10-*` only (logs, caches, temp files) and MUST be deleted
   or listed in `RESUME-NOTES.md` for cleanup. No evidence file over 1 MB under `docs/`.
   No raw build logs, no per-row report trees under `docs/`.
   Use `build/card10-gpu` and `build/card10-cpu` as CMake build directories if configuring fresh,
   or reuse existing preset directories.
6. Keep tool output small: redirect build and test logs to `/tmp/card10-logs/<name>.log`
   and inspect only using `tail -n 30` or grep. Never output entire build logs.
7. Host RAM: Before running any GPU build or test, check `free -g` and ensure at least
   15 GiB headroom. The `llama-swap.service` must stay running; do not stop services.
8. **No behavior change.** Preserve existing functionality. Card 10 is about publishing,
   classifying, and governing the public extension API surface.

## Background & Objectives

Card 10 is RFC Step 7 (first part): "Publish core headers used by models under
`include/gufo/ext/` (stable, `platform/`, `experimental/`) per the inventory classification;
rewrite model includes, one mechanical PR per model." Card 10 performs the publishing,
classification, governance, and measurement of the public API surface; Card 11 performs
the mechanical per-model include migrations.

Forwarding headers expose a transitive core/cache closure. The public API surface is defined
by `include/gufo/ext/SURFACE`, generated via `python3 tools/ci/check-extension-api.py --write`.

## Step-by-Step Implementation Plan

### Step 1: Enumerate and Classify Every Core/Cache Header Used by Models

Analyze all `#include` directives from `src/models/` and `src/families/` into `src/core/`,
`src/cache/`, and `src/cli/`.
Produce `docs/model-extensions/evidence/card10/header-classification.md` documenting:
- Every header
- Classification:
  * **Stable public API** (`include/gufo/ext/*.hpp`)
  * **Platform public API** (`include/gufo/ext/platform/*.hpp`)
  * **Experimental forwarding** (`include/gufo/ext/experimental/*.hpp`)
  * **Host wrapper / host services**
  * **Private / internal** (must have an approved removal plan, such as the ASR test serving includes being removed in Card 12)
- Intended owner & rationale

The inventoried headers include:
- Platform leaves: `src/core/platform/{fs,library,mapped_file,os_info,pages,prefault,publish}.hpp`
- Cache contracts: `src/cache/{adapter,ledger,types}.hpp`
- Core services & utilities:
  * `src/core/{cancellable_gate,image,json,mapped_prefetch,model_config,reasoning,sampling,session_mode,utf8}.hpp`
  * `src/core/crypto/sha256.hpp`
  * `src/core/gguf_reader.hpp`
  * `src/core/logging/logging.hpp` (already forwarded via `host.hpp`)
  * `src/core/quant/{ggml_dequant,ggml_gemm}.hpp`
  * `src/core/speculative/{draft_backend,prompt_lookup,speculative_verifier}.hpp`
  * `src/core/hip/{heap_placement,hip_utils,host_allocations,managed_alloc,snapshot_copy,snapshot_transfer,transfer_pool,weight_upload}.hpp`
  * `src/core/hip/detail/{dispatch_telemetry,gemv_dispatcher,hipblaslt_plan_database,hip_graph_decode_executor}.hpp`
- Runtime contracts from Card 09:
  * `src/ext/{continuation,device_probe,host_budget,text_contracts}.hpp`

### Step 2: Publish Forwarding Headers under `include/gufo/ext/`

Create the forwarding headers under `include/gufo/ext/`, `include/gufo/ext/platform/`,
and `include/gufo/ext/experimental/`.
Each forwarding header must:
- Use `#pragma once`
- Forward directly to the real implementation header, e.g.:
  `#include "src/core/platform/fs.hpp"`
  `#include "src/cache/adapter.hpp"`
- Not expose or re-export any serving policy (`src/cli/**` or `gufo_serving`)
- Match the established naming conventions (e.g. `include/gufo/ext/platform/fs.hpp`,
  `include/gufo/ext/cache.hpp` or `adapter.hpp`, `include/gufo/ext/experimental/hip_utils.hpp`, etc.)

### Step 3: Bump Extension API Version and Update Changes

Since Card 10 publishes new headers and declarations (additive source API expansion):
1. Bump API version in `include/gufo/ext/version.hpp`:
   `kApiRevision = 2;` -> version `"0.1.2"` (or appropriate minor/revision per policy)
2. Update `cmake/GufoExtension.cmake`:
   `set(GUFO_EXTENSION_API_VERSION "0.1.2")`
3. Add a clear, comprehensive `## 0.1.2` entry in `include/gufo/ext/CHANGES.md` describing:
   - Newly published forwarding headers (`stable`, `platform/`, `experimental/`)
   - Forwarded cache, platform, and core contracts
   - Retention of source compatibility without binary ABI guarantees

### Step 4: Generate and Measure `SURFACE` Closure

1. Run `python3 tools/ci/check-extension-api.py --write`.
2. Inspect `include/gufo/ext/SURFACE`. Ensure the transitive closure contains all expected
   public headers and their forwarded `src/` files, and no unexpected leaks.
3. Run `python3 tools/ci/check-extension-api.py` (without `--write`) to verify that the
   check passes cleanly with returncode 0.
4. Record metrics in `docs/model-extensions/evidence/card10/SURFACE-metrics.md`:
   - Number of public headers under `include/gufo/ext/`
   - Total files in transitive closure (`SURFACE`)
   - Breakdown of headers: public, platform, experimental, forwarded `src/` headers
   - Declaration counts and coupling analysis

### Step 5: Extend Policy & Diff Tests

1. Extend `tests/tools/check_extension_api_test.py` to cover:
   - Transitive additions, modifications, and deletions in `SURFACE`
   - Forwarded headers across subdirectories (`platform/`, `experimental/`)
   - Verification that changes require API version bump, CHANGES entry, and `ext-api` label
2. Ensure `python3 tests/tools/check_extension_api_test.py` passes cleanly.

### Step 6: Compile Representative HIP-less and HIP Consumers

1. In `tests/extensions/`, create or update tests that consume the newly published headers:
   - A CPU/HIP-less consumer (e.g., in `ext_runtime_neutral_contracts_test` or a dedicated test)
     verifying that inclusion of `<gufo/ext/...>` headers compiles cleanly in standard C++20 without HIP.
   - A HIP consumer (if HIP is enabled) verifying device/HIP headers compile cleanly.
   - Verify that no private serving headers or unclassified symbols leak into extension compilation units.
2. Ensure tests are registered and pass under CTest.

### Step 7: Run Validation Suite and Record Evidence

Run the required gates:
1. `python3 tools/ci/check-extension-api.py`
2. `python3 tests/tools/check_extension_api_test.py`
3. `python3 tools/ci/check-extension-isolation.py --fail-on-core-model-includes --fail-on-serving-boundary --fail-on-family-graph --fail-on-target-ownership`
4. CTest selection covering extension tests and identity goldens in both GPU and CPU configurations
5. Format check: `python3 tools/ci/check-format.py` (run with `--fix` if needed)
6. Docs check: `python3 tests/tools/check_docs_test.py`

### Step 8: Document Evidence and Hand Off

Write all required evidence files in `docs/model-extensions/evidence/card10/`:
- `header-classification.md`
- `SURFACE-metrics.md`
- `build-and-tests.txt`
- `pending-commands.txt`
- `RESUME-NOTES.md`

Update `docs/model-extensions/cards/10-public-api-surface.md` with implementation status,
DoD checks, and summary.
Finish with an explicit handoff statement indicating completion of worker phase and releasing editing ownership.
