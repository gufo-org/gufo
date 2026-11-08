# Public API Surface Metrics — Card 10

This document records the exact measurements of the public extension API surface (`include/gufo/ext/SURFACE`) generated and governed under Card 10.

## Overview

The public API surface closure is generated deterministically by:
```sh
python3 tools/ci/check-extension-api.py --write
```
and verified by the extension API policy diff gate.

| Metric | Measured Value |
| --- | --- |
| Total paths in `include/gufo/ext/SURFACE` | **97** |
| Public headers under `include/gufo/ext/` | **49** |
| Forwarded implementation headers under `src/` | **48** |
| Total non-comment lines of code across closure | **4,235** |
| Total code tokens across closure | **31,365** |

---

## Detailed Breakdown of Public Headers (`include/gufo/ext/`, 49 files)

### Stable Public Headers (28 files)
1. `include/gufo/ext/adapter.hpp` (Cache adapter contract)
2. `include/gufo/ext/cache.hpp` (Cache umbrella contract)
3. `include/gufo/ext/cancellable_gate.hpp` (Cancellation gate utility)
4. `include/gufo/ext/CHANGES.md` (Version changelog)
5. `include/gufo/ext/continuation.hpp` (Continuation snapshot/state bases)
6. `include/gufo/ext/core_options.def` (Reserved core CLI option definitions)
7. `include/gufo/ext/crypto.hpp` (Cryptographic hash utilities)
8. `include/gufo/ext/device_probe.hpp` (Hardware status probe)
9. `include/gufo/ext/draft_backend.hpp` (Draft model speculative backend)
10. `include/gufo/ext/gguf_reader.hpp` (GGUF file parser)
11. `include/gufo/ext/host.hpp` (Host logging context handle)
12. `include/gufo/ext/host_budget.hpp` (Host RAM headroom budgets)
13. `include/gufo/ext/image.hpp` (Image data types)
14. `include/gufo/ext/json.hpp` (JSON reader and AST)
15. `include/gufo/ext/ledger.hpp` (Resource ledger interface)
16. `include/gufo/ext/mapped_prefetch.hpp` (Mmap prefetch helper)
17. `include/gufo/ext/model_config.hpp` (Model hyperparameters)
18. `include/gufo/ext/options.hpp` (Extension options parsing)
19. `include/gufo/ext/prompt_lookup.hpp` (N-gram prompt lookup draft backend)
20. `include/gufo/ext/reasoning.hpp` (Reasoning tag parsing)
21. `include/gufo/ext/registry.hpp` (Extension registration and provider registry)
22. `include/gufo/ext/sampling.hpp` (Sampling parameters and grammars)
23. `include/gufo/ext/session_mode.hpp` (Session lifecycle states)
24. `include/gufo/ext/sha256.hpp` (SHA-256 calculation)
25. `include/gufo/ext/speculative_verifier.hpp` (Speculative candidate verifier)
26. `include/gufo/ext/text_contracts.hpp` (Text generation request/response)
27. `include/gufo/ext/types.hpp` (Cache primitive types)
28. `include/gufo/ext/utf8.hpp` (Streaming UTF-8 validation)
29. `include/gufo/ext/version.hpp` (Extension API version constants)

*(Note: `CHANGES.md` is documentation; headers and `.def` make up 28 compilation-facing files plus `version.hpp`).*

### Platform Public Headers (`include/gufo/ext/platform/`, 7 files)
1. `include/gufo/ext/platform/fs.hpp`
2. `include/gufo/ext/platform/library.hpp`
3. `include/gufo/ext/platform/mapped_file.hpp`
4. `include/gufo/ext/platform/os_info.hpp`
5. `include/gufo/ext/platform/pages.hpp`
6. `include/gufo/ext/platform/prefault.hpp`
7. `include/gufo/ext/platform/publish.hpp`

### Experimental Public Headers (`include/gufo/ext/experimental/`, 14 files)
1. `include/gufo/ext/experimental/dispatch_telemetry.hpp`
2. `include/gufo/ext/experimental/gemv_dispatcher.hpp`
3. `include/gufo/ext/experimental/ggml_dequant.hpp`
4. `include/gufo/ext/experimental/ggml_gemm.hpp`
5. `include/gufo/ext/experimental/heap_placement.hpp`
6. `include/gufo/ext/experimental/hip_graph_decode_executor.hpp`
7. `include/gufo/ext/experimental/hip_utils.hpp`
8. `include/gufo/ext/experimental/hipblaslt_plan_database.hpp`
9. `include/gufo/ext/experimental/host_allocations.hpp`
10. `include/gufo/ext/experimental/managed_alloc.hpp`
11. `include/gufo/ext/experimental/snapshot_copy.hpp`
12. `include/gufo/ext/experimental/snapshot_transfer.hpp`
13. `include/gufo/ext/experimental/transfer_pool.hpp`
14. `include/gufo/ext/experimental/weight_upload.hpp`

---

## Breakdown of Forwarded Implementation Headers (`src/`, 48 files)

### Cache Subsystem (`src/cache/`, 6 files)
- `src/cache/adapter.hpp`
- `src/cache/completion.hpp` (transitive via `adapter.hpp`)
- `src/cache/identity.hpp` (transitive via `adapter.hpp`)
- `src/cache/ledger.hpp`
- `src/cache/stream.hpp` (transitive via `adapter.hpp`)
- `src/cache/types.hpp`

### Core Subsystem (`src/core/`, 38 files)
- **Platform leaves (7)**: `src/core/platform/{fs,library,mapped_file,os_info,pages,prefault,publish}.hpp`
- **HIP experimental (8)**: `src/core/hip/{heap_placement,hip_utils,host_allocations,managed_alloc,snapshot_copy,snapshot_transfer,transfer_pool,weight_upload}.hpp`
- **HIP detail experimental (4)**: `src/core/hip/detail/{dispatch_telemetry,gemv_dispatcher,hip_graph_decode_executor,hipblaslt_plan_database}.hpp`
- **Quantization experimental (2)**: `src/core/quant/{ggml_dequant,ggml_gemm}.hpp`
- **Core services and utilities (11)**: `src/core/{cancellable_gate,image,json,json_constraint,json_schema_lexeme,mapped_prefetch,model_config,reasoning,sampling,session_mode,utf8}.hpp`
- **Crypto (1)**: `src/core/crypto/sha256.hpp`
- **GGUF (1)**: `src/core/gguf_reader.hpp`
- **Logging (1)**: `src/core/logging/logging.hpp`
- **Speculative (3)**: `src/core/speculative/{draft_backend,prompt_lookup,speculative_verifier}.hpp`

### Extension Subsystem (`src/ext/`, 4 files)
- `src/ext/continuation.hpp`
- `src/ext/device_probe.hpp`
- `src/ext/host_budget.hpp`
- `src/ext/text_contracts.hpp`

---

## Coupling and Leak Analysis

1. **Serving Policy Leaks**:
   - `grep -rn "src/cli/" include/gufo/ext/`: **0 occurrences**.
   - Neither `src/cli/serve/**` nor HTTP headers are reachable anywhere within the 97 closure files.
2. **Model Include Leaks**:
   - No `src/models/**` or `src/families/**` headers are included by any file in `SURFACE`.
3. **Grammar Constraints**:
   - `json_constraint.hpp` and `json_schema_lexeme.hpp` are included transitively by `src/core/sampling.hpp` for JSON schema grammar-guided sampling. Both are governed as part of `SURFACE`.
4. **Cache Completion and Streams**:
   - `completion.hpp`, `identity.hpp`, and `stream.hpp` are included transitively by `src/cache/adapter.hpp`. All three are governed as part of `SURFACE`.
5. **Governance**:
   - Any modification to tokens or declarations in all 97 files triggers the extension API diff gate, requiring an `ext-api` PR label, an API version increase, and an entry in `include/gufo/ext/CHANGES.md`.
