# Model Header Classification Inventory — Card 10

This inventory classifies every `#include` directive from `src/models/` and `src/families/` into `src/core/`, `src/cache/`, `src/cli/`, and `src/ext/` per RFC Step 7 (Card 10 of model-extensions RFC).

## Classification Summary

| Classification Tier | Target Prefix | Header Count | Description |
| --- | --- | --- | --- |
| **Stable public API** | `include/gufo/ext/` | 23 new (+ 5 pre-existing) | Core types, sampling, speculative contracts, cache interfaces, neutral runtime contracts |
| **Platform public API** | `include/gufo/ext/platform/` | 7 | Cross-platform OS leaves in `src/core/platform/` |
| **Experimental forwarding** | `include/gufo/ext/experimental/` | 14 | GPU kernels, device memory management, telemetry, plan databases |
| **Host wrapper / services** | `include/gufo/ext/host.hpp` | 1 | Process-wide logging service handle (`src/core/logging/logging.hpp`) |
| **Private / internal** | N/A (unexported) | 2 | Serving boundary violations in test code (`src/cli/serve/asr_service.hpp`, `src/cli/serve/audio_asr_api.hpp`); approved for removal in Card 12 |

---

## Detailed Header Classification

### 1. Platform Public API (`include/gufo/ext/platform/*.hpp`)

Platform leaves provide OS-neutral system facilities conforming to the AGENTS.md rule (no direct `windows.h` or `_WIN32` outside `src/core/platform/`).

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/core/platform/fs.hpp` | `include/gufo/ext/platform/fs.hpp` | Platform | File system path and utility routines used by loaders and models. |
| `src/core/platform/library.hpp` | `include/gufo/ext/platform/library.hpp` | Platform | Dynamic library loader for media processors (e.g. EmbeddingGemma). |
| `src/core/platform/mapped_file.hpp` | `include/gufo/ext/platform/mapped_file.hpp` | Platform | Read-only mapped memory file abstraction used by GGUF weight loaders. |
| `src/core/platform/os_info.hpp` | `include/gufo/ext/platform/os_info.hpp` | Platform | System page size, host memory, terminal status, process queries. |
| `src/core/platform/pages.hpp` | `include/gufo/ext/platform/pages.hpp` | Platform | Page commit/decommit and virtual memory management. |
| `src/core/platform/prefault.hpp` | `include/gufo/ext/platform/prefault.hpp` | Platform | Parallel page prefaulting routines for model weight mappings. |
| `src/core/platform/publish.hpp` | `include/gufo/ext/platform/publish.hpp` | Platform | Atomic memory publishing and cross-thread barriers. |

### 2. Cache Public Contracts (`include/gufo/ext/*.hpp`)

Neutral continuation cache interfaces and descriptors allowing extensions to plug into Gufo's prompt caching subsystem.

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/cache/adapter.hpp` | `include/gufo/ext/adapter.hpp` | Cache | Continuation adapter base class and mutation guard contracts. |
| `src/cache/adapter.hpp` + `src/cache/ledger.hpp` + `src/cache/types.hpp` | `include/gufo/ext/cache.hpp` | Cache | Umbrella header forwarding all fundamental cache contracts. |
| `src/cache/ledger.hpp` | `include/gufo/ext/ledger.hpp` | Cache | Resource ledger and token accounting interfaces. |
| `src/cache/types.hpp` | `include/gufo/ext/types.hpp` | Cache | Fundamental cache types (`Token`, `SlotId`, `ComponentDescriptor`, etc.). |

### 3. Core Services and Utilities (`include/gufo/ext/*.hpp`)

Core utilities, serialization readers, and runtime algorithms.

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/core/cancellable_gate.hpp` | `include/gufo/ext/cancellable_gate.hpp` | Core | Synchronous cancellation gating for long-running decodes. |
| `src/core/crypto/sha256.hpp` | `include/gufo/ext/crypto.hpp`, `include/gufo/ext/sha256.hpp` | Core | Cryptographic SHA-256 calculation for cache identity and prompts. |
| `src/core/gguf_reader.hpp` | `include/gufo/ext/gguf_reader.hpp` | Core | Zero-copy GGUF metadata parsing and tensor dictionary reader. |
| `src/core/image.hpp` | `include/gufo/ext/image.hpp` | Core | Image representation, color conversion, and spatial metadata. |
| `src/core/json.hpp` | `include/gufo/ext/json.hpp` | Core | JSON lexing, parsing, and structured value representation. |
| `src/core/mapped_prefetch.hpp` | `include/gufo/ext/mapped_prefetch.hpp` | Core | Prefetch scheduler for memory-mapped tensor regions. |
| `src/core/model_config.hpp` | `include/gufo/ext/model_config.hpp` | Core | Hyperparameter definitions and architecture configuration structs. |
| `src/core/reasoning.hpp` | `include/gufo/ext/reasoning.hpp` | Core | Thinking tag detection (`<think>`, `</think>`) and token slicing. |
| `src/core/sampling.hpp` | `include/gufo/ext/sampling.hpp` | Core | Logit sampling, temperature, top-p, min-p, and JSON grammar constraints. |
| `src/core/session_mode.hpp` | `include/gufo/ext/session_mode.hpp` | Core | Session lifecycle modes and execution phases. |
| `src/core/utf8.hpp` | `include/gufo/ext/utf8.hpp` | Core | Streaming UTF-8 validation and codepoint boundary tracking. |

### 4. Speculative Decoding Contracts (`include/gufo/ext/*.hpp`)

Interfaces for draft verification and candidate proposal.

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/core/speculative/draft_backend.hpp` | `include/gufo/ext/draft_backend.hpp` | Core | Draft model backend contract for speculative decoding. |
| `src/core/speculative/prompt_lookup.hpp` | `include/gufo/ext/prompt_lookup.hpp` | Core | N-gram prompt lookup draft proposal engine. |
| `src/core/speculative/speculative_verifier.hpp` | `include/gufo/ext/speculative_verifier.hpp` | Core | Verification engine matching draft tokens against target logits. |

### 5. Extension Runtime Contracts (`include/gufo/ext/*.hpp`)

Neutral extension contracts extracted in Card 09.

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/ext/continuation.hpp` | `include/gufo/ext/continuation.hpp` | Extension Runtime | Neutral continuation snapshot, state, and identity interfaces. |
| `src/ext/device_probe.hpp` | `include/gufo/ext/device_probe.hpp` | Extension Runtime | Hardware probe for health, readiness, and stream status. |
| `src/ext/host_budget.hpp` | `include/gufo/ext/host_budget.hpp` | Extension Runtime | Host RAM headroom query and snapshot allocation budgeting. |
| `src/ext/text_contracts.hpp` | `include/gufo/ext/text_contracts.hpp` | Extension Runtime | Text generation request structures and reasoning tags. |

### 6. Experimental Forwarding Contracts (`include/gufo/ext/experimental/*.hpp`)

Device memory management, HIP execution helpers, plan databases, and quantization kernels. These are published under `experimental/` without stability guarantees until multi-model stabilization.

| Source Header | Public Forwarding Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/core/hip/hip_utils.hpp` | `include/gufo/ext/experimental/hip_utils.hpp` | Core HIP | HIP macro wrappers (`HIP_CHECK`, `HIPBLAS_CHECK`) and error handlers. |
| `src/core/hip/heap_placement.hpp` | `include/gufo/ext/experimental/heap_placement.hpp` | Core HIP | Device heap headroom reservation and allocation strategy. |
| `src/core/hip/host_allocations.hpp` | `include/gufo/ext/experimental/host_allocations.hpp` | Core HIP | Pinned host memory allocator and huge-page management. |
| `src/core/hip/managed_alloc.hpp` | `include/gufo/ext/experimental/managed_alloc.hpp` | Core HIP | Unified memory allocation (`hipMallocManaged`) helpers. |
| `src/core/hip/snapshot_copy.hpp` | `include/gufo/ext/experimental/snapshot_copy.hpp` | Core HIP | Asynchronous state copy between host and device buffers. |
| `src/core/hip/snapshot_transfer.hpp` | `include/gufo/ext/experimental/snapshot_transfer.hpp` | Core HIP | Transfer stream lease management and synchronization. |
| `src/core/hip/transfer_pool.hpp` | `include/gufo/ext/experimental/transfer_pool.hpp` | Core HIP | Dedicated transfer stream pool for cache snapshot transfers. |
| `src/core/hip/weight_upload.hpp` | `include/gufo/ext/experimental/weight_upload.hpp` | Core HIP | Pipelined asynchronous weight uploading engine. |
| `src/core/hip/detail/dispatch_telemetry.hpp` | `include/gufo/ext/experimental/dispatch_telemetry.hpp` | Core HIP Detail | Kernel launch telemetry and performance counters. |
| `src/core/hip/detail/gemv_dispatcher.hpp` | `include/gufo/ext/experimental/gemv_dispatcher.hpp` | Core HIP Detail | Wave32/wave64 GEMV dispatch routing tables. |
| `src/core/hip/detail/hipblaslt_plan_database.hpp` | `include/gufo/ext/experimental/hipblaslt_plan_database.hpp` | Core HIP Detail | Tuned GEMM plan cache and hardware profile lookup. |
| `src/core/hip/detail/hip_graph_decode_executor.hpp` | `include/gufo/ext/experimental/hip_graph_decode_executor.hpp` | Core HIP Detail | HIP graph capture and replay executor for decode steps. |
| `src/core/quant/ggml_dequant.hpp` | `include/gufo/ext/experimental/ggml_dequant.hpp` | Core Quant | CPU/GPU block dequantization for Q4_0, Q4_K, Q6_K, Q8_0. |
| `src/core/quant/ggml_gemm.hpp` | `include/gufo/ext/experimental/ggml_gemm.hpp` | Core Quant | Matrix-vector multiplication kernels for quantized weights. |

### 7. Host Services (`include/gufo/ext/host.hpp`)

| Source Header | Public Header | Owning Subsystem | Rationale |
| --- | --- | --- | --- |
| `src/core/logging/logging.hpp` | `include/gufo/ext/host.hpp` | Core Logging | Provides `Host::Log` callback and `LogLevel` enum without exposing serving loggers. |

### 8. Private / Internal with Approved Removal Plan

| Source Header | Included By | Classification | Removal / Migration Plan |
| --- | --- | --- | --- |
| `src/cli/serve/asr_service.hpp` | `src/models/qwen3_asr/tests/audio_api_test.cpp` | Private / Internal | Serving HTTP boundary leak in test code; Card 12 adds in-tree model API test support and removes this edge. |
| `src/cli/serve/audio_asr_api.hpp` | `src/models/qwen3_asr/tests/audio_api_test.cpp` | Private / Internal | Serving HTTP boundary leak in test code; Card 12 adds in-tree model API test support and removes this edge. |
