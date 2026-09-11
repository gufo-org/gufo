# Project Status and Decisions

Status: active implementation, 2026-08-20

## Current Implementation Boundary

The native runtime loads Qwen3.8 GGUF artifacts, executes prefill and decode on
gfx1151, and serves the OpenAI-compatible HTTP path with request-owned state.
XDNA2 has a reproducible XRT/AIE lifecycle and a numerically validated
model-private MTP RMSNorm program. K-quant matrix operators and complete NPU
draft generation are the active implementation boundary.

## Product Decisions

### Platform

- The supported production platform is Linux x86-64 on AMD Strix Halo.
- The target GPU is `gfx1151`; the target NPU is XDNA2/AIE2P.
- Windows, macOS, and CUDA are out of scope, not future compatibility goals.
- Nix is the supported build and development entry point. CMake remains the
  internal native build system invoked by Nix.

### Models

- Qwen3.8-27B is the first production model.
- Qwen3.5-0.8B is the bring-up and rapid-iteration model.
- Both use the Qwen3.5 implementation family and the same repeating
  `3 x Gated DeltaNet + 1 x full attention` block pattern. They share operator
  families and format contracts, but retain model-specific dimensions, kernel
  policies, tuning records, and acceptance results.
- The runtime supports curated compiled model kinds rather than a general
  architecture loader.

Official model sources:

- `https://huggingface.co/Qwen/Qwen3.8-27B`
- `https://huggingface.co/Qwen/Qwen3.5-0.8B`

### First Artifact and Capability

- The first Qwen3.8-27B production artifact is text-only.
- It excludes the vision encoder and declares only text capability.
- Image and video inputs are rejected rather than silently ignored.
- MTP weights remain available for later speculative-decoding work.
- A multimodal artifact, if pursued, is a separate future product with its own
  model kind/capability contract and release gates.

### First Vertical Slice

Executable naming and CLI topology are intentionally left to their existing
design documents in this update; they are not material to the runtime milestone.

The first externally usable inference milestone is native, text-only, greedy
terminal generation:

```text
prompt -> native tokenizer -> native model -> GPU -> greedy tokens
```

It must load a validated artifact, maintain minimal model/KV state, and match
pinned reference token fixtures. Sampling, interactive chat, HTTP, continuous
batching, and speculative decoding follow this correctness slice.

### Quantization Evidence

The Qwen3.5-0.8B mixed-precision results are bring-up evidence, not a production
recipe for Qwen3.8-27B. The tooling's current `embed_ffn` default is an
experimental convenience. Qwen3.8-27B receives its own calibration, per-layer
attribution, quality gates, kernel measurements, and selected recipe.

The committed Qwen3.5 teacher suite currently has 78 scored positions and
SHA-256 `b33862883e78e7500cc8553cffe68ec8c44ca5ac1ceca8cbccb3070f6d42b131`.
The disjoint calibration suite SHA-256 is
`f1314e80685482cd80c99a6093054ce44ec2b326fa2edc7c87065dc249a1b37e`.
Changing a suite hash creates a new benchmark identity.

### SHQ Format Stability

`SHQ-T16` is the project-owned tiled quantized weight family. The current
Python-produced SHQ4/SHQ6/SHQ8 layout is a candidate contract, not yet a frozen
portable v1 artifact ABI.

The v1 contract is frozen only after:

1. An independent C++ parser validates bounds, offsets, alignment, and checksums.
2. CPU decoding matches byte-exact conformance vectors.
3. HIP kernels consume the packed representation and match the CPU oracle.
4. Promoted AIE programs match the same numerical contract.
5. Malformed and corrupted artifacts fail closed.

Artifacts produced before this gate may require regeneration.

## GPU and NPU Policy

The complete evidence record, quantitative results, caveats, and probe matrix
are preserved in [XDNA2 GPU/NPU Interoperability Research](NPU_RESEARCH.md).

The GPU is the protected latency path and the initial decode backend. The NPU is
an optional first-class backend promoted only where it improves a declared
end-to-end workload.

Initial NPU candidates:

- Large and bucketed prefill GEMMs.
- Prefill for one request while the GPU decodes another.
- Batched verification or auxiliary work with stable shapes.
- Coarse fused stages whose handoff cost is small relative to computation.

Initial non-goals:

- Batch-1 token decode on the NPU.
- Per-layer GPU/NPU ping-pong.
- Depending on implicit cross-driver dma-buf fencing.
- Concurrent GPU/NPU writes to one allocation.

### Shared-allocation feasibility

Current XRT and `amdxdna` expose DRM PRIME/dma-buf BO export/import, and HIP can
import Linux external-memory FDs. This makes an XRT-owned BO exported to HIP the
first interoperability experiment. The exact `amdxdna`/XRT to gfx1151/HIP
combination is not yet treated as a supported production contract.

The first probe must:

1. Allocate an XRT BO and export its DRM PRIME FD.
2. Import and map that FD through HIP external memory.
3. Alternate HIP and NPU writes/checks with explicit completion waits and cache
   maintenance where required.
4. Stress multiple sizes and partial ranges for at least 10,000 iterations.
5. Measure GPU-to-NPU and NPU-to-GPU ownership-transition latency without a
   payload copy.
6. Run sustained concurrent GPU/NPU load and detect firmware stalls, resets,
   corruption, and decode-latency regression.

Immutable simultaneous reads may be tested after producer completion. Disjoint
simultaneous writes remain disabled until explicitly proven safe.

### Promotion gates

An NPU or heterogeneous route is promoted only when it:

- Matches CPU and GPU correctness references.
- Improves its declared end-to-end metric by at least 15 percent.
- Regresses protected GPU decode latency by less than 5 percent.
- Includes synchronization, dispatch, padding, memory, and repacking costs.
- Survives cancellation, reset, and sustained concurrent-load testing.

If interoperability or stability fails, serving remains GPU-only. If one-copy
access fails but the NPU is still beneficial, explicit backend views/copies are
accounted and measured rather than hidden.

### Evidence tracked for the feasibility gate

- XRT native BO export/import API:
  `https://xilinx.github.io/XRT/2026.1/html/xrt_native.main.html`
- XDNA shim and PRIME implementation:
  `https://github.com/amd/xdna-driver/blob/main/src/shim/buffer.cpp`
  and `src/shim/host/platform_host.cpp`
- `amdxdna` GEM PRIME implementation:
  `https://github.com/amd/xdna-driver/blob/main/src/driver/amdxdna/amdxdna_gem.c`
- HIP external-memory interoperability:
  `https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html`
- Current concurrent ROCm/XDNA2 stability risk:
  `https://github.com/amd/xdna-driver/issues/1605`
- XDNA2 GEMM evidence and shape/configuration costs:
  `https://arxiv.org/html/2512.13282v1`
- AMD IRON operator and transformer examples:
  `https://github.com/amd/IRON`

## Work Tracking

Architecture, contracts, decisions, and durable rationale live in this
repository. Executable work is represented by bounded GitHub issues with
acceptance criteria. One GitHub Project is the status source of truth and tracks
at least:

- Status.
- Roadmap milestone.
- Priority.
- Backend (`core`, `CPU`, `GPU`, `NPU`, `server`, `tools`).
- Model (`shared`, `Qwen3.5-0.8B`, `Qwen3.8-27B`, later model kinds).
- Dependencies and blockers.

`docs/ROADMAP.md` defines ordering and milestone exit criteria; it is not a
manual duplicate of live issue status.
