# XDNA2 GPU/NPU Interoperability Research

Status: research record, 2026-08-17

This document preserves the evidence behind Gufo's XDNA2 interoperability and
routing decisions. It is not a claim that every referenced API combination is
supported on every ROCm, XRT, firmware, or kernel revision. Production support
is established only by the probes and promotion gates below on the pinned
Strix Halo platform.

## Executive Summary

| Question | Current evidence |
| --- | --- |
| Can HIP and XRT access one physical allocation? | The Linux plumbing exists through XRT BO export, DRM PRIME/dma-buf, `amdxdna` GEM import/export, and HIP external-memory import. No AMD-supported end-to-end guarantee for XRT/`amdxdna` dma-buf to HIP/gfx1151 was found. Prototype it before depending on it. |
| What synchronization is required? | Use explicit producer completion and consumer submission. `xrt::bo::sync()` is cache/visibility maintenance on the current XDNA shim, not proof of execution ordering and not inherently a payload copy. Do not assume implicit amdgpu-to-`amdxdna` dma-buf fencing. |
| Can concurrent GPU and NPU execution help? | Concurrent execution is physically demonstrated and one community report observes roughly 1.7–1.9x aggregate throughput for independent work. A current open `amdxdna` report also shows firmware stalls under concurrent ROCm and XDNA2 load. Benefit and stability are workload- and stack-specific. |
| What work suits XDNA2? | Large, high-arithmetic-intensity GEMM—especially INT8 and BF16—has the strongest evidence. Large/bucketed prefill is a better initial target than batch-1 decode/GEMV. Reuse configurations and overlap DMA with compute. |

## 1. Shared Physical Allocations

### XRT and `amdxdna`

XRT exposes `xrt::bo::export_buffer()` and import of an exported BO. The Linux
XDNA shim exports a BO as a file descriptor. Its host platform implementation
uses:

```text
DRM_IOCTL_PRIME_HANDLE_TO_FD
DRM_IOCTL_PRIME_FD_TO_HANDLE
```

The kernel driver's GEM PRIME paths export and import dma-bufs. Imported GEM
objects use the dma-buf reservation object, so this is actual DRM PRIME sharing,
not merely two devices using the same physical LPDDR.

Relevant implementation sources:

- XRT native BO API: <https://xilinx.github.io/XRT/2026.1/html/xrt_native.main.html>
- XDNA shim BO sharing: <https://github.com/amd/xdna-driver/blob/main/src/shim/buffer.cpp>
- XDNA host PRIME ioctls: <https://github.com/amd/xdna-driver/blob/main/src/shim/host/platform_host.cpp>
- `amdxdna` GEM PRIME paths: <https://github.com/amd/xdna-driver/blob/main/src/driver/amdxdna/amdxdna_gem.c>

### HIP

HIP supports importing external memory from a Linux FD and mapping it into GPU
virtual address space. AMD's Vulkan/HIP documentation describes both APIs as
representing the same physical memory and explicitly requires cross-API
synchronization. HIP also exposes `hipDeviceAttributeDmaBufSupported`.

Relevant sources:

- HIP external-resource interoperability:
  <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/external_interop.html>
- HIP external-memory implementation:
  <https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/hipamd/src/hip_memory.cpp>
- HIP runtime API and dma-buf capability attribute:
  <https://github.com/ROCm/HIP/blob/develop/include/hip/hip_runtime_api.h>

### Missing guarantee

No AMD document or supported sample was found that explicitly guarantees:

```text
XRT/amdxdna PRIME FD -> hipExternalMemoryHandleTypeOpaqueFd -> gfx1151
```

The interfaces align, but that exact cross-driver combination remains a
platform capability to prove. The first Gufo experiment therefore uses:

```text
XRT BO
  -> xrt::bo::export_buffer()
  -> DRM PRIME/dma-buf FD
  -> hipImportExternalMemory()
  -> hipExternalMemoryGetMappedBuffer()
  -> gfx1151 pointer
```

The public XRT export-handle representation is implementation-specific. Gufo
must isolate Linux-FD extraction behind a version-checked adapter rather than
exposing that assumption throughout the runtime.

## 2. Ordering and Visibility

Execution ordering and memory visibility are separate requirements:

- Ordering answers whether the producer has finished.
- Visibility answers whether the consumer can observe the producer's writes.

On the current XDNA shim, `xrt::bo::sync()`:

1. Is a no-op on a cache-coherent path.
2. Uses a driver BO-sync ioctl when driver synchronization is selected.
3. Otherwise performs host cache flushing.

This supports treating BO sync as cache/visibility maintenance rather than an
implicit staging-buffer copy. It does not establish that a dependent HIP or XRT
command waits for the other device.

`amdxdna` uses dma reservation objects, job fences, DRM sync objects, and
timeline points. The research did not establish a complete automatic chain in
which amdgpu publishes an implicit fence and `amdxdna` reliably waits on it, or
the reverse. Implicit cross-driver fencing is therefore not part of Gufo's
initial correctness model.

The initial handoff protocol is:

```text
GPU -> NPU
HIP producer
  -> HIP stream/event completion
  -> required BO/cache visibility operation
  -> XRT submit

NPU -> GPU
XRT producer
  -> XRT completion wait
  -> required BO/cache visibility operation
  -> HIP submit
```

Host participation in scheduling is acceptable for feasibility. The handoff
must not copy tensor payload bytes unless the runtime explicitly selects a
fallback interoperability tier.

Initial access policy:

| Access pattern | Initial policy |
| --- | --- |
| GPU and NPU read immutable data | Test after producer completion; promote if stable |
| GPU writes, NPU reads | Explicit handoff |
| NPU writes, GPU reads | Explicit handoff |
| Both write the same cache lines | Forbidden |
| Both write disjoint subranges | Disabled initially; whole-BO fencing/cache behavior is insufficiently proven |
| CPU accesses during accelerator writes | Forbidden without explicit synchronization |

Driver synchronization evidence:

- XDNA command/fence handling:
  <https://github.com/amd/xdna-driver/blob/main/src/driver/amdxdna/amdxdna_ctx.c>

## 3. Concurrent GPU and NPU Execution

### Positive evidence

The community `gufo-Linux-NPU-Concurrency` experiment runs an NPU probe
while a ROCm llama.cpp workload keeps the iGPU near 87 percent utilization. It
reports roughly 1.7–1.9x aggregate throughput compared with fully sequential
multi-tier serving.

This is useful evidence that physical concurrency can work. It is not an AMD
benchmark and does not prove that splitting one LLM across devices gives the
same improvement.

Source:

- <https://github.com/LucRoot/gufo-Linux-NPU-Concurrency/blob/main/README.md>

### Stability risk

As of this research date, `amd/xdna-driver#1605` reports reproducible NPU
firmware command timeouts and process failures after approximately one to five
minutes while a Strix Halo system concurrently runs a ROCm/HIP llama.cpp GPU
workload and FastFlowLM on XDNA2. The reporter observed previous concurrency
with a Vulkan GPU path, but had not completed a controlled Vulkan-versus-ROCm
comparison. No maintainer conclusion had established root cause or declared the
combination unsupported.

Source:

- <https://github.com/amd/xdna-driver/issues/1605>

Engineering conclusion:

- Concurrent execution is demonstrated.
- A throughput gain is plausible but not guaranteed.
- Production robustness under saturated ROCm plus XDNA2 load is not yet proven.
- GPU-only serving must remain a complete fallback.

## 4. XDNA2 Workload Evidence

The paper "Striking the Balance: GEMM Performance Optimization Across
Generations of Ryzen AI NPUs" reports end-to-end XDNA2 GEMM results that include
OS and dispatch overhead:

| GEMM type | Reported XDNA2 peak |
| --- | ---: |
| INT8 to INT8 | 38.05 TOPS |
| INT8 to INT16 | 31.52 TOPS |
| INT8 to INT32 | 25.31 TOPS |
| BF16 to BF16 | 14.71 TOPS |

Source:

- <https://arxiv.org/html/2512.13282v1>

### Arithmetic intensity

Small, low-arithmetic-intensity GEMMs are strongly memory-bound. Performance
improves as arithmetic intensity grows. This maps naturally to transformer
execution:

- Prefill has multiple rows and large matrix multiplications, making it a
  plausible NPU route.
- Batch-1 autoregressive decode is GEMV-like and is unlikely to approach the
  published GEMM peaks.

IRON contains GEMM, GEMV, attention, RMSNorm, RoPE, activation,
dequantization, and transformer examples, but this research did not find a
public batch-1 GEMV sweep comparable to the GEMM results.

### Layout

The GEMM study used row-major A and compared row- and column-major B. Reported
average improvements for column-major B were approximately:

| Type | Improvement |
| --- | ---: |
| INT8 to INT8 | 19.1% |
| INT8 to INT16 | 25.2% |
| BF16 to BF16 | 8.7% |

The implementation uses multidimensional DMA addressing to transform normal
DRAM layouts into AIE-consumable tiles. Gufo should therefore measure both its
canonical T16 representation and a lossless backend view. It should not assume
that source tensors must be permanently pre-tiled in an AIE-only layout.

### Configuration reuse

The study reports roughly 4.9 ms to reconfigure a complete XDNA2 GEMM design.
A roughly 4K-square INT8-to-INT16 GEMM in the cited comparison took about
5.2 ms. Reconfiguration can therefore cost as much as the operation itself.

Consequences:

- Compile a small set of reusable physical shape buckets.
- Reuse loaded programs.
- Pad or mask irregular logical rows when the measured cost is acceptable.
- Do not generate or reconfigure an NPU design for every request shape.

### DMA/compute overlap and memory bandwidth

The study reports an approximately 28 percent performance drop for one
roughly-4K INT8-to-INT16 GEMM when descriptor/data movement was not properly
overlapped with compute. The execution model must pipeline DMA and AIE work
rather than serialize input movement, computation, and output movement.

The study also measures roughly 50 GB/s effective DRAM bandwidth available to
XDNA2 for its GEMM transfers. This is not total Strix Halo LPDDR bandwidth, but
it shows that an active NPU can materially contend with bandwidth-sensitive GPU
decode.

### Types and operator ecosystem

Current Ryzen AI documentation includes BF16 and INT8-oriented conventions such
as XINT8, A8W8, and A16W8. IRON's operator library is heavily BF16-oriented for
transformer operators and includes AIE2/AIE2P primitives relevant to a native
runtime.

Sources:

- Ryzen AI quantization:
  <https://ryzenai.docs.amd.com/en/latest/model_quantization.html>
- AMD IRON:
  <https://github.com/amd/IRON>
- IRON README and operator inventory:
  <https://github.com/amd/IRON/blob/devel/README.md>

## 5. Routing Implications for Gufo

The initial protected route is:

```text
GPU: batch-1 decode and latency-critical work
NPU: large/bucketed prefill or independent stable-shape work
```

The most promising first concurrency case is:

```text
NPU: prefill request A
GPU: decode request B
CPU: tokenize or stream request C
```

A later NPU-prefill-to-GPU-decode transition requires compatible KV and Gated
DeltaNet state representations, explicit ownership transfer, and end-to-end
measurement. Avoid initially:

- Alternating devices every decoder layer.
- Sending tiny operators across a device boundary.
- Running NPU work solely to increase nominal utilization.
- Maintaining an unaccounted second weight layout.

The NPU may also be useful for batched verification, MTP/draft work across
requests, vision encoding in a future multimodal product, or independent
auxiliary models. Each is a separate measured route.

## 6. Required Probe Matrix

### Shared allocation correctness

1. Query `hipDeviceAttributeDmaBufSupported`.
2. Allocate an XRT BO and export its PRIME FD.
3. Import and map it through HIP external memory.
4. HIP writes pattern A; explicitly synchronize; NPU verifies A.
5. NPU writes pattern B; explicitly synchronize; HIP verifies B.
6. Repeat at 4 KiB, several MiB, and tens or hundreds of MiB.
7. Include partial-range writes and offsets.
8. Run at least 10,000 iterations initially and a longer soak before promotion.
9. Verify destruction, reset, and error paths do not leave stale mappings.

### Handoff cost

Measure separately:

- HIP completion cost.
- XRT completion cost.
- Cache/BO synchronization cost.
- GPU-to-NPU ownership transition.
- NPU-to-GPU ownership transition.
- Explicit-copy fallback cost.

### Performance matrix

Measure:

1. GPU-only.
2. NPU-only.
3. Independent GPU plus NPU.
4. Dependency-coupled GPU-to-NPU over the shared BO.
5. Dependency-coupled NPU-to-GPU over the shared BO.

For each configuration record:

- End-to-end throughput.
- TTFT and inter-token p50/p95/p99 latency.
- Actual GPU/NPU activity and memory bandwidth.
- Power steady state.
- Dispatch and ownership-transition counts.
- Firmware timeouts, resets, corruption, and recovery.

### NPU numerical/performance matrix

- INT8 and BF16 large GEMM.
- W4A8 and W8A8 candidate kernels.
- Batch-1 GEMV measured independently.
- Row-major and losslessly repacked/column-major weight views.
- Shape-bucket reuse versus reconfiguration.
- Serialized versus overlapped DMA/compute.
- Assigned AIE column counts and contention.

## 7. Promotion Policy

A route is promoted only when it:

- Matches CPU and single-backend numerical references.
- Improves its declared end-to-end metric by at least 15 percent.
- Regresses protected GPU decode latency by less than 5 percent.
- Includes dispatch, synchronization, cache maintenance, padding, repacking,
  and duplicate-memory costs.
- Survives cancellation, reset, suspend/resume, and a sustained concurrency
  soak.
- Falls back transactionally to GPU-only operation.

A microkernel TOPS result, successful dma-buf import, or simultaneous device
activity is necessary evidence for some routes but is never sufficient by
itself.
