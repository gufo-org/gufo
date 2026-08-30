# GPU and NPU Execution

Status: design draft, 2026-08-11

## Purpose

The heterogeneous runtime uses gfx1151 and XDNA2 together only when doing so
improves a declared objective:

- Single-request latency.
- Time to first token.
- Aggregate token throughput.
- Admission capacity.
- Energy per completed request.

The GPU and NPU share the same external memory bandwidth. Their peak compute
figures must never be added without a workload-specific roofline and an
end-to-end measurement.

The full evidence record is in [XDNA2 GPU/NPU Interoperability
Research](NPU_RESEARCH.md).

Current evidence makes large, reusable GEMM shapes the first NPU target and
batch-1 GEMV/decode a poor initial target. The primary concurrency hypothesis is
NPU prefill for one request while the GPU protects decode for another. Per-layer
GPU/NPU ping-pong is not an initial route.

The exact XRT/`amdxdna` dma-buf to HIP/gfx1151 path is plausible from current
kernel and runtime APIs but is not treated as a supported contract until the
feasibility probes in this document pass. Current ROCm+XDNA2 firmware-timeout
reports, including `amd/xdna-driver#1605`, require a sustained concurrency soak
before production promotion.

## Execution Routes

Every schedulable operation selects one route:

```text
GPU_ONLY
NPU_ONLY
SPLIT_N
SPLIT_K_REDUCE
EXPERT_PARALLEL
REQUEST_PIPELINE
HEDGED_PROPOSAL
```

The route is selected from a measured table keyed by:

```text
model
operator/fusion
quantization
M, N, K
logical and physical rows
context range
expert occupancy
request concurrency
backend load
```

Static heuristics provide safe defaults. An autotuner may generate the table,
but production serving does not benchmark untrusted requests.

## Shared Weights

For a selected artifact, GPU and NPU use the same tensor-encoding table,
quantized codes, scales, and zero points and, initially, one resident T16
weight layout.

The runtime may later keep a losslessly repacked backend copy for a measured hot
tensor when:

- The memory budget permits it.
- The improvement is material end to end.
- The duplicate is visible in memory accounting.
- The canonical copy remains the correctness source.

Within one artifact, different numerical quantizations for the same tensor are
avoided because they complicate split-operator equivalence and quality
evaluation. Separately published size variants may quantize that tensor
differently.

## Route Guidance

| Work | Initial route |
| --- | --- |
| Single-request decode | GPU only |
| Large prompt prefill | NPU or GPU+NPU |
| Small prompt prefill | GPU |
| Large dense batched verification | NPU or split |
| Small speculative verification | GPU |
| Batched draft/MTP heads | NPU candidate |
| **DFlash-2 Block Diffusion Drafter** | **NPU (XDNA2 via XRT unified DMA)** |
| MoE populated expert groups | Expert parallel |
| Sampling and grammar | GPU/CPU contract |
| Disk persistence | Background CPU and copy queue |

### Heterogeneous Speculative Drafting: DFlash-2 on XDNA2 NPU

In standard speculative decoding on Strix Halo, running both the target model (e.g. Qwen3.8-27B) and the draft model on the GPU causes contention for the unified 270 GB/s DRAM bus. Offloading the **DFlash-2 non-causal block diffusion drafter to the 50 TOPS XDNA2 NPU** resolves this bottleneck:

1. **Zero-Copy Feature Priming**:
   The GPU writes multi-layer feature representations (e.g., hidden states tapped at layers 16, 32, 48, 64) directly into an `xrt::bo` unified DMA buffer accessible by the NPU.
2. **Concurrent Block Prediction**:
   While the GPU processes prompt prefill or verifies prior tokens, the XDNA2 NPU executes parallel diffusion iterations on its dedicated compute array, generating a $K$-token draft proposal block (typically 5–7 tokens) in a single non-causal forward pass.
3. **Parallel Verification**:
   The GPU ingests the proposed draft block from the shared DMA buffer and verifies all $K$ tokens in parallel in a single matrix-vector pass.

## Tensor Parallel Routes

### `SPLIT_N`

Split output channels. Each device reads the shared activation and disjoint
weight tiles, then writes disjoint output ranges.

Best suited to:

- Q, K, and V projections.
- Gate and up projections.
- Independent attention heads.
- Output heads when only a bounded result is required from each shard.

No numerical reduction is required, but a completion barrier is required before
the consumer reads the full output.

### `SPLIT_K_REDUCE`

Split the reduction dimension. Each device produces a partial output, followed
by an FP32 or defined BF16 reduction.

Best suited to:

- Down projections after split FFN channels.
- Attention output projection after head splitting.
- Large tensors where the reduction cost is small relative to GEMM.

The reduction and synchronization cost must be included in route selection.

### `EXPERT_PARALLEL`

Group `(request, row, expert)` assignments across active requests. Send
sufficiently populated groups to the NPU and irregular or latency-critical
tails to the GPU.

This is preferred over splitting every expert matrix because:

- Experts are naturally independent.
- Row compaction creates better NPU shapes.
- Device ownership can be adjusted per scheduling step.
- Only the weighted combination crosses the device boundary.

## Request Pipeline

Independent requests can overlap:

```text
NPU: prefill or proposal for request A
GPU: verify or decode request B
CPU: tokenize or stream request C
```

This improves aggregate throughput, not necessarily latency for one request.
Decode protection limits NPU work if concurrent memory traffic raises GPU
inter-token latency beyond policy.

## Speculative Decoding

Use one provider-neutral cycle:

```text
PROPOSE -> VERIFY -> ACCEPT -> COMMIT -> UPDATE
```

Suggested device ownership:

| Scenario | Proposal | Verification |
| --- | --- | --- |
| One request, tiny MTP head | GPU | GPU |
| Multiple MTP requests | NPU batch | GPU batch |
| Dense draft model | NPU candidate | GPU |
| Large combined verifier batch | GPU or NPU | NPU or split |
| DeepSeek DSpark | NPU support-model candidate | GPU target |

Candidate generation and verification for one request are causally dependent.
Pipeline overlap is therefore mainly across requests.

`HEDGED_PROPOSAL` optionally computes the ordinary next target token on the GPU
while the NPU proposes candidates. It reduces low-acceptance latency at the cost
of duplicated work and is enabled only by measured policy.

Accept, selected-state copy, KV commit, rollback, and cursors should remain on
device. Only a bounded result is returned to the scheduler.

## Shared Memory and Synchronization

Unified physical memory does not remove the need for:

- Agent access permissions.
- Explicit release/acquire ordering.
- Cache visibility tests.
- Stable buffer lifetimes.
- Scheduler-visible ownership.

The first Tier-A experiment uses an XRT-owned BO exported as a DRM
PRIME/dma-buf FD and imported through HIP external memory. Public support for
this exact XDNA2-to-gfx1151 combination is not assumed merely because both API
halves exist.

The allocation broker is backend-neutral. GPU and NPU executions depend on
opaque completion tokens translated into native XRT or HIP completion
primitives by the backend. Initial correctness uses explicit host-mediated
handoffs:

```text
HIP producer -> HIP completion wait -> required BO/cache sync -> XRT submit
XRT producer -> XRT completion wait -> required BO/cache sync -> HIP submit
```

`xrt::bo::sync()` is treated as cache/visibility maintenance where required,
not as proof of execution ordering. Implicit amdgpu-to-amdxdna dma-buf fencing
is not part of the correctness contract until independently demonstrated.

Immutable simultaneous reads may be enabled after producer completion. Same-
range writes are forbidden, and disjoint simultaneous writes remain disabled
initially because reservation/fence and cache behavior may operate at whole-BO
granularity.

Avoid a CPU synchronization between every layer. A heterogeneous route should
submit a complete operator, layer group, prefill stage, or speculative stage
before crossing devices.

## Bandwidth Policy

The scheduler maintains a coarse bandwidth-pressure state:

```text
LOW
MODERATE
SATURATED
```

Inputs include:

- Recent GPU decode bandwidth and inter-token latency.
- NPU DMA time.
- Concurrent prefill and restore traffic.
- Active model weight footprint.
- KV attention traffic at current contexts.

Under `SATURATED`, latency-critical decode owns the memory budget and background
NPU work is delayed or reduced.

## Promotion Requirements

A heterogeneous route must demonstrate:

- Correct operator outputs against CPU and single-backend references.
- Full-model quality within the released model's thresholds.
- At least 15 percent improvement in its declared end-to-end metric.
- Less than 5 percent regression in protected decode latency.
- Stable behavior under cancellation and backend failure.
- Explicit memory, synchronization, and padding accounting.

Microkernel speedup alone is insufficient.

## Tests

- Alternating shared-buffer HIP-write/NPU-read and NPU-write/HIP-read stress at
  multiple sizes and partial ranges.
- Explicit GPU/NPU completion ordering and cache visibility.
- Immutable simultaneous reads; verify that concurrent writes remain rejected
  until a separate safety gate passes.
- Sustained ROCm+XDNA2 concurrency, firmware timeout, reset, and recovery.
- Split-N versus unsplit output.
- Split-K reduction accuracy.
- Expert routing and weighted combination.
- Concurrent prefill/decode pressure.
- Request pipeline fairness.
- Speculative accept and commit agreement.
- NPU failure before and after transaction commit.
- Autotune table compatibility and fallback behavior.
