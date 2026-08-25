# XDNA2 NPU Backend

Status: implementation in progress, updated 2026-08-20

## Purpose

The XDNA2 backend executes selected model operations on the Strix Halo NPU
without ONNX Runtime, PyTorch, or a Python runtime in the serving process.

The backend is a separate source module with no dependency on HIP headers:

```text
strix
  core/xdna2/
```

The NPU module is statically linked into `strix`. The core runtime may
initialize both GPU and NPU backends.

## Current Status

The Nix package builds reviewed NPU2 programs with pinned MLIR-AIE, LLVM-AIE,
and AIEBU tools. The deterministic XRT smoke validates the runtime lifecycle.
The first model-private program executes Qwen3.8 MTP RMSNorm with the actual
5120-element weight vector: its BF16 result matches the FP64 oracle at
`0.00405` RMSE and `0.999982` cosine similarity, with observed warm commands
between `0.08` and `0.12 ms`. MTP matrix operators and full draft generation
remain GPU-owned.

## Initial Platform

- Linux x86-64.
- Upstream or distribution `amdxdna` kernel driver.
- XRT NPU userspace shim compatible with that driver.
- XDNA2/AIE2P device.
- MLIR-AIE/IRON toolchain for ahead-of-time program generation.

The official Linux driver architecture uses XRT as the open-source userspace
runtime over `amdxdna`. It is the production integration baseline:

https://docs.kernel.org/accel/amdxdna/amdnpu.html

The current `ypapadop-amd/ggml` `hsa-backend` remains an experimental reference
for:

- Discovering AIE2P as an HSA agent.
- Selecting memory pools.
- Building HSA queues and signals.
- Submitting AIE execution packets.
- Loading PDI and instruction artifacts.
- Generating whole-array IRON GEMM programs.

Reference:

https://github.com/ypapadop-amd/ggml/tree/hsa-backend/src/ggml-hsa

Do not make that ggml backend a required runtime dependency. Reuse its proven
integration patterns where useful behind a Strix-owned backend API. Do not make
production NPU support depend on undocumented HSA packets or agent behavior.

## Driver Stack

The intended stack is:

```text
Strix scheduler/model implementation
            |
      Strix XDNA backend
            |
       XRT NPU shim
            |
 amdxdna DRM accel driver
            |
        XDNA2 NPU
```

MLIR-AIE and IRON are build-time dependencies. Python may run build scripts,
but the resulting PDI, instruction streams, metadata, and AIE objects are
loaded by C++ at runtime.

An HSA-backed adapter may be retained behind the same interface for research
and comparison. It is not promoted to a production route until it passes the
same allocation, synchronization, reset, performance, and compatibility tests
as XRT.

Raw driver ioctls are a last resort and must be isolated in one translation
unit. The rest of the runtime uses Strix-owned context, buffer, command, and
completion abstractions rather than XRT or HSA types.

## Backend Objects

```cpp
class XdnaDevice;
class XdnaContext;
class XdnaCommandQueue;
class XdnaProgram;
class XdnaBuffer;
class XdnaCompletion;
class XdnaExecution;
```

Responsibilities:

### `XdnaDevice`

- Enumerate and identify the XDNA2/AIE2P device.
- Query tile-array dimensions and supported program ABI.
- Query supported buffer allocation and import capabilities.
- Report driver, firmware, runtime, and compiler compatibility IDs.
- Expose health and reset diagnostics.

### `XdnaContext`

- Own one `amdxdna` workload context through the selected userspace adapter.
- Request a program-compatible number of AIE columns.
- Record the columns and resources actually assigned by the driver.
- Own the overlay, instruction buffer, and command resources for that context.
- Remain reconstructible after reset, suspend, or firmware restart.

### `XdnaCommandQueue`

- Own command-buffer submission for one workload context.
- Reserve, populate, and submit commands safely.
- Associate completion objects with executions.
- Report command stalls, driver errors, and timeouts.

### `XdnaProgram`

- Own the loaded PDI and instruction sequence.
- Validate architecture, compiler version, model implementation, and tensor
  contract.
- Expose fixed argument slots and supported shape buckets.
- Remain immutable after load.

### `XdnaBuffer`

- Wrap an allocation and its permitted CPU, GPU, and NPU access paths.
- Record size, alignment, cache policy, and ownership.
- Permit explicit import from the shared allocation broker.
- Never assume that an arbitrary HIP pointer is NPU-accessible.

### `XdnaCompletion`

- Wrap backend completion and dependency primitives.
- Use release/acquire semantics at GPU/NPU and CPU boundaries.
- Support bounded waits and timeout diagnostics.

## Shared Allocation Integration

The GPU, NPU, and CPU use physically unified memory, but pointer visibility,
cache coherency, and synchronization must still be established explicitly.

The first driver milestone must prove:

1. Allocate or import a buffer through a supported XRT/driver path.
2. Establish CPU, GPU, and XDNA access where supported.
3. Write it from the CPU and read it from both devices.
4. Write disjoint regions from GPU and NPU.
5. Synchronize with explicit release/acquire completion primitives.
6. Verify cache visibility and content without an implicit full-device sync.

The allocation broker records:

```text
allocation ID
owning pool
permitted agents
cache policy
last writer
last release signal
size and alignment
```

If one optimal common allocation is not available, use explicit backend
buffers and bounded asynchronous copies. Do not conceal copies inside a kernel
wrapper because the scheduler must account for their wall time.

## Driver Resource Management

The XDNA2 array is partitioned at column boundaries. The `amdxdna` resource
solver may allocate spatial partitions or time-share columns between workload
contexts and other processes.

Consequences for Strix:

- A program declares its required column count in immutable metadata.
- Context creation may receive fewer available resources than the physical
  array suggests or may wait behind another workload.
- Route and performance keys include the columns actually assigned.
- The scheduler does not assume exclusive NPU ownership.
- A route that misses latency targets under contention is disabled or shifted
  to the GPU without corrupting request state.
- Context creation and destruction stay outside latency-critical request paths.

Current upstream documentation describes a 4 x 8 Strix-class array, up to 16
concurrent workload contexts, and a 64 MiB host-resident instruction buffer per
context. Treat these as capability values queried and validated at startup,
not constants embedded in memory accounting.

The allocation broker accounts for:

- Instruction buffers.
- Overlay and command buffers.
- Driver-internal and shared allocations.
- Requested and assigned columns.
- Context and mailbox limits.
- Memory retained during recovery or replacement.

## Program Trust

A model weight artifact is data only. It cannot provide an NPU overlay,
`ctrlcode`, instruction stream, or compiler object.

All executable NPU artifacts are:

- Built with the server from reviewed source.
- Bound to a compiled model kind and program ABI.
- Content-hashed and recorded in server build metadata.
- Checked against the detected architecture, driver, and firmware contract
  before context creation.

The server refuses an unknown or externally substituted executable artifact.

## Program Build Pipeline

```text
models/<model>/npu/aie2p/ microkernel
                  |
         AIE compiler object
                  |
IRON/MLIR-AIE array and DMA design
                  |
    PDI + instruction stream + metadata
                  |
        embedded server resource
```

Programs are compiled ahead of time for fixed shape families:

- Tensor dtype and layout.
- M, N, and K tile class.
- Number of AIE columns.
- Output dtype.
- Group size and scale epilogue.
- Model-specific fused operations.

JIT compilation is a development feature only. Production builds contain
content-addressed artifacts embedded in `strix` and built by a pinned
compiler toolchain.

## Model Isolation

Each model owns its complete AIE2P source, IRON array design, DMA schedule,
overlay, `ctrlcode`, program metadata, and shape routing:

```text
models/<model>/npu/aie2p/
```

There is no shared production numerical AIE program catalog. Common build
scripts may invoke compilers and package artifacts, but they do not generate
one numerical program consumed by several models.

When an AIE program is adapted for another model, copy it into that model and
give it an independent content hash, tuning record, and regression suite. A
change to one model must leave every other model's embedded AIE program hashes
unchanged.

Different quantized sizes of the same model reuse its private AIE program set
when their tensors use supported SHQ4-T16, SHQ8-T16, or BF16 encodings.

## Initial Feasibility Priorities

The supporting API analysis, measurements, stability reports, and required
probe matrix are recorded in [XDNA2 GPU/NPU Interoperability
Research](NPU_RESEARCH.md).

Current XRT and `amdxdna` expose DRM PRIME/dma-buf BO sharing, while HIP exposes
Linux external-memory import. The first interoperability probe therefore uses
an XRT-owned BO exported to HIP. This exact cross-driver combination is not a
production guarantee until it passes access, explicit-ordering, cache-
visibility, partial-range, and sustained-load tests on gfx1151/XDNA2.

Initial synchronization is host-mediated and explicit. Do not depend on
implicit dma-buf fences between amdgpu and `amdxdna`. Do not enable concurrent
writes, even to disjoint ranges, in the first implementation.

Public XDNA2 GEMM evidence favors large, high-arithmetic-intensity INT8 and BF16
shapes. It also shows meaningful configuration and DMA-scheduling costs. The
backend therefore compiles a small set of reusable shape buckets, overlaps DMA
with compute, measures GEMV separately, and treats large prefill as the first
model workload. Batch-1 decode remains GPU-owned unless later measurements
reverse this decision.

Sustained concurrent ROCm+XDNA2 testing is mandatory because current field
reports include firmware command timeouts under combined load. A failed or
unstable probe selects GPU-only operation rather than reducing server
readiness.

## First Kernel Family

The first custom quantized numerical kernel is mixed W4A8 GEMM:

```text
A: dynamic INT8 rows
B: SHQ4-T16 UINT4 or signed INT4 weights
C: INT32 partial accumulators
Epilogue: zero correction, activation scale, BF16 weight scale, output convert
```

The AIE API exposes mixed INT8 x INT4/UINT4 matrix multiplication. For AIE2P the
native mixed microtile is `M=4, K=16, N=16`.

Reference:

https://github.com/Xilinx/aie_api/blob/main/include/aie_api/detail/aie2p/mmul_8_4.hpp

The current ggml-HSA GEMM interface accepts one shared input dtype and therefore
does not expose this path. Strix must provide separate A and B types, a custom
microkernel, and a fused group-scale epilogue.

## Work Suited to the NPU

- Large prompt prefill.
- Batched prompt chunks.
- Batched speculative target verification.
- Batched MTP or draft-model work across requests.
- Dense model fragments with stable shapes.
- Grouped MoE experts after rows are sorted and compacted.
- Background work that does not delay latency-critical GPU decode.

Poor initial candidates:

- Single-token full-model decode.
- Tiny dynamic kernels with many host decisions.
- Sampling and grammar state machines.
- Work requiring a device synchronization after every small operation.

## Shape Buckets

Compile and benchmark fixed physical widths such as:

```text
M = 4, 8, 16, 32, 64, 128, 256, ...
```

Logical rows may be mapped into a physical bucket with an active-row mask. The
scheduler must account for padding cost and choose GPU execution when the NPU
bucket is underfilled.

Programs should use the full 4 x 8 array only when the shape provides enough
work. Smaller column counts may win for narrow tensors or reduce DMA pressure.

## Error Handling

- Every submission has a finite timeout.
- Command and program errors include the execution ID and program content hash.
- A timed-out context is quarantined before reset.
- Requests assigned to a failed NPU execution are retried on the GPU only when
  scheduler state has not been committed.
- Repeated driver failures disable the NPU route and keep GPU-only serving
  available.
- System suspend powers down the NPU and invalidates runtime contexts. Resume
  recreates contexts, reloads programs, reimports buffers, and reruns visibility
  and numerical smoke tests before NPU routing is restored.

## Tests

- Driver and agent discovery.
- Program ABI and architecture rejection.
- Shared-buffer visibility and ordering.
- Command-queue reuse and concurrent submission.
- Resource-solver behavior with reduced assigned columns.
- Multi-process NPU contention.
- Timeout and reset recovery.
- Suspend/resume context and buffer reconstruction.
- W4A8 microkernel against a wide CPU oracle.
- Scale and zero-point epilogue.
- Shape, padding, and active-row masks.
- NPU-only full layer and model-logit comparisons.
- NPU versus GPU operator comparisons.
- Repeated graph/program execution without allocation growth.
- Driver, firmware, and compiler compatibility matrix.
