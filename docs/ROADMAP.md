# Implementation Roadmap

Status: active roadmap, updated 2026-08-17

Live decisions and the current implementation boundary are recorded in
[Project Status and Decisions](PROJECT_STATUS.md). GitHub issues and one GitHub
Project track execution; this document defines ordering and exit criteria.

## Objective

Build Strix-Halo.cpp incrementally from a deterministic, testable vertical slice.
Do not begin with the complete server, continuous batching, and heterogeneous
execution simultaneously.

Qwen3.5-0.8B is the rapid-iteration and correctness fixture. Qwen3.8-27B is the
first production model. Both use curated compiled model kinds over shared
runtime primitives; model dimensions, kernels, tuning, and acceptance records
remain explicit rather than inferred by a general architecture loader.

The first useful system is:

```text
safetensors source
    -> validated text-only conversion
    -> compiled Qwen3.5/Qwen3.8 implementation family
    -> GPU-only greedy inference
    -> terminal prompt
    -> exact-token, full-logit, and performance report
```

This path establishes the model contract, tensor orientation, tokenizer, KV
state, numerical oracle, kernel interfaces, and performance baseline required
by later work.

An independent early NPU feasibility track validates XRT, AIE programs,
XRT-BO/dma-buf import into HIP, explicit ownership transfer, and sustained
concurrent operation before the common weight layout is frozen. Its evidence
and probe matrix are recorded in [NPU_RESEARCH.md](NPU_RESEARCH.md).

## Delivery Rules

- Use Qwen3.5-0.8B for fast iteration and Qwen3.8-27B for production acceptance.
- Make single-request direct greedy execution correct before adding sampling,
  interactive chat, or HTTP.
- Make the GPU path correct before depending on the NPU.
- Establish CPU and full-quality oracles before optimizing kernels.
- Keep numerical source and tuning private to each curated model kind.
- Treat current SHQ bytes as a candidate contract; freeze v1 only after CPU,
  GPU, AIE, and malformed-artifact conformance gates pass.
- Retain a performance change only when the end-to-end workload improves.
- Every milestone leaves the repository in a runnable and testable state.

## Milestone 0: Repository and Toolchain

### Tasks

1. Initialize the repository and add the complete MIT `LICENSE`.
2. Add `NOTICE` and an initial `THIRD_PARTY_NOTICES.md`.
3. Create the C++20 CMake project and top-level source layout.
4. Add development, release, sanitizer, and test build presets.
5. Pin the supported Linux, compiler, ROCm, XRT, `amdxdna`, firmware, and AIE
   toolchain versions.
6. Add formatting, static analysis, dependency inventory, and documentation
   checks.
7. Create the initial PR test command.

### Initial layout

```text
src/
  core/
  server/
  cli/
models/
  qwen35_08b/
    cpu/
    gpu/gfx1151/
    npu/aie2p/
  qwen38_27b/
    cpu/
    gpu/gfx1151/
    npu/aie2p/
tests/
tools/
docs/
```

### Exit criteria

- A clean checkout configures and builds on the supported Strix Halo Linux
  environment.
- A placeholder `strix --version` runs.
- The PR test command succeeds from one documented entry point.

## Milestone 1: Hardware Diagnostics and Baselines

### Tasks

1. Implement `strix diagnose`.
2. Record CPU, memory, GPU, NPU, driver, runtime, firmware, clock, and power
   information.
3. Measure sustained CPU, GPU, and NPU-visible memory bandwidth.
4. Add HIP allocation, copy, launch, event, and graph smoke tests.
5. Add XRT device discovery, context, buffer, command, and completion smoke
   tests.
6. Measure first-touch, page-fault, and prefault behavior.
7. Produce a structured machine fingerprint used by every benchmark artifact.

### Exit criteria

- Diagnostics identify `gfx1151` and XDNA2 correctly.
- HIP and XRT can each execute a deterministic device program repeatedly.
- Baseline bandwidth reports are retained.
- Unsupported driver or firmware combinations fail with actionable errors.

## Milestone 2: Test Oracles and Model Contracts

### Tasks

1. Implement the common tensor descriptor and checked shape/byte arithmetic.
2. Implement safetensors index and shard inspection.
3. Add the compiled `ModelKind` registry with Qwen as the first implementation.
4. Define Qwen tensor names, orientations, dimensions, RoPE, tokenizer, and
   state contracts.
5. Implement or integrate the exact Qwen tokenizer and chat template.
6. Create small high-precision CPU operator oracles.
7. Build the full-quality teacher-logit capture workflow.
8. Define the versioned logit artifact and matched-token comparison runner.
9. Add the initial SHQ4-T16, SHQ6-T16, and SHQ8-T16 byte-exact conformance
   vectors.
10. Commit the capability evaluation suite with provenance and extraction
    fixtures (EVAL.md).

### Exit criteria

- Safetensors inventory is deterministic and rejects malformed inputs.
- Tokenization matches the pinned source tokenizer on the committed corpus.
- CPU quantization/dequantization vectors are byte exact.
- Teacher-forced full-vocabulary logits can be captured and compared.
- Tests distinguish source dtype from accumulation dtype.

## Milestone 3: Early GPU and NPU Format Feasibility

This milestone has two parallel engineering tracks. It does not yet implement a
complete model.

### GPU track

1. Implement model-private Qwen SHQ4-T16 decode GEMV.
2. Implement model-private Qwen SHQ8-T16 decode GEMV.
3. Implement small-row and prefill candidates for Q4 and Q8.
4. Compare the common T16 layout with a GPU-native repacked layout.
5. Record ISA, occupancy, bandwidth, alignment, and tail behavior.

### NPU track

1. Compile and execute a minimal model-private AIE2P program through XRT.
2. Implement W4A8, W8A8, and BF16 GEMM microbenchmarks.
3. Measure large/high-arithmetic-intensity GEMM and batch-1 GEMV separately;
   do not infer decode performance from GEMM TOPS.
4. Validate the T16 microtile ordering and scale epilogues.
5. Test a small set of reusable row/column shape buckets and record the cost of
   program/configuration changes.
6. Measure context creation, program load, command submission, DMA overlap, and
   completion overhead.

### Shared-allocation track

1. Allocate an XRT BO, export its DRM PRIME/dma-buf FD, and attempt import and
   mapping through HIP external memory on gfx1151.
2. Alternate HIP and NPU producer/consumer access with explicit completion waits
   and required cache maintenance; do not depend on implicit dma-buf fencing.
3. Stress full- and partial-range handoffs over multiple allocation sizes for at
   least 10,000 iterations.
4. Measure one-way ownership-transition latency without copying payload bytes.
5. Measure immutable simultaneous GPU/NPU reads.
6. Run sustained ROCm+XDNA2 concurrent load and record firmware stalls, resets,
   corruption, bandwidth contention, and decode-latency impact.
7. Test lossless backend views and explicit-copy fallback tiers.
8. Decide whether one-copy heterogeneous operators are viable.

### Exit criteria

- CPU, HIP, and AIE results agree under their declared numerical contracts.
- The common layout has measured results on both accelerators.
- The memory interoperability tier is selected from evidence.
- The SHQ candidate is frozen as v1 or revised before published artifacts depend
  on long-term compatibility.
- Unstable or slower NPU paths remain disabled rather than blocking GPU work.

## Milestone 4: Qwen GPU-Only Vertical Slice

### Tasks

1. Implement transactional model loading and checksum validation.
2. Implement model-private Qwen GPU operations:
   - Embeddings.
   - RMS normalization.
   - RoPE.
   - Q/K/V and output projections.
   - Decode and prefill attention.
   - Gate, up, and down projections.
   - Residual paths.
   - LM head.
3. Implement the initial contiguous KV cache.
4. Implement deterministic greedy sampling.
5. Implement request-owned model and sampling state.
6. Implement the capability evaluation drift gate, trace format, and offline
   regrade (EVAL.md).
7. Implement `strix prompt` in direct greedy mode.
8. Add exact-token fixtures comparing the native CLI with the pinned reference.
9. Add sampling and `strix chat` only after the greedy slice passes.
10. Add eager execution first; add HIP graphs only after correctness.

Start with BF16 or SHQ8 where it simplifies bring-up, then introduce SHQ4
tensor by tensor. Do not debug every low-bit kernel simultaneously.

### Exit criteria

- A safetensors-derived artifact produces text from a terminal prompt.
- Greedy token history is deterministic.
- Layer outputs and full logits pass the declared oracle thresholds.
- No general allocation occurs in timed decode.
- Direct CLI cancellation reclaims all provisional state.
- Single-request TTFT, inter-token latency, and memory baselines are retained.

## Milestone 5: Quantization Pipeline and Size Variants

### Tasks

1. Implement `strix-inspect`.
2. Implement deterministic imatrix and activation calibration.
3. Implement `strix-plan-quant`.
4. Implement streaming, resumable SHQ4-T16, SHQ6-T16, and SHQ8-T16
   conversion.
5. Implement mixed Q4/Q6/Q8/BF16 tensor selection.
6. Search candidate recipes at a small number of explicit byte targets.
7. Validate every candidate against teacher logits, perplexity, tasks, and
   hardware performance.
8. Retain only the best artifact for each selected size.
9. Implement package manifests, checksums, and model-card generation.

### Exit criteria

- Repeated conversion from identical inputs produces identical bytes.
- Every published size has an independent quality and performance report.
- Measured BPW and file/resident byte counts match the payload.
- GPU kernels consume converted artifacts without runtime repacking unless a
  retained benchmark explicitly justifies a model-private view.
- At least one artifact meets the model-wide release objective.

## Milestone 6: OpenAI Server and Single-User Fast Path

### Tasks

1. Implement configuration parsing and model alias loading.
2. Implement the single-owner scheduler and generation-tagged completions.
3. Add the direct single-request fast lane.
4. Implement `/healthz`, `/readyz`, and `/v1/models`.
5. Implement non-streaming `/v1/responses`.
6. Add SSE streaming and cancellation.
7. Add the Chat Completions adapter.
8. Implement bearer-token and loopback exposure policies.
9. Add client-mode terminal CLI operation through `/v1/responses`.
10. Add direct-runtime versus HTTP exact-token tests.

### Exit criteria

- Official-compatible clients can perform basic text generation.
- Direct CLI and HTTP greedy requests produce identical tokens.
- A lone request does not wait for a batching window.
- Socket cancellation reaches scheduler rollback and memory reclamation.
- Startup, degraded readiness, drain, and shutdown tests pass.

## Milestone 7: Paged KV and Continuous Batching

### Tasks

1. Replace the initial contiguous KV cache with fixed-size pages.
2. Add transactional page allocation and provisional speculative pages.
3. Add admission reservations and explicit pressure states.
4. Implement chunked prefill.
5. Implement continuous decode batching with physical width buckets.
6. Add request fairness, deadlines, and cancellation.
7. Add prefix caching.
8. Add GPU graph buckets for promoted batch shapes.
9. Add KV snapshot dump and restore.
10. Add disk quota, retention, encryption, corruption, and disk-full tests.

### Exit criteria

- Concurrent requests make progress without corrupting isolated state.
- The idle single-user path returns immediately to its direct route.
- Single-user latency remains within its declared regression budget.
- Throughput improves beyond measured noise under supported concurrency.
- KV snapshots restore only under exact compatibility IDs.

## Milestone 8: Qwen NPU Integration

### Tasks

1. Integrate Qwen-private AIE programs into the compiled server.
2. Implement NPU prefill only for promoted reusable matrix-shape buckets.
3. Add NPU SHQ4-T16 W4A8 and SHQ8-T16 W8A8 paths.
4. Implement activation packing and scale handling.
5. Add shape-bucket selection based on columns actually assigned by
   `amdxdna`.
6. Add reset, timeout, contention, and suspend/resume recovery.
7. Add NPU-only layer and full-logit comparisons.
8. Establish GPU/NPU crossover tables.

### Exit criteria

- NPU routes match their CPU and GPU numerical contracts.
- NPU prefill improves the selected end-to-end workload.
- Driver failure degrades to GPU-only serving without committed-state damage.
- Other models' AIE artifact hashes remain unchanged.

## Milestone 9: GPU and NPU Concurrent Execution

### Tasks

1. Implement request-level GPU/NPU pipelining, initially allowing NPU prefill
   for one request while the GPU decodes another.
2. Add bandwidth-pressure estimation and a protected GPU-decode budget.
3. Add `NPU_ONLY`, `SPLIT_N`, `SPLIT_K_REDUCE`, and expert-parallel route
   experiments.
4. Batch candidate verification and draft work before considering tensor
   splitting.
5. Add explicit cross-device completion and reduction transactions.
6. Measure concurrent prefill, decode, and verification under realistic load.
7. Run a sustained concurrency soak covering known `amdxdna` firmware-timeout
   risks before enabling a production route.
8. Promote only route/shape combinations that improve end-to-end behavior.

### Exit criteria

- Concurrent routes preserve exact request ownership and commit decisions.
- Single-request decode is not slowed merely to keep the NPU occupied.
- Promoted heterogeneous routes improve their declared wall-time or throughput
  metric.
- Shared LPDDR contention remains within the decode-latency budget.

## Milestone 10: Speculative Decoding

### Tasks

1. Implement the provider-neutral propose, verify, accept, and commit cycle.
2. Add device-side state selection and KV commit.
3. Add MTP proposal support where the model provides it.
4. Evaluate dense draft and DFlash-style proposal models.
5. Evaluate DSpark support-model execution for DeepSeek-family work.
6. Pipeline proposal and verification across requests.
7. Add acceptance, rollback, and low-acceptance hedging tests.

### Exit criteria

- Speculative output matches ordinary target-model semantics.
- Acceptance decisions and committed state pass exact regression tests.
- The complete speculative cycle is faster than autoregressive decoding on its
  promoted workload.

## Milestone 11: DeepSeek V4 Flash

### Tasks

1. Add a separate compiled model implementation.
2. Import architecture and correctness knowledge from DwarfStar without
   sharing production numerical source.
3. Copy and independently own every required HIP and AIE kernel.
4. Implement router, expert selection, shared experts, and model-specific
   state.
5. Build model-specific calibration and router-agreement suites.
6. Add grouped expert scheduling and expert-parallel GPU/NPU experiments.
7. Add model-specific quantized size variants.

### Exit criteria

- Qwen GPU ISA, AIE programs, and performance baselines remain unchanged.
- DeepSeek logits, routing, tasks, and state pass their own gates.
- The model has independently tuned GPU, NPU, and heterogeneous routes.
- At least one quantized artifact meets its release objective.

## Milestone 12: Packaging and First Release

### Tasks

1. Run the complete release suite on the supported machine configuration.
2. Produce reproducible server and model build metadata.
3. Generate the software bill of materials and third-party notices.
4. Validate clean installation against system ROCm, XRT, driver, and firmware.
5. Publish selected quantized artifacts to private Hugging Face staging.
6. Download and verify staged artifact checksums.
7. Publish the server source and approved model artifacts.
8. Record supported versions, benchmark methodology, and known limitations.

### Exit criteria

- A clean supported Strix Halo Linux system can install and run the server.
- CLI and OpenAI API smoke tests pass after installation.
- Published hashes match locally validated release artifacts.
- Licensing, model redistribution, and attribution gates pass.

## Out of Scope and Deferred Work

The following are out of scope rather than future compatibility goals:

- Windows or macOS support.
- CUDA compatibility.
- General architecture loading.

Do not place the following on the critical path for the first production model:

- Arbitrary JIT kernels.
- Packed Q5 formats.
- Sub-four-bit production weights.
- Multiple models resident by default.
- Multimodal serving. The first Qwen3.8-27B artifact is explicitly text-only and
  excludes the vision encoder; any later multimodal artifact is separate.
- Speech, image, or video generation.

## First Backlog

The first concrete issues should be opened in this order:

1. Add the MIT license and repository skeleton.
2. Create CMake presets and build `strix --version`.
3. Implement `strix diagnose`.
4. Add HIP and XRT smoke programs.
5. Define tensor descriptors and checked byte arithmetic.
6. Implement safetensors inspection.
7. Implement Qwen tokenizer conformance tests.
8. Add teacher-logit artifact capture and comparison.
9. Commit SHQ4-T16, SHQ6-T16, and SHQ8-T16 conformance vectors.
10. Implement Qwen-private HIP Q8 GEMV.
11. Implement Qwen-private HIP Q4/SHQ6 decode kernels.
12. Implement AIE2P W8A8, W4A8, and BF16 GEMM microbenchmarks.
13. Prove or reject XRT-BO/dma-buf import into HIP with explicit handoffs.
14. Decide the shared-allocation interoperability tier.
15. Freeze or revise the SHQ candidate as SHQ-T16 v1.
16. Execute one Qwen layer from converted weights.
17. Produce the first deterministic GPU-only terminal completion.
