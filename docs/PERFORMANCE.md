# Strix Halo Performance Engineering

Status: stable guidance, 2026-08-19.

This document defines the measurement and promotion rules for performance
work. Current model numbers belong in `docs/models/<model>/BENCHMARKS.md`; issue
comments and local artifacts hold detailed experiment history.

## Target

The only production target is Linux x86-64 on AMD Strix Halo:

- `gfx1151` GPU using Wave32 HIP kernels.
- Unified LPDDR5X shared by CPU and GPU.
- Nix-provided compilers, libraries, profilers, and test tools.

Runtime probes, not assumed peak specifications, determine the available CUs,
memory, firmware, and power mode.

## Optimization Loop

1. Define the numerical and mutable-state contract.
2. Measure the complete workload and identify the dominant phase.
3. Classify it as bandwidth, compute, launch, synchronization, or scheduling
   limited.
4. Build a focused reproducer with a trusted oracle.
5. Change one mechanism at a time.
6. Reject candidates that lose focused correctness or measured performance.
7. Run broader quality and integration gates only for the retained candidate.
8. Record the current result, not the full experiment diary.

Optimize end-to-end request behavior. A faster isolated kernel is not a win if
packing, synchronization, or state management erase the gain.

## Measurement Contract

Comparable runs use the same:

- Source revision and Nix build mode.
- Model revision, artifact, quantization, and KV representation.
- Prompt tokens, generated-token count, batch, depth, and concurrency.
- GPU route and dispatch configuration.
- Driver, firmware, ROCm, power mode, and memory configuration.
- Warmup policy and repetition count.

Alternate baseline and candidate runs on the same machine. Report medians and
tail values rather than the best sample. Record the starting state for
long model runs; do not attribute an increasing-length sweep when the shared
APU runs materially slower between cases.

Headline latency comes from an unprofiled run. Use separate profiler runs for
kernel timing, counters, occupancy, VGPR, LDS, and scratch because tracing
changes timing.

Benchmark artifacts may include a privacy-safe machine fingerprint. They must
not include hostnames, usernames, local paths, prompts, generated text,
credentials, process secrets, or raw token IDs.

## Workload Guide

| Workload | Typical limit | First measurements |
| --- | --- | --- |
| Single-token decode | Weight and KV bandwidth, launch count | Effective bytes/s, kernel count, context scaling |
| Prompt prefill | GEMM utilization, attention reuse, packing | Projection and attention share, batch scaling |
| Long-context attention | KV traffic and parallelism | Per-layer attention time at 4K/8K/12K/16K |
| Speculative verification | Checkpoint, rollback, acceptance | Accepted tokens, replay time, baseline token parity |
| HTTP serving | Admission, queueing, session reuse | TTFT, inter-token latency, cancellation cleanup |

For decode GEMV, reduce bytes read before chasing peak matrix throughput. For
prefill GEMM, measure reuse and matrix-instruction utilization. For every
heterogeneous route, include shared-memory-bandwidth contention and transfer
cost in the result.

## Implementation Rules

### Host runtime

- Use RAII for mappings, streams, events, graphs, contexts, and allocations.
- Allocate model, KV, recurrent, graph, and scratch resources before serving.
- Avoid general heap allocation in decode dispatch.
- Keep hot scheduler data compact and ownership explicit.
- Prefer single-owner state machines and bounded message passing.
- Do not hold a process-wide mutex while waiting for a device.
- Reject size and offset overflow before allocating or launching.

### HIP

- Compile production kernels for `gfx1151` and treat Wave32 as explicit.
- Prove alignment before vector loads and provide bounded tail paths.
- Track VGPR, LDS, occupancy, and scratch for retained kernels.
- Keep distinct decode and prefill routes where their reuse differs.
- Read only live KV spans and reuse GQA/MQA K/V across query heads.
- Retain hipBLASLt or rocBLAS as the baseline for supported matrix shapes.
- Do not inherit launch geometry from CUDA or another RDNA target without a
  new measurement on `gfx1151`.

### Quantization

- Report artifact size and effective bits per weight, including metadata.
- Fuse unpacking, scale application, and zero correction when practical.
- Avoid materializing dequantized weights in global memory.
- Verify full-vocabulary logits before reporting speed.
- Use separate kernel strategies when metadata or group shape changes.

## Commands

### Build and quality gate

Nix must see new files, so stage them before building:

```sh
git add <changed-files>
nix build
nix build .#checks.x86_64-linux.pr
```

Run focused tests during iteration. Run the full hardware or full-logit suite
when a retained kernel, numerical route, state transition, or model dispatch
changes.

### End-to-end model benchmark

```sh
MODEL=models/<model>/<artifact>.gguf

./result/bin/gufo bench \
  --model "$MODEL" \
  --n-prompt 128,512,1024,2048,4096 \
  --n-gen 0 \
  --repetitions 3

./result/bin/gufo bench \
  --model "$MODEL" \
  --n-prompt 2048 \
  --n-gen 128 \
  --n-depth 4096,8192,12288,16384 \
  --repetitions 1

./result/bin/gufo bench \
  --model "$MODEL" \
  --validate-prefill 1024 \
  --n-prompt 1024 \
  --n-gen 0 \
  --repetitions 1
```

Keep current per-model results and matching third-party commands in that
model's benchmark README.

### Canonical serving benchmark

Build and start the release server with enough resident sessions for the
largest requested concurrency:

```sh
MODEL=models/<model>/<artifact>.gguf

./result/bin/gufo serve \
  --host 127.0.0.1 \
  --port 8080 \
  --sessions 4 \
  llm \
  --model "$MODEL" \
  --context 4096
```

In another shell, run the canonical C=1/C=2/C=4 harness:

```sh
tools/serving/gufo-serving-bench.py \
  --base-url http://127.0.0.1:8080 \
  --concurrency 1,2,4 \
  --warmup 1 \
  --repetitions 3 \
  --max-tokens 128 \
  --output artifacts/serving/benchmark.json

./result/bin/gufo diagnose \
  --validate-artifact artifacts/serving/benchmark.json
```

For speculative decoding, run the shared categorized corpus. `distinct` pairs
different request types in each wave; `homogeneous` runs C copies of each case
to measure the best case for shared draft behavior:

```sh
tools/serving/gufo-serving-bench.py \
  --base-url http://127.0.0.1:8080 \
  --suite docs/models/qwen3.8-27b/artifacts/speculative-corpus.json \
  --corpus-layout distinct \
  --concurrency 1,2,4,6,8 \
  --max-tokens 128 \
  --output artifacts/serving/speculative-distinct.json

tools/serving/gufo-serving-bench.py \
  --base-url http://127.0.0.1:8080 \
  --suite docs/models/qwen3.8-27b/artifacts/speculative-corpus.json \
  --corpus-layout homogeneous \
  --concurrency 2 \
  --max-tokens 128 \
  --output artifacts/serving/speculative-homogeneous-c2.json
```

The report keeps these metrics separate:

- Prefill throughput: actual uncached prefill tokens divided by server
  prefill time.
- TTFT: both scheduler-observed and client-observed time to first useful
  streamed output.
- Decode `tg`: completion tokens divided by server decode time.
- ITL: scheduler-observed token-to-token latency; client SSE event spacing is
  retained separately.
- Whole-request throughput: actual prefill plus completion tokens divided by
  client request wall time.
- Aggregate throughput: summed useful tokens divided by the synchronized
  C=1, C=2, or C=4 round span.
- Draft acceptance: target-accepted support tokens divided by support tokens
  proposed. Also inspect drafted tokens per output token: high acceptance with
  very low proposal coverage cannot materially change end-to-end throughput.
- Category and case summaries: acceptance, proposal coverage, decode speed,
  and generated-text SHA-256 counts. The hashes support exact A/B comparison
  without storing generated text; compare counts for homogeneous waves because
  equivalent trajectories may exchange request slots.

Raw per-request samples and p50/p95/p99 summaries are retained. Endpoint hosts,
prompt text, generated text, model paths, timestamps, and token IDs are never
written to the artifact. `gufo bench` remains the direct model-path
microbenchmark; use this serving harness for TTFT, ITL, queueing, and
concurrency decisions.

### HIP allocation diagnostic

Compare allocation, mapped-registration, first-touch, warm-access, copy,
advice/prefetch, synchronization, teardown, fault, and checksum behavior:

```sh
./result/bin/gufo diagnose \
  --benchmark allocation \
  --working-set-mib 64,1024 \
  --warmup 2 \
  --repetitions 5 \
  --json \
  --output artifacts/diagnostics/hip-allocation.json
```

The command aborts before a requested size that cannot preserve its memory
headroom; it never silently substitutes a smaller working set. The production
Qwen3.8-27B loader registers mapped GGUF shards directly with HIP; see
[the model's state ownership](../src/models/qwen/README.md#state-and-arithmetic).

### Focused HIP benchmark

List cases, then run the smallest relevant matrix:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target gufo-kernel-bench tune_hipblaslt
./build/gpu-test/gufo-kernel-bench --help

./build/gpu-test/gufo-kernel-bench \
  --case decode-attention \
  --context 4096,8192,12288,16384 \
  --warmup 5 \
  --repetitions 30 \
  --json
```

The report includes raw HIP-event samples, summary latency, correctness
sentinels, dispatch choices, and the privacy-safe machine fingerprint.

### Roofline calibration

Score every kernel against measured ceilings, not spec-sheet numbers. Build the
standalone microbenchmarks and run the calibrator:

```sh
nix develop -c tools/bench/build.sh              # all of tools/bench/*.hip -> /tmp
nix develop -c tools/bench/build.sh gfx1151_peak # or one by name
/tmp/gfx1151_peak
```

It reports sustained WMMA INT8/BF16/FP16 and VALU FP32 FMA rates, a mixed
WMMA/epilogue instruction probe, LDS reads, and DRAM read/write/copy bandwidth.
Every thread writes a checksum covering every accumulator; host references
check them before timing. Distinct WMMA chains and volatile LDS reads keep
the measured work live. Compute loops use registers; memory probes read the
named memory space.

Record calibration beside the model experiment, with its power state and
compiler. When changing this tool, inspect the emitted instruction counts as
well as the checksums: a correct result alone cannot detect hoisted work.
Calibration from the earlier gated-sink implementation is invalid.

`tools/bench/build.sh` exists because `hipcc` invokes the raw HIP clang++ rather
than the Nix cc wrapper, so it forwards the include and library paths that
`nix develop` exports through `NIX_CFLAGS_COMPILE` / `NIX_LDFLAGS`. It also
prints per-kernel VGPR, occupancy, spill, and LDS usage into
`/tmp/<name>.res.txt`.

### Kernel design iteration

Fast probes live in `tools/bench/` and model-specific tool folders. The Qwen27B
probes use production kernels; standalone design experiments keep their own
references. Each reports correctness beside component timings. Use release
model comparisons before retaining a change: a microbenchmark can improve
while the model gets slower.

```sh
nix develop -c tools/bench/build.sh tools/qwen27b/prefill_gemm_bench.hip
/tmp/prefill_gemm_bench q5 17408 5120 2048 12
/tmp/prefill_gemm_bench q8-swiglu 2048 12

/tmp/bf16_gemm_bench -b 2048     # hipBLAS / hipBLASLt bar to beat
/tmp/aotriton_attn_bench -i 5    # AOTriton flash-attention capability probe
/tmp/asr_decode_gemv_bench       # batch-one decode GEMV vs the DRAM ceiling
```

The Qwen prefill probe calls the production kernels and requires exact bytes,
including quantization scales and sums. Other probes report their numerical
error beside timings. Retained changes are validated with Nix release model
measurements and the affected operator checks.

**Size the working set like the model does.** Strix Halo has a 32 MB MALL, and a
single decoder projection is 4-25 MB. Timing one shape in a repeat loop leaves it
cache resident and reports 400-860 GB/s for a kernel that sustains 124 GB/s in
the model. A decode-bandwidth harness must therefore allocate the whole per-token
weight footprint and time a full token pass, which is what
`tools/bench/asr_decode_gemv_bench.hip` does. The same caveat applies in reverse
to non-temporal loads: `__builtin_nontemporal_load` bypasses the MALL, and on
every streaming kernel measured so far it has cost more than half the achieved
bandwidth rather than helping.

### Profiling

`tools/prof/prof.py` wraps `rocprofv3` and answers the three questions a flat kernel
table cannot: which pipeline stage owns the time, whether the GPU is actually
busy, and what changed between two runs.

```sh
# profile a command and analyze in one step
nix develop -c python3 tools/prof/prof.py run --stages qwen -- \
  ./result/bin/gufo bench --model "$MODEL" -p 2048 -n 0 -r 1

# re-analyze an existing database
nix develop -c python3 tools/prof/prof.py show /tmp/prof/prof_results.db --top 20

# list the passes separated by idle gaps over 5 ms, then analyze only pass 3
nix develop -c python3 tools/prof/prof.py show /tmp/prof/prof_results.db --passes 5
nix develop -c python3 tools/prof/prof.py show /tmp/prof/prof_results.db --passes 5 --pass 3

# A/B two runs, per stage and per kernel
nix develop -c python3 tools/prof/prof.py diff before_results.db after_results.db
```

A benchmark trace holds several passes (model upload, a warm-up prefill, the
timed prefill, decoding); the first pass over freshly uploaded weights ran up to
ten times slower than the timed one, so attribute kernels within the timed pass.
`run` and `show` print a pipeline-stage rollup (kernel names grouped by model
stage), a per-kernel table with launch geometry, GPU-busy-versus-wall-span with
the idle percentage, and the largest idle gaps attributed to the dispatch on
either side. A large idle share means launch- or host-bound; a small one means
the remaining work is genuinely in the kernels. `--stages ''` disables grouping
for a non-Qwen workload; `--json` emits the same data for scripting.

For raw rocprofv3 with counters or ROCTx markers:

```sh
nix develop -c rocprofv3 \
  --kernel-trace \
  --marker-trace \
  --scratch-memory-trace \
  --stats \
  --summary \
  --output-directory /tmp/gufo-profile \
  -- ./build/gpu-test/gufo-kernel-bench \
    --case decode-attention \
    --context 16384 \
    --warmup 1 \
    --repetitions 3
```

Use the emitted ROCTx case marker to isolate the measured region. Add selected
PMC counters only in a separate diagnostic pass.

### Instruction mix

When a kernel is off its roofline, the instruction mix says why. `tools/prof/isa_mix.py`
groups one kernel's emitted instructions into matrix, VALU, LDS, global memory,
and wait/barrier categories:

```sh
nix develop -c hipcc -O3 --offload-arch=gfx1151 -std=c++20 \
  -I. --cuda-device-only -S -o /tmp/k.s tools/qwen27b/prefill_gemm_bench.hip
nix develop -c python3 tools/prof/isa_mix.py /tmp/k.s            # list kernels
nix develop -c python3 tools/prof/isa_mix.py /tmp/k.s W8A8BlockedWmmaGEMMKernel
```

Read the counts with care: the listing covers a whole kernel, so a once-per-block
store epilogue is counted alongside the K loop that repeats hundreds of times.
Attributing epilogue instructions to the inner loop led to one rejected
experiment (`opt-c163-lowoverhead`); confirm a hypothesis with an ablation in the
microbenchmark before acting on the mix.

### Offline tuning and replay

```sh
./build/gpu-test/tune_hipblaslt \
  --out /tmp/gufo-hipblaslt-plans.bin \
  --warmup 3 \
  --repetitions 10

GUFO_HIPBLASLT_PLAN_CACHE=/tmp/gufo-hipblaslt-plans.bin \
  ./result/bin/gufo bench ...

nix develop -c tools/bench/build.sh tools/qwen27b/deltanet_bench.hip
/tmp/deltanet_bench 8 48

nix develop -c tools/bench/build.sh tools/qwen27b/attention_bench.hip
/tmp/attention_bench
/tmp/attention_bench rope
```

The recurrence probe checks exact outputs/state with a rotating 144 MiB state
working set. The Qwen27B model suite checks full-logit rollback, including
replay-ring wraparound; `gufo bench` measures actual speculative throughput.

Generated profiler, plan, and replay artifacts stay outside the repository.

### Dispatch telemetry

```sh
GUFO_DISPATCH_TELEMETRY=1 ./build/gpu-test/gufo-kernel-bench ...
```

Telemetry is diagnostic JSONL. It records semantic dispatch decisions and
must not contain prompts, tokens, model paths, or machine identity.

## Promotion Gates

A performance change is retained only when:

- Focused correctness passes against the canonical oracle.
- Full-vocabulary logits remain finite and within the accepted envelope.
- Greedy output keeps the required token parity.
- Request state, rollback, cancellation, and cleanup remain correct.
- The declared workload improves outside measurement noise.
- Memory, startup, and tail latency do not regress unexpectedly.
- Required Nix checks pass on the exact staged source.

Document the winning route, current measurements, reproduction command, and
remaining gap. Keep rejected alternatives to a short summary or the relevant
issue discussion.
