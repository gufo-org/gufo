# HRX experiments

Experiments with the HRX runtime, its native API, Loom and libamdf on Strix
Halo (gfx1151), newest first. Each section names the hrx-system and Gufo
commits it ran on, so it can be repeated as hrx-system `main` moves; upstream
fix status is tracked in #1144. Background and the first measurements are in
[`../hrx-migration.md`](../hrx-migration.md).

Unless stated, Gufo runs use Qwen3.8-Flash-Next UD-Q4_K_XL (Unsloth revision
`d96201`), `-r 3`, a cooldown to 38 °C or below plus 60 s idle before each run,
and the same Gufo binary on both runtimes: the one `.#gufo-hrx` wraps
([`ab.sh`](ab.sh)). Rates are tokens per second.

## A Loom kernel inside the HIP code base (November 10, 2026)

hrx-system `dd9572875f`. Can an extension author write a kernel in Loom
instead of HIP C++, and Gufo launch it through HIP? Gufo's
`SwigluHalfKernel` (`out = silu(gate) * up` to F16, four elements per work
item) was rewritten in 25 lines of Loom ([`swiglu.loom`](swiglu.loom)) and
compiled ahead of time with `loom-compile --target=amdgpu:gfx1151
--format=amdgpu-hsaco`. The code object has a plain argument layout (count by
value at offset 0, three 8-byte buffer pointers, 32 bytes, wave32), so the
host ([`swiglu_host.hip`](swiglu_host.hip)) loads it with `hipModuleLoad` and
launches it with `hipModuleLaunchKernel` next to the original kernel. Median
of 15 x 20 chained launches, inputs written device-to-device:

| Elements | HIP runtime: Gufo | HIP runtime: Loom | HRX: Gufo | HRX: Loom |
| ---: | ---: | ---: | ---: | ---: |
| 786,432 | 30.6 µs | 29.9 µs | 40.0 µs | 39.2 µs |
| 3,145,728 | 92.9 µs | 92.3 µs | 106.6 µs | 102.5 µs |
| 20,971,520 | 978.8 µs | 970.4 µs | 980.5 µs | 966.4 µs |

The Loom kernel matches or slightly beats the HIP one on both runtimes. Its
output differs by one F16 rounding step (at most 0.0078) in about 0.008% of
elements, from the fast-math exponential inside SiLU (`siluf<afn>` against
`__expf`); a lossless port would need the same approximation. Both kernels
pay the same HRX cost on cache-resident sizes, so that cost is the runtime's,
not the authoring language's.

Kept as evidence for Loom-authored extension kernels: the GPU path needs no
HRX at run time, only the Loom compiler at build or load time (`loomc` also
offers JIT). Elementwise kernels are the easy case; GEMM-class kernels still
need the production-tile recheck (4.4 against 28.7 TFLOPS in October 2026).

## LFM3 on GPU and NPU: speed per watt (November 10, 2026)

Recorded in the extension's experiment log
([lfm3-gufo-extension#24](https://github.com/gufo-org/lfm3-gufo-extension/pull/24)).
LFM3-1.7B Q8 over HTTP, per phase with socket power: HIP prefill 7,821 tok/s
at 140.6 W (18.0 mJ/token) and decode 111.8 tok/s at 100.4 W (898 mJ/token);
HRX 7,054 and 95.6 tok/s (19.8 and 1,011 mJ/token, measured before the queue
pool). The NPU embedding below uses about 20 W for about 2,100 tok/s; projected
onto LFM3 decode it would be about 12 tok/s and 1.7 J/token unless the NPU's
weight bandwidth is well above 21 GB/s.

## Loom's Qwen3-Embedding on the XDNA NPU (November 10, 2026)

hrx-system `dd9572875f`, hrx-loom-library `8ef2e49`. Loom tools built with
`-DLOOM_TARGET_ALL_AMD=ON -DLOOM_ARCH_AMDGPU_TARGETS=gfx1151
-DLOOM_IMPORT_CXX=ON` (architectures `aie2p,amdgpu,vm`; the embedding
programs include C++ sources). The library serves `Qwen/Qwen3-Embedding-0.6B`
on the Strix Halo NPU with eight columns and four core rows
(`--device=xdna://0?columns=8 --pool_backing=fixed`), loading Qwen's
original BF16 checkpoint (revision `97b0c614`, 1,191,586,416 bytes, fetched
and checked against the card's lock).

Two problems so far:

- The card's `programs.source` and `programs.archive` lack the
  `"origin": "../../../"` that the Flash-Next card carries, so `loom-fetch`
  resolves the library-root paths against the card directory and fails with
  `local source is missing or not a file`. Adding the field locally fixes
  the fetch (a library bug to report).
- With our compiler the NPU program does not fit:
  `XDNA/049: tile [0, 2] requires 66688 bytes for resident data and loaded
  worker storage, but has 65536 bytes`. The library pins hrx-system
  `314238a81c` (November 8), one day older.

With `loom-serve` built at the library's pin `314238a81c`, the program fits
and the model is ready in 8 s (3 kernels compiled). So the tile-memory failure
is a Loom regression between `314238a81c` and `dd9572875f`. `git bisect run`
with [`bisect-npu.sh`](bisect-npu.sh) (incremental `loom-serve` build, start
on the NPU, read the registration) names `a62c887489` "[Loom] Add native AIE2P
stack spilling" (November 8) as the first bad commit: the program now reserves
tile SRAM for a spill stack and exceeds the 64 KiB tile by 1,152 bytes. To
report upstream.

Embeddings are sane ([`embbench.py`](embbench.py), 1,024 dimensions):
related sentences score 0.749 and 0.669 cosine, unrelated ones 0.283 and
0.205. A cohort holds 128 token slots, each sequence a multiple of eight, so
the batch is bounded by length, not count. Every cohort takes about 57 ms
whatever it holds:

| Requests per call (14-token input) | Wall | Requests/s |
| ---: | ---: | ---: |
| 1 | 57.0 ms | 17.5 |
| 4 | 57.7 ms | 69.3 |
| 8 | 58.6 ms | 136.5 |
| 64 concurrent single requests | 522 ms | 122.5 |

At eight per cohort that is about 2,050 tokens/s on the NPU, with the GPU
untouched. Kept as a working NPU path to study; Loom's GPU and NPU programs
are separate (AMDGPU wave32 kernels against a dataflow program pinned to an
8 x 4 tile grid with 64 KiB per tile), so kernels are not shared between the
two.

## Divergent kernel-argument loads (November 10, 2026)

hrx-system `dd9572875f`. Found while chasing LFM3's slower decode on HRX
([lfm3-gufo-extension#24](https://github.com/gufo-org/lfm3-gufo-extension/pull/24)).
A kernel that reads a by-value job table with a per-wave index (the index
comes from `threadIdx`, so the compiler emits vector loads from the
kernel-argument segment) against the same table in device memory through a
pointer ([`kernarg.hip`](kernarg.hip)):

| Grid x elements per workgroup | HIP, by value against pointer | HRX, by value against pointer |
| --- | ---: | ---: |
| 512 x 256 | 3.74 against 3.60 µs (+4%) | 7.89 against 6.18 µs (+28%) |
| 512 x 2,048 | 13.99 against 14.40 µs (-3%) | 19.19 against 14.40 µs (+33%) |
| 2,048 x 256 | 7.06 against 6.97 µs (+1%) | 13.49 against 10.32 µs (+31%) |
| 8,192 x 2,048 | 123.96 against 124.10 µs (0%) | 130.98 against 119.88 µs (+9%) |

With a uniform (per-workgroup) index both runtimes use scalar loads and the
two forms match within 4%. So HRX's kernel-argument memory is slow for vector
loads, probably an uncached placement. Gufo kernels that take large structs
by value and index them per wave pay this on HRX. Flash-Next's
`DenseF16GEMMKernel` (the `hcmix` route, still 7% slower in isolation) only
reads pointer fields of its by-value `AttentionProjectionOutput` and
`PassCheckpoints`, uniformly, so it does not explain that kernel. Not yet
reported upstream.

LFM3's own decode deficit is a different kernel: its SwiGLU `Q8_0MultiGEMV`
runs about 2x slower on HRX in a standalone copy
([`multigemv.hip`](multigemv.hip)), and selecting the weight pointer
uniformly does not change that.

## Idle GPU spin with a queue per stream (November 10, 2026)

hrx-system `dd9572875f`, Gufo `3edc2eff` with lfm3-gufo-extension `debba20`.
An idle `gufo serve llm` with LFM3-1.7B on `.#gufo-hrx` kept the GPU at 100%
busy and 2,900 MHz with no requests and no host CPU time in the server: the
socket drew 39-41 W against 11-12 W for the same server on HIP (9-12 W with no
Gufo process). [`idleprobe.hip`](idleprobe.hip) rules out HRX itself:
after init, a finished kernel, 64 MiB pageable, pinned and registered uploads,
a cross-stream fork/join and live graph executables, HRX idles at 9-12 W and
0% busy. (Stock HIP also spins at 99% and 37 W while a stream waits on a
host-written `hipStreamWaitValue32`; HRX rejects that call on `hipHostMalloc`
memory, see hrx-system#2985.)

The cause is our own patch. A dispatch profile of the idle server holds two
kernels and 70 queue records: `hrx-stream-queues.patch` gives every created
stream its own hardware queue, and LFM3 creates about 70 streams. With the
probe build's switch:

| Streams on | Idle busy | Idle socket power |
| --- | ---: | ---: |
| the context queue only | 0% | 11 W |
| a queue each | 100% | 40 W |
| a pool of 2 queues | 0% | 12 W |
| a pool of 4 queues | 0% | 11 W |
| a pool of 16 queues | 100% | 36 W |

So more than a handful of mapped queues keeps the gfx1151 scheduler busy.
The fix spreads streams round-robin over a pool of four (the context queue
plus three acquired ones; `HRX_STREAM_QUEUE_POOL` overrides). Flash-Next,
two alternating pairs, `-r 3`:

| Streams on | pp512 | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| a pool of 4 queues | 1,309.04 and 1,316.58 | 1,875.63 and 1,889.71 | 30.52 and 30.58 |
| the context queue only | 1,274.92 and 1,280.03 | 1,862.89 and 1,844.20 | 30.44 and 30.50 |

The pool keeps the concurrency gain (+2.9% pp512, +1.5% pp2048; a queue per
stream gave +3.2% and +1.9%) without the idle cost. Retained: the `.#hrx`
package uses the pool since #1147.

## Graph replay and prefill capture (November 10, 2026)

hrx-system `dd9572875f`, Gufo `3edc2eff`. HRX's graph replay is its cheapest
launch path. A chain of 200 short bounded kernels on one stream, eager or
captured once and replayed ([`graphcost.hip`](graphcost.hip)), time per
kernel:

| Grid, arguments | HIP eager | HIP graph | HRX eager | HRX graph |
| --- | ---: | ---: | ---: | ---: |
| 20 workgroups, 16 B | 2.07 µs | 1.90 µs | 4.82 µs | 0.66 µs |
| 20 workgroups, 256 B | 2.04 µs | 1.91 µs | 5.23 µs | 0.73 µs |
| 640 workgroups, 16 B | 3.83 µs | 3.79 µs | 5.03 µs | 1.90 µs |
| 640 workgroups, 256 B | 4.08 µs | 3.79 µs | 6.97 µs | 1.94 µs |

Building a graph is cheap too: capturing and instantiating 1,200 kernels takes
697 µs of host time on HRX (0.58 µs per kernel) against 1,639 µs on HIP, and
capture plus `hipGraphExecUpdate` 2,043 µs against 13,587 µs. On HRX a graph
built afresh for every pass still costs about 2.5 µs per kernel against 5-7 µs
eager.

Flash-Next captures only decode-sized batches. The probe
([`prefill-graph-probe.diff`](prefill-graph-probe.diff),
`GUFO_PROBE_PREFILL_GRAPH=1`) captures each warmed prefill pass, launches it
once and retires it. The first version never engaged: `gufo bench` opens a new
session per repetition, and the per-session warm set never matched, so its
A/B (pp2048 1,890.09 and 1,886.86 on, 1,892.51 and 1,890.35 off) measured
eager twice. With the warm set kept per process, capture engages and fails:
`probe graph capture end: operation failed due to a previous error during
capture`. A local HRX debug patch that prints a backtrace where a capture is
invalidated ([`hrx-debug-capture.patch`](hrx-debug-capture.patch),
`HRX_DEBUG_CAPTURE=1`) names the call: `Executor::RouteHints` waits on
`hipEventSynchronize` inside `MoeExperts`. Prefill reads each MoE layer's
expert counts on the host and plans the routed GEMM tiles from them, so the
pass cannot be one graph.

Blocked: prefill capture needs device-side routing (indirect dispatch of the
routed GEMMs); HRX is not the obstacle. The expected gain on HRX was small
anyway: eager prefill gaps median 5.5 µs over 786 gaps per pass, about 4 ms of
a 1,083 ms pass.

## Loom's Flash-Next on the native driver (November 10, 2026)

hrx-system `dd9572875f`, hrx-loom-library `8ef2e49`. Loom tools built from the
HRX CMake tree with `-DLOOM_TARGET_ALL_AMD_GPU=ON
-DLOOM_ARCH_AMDGPU_TARGETS=gfx1151`. The library's
`Qwen/Qwen3.8-Flash-Next` card pins Unsloth revision `766911a`; our local
shards are revision `d96201`, the same tensors split differently (total size
differs by 6 bytes), so the card was fetched with
`--origin quantized=<symlinks to the local shards>`,
`--origin checkpoint=<the pinned tokenizer.json>`,
`--no-verify-parameters --offline`. The fetch shares the shards by link; no
copy.

Loom compiles 2,563 kernels for the `chat` preset (about 75 s of wall time on
8 workers, 596 s summed); a restart reuses them and is ready in 6 s. The first
request activates the 104 GiB parameter bank (about 40 s). Output is coherent.
Rates over HTTP with the same script for every request
([`loombench.py`](loombench.py)), greedy, 3 repetitions:

| Test | Loom (`loom-serve`, HTTP) | `gufo-hrx` (`gufo bench`) |
| --- | ---: | ---: |
| pp512 (524 prompt tokens) | 1,041-1,112 | 1,318 |
| pp2048 (2,060 prompt tokens) | 1,530-1,546 | 1,884 |
| tg128 | 29.98-30.12 | 30.52 |

The Loom numbers include HTTP, tokenization and the chat template, so they
are an upper bound on its overhead against an in-process benchmark. A
dispatch-event profile (`--device_profiling_mode=dispatch-events`, which
costs under 2% here) of one 2,060-token prefill: 1,306 dispatches on one
queue, 1,316 ms of kernel time, no overlap and no idle gaps.

| Stage | Loom kernels | Loom (ms) | Gufo on HRX (ms) |
| --- | --- | ---: | ---: |
| Routed MoE | `expert_matrix` | 431.7 | 402.6 |
| Dense projections | `flash_next_q8_matrix_256`, `hc_matrix_*`, `gdn_fused_projection`, `shared_matrix` | 620.1 | 513.1 |
| GDN recurrence | `gdn_recur_window` | 77.7 | 66.9 |
| Attention | `qsa_cohort_attention` | 36.8 | 33.5 |
| HC combine, norm, injection and MoE epilogue | `hc_combine_normalize_inject`, `experts_combine` | 88.3 | 135.0 |

Gufo is ahead everywhere except the hyper-connection residual pass. Both
fuse the combine, RMS norm and next-layer injection into one kernel; the
difference is precision: Loom keeps the 4 x 2,560 residual streams in BF16,
Gufo in F32, so Loom moves half the bytes through a bandwidth-bound pass. A
BF16 residual would be worth about 47 ms per pass (about 4% of prefill) on
either runtime, but it changes numerics and needs the F32-oracle precision
gate before it could be kept. Not adopted.

## Stream priority (November 10, 2026)

hrx-system `dd9572875f`, Gufo `3edc2eff`. The probe
([`stream-priority-probe.diff`](stream-priority-probe.diff)) creates either
the main stream or the shared-expert side stream with the device's greatest
priority when `GUFO_PROBE_STREAM_PRIORITY=main|side` is set. On HRX a
non-normal priority also selects a queue of its own at that priority. One run
per setting, `-p 512,2048 -n 128`:

| Priority | HIP pp512 | HIP pp2048 | HRX pp512 | HRX pp2048 | tg128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 1,354.40 | 1,944.02 | 1,316.74 | 1,868.65 | 30.52-30.61 |
| main stream | 1,347.83 | 1,930.18 | 1,312.03 | 1,855.04 | 30.57-30.61 |
| side stream | 1,352.02 | 1,950.83 | 1,314.04 | 1,880.77 | 30.52-30.60 |

Every difference is at most 0.7% and inside the run spread (pp2048 standard
deviation 4-15 tok/s). Dropped: priority does not shorten the MoE fork/join on
either runtime.

## MTP speculative decode (November 10, 2026)

hrx-system `dd9572875f`, Gufo `3edc2eff`, `-p 512 -n 128 --speculative mtp`
with the shared Q8_0 MTP sidecar, two alternating pairs:

| Pair | HIP pp512 | HRX pp512 | HIP tg128 | HRX tg128 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,321.23 | 1,294.44 | 60.39 | 61.14 |
| 2 | 1,322.66 | 1,290.02 | 63.26 | 61.31 |

HRX's reworked graph replay brings no decode gain: tg128 differs by -1% on
average and the pairs disagree in sign. Prefill stays 2.2% behind.

## Rerunning

```sh
nix build .#gufo-hrx --out-link /tmp/gufo-hrx
export GUFO_HRX=/tmp/gufo-hrx MODEL=<first GGUF shard> OUT=/tmp/hrx-ab
docs/internal/hrx-migration/ab.sh hip base-hip -p 512,2048 -n 128 -r 3
docs/internal/hrx-migration/ab.sh hrx base-hrx -p 512,2048 -n 128 -r 3
```

For a probe diff, apply it, rebuild `.#gufo-hrx`, and pass its switch in the
environment of `ab.sh`. After bumping HRX in `.devops/nix/hrx/default.nix`,
check `gufo diagnose` and `gufo bench --validate-prefill 512` before repeating
an experiment, then add a dated section here and update #1144.
