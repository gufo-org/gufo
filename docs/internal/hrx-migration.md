# HRX migration

What it takes to run Gufo on [HRX](https://github.com/ROCm/hrx-system) instead
of stock ROCm HIP, measured on Qwen3.8-Flash-Next as of November 2026. HRX is
not a drop-in win yet: prefill is 2-3% slower than HIP even with a local
queue-policy patch, and the cause of the rest is open.

## Setup

- **HRX** is AMD's alternative HIP implementation. Its HIP layer installs as
  `libamdhip64.so.7`, so a Gufo binary switches runtime through
  `LD_LIBRARY_PATH` without rebuilding. Since HRX commit `3278a4f66f` the
  default GPU driver is native libamdf (`HRX_GPU_DRIVER=amdgpu`), which does
  not use ROCr or HSA.
- Measured: HRX `dd9572875f` against Gufo `main` `b62b1674c`
  (`nix build .#release`, ROCm 10.1), gfx1151, model
  `Qwen3.8-Flash-Next-UD-Q4_K_XL`.
- HRX build: [`hrx-configure.sh`](hrx-migration/hrx-configure.sh) uses
  `IREE_ROCM_DEPENDENCY_MODE=pinned` and the flake's ROCm 10.1 clang and device
  libraries as the device toolchain. Only the native drivers are enabled; the
  legacy HSA driver is off.
- Runs: [`run1.sh`](hrx-migration/run1.sh) refuses to start beside another
  model process, cools the GPU to 38 °C or below plus 60 s idle, and runs one
  `gufo bench` inside a memory-capped systemd scope.

## Local HRX patches

All five are in [`hrx-probe.patch`](hrx-migration/hrx-probe.patch). None is
upstream.

| Patch | Why | Needed for |
| --- | --- | --- |
| `hsaco.c`: accept `DT_RPATH`/`DT_RUNPATH` | The Nix linker wrapper writes a `DT_RUNPATH` into Gufo's code objects; HRX rejected tag `0x1d` and Flash-Next failed with "quantized GEMM tier initialization failed" | Running at all |
| `api.c`: `HRX_STREAM_DEDICATED_QUEUE=1` | Gives each `hipStreamCreate*` stream its own hardware queue (see below) | Half the prefill gap |
| `reader.c`: `IREE_PROFILE_ALLOW_OPEN_SESSION=1` | Gufo exits without ending the HRX profile session, so `iree-profile` refused the file | Profiling |
| `barrier.c`: `HRX_PROBE_NO_GL2=1` | Drops L2 writeback/invalidate from system-scope barriers. Unsafe, probe only | Ruling out barriers |
| `launch.c`: `HRX_PROBE_INITIATOR_OR`, `HRX_PROBE_LIMITS_OR`, `HRX_PROBE_INST_PREF` | Override dispatch initiator, resource limits and instruction prefetch. Probe only | Ruling out launch parameters |

The October 2026 blockers are gone on the native driver: no ROCr
`PM4_EMULATION` rejection, and `hipFuncGetAttributes` works.

## Results

Thermally controlled A/B/A/B, 4 pairs, `-r 3`, tokens per second:

| Test | HIP | HRX | HRX + dedicated queues |
| --- | --- | --- | --- |
| pp512 | 1352 | 1284 (-5.0%) | 1325 (-2.0%) |
| pp2048 | 1940 | 1851 (-4.6%) | 1886 (-2.8%) |
| tg128 | 30.60 | 30.53 (-0.2%) | 30.55 (-0.2%) |

The dedicated-queue column is a single `-r 3` run, not the full A/B. The HIP
pp2048 mean drops pair 1, where one repetition collapsed (± 639) to
hipBLASLt first-use plan timing. `--validate-prefill 512` gives identical
results on both runtimes (rmse 0.520053, cosine 0.991351, max error 3.81343).
Decode ties because it already runs through HIP graphs.

## Findings

### One hardware queue for every stream

HRX's HIP frontend creates every normal-priority stream on the context's single
queue (`hipStreamCreate` and `hipStreamCreateWithFlags` pass `context->queue`).
Only a stream with a non-normal priority acquires a queue of its own.
Flash-Next runs the shared-expert GEMMs on a side stream concurrently with the
routed MoE stream. Under HIP, 492 ms of kernel time across the whole trace
overlaps across streams; under HRX every dispatch ran on queue 0 with zero
overlap. With the patch, overlap returns (434 ms) and busy time matches HIP
within 1%.

The fix belongs in HRX: per-stream queues, or a policy that maps independent
streams to separate queues. A Gufo-side workaround is to create the side stream
with a non-normal priority, but that also changes scheduling on stock HIP.

### The remaining 2-3%

Per steady pp2048 pass, traced with rocprofv3 for HIP and
`HRX_PROFILE_MODE=dispatch` for HRX ([`passes.py`](hrx-migration/passes.py)):

| Runtime | GPU span | Busy | Idle | Median launch gap |
| --- | --- | --- | --- | --- |
| HIP (under rocprof) | 1077 ms | 1067 ms | 10.5 ms | 2.2 µs |
| HRX, shared queue | 1099 ms | 1089 ms | 9.6 ms | 5.0 µs |
| HRX, dedicated queues | 1082 ms | 1075 ms | 7.0 ms | 5.4 µs |

Both traces hold the same 12781 kernel dispatches with the same per-family call
counts, so both runtimes take the same kernel routes. These are ruled out:

- **Idle time and launch gaps.** The GPU idles 7 ms per pass under HRX. The
  large gaps outside passes are benchmark phase boundaries.
- **Hardware ceilings.** `tools/bench/gfx1151_peak.hip` gives the same WMMA
  int8/bf16/fp16, FP32 FMA, LDS and DRAM rates on both runtimes.
- **Kernel speed in isolation.** In
  `qwen38_flash_next_dense_gemm_bench`, rebuilt with 20 chained launches per
  event pair ([`dense-chain.diff`](hrx-migration/dense-chain.diff)), the large
  GEMM kernels traced 2.5-3.2% faster under HRX than under rocprof.
- **L2 barriers.** `HRX_PROBE_NO_GL2=1` moved pp2048 by +0.7%, within noise.
- **Clocks, power and host CPU.** During prefill both runtimes sit at the
  package power cap: HIP 2650 MHz at 154 W, HRX 2696 MHz at 150 W, mclk
  1000 MHz on both, about 1.5 host cores busy on both
  ([`sample.sh`](hrx-migration/sample.sh)).

rocprof itself costs HIP 1.3% (1940 to 1916 tok/s) and inflates its kernel
durations by 2-3%, while the HRX profiler costs nothing. Profiled, the two
runtimes look equal; unprofiled, HIP is 26 ms per pass faster.

Bounded microbenchmarks, run one at a time after the GPU was recovered:

- **Concurrency is not the cause.** With dedicated queues two half-GPU kernels
  overlap fully, as on HIP; on the shared queue they serialize
  ([`overlap.hip`](hrx-migration/overlap.hip)). Two full-grid kernels, one
  memory bound and one ALU bound, gain 14.7-18.7% from running in parallel
  under HRX against 13.5-14.5% under HIP
  ([`contend.hip`](hrx-migration/contend.hip)).
- **Dispatch cost is higher.** A 20-workgroup kernel takes 2.2 µs per
  chained launch under HIP and 5.0 µs under HRX; with 256 bytes of kernel
  arguments, 2.2 µs and 6.4-7.9 µs. Per-workgroup cost on large grids is
  equal ([`wgstart.hip`](hrx-migration/wgstart.hip)). Cross-stream event hops
  cost 18.9 µs under HIP and 20.6 µs under HRX. At 1219 dispatches per pass this
  accounts for about 4-8 ms, under 1% of prefill.
- **Host uploads leave slow cache state.** After a pageable
  `hipMemcpy` host-to-device copy, a buffer of 1-16 MiB reads 26% to 10x slower
  under HRX until a GPU write replaces it (16 MiB: 988 GB/s under HIP, 86 GB/s
  under HRX). The same buffer filled by `hipMemset`, or rewritten with a
  device-to-device copy, reads at HIP speed, and buffers of 32 MiB or more are
  unaffected ([`allocsize2.hip`](hrx-migration/allocsize2.hip)). The state lives
  in the MALL: random-page, strided and write traffic on GPU-written memory
  match HIP ([`tlb.hip`](hrx-migration/tlb.hip)), and changing
  `HRX_MEM_POOL_BYTES` or the 16 MiB PM4 upload span changes nothing. Gufo
  prefill streams weights from a model far larger than the MALL, so this
  mostly distorts benchmarks that upload inputs and re-read them; it is still an
  HRX defect to report.
- **Most GEMM-bench slowdowns were that upload state.** With each uploaded
  buffer rewritten on the GPU (`BENCH_D2D_REWRITE=1` in
  [`dense-chain.diff`](hrx-migration/dense-chain.diff)), `down`, `route` and
  `gates` match HIP. `hcmix` stays 7% slower (0.535 ms against 0.489-0.506 ms)
  and `ssm` 1-2% slower.
- **Launch parameters match.** These were compared and ruled out for `hcmix`:
  `ORDER_MODE` in `COMPUTE_DISPATCH_INITIATOR` (`HRX_PROBE_INITIATOR_OR=40`),
  instruction prefetch (`HRX_PROBE_INST_PREF` 0, 16 and the descriptor's 63),
  scratch wave capacity (640 per engine under both, from libamdf and KFD),
  and LDS allocation (no residency step from 1 to 64 KiB under either runtime,
  [`occ.hip`](hrx-migration/occ.hip)). The large Flash-Next GEMM kernels use no
  scratch and run in WGP mode. The `hcmix` and `ssm` gap remains unexplained;
  the next step is an instruction-level thread trace of `hcmix` under both
  runtimes.

### Other HRX defects found

- A kernel that reads `hipHostMalloc` memory page-faults
  (`PERMISSION_FAULTS 0x3`, [`memkind.hip`](hrx-migration/memkind.hip)). Gufo's
  Flash-Next path does not hit this; `hipMalloc` and `hipHostRegister` memory
  work.
- hipBLASLt's Tensile code objects fail to load ("device kernel image is
  invalid"), so every hipBLASLt route fails under HRX. The measured Flash-Next
  pp512, pp2048 and tg128 runs dispatched no hipBLASLt kernels, but Flash-Next
  and most other models link hipBLASLt routes.
- `hipDeviceProp_t::clockRate` reports 1.00 GHz (HIP: 2.90 GHz), and the device
  name is the PCI identity instead of "AMD Radeon 8060S Graphics".
- During weight load the GPU reports 100% busy at 2.9 GHz for about 15 s, against
  about 20% at 1.7 GHz under HIP. Load time is the same, about 15.6 s.

> **Warning:** On 2026-11-10 a timed-out GPU probe followed by HRX runs ended in
> "MES failed to respond to msg=REMOVE_QUEUE", "Illegal opcode in command
> stream" and a MODE2 GPU reset; afterwards even stock HIP kernels hung until
> recovery. Use bounded fixed-iteration probe kernels, run one GPU process at a
> time and check `journalctl -k` after any timeout.

## Experimental Nix packages

Two flake outputs run Gufo on HRX. Neither is built by hosted checks or
releases.

```sh
nix build .#hrx        # HRX dd9572875f: lib/libamdhip64.so.7, hrx-info, iree-profile
nix build .#gufo-hrx   # the default Gufo binaries wrapped to load HRX
./result/bin/gufo diagnose
```

- [`.devops/nix/hrx/default.nix`](../../.devops/nix/hrx/default.nix) builds HRX
  for gfx1151 with the flake's ROCm 10.1 clang. HRX downloads its
  dependencies at configure time, so a fixed-output derivation runs
  [`hrx-deps.py`](../../.devops/nix/hrx/hrx-deps.py) `fetch` over every URL
  pinned in `MODULE.cmake.lock` and Loom's ISA XML descriptor, and the build
  rewrites those URLs to `file://` paths in that store path. HRX's own hash
  checks and dependency patches still run. After an HRX update, set the
  dependency `outputHash` to `lib.fakeHash`, build `.#hrx.passthru.deps` and
  copy the reported hash.
- Two patches apply: `hrx-dt-runpath.patch` (needed to load Gufo's code
  objects) and `hrx-stream-queues.patch` (one hardware queue per created
  stream). The probe switches in `hrx-probe.patch` are not included.
- [`gufo.nix`](../../.devops/nix/hrx/gufo.nix) wraps `gufo` and `gufo-server`
  with HRX first on `LD_LIBRARY_PATH`; the Gufo binaries are unchanged.

As of November 2026, `gufo-hrx` passes `gufo diagnose`, gives the same
Flash-Next prefill validation as HIP and measures pp512 1318, pp2048 1884 and
tg128 30.52 tok/s (one `-r 3` run).

## Profiling HRX

```sh
HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=hrx.ireeprof \
  LD_LIBRARY_PATH=<hrx-build>/libhrx/src/binding/hip \
  gufo bench -m <model> -p 2048 -n 1 -r 2
IREE_PROFILE_ALLOW_OPEN_SESSION=1 iree-profile export \
  --format=ireeperf-jsonl --output=hrx.perf.jsonl hrx.ireeprof
python3 passes.py hrx.perf.jsonl out.json
```

`iree-profile` builds from the HRX tree (`ninja iree-profile`). Its dispatch
ticks are 10.019 ns. [`passes.py`](hrx-migration/passes.py) also reads rocprofv3
`rocpd` databases, so one script compares both runtimes;
[`gaps.py`](hrx-migration/gaps.py), [`hrxtl.py`](hrx-migration/hrxtl.py),
[`hiptl.py`](hrx-migration/hiptl.py) and [`cmp.py`](hrx-migration/cmp.py) break
down gaps, queues and kernel families. Set `CXXFILT` to a `c++filt` binary.

## Migration plan

1. Track the HRX fixes, reported as of November 2026: the `DT_RUNPATH` loader
   fix ([hrx-system#2983](https://github.com/ROCm/hrx-system/issues/2983)), a
   per-stream queue policy ([#2968](https://github.com/ROCm/hrx-system/issues/2968)),
   the per-dispatch cost ([#2969](https://github.com/ROCm/hrx-system/issues/2969)),
   the host-upload cache state ([#2984](https://github.com/ROCm/hrx-system/issues/2984)),
   the `hipHostMalloc` fault ([#2985](https://github.com/ROCm/hrx-system/issues/2985)),
   the hipBLASLt code-object load failure
   ([#2986](https://github.com/ROCm/hrx-system/issues/2986)) and the device
   properties ([#2987](https://github.com/ROCm/hrx-system/issues/2987)).
2. Find the remaining gap: an instruction-level thread trace of `hcmix` and the
   `ssm` projection under both runtimes, then an unprofiled per-kernel
   comparison inside the full prefill pass.
3. Switch only after a thermally controlled A/B/A/B matches HIP on prefill and
   decode for every shipped model family, and the functional suites pass.
   Any route that calls hipBLASLt (DeepSeek V4 Flash, Qwen, EmbeddingGemma 2,
   Qwen3-ASR, MiniMax H3, Qwen-Image and some Flash-Next shapes) is blocked
   until the code-object fix from item 1 lands.
4. Then promote `.#gufo-hrx` from an experimental wrapper to a supported
   build, and keep the stock-HIP build as the comparison control.
