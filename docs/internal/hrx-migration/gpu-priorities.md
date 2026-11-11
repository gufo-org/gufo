# HRX priorities for regular GPU models

November 11, 2026: prioritize measured GPU gaps in Gufo's regular models.
The investigation also found and fixed a Gufo benchmark error that affected
our ability to judge HRX. No additional HRX throughput improvement is claimed.

## Fix: propagate Qwen benchmark synchronization failures

Qwen's `restore_depth` and `prefill` callbacks used `HIP_CHECK`, whose error
branch returns `1`. This made each callback return `int`, while successful
calls reached its end without returning. That is undefined behavior. The
production build printed `HIP error: no error` even when the intercepted HIP
call returned success. A real synchronization failure returned from the
callback alone; the benchmark ignored it and still printed a throughput row.

Both callbacks now return `hipError_t` on every path. The outer benchmark
checks their status and returns an error before recording a failed run.
The existing Qwen CLI benchmark test also rejects HIP errors on successful
runs. Inference arithmetic, kernels and serving behavior do not change.

**Gufo gains trustworthy benchmark results and error propagation.** This is
a Gufo fix, rather than an HRX runtime fix or a measured inference speedup.
It belongs on Gufo's `hrx-staging`, with its own fix branch.

The fault-injection shim delegates to the real HIP synchronization first,
then substitutes `hipErrorInvalidValue` at a selected call. It confirms:

- The old executable prints a pp32 result after the injected restore failure.
- The candidate exits with code 1 and prints no pp32 result for failures at
  either warmup or measured restore/prefill: four cases on HIP and four on HRX.
- Successful runs produce no HIP errors. AR and DFlash2 at depths 0 and 32
  produce identical eight-token hashes on HIP and HRX. DFlash2 reports actual
  speculative acceptance in both runtimes.

These are focused benchmark checks, not a new model numerical qualification.
Native MTP was not executed. The changed prefill branches are covered by AR
and DFlash2; their decoder kernels are unchanged. The Nix production build
and shared C++ formatting check pass.

## Fresh Flash-Next control

HRX `8ebcdb6b70` versus stock ROCm 10.1 HIP, identical Gufo `9612db2e5`
production executable, Flash-Next UD-Q4_K_XL, native AR benchmark, depth zero,
seed 1, three repetitions per process. Loading is excluded. Each process
starts after cooling to at most 38 °C and 60 seconds idle, in the performance
profile, with no competing model/compiler. These are CLI benchmark rates,
not HTTP request throughput. Values are means ± standard deviations in tok/s.

| Pair | Phase | HIP | HRX | HRX relative to HIP |
| --- | --- | ---: | ---: | ---: |
| A | pp512 | 1336.47 ± 0.63 | 1275.89 ± 0.92 | -4.53% |
| A | pp2048 | 1754.91 ± 258.96 | 1843.14 ± 4.44 | Inconclusive |
| A | tg128 | 30.44 ± 0.21 | 30.40 ± 0.13 | -0.13% |
| B | pp512 | 1326.52 ± 5.47 | 1273.70 ± 2.70 | -3.98% |
| B | pp2048 | 1901.26 ± 1.47 | 1836.91 ± 6.08 | -3.38% |
| B | tg128 | 30.48 ± 0.19 | 30.39 ± 0.14 | -0.30% |

The first HIP pp2048 process is noisy and remains inconclusive. Pair B resolves
that phase without averaging pairs. All four processes pass the existing
512-token prefill check with identical metrics: finite=1, scalar winner rank=1,
RMSE=0.520053, cosine=0.991351, max error=3.81343.

The Qwen benchmark callback bug is in a different model driver. It does not
invalidate these Flash-Next controls.

## Corrected Qwen 27B control

One matched pair uses the corrected Nix production executable, Qwen3.8-27B
UD-Q4_K_XL (reported mixed Q5_K), AR, depth zero, pp2048/tg128, seed 1 and
three repetitions. Cooling, isolation and power settings match the Flash
control. Both processes pass the 512-token prefill check with identical
reported metrics and no HIP errors.

| Phase | HIP (tok/s) | HRX (tok/s) | HRX relative to HIP |
| --- | ---: | ---: | ---: |
| pp2048 | 635.31 ± 0.61 | 638.62 ± 0.33 | +0.52% |
| tg128 | 12.36 ± 0.00 | 12.32 ± 0.01 | -0.32% |

This pair provides no compelling Qwen runtime optimization target. The
small differences are not gains from the callback fix. The earlier Qwen
run has undefined behavior and is retained as invalid performance evidence.

The existing complete benchmark test stops earlier at a separate, pre-existing
failure: Qwen accepts non-greedy sampling controls but always benchmarks greedy
decoding. The callback checks above pass independently; the sampling contract
needs a separate fix before the complete benchmark test can pass.

## Next GPU targets

- **Flash-Next prefill:** investigate the confirmed roughly 3–4% deficit,
  starting with the previously reported HC-mixing/projection gap. Fresh
  per-kernel evidence is required before choosing a mechanism. The older
  7% isolated HC-mixing result is not a new gain estimate or a guarantee
  of a full-model improvement.
- **Eager submission:** the current empty-kernel probe still trails HIP,
  but older Flash-Next profiles put launch gaps below 1% of prefill. A
  completion-spin experiment improved the complete diagnostic by only
  about 1.2%; it does not establish a model gain and was discarded.
- **Completion interrupts:** a coalescing prototype failed the final-operation
  progress check. It was discarded and the runtime sources restored. Any
  future design must guarantee completion when the last launch is the final
  API operation; a faster isolated launch is insufficient.

Pinned-host reads and hipBLASLt code-object loading are already marked fixed
in issue #1144. Do not prioritize them from the older migration notes.
NPU compiler work is outside the selected priority.

## Evidence

[Raw logs, commands, fault shim and build identities](gpu-priorities-2026-11-11/)
retain both Flash pairs, the old invalid Qwen run and every focused check.
Fault-injection numbers are not used as performance results. The shim source
can be built with `cc -O2 -shared -fPIC qwen_sync_fault.c -o fault.so -ldl`;
set `LD_PRELOAD` to its absolute path and `GUFO_SYNC_FAIL_AT` to 2, 3, 4 or 5
for `gufo bench -m <Qwen27B> -p 32 -n 0 -r 1`. The recorded runner scripts
contain the original host paths; replace them when replaying elsewhere.
