# LFM3 benchmarks

## October 29, 2026: current

Same machine, toolchain, weights and harness as below. Production `release`
build at commit `2bf3140d` against the preserved October 28 binary (`base`,
commit `54d260b3`). Q8_0, FP32 KV, C1, single runs per depth; depth rows use
pp512/tg128 after preparing the stated history. Tokens/s.

| Test | Depth | Base | Current |
| --- | --: | --: | --: |
| pp2048 (3 reps) | 0 | 2424–2431 | 4961 / 4997 / 4986 |
| tg128 (3 reps) | 0 | 97.9–98.0 | 105.6 / 105.7 / 105.6 |
| pp512 | 4096 | 1563 | 4240 |
| tg128 | 4096 | 36.5 | 98.9 |
| pp512 | 16384 | 431 | 2893 |
| tg128 | 16384 | 13.1 | 85.1 |
| pp512 | 32768 | 236 | 2073 |
| tg128 | 32768 | 7.0 | 73.7 |
| pp512 | 65536 | — | 1320 |
| tg128 | 65536 | — | 57.4 |
| pp512 | 131072 | — | 595 |
| tg128 | 131072 | — | 38.7 |

Base was not run beyond 32K (its 32K row alone takes minutes). At 128K,
decode reads 3.2 GB of FP32 K/V per token beside 1.84 GB of weights; the
current 25.8 ms per token is about 88% of that bandwidth floor. Prompt
attention at depth runs at about 5.8 TFLOPS FP32 and dominates long prefills.
Repeated runs on this host vary by about 2% with temperature.

## October 28, 2026

October 28, 2026, Linux x86-64 / Ryzen AI Max+ 395, `gfx1151`, 128 GB unified
memory. Production CMake `release` builds, GCC 16.2 / ROCm 10.1 HIP Clang 24.
Official BF16 and Q8_0 GGUF revision
`b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf`. No speculative decoding.

## Timed scope

CLI C1, pp2048 and tg128 at depth zero, three retained repetitions with the
same token histories and settings per precision. Prompt shapes and decode are
warmed first. Prefill times model execution without the vocabulary projection;
decode includes the vocabulary projection, host logit download and greedy
sampling. Model load, session allocation and prefix preparation are outside the
timers. FP32 KV/history are used in both formats. Timings are per phase and
are not combined across requests.

The current Q8 control is the preserved native 1291 tok/s production binary,
built with the same toolchain and benchmark harness. The candidate retains
packed Q8 weights, fuses three digit products with one-pass activation encoding,
uses one-wave attention with eight-row prefetch, and increases prefill chunks
from 128 to 512. Both use FP32 KV and history. Decode projections are unchanged.

## Results

Median per-phase throughput is shown for convenience; every individual phase
and repetition is retained in [the measurement artifact](artifacts/benchmarks.json),
under `optimization_requests`. Earlier implementation measurements remain in
`requests` as historical evidence.

| Precision | Control pp2048 | Current pp2048 | Control tg128 | Current tg128 |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 1284.29 | 2348.62 | 94.51 | 97.79 |
| BF16 | — | 488.05 | — | 39.84 |

Units are tokens/s. Q8 prefill improves about 83% over the matched control.
Every candidate prefill phase improves beyond the 5% / 10 ms timing gates;
no decode phase regresses. Candidate prefill repetitions span 2338–2357 tok/s.
BF16 numbers describe the current build only; no new matched BF16 speedup is
claimed. The 5000 tok/s prefill target is not achieved.

## Scope and remaining work

Q8_0 is the deployment target; BF16 is the numerical reference. F16 has only a
compatibility check and no retained speed result. No C2+ throughput or deep
context speed claim is made. HTTP correctness covers independent sessions,
and physical autoregressive dispatch now batches up to eight sessions.

A separately pinned llama.cpp GPU build was used to identify prefill costs.
Its ordinary benchmark uses random tokens, FP16 KV and quantized activations;
those results are diagnostic and are not a matched speedup comparison. Its
prefill remains faster than this higher-precision activation path. Full traces
and diagnostics stay under ignored `artifacts/lfm3/`.

## Serving integration verification

After wiring snapshots, native tools and batched decode, the standard Q8 C1
pp2048/tg128 benchmark was repeated against the preserved wired serial binary.
Each of the three candidate phases stayed within both the 5% and 10 ms gates:
prefill 2362.33 / 2366.20 / 2357.56 tok/s, decode
97.7137 / 97.7105 / 97.2564 tok/s. Corresponding control phases were
2365.13 / 2377.46 / 2364.43 and 97.9737 / 98.0486 / 97.6067 tok/s.
This checks that integration retained C1 performance; it is not a new speedup
claim or a C2+ throughput result. Raw phase timings and binary hashes are in
`serving_wiring_verification` in the artifact and local
`artifacts/lfm3/wired-cli-benchmark-final/`. The 5000 tok/s target remains open.
