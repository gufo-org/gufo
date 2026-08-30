# Qwen3.8 27B DFlash2 XDNA2 Experiment

Date: 2026-08-30

## Result

The XDNA2 vocabulary-head offload is rejected for production use. The final
one-shard route preserved exact target output, but increased DFlash head latency
and did not improve end-to-end throughput. The normal DFlash2 GPU path remains
unchanged.

The isolated AIE artifact and hardware test are retained as a reproducible
calibration harness. No prompt, bench, server, model, or speculative-runtime
option selects the artifact.

## Baseline and candidate

The target was Qwen3.8-27B Q8_K_XL with the Qwen3.8-27B DFlash2 Q8_0 draft on
Strix Halo (`gfx1151` GPU and XDNA2 NPU).

Profiling showed:

- the complete GPU DFlash block took 23.480 ms;
- the GPU vocabulary head took 6.268 ms;
- the small-batch Q8_0 GEMM family accounted for 57.1% of profiled GPU kernel
  time;
- the measured XDNA2 projection rate was about 39.5 GB/s.

A full serial NPU port would need to stream roughly 2.06 GB of draft weights,
giving a lower bound near 52 ms. Per-layer offload was also rejected because
the repeated GPU/NPU boundaries exceeded the small stages they replaced.

The implemented candidate split the final vocabulary projection by output
rows. XDNA2 computed a contiguous 8,192-row Q8_0 shard while the GPU computed
the remaining rows, followed by the existing GPU selector. The artifact uses
all eight NPU2 columns, dynamic INT8 activations, INT32 accumulation, and FP32
output for a maximum batch of eight rows.

## Artifact validation

The final 8,192-row program:

- builds with Nix;
- produces exact synthetic identity output (`max_abs=0`);
- takes 1.233 ms for the XDNA2 command and 1.303 ms end-to-end in the focused
  hardware test;
- keeps the AIE implementation and test-only XRT session outside the production
  runtime.

Larger single-command variants also built and were exact:

| NPU rows | Standalone command | Standalone end-to-end | Integrated head |
| ---: | ---: | ---: | ---: |
| 8,192 | 1.233 ms | 1.303 ms | 6.105 ms median |
| 24,576 | 3.400 ms | 3.551 ms | 6.402 ms |
| 32,768 | 4.482 ms | 4.663 ms | 6.595 ms |

Four independent XRT contexts were much worse at 18.004 ms integrated head
latency. Submitting XDNA2 before the GPU complement also regressed the
24,576-row route to 6.541 ms.

## Interleaved A/B

The final release comparison used three interleaved GPU/NPU pairs with a
128-token prompt and 32 generated tokens:

| Route | Head samples | Median | Throughput samples |
| --- | --- | ---: | --- |
| GPU DFlash2 | 5.951, 5.955, 5.957 ms | 5.955 ms | 11.11, 11.09, 11.09 tok/s |
| GPU + XDNA2 | 6.093, 6.105, 6.110 ms | 6.105 ms | 11.10, 11.08, 11.06 tok/s |

The hybrid head was 0.150 ms, or 2.5%, slower. Acceptance was identical at
14.7% in all six runs.

The integrated processor's shared memory is the limiting factor: the standalone
NPU kernel is fast enough, but concurrent NPU weight streaming slows the
remaining GPU projection enough to erase the offload.

## Quality

The ten-prompt speculative corpus passed exact target-output comparison:

- exact completions: 10/10;
- aggregate acceptance: 45.0%;
- average draft width: 7.00;
- autoregressive throughput: 6.68 tok/s;
- hybrid speculative throughput: 18.51 tok/s.

The target verifier therefore preserved quality, but quality parity does not
override the negative A/B performance result.

## Reproduction

```sh
nix build .#aie-qwen-dflash-head
nix build
nix build .#checks.x86_64-linux.pr
```

Use `qwen_dflash_head_test` from the focused hardware-test CMake tree for NPU
correctness and the release `result/bin/gufo` binary for model measurements.
