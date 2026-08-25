# Hardware Memory Bandwidth Baseline

This document retains the hardware memory bandwidth baseline measurements for AMD Strix Halo on Linux x86-64.

## Machine Fingerprint

- **Fingerprint ID**: `bb565d5eff3a9f23b4ac3f1ff03f66bebef651e57093cd558c4cded823358849`
- **CPU**: AMD Ryzen AI MAX+ 395 (16 physical cores, 32 logical threads)
- **GPU**: AMD Radeon 8060S (`gfx1151`, 40 CUs, `amdgpu`)
- **NPU**: AMD XDNA2 / AIE2P (32 spatial tiles, `amdxdna`, `xrt` 2.21.0)
- **Unified Memory**: 128 GiB LPDDR5X (Unified System Pool)
- **Toolchain Pins**: ROCm 7.2.3, GCC 15.3.0, Linux 7.1.8

## Methodology & Rules

- **Working Set**: Minimum 256 MiB per test (exceeding all CPU L1/L2/L3 caches and on-chip SRAM to measure sustained external unified DRAM bandwidth).
- **Correctness Sentinels**: Output memory arrays are filled with deterministic pseudorandom test patterns and verified before bandwidth measurements count.
- **Timing**: High-resolution clock measurement across warmups and repeated iterations reporting minimum, median, and P95 sustained GB/s.

## Baseline Results Summary

| Backend | Path Name | Allocation Type | Working Set | Median (GB/s) | P95 (GB/s) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | `cpu_copy` | `host_pageable` | 256 MiB | 50.62 | 51.11 | `completed` |
| CPU | `cpu_read` | `host_pageable` | 256 MiB | 13.88 | 13.89 | `completed` |
| CPU | `cpu_write` | `host_pageable` | 256 MiB | 37.96 | 38.37 | `completed` |
| HIP | `hip_h2d` | `hip_device_memory` | 256 MiB | 68.36 | 68.56 | `completed` |
| HIP | `hip_device_copy` | `hip_device_memory` | 256 MiB | 211.03 | 211.63 | `completed` |
| HIP | `hip_d2h` | `hip_device_memory` | 256 MiB | 55.89 | 56.42 | `completed` |
| XRT | `xrt_bo_sync_to_device` | `xrt_bo_dma_sync` | 64 MiB | 127.06 | 127.75 | `completed` |
| XRT | `xrt_bo_sync_from_device` | `xrt_bo_dma_sync` | 64 MiB | 120.35 | 127.74 | `completed` |

## Reproduction Command

```sh
./result/bin/strix diagnose \
  --benchmark bandwidth \
  --backends cpu,hip,xrt \
  --warmup 3 \
  --repetitions 10 \
  --duration-ms 2000 \
  --json \
  --output /tmp/strix-bandwidth.json

./result/bin/strix diagnose --validate-artifact /tmp/strix-bandwidth.json
```
