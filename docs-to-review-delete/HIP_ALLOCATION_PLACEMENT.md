# HIP allocation and Qwen weight placement on gfx1151

Issues #91 and #125 evaluated HIP allocation APIs and the Qwen3.8-27B Q8
production weight-placement policy on the supported Strix Halo target. The
decision is to retain mapped GGUF weights:

```text
mmap(MAP_PRIVATE) -> hipHostRegister(Mapped|ReadOnly)
                  -> hipHostGetDevicePointer
```

This policy is selected in source. There is no runtime environment flag and no
automatic copy fallback.

## Test environment

- Linux 7.0.10, ROCm 7.2.3, gfx1151
- 125.08 GiB unified LPDDR5X visible to the host
- 4 KiB base pages; persistent huge-page and NUMA settings unchanged
- Qwen3.8-27B `UD-Q8_K_XL`, one GGUF file, 31,457,991,680 bytes
  (29.299 GiB)
- machine fingerprint
  `80851c02e122209c1e37759f2b96257c8a9a009fbd48716d822881c749aefc70`

The raw JSON repetitions and process-resource reports are retained outside the
repository. The committed diagnostic emits the same schema with:

```sh
gufo diagnose --benchmark allocation --working-set-mib 64,1024 --json
```

## Allocation API findings

All five requested paths completed at 64 MiB and 1 GiB with matching
deterministic CPU/GPU checksums. The 1 GiB medians were:

| Path | Setup (ms) | First GPU read (GB/s) | Warm GPU read (GB/s) | H2D (GB/s) |
|---|---:|---:|---:|---:|
| `hipMalloc` | 35.92 | 84.53 | 83.67 | 77.16 |
| `hipMallocManaged` | 42.43 | 74.38 | 82.22 | 83.66 |
| mapped `hipHostMalloc` | 43.09 | 73.28 | 82.36 | 84.18 |
| `mmap` + `hipHostRegister` | 158.17 | 27.48 | 28.58 | 81.25 |
| `hipMallocAsync` | 45.49 | 74.07 | 74.21 | 78.21 |

`hipMallocAsync` reuse had a 0.0085 ms median after initialization. It is an
allocation/reuse result, not evidence of a distinct physical placement.
Managed-memory advice and CPU/GPU prefetch calls succeeded.

At the rounded 30,001 MiB Q8 working set, all paths again completed with
matching checksums and no swaps:

| Path | Setup (ms) | Warm GPU read (GB/s) | H2D (GB/s) |
|---|---:|---:|---:|
| `hipMalloc` | 1,073.75 | 26.14 | 74.29 |
| `hipMallocManaged` | 1,461.65 | 26.91 | 83.15 |
| mapped `hipHostMalloc` | 1,512.69 | 26.93 | 83.20 |
| `mmap` + `hipHostRegister` | 3,411.38 | 19.55 | 83.19 |
| `hipMallocAsync` | 1,258.94 | 27.17 | 81.68 |

The API microbenchmark shows that device allocations can stream faster than a
registered file mapping at 1 GiB. It does not predict the complete Qwen
execution path, where setup, prefill, decode, context preparation, and
persistent memory must be considered together.

## Qwen3.8-27B Q8 decision

Mapped and copied placement used separate release builds with the policy
selected in source. The model, prompt tokens, generated-token count, context
depths, kernels, and machine were otherwise identical. Cold runs evicted this
file's clean page-cache entries with `posix_fadvise(POSIX_FADV_DONTNEED)`; no
privilege or persistent OS change was used.

### Controlled cold pair

| Metric | Mapped | Copy |
|---|---:|---:|
| Model load | 12.06 s | 14.44 s |
| Complete 0/4K/8K/12K/16K sweep | 2:59.55 | 3:04.39 |
| Peak RSS | 31,393,280 KiB | 31,409,748 KiB |
| Major faults | 7,288 | 7,312 |
| Swaps | 0 | 0 |

| Context | Mapped prefill (tok/s) | Copy prefill (tok/s) | Mapped decode (tok/s) | Copy decode (tok/s) |
|---:|---:|---:|---:|---:|
| 0 | 554.96 | 539.17 | 7.16 | 7.27 |
| 4K | 520.64 | 511.22 | 7.11 | 7.20 |
| 8K | 497.51 | 488.56 | 7.04 | 7.11 |
| 12K | 473.24 | 466.78 | 6.97 | 7.03 |
| 16K | 447.96 | 447.82 | 6.90 | 6.94 |

### Warm pair

| Metric | Mapped | Copy |
|---|---:|---:|
| Model load | 1.17 s | 3.05 s |
| Complete 0/4K/8K/12K/16K sweep | 2:48.78 | 2:52.75 |

| Context | Mapped prefill (tok/s) | Copy prefill (tok/s) | Mapped decode (tok/s) | Copy decode (tok/s) |
|---:|---:|---:|---:|---:|
| 0 | 554.07 | 542.18 | 7.15 | 7.27 |
| 4K | 520.18 | 510.90 | 7.10 | 7.20 |
| 8K | 496.84 | 488.29 | 7.03 | 7.11 |
| 12K | 474.19 | 468.82 | 6.96 | 7.02 |
| 16K | 453.79 | 446.23 | 6.89 | 6.94 |

Copy is about 0.6-1.7% faster in decode, but mapped is faster in every warm
prefill case, wins both complete sweeps, and avoids a second
31,457,991,680-byte allocation for the encoded Q8 weights. The decode-only
gain does not offset the prefill, setup, total-time, and persistent-memory
costs, so mapped placement remains the production policy.

### Quality and route parity

Mapped and copy produced byte-identical 128-token full-vocabulary validation
metrics: both sequential and batched execution selected token 194, all logits
were finite, and cosine similarity was 0.99976891. A deterministic 16-token
greedy prompt also produced identical output.

Short profiler runs recorded 15,521 dispatches for each placement with
identical stage call counts, kernel names, and launch geometry. Placement
therefore changes weight visibility, not the selected execution routes.
Profiled timing is not used as the headline performance result.

## Inference limits

- HIP pointer attributes expose API memory types, not physical DRAM banks or
  cache residency on this integrated APU.
- `mincore` reports host-mapping residency, not GPU cache residency.
- Process fault counts can include unrelated runtime activity.
- No privileged counters, persistent OS changes, huge-page changes, NUMA
  changes, or forced power policy were used.
