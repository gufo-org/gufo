# Ornith 1.5 122B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo host builds with ROCm
7.2.2 unless noted, `Ornith-1.5-122B-Q4_K_M.gguf`. Measured October 21, 2026,
with each comparison inside one GPU admission hold so no other model shared
the GPU. llama.cpp is `23f7aab` (`llama-bench -ngl 999 -fa 1`).

The first repetition in a fresh Gufo process is often slow (clock ramp and
first-touch faults). Rows marked "clean" either drop it or come from rounds
where it did not occur; treat any row with a large ± as unreliable.

## Prefill and decode, MTP off

| Workload | Gufo | llama.cpp |
| --- | ---: | ---: |
| pp512 (t/s) | about 680 | about 510 |
| tg128 (t/s) | 28.2 | 27.2 |

Gufo at the commit that routes Q4_K routed experts to the MMQ vector kernels.
Three later A/B rounds of the full stack measured tg128 28.11 to 28.18 and
pp512 689 to 695.

## Decode at context depth, MTP off

tg64 after the prompt is prepared to the given depth.

| Depth | Gufo, generic attention | Gufo, 16-head shared KV | llama.cpp |
| ---: | ---: | ---: | ---: |
| 0 | about 28.3 | about 28.3 | 26.7 to 27.2 |
| 16384 | 24.77 / 24.68 | 25.39 | 24.35 / 24.34 |
| 32768 | 21.79 / 21.65 | 22.91 / 22.85 | 22.10 / 22.08 |

Without the shared-KV variant, Gufo's split-K partials took 1069 us per call
at depth 32768, against 889 us for llama.cpp's `flash_attn_tile`. Clean rounds
only; one round of each arm had a slow repetition and is excluded.

## MTP draft depth

Fresh server per arm, six prompts of 256 tokens at temperature 0, thinking
off, 4 GiB RAM cache.

| Draft tokens | Median 256-token decode | Accepted |
| ---: | ---: | ---: |
| 2 | 8364 ms (30.6 t/s) | 1218 / 2162 (56.3%) |
| 3 | 8629 ms (29.7 t/s) | 1286 / 2614 (49.2%) |

Replies were identical across all arms. MTP-off tg128 on the same build was
about 27.3 t/s.

## First token after load

Three fresh loads per arm, eight chat requests each (six distinct prompts of
16 to 19 tokens, two exact repeats), MTP depth 2.

| Weights | Load | Uncached first token, median / p90 / max | Over 1 s |
| --- | ---: | ---: | ---: |
| Registered host copy | 29 to 35 s | 282 / 633 / 1077 ms | 2 of 18 |
| Device memory | 21 to 23 s | 285 / 356 / 497 ms | 0 of 18 |

With the registered copy, each slow request coincided with kernel compaction
migrating pages (up to 2.4 million pages in one 6 s stall). In those windows
one routed-expert GEMM that normally takes about 1 ms ran for 2 to 7 s. The
device-memory runs had 1.1 million pages migrated during their uncached
requests and no stall. Cache hits answered in 3 to 5 ms in both arms.

## Not measured

- The repository benchmark driver (`tools/bench/model-bench.py`) and its
  pp2048 prose and copying workloads.
- Concurrency above one session.
- Prefill at depth against llama.cpp.
