# EmbeddingGemma 2 benchmarks

Measured on 2026-10-23 on Linux with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors, a resident model, and the production `gufo` from a Nix container image.  The baseline build is #700 at its current head, which includes the device turns.  Each build ran its rows in two passes on one node, the only GPU pod on it, with passes of builds from the sibling changes between them.  Each row is 60 requests per pass, the table shows the mean of the two passes' medians, and token counts include `<bos>` and `<eos>`.

This build's first pass stopped at the 16 × 1 × 16 row with one HTTP 503 that the server never logged.  That is a race in the HTTP server's connection limit and is unrelated to this change, so the pass left one sample for the 4 × 8 × 128 and 16 × 8 × 128 rows.  Those two rows come from a second pod on the same node after a reboot, with three passes per build.  The [record](artifacts/gfx1151-2026-10-23-gemm-ab.json) holds both builds' rows, and the [earlier #700 record](artifacts/gfx1151-2026-10-23-nix.json) holds #700's own measurement.

The numbers for the matrix-core attention kernel are in the [Matrix-core attention](#matrix-core-attention) section.  The table below is main with the hipBLASLt GEMMs and the vector-unit attention kernel.

Model load takes **2.2 s**, against 0.53 s for the baseline build.  The baseline's time is mostly the 540 MB upload, and the other 1.7 s is the warm-up that times each hipBLASLt algorithm for every GEMM shape and row bucket.

## HTTP

These rows time `gufo serve embedding` on loopback with [`http_bench.py`](../../../tools/models/embedding_gemma2/http_bench.py).  Each input is a run of distinct `word<N>` words.

| Concurrency × inputs × words | Tokens/request | Median | p95 | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 6.69 ms | 7.54 ms | 5,805 |
| 1 × 1 × 128 | 404 | 20.45 ms | 21.49 ms | 19,507 |
| 1 × 1 × 512 | 1,940 | 147.96 ms | 151.34 ms | 13,101 |
| 1 × 8 × 128 | 3,495 | 149.16 ms | 150.31 ms | 23,371 |
| 1 × 32 × 128 | 15,670 | 708.58 ms | 711.57 ms | 22,088 |
| 4 × 1 × 128 | 404 | 78.93 ms | 80.81 ms | 20,337 |
| 16 × 1 × 128 | 404 | 313.23 ms | 317.89 ms | 20,462 |
| 4 × 8 × 128 | 3,495 | 584.30 ms | 588.65 ms | 23,896 |
| 16 × 8 × 128 | 3,495 | 2,342.86 ms | 2,346.70 ms | 23,840 |

Against #700, this build is 1.5 to 2.3 times faster at the median on every row of this table, and the record lists each speedup.  Concurrent requests share one device and take turns one batch at a time, so concurrency adds queueing time but not throughput.  A long request yields the device after each batch, so a short request queued behind it waits for one batch instead of the whole request.  The record keeps each pass's median.

## GEMM benchmark

`gemm_bench_hip_test` times each encoder GEMM shape through hipBLAS and through every hipBLASLt algorithm that needs no workspace.  These are kernel timings, not request latency.  At 1,024 rows, four of the shapes compare as follows.

| Shape | hipBLAS | Best hipBLASLt |
| --- | ---: | ---: |
| `gate_up` | 0.88 ms, 4.86 TFLOPS | 0.16 ms, 27.01 TFLOPS |
| `down` | 0.46 ms, 4.69 TFLOPS | 0.08 ms, 25.60 TFLOPS |
| `ple_512` | 0.12 ms, 4.49 TFLOPS | 0.03 ms, 20.98 TFLOPS |
| `final` | 0.17 ms, 4.72 TFLOPS | 0.05 ms, 17.41 TFLOPS |

## Matrix-core attention

Measured on 2026-10-24 on the node gpuworker01 with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors and the production `gufo` from one Nix container image that holds both builds.  Main is commit 479162ed, which has the hipBLASLt GEMMs from #751 and image inputs from #759.  This PR is commit ec21a41f, which differs from the PR head only in documentation.  The pod was the only GPU pod on the node.  The builds ran in the order main, this PR, this PR, main, with 60 requests per row in each pass, and the table shows the mean of the two passes' medians.  Speedup is main's median divided by this PR's median.  The text rows use `http_bench.py` with the labels of the HTTP table above.  The image rows send one 320 × 180 PNG followed by a newline in the typed `content` format, which is 269 tokens, at concurrency 1 and 4 after one warm-up request.

| Concurrency × inputs × words | Tokens/request | Main median | This PR median | Speedup | This PR tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 6.74 ms | 6.64 ms | 1.02 | 5,914 |
| 1 × 1 × 128 | 404 | 19.95 ms | 12.94 ms | 1.54 | 30,391 |
| 1 × 1 × 512 | 1,940 | 144.56 ms | 61.50 ms | 2.35 | 31,477 |
| 1 × 1 × 2100 | 8,072 | 1,650.55 ms | 341.70 ms | 4.83 | 23,589 |
| 1 × 8 × 128 | 3,495 | 148.01 ms | 94.27 ms | 1.57 | 36,954 |
| 1 × 32 × 128 | 15,670 | 701.38 ms | 397.00 ms | 1.77 | 39,366 |
| 4 × 1 × 128 | 404 | 77.88 ms | 48.45 ms | 1.61 | 32,919 |
| 16 × 1 × 128 | 404 | 308.69 ms | 192.42 ms | 1.60 | 33,264 |
| 4 × 8 × 128 | 3,495 | 574.09 ms | 346.99 ms | 1.65 | 40,215 |
| 16 × 8 × 128 | 3,495 | 2,297.35 ms | 1,388.06 ms | 1.66 | 40,139 |
| 1 × 1 image | 269 | 585.69 ms | 581.39 ms | 1.01 | 463 |
| 4 × 1 image | 269 | 2,254.74 ms | 2,234.28 ms | 1.01 | 481 |

Image requests change by about 1 percent because the vision tower, which this PR does not change, takes most of their time.  Server load took 1.98 s for this PR and 2.20 s for main.  The [record](artifacts/gfx1151-2026-10-24-attention-ab.json) holds each pass's median, the p95 values, and the quality and test results.

## Reference comparison

The llama.cpp comparison is TODO.  It requires a llama.cpp build with EmbeddingGemma 2 support and the matching GGUF.  Measure both over HTTP with the same texts and report the ratio per row.

## Image support qualification (2026-10-24)

Requalified after PR review on main `1fafd5ad`, preserving its hipBLASLt text
GEMMs and startup warm-up. Both the candidate and unchanged-main control use
the production release CMake preset with the same pinned Nix compiler and
libraries. The image request holds a 320×180 RGB PNG followed by a newline in
the typed `content` format: 264 image soft tokens and **269 total tokens**.
One resident warm-up precedes 60 sequential HTTP requests. Timing includes
PNG reading, preprocessing, vision/text inference and response transport.

| Input | Requests | Median | p95 | Requests/s |
| --- | ---: | ---: | ---: | ---: |
| One 320×180 PNG + newline | 60 | **545.31 ms** | **548.47 ms** | **1.84** |

This is a new workload with no supported main image path to compare. The first
image request additionally loads **335.5 MB** of vision/projector weights; text
startup still loads only the text backbone. Audio weights are never uploaded.

The affected text controls retain every request. Main and candidate alternate
per workload, with 60 requests each and no per-request failures at the 5% and
3 ms thresholds. These are controls, not a speedup claim; startup's measured
algorithm selection can differ between processes.

| Inputs × words | Tokens/request | Main median | Candidate median |
| --- | ---: | ---: | ---: |
| 1 × 16 | 40 | 6.63 ms | 6.49 ms |
| 1 × 128 | 404 | 19.13 ms | 18.97 ms |
| 8 × 128 | 3,495 | 146.23 ms | 139.25 ms |

The [review qualification record](artifacts/gfx1151-pr759-review.json) retains
all request timings, image checks and CPU/GPU test results. The
[pre-rebase record](artifacts/gfx1151-2026-10-24-images.json) is historical
evidence against worktree revision `8e7979cc`, which was not origin/main. Its
initial isolated 152.8 ms outlier remains marked inconclusive and is not used
to qualify the rebased PR. Earlier published text benchmark rows above remain
their original measurements.
