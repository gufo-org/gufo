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

## Cross-request batching

Measured on 2026-10-24 on the node gpuworker01 with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors and the production `gufo` from one Nix container image that holds both builds.  Main is commit 194db882, which has the hipBLASLt GEMMs from #751, image inputs from #759, and matrix-core attention from #770.  This PR is commit 9a06aa43, which differs from the PR head only in documentation, a removed include, and a rewrite of the return path of `EmbedMultimodal` that does not change behavior.  The pod was the only GPU pod on the node.  Each build ran its nine EmbeddingGemma 2 GPU test binaries, then four HTTP passes in the order main, this PR, this PR, main, main, this PR, this PR, main, each with a new server and 60 requests per row.  The table shows the mean of the four passes' medians, and speedup is main's median divided by this PR's median.  The text rows use `http_bench.py` with the labels of the HTTP table above, including the new 16 × 1 × 16 row.  The image rows send one 320 × 180 PNG followed by a newline in the typed `content` format, which is 269 tokens, at 1, 4, and 16 clients.  The earlier tables predate cross-request batching, so in them concurrent requests take turns one batch at a time.  With this change, text requests that wait for the device share batches of up to 8,192 tokens, oldest first, and a lone request runs at once.

| Concurrency × inputs × words | Tokens/request | Main median | This PR median | Speedup | This PR tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 6.63 ms | 6.59 ms | 1.01 | 5,900 |
| 1 × 1 × 128 | 404 | 12.88 ms | 13.04 ms | 0.99 | 30,372 |
| 1 × 1 × 512 | 1,940 | 61.08 ms | 61.23 ms | 1.00 | 31,589 |
| 1 × 1 × 2100 | 8,072 | 339.43 ms | 337.65 ms | 1.01 | 23,876 |
| 1 × 8 × 128 | 3,495 | 93.51 ms | 94.10 ms | 0.99 | 37,037 |
| 1 × 32 × 128 | 15,670 | 394.59 ms | 394.56 ms | 1.00 | 39,608 |
| 4 × 1 × 128 | 404 | 48.16 ms | 45.82 ms | 1.05 | 34,991 |
| 16 × 1 × 16 | 40 | 95.96 ms | 27.16 ms | 3.53 | 21,368 |
| 16 × 1 × 128 | 404 | 192.38 ms | 159.16 ms | 1.21 | 39,755 |
| 4 × 8 × 128 | 3,495 | 343.98 ms | 335.15 ms | 1.03 | 41,479 |
| 16 × 8 × 128 | 3,495 | 1,377.72 ms | 1,340.11 ms | 1.03 | 41,501 |
| 1 × 1 image | 269 | 584.58 ms | 583.75 ms | 1.00 | 461 |
| 4 × 1 image | 269 | 2,246.65 ms | 2,246.21 ms | 1.00 | 479 |
| 16 × 1 image | 269 | 8,984.98 ms | 8,979.05 ms | 1.00 | 479 |

Sixteen clients that each send one 16-word input are 3.53 times faster, at 27.16 ms against 95.96 ms, because up to sixteen of those requests now share one device batch.  Sixteen clients of 128 words are 1.21 times faster and four clients are 1.05 times faster.  The concurrent rows of 8 inputs, which already fill a batch, are about 1.03 times faster.

The single-client rows are within 1.3 percent of main.  The three shortest single-input rows differ by at most 0.16 ms, and 1 × 1 × 2100 is 1.78 ms faster.  On 1 × 8 × 128 this PR is 0.59 ms slower, which is 0.6 percent.  It is the only single-client row where the four passes of the two builds do not overlap, with main between 93.21 and 93.82 ms and this PR between 93.91 and 94.63 ms.  Image-only rows are unchanged at every client count, because an image request runs in a batch of its own and the vision tower sets its pace.  Server load took 1.99 s for this PR and 1.98 s for main.

| Mixed measurement | Main median | This PR median | Speedup | Main p95 | This PR p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Alternating, image requests | 5,094.55 ms | 4,952.76 ms | 1.03 | 6,767.15 ms | 7,315.21 ms |
| Alternating, text requests | 4,193.52 ms | 3,974.50 ms | 1.06 | 5,651.22 ms | 5,929.67 ms |
| Separate streams, image requests | 4,513.24 ms | 4,488.04 ms | 1.01 | 4,533.84 ms | 4,499.03 ms |
| Separate streams, text requests | 3,953.90 ms | 3,928.41 ms | 1.01 | 4,511.70 ms | 4,460.02 ms |

In the alternating row, 16 client threads take 120 requests that alternate an image and a 16-word text request, so each kind has 60 requests per pass.  The medians were 1.06 times faster for text and 1.03 times faster for images.  The image p95 was 8 percent higher on average, and every pass of this PR, at 7,278 to 7,406 ms, was above every pass of main, at 6,731 to 6,796 ms.  The text p95 was 4.9 percent higher on average.  Two of this PR's passes were above every pass of main, at 6,281 and 6,178 ms, and the other two were at 5,636 and 5,623 ms, inside main's range of 5,606 to 5,681 ms.

In the separate-streams run, 8 threads each send 8 image requests while 8 other threads send 16-word text requests until the image threads finish, with four passes per build in the same order.  Each value in the table is the mean of the four passes.  The image median, p95, and mean match main or are slightly lower, with means of 4,242.39 ms for this PR and 4,272.37 ms for main.  The text median and p95 match main, and the text mean is 3,546.61 ms against 3,529.07 ms.  The image p95 did not rise as it did in the alternating row.  The alternating row couples the two kinds through shared client threads, so faster text replies may make those threads send images sooner.  That is untested.

Under either mixed load, a text request that queues behind images waits about 4 s on both builds, because the queue is first come, first served and each image takes about 560 ms.  This PR does not change that.

Both functional suites passed on both builds.  The concurrent-versus-lone check of `tests/functional/embeddings.py` has a worst cosine of 0.99997162 on this PR.  The [record](artifacts/gfx1151-2026-10-24-batching-ab.json) holds each pass's median and p95, the separate-streams passes, and the quality results.

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
startup still loads only the text backbone. Audio weights load lazily on the first audio request; see the audio measurements below.

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

## Audio, October 26, 2026

Production release binary `cb49012f75b4`, checkpoint and official reference pins
as recorded in [quality](QUALITY.md#audio-and-interleaved-input-parity).
Loopback HTTP C1 server durations below keep each request separately. The first
audio request includes lazy tower/projector loading; subsequent requests reuse
weights and bounded scratch. Text and vision were already resident.

| Input | First request (ms) | Second (ms) | Third (ms) |
| --- | ---: | ---: | ---: |
| Generated one-second tone, 29 expanded tokens | 146.9 | 28.3 | 27.8 |
| Ten-second English speech, 254 tokens | 99.8 | 99.7 | 100.4 |
| Two clips with text, 35 tokens | 46.8 | 47.2 | 47.1 |
| Audio/image/text, 288 tokens | 511.4 | 498.6 | 498.1 |

An isolated native test measured **1,410,707,456 bytes (1.31 GiB)** less free
device memory after constructing the audio runtime, including widened float32
weights, fixed scratch and BLAS reservations. This is a `hipMemGetInfo` allocation
delta, not process RSS. Audio is float32 to retain the independent stage contract;
text and vision keep their existing numerical paths.

[Raw measurements](artifacts/gfx1151-2026-10-26-audio.json) retain all runs and
all standard HTTP benchmark requests (60 per workload). The first candidate pass
overlapped the CPU contract workload, so that standard comparison is
**inconclusive**, without a speed claim. After it finished, matched main
`9dd2d24d` and candidate controls repeated only the affected C1 histories:
40/404/1,940-token text inputs and a 256-token image. Every individual request's
server and client time passes the unchanged 5% and 10 ms margins. All vectors
are byte-identical and work counts match. The two unchanged-main runs are
retained as controls; the first inconclusive observations are not discarded.
