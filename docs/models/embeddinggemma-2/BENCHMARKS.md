# EmbeddingGemma 2 benchmarks

Measured on 2026-10-23 on Linux with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors, a resident model, and the `nix build` production `gufo`.  Each row is 60 requests, and token counts include `<bos>` and `<eos>`.  The [record](artifacts/gfx1151-2026-10-23-nix.json) holds the evidence.

Model load takes **0.62 s**, which is mostly the 540 MB upload.

## HTTP

These rows time `gufo serve embedding` on loopback with [`http_bench.py`](../../../tools/models/embedding_gemma2/http_bench.py).  Each input is a run of distinct `word<N>` words.

| Concurrency × inputs × words | Tokens/request | Median | p95 | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 10.04 ms | 10.17 ms | 3,944 |
| 1 × 1 × 128 | 404 | 36.50 ms | 37.29 ms | 11,035 |
| 1 × 1 × 512 | 1,940 | 230.29 ms | 232.42 ms | 8,467 |
| 1 × 8 × 128 | 3,495 | 311.00 ms | 312.47 ms | 11,222 |
| 1 × 32 × 128 | 15,670 | 1,439.87 ms | 1,443.38 ms | 10,881 |
| 4 × 1 × 128 | 404 | 149.40 ms | 154.54 ms | 10,724 |
| 16 × 1 × 128 | 404 | 597.34 ms | 601.56 ms | 10,761 |
| 4 × 8 × 128 | 3,495 | 1,227.92 ms | 1,233.70 ms | 11,370 |
| 16 × 8 × 128 | 3,495 | 4,921.57 ms | 5,006.96 ms | 11,275 |

Concurrent requests share one device and take turns one batch at a time, so concurrency adds queueing time but not throughput.  A long request yields the device after each batch, so a short request queued behind it waits for one batch instead of the whole request.  Concurrent rows depend on how long the GPU has been busy: a second run right after this one, on a fresh server, measured the single-client rows within 1 % and the concurrent rows 17–37 % slower.  The record keeps both runs.

## Reference comparison

The llama.cpp comparison is TODO.  It requires a llama.cpp build with EmbeddingGemma 2 support and the matching GGUF.  Measure both over HTTP with the same texts and report the ratio per row.
