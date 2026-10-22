# EmbeddingGemma 2 benchmarks

Measured on 2026-10-22 on Linux with a Radeon 8060S GPU of architecture gfx1151, using BF16 safetensors, a resident model, and the `nix build` production `gufo`.  Each row is 60 requests, and token counts include `<bos>` and `<eos>`.  The [record](artifacts/gfx1151-2026-10-22-nix.json) holds the evidence.

Model load takes **0.47 s**, which is mostly the 540 MB upload.

## HTTP

These rows time `gufo serve embedding` on loopback with [`http_bench.py`](../../../tools/models/embedding_gemma2/http_bench.py).  Each input is a run of distinct `word<N>` words.

| Concurrency × inputs × words | Tokens/request | Median | p95 | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 1 × 1 × 16 | 40 | 10.30 ms | 11.11 ms | 3,816 |
| 1 × 1 × 128 | 404 | 39.50 ms | 40.67 ms | 10,166 |
| 1 × 1 × 512 | 1,940 | 250.51 ms | 252.06 ms | 7,800 |
| 1 × 8 × 128 | 3,495 | 332.61 ms | 336.58 ms | 10,491 |
| 1 × 32 × 128 | 15,670 | 1,552.19 ms | 1,557.52 ms | 10,091 |
| 4 × 1 × 128 | 404 | 160.90 ms | 167.09 ms | 9,965 |
| 16 × 1 × 128 | 404 | 641.96 ms | 643.70 ms | 10,024 |
| 4 × 8 × 128 | 3,495 | 1,320.13 ms | 1,323.75 ms | 10,583 |
| 16 × 8 × 128 | 3,495 | 5,286.15 ms | 5,296.29 ms | 10,572 |

Concurrent requests share one device and run one batch at a time, so concurrency mostly adds queueing time.

## Reference comparison

The llama.cpp comparison is TODO.  It requires a llama.cpp build with EmbeddingGemma 2 support and the matching GGUF.  Measure both over HTTP with the same texts and report the ratio per row.
