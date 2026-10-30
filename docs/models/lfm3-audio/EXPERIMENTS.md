# LFM3-Audio experiments

October 29, 2026. Timings are per TTS frame (80 ms of speech), Q8_0 backbone
with the BF16 mmproj, unless noted.

| Change | Result | Decision |
| --- | --- | --- |
| First port: host sampling, three synchronized depth passes, synchronous codec | 6.2× real time; depth 3.8 ms, backbone 8.4 ms, codec 0.55 ms | Baseline |
| Whole frame on the device: reveal passes, top-k/temperature/repetition sampling, window counts and next frame embedding in one captured HIP graph, one sync per frame | Depth 3.8 → 3.5 ms; greedy frames still identical to the reference | Kept: launch gaps were not the main cost |
| Codec decode in a worker on its own stream, overlapping the next backbone step | Codec leaves the critical path | Kept |
| Profile of the depth decoder | 12-row BF16 projections stream at about 80 GB/s (88 and 32 workgroups for 2816- and 1024-row weights); they are about 2.2 ms of the 3.5 ms | Open |
| Custom 12-row GEMM, one wave per row with LDS-staged activations | 2–3× slower; then accumulators spilled to scratch on data-dependent loop exits | Rejected |
| Same with four rows per wave, weight prefetch, exact-size templates | No spills, still slower than the stock kernel | Rejected |
| Barrier-free split-K variant (eight waves per four rows) | Paired gate/up 155 µs vs 2 × 69 µs stock; activation re-reads saturate L1/L2 at 12 rows | Rejected for 12 rows |
| Same kernel for the one-row 2048 → 12288 depth input | 263 µs vs 320 µs stock (190 GB/s) | Kept as `WideGemv` |
| Q8_0 mmproj | Depth 3.5 → 2.5 ms (7.1× real time), but codec audio moves 7% | Accepted input, not the default |

Concurrent synthesis, eight streams of the same sentence (aggregate real
time; per step depth / backbone):

| Change | Result | Decision |
| --- | --- | --- |
| Batched backbone step (`ForwardBatchInputs`, frame embeddings from the device, no logits) and one depth pass over all voices | 17.4×; depth 20.1 ms, backbone 14.1 ms: the BF16 small-batch GEMM reads weights once per 16 rows, so 96 rows cost six reads; greedy codes identical to a lone stream | Kept |
| 64×64-tile FP32 GEMM over BF16 weights for 32+ rows | Depth 19.1 ms at eight, but slower than the 16-row kernel at four (14.5 vs 10.5 ms); ASR encoder 17.7 → 17.1 ms | Kept for the encoder and codec rows; depth uses hipBLAS |
| One launch per frame kernel over a device table of voices | 18.8×; depth 16.8 ms | Kept |
| hipBLAS SGEMM on FP32 copies of the depth projections (about 236 MB) | 20.8×; depth 13.6 ms (rocBLAS 32×32 tiles, about 2.7 TFLOPS) | Kept for batches of 32+ rows |
| Sixteen streams | 23.5×; backbone 24.3 ms, depth 23.5 ms (per-row launches, linear depth FLOPs) | Rejected: batches stop at eight |

Remaining levers: one attention/RoPE launch for all rows of a backbone batch
(about 5 ms of the eight-stream step) and a faster FP32-activation GEMM for the
depth decoder (BF16 WMMA would round activations).

The backbone step is near its bandwidth floor (see [benchmarks](BENCHMARKS.md)).
Its remaining overhead is about 1 ms of launch gaps; graphing it needs
position-independent attention/RoPE/cache launches in the shared LFM3 engine.
