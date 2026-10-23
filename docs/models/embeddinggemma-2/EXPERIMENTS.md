# EmbeddingGemma 2 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Float32 residual stream, BF16 matrix inputs | Retained.  Its worst probe has cosine 0.999981 to the float32 reference, while Hugging Face BF16 has 0.999867. |
| Fused QKV and gate/up projections | Retained.  Each pair of projections that share an activation runs as one GEMM. |
| BF16 safetensors only | Retained.  The text backbone is 540 MB and 268 MB of that is the token table, which is a row lookup, so quantizing saves little memory or time.  GGUF quants wait for published quality measurements in [discussion 478](https://github.com/gufo-org/gufo/discussions/478). |
| Text backbone only | Retained.  Gufo does not load the vision and audio encoders and never maps their tensors into the device. |
| hipBLASLt instead of hipBLAS | Retained.  hipBLAS has no matrix-core kernel for this BF16-in, float32-out transposed GEMM and runs at 4.5 to 5 TFLOPS in `gemm_bench_hip_test`, while hipBLASLt reaches 17 to 27 TFLOPS at 1,024 rows.  No fixed heuristic rank wins across shapes, because the best rank moves between 4 and 8, so Gufo times each shape and row bucket once at load.  Against #700 over HTTP, a 404-token request drops from 41.4 ms to 20.5 ms, which is 2.0 times faster.  Concurrent single-input requests of 404 tokens are 2.0 to 2.1 times faster, and batches of 8 and 32 inputs are 2.2 to 2.3 times faster.  A request of about 8,000 tokens is 1.3 times faster, and load time grows from 0.53 s to 2.2 s. |
| Row-chunked GEMM and consumer | Retained.  Chunks of 1,024 rows bound the float32 scratch buffers.  Unchunked, the gate/up buffer for 8,192 tokens would be 8,192 x 4,096 x 4 bytes = 134 MB and the QKV buffer about 100 MB. |
| Precomputed rotary tables | Retained.  Double-precision cos/sin tables per position replace per-element `powf` and `sincosf`. |
| Device turns per batch | Retained.  One lock held for a whole request made a one-word request wait 58.4 s behind 32 inputs of 8,000 tokens.  With arrival-order turns per device batch it waits 1.01 s, and the long request still takes 59.5 s.  A plain mutex is not enough, because the releasing thread can take it straight back. |
| Unmapping the checkpoint after upload | Retained.  Keeping the mapping held 656 MB of file-backed RSS that nothing reads after the upload.  RSS after load falls from 930 MB to 286 MB. |
| Cross-request batching | Not implemented.  Concurrent requests take device turns one batch at a time.  Merging queued requests into one batch would raise the throughput of many small requests.  Quantify the gain against real traffic first. |
| Server does not add task prefixes | Retained.  The OpenAI API has no task parameter, so clients choose the prefix. |
| Attention window radius 512 | Retained from the checkpoint configuration.  The model card's 1,024 is the full window width. |
