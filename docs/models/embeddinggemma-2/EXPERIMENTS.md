# EmbeddingGemma 2 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Float32 residual stream, BF16 matrix inputs | Retained.  It is closer to the float32 reference (0.999985 worst probe) than Hugging Face BF16 is (0.999867). |
| Fused QKV and gate/up projections | Retained.  Each pair of projections that share an activation runs as one GEMM. |
| BF16 safetensors only | Retained.  The text backbone is 540 MB and 268 MB of that is the token table, which is a row lookup, so quantizing saves little memory or time.  GGUF quants wait for published quality measurements in [discussion 478](https://github.com/gufo-org/gufo/discussions/478). |
| Text backbone only | Retained.  Gufo does not load the vision and audio encoders and never maps their tensors into the device. |
| Row-chunked GEMM and consumer | Retained.  Chunks of 1,024 rows bound the float32 scratch buffers.  Unchunked, the gate/up buffer for 8,192 tokens would be 8,192 x 4,096 x 4 bytes = 134 MB and the QKV buffer about 100 MB. |
| Precomputed rotary tables | Retained.  Double-precision cos/sin tables per position replace per-element `powf` and `sincosf`. |
| Cross-request batching | Not implemented.  Concurrent requests serialize on the device.  Merging queued requests into one batch would raise the throughput of many small requests.  Quantify the gain against real traffic first. |
| Server does not add task prefixes | Retained.  The OpenAI API has no task parameter, so clients choose the prefix. |
| Attention window radius 512 | Retained from the checkpoint configuration.  The model card's 1,024 is the full window width. |
