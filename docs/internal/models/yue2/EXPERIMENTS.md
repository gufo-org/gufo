# YuE2 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Logits over the phase's rows only | Retained; masked rows are -inf upstream, so sampling is unchanged; the semantic head reads 32769 of 184704 rows. |
| One KV cache across plan, semantic and acoustic prefix | Retained; same tokens as upstream's separate prefills; saves two prefills. |
| BF16 KV cache, split-K decode attention | Retained; upstream stores BF16 K/V; the talker's one-block FP32 kernel cannot hold 24576 keys in LDS. |
| FP32 ODE state | Retained; upstream keeps BF16 between steps. Latents still match upstream at cosine 0.99994. |
| Polyphase transposed convolutions, shifted-view k7 SGEMMs | Retained; no im2col buffers (the last stage would need 3.6 GB per tile); audio matches at 6.4e-7 relative error. |
| hipBLASLt plan for prefill-sized GEMMs | Retained; the NAR projections ran at about 4 TFLOP/s through `hipblasGemmEx` and 21-37 TFLOP/s with candidate 4; acoustic 117.5 -> 34.8 s on the same score (BENCHMARKS.md). Oracles unchanged. |
| BF16 outputs for prefill-sized projections | Retained; every consumer rounded the FP32 output to BF16 anyway; acoustic 34.8 -> 29.8 s. Velocity error 0.01105 -> 0.01090. |
| BF16 probabilities and WMMA value product in the NAR attention | Retained; matches PyTorch's fused BF16 attention; acoustic 29.8 -> 26.9 s. The same change in AR prefill flipped 2 of 26 teacher-forced argmaxes (near ties, cosine 0.99997), so AR prefill keeps the FP32 product. |
| Top-k by a k-entry min-heap | Retained; same candidates, byte-identical seeded song; sampling 305 -> 71 us per token. |
| HIP graph for AR decode | Rejected; a captured decode step replays in 16.6 ms, the same as eager launches. |
| Register-cached single-row RMSNorm | Not applied; bit-identical but 5.8 -> 3.3 us per launch, about 0.13 ms of an 18 ms token. |
| More speed work | Open: fewer small kernels per decode layer (fuse norm and RoPE into the projections), merged q/k/v and gate/up GEMMs for the NAR pass. |
