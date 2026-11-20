# YuE2 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Logits over the phase's rows only | Retained; masked rows are -inf upstream, so sampling is unchanged; the semantic head reads 32769 of 184704 rows. |
| One KV cache across plan, semantic and acoustic prefix | Retained; same tokens as upstream's separate prefills; saves two prefills. |
| BF16 KV cache, split-K decode attention | Retained; upstream stores BF16 K/V; the talker's one-block FP32 kernel cannot hold 24576 keys in LDS. |
| FP32 ODE state | Rejected 2026-11-15; upstream keeps BF16 between steps. Against fresh upstream fixtures the FP32 state drifted further from upstream (latent relative error 0.0133) than upstream's BF16 update (0.0112, the single-velocity error 0.0113). |
| Polyphase transposed convolutions, shifted-view k7 SGEMMs | Retained; no im2col buffers (the last stage would need 3.6 GB per tile); audio matches at 6.4e-7 relative error. |
| hipBLASLt plan for prefill-sized GEMMs | Retained; the NAR projections ran at about 4 TFLOP/s through `hipblasGemmEx` and 21-37 TFLOP/s with candidate 4; acoustic 117.5 -> 34.8 s on the same score (BENCHMARKS.md). Oracles unchanged. |
| BF16 outputs for prefill-sized projections | Retained; every consumer rounded the FP32 output to BF16 anyway; acoustic 34.8 -> 29.8 s. Velocity error 0.01105 -> 0.01090. |
| BF16 probabilities and WMMA value product in the NAR attention | Retained; matches PyTorch's fused BF16 attention; acoustic 29.8 -> 26.9 s. The same change in AR prefill flipped 2 of 26 teacher-forced argmaxes (near ties, cosine 0.99997), so AR prefill keeps the FP32 product. |
| Top-k by a k-entry min-heap | Retained; same candidates, byte-identical seeded song; sampling 305 -> 71 us per token. |
| HIP graph for AR decode | Rejected; a captured decode step replays in 16.6 ms, the same as eager launches. |
| Register-cached single-row RMSNorm | Not applied; bit-identical but 5.8 -> 3.3 us per launch, about 0.13 ms of an 18 ms token. |
| 12-bit decode rows (sign, 4-bit exponent code, 7 mantissa bits, per-row escapes) | Retained 2026-11-20; lossless, verified against the tensors at load; the GEMV keeps the BF16 kernel's order, so the seeded song is byte-identical; semantic 29.9 -> 24.8 s for 2.5 GiB of device memory and 0.3 s of load (BENCHMARKS.md). |
| Packing the decode rows on the CPU | Rejected; 16 threads over the mapped file added 1.9 s of load against 0.3 s on the GPU, and needed a 570 MB host buffer for the head. |
| More waves per workgroup in the acoustic attention | Rejected; 16 and 32 waves share each key tile between more queries and are bit-identical, but took 20.1 s and 21.4 s against 20.3 s: the kernel is bound by its matrix products. |
| More speed work | Open: fewer small kernels per decode layer (fuse norm and RoPE into the projections), merged q/k/v and gate/up GEMMs for the NAR pass. |
