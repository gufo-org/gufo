# YuE2 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Logits over the phase's rows only | Retained; masked rows are -inf upstream, so sampling is unchanged; the semantic head reads 32769 of 184704 rows. |
| One KV cache across plan, semantic and acoustic prefix | Retained; same tokens as upstream's separate prefills; saves two prefills. |
| BF16 KV cache, split-K decode attention | Retained; upstream stores BF16 K/V; the talker's one-block FP32 kernel cannot hold 24576 keys in LDS. |
| FP32 ODE state | Retained; upstream keeps BF16 between steps. Latents still match upstream at cosine 0.99994. |
| Polyphase transposed convolutions, shifted-view k7 SGEMMs | Retained; no im2col buffers (the last stage would need 3.6 GB per tile); audio matches at 6.4e-7 relative error. |
| Speed work on the acoustic pass | Open: WMMA value product in attention, hipBLASLt for the NAR GEMMs, graph capture of decode. |
