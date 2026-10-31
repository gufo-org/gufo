# MOSS-Image experiments

| Approach | Status |
| --- | --- |
| Run the transformer, text projection and VAE decoder on the Qwen-Image BF16 operators (projections, RMSNorm, RoPE mode 2, 3×3 convolution) | Retained; same BF16 boundaries as the PyTorch reference. Only the shared modulation, situ gate and GroupNorm needed new kernels. |
| Text encoder from a GGUF on the Qwen engine with prompt hidden-state capture | Retained; reuses the hybrid Gated DeltaNet/attention runtime instead of a second Qwen3.5 implementation. Encoder output cosine 0.99987 against Transformers. |
| Dense attention on the MiniMax H3 128-dimension WMMA kernel | Retained; MOSS attention is dense and non-causal over image+text rows, the case that kernel serves. FP32 softmax and accumulation. |
| Qwen-Image fused attention for the transformer blocks | Replaced by the WMMA kernel above; still used for the two short text-projection blocks and the VAE mid-block. |
| GroupNorm statistics in FP32 per thread, combined in FP64 | Retained; the reference VAE runs in FP32. |
| Replay reference noise through an environment variable | Rejected for upstream; the probe passes `--noise` to a development-only model option instead. |
