# MiniMax H3 experiments

| Experiment | Decision / qualification |
| --- | --- |
| Device-resident block chaining | Retained; independent complete-forward oracle. |
| Fused QKV RMSNorm/RoPE | Retained; original FP32 reduction/FMA order, byte-exact full-resolution block, unchanged teacher gates. AdaLN keeps its split normalization. |
| Streamed prompt/AdaLN weights | Retained; phase releases, bounded prefetch and zero-swap checks. |
| F32 convolution GEMMs | Retained; VisualVAE/AudioVAE teacher metrics. |
| Fixed-shape row-parallel attention | Retained for diagnostic comparison; independent block/forward gates. Production uses native fused attention. |
| Native long BF16 attention | Retained; 64-query/32-key WMMA tiles, register-held probabilities, bounded vector loads and branchless BF16 rounding. Independent FP64 and real-weight teacher checks; removes Triton/AOTriton production dependencies. |
| Native short BF16 attention | Retained; preserves CK's 128-key softmax arithmetic and explicit FMA order with no CK dependency. Byte-exact component controls and unchanged complete-forward teacher errors; 1.6–5.1× faster short-attention controls. |
| Pack long-attention values once; share K/V LDS | Retained; eight waves share K/V, vector loads replace repeated scalar gathers, same 32-key softmax, byte-exact output. Uses dead QKV scratch. |
| Native full-resolution FFN down projection | Retained; 256×128 WMMA tiles with operand prefetch, exact weight/activation packing and FP32 accumulation. No persistent duplicate weights. |
| Fuse SwiGLU and activation packing | Retained; exhaustive BF16 gate comparison and byte-identical full-resolution block. Removes an intermediate read/write. |
| Native QKV/up projections and 16-wave attention | Rejected; projection tile-order and smaller-tile trials still lost to rocBLAS; 16 waves lost to eight. |
| Per-tile V transpose / normalized-pair shuffles / extra softmax shortcuts | Rejected: transpose and shortcuts did not improve speed; pair shuffles changed BF16 rounding. |
| Wider attention tiles / more waves | Rejected; slower at the affected long shapes. |
| Extra BF16 probability refinement on short attention | Rejected; improved one-block teacher error but worsened complete-forward error. The retained native path preserves the qualified short-attention arithmetic. |
| VisualVAE wave-local LayerNorm / selected-frame pruning | Retained; same decoder context and selected-frame teacher. |
| Direct mapped reused DiT weights | Rejected: slower resident execution. |
| Alternate softmax reductions | Rejected: quality failures. |
| Fast/aggressive thinning and velocity reuse | Explicit approximate presets; complete delivery-quality comparison remains TODO. |

Run the smallest [independent oracle](EVALUATION.md) covering a change.
A recognizable frame/playable MP4 is insufficient. Full schedule and perceptual
studies require separate qualification; no full video is needed for docs cleanup.
