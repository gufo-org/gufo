# MiniMax H3 experiments

| Experiment | Decision / qualification |
| --- | --- |
| Device-resident block chaining | Retained; independent complete-forward oracle. |
| Fused QKV RMSNorm/RoPE | Retained; original FP32 reduction/FMA order, byte-exact full-resolution block, unchanged teacher gates. AdaLN keeps its split normalization. |
| Streamed prompt/AdaLN weights | Retained; phase releases, bounded prefetch and zero-swap checks. |
| F32 convolution GEMMs | Retained; VisualVAE/AudioVAE teacher metrics. |
| Native long BF16 attention | Retained; 128-query/32-key WMMA tiles, register-held probabilities, bounded vector loads and branchless BF16 rounding. Independent FP64 and real-weight teacher checks; removes Triton/AOTriton production dependencies. |
| Native short BF16 attention | Retained; preserves CK's 128-key softmax arithmetic and explicit FMA order with no CK dependency. Byte-exact component controls and unchanged complete-forward teacher errors; 1.6–5.1× faster short-attention controls. |
| Pack long-attention values once; share K/V LDS | Retained; eight waves share K/V, vector loads replace repeated scalar gathers, same 32-key softmax, byte-exact output. Uses dead QKV scratch. |
| Native full-resolution attention-output and FFN-down projections | Retained; 256×128 WMMA tiles with operand prefetch, separate interior/tail tiles and paired-lane vector stores. Same FP32 accumulation/BF16 outputs; packed weights replace originals. |
| Fuse FFN gate/up, SwiGLU and packed output | Retained; both projections still round to BF16 before the activation. Exact full-size controls and complete 50-block outputs; eliminates the gate/up tensor and saves 516 MiB. Paired lanes combine output into vector stores. |
| Fuse QKV projection/norm/RoPE and attention output packing | Retained; avoids full QKV/V intermediates and attention repacking. Same BF16 projection boundary, norm reduction and paired RoPE expressions; block/complete-forward replay is byte-exact. |
| Fuse attention/MLP AdaLN with input packing | Retained; same normalization/modulation arithmetic, byte-exact controls. Diagnostic readback preserves row-major layout. |
| Reuse dead activations and remove denoiser copies | Retained; one in-place trunk buffer, final normalization writes widened BF16 directly to F32 projection scratch. Deletes six staging tensors and redundant row maps. |
| Remove unused BLAS handles | Retained; no attention GEMM handles; sequential short blocks share one projection handle, native blocks need none. Setup-only handle is released, retained workspaces are counted. |
| Unfused native QKV/up projections and 16-wave attention | Rejected; projection tile-order and smaller-tile trials still lost to rocBLAS; 16 waves lost to eight. |
| Per-tile V transpose / normalized-pair shuffles / extra softmax shortcuts | Rejected: transpose and shortcuts did not improve speed; pair shuffles changed BF16 rounding. |
| Wider attention tiles / 6 or 12 waves / query LDS / K/V prefetch | Rejected; register and scheduling controls did not beat eight waves. Twelve waves spilled; wider softmax tiles also change rounding. Direct buffer-to-LDS did not lower on the pinned gfx1151 compiler. |
| Fused down projection plus gated residual / alternate up wave layout | Rejected; exact outputs but slower than separate residual and retained up projection. |
| Alternate fused FFN tile groups and paired weight layout | Rejected; no additional useful gain over the retained tile/vector stores. |
| Extra BF16 probability refinement on short attention | Rejected; improved one-block teacher error but worsened complete-forward error. The retained native path preserves the qualified short-attention arithmetic. |
| VisualVAE wave-local LayerNorm / selected-frame pruning | Retained; same decoder context and selected-frame teacher. |
| Direct mapped reused DiT weights | Rejected: slower resident execution. |
| Alternate softmax reductions | Rejected: quality failures. |
| Fast/aggressive thinning and velocity reuse | Explicit approximate presets; complete delivery-quality comparison remains TODO. |

Run the smallest [independent oracle](QUALITY.md) covering a change.
A recognizable frame/playable MP4 is insufficient. Full schedule and perceptual
studies require separate qualification; no full video is needed for docs cleanup.
