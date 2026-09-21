# MiniMax H3 experiments

| Experiment | Decision / qualification |
| --- | --- |
| Device-resident block chaining | Retained; independent complete-forward oracle. |
| Split QKV/AdaLN normalization | Retained with component gates; bounded live state. |
| Streamed prompt/AdaLN weights | Retained; phase releases, bounded prefetch and zero-swap checks. |
| F32 convolution GEMMs | Retained; VisualVAE/AudioVAE teacher metrics. |
| Fixed-shape row-parallel attention | Retained; independent block/forward gates. Full-resolution speed remains unmeasured. |
| Native long BF16 attention | Retained; 64-query/32-key WMMA tiles, register-held probabilities, bounded vector loads and branchless BF16 rounding. Independent FP64 and real-weight teacher checks; removes Triton/AOTriton production dependencies. |
| Wider attention tiles / more waves | Rejected; slower at the affected long shapes. |
| Extra BF16 probability refinement on short attention | Rejected; improved one-block teacher error but worsened complete-forward video-velocity relative L2 from `0.01936` to `0.03418`. Short CK arithmetic stays unchanged. |
| VisualVAE wave-local LayerNorm / selected-frame pruning | Retained; same decoder context and selected-frame teacher. |
| Direct mapped reused DiT weights | Rejected: slower resident execution. |
| Alternate softmax reductions | Rejected: quality failures. |
| Fast/aggressive thinning and velocity reuse | Explicit approximate presets; complete delivery-quality comparison remains TODO. |

Run the smallest [independent oracle](EVALUATION.md) covering a change.
A recognizable frame/playable MP4 is insufficient. Full schedule and perceptual
studies require separate qualification; no full video is needed for docs cleanup.
