# Qwen-Image-2.1 experiments

| Approach | Status |
| --- | --- |
| Cache timestep-zero text/reference K/V after the first step | Retained; first/cached-step blocks and full edit trajectory pass. |
| Fused QK, online softmax and FP32 value product | Retained for 128-wide attention; 128 queries per block, bounded LDS, no quadratic allocation or probability narrowing. |
| Separate causal prefix from fully visible image queries | Retained; avoids mask work on image rows. |
| Two-way QK unrolling and selective ILP scheduling | Retained; bounded register pressure and exact output. |
| Packed BF16 projections with direct WMMA loads | Retained; FP32 accumulation, exact byte permutation, no duplicate persistent weights. |
| Fuse gate/up projections, SiLU and the following projection's input packing | Retained; preserves each BF16 boundary; operator and independent model checks pass. |
| Branchless BF16 conversion and native exponential for BF16 SiLU | Retained; conversion boundary tests and all 65,536 BF16 activation inputs match. |
| Precompute timestep modulation; fuse normalization/modulation and RMSNorm/RoPE | Retained; original reduction order and BF16 boundaries. |
| One wave per short normalization row; fuse VAE normalization/SiLU | Retained; exact model tensors. |
| Native 3×3 convolution with packed weights and implicit im2col | Retained; original channel/tap order, BF16 product before bias. |
| Reuse overlapping convolution input windows in aligned strips | Retained; exact outputs, padding and upscaling qualified. |
| Vectorized head layouts and convolution input copies | Retained; exact aligned/ragged layouts. |
| Bound unused GPU scratch to 8 GiB; reclaim it on allocation failure | Retained; live state remains intact across request-size changes. |
| Encode PNG once with low compression and adaptive filters | Retained; exact RGBA pixels; trades larger responses for lower CPU latency. |
| Wider attention key tiles, alternative PV mappings, matrix PV with exact probability decomposition | Rejected; slower than FP32 value accumulation. |
| Attention bank swizzles, register softmax state and narrower barriers | Rejected; no useful standalone improvement. |
| Projection workgroup swizzles, larger tiles, LDS staging, raw-buffer loads, stride padding and alternate weight layouts | Rejected; no worthwhile gain over direct packed operands. |
| Larger convolution chunks or hipBLASLt workspace | Rejected; no useful end-to-end gain. |
| SiLU lookup table, refined reciprocal and alternate exact BF16 conversion | Rejected; no speed gain in the complete feed-forward kernel. |
| Fuse projection output with residual modulation | Rejected; slower despite exact output. |

The official prompt encoder uses pre-final-RMSNorm text features, omits its
vocabulary head and uses the VAE's first-frame path. These are model contracts,
not quality/speed tradeoffs. No steps, conditioning images or trained weights
were removed. Concurrent GPU batching remains future work.

Use `tools/prof/prof.py` with `--stages qwen-image`. The maintained
`tools/models/qwen_image_21/qwen_image_gemm_bench.hip` rotates weights beyond
cache capacity and checks sampled FP64 dot products. Keep transient profiles,
failed microbenchmarks and tensor dumps outside Git; qualification is in
[EVALUATION.md](EVALUATION.md).
