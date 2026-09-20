# Qwen-Image-2.1 experiments

| Approach | Status |
| --- | --- |
| Cache timestep-zero text/reference K/V after first denoising step | Retained; matched first/cached-step blocks and full edit trajectory pass. |
| Bound attention query tiles and convolution im2col scratch | Retained; no persistent quadratic attention allocation. |
| Shape-specific hipBLASLt projections and batched QK score products | Retained; FP64 dot-product controls and independent model checks pass. |
| Register-cached softmax for 4096–8448 keys; finish the existing reduction tree in one wave | Retained; bit-identical model boundaries. |
| Cached softmax below 4096 keys | Rejected; slower in the focused control. |
| Fully fuse QK, online softmax and the FP32 value product | Retained for masked/unmasked 128-wide attention with at least 128 queries and 512 keys; no global score/probability buffers or Triton dependency. |
| Cover all 128 output channels in each fused-attention block | Retained; avoids duplicate QK/softmax work, exact outputs versus the narrower tile. |
| Launch all attention query tiles together | Retained; exact output, fewer launches and better reuse across queries. |
| Exact packed BF16 projections | Retained for supported 4096/12288-wide shapes; direct WMMA fragment loads, FP32 accumulation, and no duplicate persistent weights. FP64 and independent model checks pass. |
| Traverse packing tiles along adjacent source columns | Retained; coalesced 64-row tiles, exact bytes, and lower complete-request latency. |
| One wave per short normalization row | Retained through 1152 channels; preserves the original reduction tree and exact model boundaries. |
| Split causal prefix from fully visible attention queries | Retained; exact outputs with a short masked prefix and one unmasked image launch. |
| LDS-prefetched projection tiles | Superseded by direct packed operands. |
| Larger attention key tiles, alternative PV thread mappings and exact three-part matrix PV | Rejected; no useful speed gain over FP32 value accumulation. |
| Projection workgroup swizzles, alternative tile shapes and store transposes | Rejected; no worthwhile gain over the retained packed kernel. |
| Larger VAE convolution chunks | Rejected; mixed gains and changed rounding. |
| Larger hipBLASLt workspace | Rejected; no useful gain in the measured projection shapes. |
| Coalesced convolution input reads with LDS transpose | Retained; exact layout, including padding, down/upscaling and partial tiles. |
| Bound unused GPU scratch to 8 GiB and reclaim it on allocation failure | Retained; avoids accumulating stale sizes across requests without changing live model state. |
| Encode PNG once using libpng’s bounded output size | Retained; avoids a redundant compression pass without changing pixels or encoding. |
| Fuse only softmax normalization and the value product | Superseded by complete attention fusion. |
| Vectorized tiled head-layout conversion | Retained; exact for aligned, ragged and grouped-query inputs. |
| Value-product prefetch and wider thread mappings | Rejected; more registers and slower execution. |
| Native value product for 64-query tiles or 72-wide vision heads | Rejected; established BLAS path is faster. |
| Two-way QK-loop unrolling | Retained; lower register pressure, exact output, and faster 128-wide attention. |
| Scheduling barriers in the fused QK loop | Rejected; no consistent worthwhile gain. |
| Preserve convolution-product rounding before bias | Retained; first VAE convolution now exactly matches PyTorch. |
| Skip final text RMSNorm and the unused vocabulary head | Required prompt-encoder contract; unused weights are not uploaded. |
| Omit temporal VAE convolutions on the single image frame | Required official first-frame path. |

Attention probabilities and accumulation remain FP32. No steps, conditioning
images or trained weights were removed. Concurrent GPU batching remains future work.

Use `tools/prof/prof.py --help` with the `qwen-image` stage map. The maintained
`tools/models/qwen_image_21/qwen_image_gemm_bench.hip` rotates weights beyond
cache capacity and checks every candidate against sampled FP64 dot products.
