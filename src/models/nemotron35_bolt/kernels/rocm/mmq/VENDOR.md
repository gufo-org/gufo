# HIP quantized matrix kernels

These model-private kernels derive from llama.cpp commit
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT, The ggml authors),
originally under `ggml/src/ggml-cuda/`. They were copied from DeepSeek V4
Flash's HIP adaptation in `src/models/deepseek_v4_flash/kernels/rocm/mmq/`;
that copy's lineage, including antirez/ds4 and
xangel82/DS4-GB10-GX10-DSpark-CUDA `910501e` (both MIT), is listed in
`THIRD_PARTY_NOTICES.md`.

Gufo builds this directory with HIP for AMD `gfx1151`. Device headers use
`.hip.hpp`; translation units use `.hip.cpp`. Symbols carry an `nb_` prefix
(`nb_ggml_hip_*`, `NB_GGML_HIP_*`, `nb_quantize_*`, `nb_ggml_stubs.h`), and
every translation unit wraps the tier in `namespace nb_mmq` after
`nb_mmq_prelude.h` has included the system headers. Upstream names such as
`ggml_type` and `mul_mat_q_case` therefore stay apart from the DeepSeek and
Qwen copies in the same binary; only the `nb_mmq_*` functions of `nb_mmq.h`
are exported.

`nb_mmq.h` is the raw-pointer C ABI and `nb_mmq.hip.cpp` its adapter.
Dispatch follows upstream `ggml_cuda_mul_mat`: dense products with N up to
`MMVQ_MAX_BATCH_SIZE` (8) use the MMVQ vector kernels, larger N the MMQ
tiles. Routed experts no longer use this tier: short chunks run the
model's IQ2_XXS vector kernel (`gemv.hip`), prefill its BF16 WMMA product
(`gemm.hip`), which replaced the routed MMQ path and its `mm_ids_helper`
maps. Activations are quantized to Q8_1 as upstream; the caller applies NVFP4's
per-tensor scale. Only Q8_0, IQ2_XXS and NVFP4 are instantiated: one MMQ
translation unit per format (`mmq_instance_*.hip.cpp`) and three cases in
the MMVQ type switch.

Retained from the DeepSeek copy: HIP-only dispatch without NVIDIA/MUSA
preprocessing, the `gfx1151` MMQ geometry (64-row WMMA tiles, columns capped
at 80), the `gfx1151` IQ2_XXS loader sign expansion and 8-byte shared-memory
stores, clamped `ids_dst` reads in partial column tiles, the grid-stride column
loop, the negative-id guard in MMVQ, the Q8_1 rounding that does
not depend on FMA contraction, and the ggml stub layer. Scratch comes from one
256 MiB arena per stream instead of `hipMallocAsync`, so a captured decode
graph replays correctly under ROCm 7.1 and calls avoid per-call allocation.

Removed DeepSeek-only extras: aligned SoA loaders and derepacking, D2R,
the raw IQ2 stride-16 specialization, the fused SwiGLU/router epilogue and
output sanitizing in MMQ write-back, routed tile-width and grid-bound
overrides, the column-warp (`NCW`) experiment, paired, fused, down-sum and
gate/up-mid entry points, folded Q8_1 activations, persistent scratch and the
DSpark paths. `ggml_type_size(GGML_TYPE_NVFP4)` is corrected to 36 bytes.

Gufo additions: `nb_mmq_dense_q8_1` runs the vector kernels on activations
the caller already quantized, and the single-column fusion gains
`dst_scale`, applied before `x_bias`, so a residual update with the NVFP4
per-tensor scale fuses into the product.

When importing upstream changes, keep the HIP dispatch, the `nb_` prefix and
the namespace wrap, update the pin, and rerun `nemotron35_bolt.mmq_ops`
(`tests/models/nemotron35_bolt/mmq_ops_test.cpp`). It compares every format
and path with the core ggml dequantizers, bounding each output by its Q8_1
activation-rounding noise.
