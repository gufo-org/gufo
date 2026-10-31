# Fish Audio Drama-3 experiments

| Experiment | Decision / evidence |
| --- | --- |
| Native weight-only `F8_E4M3` + `128x128` `F32` block scale MoE | Retained: avoids expanding 30 GiB of routed experts to 60 GiB BF16 on disk or in memory; cosine 1.0 (max diff 1.83e-4) vs FP32 oracle. |
| Register-resident Gated DeltaNet `[128, 128]` recurrent state | Retained: each wave32 lane holds 4 state columns in registers across steps (`14.15 us/layer`, 0 bytes scratch spill, max diff 1.79e-7). |
| 1-row-per-workgroup (8-wave coalesced) BF16 GEMV | Retained: replaced 1-row-per-wave GEMV that hit 8 distinct 4 KB pages per workgroup; Tower decode dropped from 1,688.5 ms to 1,010.2 ms (-40.2%, 200.9 GB/s). |
| 8-wave parallel-expert `MoEDownCombineKernel` | Retained: maps `wave_id = 0..7` to routed expert `k = 0..7` in parallel; eliminated serial expert loop in down projection. |
| Full-GPU parallel `MoERouterLogitsKernel` (257 workgroups) + warp tournament top-8 | Retained: replaced single-workgroup router scan; single-layer MoE dropped from 271.2 us to 132.2 us and 96-frame backbone decode dropped from 2,453.6 ms to 2,000.4 ms (-18.5%). |
| Fused QKV/Gated-DeltaNet projections, `ResidualAddRmsNorm`, and last-token LM Head | Retained: reduced per-token kernel launches and host-device synchronization. |
| Direct C++ `codec.pth` loader with online weight-norm folding | Retained: parses uncompressed ZIP + Pickle v2 directly in C++ so no Python preprocessing or `codec.safetensors` conversion is needed. |
| Coalesced 3D `CausalConv1dIm2ColKernel` + `kernel=1` GEMM bypass + prewarmed buffers | Retained: eliminated dynamic `hipFree`/`hipMalloc` device locks and uncoalesced strided gather; 16-frame encode dropped from 118.1 ms to 38.2 ms (-67.6%) and decode dropped from 103.7 ms to 58.3 ms (-43.8%). |
| Async dual-stream 4-frame Codec pipeline | Retained: overlaps `DecodeStreamChunk` on Codec's HIP stream with autoregressive decode; 610 ms TTFA, 23.9 ms total sync wait across 24 chunks, 0.0 diff vs offline. |
| LDS staging of `float[2048]` input vector in MoE kernels | Rejected: `gfx1151` L1/L2 vector cache already broadcasts `ffn_in` at full speed; adding `__shared__ s_x[2048]` reduced CU wave occupancy and regressed decode by ~60 ms. |
