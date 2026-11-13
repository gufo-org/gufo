# NPU BF16 GEMM, host operators and embedding provider options

`include/gufo/ext/experimental/npu.hpp` now also includes:
- `BfpGemm` (`src/core/npu/bfp_gemm.hpp`): BF16 x BFP16 matrix products
  `y = W x` for many rows (prefill, encoders, embeddings) on the NPU. `Open`
  loads the IRON design in `src/core/npu/kernels/bfpgemm` (`bfpgemm.xclbin`,
  `bfpgemm_dispatch.so`); `Upload`/`UploadBf16` pack a matrix into the weight
  arena as bfp16ebs8; `Run` executes independent products. Rows go in bands of
  512 and outputs in groups of 1024; each product is within about 1% relative
  of f32 arithmetic. `Run` is thread-safe and stages two threads' products at
  once, so one thread's CPU work overlaps the other's NPU work. `NewScratch`
  reserves an arena region whose `Scratch::Upload` (strided or transposed
  views) and `Scratch::Reset` serve per-request operands such as attention keys
  and values.
- `gufo::npu::host` (`src/core/npu/host_ops.hpp`): FP32 CPU operators between
  NPU products on a persistent thread pool: `ParallelFor`, bf16 conversions,
  `RmsNormRows`, `LayerNormRows`, `RopeNeox`, `ActivateRows`, `GatedRows`,
  `Attention` (grouped KV heads, per-row key ranges), `ExpInPlace`, `Dot` and
  `Axpy`.
- `FindDesign(name)` (`src/core/npu/npu.hpp`): the directory of an in-tree NPU
  design: `$GUFO_NPU_DESIGNS/<name>`, `share/gufo/npu/<name>` beside the
  executable's `bin`, or the build tree. Packages built with IRON install the
  designs there.

Embedding providers receive their declared options, as speech and decision
providers do: `EmbeddingRunnerOptions` (`model_root`, `provider_options`) and
`EmbeddingProvider::create_with_options`, preferred over `create` when set.
`gufo serve embedding --help` lists the options providers declare for
`serve embedding` under "Provider options".

Public CMake contract: `gufo_npu_design` (`cmake/GufoNpuDesigns.cmake`) builds
designs by default when `GUFO_IRON_PYTHON` names an IRON Python, keeps IRON's
kernel cache in the build tree and installs each design to
`share/gufo/npu/<name>`.

Compatibility: additive. Existing extensions need no change: the new fields
default to empty and `create` keeps working.
