# NPU BF16 GEMM, host operators and embedding provider options

`include/gufo/ext/experimental/npu.hpp` now also includes:
- `BfpGemm` (`src/core/npu/bfp_gemm.hpp`): BFP16 matrix products `y = W x`
  for many rows (prefill, encoders, embeddings) on the NPU. `Open` loads the
  IRON designs in `src/core/npu/kernels/bfpgemm` (`bfpgemm` with 512-row bands
  and, when present, `bfpgemm_s` with 128-row bands for short products, both
  sharing one weight arena); `Upload`/`UploadBf16` pack a matrix into the
  arena as bfp16ebs8; `Run` executes independent products, batching calls of
  one shape whose matrices lie at a constant stride into one NPU job. A `Call`
  may pass bf16 rows (`x16`) and an `Epilogue` that consumes the bf16 output
  tiles instead of `y`. Each product is within about 1% relative of f32
  arithmetic. `Run` is thread-safe and stages up to four threads' products at
  once, so one thread's CPU work overlaps the others' NPU work. `NewScratch`
  reserves an arena region whose `Scratch::Upload`, `UploadMany` (one pass for
  several matrices, `Spec::same_as` copies) and `Duplicate` serve per-request
  operands such as attention keys and values, until `Scratch::Reset`.
  `TakeBusyMs` reports the NPU's busy time with `GUFO_NPU_PROFILE` set.
- `gufo::npu::host` (`src/core/npu/host_ops.hpp`): FP32 CPU operators between
  NPU products on a persistent thread pool: `ParallelFor`, `SharePool` (a
  thread's share of the pool, so several threads run parallel host work at
  once), bf16 conversions, `EncodeBfp16Block(s)`, `RmsNormRows`,
  `RmsNormRowsBf16`, `AddNormRows`, `LayerNormRows`, `RopeNeox`,
  `ActivateRows`, `GatedRows`, `Attention` (grouped KV heads, per-row key
  ranges), `ExpInPlace`, `Dot` and `Axpy`.
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
kernel cache in the build tree and installs each of a design's `PROGRAMS`
(default: the design's name) to `share/gufo/npu/<name>`.

Compatibility: additive. Existing extensions need no change: the new fields
default to empty and `create` keeps working.
