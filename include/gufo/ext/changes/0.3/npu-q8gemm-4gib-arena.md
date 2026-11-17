# NPU Q8_0 GEMM arena of up to 4 GiB

`gufo::npu::Q8Gemm` (`src/core/npu/q8_gemm.hpp`, included by
`include/gufo/ext/experimental/npu.hpp`) now accepts a weight arena of up to
`Q8Gemm::kMaxArenaBytes` (4 GiB, the q8gemm design's weight tensor; before,
2 GiB). `Open` throws `Error` for a larger `arena_bytes`, instead of failing
at the first product past the tensor.

The q8gemm design and its host also run each matrix as one job of a dispatch:
the cores keep the rows across the matrix's output parts, and a K beyond
2048 codes splits over the shim streams instead of over jobs. Results are
unchanged; one-row products take fewer and cheaper jobs.

Compatibility: additive. Extensions that keep their arena within 2 GiB need
no change; those that need more require this revision.
