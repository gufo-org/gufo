# Q8_0 GEMM on the XDNA NPU

The IRON design behind `gufo::npu::Q8Gemm` (`src/core/npu/q8_gemm.hpp`):
`y = W x` with Q8_0 weights, on all 32 cores of the Strix Halo NPU.

```sh
python3 q8gemm.py <dir>  # writes <dir>/q8gemm.xclbin and <dir>/q8gemm_dispatch.so
```

Compiling needs an mlir-aie (IRON) Python with Peano (`nix develop` has them);
with one, CMake builds it into `build/<preset>/npu/q8gemm`
(`cmake/GufoNpuDesigns.cmake`). `Q8Gemm::Open(device, <dir>)` loads the result.

- `q8gemm.py`: the 16 shim streams feed two cores each, and each column joins its four cores' outputs. A job runs one matrix: the cores keep the rows and loop over the matrix's output parts (`parts`); a K beyond 2048 codes splits over sets of streams, each with its own copy of the rows (`a_ss`).
- `q8gemm.cc`: the cores' arithmetic. x is int16 with one scale per row. For each 32-code block, the int32 dot is multiplied by the block scale's int16 fraction of the output's largest scale and accumulated in acc64. For one row, a stored record holds row 0 of 8 tiles (`compact`), an eighth of the output traffic.
- `q8_gemm.cpp`: packs the weights ([stream][part][chunk][tile][parity] records) and rows, adds the K slices' partial sums, and concatenates the instruction streams of a call's jobs into one dispatch.

Measured on one-row products: about 80 µs per job plus about 49 GB/s of
weights from DDR; a 15360 x 2048 product takes 670 µs, a 2048 x 7680 one
417 µs.
