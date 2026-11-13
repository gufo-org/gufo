# Q8_0 GEMM on the XDNA NPU

The IRON design behind `gufo::npu::Q8Gemm` (`src/core/npu/q8_gemm.hpp`):
`y = W x` with Q8_0 weights, on all 32 cores of the Strix Halo NPU.

```sh
python3 q8gemm.py <dir>  # writes <dir>/q8gemm.xclbin and <dir>/q8gemm_dispatch.so
```

Compiling needs an mlir-aie (IRON) Python with Peano (`nix develop` has them);
with one, CMake builds it into `build/<preset>/npu/q8gemm`
(`cmake/GufoNpuDesigns.cmake`). `Q8Gemm::Open(device, <dir>)` loads the result.

- `q8gemm.py`: the 16 shim streams broadcast each to two cores, and each column joins its four cores' outputs.
- `q8gemm.cc`: the cores' arithmetic. x is int16 with one scale per row. For each 32-code block, the int32 dot is multiplied by the block scale's int16 fraction of the output's largest scale and accumulated in acc64.
- `q8_gemm.cpp`: packs the weights and rows, and concatenates the instruction streams of a call's jobs into one dispatch.

Measured: about 57 GB/s of weights from DDR on large matrices, and about
84 µs per dispatch.
