#ifndef GUFO_CORE_HIP_ROCBLAS_GEMM_HPP_
#define GUFO_CORE_HIP_ROCBLAS_GEMM_HPP_

#include <cstddef>

#include "src/core/platform/rocblas_kernels.hpp"

namespace gufo::hip {

/// The C operand of a beta-zero rocblas_gemm_ex that writes its result to D.
///
/// rocBLAS runs an in-place (C == D) GEMM with one output row or column
/// through its own GEMV kernels, on gfx11 always since ROCm 10.1. TheRock's
/// Windows rocBLAS ships none of its own kernels (only Tensile's), so such a
/// call fails with hipErrorInvalidKernelFile and leaves the HIP context
/// unusable. With beta zero C is never read and may be null, which keeps the
/// call on Tensile: bit for bit the result the in-place call had with TheRock
/// 10.0. Where rocBLAS has its own kernels (Linux) the call stays in place
/// and keeps its GEMV.
template<typename T>
[[nodiscard]] constexpr const T* BetaZeroGemmC(const T* d,
                                               [[maybe_unused]] std::size_t m,
                                               [[maybe_unused]] std::size_t n) {
  if constexpr (!platform::kRocblasHasOwnKernels) {
    if (m == 1 || n == 1)
      return nullptr;
  }
  return d;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_ROCBLAS_GEMM_HPP_
