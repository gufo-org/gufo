#ifndef GUFO_CORE_PLATFORM_ROCBLAS_KERNELS_HPP_
#define GUFO_CORE_PLATFORM_ROCBLAS_KERNELS_HPP_

namespace gufo::platform {

/// Whether the ROCm SDK's rocBLAS library carries rocBLAS's own GPU kernels,
/// such as its GEMV, besides the Tensile libraries. TheRock's Windows
/// `rocblas.dll` ships only Tensile's, so a call that rocBLAS routes to one
/// of its own kernels fails there with hipErrorInvalidKernelFile.
inline constexpr bool kRocblasHasOwnKernels =
#if defined(_WIN32)
    false;
#else
    true;
#endif

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_ROCBLAS_KERNELS_HPP_
