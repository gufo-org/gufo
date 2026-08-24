#if defined(ENGINE_ENABLE_HIP)
#include <cstddef>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"

namespace strix::hip {

void LaunchHipblasGEMM(hipblasHandle_t handle, const void* A, bool is_bf16,
                       const float* X, float* Y, std::size_t batch_size,
                       std::size_t M, std::size_t K, void* d_x_bf16_buf,
                       hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  if (is_bf16) {
    LaunchFloatToBfloat16(X, d_x_bf16_buf, batch_size * K, stream);
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A,
        HIP_R_16BF, static_cast<int>(K), d_x_bf16_buf, HIP_R_16BF,
        static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  } else {
    HIPBLAS_CHECK(hipblasGemmEx(
        handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
        static_cast<int>(batch_size), static_cast<int>(K), &alpha, A, HIP_R_32F,
        static_cast<int>(K), X, HIP_R_32F, static_cast<int>(K), &beta, Y,
        HIP_R_32F, static_cast<int>(M), HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT));
  }
}

void LaunchHipblasGEMMBF16(hipblasHandle_t handle, const void* A_bf16,
                           const void* d_x_bf16, float* Y,
                           std::size_t batch_size, std::size_t M, std::size_t K,
                           hipStream_t stream) {
  if (batch_size == 0 || M == 0 || K == 0) {
    return;
  }

  if (stream != nullptr) {
    HIPBLAS_CHECK(hipblasSetStream(handle, stream));
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;

  HIPBLAS_CHECK(hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(M),
      static_cast<int>(batch_size), static_cast<int>(K), &alpha, A_bf16,
      HIP_R_16BF, static_cast<int>(K), d_x_bf16, HIP_R_16BF,
      static_cast<int>(K), &beta, Y, HIP_R_32F, static_cast<int>(M),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}


}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
