#include "src/models/embedding_gemma2/hip/blas.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace gufo::models::embedding_gemma2::hip {
namespace {

constexpr float kAlpha = 1.0F;
constexpr float kBeta = 0.0F;

}  // namespace

void LaunchGemmBf16(hipblasHandle_t handle, const void* weight_bf16,
                    const void* input_bf16, float* output, std::size_t tokens,
                    std::size_t m, std::size_t k, hipStream_t stream) {
  if (handle == nullptr || weight_bf16 == nullptr || input_bf16 == nullptr ||
      output == nullptr || tokens == 0U || m == 0U || k == 0U) {
    throw std::invalid_argument("invalid EmbeddingGemma 2 GEMM inputs");
  }
  constexpr auto kLimit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (tokens > kLimit || m > kLimit || k > kLimit) {
    throw std::length_error("EmbeddingGemma 2 GEMM exceeds library limits");
  }
  if (hipblasSetStream(handle, stream) != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("EmbeddingGemma 2 hipBLAS stream setup failed");
  }
  // Row-major Y[tokens, m] is column-major Y^T[m, tokens] = W^T X^T.
  const hipblasStatus_t status = hipblasGemmEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m),
      static_cast<int>(tokens), static_cast<int>(k), &kAlpha, weight_bf16,
      HIP_R_16BF, static_cast<int>(k), input_bf16, HIP_R_16BF,
      static_cast<int>(k), &kBeta, output, HIP_R_32F, static_cast<int>(m),
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("EmbeddingGemma 2 hipBLAS GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace gufo::models::embedding_gemma2::hip

#endif  // defined(ENGINE_ENABLE_HIP)
