#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_

#include <cstddef>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::models::embedding_gemma2::hip {

/// Y[tokens, m] = X_bf16[tokens, k] * W_bf16[m, k]^T with a float32 result.
/// Throws std::runtime_error when hipBLAS rejects the call.
void LaunchGemmBf16(hipblasHandle_t handle, const void* weight_bf16,
                    const void* input_bf16, float* output, std::size_t tokens,
                    std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::models::embedding_gemma2::hip
#endif

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_
