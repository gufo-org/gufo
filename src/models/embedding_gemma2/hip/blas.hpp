#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_

#include <cstddef>
#include <memory>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::models::embedding_gemma2::hip {

/// Y[tokens, m] = X_bf16[tokens, k] * W_bf16[m, k]^T with a float32 result.
/// Throws std::runtime_error when hipBLAS rejects the call.
void LaunchGemmBf16(hipblasHandle_t handle, const void* weight_bf16,
                    const void* input_bf16, float* output, std::size_t tokens,
                    std::size_t m, std::size_t k, hipStream_t stream);

/// hipBLASLt GEMM with the algorithm chosen by measurement.
///
/// rocBLAS has no matrix-core kernel for a transposed BF16 operand with a
/// float32 result and runs a scalar kernel at about 5 TFLOPS on gfx1151.
/// hipBLASLt reaches 17 to 27 TFLOPS at 1,024 rows, but its ranked heuristic
/// list is a poor guide here. The top entry is no faster than rocBLAS, and the
/// best rank moves between 4 and 8 with the shape and the token count. See
/// tests/models/embedding_gemma2/gemm_bench_hip_test.cpp. The first call for a
/// shape therefore times every usable heuristic result on the caller's own
/// operands and keeps the fastest. Later calls reuse it.
///
/// The token count rounds up to a bucket so that nearby batch sizes share one
/// tuned plan. The encoder never passes more than 1,024 rows, which bounds the
/// buckets to twenty and lets it tune them all while the model loads. The
/// extra rows are computed and ignored, so the buffers behind `input_bf16` and
/// `output` must hold `capacity_tokens` rows.
///
/// Not thread-safe. The encoder runs it under the service's device mutex and
/// during load.
class GemmLt {
public:
  GemmLt();
  ~GemmLt();
  GemmLt(const GemmLt&) = delete;
  GemmLt& operator=(const GemmLt&) = delete;

  /// Returns false when hipBLASLt offers no usable algorithm for the shape.
  /// The caller then falls back to LaunchGemmBf16().
  [[nodiscard]] bool Run(const void* weight_bf16, const void* input_bf16,
                         float* output, std::size_t tokens, std::size_t m,
                         std::size_t k, std::size_t capacity_tokens,
                         hipStream_t stream);

  /// Rows the GEMM computes for `tokens` input rows.
  [[nodiscard]] static std::size_t Bucket(std::size_t tokens,
                                          std::size_t capacity_tokens);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::embedding_gemma2::hip
#endif

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_HIP_BLAS_HPP_
