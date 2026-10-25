#ifndef GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Computes RMSNorm on GPU: out = (x / sqrt(mean(x^2) + eps)) * weight
void LaunchRMSNorm(const float* x, const float* weight, float* out,
                   std::size_t dim, float eps = 1e-6F,
                   hipStream_t stream = nullptr);

/// Computes per-head RMSNorm across num_heads
void LaunchPerHeadRMSNorm(const float* x, const float* weight, float* out,
                          std::uint32_t num_heads, std::uint32_t head_dim,
                          float eps = 1e-6F, hipStream_t stream = nullptr);

/// Computes residual add: out = a + b
void LaunchResidualAdd(const float* a, const float* b, float* out,
                       std::size_t dim, hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
                          hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens with a strided output row pitch. The
/// arithmetic is identical to `LaunchBatchedRMSNorm`: row `row` is written at
/// `out + row * out_row_stride` and the input rows stay packed at `dim`.
/// The MTP prompt-prime path uses it to land the two normed halves of a
/// 2h-wide fusion row inside one buffer (MTP priming plan, stage A).
void LaunchBatchedRMSNormStrided(const float* x, const float* weight,
                                 float* out, std::size_t out_row_stride,
                                 std::size_t batch_size, std::size_t dim,
                                 float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Batched Per-Head RMSNorm across B tokens
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps = 1e-6F,
                                 hipStream_t stream = nullptr);

/// Batched Residual Add across B tokens
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
