#ifndef STRIX_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
#define STRIX_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

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

/// Fuses the residual add with the subsequent RMSNorm into one launch
/// (opt-c010-residual-rmsnorm): writes out_sum = a + b in place and
/// out = (out_sum / sqrt(mean(out_sum^2) + eps)) * weight. Unfused reference:
/// LaunchResidualAdd followed by LaunchRMSNorm.
void LaunchFusedResidualAddRMSNorm(const float* a, const float* b,
                                   float* out_sum, const float* weight,
                                   float* out, std::size_t dim,
                                   float eps = 1e-6F,
                                   hipStream_t stream = nullptr);

/// Batched RMSNorm across B tokens (optional BF16 output in single pass)
void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps = 1e-6F,
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

/// Batched fuse of residual add and RMSNorm across B tokens into one launch
/// (opt-c010-residual-rmsnorm), with optional BF16 normed output. Unfused
/// reference: LaunchBatchedResidualAdd followed by LaunchBatchedRMSNorm.
void LaunchBatchedFusedResidualAddRMSNorm(const float* a, const float* b,
                                          float* out_sum, const float* weight,
                                          float* out, void* out_bf16,
                                          std::size_t batch_size,
                                          std::size_t dim, float eps = 1e-6F,
                                          hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_OPS_NORM_RESIDUAL_HPP_
