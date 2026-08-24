#ifndef STRIX_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_
#define STRIX_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_

#include <cstddef>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

/// Computes Fused SwiGLU GEMV: out = SiLU(W_gate * x) * (W_up * x)
void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Fused layer pre-RMSNorm + FFN SwiGLU gate/up GEMV (BF16 weights).
void LaunchFusedRMSNormSwiGLUGEMV(const float* x, const float* norm_w,
                                  float eps, const void* gate_w,
                                  const void* up_w, float* out,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

/// Batched SwiGLU activation: out = (gate * sigmoid(gate)) * up (optional BF16
/// output)
void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream = nullptr);

/// Batched Fused SwiGLU GEMM: Out[B, intermediate] = SiLU(X[B, K] * W_gate^T) *
/// (X[B, K] * W_up^T), with optional BF16 output (opt-c010-ffn-swiglu)
void LaunchBatchedFusedSwiGLUGEMM(const void* gate_w, bool gate_is_bf16,
                                  const void* up_w, bool up_is_bf16,
                                  const float* X, float* out, void* out_bf16,
                                  std::size_t batch_size,
                                  std::size_t intermediate_size,
                                  std::size_t hidden_size,
                                  hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_
