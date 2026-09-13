#ifndef GUFO_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_

#include <cstddef>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Computes Fused SwiGLU GEMV: out = SiLU(W_gate * x) * (W_up * x)
void LaunchFusedSwiGLUGEMV(const void* gate_w, core::GgmlType gate_type,
                           const void* up_w, core::GgmlType up_type,
                           const float* x, float* out,
                           std::size_t intermediate_size,
                           std::size_t hidden_size,
                           hipStream_t stream = nullptr);

/// Batched SwiGLU activation: out = (gate * sigmoid(gate)) * up (optional BF16
/// output)
void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream = nullptr);

/// SwiGLU for adjacent gate/up rows stored as [batch, 2, intermediate_size].
void LaunchPackedSwiGLUActivation(const float* gate_up, float* out,
                                   std::size_t batch_size,
                                   std::size_t intermediate_size,
                                   hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_SWIGLU_HPP_
