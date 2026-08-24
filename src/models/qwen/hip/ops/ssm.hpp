#ifndef STRIX_MODELS_QWEN_HIP_OPS_SSM_HPP_
#define STRIX_MODELS_QWEN_HIP_OPS_SSM_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip {

inline constexpr std::size_t kSsmReplayCapacity = 16;

struct SsmReplayCapture {
  float* qkv{nullptr};
  float* alpha{nullptr};
  float* beta{nullptr};
  const std::uint32_t* position{nullptr};
  const std::uint32_t* enabled{nullptr};
};

/// Computes Fused SSM Input Projections (QKV, Gate, Alpha, Beta) in a single
/// kernel
void LaunchFusedSSMInputProjections(
    const void* qkv_w, core::GgmlType qkv_type, const void* gate_w,
    core::GgmlType gate_type, const void* alpha_w, core::GgmlType alpha_type,
    const void* beta_w, core::GgmlType beta_type, const float* x,
    float* qkv_out, float* gate_out, float* alpha_out, float* beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);


/// Fused layer pre-RMSNorm + SSM input projections (QKV, Gate, Alpha, Beta).
void LaunchFusedRMSNormSSMInputProjections(
    const float* x, const float* norm_w, float eps, const void* qkv_w,
    bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, float* qkv_out, float* gate_out, float* alpha_out,
    float* beta_out, std::size_t hidden_size, std::size_t qkv_size,
    std::size_t inner_size, std::size_t time_step_rank,
    hipStream_t stream = nullptr);


void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream = nullptr, SsmReplayCapture replay_capture = {});

/// Batched Fused SSM Input Projections across B tokens
void LaunchBatchedFusedSSMInputProjections(
    const void* qkv_w, bool qkv_is_bf16, const void* gate_w, bool gate_is_bf16,
    const void* alpha_w, bool alpha_is_bf16, const void* beta_w,
    bool beta_is_bf16, const float* X, float* qkv_out, float* gate_out,
    float* alpha_out, float* beta_out, std::size_t batch_size,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Batched Causal SSM Conv1D + DeltaNet Recurrence for B tokens
void LaunchBatchedSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr);

/// Batched Causal SSM Conv1D + DeltaNet recurrence with the per-head
/// post-RMSNorm + SiLU gate folded into the recurrence epilogue
/// (opt-c010-ssm-gate-residual). Writes the final gated output into out_buf.
void LaunchBatchedSSMConvRecurrenceNormGate(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, float* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr);

}  // namespace strix::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_OPS_SSM_HPP_
