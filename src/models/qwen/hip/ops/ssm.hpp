#ifndef GUFO_MODELS_QWEN_HIP_OPS_SSM_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_SSM_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

inline constexpr std::size_t kSsmReplayCapacity = 16;

struct SsmReplayCapture {
  float* qkv{nullptr};
  float* alpha{nullptr};
  float* beta{nullptr};
  const std::uint32_t* position{nullptr};
  const std::uint32_t* enabled{nullptr};
};

struct SsmSequenceState {
  float* conv{nullptr};
  void* recurrent{nullptr};
  SsmReplayCapture replay;
  std::uint32_t row_offset{0};
  std::uint32_t rows{0};
};

/// Captures the raw recurrent inputs for every row in a verification batch so
/// a rejected speculative suffix can restore the checkpoint and replay only
/// the committed SSM transitions.
void LaunchCaptureBatchedSsmReplay(
    const float* qkv, const float* alpha, const float* beta,
    SsmReplayCapture replay_capture, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::size_t qkv_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Computes Fused SSM Input Projections (QKV, Gate, Alpha, Beta) in a single
/// kernel
void LaunchFusedSSMInputProjections(
    const void* qkv_w, core::GgmlType qkv_type, const void* gate_w,
    core::GgmlType gate_type, const void* alpha_w, core::GgmlType alpha_type,
    const void* beta_w, core::GgmlType beta_type, const float* x,
    float* qkv_out, float* gate_out, float* alpha_out, float* beta_out,
    std::size_t hidden_size, std::size_t qkv_size, std::size_t inner_size,
    std::size_t time_step_rank, hipStream_t stream = nullptr);

/// Runs `rows` consecutive verification rows through the conv and DeltaNet
/// recurrence in a single pair of launches. Each row's arithmetic and its order
/// are identical to the one-row-per-launch form, so the result is bit-exact;
/// what it removes is the launch serialization of 2 x rows dispatches per
/// layer. A null `out_buf` updates only the recurrent/conv state for replay,
/// omitting query normalization, output dot products and output normalization.
void LaunchSSMConvRecurrenceRows(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    std::uint32_t rows, std::size_t projection_row_stride,
    std::size_t inner_row_stride, hipStream_t stream = nullptr,
    SsmReplayCapture replay_capture = {},
    QwenRecurrentStateStorage state_storage = QwenRecurrentStateStorage::kFp32);

/// Runs independent request states in one pair of launches. Row offsets index
/// the shared projection buffers; each sequence retains its own causal order
/// and replay positions. A single sequence uses the direct rows launcher.
void LaunchSSMConvRecurrenceBatch(
    const float* qkv_in, const float* conv_weights, float* conv_out,
    const float* alpha_buf, const float* beta_buf, const float* ssm_a,
    const float* ssm_dt, const float* ssm_norm, const float* gate,
    float* out_buf, std::span<const SsmSequenceState> sequences,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    std::size_t projection_row_stride, std::size_t inner_row_stride,
    hipStream_t stream = nullptr,
    QwenRecurrentStateStorage state_storage = QwenRecurrentStateStorage::kFp32);

void LaunchSSMConvRecurrence(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream = nullptr, SsmReplayCapture replay_capture = {},
    QwenRecurrentStateStorage state_storage = QwenRecurrentStateStorage::kFp32);

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
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf,
    std::uint32_t layer_idx, std::size_t batch_size, std::size_t qkv_size,
    std::uint32_t num_key_heads, std::uint32_t num_heads, std::uint32_t key_dim,
    std::uint32_t val_dim, hipStream_t stream = nullptr,
    QwenRecurrentStateStorage state_storage = QwenRecurrentStateStorage::kFp32);

/// True when the row-split DeltaNet recurrence supports this state shape. Its
/// register tile is built for key_dim == val_dim == 128.
[[nodiscard]] bool IsDeltaNetRowSplitSupported(std::uint32_t key_dim,
                                               std::uint32_t val_dim) noexcept;

/// Batched Causal SSM Conv1D + row-split DeltaNet recurrence + per-head
/// post-RMSNorm/SiLU gate (opt-c170-deltanet-rowsplit). `kq_scales` holds
/// 3 floats per (token, key head) and `alpha_beta` 2 floats per (token, value
/// head); both are pure scratch. `out_buf` carries the recurrence output and is
/// then normalized and gated in place. When `q8_out` is non-null the epilogue
/// writes the tiled Q8_1 activation there instead of the FP32 row, which is
/// valid only when nothing else reads the FP32 form. Alternatively, `fp16_out`
/// receives the gated row as FP16. Only one activation destination may be set.
void LaunchBatchedSSMConvRecurrenceRowSplit(
    const float* qkv_in, const float* conv_weights, float* conv_state,
    float* conv_out, void* deltanet_state, const float* alpha_buf,
    const float* beta_buf, const float* ssm_a, const float* ssm_dt,
    const float* ssm_norm, const float* gate, float* out_buf, void* q8_out,
    float* kq_scales, float* alpha_beta, std::uint32_t layer_idx,
    std::size_t batch_size, std::size_t qkv_size, std::uint32_t num_key_heads,
    std::uint32_t num_heads, std::uint32_t key_dim, std::uint32_t val_dim,
    hipStream_t stream = nullptr,
    QwenRecurrentStateStorage state_storage = QwenRecurrentStateStorage::kFp32,
    void* fp16_out = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_SSM_HPP_
