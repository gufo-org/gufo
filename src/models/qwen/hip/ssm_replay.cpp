#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"

namespace gufo::hip {

void QwenGpuArena::AllocateRecurrentSnapshot() {
  if (d_saved_ssm_conv_state_ != nullptr) {
    return;
  }

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  HIP_CHECK(hipMalloc(&d_saved_ssm_conv_state_, total_conv * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_saved_ssm_deltanet_state_,
                      total_deltanet * QwenRecurrentStateElementBytes(
                                           policy_.recurrent_state_storage)));
}

void QwenGpuArena::SaveState(std::uint32_t valid_context) {
  if (valid_context > max_context_) {
    throw std::length_error("saved GPU state exceeds the context length");
  }
  AllocateRecurrentSnapshot();

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  // KV entries are append-only and every attention launch is bounded by its
  // explicit position. Draft entries beyond valid_context can remain in place:
  // accepted positions reuse them and rejected positions are overwritten.
  HIP_CHECK(hipMemcpyAsync(d_saved_ssm_conv_state_, d_ssm_conv_state,
                           total_conv * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(
      d_saved_ssm_deltanet_state_, d_ssm_deltanet_state,
      total_deltanet *
          QwenRecurrentStateElementBytes(policy_.recurrent_state_storage),
      hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  saved_context_ = valid_context;
  replay_last_position_ = valid_context;
  replay_captured_positions_ = 0;
  has_saved_state_ = true;
}

void QwenGpuArena::RestoreState() {
  if (!has_saved_state_) {
    throw std::logic_error("GPU state has not been saved");
  }

  const std::size_t total_conv =
      config_.num_layers * config_.SsmQkvSize() * config_.ssm_conv_kernel;
  const std::size_t total_deltanet =
      config_.num_layers * config_.ssm_time_step_rank * config_.ssm_state_size *
      config_.SsmValueSize();

  HIP_CHECK(hipMemcpyAsync(d_ssm_conv_state, d_saved_ssm_conv_state_,
                           total_conv * sizeof(float), hipMemcpyDeviceToDevice,
                           stream));
  HIP_CHECK(hipMemcpyAsync(
      d_ssm_deltanet_state, d_saved_ssm_deltanet_state_,
      total_deltanet *
          QwenRecurrentStateElementBytes(policy_.recurrent_state_storage),
      hipMemcpyDeviceToDevice, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

bool QwenGpuArena::AllocateSsmReplayLog() {
  if (d_ssm_replay_qkv_ != nullptr) {
    return false;
  }

  const std::size_t layer_slots =
      static_cast<std::size_t>(config_.num_layers) * kSsmReplayCapacity;
  HIP_CHECK(hipMalloc(&d_ssm_replay_qkv_,
                      layer_slots * config_.SsmQkvSize() * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_ssm_replay_alpha_,
                layer_slots * config_.ssm_time_step_rank * sizeof(float)));
  HIP_CHECK(
      hipMalloc(&d_ssm_replay_beta_,
                layer_slots * config_.ssm_time_step_rank * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_ssm_replay_enabled_, sizeof(std::uint32_t)));
  return true;
}

bool QwenGpuArena::BeginSsmReplayCapture() {
  const bool allocated = AllocateSsmReplayLog();
  HIP_CHECK(
      hipMemsetAsync(d_ssm_replay_enabled_, 1, sizeof(std::uint32_t), stream));
  replay_capture_active_ = true;
  replay_last_position_ = saved_context_;
  replay_captured_positions_ = 0;
  return allocated;
}

void QwenGpuArena::DisableSsmReplayCapture() noexcept {
  if (d_ssm_replay_enabled_ != nullptr) {
    HIP_CHECK(hipMemsetAsync(d_ssm_replay_enabled_, 0, sizeof(std::uint32_t),
                             stream));
  }
  replay_capture_active_ = false;
}

void QwenGpuArena::MarkSsmReplayPosition(std::uint32_t position) noexcept {
  if (replay_capture_active_ && position >= saved_context_) {
    replay_last_position_ = std::max(replay_last_position_, position);
    replay_captured_positions_ =
        std::max(replay_captured_positions_,
                 static_cast<std::size_t>(position - saved_context_) + 1);
  }
}

bool QwenGpuArena::CanReplaySsmPosition(std::uint32_t position) const noexcept {
  if (!has_saved_state_ || d_ssm_replay_qkv_ == nullptr ||
      replay_captured_positions_ == 0) {
    return false;
  }
  return replay_captured_positions_ <= kSsmReplayCapacity &&
         position >= saved_context_ && position <= replay_last_position_;
}

SsmReplayCapture QwenGpuArena::GetSsmReplayCapture() const noexcept {
  return {
      .qkv = d_ssm_replay_qkv_,
      .alpha = d_ssm_replay_alpha_,
      .beta = d_ssm_replay_beta_,
      .position = d_prompt_tokens + 1,
      .enabled = d_ssm_replay_enabled_,
  };
}

const float* QwenGpuArena::GetReplayQkv(std::uint32_t layer,
                                        std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.SsmQkvSize();
  return d_ssm_replay_qkv_ + offset;
}

const float* QwenGpuArena::GetReplayAlpha(std::uint32_t layer,
                                          std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.ssm_time_step_rank;
  return d_ssm_replay_alpha_ + offset;
}

const float* QwenGpuArena::GetReplayBeta(std::uint32_t layer,
                                         std::uint32_t position) const {
  if (!CanReplaySsmPosition(position)) {
    throw std::out_of_range("SSM replay position is outside the capture");
  }
  const std::size_t slot =
      static_cast<std::size_t>(position) % kSsmReplayCapacity;
  const std::size_t offset =
      (static_cast<std::size_t>(layer) * kSsmReplayCapacity + slot) *
      config_.ssm_time_step_rank;
  return d_ssm_replay_beta_ + offset;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
