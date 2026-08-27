#include "src/models/qwen/hrx/qwen_hrx_arena.hpp"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace gufo::hrx {
namespace {

constexpr std::size_t BufferIndex(QwenHrxArenaBuffer buffer) noexcept {
  return static_cast<std::size_t>(buffer);
}

bool Reject(const char* message, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
  return false;
}

bool DispatchCopy(hrx_stream_t stream, hrx_executable_t executable,
                  const HrxBufferBinding& source,
                  const HrxBufferBinding& destination, std::string* error_msg) {
  constexpr std::uint32_t kCopyCapacity = 50'331'648;
  constexpr std::uint32_t kWorkgroupSize = 256;
  if (stream == nullptr || executable == nullptr || !source.IsValid() ||
      !destination.IsValid() || source.length != destination.length ||
      (source.length % sizeof(float)) != 0) {
    return Reject("native HRX state-copy operands are invalid", error_msg);
  }
  const std::size_t element_count = source.length / sizeof(float);
  if (element_count == 0 || element_count > kCopyCapacity ||
      (element_count % kWorkgroupSize) != 0) {
    return Reject("native HRX state-copy length violates artifact capacity",
                  error_msg);
  }
  hrx_buffer_ref_t bindings[2];
  if (!TryMakeBufferRef(source, source.length, &bindings[0]) ||
      !TryMakeBufferRef(destination, destination.length, &bindings[1])) {
    return Reject("native HRX state-copy binding is invalid", error_msg);
  }
  hrx_dispatch_config_t config{};
  config.workgroup_count[0] = kCopyCapacity / kWorkgroupSize;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  config.workgroup_size[0] = kWorkgroupSize;
  config.workgroup_size[1] = 1;
  config.workgroup_size[2] = 1;
  config.subgroup_size = 32;
  const auto elements = static_cast<std::uint32_t>(element_count);
  auto status = hrx_stream_dispatch(stream, executable, 0, &config, &elements,
                                    sizeof(elements), bindings, 2, 0);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return Reject("native HRX state-copy dispatch failed", error_msg);
  }
  return true;
}

bool VerifyZeroEdges(hrx_device_t device, const HrxBufferBinding& binding,
                     std::string* error_msg) {
  if (!binding.IsValid() || binding.length < sizeof(std::uint32_t) ||
      (binding.length % sizeof(std::uint32_t)) != 0) {
    return Reject("native HRX reset verification range is invalid", error_msg);
  }
  std::uint32_t first = 1;
  std::uint32_t last = 1;
  const HrxBufferBinding first_word{
      .buffer = binding.buffer,
      .offset = binding.offset,
      .length = sizeof(first),
  };
  const HrxBufferBinding last_word{
      .buffer = binding.buffer,
      .offset = binding.offset + binding.length - sizeof(last),
      .length = sizeof(last),
  };
  if (!HrxCopyToHost(device, first_word, &first, sizeof(first), error_msg) ||
      !HrxCopyToHost(device, last_word, &last, sizeof(last), error_msg)) {
    return false;
  }
  return (first == 0 && last == 0) ||
         Reject("native HRX reset readback observed non-zero state", error_msg);
}

std::array<std::size_t, static_cast<std::size_t>(QwenHrxArenaBuffer::kCount)>
BufferSizes(const QwenHrxArenaLayout& layout) {
  return {
      layout.hidden_bytes,
      layout.normed_bytes,
      layout.attention_q_gate_bytes,
      layout.attention_q_bytes,
      layout.attention_gate_bytes,
      layout.attention_k_bytes,
      layout.attention_v_bytes,
      layout.attention_output_bytes,
      layout.rope_cos_bytes,
      layout.rope_sin_bytes,
      layout.ssm_qkv_bytes,
      layout.ssm_gate_bytes,
      layout.ssm_alpha_bytes,
      layout.ssm_beta_bytes,
      layout.ssm_conv_output_bytes,
      layout.ssm_recurrent_output_bytes,
      layout.ffn_gate_bytes,
      layout.ffn_up_bytes,
      layout.ffn_activation_bytes,
      layout.ffn_output_bytes,
      layout.logits_bytes,
      layout.token_bytes,
      layout.position_bytes,
      layout.kv_cache_bytes,
      layout.ssm_conv_state_bytes,
      layout.ssm_recurrent_state_bytes,
      layout.saved_ssm_conv_state_bytes,
      layout.saved_ssm_recurrent_state_bytes,
  };
}

constexpr std::array<const char*,
                     static_cast<std::size_t>(QwenHrxArenaBuffer::kCount)>
    kBufferNames = {
        "hidden",
        "normed",
        "attention_q_gate",
        "attention_q",
        "attention_gate",
        "attention_k",
        "attention_v",
        "attention_output",
        "rope_cos",
        "rope_sin",
        "ssm_qkv",
        "ssm_gate",
        "ssm_alpha",
        "ssm_beta",
        "ssm_conv_output",
        "ssm_recurrent_output",
        "ffn_gate",
        "ffn_up",
        "ffn_activation",
        "ffn_output",
        "logits",
        "token",
        "position",
        "kv_cache",
        "ssm_conv_state",
        "ssm_recurrent_state",
        "saved_ssm_conv_state",
        "saved_ssm_recurrent_state",
};

}  // namespace

std::optional<QwenHrxArena> QwenHrxArena::Create(
    hrx_device_t device, hrx_stream_t stream,
    const QwenHrxArtifactContract& contract, std::uint32_t max_context,
    std::string* error_msg) {
  if (device == nullptr || stream == nullptr) {
    Reject("native HRX arena requires an initialized device and stream",
           error_msg);
    return std::nullopt;
  }
  auto layout = QwenHrxArenaLayout::Create(contract, max_context, error_msg);
  if (!layout.has_value()) {
    return std::nullopt;
  }

  QwenHrxArena arena(device, stream, *layout);
  const auto sizes = BufferSizes(*layout);
  for (std::size_t index = 0; index < sizes.size(); ++index) {
    std::string allocation_error;
    auto allocation =
        HrxOwnedBuffer::Allocate(stream, sizes[index], &allocation_error);
    if (!allocation.has_value()) {
      if (error_msg != nullptr) {
        *error_msg = "failed to allocate native HRX arena buffer '";
        *error_msg += kBufferNames[index];
        *error_msg += "': ";
        *error_msg += allocation_error;
      }
      return std::nullopt;
    }
    arena.buffers_[index] = std::move(*allocation);
  }

  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return std::optional<QwenHrxArena>{std::move(arena)};
}

HrxBufferBinding QwenHrxArena::Binding(QwenHrxArenaBuffer buffer,
                                       std::size_t offset_bytes,
                                       std::size_t length) const noexcept {
  const std::size_t index = BufferIndex(buffer);
  if (index >= buffers_.size()) {
    return {};
  }
  const auto base = buffers_[index].Binding();
  if (offset_bytes > base.length) {
    return {};
  }
  const std::size_t actual_length =
      (length == 0) ? (base.length - offset_bytes) : length;
  if (offset_bytes + actual_length > base.length) {
    return {};
  }
  return HrxBufferBinding{
      .buffer = base.buffer,
      .offset = base.offset + offset_bytes,
      .length = actual_length,
  };
}

bool QwenHrxArena::PrecomputeRope(float rope_theta, std::string* error_msg) {
  constexpr std::size_t kRotaryDim = 64;
  constexpr std::size_t kHalfDim = kRotaryDim / 2;
  const std::size_t total_elements = layout_.max_context * kHalfDim;
  std::vector<float> cos_table(total_elements);
  std::vector<float> sin_table(total_elements);

  for (std::uint32_t pos = 0; pos < layout_.max_context; ++pos) {
    for (std::size_t dim = 0; dim < kHalfDim; ++dim) {
      const double exponent =
          2.0 * static_cast<double>(dim) / static_cast<double>(kRotaryDim);
      const double frequency =
          1.0 / std::pow(static_cast<double>(rope_theta), exponent);
      const double angle = static_cast<double>(pos) * frequency;
      cos_table[pos * kHalfDim + dim] = static_cast<float>(std::cos(angle));
      sin_table[pos * kHalfDim + dim] = static_cast<float>(std::sin(angle));
    }
  }

  const auto cos_binding = Binding(QwenHrxArenaBuffer::kRopeCos);
  const auto sin_binding = Binding(QwenHrxArenaBuffer::kRopeSin);
  if (!HrxCopyFromHost(device_, cos_table.data(), cos_binding,
                       cos_table.size() * sizeof(float), error_msg) ||
      !HrxCopyFromHost(device_, sin_table.data(), sin_binding,
                       sin_table.size() * sizeof(float), error_msg)) {
    return false;
  }
  return true;
}

bool QwenHrxArena::Reset(std::string* error_msg) {
  static_assert(kHrxNativeDeviceFillAvailable);
  constexpr std::array state_buffers{
      QwenHrxArenaBuffer::kToken,
      QwenHrxArenaBuffer::kPosition,
      QwenHrxArenaBuffer::kKvCache,
      QwenHrxArenaBuffer::kSsmConvState,
      QwenHrxArenaBuffer::kSsmRecurrentState,
      QwenHrxArenaBuffer::kSavedSsmConvState,
      QwenHrxArenaBuffer::kSavedSsmRecurrentState,
  };
  for (const auto buffer : state_buffers) {
    const auto binding = Binding(buffer);
    if (!HrxFillBuffer(device_, stream_, binding, 0, error_msg)) {
      return false;
    }
  }
  current_position_ = 0;
  saved_position_ = 0;
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

bool QwenHrxArena::SaveState(hrx_executable_t copy_executable,
                             std::string* error_msg) {
  static_assert(kHrxNativeDeviceCopyAvailable);
  if (!DispatchCopy(
          stream_, copy_executable, Binding(QwenHrxArenaBuffer::kSsmConvState),
          Binding(QwenHrxArenaBuffer::kSavedSsmConvState), error_msg) ||
      !DispatchCopy(stream_, copy_executable,
                    Binding(QwenHrxArenaBuffer::kSsmRecurrentState),
                    Binding(QwenHrxArenaBuffer::kSavedSsmRecurrentState),
                    error_msg)) {
    return false;
  }
  const auto status = hrx_stream_synchronize(stream_);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return Reject("native HRX snapshot synchronization failed", error_msg);
  }
  saved_position_ = current_position_;
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

bool QwenHrxArena::RestoreState(hrx_executable_t copy_executable,
                                std::string* error_msg) {
  static_assert(kHrxNativeDeviceCopyAvailable);
  if (!DispatchCopy(stream_, copy_executable,
                    Binding(QwenHrxArenaBuffer::kSavedSsmConvState),
                    Binding(QwenHrxArenaBuffer::kSsmConvState), error_msg) ||
      !DispatchCopy(stream_, copy_executable,
                    Binding(QwenHrxArenaBuffer::kSavedSsmRecurrentState),
                    Binding(QwenHrxArenaBuffer::kSsmRecurrentState),
                    error_msg)) {
    return false;
  }
  const auto status = hrx_stream_synchronize(stream_);
  if (!hrx_status_is_ok(status)) {
    hrx_status_ignore(status);
    return Reject("native HRX restore synchronization failed", error_msg);
  }
  current_position_ = saved_position_;
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

}  // namespace gufo::hrx
