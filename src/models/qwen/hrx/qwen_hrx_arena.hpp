#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_HPP_

#include <hrx/hrx_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/hrx/hrx_buffer_binding.hpp"
#include "src/core/hrx/hrx_owned_buffer.hpp"
#include "src/models/qwen/hrx/qwen_hrx_arena_layout.hpp"
#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"

namespace gufo::hrx {

enum class QwenHrxArenaBuffer : std::size_t {
  kHidden,
  kNormed,
  kAttentionQGate,
  kAttentionQ,
  kAttentionGate,
  kAttentionK,
  kAttentionV,
  kAttentionOutput,
  kRopeCos,
  kRopeSin,
  kSsmQkv,
  kSsmGate,
  kSsmAlpha,
  kSsmBeta,
  kSsmConvOutput,
  kSsmRecurrentOutput,
  kFfnGate,
  kFfnUp,
  kFfnActivation,
  kFfnOutput,
  kLogits,
  kToken,
  kPosition,
  kKvCache,
  kSsmConvState,
  kSsmRecurrentState,
  kSavedSsmConvState,
  kSavedSsmRecurrentState,
  kCount,
};

/// Stable native-HRX storage for the sequential single-token MVP.
/// The arena is non-owning with respect to its device and stream and must be
/// destroyed before the HrxBackend that supplied them is shut down.
class QwenHrxArena {
public:
  QwenHrxArena(const QwenHrxArena&) = delete;
  QwenHrxArena& operator=(const QwenHrxArena&) = delete;
  QwenHrxArena(QwenHrxArena&&) noexcept = default;
  QwenHrxArena& operator=(QwenHrxArena&&) noexcept = default;
  ~QwenHrxArena() = default;

  [[nodiscard]] static std::optional<QwenHrxArena> Create(
      hrx_device_t device, hrx_stream_t stream,
      const QwenHrxArtifactContract& contract, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] hrx_device_t Device() const noexcept { return device_; }
  [[nodiscard]] hrx_stream_t Stream() const noexcept { return stream_; }
  [[nodiscard]] const QwenHrxArenaLayout& Layout() const noexcept {
    return layout_;
  }
  [[nodiscard]] HrxBufferBinding Binding(
      QwenHrxArenaBuffer buffer) const noexcept;
  void SetCurrentPosition(std::uint32_t position) noexcept {
    current_position_ = position;
  }
  [[nodiscard]] std::uint32_t CurrentPosition() const noexcept {
    return current_position_;
  }

  /// Zeros caches and recurrent state with native HRX graph fill nodes. No
  /// synchronous host roundtrip is used here.
  [[nodiscard]] bool Reset(std::string* error_msg = nullptr);
  /// Copies recurrent state with the packaged native F32 copy artifact.
  [[nodiscard]] bool SaveState(hrx_executable_t copy_executable,
                               std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreState(hrx_executable_t copy_executable,
                                  std::string* error_msg = nullptr);

private:
  static constexpr std::size_t kBufferCount =
      static_cast<std::size_t>(QwenHrxArenaBuffer::kCount);

  QwenHrxArena(hrx_device_t device, hrx_stream_t stream,
               QwenHrxArenaLayout layout) noexcept
      : device_(device), stream_(stream), layout_(layout) {}

  hrx_device_t device_{nullptr};
  hrx_stream_t stream_{nullptr};
  QwenHrxArenaLayout layout_{};
  std::array<HrxOwnedBuffer, kBufferCount> buffers_{};
  std::uint32_t current_position_{0};
  std::uint32_t saved_position_{0};
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_HPP_
