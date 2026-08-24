#ifndef STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_
#define STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_

#include <cstdint>

#include "src/core/model_config.hpp"
#include "src/models/qwen/state.hpp"

namespace strix::models::qwen {

/// Capability token for stateless CPU module calls.
struct CpuModuleContext final {};

/// CPU capabilities required by stateful layer modules. Non-owning pointers
/// keep ownership in the composition layer; the constructor guarantees they
/// are non-null.
class CpuLayerContext final {
public:
  CpuLayerContext(const core::ModelConfig& config, QwenScratchArena& scratch,
                  std::uint32_t layer_idx = 0,
                  std::uint32_t position = 0) noexcept
      : config_(&config),
        scratch_(&scratch),
        layer_idx_(layer_idx),
        position_(position) {}

  [[nodiscard]] const core::ModelConfig& Config() const noexcept {
    return *config_;
  }
  [[nodiscard]] QwenScratchArena& Scratch() const noexcept { return *scratch_; }
  [[nodiscard]] std::uint32_t LayerIndex() const noexcept { return layer_idx_; }
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

private:
  const core::ModelConfig* config_;
  QwenScratchArena* scratch_;
  std::uint32_t layer_idx_;
  std::uint32_t position_;
};

/// HIP launch capability. A null stream is valid and denotes the HIP default
/// stream; no CPU scratch/config pointers can accidentally be paired with this
/// capability.
class HipModuleContext final {
public:
  explicit HipModuleContext(void* stream = nullptr, std::uint32_t layer_idx = 0,
                            std::uint32_t position = 0) noexcept
      : stream_(stream), layer_idx_(layer_idx), position_(position) {}

  [[nodiscard]] void* Stream() const noexcept { return stream_; }
  [[nodiscard]] std::uint32_t LayerIndex() const noexcept { return layer_idx_; }
  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }

private:
  void* stream_;
  std::uint32_t layer_idx_;
  std::uint32_t position_;
};

}  // namespace strix::models::qwen

#endif  // STRIX_MODELS_QWEN_MODULES_MODULE_CTX_HPP_
