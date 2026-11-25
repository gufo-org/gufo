#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PRIMITIVES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PRIMITIVES_HPP_

// Handles the Flash-Next stages (stages.hpp) and weight sources
// (weight_source.hpp) exchange with the executor. They name no executor
// type: a Layer is an opaque handle with accessors, a Tensor a device weight
// whose format, when it is not GGUF, is described by extension-owned state.
// Experimental: no compatibility promise yet (see the Qwen family RFC).

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>

namespace gufo::ext::qwen::flash_next {

/// Which arithmetic a pass takes. Prompt passes (prefill chunks, catch-up)
/// keep one route at every chunk width; decode covers decoding and
/// speculative verification.
enum class Phase : std::uint8_t { kDecode, kPrompt };

/// One weight as the device holds it. A GGUF tensor keeps its encoding in
/// type(); a tensor in another format carries the non-null format_state()
/// its weight source attached, and the upload stores it as kI32, which no
/// GGUF kernel accepts, so only the Stages that know the format read it. cols()
/// is the input (reduction) width, rows() the output width, experts() stacks
/// matrices.
class Tensor {
public:
  constexpr Tensor() noexcept = default;
  constexpr Tensor(void* data, ::gufo::core::GgmlType type, std::uint32_t cols,
                   std::uint32_t rows, std::uint32_t experts = 1,
                   const void* format_state = nullptr) noexcept
      : data_(data),
        format_state_(format_state),
        type_(type),
        cols_(cols),
        rows_(rows),
        experts_(experts) {}

  [[nodiscard]] constexpr void* data() const noexcept { return data_; }
  [[nodiscard]] constexpr ::gufo::core::GgmlType type() const noexcept {
    return type_;
  }
  [[nodiscard]] constexpr std::uint32_t cols() const noexcept { return cols_; }
  [[nodiscard]] constexpr std::uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] constexpr std::uint32_t experts() const noexcept {
    return experts_;
  }
  /// Extension-owned description of a non-GGUF encoding; null for GGUF.
  [[nodiscard]] constexpr const void* format_state() const noexcept {
    return format_state_;
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return data_ == nullptr;
  }

private:
  void* data_{nullptr};
  const void* format_state_{nullptr};
  ::gufo::core::GgmlType type_{::gufo::core::GgmlType::kF32};
  std::uint32_t cols_{0};
  std::uint32_t rows_{0};
  std::uint32_t experts_{1};
};

/// The projections of a layer a stage may read (Layer::tensor) or replace
/// for prompt passes (LayerEdit::Set). Stacked forms exist when the upload
/// fused projections of one input: kSsmIn replaces kSsmQkv + kSsmGate,
/// kAttnQkv replaces kAttnQ + kAttnK + kAttnV.
enum class TensorId : std::uint8_t {
  kSsmIn,
  kSsmQkv,
  kSsmGate,
  kSsmAlphaBeta,
  kSsmOut,
  kAttnQkv,
  kAttnQ,
  kAttnK,
  kAttnV,
  kAttnOut,
  kIndexerQ,
  kIndexerK,
  kRouter,
  kFfnGateExps,
  kFfnUpExps,
  kFfnDownExps,
  kShexpGate,
  kShexpUp,
  kShexpDown,
  kNextnFcEmbedding,
  kNextnFcHidden,
};

struct LayerAccess;  // the executor's side of Layer and LayerEdit

/// A device layer of the trunk (index() 0..layers-1) or the MTP draft block
/// (index() == kMtp). Valid while the model is loaded.
class Layer {
public:
  static constexpr std::uint32_t kMtp = 0xFFFFFFFFU;

  constexpr Layer() noexcept = default;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return impl_ != nullptr;
  }
  [[nodiscard]] constexpr std::uint32_t index() const noexcept {
    return index_;
  }
  /// Gated DeltaNet (linear attention) layer, else full attention.
  [[nodiscard]] bool linear() const noexcept;
  [[nodiscard]] Tensor tensor(TensorId id) const noexcept;

private:
  friend struct LayerAccess;
  const void* impl_{nullptr};
  std::uint32_t index_{kMtp};
};

/// The view of one trunk layer prompt passes run (Stages::PromptLayer). It
/// starts as the uploaded layer and keeps every edit across passes.
class LayerEdit {
public:
  [[nodiscard]] Layer layer() const noexcept;
  /// Replaces one projection; an empty Tensor removes it (a stacked form set
  /// alongside takes its place).
  void Set(TensorId id, const Tensor& tensor) noexcept;

private:
  friend struct LayerAccess;
  void* impl_{nullptr};
  std::uint32_t index_{0};
};

}  // namespace gufo::ext::qwen::flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PRIMITIVES_HPP_
