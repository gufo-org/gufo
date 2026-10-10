#ifndef GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_CONTINUATION_LAYOUT_HPP_

#include <span>
#include <vector>

#include "src/cache/types.hpp"
#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::models::qwen {

enum class ContinuationPart {
  kMetadata,
  kLogits,
  kKey,
  kValue,
  kConvolution,
  kRecurrent
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{};
};

inline constexpr std::size_t kContinuationMetadataBytes = 64;

// KV rows have a canonical token/head/channel order. FP32's native head-major
// planes are gathered without conversion; their layout version differs from
// FP16. The recurrent storage policy also has its own layout version.
class ContinuationLayout {
public:
  ContinuationLayout(const core::ModelConfig&, hip::QwenExecutionPolicy,
                     std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      cache::Rows) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  [[nodiscard]] std::size_t BytesPerToken() const noexcept;

private:
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::qwen
#endif
