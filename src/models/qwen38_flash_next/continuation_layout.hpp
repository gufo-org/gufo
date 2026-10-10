#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_LAYOUT_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/cache/types.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

// State inventory shared by the adapter and its model-local capacity checks.
// IDs refer to this representation, never to an allocation or serving slot.
enum class ContinuationPart {
  kTokens,
  kLogits,
  kMetadata,
  kConvolution,
  kRecurrent,
  kPleHistory,
  kKey,
  kValue,
  kPooledKeys,
  kRawKeys,
  kDraftKey,
  kDraftValue,
  kDraftPooledKeys,
  kDraftRawKeys,
  kDraftResidual,
  kKeptHidden,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};

// Host metadata has an explicit fixed-size representation. It includes the
// model-owned length policy, PLE n-gram tail, hidden frontier and input
// identity. Vision layout is rebuilt from the matching request attachment on
// restoration.
inline constexpr std::size_t kContinuationMetadataBytes = 160;

class ContinuationLayout {
public:
  ContinuationLayout(const Config&, bool mtp, std::uint32_t context,
                     std::uint32_t kept_rows);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t target, std::uint32_t draft, std::uint32_t blocks,
      std::uint32_t draft_blocks) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  // Completed pools contribute indexer_head_dim * sizeof(f16) per block,
  // rather than per token. Returns the exact total at the supplied frontiers.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t target, std::uint32_t draft,
                                     std::uint32_t blocks,
                                     std::uint32_t draft_blocks) const;
  [[nodiscard]] std::uint32_t RawCapacity() const noexcept {
    return raw_capacity_;
  }

private:
  std::uint32_t context_, ratio_, raw_capacity_;
  bool mtp_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::qwen38_flash_next
#endif
