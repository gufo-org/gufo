#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CONTINUATION_LAYOUT_HPP_

#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/types.hpp>
#include <span>
#include <vector>

namespace gufo::models::deepseek_v4_flash {

// State inventory shared by the adapter and its model-local capacity checks.
// IDs refer to this representation, never to an allocation or serving slot.
enum class ContinuationPart {
  kTokens,
  kLogits,
  kMetadata,
  // Sliding window plus the compressor frontiers. Ring slots are overwritten
  // and frontiers change every token, so neither can be shared between
  // checkpoints.
  kLayerState,
  // Compressed rows are final once emitted, one per `ratio` tokens.
  kAttentionRows,
  kIndexerRows,
  kDraftWindow,
};

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t index{0};  // Target layer, or DSpark stage for kDraftWindow.
};

struct ContinuationGeometry {
  std::vector<std::uint32_t> ratios;  // Per layer; zero has no compressor.
  std::uint32_t head_dim{}, indexer_head_dim{}, window{}, vocabulary{};
  std::uint32_t draft_stages{};
};

// Bytes of one compressor frontier tensor (values or scores).
[[nodiscard]] std::size_t CompressorStateBytes(std::uint32_t ratio,
                                               std::uint32_t width);

inline constexpr std::size_t kContinuationMetadataBytes = 192;

class ContinuationLayout {
public:
  ContinuationLayout(ContinuationGeometry, std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] const ContinuationGeometry& Geometry() const {
    return geometry_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t position) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  // Exact row bytes at a frontier; a checkpoint adds PrivateBytes() to them.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t position) const;
  // Rows, tails and private frontiers share the same committed pages.
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  ContinuationGeometry geometry_;
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};

}  // namespace gufo::models::deepseek_v4_flash
#endif
