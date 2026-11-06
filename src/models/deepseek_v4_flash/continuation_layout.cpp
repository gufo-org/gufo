#include "src/models/deepseek_v4_flash/continuation_layout.hpp"

#include <stdexcept>
#include <utility>

namespace gufo::models::deepseek_v4_flash {
namespace {
// A chunk should fill a backing page. Sparse layers compress 32x harder, so
// their chunks cover a longer prefix than dense layers.
constexpr cache::Rows kDenseChunkRows = 256;
constexpr cache::Rows kSparseChunkRows = 32;
constexpr std::uint32_t kOverlappedRatio = 4;
}  // namespace

std::size_t CompressorStateBytes(std::uint32_t ratio, std::uint32_t width) {
  // Overlapped compressors keep two windows of twice the width.
  const std::size_t overlap = ratio == kOverlappedRatio ? 2 : 1;
  return overlap * width * overlap * ratio * sizeof(float);
}

ContinuationLayout::ContinuationLayout(ContinuationGeometry geometry,
                                       std::uint32_t context)
    : geometry_(std::move(geometry)), context_(context) {
  const auto& g = geometry_;
  if (!context || g.ratios.empty() || !g.head_dim || !g.indexer_head_dim ||
      !g.window || !g.vocabulary)
    throw std::invalid_argument("invalid DeepSeek continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       cache::Rows chunk, std::size_t state,
                       std::uint32_t index = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            1,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            chunk,
                            state},
                           part,
                           index});
  };
  const std::size_t head_row = std::size_t{g.head_dim} * sizeof(float);
  const std::size_t indexer_row =
      std::size_t{g.indexer_head_dim} * sizeof(float);
  add(ContinuationPart::kTokens, sizeof(std::int32_t), kDenseChunkRows, 0);
  add(ContinuationPart::kLogits, 0, 0,
      std::size_t{g.vocabulary} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < g.ratios.size(); ++layer) {
    const auto ratio = g.ratios[layer];
    std::size_t state = g.window * head_row;
    if (ratio)
      state += 2 * CompressorStateBytes(ratio, g.head_dim);
    if (ratio == kOverlappedRatio)
      state += 2 * CompressorStateBytes(ratio, g.indexer_head_dim);
    add(ContinuationPart::kLayerState, 0, 0, state, layer);
    if (ratio)
      add(ContinuationPart::kAttentionRows, head_row,
          ratio == kOverlappedRatio ? kDenseChunkRows : kSparseChunkRows, 0,
          layer);
    if (ratio == kOverlappedRatio)
      add(ContinuationPart::kIndexerRows, indexer_row, kDenseChunkRows, 0,
          layer);
  }
  for (std::uint32_t stage = 0; stage < g.draft_stages; ++stage)
    add(ContinuationPart::kDraftWindow, 0, 0, g.window * head_row, stage);
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position > context_)
    throw std::invalid_argument("invalid DeepSeek continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_) {
    const bool compressed =
        component.part == ContinuationPart::kAttentionRows ||
        component.part == ContinuationPart::kIndexerRows;
    positions.push_back(
        {component.descriptor.id,
         compressed ? position / geometry_.ratios[component.index] : position});
  }
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  const auto positions = Positions(position);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}

std::vector<cache::BackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // Contiguous page groups retain one payload owner and charge all slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}

}  // namespace gufo::models::deepseek_v4_flash
