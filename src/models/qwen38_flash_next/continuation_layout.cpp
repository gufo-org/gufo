#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next {
namespace {
bool IsDraft(ContinuationPart part) {
  return part == ContinuationPart::kDraftKey ||
         part == ContinuationPart::kDraftValue ||
         part == ContinuationPart::kDraftPooledKeys ||
         part == ContinuationPart::kDraftRawKeys ||
         part == ContinuationPart::kDraftResidual;
}
bool IsPool(ContinuationPart part) {
  return part == ContinuationPart::kPooledKeys ||
         part == ContinuationPart::kDraftPooledKeys;
}
}  // namespace

ContinuationLayout::ContinuationLayout(const Config& c, bool mtp,
                                       std::uint32_t context,
                                       std::uint32_t kept_rows)
    : context_(context), ratio_(c.compress_ratio), mtp_(mtp) {
  if (!context || context > c.context_length || !ratio_ ||
      !c.full_attention_interval || !c.num_layers || !c.AttentionKvDim() ||
      !c.indexer_head_dim || !c.indexer_top_k || !c.ssm_conv_kernel ||
      !c.ssm_head_dim || !c.ssm_num_v_heads || !c.HcDim() || !c.vocab_size ||
      (mtp && !kept_rows))
    throw std::invalid_argument("invalid Flash-Next continuation geometry");
  raw_capacity_ = std::min(context, std::max(c.indexer_top_k, ratio_ - 1));
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            1,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            row ? cache::Rows{256} : 0,
                            state},
                           part,
                           layer});
  };
  add(ContinuationPart::kTokens, sizeof(std::int32_t), 0);
  add(ContinuationPart::kLogits, 0, std::size_t{c.vocab_size} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    if (c.IsLinearLayer(layer)) {
      add(ContinuationPart::kConvolution, 0,
          std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels() *
              sizeof(float),
          layer);
      add(ContinuationPart::kRecurrent, 0,
          std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim *
              sizeof(float),
          layer);
    } else {
      add(ContinuationPart::kKey, std::size_t{c.AttentionKvDim()} * 2, 0,
          layer);
      add(ContinuationPart::kValue, std::size_t{c.AttentionKvDim()} * 2, 0,
          layer);
      add(ContinuationPart::kPooledKeys, std::size_t{c.indexer_head_dim} * 2, 0,
          layer);
      add(ContinuationPart::kRawKeys, 0,
          std::size_t{raw_capacity_} * c.indexer_head_dim * sizeof(float),
          layer);
    }
  }
  if (c.ple_layer >= 0)
    add(ContinuationPart::kPleHistory, 0,
        std::size_t{c.PleConvHistory()} * c.HcDim() * sizeof(float));
  if (mtp) {
    add(ContinuationPart::kDraftKey, std::size_t{c.AttentionKvDim()} * 2, 0);
    add(ContinuationPart::kDraftValue, std::size_t{c.AttentionKvDim()} * 2, 0);
    add(ContinuationPart::kDraftPooledKeys, std::size_t{c.indexer_head_dim} * 2,
        0);
    add(ContinuationPart::kDraftRawKeys, 0,
        std::size_t{raw_capacity_} * c.indexer_head_dim * sizeof(float));
    add(ContinuationPart::kDraftResidual, 0,
        std::size_t{c.HcDim()} * sizeof(float));
    add(ContinuationPart::kKeptHidden, 0,
        std::size_t{kept_rows} * c.HcDim() * sizeof(float));
  }
}

std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t target, std::uint32_t draft, std::uint32_t blocks,
    std::uint32_t draft_blocks) const {
  if (target > context_ || draft > target || blocks > target / ratio_ ||
      draft_blocks > draft / ratio_ ||
      target - blocks * ratio_ > raw_capacity_ ||
      draft - draft_blocks * ratio_ > raw_capacity_ ||
      (!mtp_ && (draft || draft_blocks)))
    throw std::invalid_argument("invalid Flash-Next continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_) {
    const bool draft_part = IsDraft(component.part);
    positions.push_back(
        {component.descriptor.id, IsPool(component.part)
                                      ? (draft_part ? draft_blocks : blocks)
                                      : (draft_part ? draft : target)});
  }
  return positions;
}

std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}

std::size_t ContinuationLayout::RowBytes(std::uint32_t target,
                                         std::uint32_t draft,
                                         std::uint32_t blocks,
                                         std::uint32_t draft_blocks) const {
  const auto positions = Positions(target, draft, blocks, draft_blocks);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}
}  // namespace gufo::models::qwen38_flash_next
