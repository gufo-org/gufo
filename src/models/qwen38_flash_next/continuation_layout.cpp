#include "src/models/qwen38_flash_next/continuation_layout.hpp"

#include <algorithm>
#include <map>
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
    // Private state belongs to this target checkpoint, including the lagging
    // predictor's residual and raw ring. Its execution frontier is recorded
    // by the draft row components and private metadata.
    positions.push_back(
        {component.descriptor.id,
         component.descriptor.kind == cache::ComponentKind::kPrivateState
             ? target
         : IsPool(component.part) ? (draft_part ? draft_blocks : blocks)
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
std::vector<ContinuationBackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram, std::uint32_t concurrency) const {
  if (!concurrency || concurrency > 8)
    throw std::invalid_argument("invalid continuation backing concurrency");
  constexpr std::size_t page = 4096, per_block = 1024;
  const auto payload_budget =
      ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  struct Class {
    std::size_t private_components{}, rows{}, tails{};
  };
  std::map<std::size_t, Class> classes;
  std::size_t private_bytes = 0, row_bytes = 0, tail_bytes = 0;
  for (const auto& component : components_) {
    const auto& d = component.descriptor;
    const auto bytes = d.kind == cache::ComponentKind::kPrivateState
                           ? d.state_bytes
                           : d.row_bytes * d.rows_per_chunk;
    const auto capacity = (bytes + page - 1) / page * page;
    auto& allocation = classes[capacity];
    if (d.kind == cache::ComponentKind::kPrivateState) {
      ++allocation.private_components;
      private_bytes += capacity + per_block;
    } else {
      const auto rows = IsPool(component.part) ? context_ / ratio_ : context_;
      const auto chunks = (rows + d.rows_per_chunk - 1) / d.rows_per_chunk;
      const auto count = std::max(2U, concurrency) * chunks;
      allocation.rows += count;
      ++allocation.tails;
      row_bytes += count * (capacity + per_block);
      tail_bytes += capacity + per_block;
    }
  }
  const auto headers = classes.size() * page;
  if (headers >= payload_budget)
    return {};
  const auto budget = payload_budget - headers;
  // Eight private frontiers cover a cold pass's four grids, stable/prompt
  // copies and subsequent continuation work. At smaller budgets, cache
  // admission still refuses checkpoints that cannot fit their complete plan.
  auto frames = std::min<std::size_t>(8, budget / (private_bytes + tail_bytes));
  if (budget >= row_bytes &&
      (budget - row_bytes) / (private_bytes + tail_bytes) >= frames)
    frames = (budget - row_bytes) / (private_bytes + tail_bytes);
  const auto remaining = budget - frames * private_bytes;
  const auto row_weight = row_bytes + frames * tail_bytes;
  std::vector<ContinuationBackingClass> result;
  for (const auto& [bytes, allocation] : classes) {
    const auto rows = allocation.rows + frames * allocation.tails;
    const auto share =
        row_weight
            ? static_cast<std::size_t>(static_cast<long double>(remaining) *
                                       rows * (bytes + per_block) / row_weight)
            : 0;
    const auto blocks =
        frames * allocation.private_components + share / (bytes + per_block);
    if (blocks)
      result.push_back({bytes, blocks, page + blocks * per_block});
  }
  return result;
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
