#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_MEMORY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_MEMORY_HPP_

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/session_mode.hpp>

#include "src/families/qwen/draft_vocab/draft_vocab.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace gufo::models::qwen38_flash_next {

// One allocation plan, shared by production admission and CPU accounting
// tests. Geometry has already passed Config validation. These are the existing
// private device-state claims; shared logits/scratch are not per-session KV.
struct SessionMemoryBreakdown {
  std::size_t trunk_kv_bytes{};
  std::size_t trunk_index_bytes{};
  std::size_t recurrent_bytes{};
  std::size_t rollback_bytes{};
  std::size_t mtp_kv_bytes{};
  std::size_t mtp_index_bytes{};
  std::size_t mtp_hidden_bytes{};
  std::size_t mtp_draft_table_bytes{};
  std::size_t control_bytes{};
  std::size_t vision_descriptor_bytes{};

  [[nodiscard]] std::size_t TotalBytes() const noexcept {
    return trunk_kv_bytes + trunk_index_bytes + recurrent_bytes +
           rollback_bytes + mtp_kv_bytes + mtp_index_bytes + mtp_hidden_bytes +
           mtp_draft_table_bytes + control_bytes + vision_descriptor_bytes;
  }
};

[[nodiscard]] inline std::uint32_t SessionIndexerCapacity(
    const Config& c, std::uint32_t batch, std::uint32_t context) noexcept {
  return static_cast<std::uint32_t>(std::bit_ceil(std::min<std::uint64_t>(
      context, std::uint64_t{c.indexer_top_k} + batch)));
}

// Recurrent rollback keeps the state after the first token, then exact FP32
// keys/decays/betas/errors for later tokens. Shared with the HIP launchers.
[[nodiscard]] constexpr std::size_t GdnRollbackRowFloats(
    std::uint32_t row, std::uint32_t k_heads, std::uint32_t v_heads,
    std::uint32_t dim) noexcept {
  return row == 0 ? std::size_t{v_heads} * dim * dim
                  : std::size_t{k_heads} * dim + v_heads * (dim + 2);
}

// Copy verification uses segments of eight rows. The neural draft limit is
// independent; optional history-copy acceleration adds further segments.
inline constexpr std::uint32_t kCopySegmentRows = 8;
inline constexpr std::size_t kMutableSnapshotSpareBlocks = 2;

[[nodiscard]] inline std::size_t PlanWideSegmentFloats(
    const Config& c, std::uint32_t rollback_rows) noexcept {
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t full = GdnRollbackRowFloats(
      0, c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_head_dim);
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  std::size_t floats = ple * (rollback_rows + 1);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (!c.IsLinearLayer(il))
      continue;
    floats += conv * (rollback_rows + 1) + full;
    for (std::uint32_t row = 0; row < rollback_rows; ++row)
      floats += GdnRollbackRowFloats(row, c.ssm_num_k_heads, c.ssm_num_v_heads,
                                     c.ssm_head_dim);
  }
  return floats;
}

// Shared by the real executor and host-only width-selection regressions.
// Already committed buffers are excluded; mutable snapshot spare ownership is
// accounted separately by the runtime coordinator after materialization.
[[nodiscard]] inline std::size_t PlanDeferredScratchBytes(
    const Config& c, bool has_mtp, std::uint32_t speculative_rows,
    std::uint32_t max_logit_rows, std::uint32_t copy_rows,
    bool batch_logits_allocated, bool checkpoint_allocated,
    std::size_t allocated_wide_segments, std::uint32_t rollback_rows) noexcept {
  std::size_t bytes = batch_logits_allocated
                          ? 0
                          : std::size_t{8} * std::min(8U, max_logit_rows) *
                                c.vocab_size * sizeof(float);
  const auto linear = c.num_layers - c.num_layers / c.full_attention_interval;
  const auto conv = std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const auto state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const auto ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const auto mutable_bytes = (linear * (conv + state) + ple) * sizeof(float);
  if (!checkpoint_allocated)
    bytes += mutable_bytes + (has_mtp ? std::size_t{speculative_rows} *
                                            c.HcDim() * sizeof(float)
                                      : 0);
  const std::size_t wide_segments = copy_rows / kCopySegmentRows;
  if (wide_segments > 1 + allocated_wide_segments)
    bytes += (wide_segments - 1 - allocated_wide_segments) *
             PlanWideSegmentFloats(c, rollback_rows) * sizeof(float);
  return bytes + kMutableSnapshotSpareBlocks * mutable_bytes;
}

[[nodiscard]] inline SessionMemoryBreakdown PlanSessionMemory(
    const Config& c, core::SessionMode mode, std::uint32_t context,
    std::uint32_t batch_rows, std::uint32_t speculative_rows,
    std::uint32_t copy_rows, std::uint32_t rollback_depth,
    std::size_t control_bytes, std::size_t draft_output_row_bytes) noexcept {
  const std::size_t attention = c.num_layers / c.full_attention_interval;
  const std::size_t linear = c.num_layers - attention;
  const std::size_t conv =
      std::size_t{c.ssm_conv_kernel - 1} * c.SsmConvChannels();
  const std::size_t state =
      std::size_t{c.ssm_num_v_heads} * c.ssm_head_dim * c.ssm_head_dim;
  const std::size_t ple =
      c.ple_layer >= 0 ? std::size_t{c.PleConvHistory()} * c.HcDim() : 0;
  const std::size_t kv =
      2 * std::size_t{context} * c.AttentionKvDim() * sizeof(std::uint16_t);
  const std::size_t index =
      std::size_t{SessionIndexerCapacity(c, batch_rows, context)} *
          c.indexer_head_dim * sizeof(float) +
      std::size_t{context / c.compress_ratio + 1} * c.indexer_head_dim *
          sizeof(std::uint16_t);
  const auto rollback_state =
      rollback_depth == 0
          ? 0
          : state + (rollback_depth - 1) *
                        GdnRollbackRowFloats(1, c.ssm_num_k_heads,
                                             c.ssm_num_v_heads, c.ssm_head_dim);
  SessionMemoryBreakdown result{
      .trunk_kv_bytes = attention * kv,
      .trunk_index_bytes = attention * index,
      .recurrent_bytes = (linear * (conv + state) + ple) * sizeof(float),
      .rollback_bytes =
          ((linear * conv + ple) * rollback_depth + linear * rollback_state) *
          sizeof(float),
      .control_bytes = control_bytes};
  if (mode == core::SessionMode::kSpeculative) {
    result.mtp_kv_bytes = kv;
    result.mtp_index_bytes = index;
    result.mtp_hidden_bytes =
        (std::size_t{std::max(speculative_rows, copy_rows)} + 1) * c.HcDim() *
        sizeof(float);
    if (draft_output_row_bytes != 0)
      result.mtp_draft_table_bytes =
          std::size_t{speculative::kDraftVocabExtraRows} *
              (draft_output_row_bytes + sizeof(std::int32_t)) +
          4096;
  }
  return result;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
