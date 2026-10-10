#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXPERT_STREAM_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXPERT_STREAM_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next {

/// Parses a byte budget: a plain number of bytes, or a number with a
/// `B`/`K`/`M`/`G` (optionally `iB`-style, e.g. `KiB`, `MiB`, `GiB`)
/// suffix, case-insensitive. Binary multipliers only; surrounding
/// whitespace is accepted. Returns nullopt for empty, malformed, negative
/// or overflowing input.
[[nodiscard]] std::optional<std::size_t> ParseByteSize(
    std::string_view text) noexcept;

/// Result of sizing the routed-expert streaming cache (issue #427) for an
/// artifact. The cache is a single size class: every streamed layer holds
/// `slots_per_layer` experts in identical per-expert byte sizes
/// (`expert_bytes` = gate + up + down). Layers whose routed-expert shape
/// differs (mixed-quant outliers, an MTP block in another class) stay
/// fully resident — the same uniform-class rule the slab layout enforces.
struct ExpertStreamPlan {
  /// 0 when streaming is off or impossible; otherwise the per-expert byte
  /// size of the streamed class.
  std::size_t expert_bytes{0};
  /// Layers (trunk plus a same-class MTP block) that stream.
  std::size_t layers{0};
  /// Resident expert slots per streamed layer derived from the budget.
  std::uint32_t slots_per_layer{0};
  /// Total routed experts per layer in the artifact (512 on Flash-Next).
  /// A slot count this high makes streaming pointless: the slabs would be
  /// as large as full residency.
  std::uint32_t num_experts{0};
  /// Bytes of the routed experts left fully resident (the non-streamed
  /// class), for load-time reporting.
  std::size_t resident_expert_bytes{0};

  [[nodiscard]] bool enabled() const noexcept {
    return expert_bytes != 0 && layers != 0 && slots_per_layer != 0;
  }
};

/// Chooses the streaming plan for the trunk layers (plus `mtp`, may be
/// null). The streamed class is the one covering the most layers among the
/// formats the streaming kernels support; `budget_bytes` is split evenly
/// across those layers. Returns a disabled plan when the budget is 0;
/// sets `error` (and leaves a disabled plan) when the budget cannot hold
/// even one token's routed experts, covers every expert outright (full
/// residency is then the cheaper plan), or no layer can stream.
[[nodiscard]] ExpertStreamPlan PlanExpertStreaming(
    const std::vector<LayerWeights>& layers, const LayerWeights* mtp,
    std::uint32_t num_experts, std::uint32_t num_experts_used,
    std::size_t budget_bytes, std::string* error = nullptr);

/// True when a layer's routed-expert formats are supported by the
/// streaming kernels (vector expert routes for Q4_K/Q5_K/Q8_0 gate+up and
/// Q5_1/Q8_0 down).
[[nodiscard]] bool ExpertStreamSupported(const LayerWeights& layer) noexcept;

/// Combined encoded bytes of one expert (gate + up + down).
[[nodiscard]] std::size_t ExpertBytes(const LayerWeights& layer) noexcept;

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_EXPERT_STREAM_HPP_
