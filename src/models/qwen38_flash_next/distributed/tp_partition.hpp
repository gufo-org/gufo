#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/weights.hpp"

namespace gufo::models::qwen38_flash_next::distributed {

/// The routed experts' intermediate dimension split evenly across ranks. Every
/// rank holds units [ff_begin, ff_begin + ff_count) of every expert: those
/// rows of the gate and up projections and those columns of the down
/// projection. Each rank computes every selected expert on its share and the
/// MoE exchange sums the shares, so all ranks route alike and none waits for a
/// peer that happened to receive more experts.
struct TpPartition {
  std::uint32_t rank{0};
  std::uint32_t world_size{1};
  std::uint32_t expert_ff{0};
  std::uint32_t ff_begin{0};
  std::uint32_t ff_count{0};

  [[nodiscard]] bool distributed() const noexcept { return world_size > 1; }

  [[nodiscard]] bool Valid() const noexcept {
    return expert_ff != 0 && world_size != 0 && rank < world_size &&
           expert_ff % world_size == 0 && ff_count == expert_ff / world_size &&
           ff_begin == ff_count * rank;
  }

  /// Builds an equal split. An uneven intermediate size is rejected rather
  /// than giving the ranks different shapes.
  [[nodiscard]] static std::optional<TpPartition> Create(
      std::uint32_t expert_ff, std::uint32_t rank, std::uint32_t world_size,
      std::string* error_msg);
};

struct RoutedWeightPlan {
  std::size_t full_encoded_bytes;
  std::size_t local_encoded_bytes;
};

/// Computes the encoded bytes of the routed tensors and of this rank's share.
/// Fails when a down projection's share does not fall on a quantization-block
/// boundary, since the share is taken without re-encoding any weight. This is
/// a source-artifact estimate; DeviceModel::resident_bytes() remains the
/// authoritative post-conversion device-memory measurement.
[[nodiscard]] std::optional<RoutedWeightPlan> PlanRoutedBytes(
    const ModelWeights& weights, const TpPartition& partition,
    const MtpWeights* mtp_weights = nullptr, std::string* error = nullptr);

}  // namespace gufo::models::qwen38_flash_next::distributed

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DISTRIBUTED_TP_PARTITION_HPP_
