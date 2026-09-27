#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

#include <limits>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next::distributed {
namespace {

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

/// Bytes of `elements` consecutive values of one row, or zero when the count
/// is not a whole number of quantization blocks.
std::size_t SpanBytes(core::GgmlType type, std::size_t elements) {
  switch (type) {
    case core::GgmlType::kF32:
      return elements * 4;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return elements * 2;
    default:
      return gufo::quant::QuantizedRowBytes(type, elements);
  }
}

bool CheckedMulAdd(std::size_t* total, std::size_t a, std::size_t b,
                   std::size_t c, std::string* error_msg) {
  if (a != 0 &&
      (b > std::numeric_limits<std::size_t>::max() / a ||
       (b != 0 && c > std::numeric_limits<std::size_t>::max() / (a * b)))) {
    SetError(error_msg, "TP routed weight plan overflows");
    return false;
  }
  const std::size_t value = a * b * c;
  if (*total > std::numeric_limits<std::size_t>::max() - value) {
    SetError(error_msg, "TP routed weight plan overflows");
    return false;
  }
  *total += value;
  return true;
}

/// A gate or up projection: `ff_count` of each expert's `expert_ff` rows.
bool AddRowShare(const TensorRef& t, const TpPartition& partition,
                 RoutedWeightPlan* plan, std::string* error_msg) {
  const std::size_t row_bytes = t.RowBytes();
  if (t.empty() || row_bytes == 0 || t.rows != partition.expert_ff) {
    SetError(error_msg, "TP routed tensor " + std::string(t.name) +
                            " does not match the intermediate split");
    return false;
  }
  return CheckedMulAdd(&plan->full_encoded_bytes, row_bytes, t.rows, t.experts,
                       error_msg) &&
         CheckedMulAdd(&plan->local_encoded_bytes, row_bytes,
                       partition.ff_count, t.experts, error_msg);
}

/// A down projection: `ff_count` of each row's `expert_ff` columns.
bool AddColumnShare(const TensorRef& t, const TpPartition& partition,
                    RoutedWeightPlan* plan, std::string* error_msg) {
  const std::size_t row_bytes = t.RowBytes();
  const std::size_t share_bytes = SpanBytes(t.type, partition.ff_count);
  if (t.empty() || row_bytes == 0 || t.cols != partition.expert_ff) {
    SetError(error_msg, "TP routed tensor " + std::string(t.name) +
                            " does not match the intermediate split");
    return false;
  }
  if (share_bytes == 0) {
    SetError(error_msg,
             "TP share of " + std::string(t.name) +
                 " does not fall on a quantization-block boundary: " +
                 std::to_string(partition.ff_count) +
                 " columns must be a multiple of the quantization "
                 "block size (32 for Q8_0 and Q5_1, 256 for Q4_K "
                 "and Q5_K)");
    return false;
  }
  return CheckedMulAdd(&plan->full_encoded_bytes, row_bytes, t.rows, t.experts,
                       error_msg) &&
         CheckedMulAdd(&plan->local_encoded_bytes, share_bytes, t.rows,
                       t.experts, error_msg);
}

bool AddLayerRouted(const LayerWeights& layer, const TpPartition& partition,
                    RoutedWeightPlan* plan, std::string* error_msg) {
  return AddRowShare(layer.ffn_gate_exps, partition, plan, error_msg) &&
         AddRowShare(layer.ffn_up_exps, partition, plan, error_msg) &&
         AddColumnShare(layer.ffn_down_exps, partition, plan, error_msg);
}

}  // namespace

std::optional<RoutedWeightPlan> PlanRoutedBytes(const ModelWeights& weights,
                                                const TpPartition& partition,
                                                const MtpWeights* mtp_weights,
                                                std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (!partition.Valid() || partition.expert_ff != weights.config.expert_ff ||
      (mtp_weights != nullptr &&
       mtp_weights->config.expert_ff != partition.expert_ff)) {
    SetError(error_msg,
             "TP routed weight plan geometry does not match the model");
    return std::nullopt;
  }
  RoutedWeightPlan plan{};
  for (const auto& layer : weights.layers) {
    if (!AddLayerRouted(layer, partition, &plan, error_msg)) {
      return std::nullopt;
    }
  }
  if (mtp_weights != nullptr &&
      !AddLayerRouted(mtp_weights->block, partition, &plan, error_msg)) {
    return std::nullopt;
  }
  return plan;
}

std::optional<TpPartition> TpPartition::Create(std::uint32_t expert_ff,
                                               std::uint32_t rank,
                                               std::uint32_t world_size,
                                               std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  if (expert_ff == 0 || world_size == 0 || rank >= world_size ||
      expert_ff % world_size != 0) {
    SetError(error_msg,
             "TP expert split requires a nonzero, evenly divisible expert "
             "intermediate size and a valid rank");
    return std::nullopt;
  }
  TpPartition result;
  result.rank = rank;
  result.world_size = world_size;
  result.expert_ff = expert_ff;
  result.ff_count = expert_ff / world_size;
  result.ff_begin = result.ff_count * rank;
  return result;
}

}  // namespace gufo::models::qwen38_flash_next::distributed
