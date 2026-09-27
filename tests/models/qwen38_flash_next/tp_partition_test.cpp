#include "src/models/qwen38_flash_next/distributed/tp_partition.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace {

using gufo::core::GgmlType;
using gufo::models::qwen38_flash_next::ModelWeights;
using gufo::models::qwen38_flash_next::TensorRef;
using gufo::models::qwen38_flash_next::distributed::PlanRoutedBytes;
using gufo::models::qwen38_flash_next::distributed::TpPartition;

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

void RequireError(const std::optional<TpPartition>& result,
                  const char* message) {
  Require(!result.has_value(), message);
}

ModelWeights OneLayer(TensorRef gate, TensorRef up, TensorRef down) {
  ModelWeights weights;
  weights.config.expert_ff = 640;
  weights.layers.resize(1);
  weights.layers[0].ffn_gate_exps = gate;
  weights.layers[0].ffn_up_exps = up;
  weights.layers[0].ffn_down_exps = down;
  return weights;
}

}  // namespace

int main() {
  std::string error;
  const auto single = TpPartition::Create(640, 0, 1, &error);
  Require(single.has_value(), error.c_str());
  Require(!single->distributed() && single->ff_begin == 0 &&
              single->ff_count == 640,
          "one rank holds every intermediate unit");

  const auto first = TpPartition::Create(640, 0, 2, &error);
  const auto second = TpPartition::Create(640, 1, 2, &error);
  Require(first.has_value() && second.has_value(), error.c_str());
  Require(first->ff_begin == 0 && first->ff_count == 320,
          "rank zero holds the first half of every expert");
  Require(second->ff_begin == 320 && second->ff_count == 320,
          "rank one holds the second half of every expert");

  // The model's shapes: 2560-wide Q4_K gate/up rows, 640 per expert; down
  // rows 640 wide, in Q5_1 or Q8_0 (32-value blocks), 2560 per expert.
  const void* data = reinterpret_cast<const void*>(0x1000);
  const TensorRef gate{data, GgmlType::kQ4_K, 2560, 640, 4, 0, 0, "gate"};
  const TensorRef down_q5{data, GgmlType::kQ5_1, 640, 2560, 4, 0, 0, "down"};
  const TensorRef down_q8{data, GgmlType::kQ8_0, 640, 2560, 4, 0, 0, "down"};
  const std::size_t gate_row = 10 * 144;  // ten Q4_K super-blocks
  for (const auto& [down, block_bytes] :
       {std::pair{down_q5, std::size_t{24}},
        std::pair{down_q8, std::size_t{34}}}) {
    const auto plan =
        PlanRoutedBytes(OneLayer(gate, gate, down), *second, nullptr, &error);
    Require(plan.has_value(), error.c_str());
    const std::size_t full =
        2 * gate_row * 640 * 4 + block_bytes * 20 * 2560 * 4;
    Require(plan->full_encoded_bytes == full, "full routed bytes");
    Require(plan->local_encoded_bytes ==
                2 * gate_row * 320 * 4 + block_bytes * 10 * 2560 * 4,
            "a rank's share is half of every routed tensor");
  }

  // A down share must be whole quantization blocks: 16 of 32 values is not.
  const TensorRef narrow_down{data, GgmlType::kQ8_0, 32, 2560, 4, 0, 0, "down"};
  const TensorRef narrow_gate{data, GgmlType::kQ4_K, 2560, 32, 4, 0, 0, "gate"};
  auto narrow = OneLayer(narrow_gate, narrow_gate, narrow_down);
  narrow.config.expert_ff = 32;
  const auto narrow_split = TpPartition::Create(32, 0, 2, &error);
  Require(narrow_split.has_value(), error.c_str());
  Require(!PlanRoutedBytes(narrow, *narrow_split, nullptr, &error).has_value(),
          "reject a share that splits a quantization block");

  Require(
      !PlanRoutedBytes(OneLayer(gate, gate, down_q8),
                       *TpPartition::Create(320, 0, 2, &error), nullptr, &error)
           .has_value(),
      "reject a split that does not match the model");
  auto invalid = *first;
  invalid.ff_begin = 1;
  Require(!invalid.Valid(), "reject a misplaced share");
  RequireError(TpPartition::Create(641, 0, 2, &error),
               "reject an uneven intermediate size");
  RequireError(TpPartition::Create(0, 0, 1, &error),
               "reject an empty intermediate size");
  RequireError(TpPartition::Create(640, 2, 2, &error),
               "reject an out-of-range rank");

  std::puts("PASS: TP=2 intermediate split and Q4_K/Q5_1/Q8_0 shares");
  return 0;
}
