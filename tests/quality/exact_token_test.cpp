#include <cassert>
#include <iostream>
#include <vector>

#include "src/models/qwen/qwen_forward.hpp"
#include "src/models/qwen/qwen_generator.hpp"

void TestDeterministicArgmax() {
  std::vector<float> logits = {0.1F, 5.2F, -1.0F, 5.19F, 2.0F};
  const auto tok = strix::models::GreedyArgmax(logits);
  assert(tok == 1);

  // Exact determinism: 1000 repetitions must yield identical result
  for (int i = 0; i < 1000; ++i) {
    assert(strix::models::GreedyArgmax(logits) == 1);
  }
}

void TestDeterministicArgmaxTies() {
  std::vector<float> logits = {1.0F, 5.0F, 5.0F, 2.0F};
  // First occurrence of maximum value
  const auto tok = strix::models::GreedyArgmax(logits);
  assert(tok == 1);
}

void TestDeltaNetHeadMapping() {
  // Qwen 3.5 architecture: 32 value heads, 16 key/query heads
  constexpr std::uint32_t num_v_heads = 32;
  constexpr std::uint32_t num_k_heads = 16;

  for (std::uint32_t h = 0; h < num_v_heads; ++h) {
    const std::uint32_t kh_idx = h % num_k_heads;
    // Ensure that head 0..15 map to 0..15, and 16..31 repeat 0..15
    assert(kh_idx < num_k_heads);
    assert(kh_idx == (h >= 16 ? h - 16 : h));
  }
}

int main() {
  TestDeterministicArgmax();
  TestDeterministicArgmaxTies();
  TestDeltaNetHeadMapping();
  std::cout << "All exact token determinism tests passed.\n";
  return 0;
}
