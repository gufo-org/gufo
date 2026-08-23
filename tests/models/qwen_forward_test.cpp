#include "src/models/qwen/qwen_forward.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

void TestEmbeddingLookup() {
  const std::size_t hidden_size = 4;
  const std::vector<float> table = {
      1.0F, 2.0F, 3.0F, 4.0F,  // token 0
      5.0F, 6.0F, 7.0F, 8.0F,  // token 1
  };
  std::vector<float> hidden(hidden_size, 0.0F);

  const strix::models::QwenTensorRef ref = {.data = table.data(),
                                            .type = strix::core::GgmlType::kF32,
                                            .num_elements = table.size()};
  strix::models::ForwardEmbedding(1, ref, hidden_size, hidden);
  assert(hidden[0] == 5.0F);
  assert(hidden[1] == 6.0F);
  assert(hidden[2] == 7.0F);
  assert(hidden[3] == 8.0F);
}

void TestRoPEPreservation() {
  const std::uint32_t num_heads = 2;
  const std::uint32_t num_kv_heads = 1;
  const std::uint32_t head_dim = 4;
  std::vector<float> q = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  std::vector<float> k = {1.0F, 1.0F, 1.0F, 1.0F};

  const float norm_before = std::sqrt(q[0] * q[0] + q[2] * q[2]);
  strix::models::ForwardRoPE(q, k, num_heads, num_kv_heads, head_dim, head_dim,
                             5, 10000.0F);
  const float norm_after = std::sqrt(q[0] * q[0] + q[2] * q[2]);

  assert(std::abs(norm_before - norm_after) < 1e-5F);
}

void TestSwiGLUFFN() {
  const std::size_t hidden_size = 2;
  const std::size_t intermediate_size = 2;
  const std::vector<float> x = {1.0F, 2.0F};
  const std::vector<float> gate_w = {1.0F, 0.0F, 0.0F, 1.0F};  // identity
  const std::vector<float> up_w = {1.0F, 0.0F, 0.0F, 1.0F};    // identity
  const std::vector<float> down_w = {1.0F, 0.0F, 0.0F, 1.0F};  // identity

  const strix::models::QwenTensorRef gate_ref = {
      .data = gate_w.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = gate_w.size()};
  const strix::models::QwenTensorRef up_ref = {
      .data = up_w.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = up_w.size()};
  const strix::models::QwenTensorRef down_ref = {
      .data = down_w.data(),
      .type = strix::core::GgmlType::kF32,
      .num_elements = down_w.size()};

  std::vector<float> gate_sc(intermediate_size);
  std::vector<float> up_sc(intermediate_size);
  std::vector<float> act_sc(intermediate_size);
  std::vector<float> ffn_out(hidden_size);

  strix::models::ForwardFFN(x, gate_ref, up_ref, down_ref, hidden_size,
                            intermediate_size, gate_sc, up_sc, act_sc, ffn_out);

  // gate = [1.0, 2.0], up = [1.0, 2.0]
  // silu(1.0) * 1.0 = (1 / (1 + exp(-1))) * 1.0 ~= 0.731058
  assert(ffn_out[0] > 0.73F && ffn_out[0] < 0.74F);
}

void TestGreedyArgmax() {
  const std::vector<float> logits = {0.1F, -2.5F, 14.8F, 3.2F, 12.0F};
  const auto best = strix::models::GreedyArgmax(logits);
  assert(best == 2);
}

void TestAttentionWithKvCache() {
  const std::uint32_t num_heads = 2;
  const std::uint32_t num_kv_heads = 2;
  const std::uint32_t head_dim = 4;
  strix::models::QwenKvCache kv_cache(1, num_kv_heads, 16, head_dim);

  std::vector<float> q = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  std::vector<float> k = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  std::vector<float> v = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(8, 0.0F);

  strix::models::ForwardAttention(q, k, v, {}, {}, kv_cache, 0, 0, num_heads,
                                  num_kv_heads, head_dim, 8, scores, out);

  // Since pos=0, softmax weight is 1.0, out should equal value vectors
  assert(std::abs(out[0] - 2.0F) < 1e-5F);
  assert(std::abs(out[1] - 3.0F) < 1e-5F);
  assert(std::abs(out[4] - 6.0F) < 1e-5F);
}

int main() {
  TestEmbeddingLookup();
  TestRoPEPreservation();
  TestSwiGLUFFN();
  TestGreedyArgmax();
  TestAttentionWithKvCache();
  std::cout << "All Qwen forward tests passed.\n";
  return 0;
}
