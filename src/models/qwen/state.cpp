#include "src/models/qwen/state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strix::models {

QwenKvCache::QwenKvCache(std::uint32_t num_layers, std::uint32_t num_kv_heads,
                         std::uint32_t max_context, std::uint32_t head_dim)
    : num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      max_context_(std::min(max_context, 8192U)),
      head_dim_(head_dim) {
  const std::size_t total_elements = static_cast<std::size_t>(num_layers_) *
                                     num_kv_heads_ * max_context_ * head_dim_;
  k_data_.resize(total_elements, 0.0F);
  v_data_.resize(total_elements, 0.0F);
}

std::span<float> QwenKvCache::GetKeySlice(std::uint32_t layer,
                                          std::uint32_t kv_head,
                                          std::uint32_t pos) noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&k_data_[offset], head_dim_};
}

std::span<const float> QwenKvCache::GetKeySlice(
    std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t pos) const noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&k_data_[offset], head_dim_};
}

std::span<float> QwenKvCache::GetValueSlice(std::uint32_t layer,
                                            std::uint32_t kv_head,
                                            std::uint32_t pos) noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&v_data_[offset], head_dim_};
}

std::span<const float> QwenKvCache::GetValueSlice(
    std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t pos) const noexcept {
  const std::size_t offset =
      (((static_cast<std::size_t>(layer) * num_kv_heads_ + kv_head) *
            max_context_ +
        pos) *
       head_dim_);
  return {&v_data_[offset], head_dim_};
}

QwenScratchArena::QwenScratchArena(const core::ModelConfig& config) {
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t q_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t max_context =
      std::min(config.context_length > 0 ? config.context_length : 4096, 8192U);
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;

  const std::size_t ssm_qkv_size =
      std::max<std::size_t>(config.SsmQkvSize(), 2 * q_size);
  const std::size_t ssm_gate_size =
      std::max<std::size_t>(config.ssm_inner_size, q_size);

  const std::size_t total_size = (hidden_size * 4) + q_size + (kv_size * 2) +
                                 max_context + (intermediate_size * 3) +
                                 ssm_qkv_size + ssm_gate_size +
                                 config.ssm_inner_size + vocab_size;
  buffer.resize(total_size, 0.0F);

  std::size_t cur = 0;
  auto alloc_span = [&](std::size_t sz) {
    auto sp = std::span<float>(&buffer[cur], sz);
    cur += sz;
    return sp;
  };

  hidden = alloc_span(hidden_size);
  normed = alloc_span(hidden_size);
  q = alloc_span(q_size);
  k = alloc_span(kv_size);
  v = alloc_span(kv_size);
  attn_scores = alloc_span(max_context);
  attn_out = alloc_span(hidden_size);
  mlp_gate = alloc_span(intermediate_size);
  mlp_up = alloc_span(intermediate_size);
  mlp_act = alloc_span(intermediate_size);
  mlp_out = alloc_span(hidden_size);
  ssm_qkv = alloc_span(ssm_qkv_size);
  ssm_gate = alloc_span(ssm_gate_size);
  ssm_out_buf = alloc_span(config.ssm_inner_size);
  logits = alloc_span(vocab_size);
}


}  // namespace strix::models
