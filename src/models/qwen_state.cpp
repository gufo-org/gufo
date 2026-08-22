#include "src/models/qwen_state.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::models {
namespace {

QwenTensorRef ExtractTensorRef(const core::GgufReader& reader,
                               std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

bool ValidateTensor(const QwenTensorRef& tensor, std::size_t expected_elements,
                    std::string_view name, std::string* error_msg) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required Qwen tensor: " + std::string(name);
    }
    return false;
  }
  // Issue #162: accept quantized tensors during model load.
  if (tensor.type != core::GgmlType::kF32 &&
      tensor.type != core::GgmlType::kBF16 &&
      tensor.type != core::GgmlType::kQ8_K &&
      tensor.type != core::GgmlType::kQ8_0 &&
      tensor.type != core::GgmlType::kQ5_K &&
      tensor.type != core::GgmlType::kQ6_K) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported tensor type for " + std::string(name) + ": " +
                   std::string(core::ToString(tensor.type));
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  return true;
}

}  // namespace

std::optional<QwenModelWeights> QwenModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto config_opt = reader.ExtractModelConfig(error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }

  QwenModelWeights weights;
  weights.config = *config_opt;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
  }
  const std::size_t hidden_size = weights.config.hidden_size;
  const std::size_t vocab_size = weights.config.vocab_size;
  if (!ValidateTensor(weights.token_embd, vocab_size * hidden_size,
                      "token_embd.weight", error_msg) ||
      !ValidateTensor(weights.output_norm, hidden_size, "output_norm.weight",
                      error_msg) ||
      !ValidateTensor(weights.output, vocab_size * hidden_size, "output.weight",
                      error_msg)) {
    return std::nullopt;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    if (l.attn_norm.empty()) {
      l.attn_norm = ExtractTensorRef(reader, prefix + "input_norm.weight");
    }

    l.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (l.ffn_norm.empty()) {
      l.ffn_norm =
          ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    }
    if (l.ffn_norm.empty()) {
      l.ffn_norm = ExtractTensorRef(reader, prefix + "attn_post_norm.weight");
    }

    // Check if full attention layer
    l.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    if (l.attn_q.empty()) {
      l.attn_q = ExtractTensorRef(reader, prefix + "wq.weight");
    }

    if (!l.attn_q.empty()) {
      l.is_full_attention = true;
      l.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      if (l.attn_k.empty()) {
        l.attn_k = ExtractTensorRef(reader, prefix + "wk.weight");
      }
      l.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      if (l.attn_v.empty()) {
        l.attn_v = ExtractTensorRef(reader, prefix + "wv.weight");
      }
      l.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "attn_out.weight");
      }
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "wo.weight");
      }
      l.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      l.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    } else {
      l.is_full_attention = false;
      l.attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
      if (l.attn_qkv.empty()) {
        l.attn_qkv = ExtractTensorRef(reader, prefix + "wqkv.weight");
      }
      l.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
      if (l.attn_gate.empty()) {
        l.attn_gate = ExtractTensorRef(reader, prefix + "wqkv_gate.weight");
      }
      l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
      if (l.ssm_a.empty()) {
        l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a.weight");
      }
      l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
      if (l.ssm_conv1d.empty()) {
        l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d");
      }
      l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.weight");
      }
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt");
      }
      l.ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
      l.ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
      l.ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
      l.ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
    }

    l.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");

    const bool expected_full_attention =
        ((i + 1) % weights.config.full_attention_interval) == 0;
    if (l.is_full_attention != expected_full_attention) {
      if (error_msg != nullptr) {
        *error_msg =
            "Unexpected Qwen layer kind at blk." + std::to_string(i) +
            ": expected " +
            (expected_full_attention ? "full attention" : "Gated DeltaNet");
      }
      return std::nullopt;
    }

    const std::size_t intermediate_size = weights.config.intermediate_size;
    if (!ValidateTensor(l.attn_norm, hidden_size, prefix + "attn_norm.weight",
                        error_msg) ||
        !ValidateTensor(l.ffn_norm, hidden_size,
                        prefix + "post_attention_norm.weight", error_msg) ||
        !ValidateTensor(l.ffn_gate, intermediate_size * hidden_size,
                        prefix + "ffn_gate.weight", error_msg) ||
        !ValidateTensor(l.ffn_up, intermediate_size * hidden_size,
                        prefix + "ffn_up.weight", error_msg) ||
        !ValidateTensor(l.ffn_down, hidden_size * intermediate_size,
                        prefix + "ffn_down.weight", error_msg)) {
      return std::nullopt;
    }

    if (l.is_full_attention) {
      const std::size_t attention_size = weights.config.AttentionSize();
      const std::size_t kv_size =
          static_cast<std::size_t>(weights.config.num_key_value_heads) *
          weights.config.head_dim;
      if (!ValidateTensor(l.attn_q, 2 * attention_size * hidden_size,
                          prefix + "attn_q.weight", error_msg) ||
          !ValidateTensor(l.attn_k, kv_size * hidden_size,
                          prefix + "attn_k.weight", error_msg) ||
          !ValidateTensor(l.attn_v, kv_size * hidden_size,
                          prefix + "attn_v.weight", error_msg) ||
          !ValidateTensor(l.attn_output, hidden_size * attention_size,
                          prefix + "attn_output.weight", error_msg) ||
          !ValidateTensor(l.attn_q_norm, weights.config.head_dim,
                          prefix + "attn_q_norm.weight", error_msg) ||
          !ValidateTensor(l.attn_k_norm, weights.config.head_dim,
                          prefix + "attn_k_norm.weight", error_msg)) {
        return std::nullopt;
      }
    } else {
      const std::size_t qkv_size = weights.config.SsmQkvSize();
      const std::size_t inner_size = weights.config.ssm_inner_size;
      const std::size_t rank = weights.config.ssm_time_step_rank;
      if (!ValidateTensor(l.attn_qkv, qkv_size * hidden_size,
                          prefix + "attn_qkv.weight", error_msg) ||
          !ValidateTensor(l.attn_gate, inner_size * hidden_size,
                          prefix + "attn_gate.weight", error_msg) ||
          !ValidateTensor(l.ssm_a, rank, prefix + "ssm_a", error_msg) ||
          !ValidateTensor(l.ssm_conv1d,
                          qkv_size * weights.config.ssm_conv_kernel,
                          prefix + "ssm_conv1d.weight", error_msg) ||
          !ValidateTensor(l.ssm_dt, rank, prefix + "ssm_dt.bias", error_msg) ||
          !ValidateTensor(l.ssm_alpha, rank * hidden_size,
                          prefix + "ssm_alpha.weight", error_msg) ||
          !ValidateTensor(l.ssm_beta, rank * hidden_size,
                          prefix + "ssm_beta.weight", error_msg) ||
          !ValidateTensor(l.ssm_norm, weights.config.SsmValueSize(),
                          prefix + "ssm_norm.weight", error_msg) ||
          !ValidateTensor(l.ssm_out, hidden_size * inner_size,
                          prefix + "ssm_out.weight", error_msg)) {
        return std::nullopt;
      }
    }
  }

  return weights;
}

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
