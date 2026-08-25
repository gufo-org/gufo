#include "src/models/qwen/dflash_reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/forward.hpp"

namespace strix::speculative {
namespace {

models::QwenTensorRef TensorRef(const core::GgufReader& reader,
                                std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount()};
}

void DequantizeOrCopyRow(const models::QwenTensorRef& tensor, std::size_t row,
                         std::size_t columns, float* output) {
  if (tensor.type == core::GgmlType::kF32) {
    const auto* source =
        static_cast<const float*>(tensor.data) + (row * columns);
    std::copy_n(source, columns, output);
    return;
  }
  if (tensor.type == core::GgmlType::kBF16) {
    const std::size_t offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto bits =
          static_cast<const std::uint16_t*>(tensor.data)[offset + column];
      output[column] =
          std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return;
  }
  const std::size_t row_bytes = quant::QuantizedRowBytes(tensor.type, columns);
  const void* source =
      static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
  switch (tensor.type) {
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(source, output, columns);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(source, output, columns);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(source, output, columns);
      return;
    default:
      throw std::runtime_error("unsupported tensor format for row dequant");
  }
}

void MatMulVector(const models::QwenTensorRef& weight,
                  std::span<const float> input, std::span<float> output,
                  std::size_t rows, std::size_t cols,
                  const models::QwenTensorRef& scale = {}) {
  std::vector<float> row_buf(cols);
  const float s = scale.empty() ? 1.0F : scale.Get(0);
  for (std::size_t r = 0; r < rows; ++r) {
    DequantizeOrCopyRow(weight, r, cols, row_buf.data());
    double dot = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      dot += static_cast<double>(row_buf[c]) * static_cast<double>(input[c]);
    }
    output[r] = static_cast<float>(dot * static_cast<double>(s));
  }
}

void SiLU(std::span<float> values) {
  for (float& val : values) {
    val = val / (1.0F + std::exp(-val));
  }
}

}  // namespace

std::optional<QwenDFlashWeights> QwenDFlashWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  core::ModelConfig config;
  auto arch_str = reader.GetMetadataString("general.architecture").value_or("qwen2");
  config.architecture = std::string(arch_str);
  const std::string prefix = config.architecture + ".";

  config.hidden_size =
      reader.GetMetadataUint32(prefix + "embedding_length").value_or(0U);
  if (config.hidden_size == 0) {
    config.hidden_size =
        reader.GetMetadataUint32("embedding_length").value_or(4096U);
  }
  config.intermediate_size =
      reader.GetMetadataUint32(prefix + "feed_forward_length").value_or(0U);
  if (config.intermediate_size == 0) {
    config.intermediate_size =
        reader.GetMetadataUint32("feed_forward_length").value_or(11008U);
  }
  config.num_attention_heads =
      reader.GetMetadataUint32(prefix + "attention.head_count").value_or(0U);
  if (config.num_attention_heads == 0) {
    config.num_attention_heads =
        reader.GetMetadataUint32("attention.head_count").value_or(32U);
  }
  config.num_key_value_heads =
      reader.GetMetadataUint32(prefix + "attention.head_count_kv")
          .value_or(config.num_attention_heads);
  config.head_dim =
      reader.GetMetadataUint32(prefix + "attention.key_length").value_or(128U);
  if (config.head_dim == 0 && config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }
  config.rotary_dim =
      reader.GetMetadataUint32(prefix + "rope.dimension_count")
          .value_or(config.head_dim);
  config.rope_theta =
      reader.GetMetadataFloat32(prefix + "rope.freq_base").value_or(1000000.0F);
  config.context_length =
      reader.GetMetadataUint32(prefix + "context_length").value_or(32768U);
  config.num_layers =
      reader.GetMetadataUint32(prefix + "block_count").value_or(5U);

  // Vocab size
  if (const auto* token_embd = reader.FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0) {
    const auto elements = token_embd->ElementCount();
    if ((elements % config.hidden_size) == 0) {
      config.vocab_size =
          static_cast<std::uint32_t>(elements / config.hidden_size);
    }
  }
  if (config.vocab_size == 0) {
    config.vocab_size =
        reader.GetMetadataUint32(prefix + "vocab_size").value_or(152064U);
  }

  QwenDFlashWeights weights;
  weights.config = config;
  auto& df_cfg = weights.dflash_config;

  // Read DFlash metadata from GGUF
  df_cfg.block_size = reader.GetMetadataUint32("dflash.block_size").value_or(16U);
  df_cfg.sample_from_anchor =
      reader.GetMetadataBool("dflash.sample_from_anchor").value_or(true);
  df_cfg.mask_token_id =
      reader.GetMetadataUint32("mask_token_id").value_or(0U);

  // Target layer indices
  if (const auto* val = reader.FindMetadata("target_layers")) {
    if (std::holds_alternative<std::vector<std::uint64_t>>(val->value)) {
      const auto& vec = std::get<std::vector<std::uint64_t>>(val->value);
      df_cfg.target_layer_ids.assign(vec.begin(), vec.end());
    }
  }
  if (df_cfg.target_layer_ids.empty()) {
    // Default uniform 5-layer sampling if not explicitly provided
    df_cfg.target_layer_ids = {1, 7, 13, 19, 25};
  }

  // Base / tied tensors
  weights.token_embedding = TensorRef(reader, "token_embd.weight");
  weights.output = TensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    weights.output = weights.token_embedding;
  }
  weights.output_norm = TensorRef(reader, "output_norm.weight");
  weights.d2t = TensorRef(reader, "d2t");
  df_cfg.draft_vocab_size =
      weights.d2t.empty() ? weights.config.vocab_size : weights.d2t.num_elements;

  // Feature fusion encoder
  weights.fc_projection = TensorRef(reader, "fc.weight");
  weights.fc_scale = TensorRef(reader, "fc.scale");
  weights.fc_norm = TensorRef(reader, "output_norm_enc.weight");
  if (weights.fc_norm.empty()) {
    weights.fc_norm = TensorRef(reader, "enc.output_norm.weight");
  }

  // Check for DFlash-2 dynamic depthwise convolutions
  weights.in_conv_weight = TensorRef(reader, "in_conv.weight");
  weights.in_conv_bias = TensorRef(reader, "in_conv.bias");
  weights.out_conv_weight = TensorRef(reader, "out_conv.weight");
  weights.out_conv_bias = TensorRef(reader, "out_conv.bias");
  df_cfg.has_dynamic_conv = !weights.in_conv_weight.empty();

  // Check for DFlash-2 / DSpark Markov candidate path selector
  weights.markov_w1 = TensorRef(reader, "markov_w1.weight");
  if (weights.markov_w1.empty()) {
    weights.markov_w1 = TensorRef(reader, "selector_predecessor.weight");
  }
  weights.markov_w2 = TensorRef(reader, "markov_w2.weight");
  if (weights.markov_w2.empty()) {
    weights.markov_w2 = TensorRef(reader, "selector_successor.weight");
  }
  weights.conf_proj = TensorRef(reader, "conf_proj.weight");
  if (weights.conf_proj.empty()) {
    weights.conf_proj = TensorRef(reader, "selector_hidden.weight");
  }
  weights.conf_bias = TensorRef(reader, "conf_proj.bias");
  df_cfg.has_path_selector =
      !weights.markov_w1.empty() && !weights.markov_w2.empty();
  if (df_cfg.has_path_selector) {
    df_cfg.markov_rank = 256;
  }

  // Load draft decoder transformer layers (e.g. blk.0, blk.1, ...)
  df_cfg.num_layers = weights.config.num_layers;
  weights.layers.resize(df_cfg.num_layers);
  for (std::uint32_t i = 0; i < df_cfg.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    layer.attn_norm = TensorRef(reader, prefix + "attn_norm.weight");
    layer.attn_q = TensorRef(reader, prefix + "attn_q.weight");
    layer.attn_k = TensorRef(reader, prefix + "attn_k.weight");
    layer.attn_v = TensorRef(reader, prefix + "attn_v.weight");
    layer.attn_output = TensorRef(reader, prefix + "attn_output.weight");
    if (layer.attn_output.empty()) {
      layer.attn_output = TensorRef(reader, prefix + "attn_out.weight");
    }
    layer.attn_q_norm = TensorRef(reader, prefix + "attn_q_norm.weight");
    layer.attn_k_norm = TensorRef(reader, prefix + "attn_k_norm.weight");
    layer.ffn_norm = TensorRef(reader, prefix + "ffn_norm.weight");
    layer.ffn_gate = TensorRef(reader, prefix + "ffn_gate.weight");
    layer.ffn_up = TensorRef(reader, prefix + "ffn_up.weight");
    layer.ffn_down = TensorRef(reader, prefix + "ffn_down.weight");
  }

  // Deduce actual dimensions from loaded layer tensors
  if (!weights.layers.empty()) {
    const auto& l0 = weights.layers[0];
    if (!l0.attn_q.empty() && weights.config.hidden_size > 0) {
      const std::size_t q_dim =
          l0.attn_q.num_elements / weights.config.hidden_size;
      weights.config.head_dim =
          l0.attn_q_norm.empty()
              ? 128U
              : static_cast<std::uint32_t>(l0.attn_q_norm.num_elements);
      if (weights.config.head_dim > 0) {
        weights.config.num_attention_heads =
            static_cast<std::uint32_t>(q_dim / weights.config.head_dim);
      }
    }
    if (!l0.attn_k.empty() && weights.config.hidden_size > 0 &&
        weights.config.head_dim > 0) {
      const std::size_t kv_dim =
          l0.attn_k.num_elements / weights.config.hidden_size;
      weights.config.num_key_value_heads =
          static_cast<std::uint32_t>(kv_dim / weights.config.head_dim);
    }
    if (!l0.ffn_gate.empty() && weights.config.hidden_size > 0) {
      weights.config.intermediate_size = static_cast<std::uint32_t>(
          l0.ffn_gate.num_elements / weights.config.hidden_size);
    }
  }

  return weights;
}

QwenDFlashReference::QwenDFlashReference(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const core::GgufReader> tied_reader,
    QwenDFlashWeights weights, std::uint32_t max_context)
    : reader_(std::move(reader)),
      tied_reader_(std::move(tied_reader)),
      weights_(std::move(weights)),
      max_context_(max_context) {
  const std::size_t num_layers = weights_.layers.size();
  const std::size_t kv_dim =
      static_cast<std::size_t>(weights_.config.num_key_value_heads) *
      weights_.config.head_dim;
  injected_k_.resize(num_layers);
  injected_v_.resize(num_layers);
  for (std::size_t i = 0; i < num_layers; ++i) {
    injected_k_[i].resize(max_context_ * kv_dim, 0.0F);
    injected_v_[i].resize(max_context_ * kv_dim, 0.0F);
  }
}

std::unique_ptr<QwenDFlashReference> QwenDFlashReference::Create(
    std::shared_ptr<const core::GgufReader> reader, std::uint32_t max_context,
    std::string* error_msg) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "null reader provided";
    }
    return nullptr;
  }
  auto weights = QwenDFlashWeights::LoadFromGguf(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  return std::unique_ptr<QwenDFlashReference>(
      new QwenDFlashReference(reader, nullptr, std::move(*weights), max_context));
}

std::unique_ptr<QwenDFlashReference> QwenDFlashReference::CreateWithTiedWeights(
    std::shared_ptr<const core::GgufReader> reader,
    std::shared_ptr<const core::GgufReader> tied_reader,
    std::uint32_t max_context, std::string* error_msg) {
  if (reader == nullptr) {
    if (error_msg != nullptr) {
      *error_msg = "null reader provided";
    }
    return nullptr;
  }
  auto weights = QwenDFlashWeights::LoadFromGguf(*reader, error_msg);
  if (!weights.has_value()) {
    return nullptr;
  }
  if (tied_reader != nullptr) {
    if (weights->token_embedding.empty()) {
      weights->token_embedding = TensorRef(*tied_reader, "token_embd.weight");
    }
    if (weights->output.empty()) {
      weights->output = TensorRef(*tied_reader, "output.weight");
      if (weights->output.empty()) {
        weights->output = weights->token_embedding;
      }
    }
  }
  return std::unique_ptr<QwenDFlashReference>(new QwenDFlashReference(
      reader, tied_reader, std::move(*weights), max_context));
}

void QwenDFlashReference::Reset() noexcept {
  injected_context_len_ = 0;
  for (auto& k : injected_k_) {
    std::ranges::fill(k, 0.0F);
  }
  for (auto& v : injected_v_) {
    std::ranges::fill(v, 0.0F);
  }
}

bool QwenDFlashReference::InjectTargetContext(
    std::span<const float> target_features, std::uint32_t position,
    std::uint32_t num_tokens) {
  const std::size_t hidden_size = weights_.config.hidden_size;
  const std::size_t num_target_layers =
      weights_.dflash_config.target_layer_ids.size();
  const std::size_t enc_in_dim = num_target_layers * hidden_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(weights_.config.num_key_value_heads) *
      weights_.config.head_dim;

  if (target_features.size() != num_tokens * enc_in_dim) {
    return false;
  }
  if (position + num_tokens > max_context_) {
    return false;
  }

  std::vector<float> fused(hidden_size);
  std::vector<float> normed_fused(hidden_size);
  std::vector<float> k_proj(kv_dim);
  std::vector<float> v_proj(kv_dim);

  for (std::size_t t = 0; t < num_tokens; ++t) {
    const auto token_features =
        target_features.subspan(t * enc_in_dim, enc_in_dim);
    const std::uint32_t token_pos = position + static_cast<std::uint32_t>(t);

    // FC Projection & Norm
    MatMulVector(weights_.fc_projection, token_features, fused, hidden_size,
                 enc_in_dim, weights_.fc_scale);
    models::ForwardRMSNorm(fused, weights_.fc_norm, 1e-6F, normed_fused);

    for (std::size_t layer_idx = 0; layer_idx < weights_.layers.size();
         ++layer_idx) {
      const auto& layer = weights_.layers[layer_idx];

      // K and V projection
      MatMulVector(layer.attn_k, normed_fused, k_proj, kv_dim, hidden_size);
      MatMulVector(layer.attn_v, normed_fused, v_proj, kv_dim, hidden_size);

      // K RMSNorm
      if (!layer.attn_k_norm.empty()) {
        models::ForwardRMSNorm(k_proj, layer.attn_k_norm, 1e-6F, k_proj);
      }

      // Apply RoPE to K
      const std::size_t head_dim = weights_.config.head_dim;
      const std::size_t num_kv_heads = weights_.config.num_key_value_heads;
      models::ForwardRoPE(k_proj, k_proj,
                          static_cast<std::uint32_t>(num_kv_heads),
                          static_cast<std::uint32_t>(num_kv_heads),
                          static_cast<std::uint32_t>(head_dim),
                          static_cast<std::uint32_t>(head_dim),
                          token_pos, weights_.config.rope_theta);

      // Store into injected KV cache
      const std::size_t cache_offset = token_pos * kv_dim;
      std::copy_n(k_proj.begin(), kv_dim,
                  injected_k_[layer_idx].begin() + cache_offset);
      std::copy_n(v_proj.begin(), kv_dim,
                  injected_v_[layer_idx].begin() + cache_offset);
    }
  }

  injected_context_len_ =
      std::max(injected_context_len_, position + num_tokens);
  return true;
}

void QwenDFlashReference::ApplyDynamicConv2Tap(
    std::span<float> sequence_hidden, std::size_t num_tokens,
    std::size_t hidden_size, const models::QwenTensorRef& weight,
    const models::QwenTensorRef& bias) {
  if (weight.empty() || num_tokens == 0) {
    return;
  }
  std::vector<float> temp(sequence_hidden.begin(), sequence_hidden.end());
  for (std::size_t t = 0; t < num_tokens; ++t) {
    for (std::size_t c = 0; c < hidden_size; ++c) {
      const float w0 = weight.Get(c * 2 + 0);
      const float w1 = weight.Get(c * 2 + 1);
      const float b = bias.empty() ? 0.0F : bias.Get(c);

      const float x_curr = temp[t * hidden_size + c];
      const float x_prev = (t > 0) ? temp[(t - 1) * hidden_size + c] : x_curr;

      sequence_hidden[t * hidden_size + c] = (w0 * x_curr) + (w1 * x_prev) + b;
    }
  }
}

void QwenDFlashReference::ApplyMarkovPathScoring(
    std::span<float> block_logits, std::size_t num_tokens,
    std::size_t vocab_size, std::size_t draft_vocab_size,
    const models::QwenTensorRef& w1, const models::QwenTensorRef& w2,
    const models::QwenTensorRef& d2t, tokenization::TokenId anchor_token,
    bool sample_from_anchor, std::vector<tokenization::TokenId>& out_tokens) {
  out_tokens.clear();
  out_tokens.reserve(num_tokens);

  const std::size_t markov_rank = 256;
  const std::size_t selector_vocab =
      (w1.empty() || markov_rank == 0) ? 0 : (w1.num_elements / markov_rank);

  std::vector<float> w1_row(markov_rank, 0.0F);
  const std::size_t bias_len =
      std::min(draft_vocab_size, selector_vocab > 0 ? selector_vocab : vocab_size);
  std::vector<float> bias(bias_len, 0.0F);

  tokenization::TokenId prev_token = anchor_token;
  const std::size_t start_idx = sample_from_anchor ? 0 : 1;

  if (!sample_from_anchor && num_tokens > 0) {
    out_tokens.push_back(anchor_token);
  }

  for (std::size_t i = start_idx; i < num_tokens; ++i) {
    auto logits_row =
        block_logits.subspan(i * vocab_size, vocab_size);

    if (selector_vocab > 0 && prev_token < selector_vocab && !w1.empty() &&
        !w2.empty()) {
      DequantizeOrCopyRow(w1, prev_token, markov_rank, w1_row.data());
      MatMulVector(w2, w1_row, bias, bias_len, markov_rank);

      if (!d2t.empty()) {
        for (std::size_t d = 0;
             d < std::min(draft_vocab_size, d2t.num_elements); ++d) {
          const auto target_id = static_cast<std::size_t>(d2t.Get(d));
          if (target_id < vocab_size && d < bias_len) {
            logits_row[target_id] += bias[d];
          }
        }
      } else {
        for (std::size_t v = 0; v < std::min(vocab_size, bias_len); ++v) {
          logits_row[v] += bias[v];
        }
      }
    }

    // Greedy pick for step i
    float max_logit = -std::numeric_limits<float>::infinity();
    tokenization::TokenId best_token = 0;
    for (std::size_t v = 0; v < vocab_size; ++v) {
      if (logits_row[v] > max_logit) {
        max_logit = logits_row[v];
        best_token = static_cast<tokenization::TokenId>(v);
      }
    }

    out_tokens.push_back(best_token);
    prev_token = best_token;
  }
}

void QwenDFlashReference::ComputeBlockLogits(tokenization::TokenId anchor_token,
                                            std::uint32_t current_pos,
                                            std::uint32_t draft_count,
                                            std::vector<float>& out_logits) {
  const std::size_t hidden_size = weights_.config.hidden_size;
  const std::size_t intermediate_size = weights_.config.intermediate_size;
  const std::size_t head_dim = weights_.config.head_dim;
  const std::size_t num_q_heads = weights_.config.num_attention_heads;
  const std::size_t num_kv_heads = weights_.config.num_key_value_heads;
  const std::size_t q_dim = num_q_heads * head_dim;
  const std::size_t kv_dim = num_kv_heads * head_dim;
  const std::size_t vocab_size = weights_.config.vocab_size;
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> block_hidden(draft_count * hidden_size, 0.0F);

  // Embed noise tokens: position 0 is anchor, positions 1..draft_count-1 are mask tokens
  for (std::size_t t = 0; t < draft_count; ++t) {
    const tokenization::TokenId tok =
        (t == 0) ? anchor_token : weights_.dflash_config.mask_token_id;
    DequantizeOrCopyRow(weights_.token_embedding, tok, hidden_size,
                        block_hidden.data() + (t * hidden_size));
  }

  // DFlash-2 Input Dynamic Conv
  if (weights_.dflash_config.has_dynamic_conv) {
    ApplyDynamicConv2Tap(block_hidden, draft_count, hidden_size,
                         weights_.in_conv_weight, weights_.in_conv_bias);
  }

  // Decoder transformer layers
  std::vector<float> q_all(draft_count * q_dim);
  std::vector<float> k_block(draft_count * kv_dim);
  std::vector<float> v_block(draft_count * kv_dim);
  std::vector<float> attn_out(draft_count * hidden_size);
  std::vector<float> ffn_gate(draft_count * intermediate_size);
  std::vector<float> ffn_up(draft_count * intermediate_size);
  std::vector<float> ffn_down(draft_count * hidden_size);

  const std::size_t total_k_len = current_pos + draft_count;

  for (std::size_t layer_idx = 0; layer_idx < weights_.layers.size();
       ++layer_idx) {
    const auto& layer = weights_.layers[layer_idx];

    // Attention pre-norm & Q/K/V projections
    for (std::size_t t = 0; t < draft_count; ++t) {
      const std::uint32_t pos = current_pos + static_cast<std::uint32_t>(t);
      auto tok_hidden = std::span<float>(block_hidden.data() + (t * hidden_size), hidden_size);
      std::vector<float> normed(hidden_size);
      models::ForwardRMSNorm(tok_hidden, layer.attn_norm, 1e-6F, normed);

      auto q_t = std::span<float>(q_all.data() + (t * q_dim), q_dim);
      auto k_t = std::span<float>(k_block.data() + (t * kv_dim), kv_dim);
      auto v_t = std::span<float>(v_block.data() + (t * kv_dim), kv_dim);

      MatMulVector(layer.attn_q, normed, q_t, q_dim, hidden_size);
      MatMulVector(layer.attn_k, normed, k_t, kv_dim, hidden_size);
      MatMulVector(layer.attn_v, normed, v_t, kv_dim, hidden_size);

      if (!layer.attn_q_norm.empty()) {
        models::ForwardRMSNorm(q_t, layer.attn_q_norm, 1e-6F, q_t);
      }
      if (!layer.attn_k_norm.empty()) {
        models::ForwardRMSNorm(k_t, layer.attn_k_norm, 1e-6F, k_t);
      }

      models::ForwardRoPE(q_t, k_t,
                          static_cast<std::uint32_t>(num_q_heads),
                          static_cast<std::uint32_t>(num_kv_heads),
                          static_cast<std::uint32_t>(head_dim),
                          static_cast<std::uint32_t>(head_dim),
                          pos, weights_.config.rope_theta);
    }

    // Non-causal block attention over [injected_KV; block_KV]
    const std::size_t gqa_group = num_q_heads / num_kv_heads;
    for (std::size_t t = 0; t < draft_count; ++t) {
      std::vector<float> head_out(q_dim, 0.0F);

      for (std::size_t h = 0; h < num_q_heads; ++h) {
        const std::size_t kv_h = h / gqa_group;
        const auto* q_head = q_all.data() + (t * q_dim) + (h * head_dim);

        std::vector<float> scores(total_k_len);
        float max_score = -std::numeric_limits<float>::infinity();

        // 1. Scores against historical injected KV cache
        for (std::size_t p = 0; p < current_pos; ++p) {
          const auto* k_hist = injected_k_[layer_idx].data() + (p * kv_dim) + (kv_h * head_dim);
          double dot = 0.0;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q_head[d]) * static_cast<double>(k_hist[d]);
          }
          const float score = static_cast<float>(dot) * kq_scale;
          scores[p] = score;
          max_score = std::max(max_score, score);
        }

        // 2. Non-causal scores against ALL tokens in the noise block
        for (std::size_t p = 0; p < draft_count; ++p) {
          const auto* k_curr = k_block.data() + (p * kv_dim) + (kv_h * head_dim);
          double dot = 0.0;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(q_head[d]) * static_cast<double>(k_curr[d]);
          }
          const float score = static_cast<float>(dot) * kq_scale;
          scores[current_pos + p] = score;
          max_score = std::max(max_score, score);
        }

        // Softmax
        double sum_exp = 0.0;
        for (std::size_t p = 0; p < total_k_len; ++p) {
          scores[p] = std::exp(scores[p] - max_score);
          sum_exp += scores[p];
        }
        const float inv_sum = static_cast<float>(1.0 / (sum_exp + 1e-9));
        for (std::size_t p = 0; p < total_k_len; ++p) {
          scores[p] *= inv_sum;
        }

        // Weighted sum with V
        auto* out_head = head_out.data() + (h * head_dim);
        for (std::size_t p = 0; p < current_pos; ++p) {
          const auto* v_hist = injected_v_[layer_idx].data() + (p * kv_dim) + (kv_h * head_dim);
          const float w = scores[p];
          for (std::size_t d = 0; d < head_dim; ++d) {
            out_head[d] += w * v_hist[d];
          }
        }
        for (std::size_t p = 0; p < draft_count; ++p) {
          const auto* v_curr = v_block.data() + (p * kv_dim) + (kv_h * head_dim);
          const float w = scores[current_pos + p];
          for (std::size_t d = 0; d < head_dim; ++d) {
            out_head[d] += w * v_curr[d];
          }
        }
      }

      // Attention output projection + residual add
      auto tok_attn_out = std::span<float>(attn_out.data() + (t * hidden_size), hidden_size);
      MatMulVector(layer.attn_output, head_out, tok_attn_out, hidden_size, q_dim);
      for (std::size_t d = 0; d < hidden_size; ++d) {
        block_hidden[t * hidden_size + d] += tok_attn_out[d];
      }
    }

    // SwiGLU FFN
    for (std::size_t t = 0; t < draft_count; ++t) {
      auto tok_hidden = std::span<float>(block_hidden.data() + (t * hidden_size), hidden_size);
      std::vector<float> normed(hidden_size);
      models::ForwardRMSNorm(tok_hidden, layer.ffn_norm, 1e-6F, normed);

      auto gate = std::span<float>(ffn_gate.data() + (t * intermediate_size), intermediate_size);
      auto up = std::span<float>(ffn_up.data() + (t * intermediate_size), intermediate_size);
      auto down = std::span<float>(ffn_down.data() + (t * hidden_size), hidden_size);

      MatMulVector(layer.ffn_gate, normed, gate, intermediate_size, hidden_size);
      MatMulVector(layer.ffn_up, normed, up, intermediate_size, hidden_size);

      SiLU(gate);
      for (std::size_t i = 0; i < intermediate_size; ++i) {
        gate[i] *= up[i];
      }

      MatMulVector(layer.ffn_down, gate, down, hidden_size, intermediate_size);
      for (std::size_t d = 0; d < hidden_size; ++d) {
        block_hidden[t * hidden_size + d] += down[d];
      }
    }
  }

  // DFlash-2 Output Dynamic Conv
  if (weights_.dflash_config.has_dynamic_conv) {
    ApplyDynamicConv2Tap(block_hidden, draft_count, hidden_size,
                         weights_.out_conv_weight, weights_.out_conv_bias);
  }

  // Final Output Norm & LM Head projection
  out_logits.resize(draft_count * vocab_size);
  std::vector<float> normed_out(hidden_size);
  std::vector<float> draft_logits(weights_.dflash_config.draft_vocab_size);

  for (std::size_t t = 0; t < draft_count; ++t) {
    auto tok_hidden = std::span<float>(block_hidden.data() + (t * hidden_size), hidden_size);
    models::ForwardRMSNorm(tok_hidden, weights_.output_norm, 1e-6F, normed_out);

    auto target_logits_row = std::span<float>(out_logits.data() + (t * vocab_size), vocab_size);

    if (!weights_.d2t.empty()) {
      MatMulVector(weights_.output, normed_out, draft_logits,
                   weights_.dflash_config.draft_vocab_size, hidden_size);
      std::fill(target_logits_row.begin(), target_logits_row.end(),
                -std::numeric_limits<float>::infinity());
      for (std::size_t d = 0; d < weights_.dflash_config.draft_vocab_size; ++d) {
        const auto target_id = static_cast<std::size_t>(weights_.d2t.Get(d));
        if (target_id < vocab_size) {
          target_logits_row[target_id] = draft_logits[d];
        }
      }
    } else {
      MatMulVector(weights_.output, normed_out, target_logits_row, vocab_size,
                   hidden_size);
    }
  }
}

std::vector<tokenization::TokenId> QwenDFlashReference::ForwardBlock(
    tokenization::TokenId anchor_token, std::uint32_t current_pos,
    std::uint32_t draft_count, std::vector<float>* out_confidences) {
  std::vector<float> logits;
  ComputeBlockLogits(anchor_token, current_pos, draft_count, logits);

  std::vector<tokenization::TokenId> tokens;
  const std::size_t vocab_size = weights_.config.vocab_size;

  if (weights_.dflash_config.has_path_selector) {
    ApplyMarkovPathScoring(
        logits, draft_count, vocab_size, weights_.dflash_config.draft_vocab_size,
        weights_.markov_w1, weights_.markov_w2, weights_.d2t, anchor_token,
        weights_.dflash_config.sample_from_anchor, tokens);
  } else {
    tokens.reserve(draft_count);
    for (std::size_t t = 0; t < draft_count; ++t) {
      const auto row = std::span<const float>(logits.data() + (t * vocab_size), vocab_size);
      float max_val = -std::numeric_limits<float>::infinity();
      tokenization::TokenId best_tok = 0;
      for (std::size_t v = 0; v < vocab_size; ++v) {
        if (row[v] > max_val) {
          max_val = row[v];
          best_tok = static_cast<tokenization::TokenId>(v);
        }
      }
      tokens.push_back(best_tok);
    }
  }

  if (out_confidences != nullptr) {
    out_confidences->assign(tokens.size(), 1.0F);
  }

  return tokens;
}

}  // namespace strix::speculative
