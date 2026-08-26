#include "src/models/qwen/dflash_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
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

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address < region_address) {
      continue;
    }
    const auto offset = tensor_address - region_address;
    if (offset < region.size) {
      available_bytes = region.size - offset;
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

[[nodiscard]] bool IsSupportedMatrixType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
         type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
         type == core::GgmlType::kQ6_K;
}

[[nodiscard]] bool IsSupportedVectorType(core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

[[nodiscard]] bool BindRequiredTensor(const core::GgufReader& reader,
                                      std::string_view name,
                                      std::size_t expected_elements,
                                      bool matrix,
                                      models::QwenTensorRef& destination,
                                      std::string* error_msg) {
  destination = TensorRef(reader, name);
  if (destination.empty()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required tensor '" +
                            std::string(name) + "'");
    return false;
  }
  if (destination.num_elements != expected_elements) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) + "' has " +
                            std::to_string(destination.num_elements) +
                            " elements; expected " +
                            std::to_string(expected_elements));
    return false;
  }
  const bool supported = matrix ? IsSupportedMatrixType(destination.type)
                                : IsSupportedVectorType(destination.type);
  if (!supported) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' has unsupported type " +
                            std::string(core::ToString(destination.type)));
    return false;
  }
  if (!destination.FitsAvailableStorage()) {
    SetError(error_msg, "DFlash-2 tensor '" + std::string(name) +
                            "' exceeds its mapped GGUF storage");
    return false;
  }
  return true;
}

[[nodiscard]] std::optional<std::uint32_t> RequiredUint32(
    const core::GgufReader& reader, std::string_view key,
    std::string* error_msg) {
  auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    SetError(error_msg, "DFlash-2 GGUF is missing required metadata '" +
                            std::string(key) + "'");
  }
  return value;
}

[[nodiscard]] std::optional<std::vector<std::uint64_t>>
GetNonnegativeIntegerArray(const core::GgufReader& reader,
                           std::string_view key) {
  const auto* metadata = reader.FindMetadata(key);
  if (metadata == nullptr) {
    return std::nullopt;
  }
  if (const auto* values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    return *values;
  }
  if (const auto* values =
          std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    std::vector<std::uint64_t> converted;
    converted.reserve(values->size());
    for (const std::int64_t value : *values) {
      if (value < 0) {
        return std::nullopt;
      }
      converted.push_back(static_cast<std::uint64_t>(value));
    }
    return converted;
  }
  return std::nullopt;
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
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    SetError(error_msg, "DFlash-2 GGUF requires general.architecture='dflash'");
    return std::nullopt;
  }

  core::ModelConfig config;
  config.architecture = "dflash";
  const std::string prefix = "dflash.";

  const auto hidden_size =
      RequiredUint32(reader, prefix + "embedding_length", error_msg);
  const auto intermediate_size =
      RequiredUint32(reader, prefix + "feed_forward_length", error_msg);
  const auto num_attention_heads =
      RequiredUint32(reader, prefix + "attention.head_count", error_msg);
  const auto num_key_value_heads =
      RequiredUint32(reader, prefix + "attention.head_count_kv", error_msg);
  const auto head_dim =
      RequiredUint32(reader, prefix + "attention.key_length", error_msg);
  auto rotary_dim = reader.GetMetadataUint32(prefix + "rope.dimension_count");
  if (!rotary_dim.has_value()) {
    const auto sections =
        GetNonnegativeIntegerArray(reader, prefix + "rope.dimension_sections");
    if (sections.has_value()) {
      const auto& values = *sections;
      if (values.size() == 4 && values[0] > 0 && values[1] == 0 &&
          values[2] == 0 && values[3] == 0 &&
          values[0] <= std::numeric_limits<std::uint32_t>::max() / 2U) {
        rotary_dim = static_cast<std::uint32_t>(values[0] * 2U);
      }
    }
  }
  const auto context_length =
      RequiredUint32(reader, prefix + "context_length", error_msg);
  const auto num_layers =
      RequiredUint32(reader, prefix + "block_count", error_msg);
  const auto rope_theta = reader.GetMetadataFloat32(prefix + "rope.freq_base");
  if (!hidden_size || !intermediate_size || !num_attention_heads ||
      !num_key_value_heads || !head_dim || !rotary_dim || !context_length ||
      !num_layers || !rope_theta) {
    if (!rotary_dim) {
      SetError(error_msg,
               "DFlash-2 GGUF requires a valid rotary dimension or "
               "degenerate four-section M-RoPE layout");
    }
    if (!rope_theta) {
      SetError(
          error_msg,
          "DFlash-2 GGUF is missing required metadata 'dflash.rope.freq_base'");
    }
    return std::nullopt;
  }

  config.hidden_size = *hidden_size;
  config.intermediate_size = *intermediate_size;
  config.num_attention_heads = *num_attention_heads;
  config.num_key_value_heads = *num_key_value_heads;
  config.head_dim = *head_dim;
  config.rotary_dim = *rotary_dim;
  config.rope_theta = *rope_theta;
  config.context_length = *context_length;
  config.num_layers = *num_layers;
  config.full_attention_interval = 1;
  config.mtp_num_layers = 0;

  if (config.hidden_size == 0 || config.intermediate_size == 0 ||
      config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
      config.head_dim == 0 || config.num_layers == 0 ||
      config.num_attention_heads % config.num_key_value_heads != 0 ||
      config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
      (config.rotary_dim % 2U) != 0) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid transformer topology");
    return std::nullopt;
  }

  QwenDFlashWeights weights;
  weights.config = config;
  auto& df_cfg = weights.dflash_config;

  const auto block_size =
      RequiredUint32(reader, prefix + "block_size", error_msg);
  const auto conv_kernel_size =
      RequiredUint32(reader, prefix + "conv_kernel_size", error_msg);
  const auto conv_group_size =
      RequiredUint32(reader, prefix + "conv_group_size", error_msg);
  const auto selector_rank =
      RequiredUint32(reader, prefix + "selector_rank", error_msg);
  const auto selector_top_k =
      RequiredUint32(reader, prefix + "selector_top_k", error_msg);
  const auto sliding_window =
      RequiredUint32(reader, prefix + "attention.sliding_window", error_msg);
  const auto mask_token_id =
      RequiredUint32(reader, "tokenizer.ggml.mask_token_id", error_msg);
  const auto is_causal = reader.GetMetadataBool(prefix + "attention.causal");
  if (!block_size || !conv_kernel_size || !conv_group_size || !selector_rank ||
      !selector_top_k || !sliding_window || !mask_token_id ||
      !is_causal.has_value()) {
    if (!is_causal.has_value()) {
      SetError(error_msg,
               "DFlash-2 GGUF is missing required metadata "
               "'dflash.attention.causal'");
    }
    return std::nullopt;
  }

  df_cfg.block_size = *block_size;
  df_cfg.conv_kernel_size = *conv_kernel_size;
  df_cfg.conv_group_size = *conv_group_size;
  df_cfg.selector_rank = *selector_rank;
  df_cfg.selector_top_k = *selector_top_k;
  df_cfg.sliding_window = *sliding_window;
  df_cfg.mask_token_id = *mask_token_id;
  df_cfg.is_causal = *is_causal;
  df_cfg.num_layers = config.num_layers;

  if (df_cfg.block_size < 2 || df_cfg.conv_kernel_size == 0 ||
      df_cfg.conv_group_size == 0 ||
      config.hidden_size % df_cfg.conv_group_size != 0 ||
      df_cfg.selector_rank == 0 || df_cfg.selector_top_k == 0 ||
      df_cfg.selector_top_k > 16U ||
      df_cfg.sliding_window < df_cfg.block_size || df_cfg.is_causal) {
    SetError(error_msg, "DFlash-2 GGUF has an invalid diffusion topology");
    return std::nullopt;
  }

  // GGUF target_layers are layer-input indices (the convention used by the
  // converter). Strix taps layer outputs, so normalize N to zero-based N - 1.
  const auto encoded_target_layers =
      GetNonnegativeIntegerArray(reader, prefix + "target_layers");
  if (!encoded_target_layers.has_value()) {
    SetError(error_msg,
             "DFlash-2 GGUF is missing required integer array "
             "'dflash.target_layers'");
    return std::nullopt;
  }
  if (encoded_target_layers->empty()) {
    SetError(error_msg, "DFlash-2 target layer list must not be empty");
    return std::nullopt;
  }
  df_cfg.target_layer_ids.reserve(encoded_target_layers->size());
  for (const std::uint64_t encoded_layer : *encoded_target_layers) {
    if (encoded_layer == 0 ||
        !std::in_range<std::uint32_t>(encoded_layer - 1U)) {
      SetError(error_msg,
               "DFlash-2 target layer indices must use positive GGUF "
               "layer-input numbering");
      return std::nullopt;
    }
    const auto normalized = static_cast<std::uint32_t>(encoded_layer - 1U);
    if (!df_cfg.target_layer_ids.empty() &&
        normalized <= df_cfg.target_layer_ids.back()) {
      SetError(error_msg,
               "DFlash-2 target layer indices must be strictly increasing");
      return std::nullopt;
    }
    df_cfg.target_layer_ids.push_back(normalized);
  }

  const std::size_t hidden = config.hidden_size;
  const std::size_t intermediate = config.intermediate_size;
  const std::size_t attention = config.AttentionSize();
  const std::size_t kv =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t groups = hidden / df_cfg.conv_group_size;
  const std::size_t dynamic_coefficients =
      std::size_t{2} * df_cfg.conv_kernel_size * groups;

  // DFlash-2 uses the target model's embedding and output head. Private copies
  // are accepted but not required.
  weights.token_embedding = TensorRef(reader, "token_embd.weight");
  weights.output = TensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    weights.output = weights.token_embedding;
  }
  weights.d2t = TensorRef(reader, "d2t");
  if (!weights.d2t.empty()) {
    SetError(error_msg,
             "DFlash-2 reduced-vocabulary d2t exports are not supported yet");
    return std::nullopt;
  }

  const auto* predecessor_tensor =
      reader.FindTensor("selector_predecessor.weight");
  if (predecessor_tensor == nullptr || predecessor_tensor->data == nullptr ||
      predecessor_tensor->ElementCount() % df_cfg.selector_rank != 0) {
    SetError(error_msg,
             "DFlash-2 selector_predecessor.weight has invalid dimensions");
    return std::nullopt;
  }
  weights.selector_predecessor =
      TensorRef(reader, "selector_predecessor.weight");
  if (!IsSupportedMatrixType(weights.selector_predecessor.type) ||
      !weights.selector_predecessor.FitsAvailableStorage()) {
    SetError(error_msg,
             "DFlash-2 selector_predecessor.weight has unsupported storage");
    return std::nullopt;
  }
  const std::size_t inferred_vocab =
      weights.selector_predecessor.num_elements / df_cfg.selector_rank;
  if (inferred_vocab == 0 || !std::in_range<std::uint32_t>(inferred_vocab)) {
    SetError(error_msg, "DFlash-2 selector vocabulary size is invalid");
    return std::nullopt;
  }
  config.vocab_size = static_cast<std::uint32_t>(inferred_vocab);
  weights.config.vocab_size = config.vocab_size;
  df_cfg.draft_vocab_size = inferred_vocab;

  if (!BindRequiredTensor(reader, "selector_successor.weight",
                          inferred_vocab * df_cfg.selector_rank, true,
                          weights.selector_successor, error_msg) ||
      !BindRequiredTensor(reader, "selector_hidden.weight",
                          hidden * df_cfg.selector_rank, true,
                          weights.selector_hidden, error_msg)) {
    return std::nullopt;
  }

  const std::size_t encoder_input = df_cfg.target_layer_ids.size() * hidden;
  if (!BindRequiredTensor(reader, "fc.weight", hidden * encoder_input, true,
                          weights.fc_projection, error_msg) ||
      !BindRequiredTensor(reader, "enc.output_norm.weight", hidden, false,
                          weights.fc_norm, error_msg) ||
      !BindRequiredTensor(reader, "output_norm.weight", hidden, false,
                          weights.output_norm, error_msg)) {
    return std::nullopt;
  }
  weights.fc_scale = TensorRef(reader, "fc.scale");
  if (!weights.fc_scale.empty() &&
      (weights.fc_scale.num_elements != 1 ||
       !IsSupportedVectorType(weights.fc_scale.type) ||
       !weights.fc_scale.FitsAvailableStorage())) {
    SetError(error_msg, "DFlash-2 fc.scale must be one F32/BF16 value");
    return std::nullopt;
  }

  weights.layers.resize(df_cfg.num_layers);
  for (std::uint32_t i = 0; i < df_cfg.num_layers; ++i) {
    const std::string layer_prefix = "blk." + std::to_string(i) + ".";
    auto& layer = weights.layers[i];
    auto& transformer = layer.transformer;
    transformer.is_full_attention = true;

    if (!BindRequiredTensor(reader, layer_prefix + "attn_norm.weight", hidden,
                            false, transformer.attn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q.weight",
                            attention * hidden, true, transformer.attn_q,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k.weight", kv * hidden,
                            true, transformer.attn_k, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_v.weight", kv * hidden,
                            true, transformer.attn_v, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_output.weight",
                            hidden * attention, true, transformer.attn_output,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_q_norm.weight",
                            config.head_dim, false, transformer.attn_q_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_k_norm.weight",
                            config.head_dim, false, transformer.attn_k_norm,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_norm.weight", hidden,
                            false, transformer.ffn_norm, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_gate.weight",
                            intermediate * hidden, true, transformer.ffn_gate,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_up.weight",
                            intermediate * hidden, true, transformer.ffn_up,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_down.weight",
                            hidden * intermediate, true, transformer.ffn_down,
                            error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_conv_base",
                            hidden * df_cfg.conv_kernel_size * 2U, false,
                            layer.attention_conv_base, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "attn_conv_proj.weight",
                            dynamic_coefficients * hidden, true,
                            layer.attention_conv_projection, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_conv_base",
                            hidden * df_cfg.conv_kernel_size * 2U, false,
                            layer.ffn_conv_base, error_msg) ||
        !BindRequiredTensor(reader, layer_prefix + "ffn_conv_proj.weight",
                            dynamic_coefficients * hidden, true,
                            layer.ffn_conv_projection, error_msg)) {
      return std::nullopt;
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
    const std::shared_ptr<const core::GgufReader>& reader,
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
  return std::unique_ptr<QwenDFlashReference>(new QwenDFlashReference(
      reader, nullptr, std::move(*weights), max_context));
}

std::unique_ptr<QwenDFlashReference> QwenDFlashReference::CreateWithTiedWeights(
    const std::shared_ptr<const core::GgufReader>& reader,
    const std::shared_ptr<const core::GgufReader>& tied_reader,
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
      const auto& layer = weights_.layers[layer_idx].transformer;

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
                          static_cast<std::uint32_t>(head_dim), token_pos,
                          weights_.config.rope_theta);

      // Store into injected KV cache
      const std::size_t cache_offset = token_pos * kv_dim;
      const auto cache_iterator_offset =
          static_cast<std::vector<float>::difference_type>(cache_offset);
      std::copy_n(k_proj.begin(), kv_dim,
                  injected_k_[layer_idx].begin() + cache_iterator_offset);
      std::copy_n(v_proj.begin(), kv_dim,
                  injected_v_[layer_idx].begin() + cache_iterator_offset);
    }
  }

  injected_context_len_ =
      std::max(injected_context_len_, position + num_tokens);
  return true;
}

void QwenDFlashReference::ApplyGroupedDynamicCausalConv(
    std::span<const float> input, std::span<const float> dynamic_coefficients,
    std::size_t num_tokens, std::size_t hidden_size, std::size_t kernel_size,
    std::size_t group_size, std::size_t side,
    const models::QwenTensorRef& base_kernel, std::span<float> output) {
  if (num_tokens == 0) {
    return;
  }
  if (kernel_size == 0 || group_size == 0 || hidden_size % group_size != 0 ||
      side >= 2 || input.size() != num_tokens * hidden_size ||
      output.size() != input.size() ||
      base_kernel.num_elements != 2U * kernel_size * hidden_size) {
    throw std::invalid_argument("invalid DFlash-2 dynamic convolution shape");
  }

  const std::size_t groups = hidden_size / group_size;
  const std::size_t coefficients_per_token = 2U * kernel_size * groups;
  if (dynamic_coefficients.size() != num_tokens * coefficients_per_token) {
    throw std::invalid_argument("invalid DFlash-2 dynamic coefficient shape");
  }

  std::ranges::fill(output, 0.0F);
  for (std::size_t token = 0; token < num_tokens; ++token) {
    for (std::size_t tap = 0; tap < kernel_size; ++tap) {
      if (token < tap) {
        continue;
      }
      const std::size_t input_token = token - tap;
      for (std::size_t channel = 0; channel < hidden_size; ++channel) {
        const std::size_t group = channel / group_size;
        const std::size_t dynamic_index =
            token * coefficients_per_token +
            ((side * kernel_size + tap) * groups) + group;
        const std::size_t base_index =
            ((side * kernel_size + tap) * hidden_size) + channel;
        output[token * hidden_size + channel] +=
            (base_kernel.Get(base_index) +
             dynamic_coefficients[dynamic_index]) *
            input[input_token * hidden_size + channel];
      }
    }
  }
}

void QwenDFlashReference::SelectCandidatePath(
    std::span<const float> normalized_hidden,
    std::span<const float> proposal_logits, std::size_t num_tokens,
    std::size_t hidden_size, std::size_t vocab_size, std::size_t selector_rank,
    std::size_t selector_top_k,
    const models::QwenTensorRef& selector_predecessor,
    const models::QwenTensorRef& selector_successor,
    const models::QwenTensorRef& selector_hidden,
    tokenization::TokenId anchor_token,
    std::vector<tokenization::TokenId>& out_tokens,
    std::vector<float>* out_confidences) {
  if (normalized_hidden.size() != num_tokens * hidden_size ||
      proposal_logits.size() != num_tokens * vocab_size || selector_rank == 0 ||
      selector_top_k == 0 ||
      selector_predecessor.num_elements != vocab_size * selector_rank ||
      selector_successor.num_elements != vocab_size * selector_rank ||
      selector_hidden.num_elements != hidden_size * selector_rank ||
      anchor_token >= vocab_size) {
    throw std::invalid_argument("invalid DFlash-2 selector shape");
  }

  out_tokens.clear();
  out_tokens.reserve(num_tokens);
  if (out_confidences != nullptr) {
    out_confidences->clear();
    out_confidences->reserve(num_tokens);
  }

  const std::size_t top_k = std::min(selector_top_k, vocab_size);
  std::vector<float> projected_hidden(selector_rank);
  std::vector<float> predecessor_code(selector_rank);
  std::vector<float> successor_code(selector_rank);
  std::vector<std::size_t> candidates(vocab_size);
  std::vector<float> candidate_scores(top_k);
  tokenization::TokenId predecessor = anchor_token;

  for (std::size_t token = 0; token < num_tokens; ++token) {
    const auto hidden_row =
        normalized_hidden.subspan(token * hidden_size, hidden_size);
    const auto logits_row =
        proposal_logits.subspan(token * vocab_size, vocab_size);
    MatMulVector(selector_hidden, hidden_row, projected_hidden, selector_rank,
                 hidden_size);
    DequantizeOrCopyRow(selector_predecessor, predecessor, selector_rank,
                        predecessor_code.data());

    std::iota(candidates.begin(), candidates.end(), 0U);
    const auto top_k_offset =
        static_cast<std::vector<std::size_t>::difference_type>(top_k);
    std::partial_sort(candidates.begin(), candidates.begin() + top_k_offset,
                      candidates.end(),
                      [&logits_row](std::size_t left, std::size_t right) {
                        return logits_row[left] > logits_row[right];
                      });

    std::size_t best_index = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < top_k; ++index) {
      const std::size_t candidate = candidates[index];
      DequantizeOrCopyRow(selector_successor, candidate, selector_rank,
                          successor_code.data());
      double transition = 0.0;
      for (std::size_t rank = 0; rank < selector_rank; ++rank) {
        transition += static_cast<double>(predecessor_code[rank]) *
                      static_cast<double>(projected_hidden[rank]) *
                      static_cast<double>(successor_code[rank]);
      }
      const float score =
          logits_row[candidate] + static_cast<float>(transition);
      candidate_scores[index] = score;
      if (score > best_score) {
        best_score = score;
        best_index = index;
      }
    }

    predecessor = static_cast<tokenization::TokenId>(candidates[best_index]);
    out_tokens.push_back(predecessor);

    if (out_confidences != nullptr) {
      double denominator = 0.0;
      for (const float score : candidate_scores) {
        denominator += std::exp(static_cast<double>(score - best_score));
      }
      out_confidences->push_back(
          static_cast<float>(1.0 / std::max(denominator, 1e-30)));
    }
  }
}

void QwenDFlashReference::ComputeBlockLogits(tokenization::TokenId anchor_token,
                                             std::uint32_t current_pos,
                                             std::uint32_t draft_count,
                                             std::vector<float>& out_logits) {
  const auto& dflash_config = weights_.dflash_config;
  draft_count = std::min(draft_count, dflash_config.block_size > 0
                                          ? dflash_config.block_size - 1U
                                          : 0U);
  if (draft_count == 0) {
    out_logits.clear();
    layer_scratch_.clear();
    return;
  }
  if (weights_.token_embedding.empty() || weights_.output.empty()) {
    throw std::logic_error(
        "DFlash-2 reference execution requires tied target embedding/output "
        "weights");
  }

  const std::size_t hidden_size = weights_.config.hidden_size;
  const std::size_t intermediate_size = weights_.config.intermediate_size;
  const std::size_t head_dim = weights_.config.head_dim;
  const std::size_t num_q_heads = weights_.config.num_attention_heads;
  const std::size_t num_kv_heads = weights_.config.num_key_value_heads;
  const std::size_t q_dim = num_q_heads * head_dim;
  const std::size_t kv_dim = num_kv_heads * head_dim;
  const std::size_t vocab_size = weights_.config.vocab_size;
  const float kq_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  const std::size_t total_tokens = draft_count + 1U;
  const std::size_t groups = hidden_size / dflash_config.conv_group_size;
  const std::size_t dynamic_size =
      std::size_t{2} * dflash_config.conv_kernel_size * groups;

  std::vector<float> block_hidden(total_tokens * hidden_size, 0.0F);

  // Position zero is the committed anchor. All proposal positions are masks.
  for (std::size_t t = 0; t < total_tokens; ++t) {
    const tokenization::TokenId tok =
        (t == 0) ? anchor_token : dflash_config.mask_token_id;
    DequantizeOrCopyRow(weights_.token_embedding, tok, hidden_size,
                        block_hidden.data() + (t * hidden_size));
  }

  std::vector<float> normed(total_tokens * hidden_size);
  std::vector<float> convolved(total_tokens * hidden_size);
  std::vector<float> dynamic(total_tokens * dynamic_size);
  std::vector<float> q_all(total_tokens * q_dim);
  std::vector<float> k_block(total_tokens * kv_dim);
  std::vector<float> v_block(total_tokens * kv_dim);
  std::vector<float> attn_projected(total_tokens * hidden_size);
  std::vector<float> conv_finished(total_tokens * hidden_size);
  std::vector<float> ffn_gate(total_tokens * intermediate_size);
  std::vector<float> ffn_up(total_tokens * intermediate_size);
  std::vector<float> ffn_down(total_tokens * hidden_size);

  for (std::size_t layer_idx = 0; layer_idx < weights_.layers.size();
       ++layer_idx) {
    const auto& dflash_layer = weights_.layers[layer_idx];
    const auto& layer = dflash_layer.transformer;

    // Attention pre-norm, dynamic prepare convolution, and Q/K/V.
    for (std::size_t t = 0; t < total_tokens; ++t) {
      const auto tok_hidden = std::span<const float>(
          block_hidden.data() + (t * hidden_size), hidden_size);
      auto tok_normed =
          std::span<float>(normed.data() + (t * hidden_size), hidden_size);
      models::ForwardRMSNorm(tok_hidden, layer.attn_norm, 1e-6F, tok_normed);

      auto dynamic_row =
          std::span<float>(dynamic.data() + (t * dynamic_size), dynamic_size);
      MatMulVector(dflash_layer.attention_conv_projection, tok_normed,
                   dynamic_row, dynamic_size, hidden_size);
    }
    ApplyGroupedDynamicCausalConv(normed, dynamic, total_tokens, hidden_size,
                                  dflash_config.conv_kernel_size,
                                  dflash_config.conv_group_size, 0,
                                  dflash_layer.attention_conv_base, convolved);

    for (std::size_t t = 0; t < total_tokens; ++t) {
      const std::uint32_t pos = current_pos + static_cast<std::uint32_t>(t);
      const auto tok_convolved = std::span<const float>(
          convolved.data() + (t * hidden_size), hidden_size);
      auto q_t = std::span<float>(q_all.data() + (t * q_dim), q_dim);
      auto k_t = std::span<float>(k_block.data() + (t * kv_dim), kv_dim);
      auto v_t = std::span<float>(v_block.data() + (t * kv_dim), kv_dim);

      MatMulVector(layer.attn_q, tok_convolved, q_t, q_dim, hidden_size);
      MatMulVector(layer.attn_k, tok_convolved, k_t, kv_dim, hidden_size);
      MatMulVector(layer.attn_v, tok_convolved, v_t, kv_dim, hidden_size);

      models::ForwardRMSNorm(q_t, layer.attn_q_norm, 1e-6F, q_t);
      models::ForwardRMSNorm(k_t, layer.attn_k_norm, 1e-6F, k_t);

      models::ForwardRoPE(q_t, k_t, static_cast<std::uint32_t>(num_q_heads),
                          static_cast<std::uint32_t>(num_kv_heads),
                          static_cast<std::uint32_t>(head_dim),
                          static_cast<std::uint32_t>(head_dim), pos,
                          weights_.config.rope_theta);
    }

    // Non-causal sliding-window attention over [injected context; block].
    const std::size_t gqa_group = num_q_heads / num_kv_heads;
    for (std::size_t t = 0; t < total_tokens; ++t) {
      const std::size_t query_position = current_pos + t;
      std::vector<float> head_out(q_dim, 0.0F);

      for (std::size_t h = 0; h < num_q_heads; ++h) {
        const std::size_t kv_h = h / gqa_group;
        const auto* q_head = q_all.data() + (t * q_dim) + (h * head_dim);

        const std::size_t history_end =
            std::min<std::size_t>(current_pos, injected_context_len_);
        const std::size_t history_start =
            query_position + 1U > dflash_config.sliding_window
                ? query_position + 1U - dflash_config.sliding_window
                : 0U;
        const std::size_t history_count =
            history_end > history_start ? history_end - history_start : 0U;
        std::vector<float> scores(history_count + total_tokens);
        float max_score = -std::numeric_limits<float>::infinity();

        for (std::size_t p = history_start; p < history_end; ++p) {
          const auto* k_hist =
              injected_k_[layer_idx].data() + (p * kv_dim) + (kv_h * head_dim);
          double dot = 0.0;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot +=
                static_cast<double>(q_head[d]) * static_cast<double>(k_hist[d]);
          }
          const float score = static_cast<float>(dot) * kq_scale;
          scores[p - history_start] = score;
          max_score = std::max(max_score, score);
        }

        for (std::size_t p = 0; p < total_tokens; ++p) {
          const auto block_position = current_pos + p;
          if (block_position + dflash_config.sliding_window <= query_position ||
              query_position + dflash_config.sliding_window <= block_position) {
            continue;
          }
          const auto* k_curr =
              k_block.data() + (p * kv_dim) + (kv_h * head_dim);
          double dot = 0.0;
          for (std::size_t d = 0; d < head_dim; ++d) {
            dot +=
                static_cast<double>(q_head[d]) * static_cast<double>(k_curr[d]);
          }
          const float score = static_cast<float>(dot) * kq_scale;
          scores[history_count + p] = score;
          max_score = std::max(max_score, score);
        }

        double sum_exp = 0.0;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          sum_exp += score;
        }
        const float inv_sum = static_cast<float>(1.0 / (sum_exp + 1e-9));
        for (float& score : scores) {
          score *= inv_sum;
        }

        auto* out_head = head_out.data() + (h * head_dim);
        for (std::size_t p = history_start; p < history_end; ++p) {
          const auto* v_hist =
              injected_v_[layer_idx].data() + (p * kv_dim) + (kv_h * head_dim);
          const float weight = scores[p - history_start];
          for (std::size_t d = 0; d < head_dim; ++d) {
            out_head[d] += weight * v_hist[d];
          }
        }
        for (std::size_t p = 0; p < total_tokens; ++p) {
          const auto* v_curr =
              v_block.data() + (p * kv_dim) + (kv_h * head_dim);
          const float weight = scores[history_count + p];
          for (std::size_t d = 0; d < head_dim; ++d) {
            out_head[d] += weight * v_curr[d];
          }
        }
      }

      auto tok_attn = std::span<float>(
          attn_projected.data() + (t * hidden_size), hidden_size);
      MatMulVector(layer.attn_output, head_out, tok_attn, hidden_size, q_dim);
    }
    ApplyGroupedDynamicCausalConv(
        attn_projected, dynamic, total_tokens, hidden_size,
        dflash_config.conv_kernel_size, dflash_config.conv_group_size, 1,
        dflash_layer.attention_conv_base, conv_finished);
    for (std::size_t index = 0; index < block_hidden.size(); ++index) {
      block_hidden[index] += conv_finished[index];
    }

    // FFN pre-norm, dynamic prepare convolution, SwiGLU, and finish.
    for (std::size_t t = 0; t < total_tokens; ++t) {
      const auto tok_hidden = std::span<const float>(
          block_hidden.data() + (t * hidden_size), hidden_size);
      auto tok_normed =
          std::span<float>(normed.data() + (t * hidden_size), hidden_size);
      models::ForwardRMSNorm(tok_hidden, layer.ffn_norm, 1e-6F, tok_normed);
      auto dynamic_row =
          std::span<float>(dynamic.data() + (t * dynamic_size), dynamic_size);
      MatMulVector(dflash_layer.ffn_conv_projection, tok_normed, dynamic_row,
                   dynamic_size, hidden_size);
    }
    ApplyGroupedDynamicCausalConv(normed, dynamic, total_tokens, hidden_size,
                                  dflash_config.conv_kernel_size,
                                  dflash_config.conv_group_size, 0,
                                  dflash_layer.ffn_conv_base, convolved);

    for (std::size_t t = 0; t < total_tokens; ++t) {
      const auto tok_convolved = std::span<const float>(
          convolved.data() + (t * hidden_size), hidden_size);
      auto gate = std::span<float>(ffn_gate.data() + (t * intermediate_size),
                                   intermediate_size);
      auto up = std::span<float>(ffn_up.data() + (t * intermediate_size),
                                 intermediate_size);
      auto down =
          std::span<float>(ffn_down.data() + (t * hidden_size), hidden_size);

      MatMulVector(layer.ffn_gate, tok_convolved, gate, intermediate_size,
                   hidden_size);
      MatMulVector(layer.ffn_up, tok_convolved, up, intermediate_size,
                   hidden_size);
      SiLU(gate);
      for (std::size_t i = 0; i < intermediate_size; ++i) {
        gate[i] *= up[i];
      }
      MatMulVector(layer.ffn_down, gate, down, hidden_size, intermediate_size);
    }
    ApplyGroupedDynamicCausalConv(ffn_down, dynamic, total_tokens, hidden_size,
                                  dflash_config.conv_kernel_size,
                                  dflash_config.conv_group_size, 1,
                                  dflash_layer.ffn_conv_base, conv_finished);
    for (std::size_t index = 0; index < block_hidden.size(); ++index) {
      block_hidden[index] += conv_finished[index];
    }
  }

  // The selector and tied output head consume final-normalized proposal
  // positions only; position zero is the already committed anchor.
  layer_scratch_.resize(draft_count * hidden_size);
  out_logits.resize(draft_count * vocab_size);
  std::vector<float> draft_logits(dflash_config.draft_vocab_size);
  for (std::size_t proposal = 0; proposal < draft_count; ++proposal) {
    const std::size_t block_position = proposal + 1U;
    const auto tok_hidden = std::span<const float>(
        block_hidden.data() + (block_position * hidden_size), hidden_size);
    auto normalized = std::span<float>(
        layer_scratch_.data() + (proposal * hidden_size), hidden_size);
    models::ForwardRMSNorm(tok_hidden, weights_.output_norm, 1e-6F, normalized);

    auto target_logits = std::span<float>(
        out_logits.data() + (proposal * vocab_size), vocab_size);
    if (!weights_.d2t.empty()) {
      MatMulVector(weights_.output, normalized, draft_logits,
                   dflash_config.draft_vocab_size, hidden_size);
      std::ranges::fill(target_logits, -std::numeric_limits<float>::infinity());
      for (std::size_t draft_id = 0; draft_id < dflash_config.draft_vocab_size;
           ++draft_id) {
        const auto target_id =
            static_cast<std::size_t>(weights_.d2t.Get(draft_id));
        if (target_id < vocab_size) {
          target_logits[target_id] = draft_logits[draft_id];
        }
      }
    } else {
      MatMulVector(weights_.output, normalized, target_logits, vocab_size,
                   hidden_size);
    }
  }
}

std::vector<tokenization::TokenId> QwenDFlashReference::ForwardBlock(
    tokenization::TokenId anchor_token, std::uint32_t current_pos,
    std::uint32_t draft_count, std::vector<float>* out_confidences) {
  std::vector<float> logits;
  ComputeBlockLogits(anchor_token, current_pos, draft_count, logits);
  const std::size_t actual_count =
      weights_.config.vocab_size == 0
          ? 0
          : logits.size() / weights_.config.vocab_size;
  if (actual_count == 0) {
    if (out_confidences != nullptr) {
      out_confidences->clear();
    }
    return {};
  }

  std::vector<tokenization::TokenId> tokens;
  SelectCandidatePath(
      layer_scratch_, logits, actual_count, weights_.config.hidden_size,
      weights_.config.vocab_size, weights_.dflash_config.selector_rank,
      weights_.dflash_config.selector_top_k, weights_.selector_predecessor,
      weights_.selector_successor, weights_.selector_hidden, anchor_token,
      tokens, out_confidences);
  return tokens;
}

}  // namespace strix::speculative
