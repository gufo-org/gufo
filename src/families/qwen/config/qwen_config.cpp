#include "src/families/qwen/config/qwen_config.hpp"

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen {

std::optional<core::ModelConfig> ExtractQwenConfig(
    const core::GgufReader& reader, std::string* error_msg) {
  using core::ModelConfig;
  ModelConfig config;

  // 1. Check architecture prefix (e.g. "qwen2", "qwen3", "qwen35", "llama")
  auto arch_str =
      reader.GetMetadataString("general.architecture").value_or("qwen");
  config.architecture = std::string(arch_str);
  const std::string prefix = config.architecture + ".";

  // 2. Read standard architectural hyperparameters
  const auto total_layers = reader.GetMetadataUint32(prefix + "block_count")
                                .value_or(config.num_layers);
  if (auto val = reader.GetMetadataUint32(prefix + "embedding_length")) {
    config.hidden_size = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "feed_forward_length")) {
    config.intermediate_size = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "attention.head_count")) {
    config.num_attention_heads = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "attention.head_count_kv")) {
    config.num_key_value_heads = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "attention.key_length")) {
    config.head_dim = *val;
  } else if (config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "context_length")) {
    config.context_length = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "full_attention_interval")) {
    config.full_attention_interval = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "ssm.conv_kernel")) {
    config.ssm_conv_kernel = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "ssm.state_size")) {
    config.ssm_state_size = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "ssm.group_count")) {
    config.ssm_group_count = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "ssm.time_step_rank")) {
    config.ssm_time_step_rank = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "ssm.inner_size")) {
    config.ssm_inner_size = *val;
  }
  if (auto val = reader.GetMetadataFloat32(prefix + "rope.freq_base")) {
    config.rope_theta = *val;
  }
  if (auto val = reader.GetMetadataUint32(prefix + "rope.dimension_count")) {
    config.rotary_dim = *val;
  } else {
    config.rotary_dim = config.head_dim;
  }

  // 3. MTP speculative layers are included in qwen35.block_count but are not
  // part of the main autoregressive transformer stack.
  const auto embedded_mtp_layers =
      reader.GetMetadataUint32(prefix + "nextn_predict_layers");
  config.mtp_num_layers =
      embedded_mtp_layers.value_or(reader.HasMtpTensors() ? 1U : 0U);
  const std::uint32_t layers_in_block_count = embedded_mtp_layers.value_or(0U);
  if (layers_in_block_count > total_layers) {
    if (error_msg != nullptr) {
      *error_msg = "MTP layer count exceeds qwen block count";
    }
    return std::nullopt;
  }
  config.num_layers = total_layers - layers_in_block_count;

  // 4. Assert text-only contract
  if (reader.HasVisionTensors()) {
    config.is_text_only = false;
    if (error_msg != nullptr) {
      *error_msg =
          "Model contains vision encoder tensors; text-only contract violation";
    }
    return std::nullopt;
  }
  config.is_text_only = true;

  // 5. Model name
  if (auto name = reader.GetMetadataString("general.name")) {
    config.model_name = std::string(*name);
  }
  if (const auto* token_embd = reader.FindTensor("token_embd.weight");
      token_embd != nullptr && config.hidden_size > 0) {
    const auto elements = token_embd->ElementCount();
    if ((elements % config.hidden_size) == 0) {
      config.vocab_size =
          static_cast<std::uint32_t>(elements / config.hidden_size);
    }
  }

  // 6. Validate configuration against Qwen3.5/3.8 structural rules. The rules
  // themselves stay with core::ModelConfig until the text migration moves that
  // type; Card 06's continuation layout validates with the same member.
  if (!config.IsValidQwen()) {
    if (error_msg != nullptr) {
      *error_msg = "Model config failed Qwen structural validation (head_dim=" +
                   std::to_string(config.head_dim) +
                   ", layers=" + std::to_string(config.num_layers) + ")";
    }
    return std::nullopt;
  }

  return config;
}

}  // namespace gufo::models::qwen
