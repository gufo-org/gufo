#include "src/models/fish_drama3/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "src/core/json.hpp"

namespace gufo::models::fish_drama3 {
namespace {

namespace json = gufo::json;

bool ReadU32(const json::Value& object, const char* key, std::uint32_t* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_number()) {
    return false;
  }
  const double num = value->as_double();
  if (!std::isfinite(num) || num < 0.0 || std::floor(num) != num) {
    return false;
  }
  *out = static_cast<std::uint32_t>(num);
  return true;
}

bool ReadI32(const json::Value& object, const char* key, std::int32_t* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_number()) {
    return false;
  }
  const double num = value->as_double();
  if (!std::isfinite(num) || std::floor(num) != num) {
    return false;
  }
  *out = static_cast<std::int32_t>(num);
  return true;
}

bool ReadFloat(const json::Value& object, const char* key, float* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_number()) {
    return false;
  }
  *out = static_cast<float>(value->as_double());
  return std::isfinite(*out);
}

bool ReadBool(const json::Value& object, const char* key, bool* out) {
  const auto* value = object.find(key);
  if (value == nullptr || !value->is_bool()) {
    return false;
  }
  *out = value->as_bool();
  return true;
}

}  // namespace

bool IsSupportedModelConfig(const ModelConfig& config) {
  if (config.model_type != "fish_omni" ||
      config.speech_template_version != "reference-bracket-v1") {
    return false;
  }
  const auto& b = config.backbone;
  const auto& t = config.tower;
  if (b.hidden_size == 0 || b.num_hidden_layers == 0 ||
      b.num_attention_heads == 0 || b.num_key_value_heads == 0 ||
      b.num_attention_heads % b.num_key_value_heads != 0 || b.head_dim == 0 ||
      b.RotaryDim() == 0 || b.RotaryDim() % 2 != 0 || b.num_experts == 0 ||
      b.num_experts_per_tok == 0 || b.num_experts_per_tok > b.num_experts ||
      b.intermediate_size != b.shared_expert_intermediate_size ||
      !b.layernorm_zero_centered_gamma ||
      b.linear_num_value_heads % b.linear_num_key_heads != 0) {
    return false;
  }
  if (!b.experts_per_layer.empty()) {
    if (b.experts_per_layer.size() != b.num_hidden_layers) {
      return false;
    }
    for (const auto count : b.experts_per_layer) {
      if (count < b.num_experts_per_tok || count > b.num_experts) {
        return false;
      }
    }
  }
  if (b.lm_head_rows != 0 && b.lm_head_rows != t.codebook_size + 1) {
    return false;
  }
  if (t.hidden_size != b.hidden_size || t.residual_layers != 9 ||
      t.shared_layers == 0 || t.heads == 0 || t.kv_heads == 0 ||
      t.heads % t.kv_heads != 0 || t.RotaryDim() == 0 ||
      t.RotaryDim() % 2 != 0 || !t.rotary_interleaved ||
      !t.layernorm_zero_centered_gamma || !t.attention_output_gate) {
    return false;
  }
  if (static_cast<std::int64_t>(config.semantic_end_token_id) + 1 -
          config.semantic_start_token_id !=
      static_cast<std::int64_t>(t.codebook_size)) {
    return false;
  }
  return true;
}

std::optional<ModelConfig> ParseModelConfig(const std::string& json_contents) {
  json::Value root;
  try {
    root = json::parse(json_contents);
  } catch (const std::exception&) {
    return std::nullopt;
  }
  if (!root.is_object()) {
    return std::nullopt;
  }

  ModelConfig config;
  config.model_type = root.member_str("model_type");
  config.speech_template_version = root.member_str("speech_template_version");
  (void)ReadI32(root, "eos_token_id", &config.eos_token_id);
  (void)ReadI32(root, "pad_token_id", &config.pad_token_id);
  (void)ReadI32(root, "audio_pad_token_id", &config.audio_pad_token_id);
  (void)ReadI32(root, "semantic_start_token_id",
                &config.semantic_start_token_id);
  (void)ReadI32(root, "semantic_end_token_id", &config.semantic_end_token_id);

  const auto* text = root.find("text_config");
  const auto* audio = root.find("audio_decoder_config");
  if (text == nullptr || !text->is_object() || audio == nullptr ||
      !audio->is_object()) {
    return std::nullopt;
  }

  auto& b = config.backbone;
  (void)ReadU32(*text, "vocab_size", &b.vocab_size);
  (void)ReadU32(*text, "hidden_size", &b.hidden_size);
  (void)ReadU32(*text, "moe_intermediate_size", &b.intermediate_size);
  (void)ReadU32(*text, "shared_expert_intermediate_size",
                &b.shared_expert_intermediate_size);
  (void)ReadU32(*text, "num_experts", &b.num_experts);
  (void)ReadU32(*text, "num_experts_per_tok", &b.num_experts_per_tok);
  (void)ReadU32(*text, "num_hidden_layers", &b.num_hidden_layers);
  (void)ReadU32(*text, "num_attention_heads", &b.num_attention_heads);
  (void)ReadU32(*text, "num_key_value_heads", &b.num_key_value_heads);
  (void)ReadU32(*text, "head_dim", &b.head_dim);
  (void)ReadU32(*text, "linear_conv_kernel_dim", &b.linear_conv_kernel_dim);
  (void)ReadU32(*text, "linear_key_head_dim", &b.linear_key_head_dim);
  (void)ReadU32(*text, "linear_num_key_heads", &b.linear_num_key_heads);
  (void)ReadU32(*text, "linear_num_value_heads", &b.linear_num_value_heads);
  (void)ReadU32(*text, "linear_value_head_dim", &b.linear_value_head_dim);
  (void)ReadFloat(*text, "rms_norm_eps", &b.rms_norm_eps);
  (void)ReadBool(*text, "layernorm_zero_centered_gamma",
                 &b.layernorm_zero_centered_gamma);
  (void)ReadBool(*text, "tie_word_embeddings", &b.tie_word_embeddings);
  (void)ReadU32(root, "semantic_head_rows", &b.lm_head_rows);

  if (const auto* rope = text->find("rope_parameters");
      rope != nullptr && rope->is_object()) {
    (void)ReadFloat(*rope, "rope_theta", &b.rope_theta);
    (void)ReadFloat(*rope, "partial_rotary_factor", &b.partial_rotary_factor);
  }

  if (const auto* layer_types = text->find("layer_types");
      layer_types != nullptr && layer_types->is_array()) {
    if (layer_types->size() != b.num_hidden_layers) {
      return std::nullopt;
    }
    for (std::size_t i = 0; i < layer_types->size(); ++i) {
      const std::string expected = ((i + 1) % b.full_attention_interval == 0)
                                       ? "full_attention"
                                       : "linear_attention";
      if (!layer_types->items()[i].is_string() ||
          layer_types->items()[i].str() != expected) {
        return std::nullopt;
      }
    }
  }

  if (const auto* per_layer = text->find("num_experts_per_layer");
      per_layer != nullptr && per_layer->is_array()) {
    for (const auto& item : per_layer->items()) {
      if (!item.is_number()) {
        return std::nullopt;
      }
      b.experts_per_layer.push_back(static_cast<std::uint32_t>(item.as_size()));
    }
  }

  auto& t = config.tower;
  if (audio->member_str("decoder_type") != "codebook_mtp_shared") {
    return std::nullopt;
  }
  (void)ReadU32(*audio, "dim", &t.hidden_size);
  (void)ReadU32(*audio, "intermediate_size", &t.intermediate_size);
  (void)ReadU32(*audio, "n_head", &t.heads);
  (void)ReadU32(*audio, "n_local_heads", &t.kv_heads);
  (void)ReadU32(*audio, "head_dim", &t.head_dim);
  (void)ReadU32(*audio, "vocab_size", &t.codebook_size);
  (void)ReadU32(*audio, "shared_enrichment_layers", &t.shared_layers);
  std::uint32_t num_codebooks = 10;
  if (ReadU32(*audio, "num_codebooks", &num_codebooks) && num_codebooks > 0) {
    t.residual_layers = num_codebooks - 1;
  }
  (void)ReadU32(*audio, "vq_backbone_codebooks", &t.backbone_codebooks);
  (void)ReadFloat(*audio, "norm_eps", &t.norm_eps);
  (void)ReadFloat(*audio, "rope_base", &t.rope_theta);
  (void)ReadFloat(*audio, "rotary_percent", &t.rotary_fraction);
  (void)ReadBool(*audio, "rotary_interleaved", &t.rotary_interleaved);
  (void)ReadBool(*audio, "layernorm_zero_centered_gamma",
                 &t.layernorm_zero_centered_gamma);
  (void)ReadBool(*audio, "attention_output_gate", &t.attention_output_gate);
  t.semantic_begin = config.semantic_start_token_id;

  return config;
}

std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir) {
  const std::filesystem::path config_path =
      std::filesystem::path(model_dir) / "config.json";
  std::ifstream file(config_path);
  if (!file.is_open()) {
    return std::nullopt;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return ParseModelConfig(buffer.str());
}

bool LooksLikeFishDrama3(const std::string& model_dir) {
  const auto config = LoadModelConfigFromPath(model_dir);
  return config.has_value() && IsSupportedModelConfig(*config);
}

}  // namespace gufo::models::fish_drama3
