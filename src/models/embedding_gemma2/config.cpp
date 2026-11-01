#include "src/models/embedding_gemma2/config.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <sstream>
#include <string>
#include <string_view>

namespace gufo::models::embedding_gemma2 {
namespace {

void ParseU32(const json::Value& object, std::string_view name,
              std::uint32_t* output) {
  if (!object.is_object()) {
    return;
  }
  const json::Value* value = object.find(std::string(name));
  if (value != nullptr && value->is_number()) {
    *output = static_cast<std::uint32_t>(value->as_size(*output));
  }
}

void ParseFloat(const json::Value& object, std::string_view name,
                float* output) {
  if (!object.is_object()) {
    return;
  }
  const json::Value* value = object.find(std::string(name));
  if (value != nullptr && value->is_number()) {
    *output = static_cast<float>(value->as_double(*output));
  }
}

std::optional<LayerType> ParseLayerType(std::string_view name) {
  if (name == "sliding_attention") {
    return LayerType::kSlidingAttention;
  }
  if (name == "full_attention") {
    return LayerType::kFullAttention;
  }
  return std::nullopt;
}

const char* LayerTypeName(LayerType type) {
  return type == LayerType::kFullAttention ? "full_attention"
                                           : "sliding_attention";
}

bool ParseLayerIndex(const std::string& key, std::uint32_t* index) {
  const char* const end = key.data() + key.size();
  const auto parsed = std::from_chars(key.data(), end, *index);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

bool ParseTextConfig(const json::Value& value, TextConfig* output) {
  ParseU32(value, "vocab_size", &output->vocab_size);
  ParseU32(value, "hidden_size", &output->hidden_size);
  ParseU32(value, "hidden_size_per_layer_input",
           &output->hidden_size_per_layer_input);
  ParseU32(value, "intermediate_size", &output->intermediate_size);
  ParseU32(value, "num_hidden_layers", &output->num_hidden_layers);
  ParseU32(value, "num_attention_heads", &output->num_attention_heads);
  ParseU32(value, "embedding_dim", &output->embedding_dim);
  ParseU32(value, "sliding_window", &output->sliding_window);
  ParseFloat(value, "rms_norm_eps", &output->rms_norm_eps);

  std::uint32_t head_dim = 256;
  std::uint32_t num_key_value_heads = 2;
  ParseU32(value, "head_dim", &head_dim);
  ParseU32(value, "num_key_value_heads", &num_key_value_heads);

  const json::Value* types = value.find("layer_types");
  if (types == nullptr || !types->is_array() ||
      types->size() != output->num_hidden_layers) {
    return false;
  }
  output->layers.assign(output->num_hidden_layers, LayerGeometry{});
  for (std::size_t index = 0; index < output->layers.size(); ++index) {
    const json::Value& name = types->items()[index];
    const auto type = name.is_string() ? ParseLayerType(name.str())
                                       : std::optional<LayerType>();
    if (!type.has_value()) {
      return false;
    }
    output->layers[index].type = *type;
    output->layers[index].head_dim = head_dim;
    output->layers[index].num_key_value_heads = num_key_value_heads;
  }

  // The checkpoint keys overrides by layer index, zero-padded as in "05".
  if (const json::Value* overrides = value.find("per_layer_config");
      overrides != nullptr && overrides->is_object()) {
    for (const auto& [key, layer] : overrides->members()) {
      std::uint32_t index = 0;
      if (!ParseLayerIndex(key, &index) || index >= output->layers.size()) {
        return false;
      }
      ParseU32(layer, "head_dim", &output->layers[index].head_dim);
      ParseU32(layer, "num_key_value_heads",
               &output->layers[index].num_key_value_heads);
    }
  }

  const json::Value* rope = value.find("rope_parameters");
  if (rope == nullptr || !rope->is_object()) {
    return false;
  }
  for (LayerGeometry& layer : output->layers) {
    const json::Value* parameters = rope->find(LayerTypeName(layer.type));
    if (parameters == nullptr || !parameters->is_object() ||
        parameters->member_str("rope_type") != "default") {
      return false;
    }
    ParseFloat(*parameters, "rope_theta", &layer.rope_theta);
  }
  return true;
}

}  // namespace

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
  config.dtype = root.member_str("dtype");
  if (const json::Value* architectures = root.find("architectures");
      architectures != nullptr && architectures->is_array() &&
      !architectures->items().empty() &&
      architectures->items().front().is_string()) {
    config.architecture = architectures->items().front().str();
  }
  const json::Value* text = root.find("text_config");
  if (text == nullptr || !text->is_object()) {
    return std::nullopt;
  }
  config.text_model_type = text->member_str("model_type");
  config.hidden_activation = text->member_str("hidden_activation");
  if (!ParseTextConfig(*text, &config.text)) {
    return std::nullopt;
  }
  if (const auto* vision = root.find("vision_config");
      vision && vision->is_object()) {
    const auto* rope = vision->find("rope_parameters");
    config.supported_vision =
        vision->member_str("model_type") == "gemma4_vision" &&
        vision->member_str("dtype") == "bfloat16" &&
        vision->member_str("hidden_activation") == "gelu_pytorch_tanh" &&
        vision->member_size("hidden_size") == 768 &&
        vision->member_size("intermediate_size") == 3072 &&
        vision->member_size("num_hidden_layers") == 16 &&
        vision->member_size("num_attention_heads") == 12 &&
        vision->member_size("num_key_value_heads") == 12 &&
        vision->member_size("head_dim") == 64 &&
        vision->member_size("patch_size") == 16 &&
        vision->member_size("pooling_kernel_size") == 3 &&
        vision->member_size("position_embedding_size") == 10240 &&
        (vision->find("standardize") &&
         !vision->find("standardize")->as_bool(true)) &&
        (vision->find("use_clipped_linears") &&
         !vision->find("use_clipped_linears")->as_bool(true)) &&
        (vision->find("attention_bias") &&
         !vision->find("attention_bias")->as_bool(true)) &&
        vision->member_double("rms_norm_eps") == 1e-6 && rope &&
        rope->member_str("rope_type") == "axial" &&
        rope->member_double("rope_theta") == 100.0 &&
        root.member_size("vision_soft_tokens_per_image") == 280 &&
        root.member_size("boi_token_id") == 255999 &&
        root.member_size("image_token_id") == 258880 &&
        root.member_size("eoi_token_id") == 258882;
  }
  config.supported_video = config.supported_vision &&
                           root.member_size("video_token_id") == kVideoTokenId;
  if (const auto* audio = root.find("audio_config");
      audio && audio->is_object()) {
    const auto* channels = audio->find("subsampling_conv_channels");
    const auto* clipped = audio->find("use_clipped_linears");
    config.supported_audio =
        audio->member_str("model_type") == "gemma4_audio" &&
        audio->member_str("dtype") == "bfloat16" &&
        audio->member_str("hidden_act") == "silu" &&
        audio->member_size("hidden_size") == 1024 &&
        audio->member_size("num_hidden_layers") == 12 &&
        audio->member_size("num_attention_heads") == 8 &&
        audio->member_size("output_proj_dims") == 1536 &&
        audio->member_size("conv_kernel_size") == 5 &&
        audio->member_size("attention_chunk_size") == 12 &&
        audio->member_size("attention_context_left") == 13 &&
        audio->member_double("attention_context_right", -1) == 0 &&
        audio->member_double("attention_invalid_logits_value") == -1e9 &&
        audio->member_double("attention_logit_cap") == 50 &&
        audio->member_double("gradient_clipping") == 1e10 &&
        audio->member_double("residual_weight") == 0.5 &&
        audio->member_double("rms_norm_eps") == 1e-6 && clipped &&
        clipped->is_bool() && clipped->as_bool() && channels &&
        channels->is_array() && channels->size() == 2 &&
        channels->items()[0].as_size() == 128 &&
        channels->items()[1].as_size() == 32 &&
        root.member_size("audio_token_id") == 258881 &&
        root.member_size("boa_token_id") == 256000 &&
        root.member_size("eoa_token_index") == 258883;
  }
  return config;
}

std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "config.json";
  const std::ifstream input(path);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  return ParseModelConfig(contents.str());
}

bool IsSupportedModelConfig(const ModelConfig& config) {
  const TextConfig& text = config.text;
  if (config.model_type != "embedding_gemma2" ||
      config.architecture != "EmbeddingGemma2Model" ||
      config.text_model_type != "embedding_gemma2_text" ||
      config.dtype != "bfloat16" ||
      config.hidden_activation != "gelu_pytorch_tanh" ||
      text.vocab_size != 262144 || text.hidden_size != 512 ||
      text.hidden_size_per_layer_input != 512 ||
      text.intermediate_size != 2048 || text.num_hidden_layers != 24 ||
      text.num_attention_heads != 4 || text.embedding_dim != 768 ||
      text.sliding_window != 512 || text.layers.size() != 24) {
    return false;
  }
  for (const LayerGeometry& layer : text.layers) {
    const bool full = layer.type == LayerType::kFullAttention;
    if (layer.head_dim != (full ? 512U : 256U) ||
        layer.num_key_value_heads != (full ? 1U : 2U) ||
        layer.rope_theta != (full ? 1000000.0F : 10000.0F)) {
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::embedding_gemma2
