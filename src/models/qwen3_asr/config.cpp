#include "src/models/qwen3_asr/config.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "src/models/qwen3_tts/json.hpp"

namespace gufo::models::qwen3_asr {
namespace {

namespace json = gufo::models::qwen3_tts::json;

const json::Value* FindPath(const json::Value& root,
                            std::initializer_list<std::string_view> path) {
  const json::Value* current = &root;
  for (const std::string_view name : path) {
    if (current == nullptr || !current->is_object()) {
      return nullptr;
    }
    current = current->find(std::string(name));
  }
  return current;
}

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

std::vector<std::uint32_t> ParseU32Array(const json::Value& value) {
  std::vector<std::uint32_t> result;
  if (!value.is_array()) {
    return result;
  }
  result.reserve(value.size());
  for (const auto& item : value.items()) {
    if (item.is_number()) {
      result.push_back(static_cast<std::uint32_t>(item.as_size()));
    }
  }
  return result;
}

void ParseAudioConfig(const json::Value& value, AudioEncoderConfig* output) {
  ParseU32(value, "num_mel_bins", &output->num_mel_bins);
  ParseU32(value, "d_model", &output->d_model);
  ParseU32(value, "encoder_layers", &output->encoder_layers);
  ParseU32(value, "encoder_attention_heads", &output->encoder_attention_heads);
  ParseU32(value, "encoder_ffn_dim", &output->encoder_ffn_dim);
  ParseU32(value, "max_source_positions", &output->max_source_positions);
  ParseU32(value, "n_window", &output->n_window);
  ParseU32(value, "n_window_infer", &output->n_window_infer);
  ParseU32(value, "conv_chunksize", &output->conv_chunksize);
  ParseU32(value, "downsample_hidden_size", &output->downsample_hidden_size);
  ParseU32(value, "output_dim", &output->output_dim);
  ParseFloat(value, "attention_dropout", &output->attention_dropout);
}

void ParseTextConfig(const json::Value& value, TextConfig* output) {
  ParseU32(value, "vocab_size", &output->vocab_size);
  ParseU32(value, "hidden_size", &output->hidden_size);
  ParseU32(value, "intermediate_size", &output->intermediate_size);
  ParseU32(value, "num_hidden_layers", &output->num_hidden_layers);
  ParseU32(value, "num_attention_heads", &output->num_attention_heads);
  ParseU32(value, "num_key_value_heads", &output->num_key_value_heads);
  ParseU32(value, "head_dim", &output->head_dim);
  ParseU32(value, "max_position_embeddings", &output->max_position_embeddings);
  ParseFloat(value, "rms_norm_eps", &output->rms_norm_eps);
  ParseFloat(value, "rope_theta", &output->rope_theta);
  if (const json::Value* rope = value.find("rope_scaling");
      rope != nullptr && rope->is_object()) {
    if (const json::Value* section = rope->find("mrope_section")) {
      auto parsed = ParseU32Array(*section);
      if (!parsed.empty()) {
        output->mrope_section = std::move(parsed);
      }
    }
    if (const json::Value* interleaved = rope->find("mrope_interleaved");
        interleaved != nullptr && interleaved->is_bool()) {
      output->mrope_interleaved = interleaved->as_bool();
    } else if (const json::Value* interleaved = rope->find("interleaved");
               interleaved != nullptr && interleaved->is_bool()) {
      output->mrope_interleaved = interleaved->as_bool();
    }
  }
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
  if (const json::Value* architectures = root.find("architectures");
      architectures != nullptr && architectures->is_array() &&
      !architectures->items().empty() &&
      architectures->items().front().is_string()) {
    config.architecture = architectures->items().front().str();
  }
  if (const json::Value* languages = root.find("support_languages");
      languages != nullptr && languages->is_array()) {
    for (const auto& language : languages->items()) {
      if (language.is_string()) {
        config.supported_languages.push_back(language.str());
      }
    }
  }

  const json::Value* thinker = FindPath(root, {"thinker_config"});
  if (thinker == nullptr || !thinker->is_object()) {
    return std::nullopt;
  }
  config.dtype = thinker->member_str("dtype");
  ParseU32(*thinker, "audio_start_token_id", &config.audio_start_token_id);
  ParseU32(*thinker, "audio_end_token_id", &config.audio_end_token_id);
  ParseU32(*thinker, "audio_token_id", &config.audio_token_id);

  const json::Value* audio = FindPath(root, {"thinker_config", "audio_config"});
  const json::Value* text = FindPath(root, {"thinker_config", "text_config"});
  if (audio == nullptr || text == nullptr || !audio->is_object() ||
      !text->is_object()) {
    return std::nullopt;
  }
  ParseAudioConfig(*audio, &config.audio);
  ParseTextConfig(*text, &config.text);
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
  const AudioEncoderConfig& audio = config.audio;
  const TextConfig& text = config.text;
  return config.model_type == "qwen3_asr" &&
         config.architecture == "Qwen3ASRForConditionalGeneration" &&
         config.dtype == "bfloat16" && audio.num_mel_bins == 128 &&
         audio.d_model == 1024 && audio.encoder_layers == 24 &&
         audio.encoder_attention_heads == 16 && audio.encoder_ffn_dim == 4096 &&
         audio.max_source_positions == 1500 && audio.n_window == 50 &&
         audio.n_window_infer == 800 && audio.conv_chunksize == 500 &&
         audio.downsample_hidden_size == 480 && audio.output_dim == 2048 &&
         text.vocab_size == 151936 && text.hidden_size == 2048 &&
         text.intermediate_size == 6144 && text.num_hidden_layers == 28 &&
         text.num_attention_heads == 16 && text.num_key_value_heads == 8 &&
         text.head_dim == 128 && text.max_position_embeddings == 65536 &&
         text.mrope_section == std::vector<std::uint32_t>({24, 20, 20}) &&
         text.mrope_interleaved && config.audio_start_token_id == 151669 &&
         config.audio_end_token_id == 151670 && config.audio_token_id == 151676;
}

}  // namespace gufo::models::qwen3_asr
