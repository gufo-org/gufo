#include "src/models/qwen3_tts/config.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <sstream>
#include <string>
#include <string_view>

namespace gufo::models::qwen3_tts {
namespace {

namespace json = gufo::json;

const json::Value* FindPath(const json::Value& root,
                            std::initializer_list<std::string_view> path) {
  const json::Value* current = &root;
  for (const std::string_view key : path) {
    if (current == nullptr || !current->is_object()) {
      return nullptr;
    }
    current = current->find(std::string(key));
  }
  return current;
}

void ParseU32(const json::Value& object, std::string_view key,
              std::uint32_t& out) {
  if (!object.is_object()) {
    return;
  }
  const json::Value* value = object.find(std::string(key));
  if (value != nullptr && value->is_number()) {
    out = static_cast<std::uint32_t>(value->as_size(out));
  }
}

std::vector<std::uint32_t> ParseU32Array(const json::Value& value) {
  std::vector<std::uint32_t> out;
  if (!value.is_array()) {
    return out;
  }
  for (const auto& item : value.items()) {
    if (item.is_number()) {
      out.push_back(static_cast<std::uint32_t>(item.as_size()));
    }
  }
  return out;
}

std::map<std::string, std::uint32_t> ParseStringU32Map(
    const json::Value& value) {
  std::map<std::string, std::uint32_t> out;
  if (!value.is_object()) {
    return out;
  }
  for (const auto& [key, member] : value.members()) {
    if (member.is_number()) {
      out[key] = static_cast<std::uint32_t>(member.as_size());
    }
  }
  return out;
}

std::map<std::string, std::optional<std::string>> ParseSpeakerDialects(
    const json::Value& value) {
  std::map<std::string, std::optional<std::string>> out;
  if (!value.is_object()) {
    return out;
  }
  for (const auto& [key, member] : value.members()) {
    if (member.is_bool()) {
      if (!member.as_bool()) {
        out.emplace(key, std::nullopt);
      }
    } else if (member.is_string()) {
      out.emplace(key, member.str());
    }
  }
  return out;
}

void ParseTalker(TalkerConfig& out, const json::Value& talker) {
  ParseU32(talker, "vocab_size", out.vocab_size);
  ParseU32(talker, "text_vocab_size", out.text_vocab_size);
  ParseU32(talker, "hidden_size", out.hidden_size);
  ParseU32(talker, "intermediate_size", out.intermediate_size);
  ParseU32(talker, "num_hidden_layers", out.num_hidden_layers);
  ParseU32(talker, "num_attention_heads", out.num_attention_heads);
  ParseU32(talker, "num_key_value_heads", out.num_key_value_heads);
  ParseU32(talker, "head_dim", out.head_dim);
  ParseU32(talker, "num_code_groups", out.num_code_groups);
  ParseU32(talker, "text_hidden_size", out.text_hidden_size);
  ParseU32(talker, "max_position_embeddings", out.max_position_embeddings);
  ParseU32(talker, "codec_eos_token_id", out.codec_eos_token_id);
  ParseU32(talker, "codec_bos_id", out.codec_bos_id);
  ParseU32(talker, "codec_pad_id", out.codec_pad_id);
  ParseU32(talker, "codec_think_id", out.codec_think_id);
  ParseU32(talker, "codec_nothink_id", out.codec_nothink_id);
  ParseU32(talker, "codec_think_bos_id", out.codec_think_bos_id);
  ParseU32(talker, "codec_think_eos_id", out.codec_think_eos_id);
  if (const json::Value* eps = talker.find("rms_norm_eps");
      eps != nullptr && eps->is_number()) {
    out.rms_norm_eps = static_cast<float>(eps->as_double(out.rms_norm_eps));
  }
  if (const json::Value* theta = talker.find("rope_theta");
      theta != nullptr && theta->is_number()) {
    out.rope_theta = static_cast<float>(theta->as_double(out.rope_theta));
  }
  if (const auto* speaker = talker.find("spk_id")) {
    out.spk_id = ParseStringU32Map(*speaker);
  }
  if (const auto* lang = talker.find("codec_language_id")) {
    out.codec_language_id = ParseStringU32Map(*lang);
  }
  if (const auto* dialect = talker.find("spk_is_dialect")) {
    out.spk_dialect = ParseSpeakerDialects(*dialect);
  }
  if (const json::Value* rope = talker.find("rope_scaling");
      rope != nullptr && rope->is_object()) {
    if (const json::Value* section = rope->find("mrope_section")) {
      auto parsed = ParseU32Array(*section);
      if (!parsed.empty()) {
        out.mrope_section = std::move(parsed);
      }
    }
    if (const json::Value* interleaved = rope->find("interleaved");
        interleaved != nullptr && interleaved->is_bool()) {
      out.mrope_interleaved = interleaved->as_bool();
    }
  }
}

void ParseCodePredictor(CodePredictorConfig& out, const json::Value& config) {
  ParseU32(config, "vocab_size", out.vocab_size);
  ParseU32(config, "hidden_size", out.hidden_size);
  ParseU32(config, "intermediate_size", out.intermediate_size);
  ParseU32(config, "num_hidden_layers", out.num_hidden_layers);
  ParseU32(config, "num_attention_heads", out.num_attention_heads);
  ParseU32(config, "num_key_value_heads", out.num_key_value_heads);
  ParseU32(config, "head_dim", out.head_dim);
  ParseU32(config, "num_code_groups", out.num_code_groups);
  ParseU32(config, "max_position_embeddings", out.max_position_embeddings);
  if (const json::Value* eps = config.find("rms_norm_eps");
      eps != nullptr && eps->is_number()) {
    out.rms_norm_eps = static_cast<float>(eps->as_double(out.rms_norm_eps));
  }
  if (const json::Value* theta = config.find("rope_theta");
      theta != nullptr && theta->is_number()) {
    out.rope_theta = static_cast<float>(theta->as_double(out.rope_theta));
  }
}

void ParseSpeechEncoder(SpeechTokenizerConfig& out,
                        const json::Value& encoder) {
  ParseU32(encoder, "audio_channels", out.encoder_audio_channels);
  ParseU32(encoder, "codebook_dim", out.encoder_codebook_dim);
  ParseU32(encoder, "codebook_size", out.encoder_codebook_size);
  ParseU32(encoder, "hidden_size", out.encoder_hidden_size);
  ParseU32(encoder, "intermediate_size", out.encoder_intermediate_size);
  ParseU32(encoder, "head_dim", out.encoder_head_dim);
  ParseU32(encoder, "num_attention_heads", out.encoder_num_attention_heads);
  ParseU32(encoder, "num_hidden_layers", out.encoder_num_hidden_layers);
  ParseU32(encoder, "num_key_value_heads", out.encoder_num_key_value_heads);
  ParseU32(encoder, "num_quantizers", out.encoder_num_quantizers);
  ParseU32(encoder, "num_semantic_quantizers",
           out.encoder_num_semantic_quantizers);
  ParseU32(encoder, "num_filters", out.encoder_num_filters);
  ParseU32(encoder, "num_residual_layers", out.encoder_num_residual_layers);
  ParseU32(encoder, "kernel_size", out.encoder_kernel_size);
  ParseU32(encoder, "last_kernel_size", out.encoder_last_kernel_size);
  ParseU32(encoder, "residual_kernel_size", out.encoder_residual_kernel_size);
  ParseU32(encoder, "compress", out.encoder_compress);
  ParseU32(encoder, "dilation_growth_rate", out.encoder_dilation_growth_rate);
  ParseU32(encoder, "sliding_window", out.encoder_sliding_window);
  if (const json::Value* scale = encoder.find("layer_scale_initial_scale");
      scale != nullptr && scale->is_number()) {
    out.encoder_layer_scale_initial_scale = static_cast<float>(
        scale->as_double(out.encoder_layer_scale_initial_scale));
  }
  if (const json::Value* eps = encoder.find("norm_eps");
      eps != nullptr && eps->is_number()) {
    out.encoder_norm_eps =
        static_cast<float>(eps->as_double(out.encoder_norm_eps));
  }
  if (const json::Value* theta = encoder.find("rope_theta");
      theta != nullptr && theta->is_number()) {
    out.encoder_rope_theta =
        static_cast<float>(theta->as_double(out.encoder_rope_theta));
  }
  if (const json::Value* ratios = encoder.find("upsampling_ratios")) {
    auto parsed = ParseU32Array(*ratios);
    if (!parsed.empty()) {
      out.encoder_upsampling_ratios = std::move(parsed);
    }
  }
}

}  // namespace

ModelVariant ParseModelVariant(std::string_view model_type) {
  if (model_type == "base") {
    return ModelVariant::kBase;
  }
  if (model_type == "voice_design") {
    return ModelVariant::kVoiceDesign;
  }
  if (model_type == "custom_voice") {
    return ModelVariant::kCustomVoice;
  }
  return ModelVariant::kUnsupported;
}

bool IsSupportedModelConfig(const ModelConfig& config) {
  if (config.model_type != "qwen3_tts" ||
      config.tokenizer_type != "qwen3_tts_tokenizer_12hz" ||
      config.tts_model_size != "1b7" ||
      config.variant == ModelVariant::kUnsupported) {
    return false;
  }
  return config.variant != ModelVariant::kBase ||
         config.speaker_encoder.has_value();
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
  config.tokenizer_type = root.member_str("tokenizer_type");
  config.tts_model_size = root.member_str("tts_model_size");
  config.tts_model_type = root.member_str("tts_model_type");
  config.variant = ParseModelVariant(config.tts_model_type);
  ParseU32(root, "im_start_token_id", config.im_start_token_id);
  ParseU32(root, "im_end_token_id", config.im_end_token_id);
  ParseU32(root, "tts_pad_token_id", config.tts_pad_token_id);
  ParseU32(root, "tts_bos_token_id", config.tts_bos_token_id);
  ParseU32(root, "tts_eos_token_id", config.tts_eos_token_id);

  if (const auto* talker = FindPath(root, {"talker_config"})) {
    ParseTalker(config.talker, *talker);
    if (const auto* predictor =
            FindPath(root, {"talker_config", "code_predictor_config"})) {
      ParseCodePredictor(config.code_predictor, *predictor);
    }
  }
  if (const auto* speaker = FindPath(root, {"speaker_encoder_config"});
      speaker != nullptr && speaker->is_object()) {
    SpeakerEncoderConfig parsed;
    ParseU32(*speaker, "enc_dim", parsed.embedding_dim);
    ParseU32(*speaker, "sample_rate", parsed.sample_rate);
    config.speaker_encoder = parsed;
  }
  return config;
}

std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir) {
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "config.json";
  const std::ifstream stream(path);
  if (!stream) {
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  auto config = ParseModelConfig(buffer.str());
  if (config.has_value()) {
    (void)LoadSpeechTokenizerConfigFromPath(model_dir,
                                            &config->speech_tokenizer);
  }
  return config;
}

bool LoadSpeechTokenizerConfigFromPath(const std::string& model_dir,
                                       SpeechTokenizerConfig* config) {
  if (config == nullptr) {
    return false;
  }
  const std::filesystem::path path =
      std::filesystem::path(model_dir) / "speech_tokenizer" / "config.json";
  const std::ifstream stream(path);
  if (!stream) {
    return false;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  try {
    const json::Value root = json::parse(buffer.str());
    if (!root.is_object()) {
      return false;
    }
    config->model_type = root.member_str("model_type");
    ParseU32(root, "input_sample_rate", config->input_sample_rate);
    ParseU32(root, "output_sample_rate", config->output_sample_rate);
    ParseU32(root, "decode_upsample_rate", config->decode_upsample_rate);
    ParseU32(root, "encode_downsample_rate", config->encode_downsample_rate);
    ParseU32(root, "encoder_valid_num_quantizers",
             config->encoder_valid_num_quantizers);
    if (const json::Value* decoder = root.find("decoder_config");
        decoder != nullptr && decoder->is_object()) {
      ParseU32(*decoder, "latent_dim", config->latent_dim);
      ParseU32(*decoder, "codebook_dim", config->codebook_dim);
      ParseU32(*decoder, "codebook_size", config->codebook_size);
      ParseU32(*decoder, "decoder_dim", config->decoder_dim);
      ParseU32(*decoder, "hidden_size", config->hidden_size);
      ParseU32(*decoder, "intermediate_size", config->intermediate_size);
      ParseU32(*decoder, "head_dim", config->head_dim);
      ParseU32(*decoder, "num_attention_heads", config->num_attention_heads);
      ParseU32(*decoder, "num_hidden_layers", config->num_hidden_layers);
      ParseU32(*decoder, "num_key_value_heads", config->num_key_value_heads);
      ParseU32(*decoder, "num_quantizers", config->num_quantizers);
      ParseU32(*decoder, "num_semantic_quantizers",
               config->num_semantic_quantizers);
      ParseU32(*decoder, "sliding_window", config->sliding_window);
      if (const json::Value* scale = decoder->find("layer_scale_initial_scale");
          scale != nullptr && scale->is_number()) {
        config->layer_scale_initial_scale = static_cast<float>(
            scale->as_double(config->layer_scale_initial_scale));
      }
      if (const json::Value* eps = decoder->find("rms_norm_eps");
          eps != nullptr && eps->is_number()) {
        config->rms_norm_eps =
            static_cast<float>(eps->as_double(config->rms_norm_eps));
      }
      if (const json::Value* theta = decoder->find("rope_theta");
          theta != nullptr && theta->is_number()) {
        config->rope_theta =
            static_cast<float>(theta->as_double(config->rope_theta));
      }
      if (const json::Value* rates = decoder->find("upsample_rates")) {
        auto parsed = ParseU32Array(*rates);
        if (!parsed.empty()) {
          config->upsample_rates = std::move(parsed);
        }
      }
      if (const json::Value* ratios = decoder->find("upsampling_ratios")) {
        auto parsed = ParseU32Array(*ratios);
        if (!parsed.empty()) {
          config->upsampling_ratios = std::move(parsed);
        }
      }
    }
    if (const json::Value* encoder = root.find("encoder_config");
        encoder != nullptr && encoder->is_object()) {
      ParseSpeechEncoder(*config, *encoder);
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace gufo::models::qwen3_tts
