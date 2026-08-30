#include "src/models/qwen3_asr/config.hpp"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace qwen3_asr = gufo::models::qwen3_asr;

namespace {

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL qwen3_asr_config_test: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::string_view kConfig = R"({
  "architectures": ["Qwen3ASRForConditionalGeneration"],
  "model_type": "qwen3_asr",
  "support_languages": ["Chinese", "English"],
  "thinker_config": {
    "dtype": "bfloat16",
    "audio_start_token_id": 151669,
    "audio_end_token_id": 151670,
    "audio_token_id": 151676,
    "audio_config": {
      "num_mel_bins": 128,
      "d_model": 1024,
      "encoder_layers": 24,
      "encoder_attention_heads": 16,
      "encoder_ffn_dim": 4096,
      "max_source_positions": 1500,
      "n_window": 50,
      "n_window_infer": 800,
      "conv_chunksize": 500,
      "downsample_hidden_size": 480,
      "output_dim": 2048
    },
    "text_config": {
      "vocab_size": 151936,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "num_hidden_layers": 28,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "max_position_embeddings": 65536,
      "rms_norm_eps": 0.000001,
      "rope_theta": 1000000,
      "rope_scaling": {
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": true
      }
    }
  }
})";

}  // namespace

int main() {
  const auto config = qwen3_asr::ParseModelConfig(std::string(kConfig));
  Check(config.has_value(), "valid 1.7B config must parse");
  Check(qwen3_asr::IsSupportedModelConfig(*config),
        "exact 1.7B config must be accepted");
  Check(config->audio.encoder_layers == 24 && config->audio.d_model == 1024 &&
            config->text.num_hidden_layers == 28 &&
            config->text.hidden_size == 2048,
        "architecture dimensions must be retained");
  Check(config->supported_languages.size() == 2,
        "supported language inventory must parse");

  auto small = *config;
  small.text.hidden_size = 1024;
  Check(!qwen3_asr::IsSupportedModelConfig(small),
        "non-1.7B text geometry must be rejected");
  auto aligner = *config;
  aligner.architecture = "Qwen3ForcedAlignerForConditionalGeneration";
  Check(!qwen3_asr::IsSupportedModelConfig(aligner),
        "forced aligner must be rejected");
  Check(!qwen3_asr::ParseModelConfig("{").has_value(),
        "malformed JSON must fail");

  std::cout << "PASS qwen3_asr_config_test\n";
  return 0;
}
