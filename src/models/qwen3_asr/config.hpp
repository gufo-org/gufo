#ifndef GUFO_MODELS_QWEN3_ASR_CONFIG_HPP_
#define GUFO_MODELS_QWEN3_ASR_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::qwen3_asr {

struct AudioEncoderConfig {
  std::uint32_t num_mel_bins{128};
  std::uint32_t d_model{1024};
  std::uint32_t encoder_layers{24};
  std::uint32_t encoder_attention_heads{16};
  std::uint32_t encoder_ffn_dim{4096};
  std::uint32_t max_source_positions{1500};
  std::uint32_t n_window{50};
  std::uint32_t n_window_infer{800};
  std::uint32_t conv_chunksize{500};
  std::uint32_t downsample_hidden_size{480};
  std::uint32_t output_dim{2048};
  float attention_dropout{0.0F};
};

struct TextConfig {
  std::uint32_t vocab_size{151936};
  std::uint32_t hidden_size{2048};
  std::uint32_t intermediate_size{6144};
  std::uint32_t num_hidden_layers{28};
  std::uint32_t num_attention_heads{16};
  std::uint32_t num_key_value_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t max_position_embeddings{65536};
  float rms_norm_eps{1.0e-6F};
  float rope_theta{1000000.0F};
  std::vector<std::uint32_t> mrope_section{24, 20, 20};
  bool mrope_interleaved{true};
};

struct ModelConfig {
  std::string model_type;
  std::string architecture;
  std::string dtype;
  AudioEncoderConfig audio;
  TextConfig text;
  std::uint32_t audio_start_token_id{151669};
  std::uint32_t audio_end_token_id{151670};
  std::uint32_t audio_token_id{151676};
  std::uint32_t im_start_token_id{151644};
  std::uint32_t im_end_token_id{151645};
  std::uint32_t endoftext_token_id{151643};
  std::uint32_t asr_text_token_id{151704};
  std::vector<std::string> supported_languages;
};

[[nodiscard]] std::optional<ModelConfig> ParseModelConfig(
    const std::string& json_contents);
[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir);

/// Accepts only Qwen3-ASR-1.7B. The 0.6B and forced-aligner checkpoints are
/// intentionally outside this implementation.
[[nodiscard]] bool IsSupportedModelConfig(const ModelConfig& config);

}  // namespace gufo::models::qwen3_asr

#endif  // GUFO_MODELS_QWEN3_ASR_CONFIG_HPP_
