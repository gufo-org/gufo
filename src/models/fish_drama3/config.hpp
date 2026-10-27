#ifndef GUFO_MODELS_FISH_DRAMA3_CONFIG_HPP_
#define GUFO_MODELS_FISH_DRAMA3_CONFIG_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::fish_drama3 {

struct BackboneConfig {
  std::uint32_t vocab_size = 252288;
  std::uint32_t hidden_size = 2048;
  std::uint32_t intermediate_size = 512;
  std::uint32_t shared_expert_intermediate_size = 512;
  std::uint32_t num_experts = 256;
  std::uint32_t num_experts_per_tok = 8;
  std::uint32_t num_hidden_layers = 40;
  std::uint32_t num_attention_heads = 16;
  std::uint32_t num_key_value_heads = 2;
  std::uint32_t head_dim = 256;
  std::uint32_t full_attention_interval = 4;
  std::uint32_t linear_conv_kernel_dim = 4;
  std::uint32_t linear_key_head_dim = 128;
  std::uint32_t linear_num_key_heads = 16;
  std::uint32_t linear_num_value_heads = 32;
  std::uint32_t linear_value_head_dim = 128;
  float rms_norm_eps = 1.0e-6F;
  float rope_theta = 10000000.0F;
  float partial_rotary_factor = 0.25F;
  bool layernorm_zero_centered_gamma = true;
  bool tie_word_embeddings = false;
  std::uint32_t lm_head_rows = 0;
  std::vector<std::uint32_t> experts_per_layer;

  [[nodiscard]] bool IsFullAttentionLayer(std::uint32_t layer) const noexcept {
    return full_attention_interval > 0 &&
           ((layer + 1) % full_attention_interval) == 0;
  }
  [[nodiscard]] std::uint32_t RotaryDim() const noexcept {
    return static_cast<std::uint32_t>(head_dim * partial_rotary_factor + 0.5F);
  }
  [[nodiscard]] std::uint32_t LayerExperts(std::uint32_t layer) const noexcept {
    return layer < experts_per_layer.size() ? experts_per_layer[layer]
                                            : num_experts;
  }
};

struct TowerConfig {
  std::uint32_t hidden_size = 2048;
  std::uint32_t intermediate_size = 6144;
  std::uint32_t heads = 16;
  std::uint32_t kv_heads = 2;
  std::uint32_t head_dim = 256;
  std::uint32_t codebook_size = 4096;
  std::uint32_t shared_layers = 4;
  std::uint32_t residual_layers = 9;
  std::uint32_t backbone_codebooks = 1;
  std::int32_t semantic_begin = 248099;
  float norm_eps = 1.0e-6F;
  float rope_theta = 10000000.0F;
  float rotary_fraction = 0.25F;
  bool rotary_interleaved = true;
  bool layernorm_zero_centered_gamma = true;
  bool attention_output_gate = true;

  [[nodiscard]] std::uint32_t TotalLayers() const noexcept {
    return shared_layers + residual_layers;
  }
  [[nodiscard]] std::uint32_t RotaryDim() const noexcept {
    return static_cast<std::uint32_t>(head_dim * rotary_fraction + 0.5F);
  }
};

struct CodecConfig {
  std::uint32_t sample_rate = 44100;
  std::uint32_t semantic_codebook_size = 4096;
  std::uint32_t residual_codebook_size = 1024;
  std::uint32_t quantizer_codebooks = 9;
  std::uint32_t total_codebooks = 10;
  std::uint32_t codebook_dim = 8;
  std::uint32_t latent_dim = 1024;
  std::uint32_t frame_length = 2048;
};

struct ModelConfig {
  std::string model_type;
  std::string speech_template_version;
  std::int32_t eos_token_id = 248046;
  std::int32_t pad_token_id = 248044;
  std::int32_t audio_pad_token_id = 248076;
  std::int32_t semantic_start_token_id = 248099;
  std::int32_t semantic_end_token_id = 252194;

  BackboneConfig backbone;
  TowerConfig tower;
  CodecConfig codec;
};

[[nodiscard]] bool IsSupportedModelConfig(const ModelConfig& config);
[[nodiscard]] std::optional<ModelConfig> ParseModelConfig(
    const std::string& json_contents);
[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir);
[[nodiscard]] bool LooksLikeFishDrama3(const std::string& model_dir);

}  // namespace gufo::models::fish_drama3

#endif  // GUFO_MODELS_FISH_DRAMA3_CONFIG_HPP_
