#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_CONFIG_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::embedding_gemma2 {

/// Square root of the hidden size, rounded to BF16 (sqrt(512) = 22.6274 becomes
/// 22.625) the way the checkpoint's BF16 execution does. The device kernels
/// also round the scaled embedding to BF16 as that execution does; the CPU
/// reference applies the same constant but keeps the product in float32.
inline constexpr float kEmbedScale = 22.625F;
inline constexpr std::uint32_t kImageTokenId = 258880;
inline constexpr std::uint32_t kAudioTokenId = 258881;
inline constexpr const char* kAudioToken = "<|audio|>";
inline constexpr const char* kBeginAudioToken = "<|audio>";
inline constexpr const char* kEndAudioToken = "<audio|>";
inline constexpr const char* kImageToken = "<|image|>";
inline constexpr const char* kBeginImageToken = "<|image>";
inline constexpr const char* kEndImageToken = "<image|>";

enum class LayerType : std::uint8_t {
  kSlidingAttention,
  kFullAttention,
};

/// Attention geometry of one decoder layer. Full-attention layers use a wider
/// head and fewer key/value heads than sliding-window layers.
struct LayerGeometry {
  LayerType type{LayerType::kSlidingAttention};
  std::uint32_t head_dim{256};
  std::uint32_t num_key_value_heads{2};
  float rope_theta{10000.0F};
};

/// The text backbone shared by text, image and audio inputs.
struct TextConfig {
  std::uint32_t vocab_size{262144};
  std::uint32_t hidden_size{512};
  std::uint32_t hidden_size_per_layer_input{512};
  std::uint32_t intermediate_size{2048};
  std::uint32_t num_hidden_layers{24};
  std::uint32_t num_attention_heads{4};
  std::uint32_t embedding_dim{768};
  /// Inclusive radius. A sliding layer attends to |query - key| <= radius,
  /// which is 1,025 keys for 512. Hugging Face's
  /// sliding_window_bidirectional_overlay in transformers/masking_utils.py
  /// returns abs(q_idx - kv_idx) <= sliding_window with config.sliding_window.
  std::uint32_t sliding_window{512};
  float rms_norm_eps{1.0e-6F};
  std::vector<LayerGeometry> layers;
};

struct ModelConfig {
  std::string model_type;
  std::string architecture;
  std::string dtype;
  std::string text_model_type;
  std::string hidden_activation;
  TextConfig text;
  bool supported_vision{false};
  bool supported_audio{false};
};

[[nodiscard]] std::optional<ModelConfig> ParseModelConfig(
    const std::string& json_contents);
[[nodiscard]] std::optional<ModelConfig> LoadModelConfigFromPath(
    const std::string& model_dir);

/// Accepts only google/embeddinggemma-2. It rejects unsupported geometry
/// instead of running it with the wrong kernels.
[[nodiscard]] bool IsSupportedModelConfig(const ModelConfig& config);

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_CONFIG_HPP_
