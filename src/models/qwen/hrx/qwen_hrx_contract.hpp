#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_CONTRACT_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_CONTRACT_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "src/core/model_config.hpp"

namespace gufo::hrx {

/// Fixed shape contract baked into the prototype BF16 Qwen3.8-27B Loom
/// artifacts.
///
/// Instances can only be created from a validated ModelConfig so dispatch code
/// cannot drift from the GGUF metadata that selected the artifact set. Shape
/// compatibility does not imply that a GGUF tensor's encoded type or layout is
/// compatible with the BF16 artifact ABI.
class QwenHrxArtifactContract {
public:
  [[nodiscard]] static std::optional<QwenHrxArtifactContract> FromConfig(
      const core::ModelConfig& config, std::string* error_msg = nullptr) {
    const std::uint32_t attention_query_width = config.AttentionSize();
    const std::uint32_t attention_q_gate_width = 2 * attention_query_width;
    const std::uint32_t attention_key_width =
        config.num_key_value_heads * config.head_dim;
    const std::uint32_t attention_value_width = attention_key_width;
    const std::uint32_t attention_packed_projection_width =
        attention_q_gate_width + attention_key_width + attention_value_width;
    const std::uint32_t full_attention_layers =
        config.FullAttentionLayerCount();
    const std::uint32_t ssm_layers = config.num_layers - full_attention_layers;
    const std::uint32_t ssm_qkv_width = config.SsmQkvSize();

    const bool supported =
        config.IsValidQwen() && config.num_layers == 64 &&
        config.full_attention_interval == 4 && full_attention_layers == 16 &&
        ssm_layers == 48 && config.hidden_size == 5120 &&
        config.intermediate_size == 17408 && config.num_attention_heads == 24 &&
        config.num_key_value_heads == 4 && config.head_dim == 256 &&
        config.rotary_dim == 64 && attention_query_width == 6144 &&
        attention_q_gate_width == 12288 && attention_key_width == 1024 &&
        attention_value_width == 1024 &&
        attention_packed_projection_width == 14336 &&
        config.ssm_group_count == 16 && config.ssm_time_step_rank == 48 &&
        config.ssm_state_size == 128 && config.ssm_inner_size == 6144 &&
        config.ssm_conv_kernel == 4 && config.SsmValueSize() == 128 &&
        ssm_qkv_width == 10240 && config.vocab_size == 248320;
    if (!supported) {
      if (error_msg != nullptr) {
        *error_msg =
            "native HRX prototype artifacts require Qwen3.8-27B text: "
            "layers=64 (16 full-attention, 48 SSM), hidden=5120, ffn=17408, "
            "attention q+gate/k/v=12288/1024/1024 (diagnostic packed "
            "projection width=14336), q_heads=24, kv_heads=4, head_dim=256, "
            "rotary_dim=64, SSM key/value heads=16/48, key/value dims=128/128, "
            "SSM qkv/gate/alpha-beta=10240/6144/48, conv_kernel=4, "
            "vocab=248320";
      }
      return std::nullopt;
    }

    if (error_msg != nullptr) {
      error_msg->clear();
    }
    return QwenHrxArtifactContract{
        config.num_layers,
        full_attention_layers,
        ssm_layers,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rotary_dim,
        attention_query_width,
        attention_q_gate_width,
        attention_key_width,
        attention_value_width,
        attention_packed_projection_width,
        ssm_qkv_width,
        config.ssm_inner_size,
        config.ssm_group_count,
        config.ssm_time_step_rank,
        config.ssm_state_size,
        config.SsmValueSize(),
        config.ssm_conv_kernel,
        config.vocab_size,
    };
  }

  [[nodiscard]] static bool Supports(const core::ModelConfig& config,
                                     std::string* error_msg = nullptr) {
    return FromConfig(config, error_msg).has_value();
  }

  [[nodiscard]] constexpr std::uint32_t NumLayers() const noexcept {
    return num_layers_;
  }
  [[nodiscard]] constexpr std::uint32_t FullAttentionLayerCount()
      const noexcept {
    return full_attention_layer_count_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmLayerCount() const noexcept {
    return ssm_layer_count_;
  }
  [[nodiscard]] constexpr std::uint32_t HiddenSize() const noexcept {
    return hidden_size_;
  }
  [[nodiscard]] constexpr std::uint32_t FfnSize() const noexcept {
    return ffn_size_;
  }
  [[nodiscard]] constexpr std::uint32_t QHeadCount() const noexcept {
    return q_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t KvHeadCount() const noexcept {
    return kv_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t HeadDim() const noexcept {
    return head_dim_;
  }
  [[nodiscard]] constexpr std::uint32_t RotaryDim() const noexcept {
    return rotary_dim_;
  }
  [[nodiscard]] constexpr std::uint32_t FullAttentionQueryWidth()
      const noexcept {
    return attention_query_width_;
  }
  [[nodiscard]] constexpr std::uint32_t FullAttentionQGateWidth()
      const noexcept {
    return attention_q_gate_width_;
  }
  [[nodiscard]] constexpr std::uint32_t FullAttentionKeyWidth() const noexcept {
    return attention_key_width_;
  }
  [[nodiscard]] constexpr std::uint32_t FullAttentionValueWidth()
      const noexcept {
    return attention_value_width_;
  }
  /// Sum of separate Q+gate, K, and V projection rows for diagnostics and
  /// allocation accounting only. No native artifact accepts a packed matrix.
  [[nodiscard]] constexpr std::uint32_t FullAttentionPackedProjectionWidth()
      const noexcept {
    return attention_packed_projection_width_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmQkvWidth() const noexcept {
    return ssm_qkv_width_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmGateWidth() const noexcept {
    return ssm_gate_width_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmKeyHeadCount() const noexcept {
    return ssm_key_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmValueHeadCount() const noexcept {
    return ssm_value_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmKeyDim() const noexcept {
    return ssm_key_dim_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmValueDim() const noexcept {
    return ssm_value_dim_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmAlphaBetaWidth() const noexcept {
    return ssm_value_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmConvKernel() const noexcept {
    return ssm_conv_kernel_;
  }
  [[nodiscard]] constexpr std::uint32_t VocabSize() const noexcept {
    return vocab_size_;
  }

private:
  constexpr QwenHrxArtifactContract(
      std::uint32_t num_layers, std::uint32_t full_attention_layer_count,
      std::uint32_t ssm_layer_count, std::uint32_t hidden_size,
      std::uint32_t ffn_size, std::uint32_t q_head_count,
      std::uint32_t kv_head_count, std::uint32_t head_dim,
      std::uint32_t rotary_dim, std::uint32_t attention_query_width,
      std::uint32_t attention_q_gate_width, std::uint32_t attention_key_width,
      std::uint32_t attention_value_width,
      std::uint32_t attention_packed_projection_width,
      std::uint32_t ssm_qkv_width, std::uint32_t ssm_gate_width,
      std::uint32_t ssm_key_head_count, std::uint32_t ssm_value_head_count,
      std::uint32_t ssm_key_dim, std::uint32_t ssm_value_dim,
      std::uint32_t ssm_conv_kernel, std::uint32_t vocab_size) noexcept
      : num_layers_(num_layers),
        full_attention_layer_count_(full_attention_layer_count),
        ssm_layer_count_(ssm_layer_count),
        hidden_size_(hidden_size),
        ffn_size_(ffn_size),
        q_head_count_(q_head_count),
        kv_head_count_(kv_head_count),
        head_dim_(head_dim),
        rotary_dim_(rotary_dim),
        attention_query_width_(attention_query_width),
        attention_q_gate_width_(attention_q_gate_width),
        attention_key_width_(attention_key_width),
        attention_value_width_(attention_value_width),
        attention_packed_projection_width_(attention_packed_projection_width),
        ssm_qkv_width_(ssm_qkv_width),
        ssm_gate_width_(ssm_gate_width),
        ssm_key_head_count_(ssm_key_head_count),
        ssm_value_head_count_(ssm_value_head_count),
        ssm_key_dim_(ssm_key_dim),
        ssm_value_dim_(ssm_value_dim),
        ssm_conv_kernel_(ssm_conv_kernel),
        vocab_size_(vocab_size) {}

  std::uint32_t num_layers_;
  std::uint32_t full_attention_layer_count_;
  std::uint32_t ssm_layer_count_;
  std::uint32_t hidden_size_;
  std::uint32_t ffn_size_;
  std::uint32_t q_head_count_;
  std::uint32_t kv_head_count_;
  std::uint32_t head_dim_;
  std::uint32_t rotary_dim_;
  std::uint32_t attention_query_width_;
  std::uint32_t attention_q_gate_width_;
  std::uint32_t attention_key_width_;
  std::uint32_t attention_value_width_;
  std::uint32_t attention_packed_projection_width_;
  std::uint32_t ssm_qkv_width_;
  std::uint32_t ssm_gate_width_;
  std::uint32_t ssm_key_head_count_;
  std::uint32_t ssm_value_head_count_;
  std::uint32_t ssm_key_dim_;
  std::uint32_t ssm_value_dim_;
  std::uint32_t ssm_conv_kernel_;
  std::uint32_t vocab_size_;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_CONTRACT_HPP_
