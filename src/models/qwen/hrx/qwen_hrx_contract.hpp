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
    const std::uint32_t q_width = config.AttentionSize();
    const std::uint32_t k_width =
        config.num_key_value_heads * config.head_dim;
    const std::uint32_t v_width = k_width;
    const std::uint32_t qkv_width = q_width + k_width + v_width;
    const std::uint32_t ssm_qkv_width = config.SsmQkvSize();

    const bool supported =
        config.IsValidQwen() && config.num_layers == 64 &&
        config.hidden_size == 5120 && config.intermediate_size == 17408 &&
        config.num_attention_heads == 24 &&
        config.num_key_value_heads == 4 && config.head_dim == 256 &&
        config.rotary_dim == 64 && q_width == 6144 && k_width == 1024 &&
        v_width == 1024 && qkv_width == 8192 &&
        config.ssm_group_count == 16 && config.ssm_state_size == 128 &&
        config.ssm_inner_size == 6144 && config.SsmValueSize() == 128 &&
        ssm_qkv_width == 10240 && config.vocab_size == 248320;
    if (!supported) {
      if (error_msg != nullptr) {
        *error_msg =
            "native HRX artifacts require Qwen3.8-27B text: layers=64, "
            "hidden=5120, ffn=17408, q_heads=24, kv_heads=4, head_dim=256, "
            "rotary_dim=64, q/k/v=6144/1024/1024, qkv=8192, "
            "ssm_qkv=10240, vocab=248320";
      }
      return std::nullopt;
    }

    if (error_msg != nullptr) {
      error_msg->clear();
    }
    return QwenHrxArtifactContract{
        config.num_layers,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.rotary_dim,
        q_width,
        k_width,
        v_width,
        qkv_width,
        ssm_qkv_width,
        config.ssm_group_count,
        config.ssm_state_size,
        config.SsmValueSize(),
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
  [[nodiscard]] constexpr std::uint32_t QWidth() const noexcept {
    return q_width_;
  }
  [[nodiscard]] constexpr std::uint32_t KWidth() const noexcept {
    return k_width_;
  }
  [[nodiscard]] constexpr std::uint32_t VWidth() const noexcept {
    return v_width_;
  }
  [[nodiscard]] constexpr std::uint32_t QkvWidth() const noexcept {
    return qkv_width_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmQkvWidth() const noexcept {
    return ssm_qkv_width_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmHeadCount() const noexcept {
    return ssm_head_count_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmStateSize() const noexcept {
    return ssm_state_size_;
  }
  [[nodiscard]] constexpr std::uint32_t SsmValueSize() const noexcept {
    return ssm_value_size_;
  }
  [[nodiscard]] constexpr std::uint32_t VocabSize() const noexcept {
    return vocab_size_;
  }

private:
  constexpr QwenHrxArtifactContract(
      std::uint32_t num_layers, std::uint32_t hidden_size,
      std::uint32_t ffn_size, std::uint32_t q_head_count,
      std::uint32_t kv_head_count, std::uint32_t head_dim,
      std::uint32_t rotary_dim, std::uint32_t q_width,
      std::uint32_t k_width, std::uint32_t v_width,
      std::uint32_t qkv_width, std::uint32_t ssm_qkv_width,
      std::uint32_t ssm_head_count, std::uint32_t ssm_state_size,
      std::uint32_t ssm_value_size, std::uint32_t vocab_size) noexcept
      : num_layers_(num_layers),
        hidden_size_(hidden_size),
        ffn_size_(ffn_size),
        q_head_count_(q_head_count),
        kv_head_count_(kv_head_count),
        head_dim_(head_dim),
        rotary_dim_(rotary_dim),
        q_width_(q_width),
        k_width_(k_width),
        v_width_(v_width),
        qkv_width_(qkv_width),
        ssm_qkv_width_(ssm_qkv_width),
        ssm_head_count_(ssm_head_count),
        ssm_state_size_(ssm_state_size),
        ssm_value_size_(ssm_value_size),
        vocab_size_(vocab_size) {}

  std::uint32_t num_layers_;
  std::uint32_t hidden_size_;
  std::uint32_t ffn_size_;
  std::uint32_t q_head_count_;
  std::uint32_t kv_head_count_;
  std::uint32_t head_dim_;
  std::uint32_t rotary_dim_;
  std::uint32_t q_width_;
  std::uint32_t k_width_;
  std::uint32_t v_width_;
  std::uint32_t qkv_width_;
  std::uint32_t ssm_qkv_width_;
  std::uint32_t ssm_head_count_;
  std::uint32_t ssm_state_size_;
  std::uint32_t ssm_value_size_;
  std::uint32_t vocab_size_;
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_CONTRACT_HPP_
