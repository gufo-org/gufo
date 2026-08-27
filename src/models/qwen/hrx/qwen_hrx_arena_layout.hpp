#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_LAYOUT_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>

#include "src/models/qwen/hrx/qwen_hrx_contract.hpp"

namespace gufo::hrx {

[[nodiscard]] inline std::optional<std::size_t> HrxCheckedProduct(
    std::initializer_list<std::size_t> factors) noexcept {
  std::size_t product = 1;
  for (const std::size_t factor : factors) {
    if (factor == 0 || product > std::numeric_limits<std::size_t>::max() / factor) {
      return std::nullopt;
    }
    product *= factor;
  }
  return product;
}

/// Exact byte layout for the sequential, single-token native-HRX MVP.
struct QwenHrxArenaLayout {
  std::uint32_t max_context{0};

  std::size_t hidden_bytes{0};
  std::size_t normed_bytes{0};

  std::size_t attention_q_gate_bytes{0};
  std::size_t attention_q_bytes{0};
  std::size_t attention_gate_bytes{0};
  std::size_t attention_k_bytes{0};
  std::size_t attention_v_bytes{0};
  std::size_t attention_output_bytes{0};
  std::size_t rope_cos_bytes{0};
  std::size_t rope_sin_bytes{0};

  std::size_t ssm_qkv_bytes{0};
  std::size_t ssm_gate_bytes{0};
  std::size_t ssm_alpha_bytes{0};
  std::size_t ssm_beta_bytes{0};
  std::size_t ssm_conv_output_bytes{0};
  std::size_t ssm_recurrent_output_bytes{0};

  std::size_t ffn_gate_bytes{0};
  std::size_t ffn_up_bytes{0};
  std::size_t ffn_activation_bytes{0};
  std::size_t ffn_output_bytes{0};

  std::size_t logits_bytes{0};
  std::size_t token_bytes{0};
  std::size_t position_bytes{0};
  std::size_t kv_cache_bytes{0};
  std::size_t ssm_conv_state_bytes{0};
  std::size_t ssm_recurrent_state_bytes{0};
  std::size_t saved_ssm_conv_state_bytes{0};
  std::size_t saved_ssm_recurrent_state_bytes{0};

  [[nodiscard]] static std::optional<QwenHrxArenaLayout> Create(
      const QwenHrxArtifactContract& contract, std::uint32_t max_context,
      std::string* error_msg = nullptr) {
    const auto bytes = [](std::initializer_list<std::size_t> dimensions) {
      return HrxCheckedProduct(dimensions);
    };
    const auto hidden = bytes({contract.HiddenSize(), sizeof(float)});
    const auto q_gate =
        bytes({contract.FullAttentionQGateWidth(), sizeof(float)});
    const auto q = bytes({contract.FullAttentionQueryWidth(), sizeof(float)});
    const auto k = bytes({contract.FullAttentionKeyWidth(), sizeof(float)});
    const auto v = bytes({contract.FullAttentionValueWidth(), sizeof(float)});
    const auto rope = bytes({contract.RotaryDim() / 2, sizeof(float)});
    const auto ssm_qkv = bytes({contract.SsmQkvWidth(), sizeof(float)});
    const auto ssm_gate = bytes({contract.SsmGateWidth(), sizeof(float)});
    const auto ssm_ab = bytes({contract.SsmAlphaBetaWidth(), sizeof(float)});
    const auto ssm_recurrent_output = bytes(
        {contract.SsmValueHeadCount(), contract.SsmValueDim(), sizeof(float)});
    const auto ffn = bytes({contract.FfnSize(), sizeof(float)});
    const auto logits = bytes({contract.VocabSize(), sizeof(float)});
    const auto token = bytes({sizeof(std::uint32_t)});
    // Only full-attention layers own KV rows. SSM layers keep independent
    // convolution and recurrent state below.
    const auto kv_cache =
        bytes({contract.FullAttentionLayerCount(), 2U,
               contract.KvHeadCount(), max_context, contract.HeadDim(),
               sizeof(float)});
    const auto conv_state =
        bytes({contract.NumLayers(), contract.SsmQkvWidth(),
               contract.SsmConvKernel(), sizeof(float)});
    const auto recurrent_state =
        bytes({contract.NumLayers(), contract.SsmValueHeadCount(),
               contract.SsmKeyDim(), contract.SsmValueDim(), sizeof(float)});

    if (max_context == 0 || !hidden || !q_gate || !q || !k || !v || !rope ||
        !ssm_qkv || !ssm_gate || !ssm_ab || !ssm_recurrent_output || !ffn ||
        !logits || !token || !kv_cache || !conv_state || !recurrent_state) {
      if (error_msg != nullptr) {
        *error_msg =
            "native HRX arena dimensions are zero or overflow size_t";
      }
      return std::nullopt;
    }
    if (error_msg != nullptr) {
      error_msg->clear();
    }

    return QwenHrxArenaLayout{
        .max_context = max_context,
        .hidden_bytes = *hidden,
        .normed_bytes = *hidden,
        .attention_q_gate_bytes = *q_gate,
        .attention_q_bytes = *q,
        .attention_gate_bytes = *q,
        .attention_k_bytes = *k,
        .attention_v_bytes = *v,
        .attention_output_bytes = *hidden,
        .rope_cos_bytes = *rope,
        .rope_sin_bytes = *rope,
        .ssm_qkv_bytes = *ssm_qkv,
        .ssm_gate_bytes = *ssm_gate,
        .ssm_alpha_bytes = *ssm_ab,
        .ssm_beta_bytes = *ssm_ab,
        .ssm_conv_output_bytes = *ssm_qkv,
        .ssm_recurrent_output_bytes = *ssm_recurrent_output,
        .ffn_gate_bytes = *ffn,
        .ffn_up_bytes = *ffn,
        .ffn_activation_bytes = *ffn,
        .ffn_output_bytes = *hidden,
        .logits_bytes = *logits,
        .token_bytes = *token,
        .position_bytes = *token,
        .kv_cache_bytes = *kv_cache,
        .ssm_conv_state_bytes = *conv_state,
        .ssm_recurrent_state_bytes = *recurrent_state,
        .saved_ssm_conv_state_bytes = *conv_state,
        .saved_ssm_recurrent_state_bytes = *recurrent_state,
    };
  }
};

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_ARENA_LAYOUT_HPP_
