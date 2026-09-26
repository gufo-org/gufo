#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/models/qwen/state.hpp"

namespace gufo::models {
namespace {

QwenTensorRef ExtractTensorRef(const core::GgufReader& reader,
                               std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }

  const auto tensor_address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available_bytes = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (tensor_address >= region_address) {
      const auto offset = tensor_address - region_address;
      if (offset < region.size) {
        available_bytes = region.size - offset;
        break;
      }
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available_bytes};
}

enum class TensorRole : std::uint8_t {
  kEmbedding,
  kNorm,
  kProjection,
  kSsmParameter,
};

[[nodiscard]] constexpr std::string_view TensorRoleName(
    TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      return "embedding";
    case TensorRole::kNorm:
      return "normalization";
    case TensorRole::kProjection:
      return "projection";
    case TensorRole::kSsmParameter:
      return "SSM parameter";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool SupportsTensorType(core::GgmlType type,
                                                TensorRole role) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      // The production HIP embedding kernel has exact F32, BF16, and Q8_0
      // implementations. Treating every other type as Q8_0 would decode the
      // wrong block layout.
      // opt-q4kxl: token_embd.weight is Q4_K in the UD-Q4_K_XL shard; the
      // lookup kernels decode any DecodeQuantSub16-backed format directly.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K ||
             type == core::GgmlType::kQ3_K || type == core::GgmlType::kQ5_K ||
             type == core::GgmlType::kQ6_K || type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
    case TensorRole::kNorm:
    case TensorRole::kSsmParameter:
      // HIP norm, convolution, and recurrence kernels consume these tensors as
      // float pointers. Accepting BF16 or packed data would reinterpret bytes.
      return type == core::GgmlType::kF32;
    case TensorRole::kProjection:
      // opt-q4kxl: the Unsloth UD-Q4_K_XL shard mixes Q5_K, IQ4_XS, Q4_K, Q6_K,
      // IQ4_NL, Q3_K and IQ3_S across projections, and every one of them has an
      // in-kernel decoder, so all are accepted here.
      return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16 ||
             type == core::GgmlType::kQ8_K || type == core::GgmlType::kQ8_0 ||
             type == core::GgmlType::kQ5_K || type == core::GgmlType::kQ6_K ||
             type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ3_K ||
             type == core::GgmlType::kIQ4_NL ||
             type == core::GgmlType::kIQ4_XS || type == core::GgmlType::kIQ3_S;
  }
  return false;
}

bool ValidateTensor(const QwenTensorRef& tensor, std::size_t expected_elements,
                    TensorRole role, std::string_view name,
                    std::string* error_msg, std::size_t row_elements = 0) {
  if (tensor.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "Missing required Qwen tensor: " + std::string(name);
    }
    return false;
  }
  if (!SupportsTensorType(tensor.type, role)) {
    if (error_msg != nullptr) {
      *error_msg = "Unsupported " + std::string(TensorRoleName(role)) +
                   " tensor type for " + std::string(name) + ": " +
                   std::string(core::ToString(tensor.type));
    }
    return false;
  }
  if (tensor.num_elements != expected_elements) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor shape mismatch for " + std::string(name) +
                   ": expected " + std::to_string(expected_elements) +
                   " elements, found " + std::to_string(tensor.num_elements);
    }
    return false;
  }
  if (!tensor.FitsAvailableStorage()) {
    if (error_msg != nullptr) {
      *error_msg = "Tensor payload is truncated or has invalid encoded size: " +
                   std::string(name);
    }
    return false;
  }
  const bool quantized = tensor.type != core::GgmlType::kF32 &&
                         tensor.type != core::GgmlType::kF16 &&
                         tensor.type != core::GgmlType::kBF16;
  if (quantized && row_elements != 0 &&
      gufo::quant::QuantizedRowBytes(tensor.type, row_elements) == 0) {
    if (error_msg != nullptr) {
      *error_msg =
          "Quantized tensor row is not block aligned: " + std::string(name) +
          " (row elements " + std::to_string(row_elements) + ")";
    }
    return false;
  }
  return true;
}

}  // namespace

std::optional<QwenModelWeights> QwenModelWeights::LoadFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  const auto config_opt = reader.ExtractModelConfig(error_msg);
  if (!config_opt.has_value()) {
    return std::nullopt;
  }

  QwenModelWeights weights;
  weights.config = *config_opt;

  weights.token_embd = ExtractTensorRef(reader, "token_embd.weight");
  weights.output_norm = ExtractTensorRef(reader, "output_norm.weight");
  weights.output = ExtractTensorRef(reader, "output.weight");
  if (weights.output.empty()) {
    // Tied LM head shares token embeddings
    weights.output = weights.token_embd;
  }
  const std::size_t hidden_size = weights.config.hidden_size;
  const std::size_t vocab_size = weights.config.vocab_size;
  if (!ValidateTensor(weights.token_embd, vocab_size * hidden_size,
                      TensorRole::kEmbedding, "token_embd.weight", error_msg,
                      hidden_size) ||
      !ValidateTensor(weights.output_norm, hidden_size, TensorRole::kNorm,
                      "output_norm.weight", error_msg) ||
      !ValidateTensor(weights.output, vocab_size * hidden_size,
                      TensorRole::kProjection, "output.weight", error_msg,
                      hidden_size)) {
    return std::nullopt;
  }

  weights.layers.resize(weights.config.num_layers);
  for (std::uint32_t i = 0; i < weights.config.num_layers; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    auto& l = weights.layers[i];

    l.attn_norm = ExtractTensorRef(reader, prefix + "attn_norm.weight");
    if (l.attn_norm.empty()) {
      l.attn_norm = ExtractTensorRef(reader, prefix + "input_norm.weight");
    }

    l.ffn_norm = ExtractTensorRef(reader, prefix + "ffn_norm.weight");
    if (l.ffn_norm.empty()) {
      l.ffn_norm =
          ExtractTensorRef(reader, prefix + "post_attention_norm.weight");
    }
    if (l.ffn_norm.empty()) {
      l.ffn_norm = ExtractTensorRef(reader, prefix + "attn_post_norm.weight");
    }

    // Check if full attention layer
    l.attn_q = ExtractTensorRef(reader, prefix + "attn_q.weight");
    if (l.attn_q.empty()) {
      l.attn_q = ExtractTensorRef(reader, prefix + "wq.weight");
    }

    if (!l.attn_q.empty()) {
      l.is_full_attention = true;
      l.attn_k = ExtractTensorRef(reader, prefix + "attn_k.weight");
      if (l.attn_k.empty()) {
        l.attn_k = ExtractTensorRef(reader, prefix + "wk.weight");
      }
      l.attn_v = ExtractTensorRef(reader, prefix + "attn_v.weight");
      if (l.attn_v.empty()) {
        l.attn_v = ExtractTensorRef(reader, prefix + "wv.weight");
      }
      l.attn_output = ExtractTensorRef(reader, prefix + "attn_output.weight");
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "attn_out.weight");
      }
      if (l.attn_output.empty()) {
        l.attn_output = ExtractTensorRef(reader, prefix + "wo.weight");
      }
      l.attn_q_norm = ExtractTensorRef(reader, prefix + "attn_q_norm.weight");
      l.attn_k_norm = ExtractTensorRef(reader, prefix + "attn_k_norm.weight");
    } else {
      l.is_full_attention = false;
      l.attn_qkv = ExtractTensorRef(reader, prefix + "attn_qkv.weight");
      if (l.attn_qkv.empty()) {
        l.attn_qkv = ExtractTensorRef(reader, prefix + "wqkv.weight");
      }
      l.attn_gate = ExtractTensorRef(reader, prefix + "attn_gate.weight");
      if (l.attn_gate.empty()) {
        l.attn_gate = ExtractTensorRef(reader, prefix + "wqkv_gate.weight");
      }
      l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a");
      if (l.ssm_a.empty()) {
        l.ssm_a = ExtractTensorRef(reader, prefix + "ssm_a.weight");
      }
      l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d.weight");
      if (l.ssm_conv1d.empty()) {
        l.ssm_conv1d = ExtractTensorRef(reader, prefix + "ssm_conv1d");
      }
      l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.bias");
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt.weight");
      }
      if (l.ssm_dt.empty()) {
        l.ssm_dt = ExtractTensorRef(reader, prefix + "ssm_dt");
      }
      l.ssm_alpha = ExtractTensorRef(reader, prefix + "ssm_alpha.weight");
      l.ssm_beta = ExtractTensorRef(reader, prefix + "ssm_beta.weight");
      l.ssm_norm = ExtractTensorRef(reader, prefix + "ssm_norm.weight");
      l.ssm_out = ExtractTensorRef(reader, prefix + "ssm_out.weight");
    }

    l.ffn_gate = ExtractTensorRef(reader, prefix + "ffn_gate.weight");
    l.ffn_up = ExtractTensorRef(reader, prefix + "ffn_up.weight");
    l.ffn_down = ExtractTensorRef(reader, prefix + "ffn_down.weight");

    if (weights.config.IsMoE()) {
      l.ffn_gate_inp = ExtractTensorRef(reader, prefix + "ffn_gate_inp.weight");
      l.ffn_gate_inp_shexp =
          ExtractTensorRef(reader, prefix + "ffn_gate_inp_shexp.weight");
      l.ffn_gate_exps =
          ExtractTensorRef(reader, prefix + "ffn_gate_exps.weight");
      l.ffn_up_exps = ExtractTensorRef(reader, prefix + "ffn_up_exps.weight");
      l.ffn_down_exps =
          ExtractTensorRef(reader, prefix + "ffn_down_exps.weight");
      l.ffn_gate_shexp =
          ExtractTensorRef(reader, prefix + "ffn_gate_shexp.weight");
      l.ffn_up_shexp = ExtractTensorRef(reader, prefix + "ffn_up_shexp.weight");
      l.ffn_down_shexp =
          ExtractTensorRef(reader, prefix + "ffn_down_shexp.weight");
    }

    const bool expected_full_attention =
        ((i + 1) % weights.config.full_attention_interval) == 0;
    if (l.is_full_attention != expected_full_attention) {
      if (error_msg != nullptr) {
        *error_msg =
            "Unexpected Qwen layer kind at blk." + std::to_string(i) +
            ": expected " +
            (expected_full_attention ? "full attention" : "Gated DeltaNet");
      }
      return std::nullopt;
    }

    const std::size_t intermediate_size = weights.config.intermediate_size;
    if (!ValidateTensor(l.attn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "attn_norm.weight", error_msg) ||
        !ValidateTensor(l.ffn_norm, hidden_size, TensorRole::kNorm,
                        prefix + "post_attention_norm.weight", error_msg)) {
      return std::nullopt;
    }

    if (weights.config.IsMoE()) {
      const std::size_t n_experts = weights.config.expert_count;
      const std::size_t expert_ff = weights.config.expert_ff_length;
      const std::size_t shared_ff = weights.config.expert_shared_ff_length;
      if (!ValidateTensor(l.ffn_gate_inp, hidden_size * n_experts,
                          TensorRole::kNorm, prefix + "ffn_gate_inp.weight",
                          error_msg) ||
          !ValidateTensor(l.ffn_gate_inp_shexp, hidden_size, TensorRole::kNorm,
                          prefix + "ffn_gate_inp_shexp.weight", error_msg) ||
          !ValidateTensor(l.ffn_gate_exps, hidden_size * expert_ff * n_experts,
                          TensorRole::kProjection,
                          prefix + "ffn_gate_exps.weight", error_msg,
                          hidden_size) ||
          !ValidateTensor(l.ffn_up_exps, hidden_size * expert_ff * n_experts,
                          TensorRole::kProjection,
                          prefix + "ffn_up_exps.weight", error_msg,
                          hidden_size) ||
          !ValidateTensor(l.ffn_down_exps, expert_ff * hidden_size * n_experts,
                          TensorRole::kProjection,
                          prefix + "ffn_down_exps.weight", error_msg,
                          expert_ff) ||
          !ValidateTensor(l.ffn_gate_shexp, hidden_size * shared_ff,
                          TensorRole::kProjection,
                          prefix + "ffn_gate_shexp.weight", error_msg,
                          hidden_size) ||
          !ValidateTensor(l.ffn_up_shexp, hidden_size * shared_ff,
                          TensorRole::kProjection,
                          prefix + "ffn_up_shexp.weight", error_msg,
                          hidden_size) ||
          !ValidateTensor(l.ffn_down_shexp, shared_ff * hidden_size,
                          TensorRole::kProjection,
                          prefix + "ffn_down_shexp.weight", error_msg,
                          shared_ff)) {
        return std::nullopt;
      }
    } else if (!ValidateTensor(l.ffn_gate, intermediate_size * hidden_size,
                               TensorRole::kProjection,
                               prefix + "ffn_gate.weight", error_msg,
                               hidden_size) ||
               !ValidateTensor(l.ffn_up, intermediate_size * hidden_size,
                               TensorRole::kProjection,
                               prefix + "ffn_up.weight", error_msg,
                               hidden_size) ||
               !ValidateTensor(l.ffn_down, hidden_size * intermediate_size,
                               TensorRole::kProjection,
                               prefix + "ffn_down.weight", error_msg,
                               intermediate_size)) {
      return std::nullopt;
    }

    if (l.is_full_attention) {
      const std::size_t attention_size = weights.config.AttentionSize();
      const std::size_t kv_size =
          static_cast<std::size_t>(weights.config.num_key_value_heads) *
          weights.config.head_dim;
      if (!ValidateTensor(l.attn_q, 2 * attention_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_q.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_k, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_k.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_v, kv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_v.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_output, hidden_size * attention_size,
                          TensorRole::kProjection,
                          prefix + "attn_output.weight", error_msg,
                          attention_size) ||
          !ValidateTensor(l.attn_q_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_q_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.attn_k_norm, weights.config.head_dim,
                          TensorRole::kNorm, prefix + "attn_k_norm.weight",
                          error_msg)) {
        return std::nullopt;
      }
    } else {
      const std::size_t qkv_size = weights.config.SsmQkvSize();
      const std::size_t inner_size = weights.config.ssm_inner_size;
      const std::size_t rank = weights.config.ssm_time_step_rank;
      if (!ValidateTensor(l.attn_qkv, qkv_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_qkv.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.attn_gate, inner_size * hidden_size,
                          TensorRole::kProjection, prefix + "attn_gate.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_a, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_a", error_msg) ||
          !ValidateTensor(l.ssm_conv1d,
                          qkv_size * weights.config.ssm_conv_kernel,
                          TensorRole::kSsmParameter,
                          prefix + "ssm_conv1d.weight", error_msg) ||
          !ValidateTensor(l.ssm_dt, rank, TensorRole::kSsmParameter,
                          prefix + "ssm_dt.bias", error_msg) ||
          !ValidateTensor(l.ssm_alpha, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_alpha.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_beta, rank * hidden_size,
                          TensorRole::kProjection, prefix + "ssm_beta.weight",
                          error_msg, hidden_size) ||
          !ValidateTensor(l.ssm_norm, weights.config.SsmValueSize(),
                          TensorRole::kSsmParameter, prefix + "ssm_norm.weight",
                          error_msg) ||
          !ValidateTensor(l.ssm_out, hidden_size * inner_size,
                          TensorRole::kProjection, prefix + "ssm_out.weight",
                          error_msg, inner_size)) {
        return std::nullopt;
      }
    }
  }

  return weights;
}

}  // namespace gufo::models
