#ifndef GUFO_MODELS_NEMOTRON35_BOLT_WEIGHTS_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/nemotron35_bolt/config.hpp"

namespace gufo::models::nemotron35_bolt {

/// Non-owning view of one GGUF tensor. `rows` x `cols` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  /// Per-tensor output scale from the `<name>.scale` companion (NVFP4).
  float scale{1.0F};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked expert matrix.
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

struct LayerWeights {
  BlockType type{BlockType::kMamba};
  TensorRef norm;  ///< [hidden] pre-mixer RMSNorm gamma, on every block.

  // Mamba-2.
  TensorRef ssm_in;        ///< [hidden -> z|x|B|C|dt]
  TensorRef ssm_conv1d;    ///< [conv_kernel, channels], oldest tap first
  TensorRef ssm_conv1d_b;  ///< [channels]
  TensorRef ssm_dt_b;      ///< [heads] softplus bias
  TensorRef ssm_a;         ///< [1, heads] = -exp(A_log)
  TensorRef ssm_d;         ///< [1, heads] skip connection
  TensorRef ssm_norm;      ///< [inner / groups, groups] grouped RMSNorm gamma
  TensorRef ssm_out;       ///< [inner -> hidden]

  // NoPE GQA.
  TensorRef attn_q;    ///< [hidden -> heads * head_dim]
  TensorRef attn_k;    ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_v;    ///< [hidden -> kv_heads * head_dim]
  TensorRef attn_out;  ///< [heads * head_dim -> hidden]

  // LatentMoE.
  std::uint32_t num_experts{0};  ///< Routed experts of this block.
  TensorRef router;              ///< [hidden -> experts] F32 logits
  TensorRef router_bias;         ///< [experts] selection-only score bias
  TensorRef latent_down;         ///< [hidden -> latent]
  TensorRef latent_up;           ///< [latent -> hidden]
  TensorRef up_exps;             ///< [latent -> expert_ff] x experts
  TensorRef down_exps;           ///< [expert_ff -> latent] x experts
  TensorRef up_shexp;            ///< [hidden -> shared_ff]
  TensorRef down_shexp;          ///< [shared_ff -> hidden]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab]
  TensorRef output_norm;  ///< [hidden]
  TensorRef output;       ///< [hidden -> vocab]
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Tensor payloads stay mapped and
  /// untouched; only headers and the scalar scale companions are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

struct DrafterLayerWeights {
  TensorRef attn_norm;       ///< [hidden]
  TensorRef attn_conv_base;  ///< [hidden, kernel, 2 sides] static taps
  TensorRef attn_conv_proj;  ///< [hidden -> 2 * kernel * groups] BF16
  TensorRef attn_q;          ///< [hidden -> heads * head_dim] BF16
  TensorRef attn_k;          ///< [hidden -> kv_heads * head_dim] BF16
  TensorRef attn_v;          ///< [hidden -> kv_heads * head_dim] BF16
  TensorRef attn_q_norm;     ///< [head_dim]
  TensorRef attn_k_norm;     ///< [head_dim]
  TensorRef attn_out;        ///< [heads * head_dim -> hidden] BF16
  TensorRef attn_sinks;      ///< [heads] softmax sink logits
  TensorRef ffn_norm;        ///< [hidden]
  TensorRef ffn_conv_base;   ///< [hidden, kernel, 2 sides]
  TensorRef ffn_conv_proj;   ///< [hidden -> 2 * kernel * groups] BF16
  TensorRef ffn_gate;        ///< [hidden -> ff] NVFP4
  TensorRef ffn_up;          ///< [hidden -> ff] NVFP4
  TensorRef ffn_down;        ///< [ff -> hidden] NVFP4
};

/// One LiLiCorr transformer layer over the candidate nodes.
struct LiliCorrLayerWeights {
  TensorRef attn_norm, mlp_norm;   ///< [lc_hidden]
  TensorRef in_proj, in_proj_b;    ///< [lc_hidden -> 3 lc_hidden] F32
  TensorRef out_proj, out_proj_b;  ///< [lc_hidden -> lc_hidden] NVFP4
  TensorRef mlp_up, mlp_up_b;      ///< [lc_hidden -> lc_mlp] NVFP4
  TensorRef mlp_down, mlp_down_b;  ///< [lc_mlp -> lc_hidden] NVFP4
};

/// The LiLiCorr DFlash drafter. It has no embedding or head of its own and
/// uses the trunk's.
struct DrafterWeights {
  DrafterConfig config;
  TensorRef fc;           ///< [features -> hidden] NVFP4
  TensorRef enc_norm;     ///< [hidden] after fc
  TensorRef output_norm;  ///< [hidden] before the trunk's head
  std::vector<DrafterLayerWeights> layers;
  // LiLiCorr.
  TensorRef token_proj, token_proj_b;      ///< [hidden -> lc_hidden] NVFP4
  TensorRef pass_proj, pass_proj_b;        ///< [hidden -> lc_hidden] NVFP4
  TensorRef context_proj, context_proj_b;  ///< [hidden -> lc_hidden] NVFP4
  TensorRef feature_norm, feature_norm_b;  ///< [5] LayerNorm affine
  TensorRef feature_up, feature_up_b;      ///< [5 -> lc_hidden] F32
  TensorRef feature_down, feature_down_b;  ///< [lc_hidden -> lc_hidden] F32
  TensorRef slot_embd;                     ///< [lc_hidden, slots]
  TensorRef rank_embd;                     ///< [lc_hidden, top_k]
  TensorRef attention_bias;  ///< [nodes, nodes, lc_heads], key fastest
  std::vector<LiliCorrLayerWeights> lc_layers;
  TensorRef lc_output_norm, anchor_norm;  ///< [lc_hidden]
  TensorRef factor_in, factor_in_b;       ///< [3 lc_hidden -> lc_hidden] F32
  TensorRef out_head, out_head_b;         ///< [lc_hidden -> factor] F32
  TensorRef in_head, in_head_b;           ///< [lc_hidden -> factor] F32
  TensorRef anchor_head, anchor_head_b;   ///< [lc_hidden -> factor] F32

  [[nodiscard]] static std::optional<DrafterWeights> Bind(
      const core::GgufReader& reader, const Config& target,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_WEIGHTS_HPP_
