#ifndef GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_
#define GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "staging/models/glm5_next/config.hpp"

namespace gufo::models::glm5_next {

/// Non-owning view of one GGUF tensor. `cols` x `rows` follows the GGUF
/// convention: cols (ne[0]) is the contiguous reduction dimension, rows
/// (ne[1]) the output dimension, and `experts` (ne[2]) stacks whole matrices.
/// The payload stays exactly as stored in the file (quantized blocks for the
/// ROCm kernels, F32 for the CPU oracle); nothing is dequantized at load.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  std::uint64_t file_offset{0};  ///< Byte offset inside the owning shard.
  std::uint32_t shard{0};        ///< Mapped region index in the reader.
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] std::uint64_t ElementCount() const noexcept {
    return cols * rows * experts;
  }
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
  /// Base address of one stacked matrix (expert or attention head).
  [[nodiscard]] const std::uint8_t* Expert(std::uint64_t e) const noexcept {
    return static_cast<const std::uint8_t*>(data) + RowBytes() * rows * e;
  }
};

struct LayerWeights {
  bool dsa{false};  ///< True on MLA+DSA layers (incl. the nextn block).

  // Four-stream hyper-connections (`fn` maps the flattened stream state,
  // [4*hidden] -> 24 = pre(4) + post(4) + comb(16) mixing weights).
  TensorRef hc_attn_fn;     ///< [hc*h -> 24], Q8_0 in the measured artifact.
  TensorRef hc_attn_base;   ///< [24] F32
  TensorRef hc_attn_scale;  ///< [3] F32
  TensorRef hc_ffn_fn;
  TensorRef hc_ffn_base;
  TensorRef hc_ffn_scale;

  TensorRef attn_norm;  ///< [hidden] pre-attention RMSNorm, F32.
  TensorRef ffn_norm;   ///< [hidden] pre-FFN RMSNorm, F32.

  // Absorbed MLA + DSA (dsa layers). The per-head query (attn_q_a -> norm ->
  // attn_q_b, 64 x 256, NoPE) is projected onto the latent width by
  // attn_k_b; attn_v_b maps the cached latent to per-head values inside the
  // attention over indexer-selected rows.
  TensorRef attn_q_a;        ///< [h -> 1536]
  TensorRef attn_q_a_norm;   ///< [1536] F32
  TensorRef attn_q_b;        ///< [1536 -> 64*256]
  TensorRef attn_kv_a;       ///< [h -> 512] latent (the whole cache row)
  TensorRef attn_kv_a_norm;  ///< [512] F32
  TensorRef attn_k_b;        ///< [256 -> 512] per head, absorbs q onto latent
  TensorRef attn_v_b;        ///< [512 -> 256] per head
  TensorRef attn_out;        ///< [64*256 -> h]

  // DSA indexer (dsa layers). iq sees the q_lora vector; the cache packs
  // [128 ik | 128 gate] rows and is pooled kpool-wise with `ape`.
  TensorRef idx_q_b;        ///< [1536 -> 32*128]
  TensorRef idx_k;          ///< [h -> 128]
  TensorRef idx_k_norm;     ///< [128] F32 LayerNorm weight
  TensorRef idx_k_bias;     ///< [128] F32 LayerNorm bias
  TensorRef idx_proj;       ///< [h -> 32] per-head score weights
  TensorRef idx_comp_gate;  ///< [h -> 128] pooling gate
  TensorRef idx_comp_ape;   ///< [128, 4] F32 absolute-position pooling weights

  // KDA linear layers (non-dsa blocks).
  TensorRef ssm_q;       ///< [h -> 8192]
  TensorRef ssm_k;       ///< [h -> 8192]
  TensorRef ssm_v;       ///< [h -> 8192]
  TensorRef ssm_conv_q;  ///< [conv_kernel, 8192] F32, taps contiguous
  TensorRef ssm_conv_k;
  TensorRef ssm_conv_v;
  TensorRef ssm_f_a;   ///< [h -> 128] decay low-rank in
  TensorRef ssm_f_b;   ///< [128 -> 8192] decay low-rank out (+dt bias)
  TensorRef ssm_g_a;   ///< [h -> 128] output gate in
  TensorRef ssm_g_b;   ///< [128 -> 8192] output gate out
  TensorRef ssm_beta;  ///< [h -> 64]
  TensorRef ssm_a;     ///< [64] F32, -exp(A_log)
  TensorRef ssm_dt;    ///< [8192] F32 bias added to the f_b output
  TensorRef ssm_norm;  ///< [128] F32 head RMSNorm
  TensorRef ssm_out;   ///< [8192 -> h]

  // Dense FFN (leading blocks) or MoE FFN (all other blocks).
  TensorRef ffn_gate;       ///< [h -> dense_ff]
  TensorRef ffn_up;         ///< [h -> dense_ff]
  TensorRef ffn_down;       ///< [dense_ff -> h]
  TensorRef router;         ///< [h -> 288] F32
  TensorRef exp_probs_b;    ///< [288] F32 routing bias
  TensorRef ffn_gate_exps;  ///< [h -> 2048] stacked 288 experts
  TensorRef ffn_up_exps;
  TensorRef ffn_down_exps;  ///< [2048 -> h] stacked 288 experts
  TensorRef shexp_gate;     ///< [h -> 2048]
  TensorRef shexp_up;       ///< [h -> 2048]
  TensorRef shexp_down;     ///< [2048 -> h]

  // nextn block only: fuses the previous hidden and the embedded token.
  TensorRef nextn_enorm;             ///< [hidden] F32
  TensorRef nextn_hnorm;             ///< [hidden] F32
  TensorRef nextn_eh_proj;           ///< [2*h -> h]
  TensorRef nextn_shared_head_norm;  ///< [hidden] F32, before the shared head
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;              ///< [h -> vocab]
  TensorRef output;                  ///< [h -> vocab], falls back to token_embd
  TensorRef output_norm;             ///< [h] F32 after the mHC mean
  std::vector<LayerWeights> layers;  ///< [num_layers_all], nextn last.

  /// Reads the architecture profile and binds every tensor (trunk and nextn)
  /// with exact shape and accepted-format validation.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg);
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_WEIGHTS_HPP_
