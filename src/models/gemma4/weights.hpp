#ifndef GUFO_MODELS_GEMMA4_WEIGHTS_HPP_
#define GUFO_MODELS_GEMMA4_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/config.hpp"

namespace gufo::models::gemma4 {

/// Non-owning view of one GGUF tensor. `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1]) the output dimension and `experts`
/// (ne[2]) the number of stacked expert matrices.
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
  /// Encoded bytes of one row (`cols` elements) in this format.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * rows * experts;
  }
};

struct LayerWeights {
  TensorRef attn_norm;    ///< [hidden]
  TensorRef attn_q;       ///< [hidden -> heads * head_dim]
  TensorRef attn_k;       ///< [hidden -> kv_heads * head_dim]; target only
  TensorRef attn_v;       ///< Absent on global layers: V is the K projection.
  TensorRef attn_q_norm;  ///< [head_dim]
  TensorRef attn_k_norm;  ///< [head_dim]; target only
  TensorRef attn_output;  ///< [heads * head_dim -> hidden]
  TensorRef post_attn_norm;  ///< [hidden]
  TensorRef ffn_norm;        ///< [hidden]
  TensorRef ffn_gate;        ///< [hidden -> ffn]
  TensorRef ffn_up;          ///< [hidden -> ffn]
  TensorRef ffn_down;        ///< [ffn -> hidden]
  TensorRef post_ffn_norm;   ///< [hidden]
  float output_scale{1.0F};  ///< layer_output_scale, applied to the residual.

  // Routed experts (26B-A4B), run beside the dense MLP above; empty on dense
  // targets and drafts.
  TensorRef router;           ///< ffn_gate_inp, F32 [hidden -> experts]
  TensorRef router_scale;     ///< ffn_gate_inp.scale, F32 [hidden]
  TensorRef pre_ffn_norm_2;   ///< Expert input norm, [hidden]
  TensorRef post_ffn_norm_1;  ///< Dense MLP output norm, [hidden]
  TensorRef post_ffn_norm_2;  ///< Expert mixture output norm, [hidden]
  /// [experts][2 * expert_ffn][hidden]: the gate rows, then the up rows.
  TensorRef gate_up_exps;
  TensorRef down_exps;        ///< [experts][hidden][expert_ffn]
  TensorRef down_exps_scale;  ///< F32 [experts], scales each expert output.
};

/// Host copy of the rope frequency divisors used by global layers. A value
/// of 1 rotates the pair; the GGUF marks non-rotating pairs with 1e30.
struct RopeFactors {
  std::vector<float> values;  ///< [head_dim_global / 2]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [hidden -> vocab]; scaled by sqrt(hidden).
  TensorRef output;       ///< LM head; the token embedding when tied.
  TensorRef output_norm;  ///< [hidden]
  RopeFactors rope_factors;
  std::uint32_t vocab_size{0};
  std::vector<LayerWeights> layers;

  [[nodiscard]] bool TiedOutput() const noexcept {
    return output.data == token_embd.data;
  }
  /// Binds and validates the target. Payloads stay mapped; only headers and
  /// the tiny F32 scalars are read.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// The `gemma4-assistant` drafter: Q-only attention over the target's KV,
/// its own vocabulary head, and projections to and from the target width.
struct DraftWeights {
  Config config;
  TensorRef pre_projection;   ///< [2 * target_hidden -> hidden]
  TensorRef post_projection;  ///< [hidden -> target_hidden]
  TensorRef token_embd;       ///< [hidden -> vocab], the draft LM head.
  TensorRef output_norm;      ///< [hidden]
  RopeFactors rope_factors;
  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<DraftWeights> Bind(
      const core::GgufReader& reader, const ModelWeights& target,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_WEIGHTS_HPP_
