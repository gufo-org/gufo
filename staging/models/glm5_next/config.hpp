#ifndef GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_
#define GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::glm5_next {

/// Architecture parameters of a GLM 5.3 Flash (`glm5-next`) GGUF artifact:
/// a trunk of Kimi-style Delta Attention (KDA) linear layers on a 3:1 stride
/// with absorbed Multi-head Latent Attention (MLA) layers carrying a DeepSeek
/// sparse attention (DSA) indexer, four-stream hyper-connections (mHC) around
/// every block, a sigmoid-gated Mixture-of-Experts FFN with one shared
/// expert (dense FFN in the leading blocks), and a single self-speculative
/// nextn block stored at index `block_count - 1`. Every value is read from
/// the file; `FromGguf` accepts only the measured geometry profile and
/// records it in `profile_name`. The documented dimensions below are the
/// GLM-5.3-Flash profile.
struct Config {
  std::string_view profile_name{""};  ///< Matched kernel geometry profile.
  std::uint32_t num_layers{0};        ///< Trunk layers (nextn excluded), 45.
  std::uint32_t num_layers_all{0};    ///< block_count, 46 (trunk + nextn).
  std::uint32_t nextn_layers{0};      ///< 1; the draft block sits last.
  std::uint32_t hidden_size{0};       ///< embedding_length, 4096.
  std::uint32_t vocab_size{0};        ///< 154880.
  std::uint32_t context_length{0};    ///< 1048576.
  float rms_eps{1e-5F};               ///< attention.layer_norm_rms_epsilon.
  float ln_eps{1e-6F};                ///< LayerNorm eps (indexer k_norm).

  // Absorbed MLA + DSA on every layer with a non-zero head_count_kv entry.
  // The block is NoPE (rope.dimension_count == 0): the per-head query is
  // absorbed by attn_k_b into the 512-wide latent and no rotary component is
  // ever produced or cached; attn_v_b maps the latent to per-head values.
  std::uint32_t num_heads{0};      ///< 64
  std::uint32_t q_lora_rank{0};    ///< 1536 (q_a output, indexer q input).
  std::uint32_t kv_lora_rank{0};   ///< 512 (latent width, cache row).
  std::uint32_t mla_head_dim{0};   ///< per-head q/k width, 256.
  std::uint32_t mla_value_dim{0};  ///< per-head value width, 256.
  std::vector<std::uint32_t> head_count_kv;  ///< [num_layers_all], 0 on KDA.

  // KDA linear layers: 64 heads of 128; q/k/v are separate projections into
  // the 8192-wide inner space, convolved per tap with a causal kernel of 4.
  std::uint32_t kda_head_dim{0};     ///< kda.head_dim, 128.
  std::uint32_t kda_heads{0};        ///< 64 (profile; verified on q).
  std::uint32_t ssm_conv_kernel{0};  ///< 4.
  float kda_gate_lower_bound{0.0F};  ///< -5.0
  std::uint32_t kda_stride{4};       ///< DSA at (il + 1) % stride == 0.

  // DSA indexer: 32 heads scoring 128-wide keys, pooled kpool-at-a-time with
  // learned absolute-position weights, top_k token slots fed to the MLA.
  std::uint32_t indexer_heads{0};    ///< attention.indexer.head_count, 32
  std::uint32_t indexer_key_len{0};  ///< key_length, 128
  std::uint32_t indexer_top_k{0};    ///< 2048 token slots
  std::uint32_t indexer_kpool{0};    ///< 4
  bool index_share_mtp{false};       ///< nextn reuses the last selection
  bool kpool_select_tail{false};     ///< absent keeps the reference default true

  // Four-stream hyper-connections around the attention and FFN sub-layers.
  std::uint32_t hc_count{0};           ///< 4
  std::uint32_t hc_sinkhorn_iters{0};  ///< 20
  float hc_eps{1e-6F};                 ///< hyper_connection.epsilon

  // Mixture of experts with sigmoid gating and one shared expert.
  std::uint32_t num_experts{0};       ///< 288
  std::uint32_t num_experts_used{0};  ///< 8
  std::uint32_t expert_ff{0};         ///< 2048
  std::uint32_t shared_expert_ff{0};  ///< 2048
  float expert_weights_scale{1.0F};   ///< 2.5
  bool expert_weights_norm{false};
  std::uint32_t expert_gating_func{0};  ///< 2 (sigmoid + bias)

  // Dense FFN blocks 0..leading_dense_blocks-1 replace the expert trios.
  std::uint32_t leading_dense_blocks{0};  ///< 3
  std::uint32_t dense_ff{0};              ///< 12288
  // swiglu_clamp_{exp,shexp} are scalars or per-block arrays in the file;
  // stored expanded to num_layers_all with the scalar semantics.
  std::vector<float> swiglu_clamp_exp;
  std::vector<float> swiglu_clamp_shexp;

  [[nodiscard]] bool IsDsaLayer(std::uint32_t il) const noexcept {
    return il < head_count_kv.size() && head_count_kv[il] != 0;
  }
  [[nodiscard]] bool IsDenseFfnLayer(std::uint32_t il) const noexcept {
    return il < leading_dense_blocks;
  }
  [[nodiscard]] std::uint32_t KdaInnerSize() const noexcept {
    return kda_heads * kda_head_dim;  // 8192
  }
  [[nodiscard]] std::uint32_t MlaQSize() const noexcept {
    return num_heads * mla_head_dim;  // 16384
  }
  [[nodiscard]] std::uint32_t MlaVSize() const noexcept {
    return num_heads * mla_value_dim;  // 16384
  }
  [[nodiscard]] std::uint32_t IndexerPoolSize() const noexcept {
    return indexer_top_k * indexer_key_len;
  }
  /// Number of pool groups selected (top_k token slots at kpool tokens each).
  [[nodiscard]] std::uint32_t IndexerTopPools() const noexcept {
    return indexer_top_k / indexer_kpool;
  }
  [[nodiscard]] float SwigluClampExp(std::uint32_t il) const noexcept {
    return swiglu_clamp_exp[il];
  }
  [[nodiscard]] float SwigluClampShexp(std::uint32_t il) const noexcept {
    return swiglu_clamp_shexp[il];
  }

  /// Parses every `glm5-next.*` key, validates the geometry against the
  /// measured kernel profile and (with `require_trunk`) the 3:1 KDA/DSA
  /// placement. Fills `error_msg` and returns nullopt on any mismatch.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, bool require_trunk, std::string* error_msg);
};

}  // namespace gufo::models::glm5_next

#endif  // GUFO_MODELS_GLM5_NEXT_CONFIG_HPP_
