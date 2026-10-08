#ifndef GUFO_MODELS_GEMMA4_CONFIG_HPP_
#define GUFO_MODELS_GEMMA4_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4 {

/// Architecture parameters of a `gemma4` target or its `gemma4-assistant`
/// MTP drafter. Every value is read from the file; the checks only bound what
/// this runtime implements: text layers with interleaved sliding-window and
/// global attention, either dense or with routed experts beside the dense
/// MLP in every layer (26B-A4B), no per-layer embeddings and no KV sharing
/// inside the target.
/// Routed experts per token and experts per layer the kernels support.
inline constexpr std::uint32_t kMaxExpertsUsed = 8;
inline constexpr std::uint32_t kMaxExperts = 256;

struct Config {
  std::uint32_t num_layers{0};      ///< 60 for the 31B target, 4 for its draft.
  std::uint32_t hidden_size{0};     ///< 5376 (target) or 1024 (draft).
  std::uint32_t ffn_size{0};        ///< Dense MLP: 21504 (31B), 2112 (26B-A4B).
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  std::uint32_t num_heads{0};           ///< Query heads, identical per layer.
  std::vector<std::uint32_t> kv_heads;  ///< Per layer: 16 sliding, 4 global.
  std::vector<std::uint8_t> sliding;    ///< Per layer: 1 sliding, 0 global.

  std::uint32_t head_dim_global{0};   ///< 512
  std::uint32_t head_dim_sliding{0};  ///< 256
  std::uint32_t rope_dim_global{0};   ///< 512; rope_freqs selects the pairs.
  std::uint32_t rope_dim_sliding{0};  ///< 256
  float rope_theta_global{0.0F};      ///< 1e6
  float rope_theta_sliding{0.0F};     ///< 1e4
  std::uint32_t sliding_window{0};    ///< 1024 keys, including the query.
  float final_logit_softcap{0.0F};    ///< 30; zero disables the cap.

  // Mixture of experts (26B-A4B); zero experts means a dense target.
  std::uint32_t num_experts{0};      ///< 128
  std::uint32_t experts_used{0};     ///< 8 routed experts per token.
  std::uint32_t expert_ffn_size{0};  ///< 704 per expert projection.

  // Draft (`gemma4-assistant`) only.
  std::uint32_t target_hidden_size{0};  ///< embedding_length_out, 5376.
  std::uint32_t shared_kv_layers{0};    ///< Draft layers reading target KV.

  [[nodiscard]] bool HasExperts() const noexcept { return num_experts != 0; }
  [[nodiscard]] bool IsSliding(std::uint32_t layer) const noexcept {
    return sliding[layer] != 0;
  }
  [[nodiscard]] std::uint32_t HeadDim(std::uint32_t layer) const noexcept {
    return IsSliding(layer) ? head_dim_sliding : head_dim_global;
  }
  [[nodiscard]] std::uint32_t RopeDim(std::uint32_t layer) const noexcept {
    return IsSliding(layer) ? rope_dim_sliding : rope_dim_global;
  }
  [[nodiscard]] float RopeTheta(std::uint32_t layer) const noexcept {
    return IsSliding(layer) ? rope_theta_sliding : rope_theta_global;
  }
  [[nodiscard]] std::uint32_t QDim(std::uint32_t layer) const noexcept {
    return num_heads * HeadDim(layer);
  }
  [[nodiscard]] std::uint32_t KvDim(std::uint32_t layer) const noexcept {
    return kv_heads[layer] * HeadDim(layer);
  }
  [[nodiscard]] std::uint32_t GlobalLayerCount() const noexcept;
  /// Target layer whose KV cache a shared-KV draft layer attends: the last
  /// target layer of the same attention kind.
  [[nodiscard]] std::uint32_t SharedKvSource(
      std::uint32_t draft_layer, const Config& target) const noexcept;

  /// Reads a `gemma4` target. Rejects variants this runtime does not
  /// implement (per-layer embeddings, KV-shared target layers, more than
  /// kMaxExpertsUsed routed experts per token).
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Reads a `gemma4-assistant` MTP drafter and checks that every draft
  /// layer can attend its target KV source with identical geometry.
  [[nodiscard]] static std::optional<Config> DraftFromGguf(
      const core::GgufReader& reader, const Config& target,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CONFIG_HPP_
