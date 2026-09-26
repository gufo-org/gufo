#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace gufo::models::qwen38_flash_next {

/// Architecture parameters of a `qwen4exp` GGUF artifact. Every value is read
/// from the file; the fixed limits below only bound what this runtime was
/// written and validated for (Qwen3.8-Flash-Next, 48 trunk layers).
struct Config {
  static constexpr std::uint32_t kMaxPleHeads = 16;
  static constexpr std::uint32_t kMaxPleNgram = 3;

  std::uint32_t num_layers{0};      ///< Trunk layers (MTP block excluded).
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP block.
  std::uint32_t hidden_size{0};     ///< n_embd, 2560.
  std::uint32_t vocab_size{0};      ///< 248320.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-6F};

  // Hyper-connections: hc_count parallel residual streams of hidden_size.
  std::uint32_t hc_count{0};     ///< 4
  std::uint32_t hc_low_rank{0};  ///< 320

  // Full (gated GQA) attention on every full_attention_interval-th layer.
  std::uint32_t full_attention_interval{0};  ///< 4
  std::uint32_t num_heads{0};                ///< 24
  std::uint32_t num_kv_heads{0};             ///< 2
  std::uint32_t head_dim{0};                 ///< 256
  std::uint32_t rotary_dim{0};               ///< 64
  float rope_theta{0.0F};                    ///< 1e7
  std::array<std::uint32_t, 4> rope_sections{};
  /// Static YaRN. Off unless the loader or the artifact enables it; the
  /// native `context_length` above is never rewritten.
  RopeScaling rope_scaling{};

  // Qwen Sparse Attention indexer (block top-k selection).
  std::uint32_t indexer_heads{0};     ///< 4
  std::uint32_t indexer_head_dim{0};  ///< 128
  std::uint32_t indexer_top_k{0};     ///< 2048 tokens
  std::uint32_t compress_ratio{0};    ///< 4 tokens per scored block

  // Gated DeltaNet linear attention.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{0};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{0};  ///< 48 (time_step_rank)
  std::uint32_t ssm_inner_size{0};   ///< 6144

  // Mixture of experts with one always-on shared expert.
  std::uint32_t num_experts{0};       ///< 512
  std::uint32_t num_experts_used{0};  ///< 10
  std::uint32_t expert_ff{0};         ///< 640
  std::uint32_t shared_expert_ff{0};  ///< 640

  // PLE n-gram hash embedding, injected into one linear-attention layer.
  std::int32_t ple_layer{-1};       ///< -1 when the artifact carries no PLE.
  std::uint32_t ple_ngram_size{0};  ///< 3: bigrams and trigrams
  std::uint32_t ple_heads_per_ngram{0};  ///< 8
  std::uint32_t ple_heads{0};            ///< (ngram_size - 1) * heads_per_ngram
  std::uint32_t ple_head_dim{0};     ///< 160 (embedding_length_per_layer_input)
  std::uint32_t ple_conv_kernel{0};  ///< 4
  std::uint32_t ple_eos_token{0};    ///< resets the n-gram window
  std::array<std::uint64_t, kMaxPleNgram> ple_multipliers{};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_offsets{};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_vocab{};
  std::uint64_t ple_rows{0};  ///< max(offset + vocab) over heads

  // Speculative MTP block (`nextn`), present in the draft sidecar only.
  std::uint32_t nextn_layers{0};

  [[nodiscard]] std::uint32_t HcDim() const noexcept {
    return hc_count * hidden_size;
  }
  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  /// Channels of the fused q|k|v projection and its causal convolution.
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t PleEmbeddingDim() const noexcept {
    return ple_heads * ple_head_dim;
  }
  /// Tokens of PLE conv history kept per sequence: (kernel - 1) * dilation.
  [[nodiscard]] std::uint32_t PleConvHistory() const noexcept {
    return (ple_conv_kernel - 1) * ple_ngram_size;
  }

  [[nodiscard]] bool IsLinearLayer(std::uint32_t layer) const noexcept {
    return layer < num_layers && ((layer + 1) % full_attention_interval) != 0;
  }
  [[nodiscard]] bool IsPleLayer(std::uint32_t layer) const noexcept {
    return ple_layer >= 0 && layer == static_cast<std::uint32_t>(ple_layer);
  }

  /// Longest session the rope configuration supports.
  [[nodiscard]] std::uint32_t MaxContextLength() const noexcept {
    return ScaledContextLength(rope_scaling, context_length);
  }
  /// uint32 words per sparse-mask row: one bit per compress_ratio block.
  [[nodiscard]] std::uint32_t SparseMaskWords(
      std::uint32_t context) const noexcept {
    const std::uint32_t blocks =
        (context + compress_ratio - 1) / compress_ratio;
    return (blocks + 31) / 32;
  }

  /// The draft executes with trunk constants and shared vocabulary weights.
  /// PLE/SSM parameters are intentionally excluded: the draft is full
  /// attention.
  [[nodiscard]] bool MtpMatches(const Config& trunk) const noexcept {
    return nextn_layers == 1 && num_layers == trunk.num_layers &&
           hidden_size == trunk.hidden_size &&
           context_length == trunk.context_length && rms_eps == trunk.rms_eps &&
           hc_count == trunk.hc_count && hc_low_rank == trunk.hc_low_rank &&
           num_heads == trunk.num_heads && num_kv_heads == trunk.num_kv_heads &&
           head_dim == trunk.head_dim && rotary_dim == trunk.rotary_dim &&
           rope_theta == trunk.rope_theta &&
           rope_sections == trunk.rope_sections &&
           indexer_heads == trunk.indexer_heads &&
           indexer_head_dim == trunk.indexer_head_dim &&
           indexer_top_k == trunk.indexer_top_k &&
           compress_ratio == trunk.compress_ratio &&
           num_experts == trunk.num_experts &&
           num_experts_used == trunk.num_experts_used &&
           expert_ff == trunk.expert_ff &&
           shared_expert_ff == trunk.shared_expert_ff;
  }

  /// Reads and validates the `qwen4exp.*` metadata. `require_trunk` rejects
  /// draft-only sidecars; the MTP sidecar is read with it false.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, bool require_trunk,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_CONFIG_HPP_
