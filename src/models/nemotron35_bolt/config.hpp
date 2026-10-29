#ifndef GUFO_MODELS_NEMOTRON35_BOLT_CONFIG_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::nemotron35_bolt {

/// Mixer of one trunk block. Every block is pre-norm with a residual add.
enum class BlockType : std::uint8_t {
  kMamba,      ///< Mamba-2 selective state space mixer.
  kAttention,  ///< GQA without positional encoding (NoPE).
  kMoe,        ///< LatentMoE: routed ReLU^2 experts plus one shared expert.
};

/// Architecture parameters of a `nemotron_h_moe` GGUF artifact. Every value is
/// read from the file; the fixed limits below only bound what this runtime was
/// written and validated for (Nemotron 3.5 Bolt 74B-A9B, 88 blocks).
struct Config {
  std::uint32_t num_layers{0};      ///< block_count, 88.
  std::uint32_t hidden_size{0};     ///< n_embd, 4096.
  std::uint32_t vocab_size{0};      ///< 131072.
  std::uint32_t context_length{0};  ///< 262144.
  float rms_eps{1e-5F};

  // Attention blocks.
  std::uint32_t num_heads{0};     ///< 32
  std::uint32_t num_kv_heads{0};  ///< 2
  std::uint32_t head_dim{0};      ///< 128

  // Mamba-2 blocks.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_state_size{0};   ///< 128 (d_state)
  std::uint32_t ssm_groups{0};       ///< 8 (B/C groups)
  std::uint32_t ssm_inner_size{0};   ///< 8192 (d_inner)
  std::uint32_t ssm_heads{0};        ///< 128 (time_step_rank)

  // MoE blocks. Routed experts per layer come from the tensor shapes; the
  // metadata expert_count is only their maximum.
  std::uint32_t max_experts{0};       ///< 496
  std::uint32_t latent_size{0};       ///< 1024, routed experts' model width.
  std::uint32_t shared_expert_ff{0};  ///< 5376
  float expert_weights_scale{1.0F};   ///< 5.0
  bool expert_weights_norm{false};    ///< true

  std::vector<BlockType> blocks;
  std::vector<std::uint32_t> experts_used;  ///< Per block; 0 off MoE blocks.
  std::vector<std::uint32_t> expert_ff;     ///< Per block; 0 off MoE blocks.

  [[nodiscard]] std::uint32_t SsmHeadDim() const noexcept {
    return ssm_inner_size / ssm_heads;
  }
  /// Channels of the x|B|C slice and its causal convolution.
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return ssm_inner_size + 2 * ssm_groups * ssm_state_size;
  }
  /// Rows of ssm_in: z | x | B | C | dt.
  [[nodiscard]] std::uint32_t SsmInProjection() const noexcept {
    return ssm_inner_size + SsmConvChannels() + ssm_heads;
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }

  /// Reads and validates the `nemotron_h_moe.*` metadata.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

/// NVIDIA's LiLiCorr DFlash drafter (`dflash` architecture): a five-layer
/// block-diffusion transformer that reads the trunk's residual stream at
/// `target_layers`, drafts `block_size - 1` tokens in one pass, and a
/// two-layer LiLiCorr head that scores the drafted candidates jointly.
struct DrafterConfig {
  std::uint32_t num_layers{0};    ///< 5
  std::uint32_t hidden_size{0};   ///< 4096, equal to the trunk's
  std::uint32_t ff_size{0};       ///< 12288, SiLU-gated FFN width
  std::uint32_t num_heads{0};     ///< 32
  std::uint32_t num_kv_heads{0};  ///< 2
  std::uint32_t head_dim{0};      ///< 128, NeoX RoPE over all of it
  float rms_eps{1e-6F};
  float rope_theta{10000.0F};
  std::uint32_t block_size{0};      ///< 8: the anchor and 7 mask tokens
  std::uint32_t sliding_window{0};  ///< 1024, on every layer
  std::uint32_t conv_kernel{0};     ///< 2, causal within the block
  std::uint32_t conv_group{0};      ///< 16 channels per dynamic tap
  std::uint32_t mask_token{0};      ///< 990
  std::vector<std::uint32_t> target_layers;  ///< Trunk blocks whose input
                                             ///< residual feeds the drafter.
  // LiLiCorr candidate head.
  std::uint32_t lc_hidden{0};  ///< 1024
  std::uint32_t lc_top_k{0};   ///< 8 candidates per drafted slot
  std::uint32_t lc_heads{0};   ///< 8
  std::uint32_t lc_layers{0};  ///< 2
  std::uint32_t lc_factor{0};  ///< 1024
  std::uint32_t lc_mlp{0};     ///< 2048
  float lc_vector_eps{1e-4F};
  float lc_logit_scale{8.0F};

  /// Drafted slots per block, also the most tokens one round can draft.
  [[nodiscard]] std::uint32_t Slots() const noexcept { return block_size - 1; }
  [[nodiscard]] std::uint32_t FeatureSize() const noexcept {
    return static_cast<std::uint32_t>(target_layers.size()) * hidden_size;
  }
  /// Rows of a conv projection: per side and tap, one weight per group.
  [[nodiscard]] std::uint32_t ConvProjection() const noexcept {
    return 2 * conv_kernel * (hidden_size / conv_group);
  }
  [[nodiscard]] std::uint32_t QDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t KvDim() const noexcept {
    return num_kv_heads * head_dim;
  }

  /// Reads and validates the `dflash.*` and `lilicorr.*` metadata against the
  /// trunk it drafts for.
  [[nodiscard]] static std::optional<DrafterConfig> FromGguf(
      const core::GgufReader& reader, const Config& target,
      std::string* error_msg = nullptr);
};

}  // namespace gufo::models::nemotron35_bolt

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_CONFIG_HPP_
