#pragma once

#include <array>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <string>

namespace gufo::models::inkling2 {

/// The one hyper-parameter set the extension serves. Defaults are the exact
/// values of Inkling 2 Mini; `ParseConfig` refuses a file that differs.
struct Inkling2Config {
  static constexpr std::uint32_t kPhysLayers = 24;

  // Trunk.
  std::uint32_t n_phys = kPhysLayers;
  std::uint32_t n_visits = 48;  // layer applications per token
  std::uint32_t n_embd = 2304;
  std::uint32_t n_head = 24;
  std::uint32_t n_head_kv = 8;
  std::uint32_t head_dim = 128;
  std::uint32_t n_vocab = 201024;       // padded embedding rows
  std::uint32_t n_vocab_text = 200058;  // tokenizer size; logits are cropped
  std::uint32_t n_ctx_train = 1048576;
  float rms_eps = 1e-6F;
  float logit_softcap = 30.0F;

  // MoE.
  std::uint32_t n_expert = 128;
  std::uint32_t n_expert_used = 8;
  std::uint32_t n_ff_expert = 1152;
  float expert_weights_scale = 8.0F;
  float situ_beta = 4.0F;
  float situ_linear_beta = 25.0F;

  // Attention: no RoPE, content-conditioned relative bias.
  std::uint32_t n_swa = 1024;
  std::uint32_t rel_dim = 16;
  std::uint32_t rel_extent = 1024;
  float qk_clip = 22.627417F;
  float rel_clip = 8.0F;
  std::uint32_t log_scale_floor = 8192;
  float log_scale_alpha = 0.1F;
  std::uint32_t conv_kernel = 4;
  std::uint32_t attn_res_block = 6;  // layers per attention-residual block

  // Loop: layers [core_start, core_end) run `n_passes` times.
  std::uint32_t core_start = 6;
  std::uint32_t core_end = 18;
  std::uint32_t n_passes = 3;
  std::uint32_t gate_rank = 64;

  /// True for the full-attention layers (5, 11, 17, 23).
  std::array<bool, kPhysLayers> global_layer{};

  [[nodiscard]] std::uint32_t CoreLen() const { return core_end - core_start; }
  [[nodiscard]] bool IsCore(std::uint32_t il) const {
    return il >= core_start && il < core_end;
  }
  /// Physical layer applied at visit `v` (0 <= v < n_visits).
  [[nodiscard]] std::uint32_t PhysOfVisit(std::uint32_t v) const;
};

/// Reads and validates the metadata of `reader`. Returns false and sets
/// `*why` (when given) to the first difference from Inkling 2 Mini.
[[nodiscard]] bool ParseConfig(const core::GgufReader& reader,
                               Inkling2Config* out, std::string* why);

}  // namespace gufo::models::inkling2
