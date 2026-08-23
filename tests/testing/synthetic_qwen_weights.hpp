#pragma once

// Synthetic Qwen weights builder for the Phase 3 L1 module-test tier.
//
// QwenModelWeights holds NON-OWNING QwenTensorRef that, in production, point
// directly into GgufReader-mapped GGUF storage (see QwenModelWeights::
// LoadFromGguf). A module test cannot rely on a reader, so we instead build a
// QwenModelWeights whose refs point at an owned, deterministic, in-memory
// buffer. This header owns that backing storage (SyntheticQwenWeights::pool)
// and guarantees the refs stay valid for the lifetime of the holder. It relies
// on no GgufReader / model file.
//
// Purely additive Phase 1. Header-only + inline (no CMake wiring yet). It
// depends only on src headers (model_config.hpp, qwen_state.hpp) and the test
// RNG helpers in tests/testing/test_common.hpp, so it can be included by both
// CPU-only and HIP module test binaries.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "src/core/model_config.hpp"
#include "src/models/qwen_state.hpp"
#include "tests/testing/test_common.hpp"

namespace strix::models::qwen {

/// A small ModelConfig satisfying core::ModelConfig::IsValidQwen() (the full/
/// SSM repeating-block structure with a 4-block interval, 128-wide SSM value
/// space, etc.). Sized for a sub-second synthetic model: hidden 128, 4 layers.
inline core::ModelConfig make_small_qwen_config() {
  core::ModelConfig c;
  c.architecture = "qwen35";
  c.model_name = "synthetic-small";
  c.num_layers = 4;
  c.hidden_size = 128;
  c.intermediate_size = 256;
  c.num_attention_heads = 2;
  c.num_key_value_heads = 2;
  c.head_dim = 64;
  c.vocab_size = 256;
  c.context_length = 32;
  c.full_attention_interval = 4;
  c.mtp_num_layers = 0;
  c.ssm_conv_kernel = 4;
  c.ssm_state_size = 128;
  c.ssm_group_count = 2;
  c.ssm_time_step_rank = 1;
  c.ssm_inner_size = 128;
  c.rotary_dim = 64;
  c.rope_theta = 10000000.0F;
  c.rope_scale = 1.0F;
  c.is_text_only = true;
  return c;
}

/// Owns the backing buffer + the built QwenModelWeights. The weights'
/// QwenTensorRef all point into `pool`, so the holder must outlive any use of
/// `weights`. Non-copyable (refs bind to pool); movable (pool heap pointer is
/// stable across a vector move).
struct SyntheticQwenWeights {
  core::ModelConfig config;
  std::vector<float> pool;
  QwenModelWeights weights;

  SyntheticQwenWeights() = default;
  SyntheticQwenWeights(const SyntheticQwenWeights&) = delete;
  SyntheticQwenWeights& operator=(const SyntheticQwenWeights&) = delete;
  SyntheticQwenWeights(SyntheticQwenWeights&&) = default;
  SyntheticQwenWeights& operator=(SyntheticQwenWeights&&) = default;
};

/// Builds a QwenModelWeights for `config` whose tensor refs point at
/// deterministic, owned F32 storage. Uses the same full/attention layer-kind
/// rule as QwenModelWeights::LoadFromGguf (`((i+1) % interval) == 0`). The LM
/// head is tied to the token embeddings (matches the LoadFromGguf fallback).
inline SyntheticQwenWeights build_synthetic_qwen_weights(
    const core::ModelConfig& config, std::uint32_t seed = 0xC0FFEEu) {
  SyntheticQwenWeights out;
  out.config = config;
  out.weights.config = config;

  auto rng = strix::test::make_seeded_rng(seed);

  // First pass: compute the exact number of float elements needed so the pool
  // can be resized once (pool.data() is then stable for the holder's lifetime).
  const std::size_t hidden = config.hidden_size;
  const std::size_t vocab = config.vocab_size;
  const std::size_t inter = config.intermediate_size;
  std::size_t total = (vocab * hidden) + hidden;  // token_embd + output_norm
  // (output is tied to token_embd: no separate buffer)
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const bool full = ((i + 1) % config.full_attention_interval) == 0;
    total += hidden + hidden;                        // attn_norm, ffn_norm
    total += inter * hidden + inter * hidden;        // ffn_gate, ffn_up
    total += hidden * inter;                         // ffn_down
    if (full) {
      const std::size_t attn = config.AttentionSize();
      const std::size_t kv =
          static_cast<std::size_t>(config.num_key_value_heads) *
          config.head_dim;
      total += 2 * attn * hidden;                    // attn_q
      total += kv * hidden + kv * hidden;            // attn_k, attn_v
      total += hidden * attn;                        // attn_output
      total += config.head_dim + config.head_dim;    // attn_q/k_norm
    } else {
      const std::size_t qkv = config.SsmQkvSize();
      const std::size_t inner = config.ssm_inner_size;
      const std::size_t rank = config.ssm_time_step_rank;
      total += qkv * hidden;                             // attn_qkv
      total += inner * hidden;                           // attn_gate
      total += rank;                                     // ssm_a
      total += qkv * config.ssm_conv_kernel;             // ssm_conv1d
      total += rank;                                     // ssm_dt
      total += rank * hidden + rank * hidden;            // ssm_alpha, ssm_beta
      total += config.SsmValueSize();                    // ssm_norm
      total += hidden * inner;                           // ssm_out
    }
  }

  out.pool.assign(total, 0.0F);

  std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
  std::size_t cur = 0;
  auto alloc = [&](std::size_t n) {
    if (cur + n > out.pool.size()) {
      // Should never happen given the first-pass total.
      throw std::length_error("synthetic weights: pool overflow");
    }
    for (std::size_t i = 0; i < n; ++i) out.pool[cur + i] = dist(rng);
    QwenTensorRef ref{out.pool.data() + cur, core::GgmlType::kF32, n};
    cur += n;
    return ref;
  };

  out.weights.token_embd = alloc(vocab * hidden);
  out.weights.output_norm = alloc(hidden);
  out.weights.output = out.weights.token_embd;  // tied LM head

  out.weights.layers.resize(config.num_layers);
  for (std::uint32_t i = 0; i < config.num_layers; ++i) {
    const bool full = ((i + 1) % config.full_attention_interval) == 0;
    auto& l = out.weights.layers[i];
    l.is_full_attention = full;

    l.attn_norm = alloc(hidden);
    l.ffn_norm = alloc(hidden);
    l.ffn_gate = alloc(inter * hidden);
    l.ffn_up = alloc(inter * hidden);
    l.ffn_down = alloc(hidden * inter);

    if (full) {
      const std::size_t attn = config.AttentionSize();
      const std::size_t kv =
          static_cast<std::size_t>(config.num_key_value_heads) *
          config.head_dim;
      l.attn_q = alloc(2 * attn * hidden);
      l.attn_k = alloc(kv * hidden);
      l.attn_v = alloc(kv * hidden);
      l.attn_output = alloc(hidden * attn);
      l.attn_q_norm = alloc(config.head_dim);
      l.attn_k_norm = alloc(config.head_dim);
    } else {
      const std::size_t qkv = config.SsmQkvSize();
      const std::size_t inner = config.ssm_inner_size;
      const std::size_t rank = config.ssm_time_step_rank;
      l.attn_qkv = alloc(qkv * hidden);
      l.attn_gate = alloc(inner * hidden);
      l.ssm_a = alloc(rank);
      l.ssm_conv1d = alloc(qkv * config.ssm_conv_kernel);
      l.ssm_dt = alloc(rank);
      l.ssm_alpha = alloc(rank * hidden);
      l.ssm_beta = alloc(rank * hidden);
      l.ssm_norm = alloc(config.SsmValueSize());
      l.ssm_out = alloc(hidden * inner);
    }
  }

  return out;
}

}  // namespace strix::models::qwen
