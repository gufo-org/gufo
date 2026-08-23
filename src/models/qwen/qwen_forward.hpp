#ifndef STRIX_MODELS_QWEN_FORWARD_HPP_
#define STRIX_MODELS_QWEN_FORWARD_HPP_

#include <cstddef>
#include <cstdint>
#include <span>

#include "src/models/qwen/qwen_ssm.hpp"
#include "src/models/qwen/qwen_state.hpp"

namespace strix::models {

/// Performs GEMV: y = A @ x supporting both F32 and BF16 weights.
void TensorGEMV(const QwenTensorRef& A, std::span<const float> x, std::size_t M,
                std::size_t K, std::span<float> y) noexcept;

/// Copies embedding weights for the given token_id into hidden_out.
void ForwardEmbedding(std::uint32_t token_id, const QwenTensorRef& token_embd,
                      std::size_t hidden_size,
                      std::span<float> hidden_out) noexcept;

/// Computes RMSNorm: out = (x / rms(x)) * weight.
void ForwardRMSNorm(std::span<const float> x, const QwenTensorRef& weight,
                    float eps, std::span<float> out) noexcept;

/// Computes RoPE rotation on Q and K heads for a given position.
void ForwardRoPE(std::span<float> q, std::span<float> k,
                 std::uint32_t num_heads, std::uint32_t num_kv_heads,
                 std::uint32_t head_dim, std::uint32_t rotary_dim,
                 std::uint32_t pos, float rope_theta) noexcept;

/// Computes Grouped-Query Attention with KV-cache and optional gating for a
/// single sequence position.
void ForwardAttention(std::span<const float> q, std::span<const float> k,
                      std::span<const float> v, std::span<const float> gate,
                      const QwenTensorRef& o_weight, QwenKvCache& kv_cache,
                      std::uint32_t layer_idx, std::uint32_t pos,
                      std::uint32_t num_heads, std::uint32_t num_kv_heads,
                      std::uint32_t head_dim, std::size_t hidden_size,
                      std::span<float> attn_scores_scratch,
                      std::span<float> attn_out) noexcept;

/// Computes SwiGLU FFN: out = (SiLU(gate) * up) * down.
void ForwardFFN(std::span<const float> x, const QwenTensorRef& gate_weight,
                const QwenTensorRef& up_weight,
                const QwenTensorRef& down_weight, std::size_t hidden_size,
                std::size_t intermediate_size, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> ffn_out) noexcept;

/// Computes one forward transformer layer block.
void ForwardLayer(std::span<float> hidden, const QwenLayerWeights& layer,
                  const core::ModelConfig& config, QwenKvCache& kv_cache,
                  QwenSsmCache& ssm_cache, std::uint32_t layer_idx,
                  std::uint32_t pos, QwenScratchArena& arena) noexcept;

/// Computes a full model forward pass on a single token, producing output
/// logits.
void ForwardModel(std::uint32_t token_id, std::uint32_t pos,
                  const QwenModelWeights& weights, QwenKvCache& kv_cache,
                  QwenSsmCache& ssm_cache, QwenScratchArena& arena,
                  std::span<float> logits_out) noexcept;

/// Computes greedy argmax over logit distribution.
[[nodiscard]] std::uint32_t GreedyArgmax(
    std::span<const float> logits) noexcept;

}  // namespace strix::models

#endif  // STRIX_MODELS_QWEN_FORWARD_HPP_
