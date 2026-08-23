#include "src/models/qwen_forward.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/modules/attention.hpp"
#include "src/models/qwen/modules/embed.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "src/models/qwen/modules/rope.hpp"
#include "src/models/qwen/modules/unembed.hpp"
#include "src/models/qwen_oracles.hpp"

namespace strix::models {
namespace {

// Packed-row byte stride for the six canonical quantized block types.
// Mirrors quant::QuantizedRowBytes but also covers kQ8_K (which the older
// helper omits), so kQ8_K routes through the unified quant:: dispatch.
[[nodiscard]] std::size_t PackedQuantizedRowBytes(core::GgmlType type,
                                                  std::size_t columns) noexcept {
  std::size_t block_qk = 0;
  std::size_t block_bytes = 0;
  switch (type) {
    case core::GgmlType::kQ3_K:
      block_qk = 256;
      block_bytes = sizeof(strix::quant::block_q3_K);
      break;
    case core::GgmlType::kQ4_K:
      block_qk = 256;
      block_bytes = sizeof(strix::quant::block_q4_K);
      break;
    case core::GgmlType::kQ5_K:
      block_qk = 256;
      block_bytes = sizeof(strix::quant::block_q5_K);
      break;
    case core::GgmlType::kQ6_K:
      block_qk = 256;
      block_bytes = sizeof(strix::quant::block_q6_K);
      break;
    case core::GgmlType::kQ8_K:
      block_qk = 256;
      block_bytes = sizeof(strix::quant::block_q8_K);
      break;
    case core::GgmlType::kQ8_0:
      block_qk = 32;
      block_bytes = sizeof(strix::quant::block_q8_0);
      break;
    default:
      return 0;
  }
  if ((columns % block_qk) != 0) {
    return 0;
  }
  return (columns / block_qk) * block_bytes;
}

const void* QuantizedRow(const QwenTensorRef& tensor, std::size_t row,
                         std::size_t columns) noexcept {
  const std::size_t row_bytes = PackedQuantizedRowBytes(tensor.type, columns);
  if (row_bytes == 0) {
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(tensor.data) + (row * row_bytes);
}

// Kept for the transition; the cut-over routes TensorGEMV directly through the
// canonical quant::DotProductQ* helpers, so this is now a dead-code candidate.
[[maybe_unused]] float QuantizedDot(const QwenTensorRef& tensor, const void* row,
                                    std::span<const float> input,
                                    std::size_t columns) noexcept {
  switch (tensor.type) {
    case core::GgmlType::kQ3_K:
      return quant::DotProductQ3_K(row, input, columns);
    case core::GgmlType::kQ4_K:
      return quant::DotProductQ4_K(row, input, columns);
    case core::GgmlType::kQ6_K:
      return quant::DotProductQ6_K(row, input, columns);
    default:
      return 0.0F;
  }
}

void DequantizeRow(const QwenTensorRef& tensor, const void* row, float* output,
                   std::size_t columns) noexcept {
  switch (tensor.type) {
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(row, output, columns);
      break;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(row, output, columns);
      break;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(row, output, columns);
      break;
    default:
      break;
  }
}

}  // namespace

void TensorGEMV(const QwenTensorRef& A, std::span<const float> x, std::size_t M,
                std::size_t K, std::span<float> y) noexcept {
  if (A.empty() || x.size() < K || y.size() < M) {
    return;
  }
  if (A.type == core::GgmlType::kF32) {
    const auto* ptr = static_cast<const float*>(A.data);
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < M; ++m) {
      const auto* row = ptr + (m * K);
      float dot = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        dot += row[k] * x[k];
      }
      y[m] = dot;
    }
  } else if (A.type == core::GgmlType::kBF16) {
    const auto* ptr = static_cast<const std::uint16_t*>(A.data);
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < M; ++m) {
      const auto* row = ptr + (m * K);
      float dot = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        const std::uint32_t u32 = static_cast<std::uint32_t>(row[k]) << 16;
        float val = 0.0F;
        std::memcpy(&val, &u32, sizeof(float));
        dot += val * x[k];
      }
      y[m] = dot;
    }
  } else if (A.type == core::GgmlType::kF16) {
    const auto* ptr = static_cast<const std::uint16_t*>(A.data);
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < M; ++m) {
      const auto* row = ptr + (m * K);
      float dot = 0.0F;
      for (std::size_t k = 0; k < K; ++k) {
        dot += quant::Fp16ToFloat(row[k]) * x[k];
      }
      y[m] = dot;
    }
  } else if (PackedQuantizedRowBytes(A.type, K) != 0) {
    // Canonical parity-tested dot helpers (the routines the unified quant::Dot
    // dispatch wraps). All six quantized types route here; the guard above
    // guarantees one of them, so genuinely unsupported types never reach the
    // switch and instead fall to the loud-fail below.
#pragma omp parallel for schedule(static)
    for (std::size_t m = 0; m < M; ++m) {
      const void* row = QuantizedRow(A, m, K);
      float dot = 0.0F;
      switch (A.type) {
        case core::GgmlType::kQ3_K:
          dot = quant::DotProductQ3_K(row, x, K);
          break;
        case core::GgmlType::kQ4_K:
          dot = quant::DotProductQ4_K(row, x, K);
          break;
        case core::GgmlType::kQ5_K:
          dot = quant::DotProductQ5_K(row, x, K);
          break;
        case core::GgmlType::kQ6_K:
          dot = quant::DotProductQ6_K(row, x, K);
          break;
        case core::GgmlType::kQ8_K:
          dot = quant::DotProductQ8_K(row, x, K);
          break;
        case core::GgmlType::kQ8_0:
          dot = quant::DotProductQ8_0(row, x, K);
          break;
        default:
          assert(false && "TensorGEMV: unsupported quantized GgmlType");
          std::abort();
      }
      y[m] = dot;
    }
  } else {
    // Unsupported (or non-block-aligned) type: fail loudly instead of the old
    // silent 0.0F / uninitialized fallthrough.
    assert(false && "TensorGEMV: unsupported GgmlType");
    std::abort();
  }
}

void ForwardEmbedding(std::uint32_t token_id, const QwenTensorRef& token_embd,
                      std::size_t hidden_size,
                      std::span<float> hidden_out) noexcept {
  const std::size_t offset = static_cast<std::size_t>(token_id) * hidden_size;
  if (offset + hidden_size <= token_embd.num_elements &&
      hidden_out.size() >= hidden_size) {
    if (token_embd.type == core::GgmlType::kF32) {
      const auto* ptr = static_cast<const float*>(token_embd.data) + offset;
      std::copy_n(ptr, hidden_size, hidden_out.data());
    } else if (token_embd.type == core::GgmlType::kBF16) {
      const auto* ptr =
          static_cast<const std::uint16_t*>(token_embd.data) + offset;
      for (std::size_t i = 0; i < hidden_size; ++i) {
        const std::uint32_t u32 = static_cast<std::uint32_t>(ptr[i]) << 16;
        float f = 0.0F;
        std::memcpy(&f, &u32, sizeof(float));
        hidden_out[i] = f;
      }
    } else if (quant::QuantizedRowBytes(token_embd.type, hidden_size) != 0) {
      const void* row = QuantizedRow(token_embd, token_id, hidden_size);
      DequantizeRow(token_embd, row, hidden_out.data(), hidden_size);
    }
  }
}

void ForwardRMSNorm(std::span<const float> x, const QwenTensorRef& weight,
                    float eps, std::span<float> out) noexcept {
  // Thin wrapper: the norm body now lives in the norm module's CPU backend
  // (NormForward). This free function is kept as-is for the existing CPU
  // callers (ForwardLayer / ForwardSSM / MTP reference) so behavior is
  // unchanged. NormForward ignores the (defaulted) ModuleCtx fields and just
  // reproduces the RMSNorm computation above.
  qwen::NormLayerView view{weight, eps};
  qwen::ModuleCtx ctx{};
  qwen::NormForward(ctx, view, x, out);
}

void ForwardRoPE(std::span<float> q, std::span<float> k,
                 std::uint32_t num_heads, std::uint32_t num_kv_heads,
                 std::uint32_t head_dim, std::uint32_t rotary_dim,
                 std::uint32_t pos, float rope_theta) noexcept {
  // Thin wrapper: the RoPE body now lives in the rope module's CPU backend
  // (RopeForward). This free function is kept for the existing CPU callers
  // (ForwardLayer / qwen_forward_test) so behavior is unchanged.
  qwen::RopeLayerView view{num_heads, num_kv_heads, head_dim, rotary_dim,
                           rope_theta};
  qwen::ModuleCtx ctx{};
  qwen::RopeForward(ctx, view, q, k, pos);
}

void ForwardAttention(std::span<const float> q, std::span<const float> k,
                      std::span<const float> v, std::span<const float> gate,
                      const QwenTensorRef& o_weight, QwenKvCache& kv_cache,
                      std::uint32_t layer_idx, std::uint32_t pos,
                      std::uint32_t num_heads, std::uint32_t num_kv_heads,
                      std::uint32_t head_dim, std::size_t hidden_size,
                      std::span<float> attn_scores_scratch,
                      std::span<float> attn_out) noexcept {
  for (std::uint32_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
    const auto k_src =
        k.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);
    const auto v_src =
        v.subspan(static_cast<std::size_t>(kv_h) * head_dim, head_dim);

    const auto k_dst = kv_cache.GetKeySlice(layer_idx, kv_h, pos);
    const auto v_dst = kv_cache.GetValueSlice(layer_idx, kv_h, pos);

    std::ranges::copy(k_src, k_dst.begin());
    std::ranges::copy(v_src, v_dst.begin());
  }

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  std::vector<float> context(static_cast<std::size_t>(num_heads) * head_dim,
                             0.0F);

  for (std::uint32_t h = 0; h < num_heads; ++h) {
    const std::uint32_t kv_h = h / group_size;
    const auto q_head =
        q.subspan(static_cast<std::size_t>(h) * head_dim, head_dim);

    float max_score = -1e30F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto k_p =
          kv_cache.GetKeySlice(layer_idx, kv_h, static_cast<std::uint32_t>(p));
      double dot = 0.0;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q_head[d]) * static_cast<double>(k_p[d]);
      }
      const auto score = static_cast<float>(dot * scale);
      attn_scores_scratch[p] = score;
      max_score = std::max(score, max_score);
    }

    double sum_exp = 0.0;
    for (std::size_t p = 0; p < seq_len; ++p) {
      sum_exp +=
          std::exp(static_cast<double>(attn_scores_scratch[p] - max_score));
    }
    const double inv_sum = 1.0 / sum_exp;
    for (std::size_t p = 0; p < seq_len; ++p) {
      attn_scores_scratch[p] = static_cast<float>(
          std::exp(static_cast<double>(attn_scores_scratch[p] - max_score)) *
          inv_sum);
    }

    auto ctx_head = std::span<float>(
        &context[static_cast<std::size_t>(h) * head_dim], head_dim);
    std::ranges::fill(ctx_head, 0.0F);

    for (std::size_t p = 0; p < seq_len; ++p) {
      const auto v_p = kv_cache.GetValueSlice(layer_idx, kv_h,
                                              static_cast<std::uint32_t>(p));
      const auto weight = static_cast<double>(attn_scores_scratch[p]);
      for (std::size_t d = 0; d < head_dim; ++d) {
        ctx_head[d] =
            static_cast<float>(static_cast<double>(ctx_head[d]) +
                               (weight * static_cast<double>(v_p[d])));
      }
    }
  }

  // Attention gating (Sigmoid gating for Qwen 3.5 full attention)
  if (!gate.empty() && gate.size() >= context.size()) {
    for (std::size_t i = 0; i < context.size(); ++i) {
      const float g = gate[i];
      const float sig = 1.0F / (1.0F + std::exp(-g));
      context[i] *= sig;
    }
  }

  const std::size_t ctx_dim = static_cast<std::size_t>(num_heads) * head_dim;
  if (!o_weight.empty()) {
    TensorGEMV(o_weight, context, hidden_size, ctx_dim, attn_out);
  } else {
    std::copy_n(context.data(), std::min(hidden_size, ctx_dim),
                attn_out.data());
  }
}

void ForwardFFN(std::span<const float> x, const QwenTensorRef& gate_weight,
                const QwenTensorRef& up_weight,
                const QwenTensorRef& down_weight, std::size_t hidden_size,
                std::size_t intermediate_size, std::span<float> gate_scratch,
                std::span<float> up_scratch, std::span<float> act_scratch,
                std::span<float> ffn_out) noexcept {
  // Thin wrapper: the SwiGLU FFN body now lives in the ffn module's CPU
  // backend (FfnForward). Kept for the existing CPU callers (ForwardLayer /
  // qwen_forward_test) so behavior is unchanged.
  qwen::FfnLayerView view{gate_weight, up_weight, down_weight, hidden_size,
                          intermediate_size};
  qwen::ModuleCtx ctx{};
  qwen::FfnForward(ctx, view, x, gate_scratch, up_scratch, act_scratch,
                   ffn_out);
}

void ForwardLayer(std::span<float> hidden, const QwenLayerWeights& layer,
                  const core::ModelConfig& config, QwenKvCache& kv_cache,
                  QwenSsmCache& ssm_cache, std::uint32_t layer_idx,
                  std::uint32_t pos, QwenScratchArena& arena) noexcept {
  const std::size_t hidden_size = config.hidden_size;

  // 1. Pre-RMSNorm
  ForwardRMSNorm(hidden, layer.attn_norm, 1e-6F, arena.normed);

  // 2. Self-Attention / SSM
  if (layer.is_full_attention) {
    qwen::ModuleCtx actx{};
    actx.config = &config;
    actx.arena = &arena;
    actx.layer_idx = layer_idx;
    actx.pos = pos;
    qwen::AttnLayerView av = qwen::MakeAttnView(layer, config);
    qwen::AttnForward(actx, av, arena.normed, kv_cache, pos, arena.attn_out);
  } else {
    ForwardSSM(arena.normed, layer, config, ssm_cache, layer_idx, arena.ssm_qkv,
               arena.ssm_gate, arena.ssm_out_buf, arena.attn_out);
  }

  // 3. Residual Add (composition layer drives the residual module)
  qwen::ModuleCtx ctx{};
  qwen::ResidualAdd(ctx, hidden, arena.attn_out);

  // 4. FFN Pre-RMSNorm
  ForwardRMSNorm(hidden, layer.ffn_norm, 1e-6F, arena.normed);

  // 5. SwiGLU FFN
  ForwardFFN(arena.normed, layer.ffn_gate, layer.ffn_up, layer.ffn_down,
             hidden_size, config.intermediate_size, arena.mlp_gate,
             arena.mlp_up, arena.mlp_act, arena.mlp_out);

  // 6. Residual Add
  qwen::ResidualAdd(ctx, hidden, arena.mlp_out);
}

void ForwardModel(std::uint32_t token_id, std::uint32_t pos,
                  const QwenModelWeights& weights, QwenKvCache& kv_cache,
                  QwenSsmCache& ssm_cache, QwenScratchArena& arena,
                  std::span<float> logits_out) noexcept {
  qwen::ModuleCtx ctx{};
  ctx.config = &weights.config;
  ctx.arena = &arena;
  qwen::EmbedForward(ctx, token_id, weights.token_embd,
                     weights.config.hidden_size, arena.hidden);

  for (std::uint32_t l = 0; l < weights.config.num_layers; ++l) {
    ForwardLayer(arena.hidden, weights.layers[l], weights.config, kv_cache,
                 ssm_cache, l, pos, arena);
  }

  qwen::UnembedForward(ctx, weights.output_norm, weights.output, arena.hidden,
                       logits_out);
}

std::uint32_t GreedyArgmax(std::span<const float> logits) noexcept {
  if (logits.empty()) {
    return 0;
  }
  std::size_t best_idx = 0;
  float best_val = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return static_cast<std::uint32_t>(best_idx);
}

}  // namespace strix::models
