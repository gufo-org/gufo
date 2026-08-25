#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/dflash.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen/dflash_reference.hpp"
#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen/hip/mtp/detail/allocation.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace strix::hip {
namespace {

template<typename T>
void AllocateBuffer(T*& pointer, std::size_t elements) {
  pointer = static_cast<T*>(detail::AllocateDevice(elements * sizeof(T)));
}

}  // namespace

QwenDFlashGpuExecutor::QwenDFlashGpuExecutor(
    std::shared_ptr<const QwenDFlashGpuModel> model,
    std::uint32_t max_context)
    : model_(std::move(model)),
      max_context_(max_context),
      h_target_features_(model_->GetDFlashConfig().target_layer_ids.size() *
                         model_->GetConfig().hidden_size),
      h_logits_(model_->GetDFlashConfig().block_size *
                model_->GetConfig().vocab_size) {
  try {
    Allocate();
    Reset();
  } catch (...) {
    Free();
    throw;
  }
}

QwenDFlashGpuExecutor::~QwenDFlashGpuExecutor() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  Free();
}

std::unique_ptr<QwenDFlashGpuExecutor> QwenDFlashGpuExecutor::Create(
    std::shared_ptr<const QwenDFlashGpuModel> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0 ||
      max_context > model->GetConfig().context_length) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash GPU executor context is invalid";
    }
    return nullptr;
  }
  try {
    return std::unique_ptr<QwenDFlashGpuExecutor>(
        new QwenDFlashGpuExecutor(std::move(model), max_context));
  } catch (const std::exception& exception) {
    if (error_msg != nullptr) {
      *error_msg = exception.what();
    }
    return nullptr;
  }
}

void QwenDFlashGpuExecutor::Allocate() {
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t block_size = df_cfg.block_size;
  const std::size_t num_layers = df_cfg.num_layers;
  const std::size_t enc_in_dim = df_cfg.target_layer_ids.size() * hidden_size;

  const auto error = hipStreamCreate(&stream_);
  if (error != hipSuccess) {
    throw std::runtime_error("DFlash hipStreamCreate failed");
  }

  // Allocate KV caches per layer
  d_injected_k_.resize(num_layers, nullptr);
  d_injected_v_.resize(num_layers, nullptr);
  for (std::size_t i = 0; i < num_layers; ++i) {
    AllocateBuffer(d_injected_k_[i], max_context_ * kv_dim);
    AllocateBuffer(d_injected_v_[i], max_context_ * kv_dim);
  }

  // Scratch buffers for prompt injection & block drafting
  AllocateBuffer(d_target_features_, max_context_ * enc_in_dim);
  AllocateBuffer(d_fused_features_, max_context_ * hidden_size);
  AllocateBuffer(d_block_normed_, max_context_ * hidden_size);
  AllocateBuffer(d_k_block_, max_context_ * kv_dim);
  AllocateBuffer(d_v_block_, max_context_ * kv_dim);

  // Scratch buffers for block drafting (size = block_size)
  AllocateBuffer(d_block_hidden_, block_size * hidden_size);
  AllocateBuffer(d_q_, block_size * q_dim);
  AllocateBuffer(d_attn_out_, block_size * hidden_size);
  AllocateBuffer(d_ffn_gate_, block_size * intermediate_size);
  AllocateBuffer(d_ffn_up_, block_size * intermediate_size);
  AllocateBuffer(d_ffn_down_, block_size * hidden_size);
  AllocateBuffer(d_logits_, block_size * cfg.vocab_size);
  AllocateBuffer(d_out_token_, block_size);
  AllocateBuffer(d_argmax_out_, block_size);
}

void QwenDFlashGpuExecutor::Free() noexcept {
  for (auto*& ptr : d_injected_k_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  for (auto*& ptr : d_injected_v_) {
    if (ptr != nullptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  if (d_target_features_ != nullptr) (void)hipFree(d_target_features_);
  if (d_fused_features_ != nullptr) (void)hipFree(d_fused_features_);
  if (d_block_hidden_ != nullptr) (void)hipFree(d_block_hidden_);
  if (d_block_normed_ != nullptr) (void)hipFree(d_block_normed_);
  if (d_q_ != nullptr) (void)hipFree(d_q_);
  if (d_k_block_ != nullptr) (void)hipFree(d_k_block_);
  if (d_v_block_ != nullptr) (void)hipFree(d_v_block_);
  if (d_attn_out_ != nullptr) (void)hipFree(d_attn_out_);
  if (d_ffn_gate_ != nullptr) (void)hipFree(d_ffn_gate_);
  if (d_ffn_up_ != nullptr) (void)hipFree(d_ffn_up_);
  if (d_ffn_down_ != nullptr) (void)hipFree(d_ffn_down_);
  if (d_logits_ != nullptr) (void)hipFree(d_logits_);
  if (d_out_token_ != nullptr) (void)hipFree(d_out_token_);
  if (d_argmax_out_ != nullptr) (void)hipFree(d_argmax_out_);
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void QwenDFlashGpuExecutor::Reset() noexcept {
  injected_context_len_ = 0;
  const std::size_t kv_dim =
      static_cast<std::size_t>(model_->GetConfig().num_key_value_heads) *
      model_->GetConfig().head_dim;
  for (auto* ptr : d_injected_k_) {
    if (ptr != nullptr) {
      (void)hipMemsetAsync(ptr, 0, max_context_ * kv_dim * sizeof(float), stream_);
    }
  }
  for (auto* ptr : d_injected_v_) {
    if (ptr != nullptr) {
      (void)hipMemsetAsync(ptr, 0, max_context_ * kv_dim * sizeof(float), stream_);
    }
  }
}

bool QwenDFlashGpuExecutor::InjectTargetContext(
    std::span<const float> target_features, std::uint32_t position,
    std::uint32_t num_tokens) {
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t enc_in_dim = GetTargetFeaturesSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const bool is_bf16 = (weights.fc_projection.type == core::GgmlType::kBF16);

  if (target_features.size() != num_tokens * enc_in_dim ||
      position + num_tokens > max_context_) {
    return false;
  }

  // Upload target features
  const std::size_t bytes = target_features.size() * sizeof(float);
  const auto cpy_err =
      hipMemcpyAsync(d_target_features_, target_features.data(), bytes,
                     hipMemcpyHostToDevice, stream_);
  if (cpy_err != hipSuccess) {
    return false;
  }

  // FC Projection & Norm
  if (weights.fc_projection.data != nullptr) {
    LaunchBatchedGEMM(weights.fc_projection.data, is_bf16, d_target_features_,
                      d_fused_features_, num_tokens, hidden_size, enc_in_dim,
                      stream_);
  }
  LaunchBatchedRMSNorm(d_fused_features_,
                       static_cast<const float*>(weights.fc_norm.data),
                       d_block_normed_, nullptr, num_tokens, hidden_size, 1e-6F,
                       stream_);

  // Project and store K / V for each draft layer
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const auto& layer = weights.layers[i];
    if (layer.attn_k.data != nullptr && layer.attn_v.data != nullptr) {
      LaunchBatchedGEMM(layer.attn_k.data, is_bf16, d_block_normed_, d_k_block_,
                        num_tokens, kv_dim, hidden_size, stream_);
      LaunchBatchedGEMM(layer.attn_v.data, is_bf16, d_block_normed_, d_v_block_,
                        num_tokens, kv_dim, hidden_size, stream_);
    }

    if (layer.attn_k_norm.data != nullptr) {
      LaunchBatchedPerHeadRMSNorm(d_k_block_,
                                  static_cast<const float*>(layer.attn_k_norm.data),
                                  d_k_block_, num_tokens,
                                  cfg.num_key_value_heads, cfg.head_dim, 1e-6F,
                                  stream_);
    }

    // Apply RoPE across injected tokens
    for (std::size_t t = 0; t < num_tokens; ++t) {
      LaunchRoPE(nullptr, d_k_block_ + t * kv_dim, 0,
                 cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim,
                 position + static_cast<std::uint32_t>(t), cfg.rope_theta,
                 stream_);
    }

    // Copy to injected KV cache
    const std::size_t cache_bytes = num_tokens * kv_dim * sizeof(float);
    (void)hipMemcpyAsync(d_injected_k_[i] + position * kv_dim, d_k_block_,
                         cache_bytes, hipMemcpyDeviceToDevice, stream_);
    (void)hipMemcpyAsync(d_injected_v_[i] + position * kv_dim, d_v_block_,
                         cache_bytes, hipMemcpyDeviceToDevice, stream_);
  }

  injected_context_len_ =
      std::max(injected_context_len_, position + num_tokens);
  (void)hipStreamSynchronize(stream_);
  return true;
}

std::vector<tokenization::TokenId> QwenDFlashGpuExecutor::ForwardBlock(
    tokenization::TokenId anchor_token, std::uint32_t current_pos,
    std::uint32_t draft_count, std::vector<float>* out_confidences) {
  const auto& weights = model_->GetWeights();
  const auto& cfg = model_->GetConfig();
  const auto& df_cfg = model_->GetDFlashConfig();
  const std::size_t hidden_size = cfg.hidden_size;
  const std::size_t intermediate_size = cfg.intermediate_size;
  const std::size_t num_q_heads = cfg.num_attention_heads;
  const std::size_t num_kv_heads = cfg.num_key_value_heads;
  const std::size_t head_dim = cfg.head_dim;
  const std::size_t q_dim = cfg.AttentionSize();
  const std::size_t kv_dim =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t vocab_size = cfg.vocab_size;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  constexpr bool is_bf16 = true;

  draft_count = std::min(draft_count, df_cfg.block_size);
  if (draft_count == 0) {
    return {};
  }

  // 1. Embed noise block: slot 0 is anchor token, slots 1..draft_count-1 are mask tokens
  if (weights.token_embedding.data != nullptr) {
    LaunchEmbeddingLookup(weights.token_embedding.data,
                          weights.token_embedding.type, anchor_token,
                          d_block_hidden_, hidden_size, stream_);
    for (std::size_t t = 1; t < draft_count; ++t) {
      LaunchEmbeddingLookup(weights.token_embedding.data,
                            weights.token_embedding.type, df_cfg.mask_token_id,
                            d_block_hidden_ + t * hidden_size, hidden_size,
                            stream_);
    }
  }

  // 2. DFlash-2 Input Dynamic Conv
  if (df_cfg.has_dynamic_conv && weights.in_conv_weight.data != nullptr) {
    kernels::LaunchDFlashDynamicConv2Tap(
        d_block_hidden_,
        static_cast<const float*>(weights.in_conv_weight.data),
        static_cast<const float*>(weights.in_conv_bias.data),
        d_block_hidden_,
        static_cast<std::uint32_t>(draft_count),
        static_cast<std::uint32_t>(hidden_size),
        stream_);
  }

  // 3. Draft transformer decoder blocks
  for (std::size_t i = 0; i < df_cfg.num_layers; ++i) {
    const auto& layer = weights.layers[i];

    // Pre-RMSNorm
    LaunchBatchedRMSNorm(d_block_hidden_,
                         static_cast<const float*>(layer.attn_norm.data),
                         d_block_normed_, nullptr, draft_count, hidden_size,
                         1e-6F, stream_);

    // Q/K/V Projections
    if (layer.attn_q.data != nullptr && layer.attn_k.data != nullptr &&
        layer.attn_v.data != nullptr) {
      LaunchBatchedGEMM(layer.attn_q.data, is_bf16, d_block_normed_, d_q_,
                        draft_count, q_dim, hidden_size, stream_);
      LaunchBatchedGEMM(layer.attn_k.data, is_bf16, d_block_normed_, d_k_block_,
                        draft_count, kv_dim, hidden_size, stream_);
      LaunchBatchedGEMM(layer.attn_v.data, is_bf16, d_block_normed_, d_v_block_,
                        draft_count, kv_dim, hidden_size, stream_);
    }

    if (layer.attn_q_norm.data != nullptr) {
      LaunchBatchedPerHeadRMSNorm(d_q_,
                                  static_cast<const float*>(layer.attn_q_norm.data),
                                  d_q_, draft_count, num_q_heads, head_dim,
                                  1e-6F, stream_);
    }
    if (layer.attn_k_norm.data != nullptr) {
      LaunchBatchedPerHeadRMSNorm(d_k_block_,
                                  static_cast<const float*>(layer.attn_k_norm.data),
                                  d_k_block_, draft_count, num_kv_heads,
                                  head_dim, 1e-6F, stream_);
    }

    // Apply RoPE across block tokens
    for (std::size_t t = 0; t < draft_count; ++t) {
      LaunchRoPE(d_q_ + t * q_dim, d_k_block_ + t * kv_dim, num_q_heads,
                 num_kv_heads, head_dim, head_dim,
                 current_pos + static_cast<std::uint32_t>(t), cfg.rope_theta,
                 stream_);
    }

    // Non-causal attention across injected KV and block KV
    kernels::LaunchDFlashNonCausalAttention(
        d_q_,
        d_injected_k_[i],
        d_injected_v_[i],
        d_k_block_,
        d_v_block_,
        d_attn_out_,
        current_pos,
        draft_count,
        static_cast<std::uint32_t>(num_q_heads),
        static_cast<std::uint32_t>(num_kv_heads),
        static_cast<std::uint32_t>(head_dim),
        scale,
        stream_);

    // Attention Output Projection & Residual
    if (layer.attn_output.data != nullptr) {
      LaunchBatchedGEMM(layer.attn_output.data, is_bf16, d_attn_out_,
                        d_fused_features_, draft_count, hidden_size, q_dim,
                        stream_);
      LaunchBatchedResidualAdd(d_block_hidden_, d_fused_features_,
                               d_block_hidden_, draft_count, hidden_size,
                               stream_);
    }

    // FFN Pre-RMSNorm
    LaunchBatchedRMSNorm(d_block_hidden_,
                         static_cast<const float*>(layer.ffn_norm.data),
                         d_block_normed_, nullptr, draft_count, hidden_size,
                         1e-6F, stream_);

    // SwiGLU FFN
    if (layer.ffn_gate.data != nullptr && layer.ffn_up.data != nullptr &&
        layer.ffn_down.data != nullptr) {
      LaunchBatchedGEMM(layer.ffn_gate.data, is_bf16, d_block_normed_,
                        d_ffn_gate_, draft_count, intermediate_size,
                        hidden_size, stream_);
      LaunchBatchedGEMM(layer.ffn_up.data, is_bf16, d_block_normed_, d_ffn_up_,
                        draft_count, intermediate_size, hidden_size, stream_);
      kernels::LaunchDFlashSiLUMul(d_ffn_gate_, d_ffn_up_,
                                   draft_count * intermediate_size, stream_);
      LaunchBatchedGEMM(layer.ffn_down.data, is_bf16, d_ffn_gate_, d_ffn_down_,
                        draft_count, hidden_size, intermediate_size, stream_);
      LaunchBatchedResidualAdd(d_block_hidden_, d_ffn_down_, d_block_hidden_,
                               draft_count, hidden_size, stream_);
    }
  }

  // 4. DFlash-2 Output Dynamic Conv
  if (df_cfg.has_dynamic_conv && weights.out_conv_weight.data != nullptr) {
    kernels::LaunchDFlashDynamicConv2Tap(
        d_block_hidden_,
        static_cast<const float*>(weights.out_conv_weight.data),
        static_cast<const float*>(weights.out_conv_bias.data),
        d_block_hidden_,
        static_cast<std::uint32_t>(draft_count),
        static_cast<std::uint32_t>(hidden_size),
        stream_);
  }

  // 5. Output RMSNorm & LM Head Projection
  LaunchBatchedRMSNorm(d_block_hidden_,
                       static_cast<const float*>(weights.output_norm.data),
                       d_block_normed_, nullptr, draft_count, hidden_size, 1e-6F,
                       stream_);

  std::vector<tokenization::TokenId> tokens(draft_count, 0);

  if (weights.output.data != nullptr) {
    for (std::size_t t = 0; t < draft_count; ++t) {
      LaunchGEMV(weights.output.data, weights.output.type,
                 d_block_normed_ + (t * hidden_size),
                 d_logits_ + (t * vocab_size),
                 vocab_size, hidden_size, stream_,
                 models::qwen::QwenGemmMode::kHipMtp);
      LaunchGPUArgmax(d_logits_ + (t * vocab_size), d_out_token_ + t, vocab_size,
                      stream_);
    }
    const auto cpy_err =
        hipMemcpyAsync(tokens.data(), d_out_token_,
                       draft_count * sizeof(tokenization::TokenId),
                       hipMemcpyDeviceToHost, stream_);
    if (cpy_err != hipSuccess || hipStreamSynchronize(stream_) != hipSuccess) {
      throw std::runtime_error("DFlash GPU argmax sync failed");
    }
  }

  if (out_confidences != nullptr) {
    out_confidences->assign(tokens.size(), 1.0F);
  }

  return tokens;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
