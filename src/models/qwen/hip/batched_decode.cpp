#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace gufo::hip {
namespace {

constexpr std::size_t kMaxDecodeBatch = 8;

[[nodiscard]] constexpr bool SupportsDenseFusedProjection(
    core::GgmlType type) noexcept {
  return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
}

[[nodiscard]] constexpr bool SupportsExactSharedProjection(
    core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K;
}

[[nodiscard]] bool UseExactBf16Lds8Projection() noexcept {
  static const bool enabled = [] {
    const char* value = std::getenv("GUFO_BF16_SMALL_BATCH_EXACT_LDS8");
    if (value == nullptr) {
      return true;
    }
    const std::string_view setting{value};
    return setting != "0" && setting != "false" && setting != "off";
  }();
  return enabled;
}

[[nodiscard]] bool VerifyTimingEnabled() noexcept {
  static const bool enabled = std::getenv("GUFO_VERIFY_TIMING") != nullptr;
  return enabled;
}

/// Stage rollup for the decode-equivalent verification chunk, printed at
/// process exit when GUFO_VERIFY_TIMING is set. Stage boundaries synchronize,
/// so the accounting is only wired up under the flag.
struct VerifyStageTimings {
  double projections_ms{0.0};
  double ssm_recurrence_ms{0.0};
  double attention_ms{0.0};
  double ffn_ms{0.0};
  double head_ms{0.0};
  double other_ms{0.0};
  std::uint64_t chunks{0};

  ~VerifyStageTimings() {
    if (!VerifyTimingEnabled() || chunks == 0) {
      return;
    }
    const double n = static_cast<double>(chunks);
    const double total = projections_ms + ssm_recurrence_ms + attention_ms +
                         ffn_ms + head_ms + other_ms;
    std::cerr << "\n[VERIFY_TIMING] chunks=" << chunks << '\n'
              << "  total          " << total << " ms  (" << (total / n)
              << " ms/chunk)\n"
              << "    projections  " << projections_ms << " ms  ("
              << (projections_ms / n) << ")\n"
              << "    ssm recur    " << ssm_recurrence_ms << " ms  ("
              << (ssm_recurrence_ms / n) << ")\n"
              << "    attention    " << attention_ms << " ms  ("
              << (attention_ms / n) << ")\n"
              << "    ffn          " << ffn_ms << " ms  (" << (ffn_ms / n)
              << ")\n"
              << "    lm head      " << head_ms << " ms  (" << (head_ms / n)
              << ")\n"
              << "    other        " << other_ms << " ms  (" << (other_ms / n)
              << ")\n";
  }
};

[[nodiscard]] VerifyStageTimings& VerifyTimings() {
  static VerifyStageTimings timings;
  return timings;
}

class VerifyStageTimer {
public:
  VerifyStageTimer(double* sink, hipStream_t stream)
      : sink_(VerifyTimingEnabled() ? sink : nullptr), stream_(stream) {
    if (sink_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
      start_ = std::chrono::steady_clock::now();
    }
  }

  VerifyStageTimer(const VerifyStageTimer&) = delete;
  VerifyStageTimer& operator=(const VerifyStageTimer&) = delete;
  VerifyStageTimer(VerifyStageTimer&&) = delete;
  VerifyStageTimer& operator=(VerifyStageTimer&&) = delete;

  ~VerifyStageTimer() {
    if (sink_ == nullptr) {
      return;
    }
    (void)hipStreamSynchronize(stream_);
    *sink_ += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start_)
                  .count();
  }

private:
  double* sink_{nullptr};
  hipStream_t stream_{nullptr};
  std::chrono::steady_clock::time_point start_{};
};

void LaunchProjection(const models::QwenTensorRef& weight, const float* input,
                      float* output, std::size_t batch_size,
                      std::size_t output_size, std::size_t input_size,
                      hipStream_t stream) {
  if (SupportsExactSharedProjection(weight.type)) {
    LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                               batch_size, output_size, input_size, stream);
    return;
  }
  if (weight.type == core::GgmlType::kBF16 && batch_size <= kMaxDecodeBatch &&
      UseExactBf16Lds8Projection()) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, batch_size,
                                      output_size, input_size, stream);
    return;
  }

  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchGEMV(weight.data, weight.type, input + (row * input_size),
               output + (row * output_size), output_size, input_size, stream);
  }
}

}  // namespace

std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardTokenBatch(
    std::span<const QwenGpuBatchItem> items) {
  if (items.size() < 2 || items.size() > kMaxDecodeBatch) {
    throw std::invalid_argument(
        "Qwen GPU decode batch requires two to eight sessions");
  }

  QwenGpuExecutor* coordinator = items.front().executor;
  if (coordinator == nullptr) {
    throw std::invalid_argument(
        "Qwen GPU decode batch contains a null session");
  }
  const auto* shared_model = coordinator->model_.get();
  const std::uint64_t shared_policy = coordinator->policy_.Fingerprint();
  for (std::size_t index = 0; index < items.size(); ++index) {
    QwenGpuExecutor* executor = items[index].executor;
    if (executor == nullptr || executor->model_.get() != shared_model ||
        executor->policy_.Fingerprint() != shared_policy) {
      throw std::invalid_argument(
          "Qwen GPU decode batch sessions are not execution-compatible");
    }
    if (items[index].position >= executor->arena_.GetMaxContext()) {
      throw std::length_error(
          "token position exceeds a batched Qwen session context");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor) {
        throw std::invalid_argument(
            "Qwen GPU decode batch contains a duplicate session");
      }
    }
  }
  if (items.size() > coordinator->arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen GPU decode width exceeds the coordinator arena capacity");
  }

  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (executor.replaying_ssm_state_) {
      executor.replaying_ssm_state_ = false;
      executor.arena_.DisableSsmReplayCapture();
    }
    executor.arena_.MarkSsmReplayPosition(item.position);
    executor.last_hidden_offset_ = 0;
  }

  const std::size_t batch_size = items.size();
  const auto& weights = coordinator->weights_;
  const auto& config = weights.config;
  auto& arena = coordinator->arena_;
  const auto scratch = arena.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;

  std::array<std::uint32_t, kMaxDecodeBatch> host_tokens{};
  for (std::size_t row = 0; row < batch_size; ++row) {
    host_tokens[row] = items[row].token_id;
  }
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.prompt_tokens.data(), host_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyHostToDevice, arena.stream));
  LaunchBatchedEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                               scratch.decode.prompt_tokens.data(),
                               scratch.decode.hidden.data(), batch_size,
                               hidden_size, arena.stream);

  EmitDecodeRouteTelemetry(weights, coordinator->policy_);
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto route_plan =
        ResolveQwenLayerRoute(coordinator->policy_, QwenExecutionMode::kDecode,
                              layer.is_full_attention);
    const bool fuse_ffn_norm_swiglu =
        route_plan.fuse_ffn_swiglu &&
        layer.ffn_gate.type == core::GgmlType::kBF16 &&
        layer.ffn_up.type == core::GgmlType::kBF16;

    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    bool ssm_residual_folded = false;
    if (layer.is_full_attention) {
      LaunchProjection(layer.attn_q, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, q_projection_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_k, scratch.decode.normed.data(),
                       scratch.attention.k.data(), batch_size, kv_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_v, scratch.decode.normed.data(),
                       scratch.attention.v.data(), batch_size, kv_size,
                       hidden_size, arena.stream);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena.stream);

      if (!layer.attn_q_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.q.data(),
            static_cast<const float*>(layer.attn_q_norm.data),
            scratch.attention.q.data(), batch_size, config.num_attention_heads,
            config.head_dim, 1e-6F, arena.stream);
      }
      if (!layer.attn_k_norm.empty()) {
        LaunchBatchedPerHeadRMSNorm(
            scratch.attention.k.data(),
            static_cast<const float*>(layer.attn_k_norm.data),
            scratch.attention.k.data(), batch_size, config.num_key_value_heads,
            config.head_dim, 1e-6F, arena.stream);
      }

      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena.GetMaxContext() * config.head_dim;
      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        float* query = scratch.attention.q.data() + (row * attention_size);
        float* key = scratch.attention.k.data() + (row * kv_size);
        float* value = scratch.attention.v.data() + (row * kv_size);
        float* gate = scratch.ssm.gate.data() + (row * attention_size);
        float* context = scratch.ssm.out.data() + (row * attention_size);
        const std::uint32_t position = items[row].position;

        LaunchRoPE(query, key, config.num_attention_heads,
                   config.num_key_value_heads, config.head_dim,
                   config.rotary_dim, position, config.rope_theta,
                   arena.stream);
        const bool use_split_k = detail::IsSplitKDecodeAttentionSupported(
            static_cast<std::size_t>(position) + 1, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim);
        LaunchAttention(
            query, key, value, gate, state_arena.d_kv_cache,
            OffsetIfPresent(state_arena.d_kv_cache, total_k),
            state_arena.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                total_k),
            context, attention_layer, position, state_arena.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena.stream,
            use_split_k ? arena.d_split_k_attention : nullptr);
      }
      LaunchProjection(layer.attn_output, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       attention_size, arena.stream);
    } else {
      LaunchProjection(layer.attn_qkv, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, ssm_qkv_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.attn_gate, scratch.decode.normed.data(),
                       scratch.ssm.gate.data(), batch_size, ssm_inner_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.ssm_alpha, scratch.decode.normed.data(),
                       scratch.ssm.alpha.data(), batch_size, time_step_rank,
                       hidden_size, arena.stream);
      LaunchProjection(layer.ssm_beta, scratch.decode.normed.data(),
                       scratch.ssm.beta.data(), batch_size, time_step_rank,
                       hidden_size, arena.stream);

      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        LaunchSSMConvRecurrence(
            scratch.ssm.qkv.data() + (row * ssm_qkv_size),
            static_cast<const float*>(layer.ssm_conv1d.data),
            state_arena.d_ssm_conv_state,
            scratch.ssm.conv_out.data() + (row * ssm_qkv_size),
            state_arena.d_ssm_deltanet_state,
            scratch.ssm.alpha.data() + (row * time_step_rank),
            scratch.ssm.beta.data() + (row * time_step_rank),
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data),
            scratch.ssm.gate.data() + (row * ssm_inner_size),
            scratch.ssm.out.data() + (row * ssm_inner_size), layer_index,
            ssm_qkv_size, config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena.stream,
            state_arena.GetSsmReplayCapture(),
            state_arena.GetRecurrentStateStorage());
      }

      ssm_residual_folded = route_plan.fuse_ssm_epilogue &&
                            SupportsDenseFusedProjection(layer.ssm_out.type);
      if (ssm_residual_folded) {
        for (std::size_t row = 0; row < batch_size; ++row) {
          float* hidden = scratch.decode.hidden.data() + (row * hidden_size);
          LaunchGEMVResidual(layer.ssm_out.data, layer.ssm_out.type,
                             scratch.ssm.out.data() + (row * ssm_inner_size),
                             hidden, hidden, hidden_size, ssm_inner_size,
                             arena.stream);
        }
      } else {
        LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                         scratch.attention.output.data(), batch_size,
                         hidden_size, ssm_inner_size, arena.stream);
      }
    }

    if (ssm_residual_folded) {
      if (!fuse_ffn_norm_swiglu) {
        LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                             static_cast<const float*>(layer.ffn_norm.data),
                             scratch.decode.normed.data(), nullptr, batch_size,
                             hidden_size, 1e-6F, arena.stream);
      }
    } else if (route_plan.fuse_residual_rmsnorm) {
      LaunchBatchedFusedResidualAddRMSNorm(
          scratch.decode.hidden.data(), scratch.attention.output.data(),
          scratch.decode.hidden.data(),
          static_cast<const float*>(layer.ffn_norm.data),
          scratch.decode.normed.data(), nullptr, batch_size, hidden_size, 1e-6F,
          arena.stream);
    } else {
      LaunchBatchedResidualAdd(
          scratch.decode.hidden.data(), scratch.attention.output.data(),
          scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);
      if (!fuse_ffn_norm_swiglu) {
        LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                             static_cast<const float*>(layer.ffn_norm.data),
                             scratch.decode.normed.data(), nullptr, batch_size,
                             hidden_size, 1e-6F, arena.stream);
      }
    }

    if (fuse_ffn_norm_swiglu) {
      for (std::size_t row = 0; row < batch_size; ++row) {
        LaunchFusedRMSNormSwiGLUGEMV(
            scratch.decode.hidden.data() + (row * hidden_size),
            static_cast<const float*>(layer.ffn_norm.data), 1e-6F,
            layer.ffn_gate.data, layer.ffn_up.data,
            scratch.ffn.activation.data() + (row * intermediate_size),
            intermediate_size, hidden_size, arena.stream);
      }
    } else {
      LaunchProjection(layer.ffn_gate, scratch.decode.normed.data(),
                       scratch.ffn.gate.data(), batch_size, intermediate_size,
                       hidden_size, arena.stream);
      LaunchProjection(layer.ffn_up, scratch.decode.normed.data(),
                       scratch.ffn.up.data(), batch_size, intermediate_size,
                       hidden_size, arena.stream);
      LaunchBatchedSwiGLUActivation(
          scratch.ffn.gate.data(), scratch.ffn.up.data(),
          scratch.ffn.activation.data(), nullptr,
          batch_size * intermediate_size, arena.stream);
    }
    LaunchProjection(layer.ffn_down, scratch.ffn.activation.data(),
                     scratch.ffn.out.data(), batch_size, hidden_size,
                     intermediate_size, arena.stream);
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.ffn.out.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);

    for (std::size_t row = 0; row < batch_size; ++row) {
      auto& state_arena = items[row].executor->arena_;
      if (const auto tap = state_arena.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        HIP_CHECK(hipMemcpyAsync(
            state_arena.d_target_layer_features + (*tap * hidden_size),
            scratch.decode.hidden.data() + (row * hidden_size),
            hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
            arena.stream));
      }
    }
  }

  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena.stream);
  coordinator->EnsureVerificationLogits(batch_size);
  LaunchProjection(weights.output, scratch.decode.normed.data(),
                   coordinator->d_verification_logits_, batch_size, vocab_size,
                   hidden_size, arena.stream);
  LaunchBatchedGPUArgmax(coordinator->d_verification_logits_,
                         scratch.decode.prompt_tokens.data(), batch_size,
                         vocab_size, arena.stream);

  std::array<std::uint32_t, kMaxDecodeBatch> host_frontiers{};
  HIP_CHECK(hipMemcpyAsync(
      host_frontiers.data(), scratch.decode.prompt_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyDeviceToHost, arena.stream));
  for (std::size_t row = 0; row < batch_size; ++row) {
    auto& state_arena = items[row].executor->arena_;
    if (&state_arena != &arena || row != 0) {
      HIP_CHECK(hipMemcpyAsync(
          state_arena.d_hidden,
          scratch.decode.hidden.data() + (row * hidden_size),
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
    }
    HIP_CHECK(hipMemcpyAsync(
        state_arena.d_logits,
        coordinator->d_verification_logits_ + (row * vocab_size),
        vocab_size * sizeof(float), hipMemcpyDeviceToDevice, arena.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena.stream));

  return {host_frontiers.begin(),
          host_frontiers.begin() + static_cast<std::ptrdiff_t>(batch_size)};
}

std::vector<tokenization::TokenId>
QwenGpuExecutor::ForwardDecodeEquivalentVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  const std::size_t batch_size = candidate_tokens.size();
  if (batch_size == 0) {
    h_verification_hidden_.clear();
    h_verification_logits_.clear();
    last_verification_rows_ = 0;
    return {};
  }
  if (batch_size > kMaxDecodeBatch) {
    throw std::invalid_argument(
        "Qwen exact verification supports at most eight tokens");
  }
  if (static_cast<std::size_t>(start_pos) + batch_size >
      arena_.GetMaxContext()) {
    throw std::length_error(
        "Qwen verification chunk exceeds the GPU context length");
  }

  replaying_ssm_state_ = false;
  last_verification_rows_ = batch_size;
  auto& stages = VerifyTimings();
  if (VerifyTimingEnabled()) {
    ++stages.chunks;
  }
  h_verification_hidden_.clear();
  h_verification_logits_.clear();

  const auto& config = weights_.config;
  const auto scratch = arena_.GetScratchView(batch_size);
  const std::size_t hidden_size = config.hidden_size;
  const std::size_t intermediate_size = config.intermediate_size;
  const std::size_t vocab_size = config.vocab_size;
  const std::size_t attention_size = config.AttentionSize();
  const std::size_t kv_size =
      static_cast<std::size_t>(config.num_key_value_heads) * config.head_dim;
  const std::size_t q_projection_size = 2 * attention_size;
  const std::size_t ssm_qkv_size = config.SsmQkvSize();
  const std::size_t ssm_inner_size = config.ssm_inner_size;
  const std::size_t time_step_rank = config.ssm_time_step_rank;
  const std::size_t captured_layer_count =
      arena_.GetTargetLayerCapture().size();

  std::array<std::uint32_t, kMaxDecodeBatch> host_tokens{};
  std::array<std::uint32_t, kMaxDecodeBatch> host_positions{};
  for (std::size_t row = 0; row < batch_size; ++row) {
    host_tokens[row] = candidate_tokens[row];
    host_positions[row] = start_pos + static_cast<std::uint32_t>(row);
    arena_.MarkSsmReplayPosition(host_positions[row]);
  }
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_tokens.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));
  LaunchBatchedEmbeddingLookup(
      weights_.token_embd.data, weights_.token_embd.type,
      scratch.decode.prompt_tokens.data(), scratch.decode.hidden.data(),
      batch_size, hidden_size, arena_.stream);
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(),
                           host_positions.data(),
                           batch_size * sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, arena_.stream));

  if (capture_prompt_hidden_ && captured_layer_count > 0) {
    h_verification_hidden_.resize(batch_size * captured_layer_count *
                                  hidden_size);
  }

  EmitDecodeRouteTelemetry(weights_, policy_);
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights_.layers[layer_index];
    const auto route_plan = ResolveQwenLayerRoute(
        policy_, QwenExecutionMode::kDecode, layer.is_full_attention);
    const bool fuse_ffn_norm_swiglu =
        route_plan.fuse_ffn_swiglu &&
        layer.ffn_gate.type == core::GgmlType::kBF16 &&
        layer.ffn_up.type == core::GgmlType::kBF16;

    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);

    bool ssm_residual_folded = false;
    if (layer.is_full_attention) {
      {
        VerifyStageTimer timer(&stages.projections_ms, arena_.stream);
        LaunchProjection(layer.attn_q, scratch.decode.normed.data(),
                         scratch.ssm.qkv.data(), batch_size, q_projection_size,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.attn_k, scratch.decode.normed.data(),
                         scratch.attention.k.data(), batch_size, kv_size,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.attn_v, scratch.decode.normed.data(),
                         scratch.attention.v.data(), batch_size, kv_size,
                         hidden_size, arena_.stream);
      }
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena_.stream);

      const std::size_t total_k = config.FullAttentionLayerCount() *
                                  config.num_key_value_heads *
                                  arena_.GetMaxContext() * config.head_dim;
      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      const bool fused_qknorm_rope_kv =
          route_plan.fuse_qk_norm_rope_kv &&
          detail::IsFusedQkNormSupported(config.head_dim);
      if (!fused_qknorm_rope_kv) {
        if (!layer.attn_q_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.q.data(),
              static_cast<const float*>(layer.attn_q_norm.data),
              scratch.attention.q.data(), batch_size,
              config.num_attention_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
        if (!layer.attn_k_norm.empty()) {
          LaunchBatchedPerHeadRMSNorm(
              scratch.attention.k.data(),
              static_cast<const float*>(layer.attn_k_norm.data),
              scratch.attention.k.data(), batch_size,
              config.num_key_value_heads, config.head_dim, 1e-6F,
              arena_.stream);
        }
      }

      VerifyStageTimer attention_timer(&stages.attention_ms, arena_.stream);
      for (std::size_t row = 0; row < batch_size; ++row) {
        float* const query =
            scratch.attention.q.data() + (row * attention_size);
        float* const key = scratch.attention.k.data() + (row * kv_size);
        float* const value = scratch.attention.v.data() + (row * kv_size);
        float* const gate = scratch.ssm.gate.data() + (row * attention_size);
        float* const context = scratch.ssm.out.data() + (row * attention_size);
        const std::uint32_t position = host_positions[row];

        if (fused_qknorm_rope_kv) {
          LaunchFusedQKNormRoPEKvWrite(
              query, key, value,
              static_cast<const float*>(layer.attn_q_norm.data),
              static_cast<const float*>(layer.attn_k_norm.data), query, key,
              arena_.d_kv_cache, OffsetIfPresent(arena_.d_kv_cache, total_k),
              arena_.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                  total_k),
              attention_layer, scratch.decode.prompt_tokens.data() + row,
              arena_.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, config.rotary_dim,
              config.rope_theta, 1e-6F, arena_.stream);
        } else {
          LaunchRoPE(query, key, config.num_attention_heads,
                     config.num_key_value_heads, config.head_dim,
                     config.rotary_dim, position, config.rope_theta,
                     arena_.stream);
        }

        const bool use_split_k = detail::IsSplitKDecodeAttentionSupported(
            static_cast<std::size_t>(position) + 1, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim);
        LaunchAttention(
            query, key, value, gate, arena_.d_kv_cache,
            OffsetIfPresent(arena_.d_kv_cache, total_k),
            arena_.d_attention_kv_f16,
            OffsetIfPresent(
                static_cast<std::uint16_t*>(arena_.d_attention_kv_f16),
                total_k),
            context, attention_layer, position, arena_.GetMaxContext(),
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, arena_.stream,
            use_split_k ? arena_.d_split_k_attention : nullptr,
            fused_qknorm_rope_kv);
      }
      LaunchProjection(layer.attn_output, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       attention_size, arena_.stream);
    } else {
      {
        VerifyStageTimer timer(&stages.projections_ms, arena_.stream);
        LaunchProjection(layer.attn_qkv, scratch.decode.normed.data(),
                         scratch.ssm.qkv.data(), batch_size, ssm_qkv_size,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.attn_gate, scratch.decode.normed.data(),
                         scratch.ssm.gate.data(), batch_size, ssm_inner_size,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.ssm_alpha, scratch.decode.normed.data(),
                         scratch.ssm.alpha.data(), batch_size, time_step_rank,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.ssm_beta, scratch.decode.normed.data(),
                         scratch.ssm.beta.data(), batch_size, time_step_rank,
                         hidden_size, arena_.stream);
      }

      {
        // One pair of launches walks the whole verification batch. The rows are
        // still applied in order with identical arithmetic, so this is
        // bit-exact; it removes 2 x batch_size dispatches per SSM layer, which
        // dominated the stage at width 8.
        VerifyStageTimer ssm_timer(&stages.ssm_recurrence_ms, arena_.stream);
        auto replay_capture = arena_.GetSsmReplayCapture();
        replay_capture.position = scratch.decode.prompt_tokens.data();
        LaunchSSMConvRecurrenceRows(
            scratch.ssm.qkv.data(),
            static_cast<const float*>(layer.ssm_conv1d.data),
            arena_.d_ssm_conv_state, scratch.ssm.conv_out.data(),
            arena_.d_ssm_deltanet_state, scratch.ssm.alpha.data(),
            scratch.ssm.beta.data(),
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data),
            scratch.ssm.gate.data(), scratch.ssm.out.data(), layer_index,
            ssm_qkv_size, config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(),
            static_cast<std::uint32_t>(batch_size), time_step_rank,
            ssm_inner_size, arena_.stream, replay_capture,
            arena_.GetRecurrentStateStorage());
      }

      ssm_residual_folded = route_plan.fuse_ssm_epilogue &&
                            SupportsDenseFusedProjection(layer.ssm_out.type);
      if (ssm_residual_folded) {
        for (std::size_t row = 0; row < batch_size; ++row) {
          float* const hidden =
              scratch.decode.hidden.data() + (row * hidden_size);
          LaunchGEMVResidual(layer.ssm_out.data, layer.ssm_out.type,
                             scratch.ssm.out.data() + (row * ssm_inner_size),
                             hidden, hidden, hidden_size, ssm_inner_size,
                             arena_.stream);
        }
      } else {
        LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                         scratch.attention.output.data(), batch_size,
                         hidden_size, ssm_inner_size, arena_.stream);
      }
    }

    if (ssm_residual_folded) {
      if (!fuse_ffn_norm_swiglu) {
        LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                             static_cast<const float*>(layer.ffn_norm.data),
                             scratch.decode.normed.data(), nullptr, batch_size,
                             hidden_size, 1e-6F, arena_.stream);
      }
    } else if (route_plan.fuse_residual_rmsnorm) {
      LaunchBatchedFusedResidualAddRMSNorm(
          scratch.decode.hidden.data(), scratch.attention.output.data(),
          scratch.decode.hidden.data(),
          static_cast<const float*>(layer.ffn_norm.data),
          scratch.decode.normed.data(), nullptr, batch_size, hidden_size, 1e-6F,
          arena_.stream);
    } else {
      LaunchBatchedResidualAdd(
          scratch.decode.hidden.data(), scratch.attention.output.data(),
          scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
      if (!fuse_ffn_norm_swiglu) {
        LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                             static_cast<const float*>(layer.ffn_norm.data),
                             scratch.decode.normed.data(), nullptr, batch_size,
                             hidden_size, 1e-6F, arena_.stream);
      }
    }

    {
      VerifyStageTimer ffn_timer(&stages.ffn_ms, arena_.stream);
      if (fuse_ffn_norm_swiglu) {
        for (std::size_t row = 0; row < batch_size; ++row) {
          LaunchFusedRMSNormSwiGLUGEMV(
              scratch.decode.hidden.data() + (row * hidden_size),
              static_cast<const float*>(layer.ffn_norm.data), 1e-6F,
              layer.ffn_gate.data, layer.ffn_up.data,
              scratch.ffn.activation.data() + (row * intermediate_size),
              intermediate_size, hidden_size, arena_.stream);
        }
      } else {
        LaunchProjection(layer.ffn_gate, scratch.decode.normed.data(),
                         scratch.ffn.gate.data(), batch_size, intermediate_size,
                         hidden_size, arena_.stream);
        LaunchProjection(layer.ffn_up, scratch.decode.normed.data(),
                         scratch.ffn.up.data(), batch_size, intermediate_size,
                         hidden_size, arena_.stream);
        LaunchBatchedSwiGLUActivation(
            scratch.ffn.gate.data(), scratch.ffn.up.data(),
            scratch.ffn.activation.data(), nullptr,
            batch_size * intermediate_size, arena_.stream);
      }
      LaunchProjection(layer.ffn_down, scratch.ffn.activation.data(),
                       scratch.ffn.out.data(), batch_size, hidden_size,
                       intermediate_size, arena_.stream);
      LaunchBatchedResidualAdd(
          scratch.decode.hidden.data(), scratch.ffn.out.data(),
          scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    }

    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        float* const destination =
            h_verification_hidden_.data() + (*tap * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(
            destination, captured_layer_count * hidden_size * sizeof(float),
            scratch.decode.hidden.data(), hidden_size * sizeof(float),
            hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToHost,
            arena_.stream));
        HIP_CHECK(hipMemcpyAsync(
            arena_.d_target_layer_features + (*tap * hidden_size),
            scratch.decode.hidden.data() + ((batch_size - 1) * hidden_size),
            hidden_size * sizeof(float), hipMemcpyDeviceToDevice,
            arena_.stream));
      }
    }
  }

  if (capture_prompt_hidden_ && captured_layer_count == 0) {
    h_verification_hidden_.resize(batch_size * hidden_size);
    HIP_CHECK(hipMemcpyAsync(h_verification_hidden_.data(),
                             scratch.decode.hidden.data(),
                             h_verification_hidden_.size() * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  }
  last_hidden_offset_ = (batch_size - 1) * hidden_size;

  VerifyStageTimer head_timer(&stages.head_ms, arena_.stream);
  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights_.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena_.stream);
  EnsureVerificationLogits(batch_size);
  LaunchProjection(weights_.output, scratch.decode.normed.data(),
                   d_verification_logits_, batch_size, vocab_size, hidden_size,
                   arena_.stream);
  auto* const d_out_tokens = scratch.decode.sampled_token.data();
  LaunchBatchedGPUArgmax(d_verification_logits_, d_out_tokens, batch_size,
                         vocab_size, arena_.stream);

  std::vector<tokenization::TokenId> predictions(batch_size, 0);
  HIP_CHECK(hipMemcpyAsync(predictions.data(), d_out_tokens,
                           batch_size * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  if (capture_logits) {
    h_verification_logits_.resize(batch_size * vocab_size);
    HIP_CHECK(hipMemcpyAsync(h_verification_logits_.data(),
                             d_verification_logits_,
                             h_verification_logits_.size() * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  arena_.DisableSsmReplayCapture();
  return predictions;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
