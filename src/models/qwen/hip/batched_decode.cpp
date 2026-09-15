#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

[[nodiscard]] constexpr bool SupportsExactSharedProjection(
    core::GgmlType type) noexcept {
  // opt-q4kxl: the K-quant/IQ formats have an exact shared-weight small-batch
  // kernel too (SmallBatchKQuantExactFp32GEMMKernel). Without them here, every
  // projection in the UD-Q4_K_XL shard fell to the per-row loop below, which
  // re-reads the weight matrix once per token in the draft block.
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ8_K ||
         detail::IsNativeWmmaQuant(type);
}

void LaunchProjection(const models::QwenTensorRef& weight, const float* input,
                      float* output, std::size_t batch_size,
                      std::size_t output_size, std::size_t input_size,
                      hipStream_t stream) {
  if (batch_size > kMaxDecodeBatch) {
    for (std::size_t row = 0; row < batch_size; row += kMaxDecodeBatch) {
      const std::size_t width = std::min(kMaxDecodeBatch, batch_size - row);
      LaunchProjection(weight, input + row * input_size,
                       output + row * output_size, width, output_size,
                       input_size, stream);
    }
    return;
  }
  if (SupportsExactSharedProjection(weight.type)) {
    LaunchBatchedQuantGEMMFp32(weight.type, weight.data, input, output,
                               batch_size, output_size, input_size, stream);
    return;
  }
  if (weight.type == core::GgmlType::kBF16 && batch_size <= kMaxDecodeBatch) {
    LaunchExactBf16GEMMFp32SmallBatch(weight.data, input, output, batch_size,
                                      output_size, input_size, stream);
    return;
  }

  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchGEMV(weight.data, weight.type, input + (row * input_size),
               output + (row * output_size), output_size, input_size, stream);
  }
}

void LaunchFfnActivation(const models::QwenLayerWeights& layer,
                         const QwenGpuScratchView& scratch,
                         std::size_t batch_size, std::size_t intermediate_size,
                         std::size_t hidden_size, hipStream_t stream) {
  const auto& gate = layer.ffn_gate;
  const auto& up = layer.ffn_up;
  const bool packed_format = gate.type == core::GgmlType::kQ4_K ||
                             gate.type == core::GgmlType::kQ5_K ||
                             gate.type == core::GgmlType::kIQ4_XS ||
                             gate.type == core::GgmlType::kQ8_0;
  if (batch_size >= 2 && intermediate_size == 17408 && hidden_size == 5120 &&
      packed_format && gate.type == up.type &&
      gate.num_elements == intermediate_size * hidden_size &&
      up.num_elements == gate.num_elements) {
    const std::size_t matrix_bytes = gate.EncodedSizeBytes();
    const std::size_t packed_elements = 2 * batch_size * intermediate_size;
    if (gate.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(gate.data) + matrix_bytes == up.data &&
        scratch.decode.weight_bf16.size_bytes() >=
            packed_elements * sizeof(float)) {
      // Adjacent GGUF tensors form one matrix with twice as many output rows.
      // Exact projections do not need dequantization scratch; reuse it until
      // SwiGLU has consumed the packed gate/up rows on this stream.
      auto* const packed =
          reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
      LaunchProjection(gate, scratch.decode.normed.data(), packed, batch_size,
                       2 * intermediate_size, hidden_size, stream);
      LaunchPackedSwiGLUActivation(packed, scratch.ffn.activation.data(),
                                   batch_size, intermediate_size, stream);
      return;
    }
  }
  LaunchProjection(gate, scratch.decode.normed.data(), scratch.ffn.gate.data(),
                   batch_size, intermediate_size, hidden_size, stream);
  LaunchProjection(up, scratch.decode.normed.data(), scratch.ffn.up.data(),
                   batch_size, intermediate_size, hidden_size, stream);
  LaunchBatchedSwiGLUActivation(
      scratch.ffn.gate.data(), scratch.ffn.up.data(),
      scratch.ffn.activation.data(), nullptr, batch_size * intermediate_size,
      stream);
}

struct SsmControls {
  const float* alpha;
  const float* beta;
  std::size_t row_stride;
};

SsmControls LaunchSsmControls(const models::QwenLayerWeights& layer,
                             const QwenGpuScratchView& scratch,
                             std::size_t batch_size, std::size_t time_step_rank,
                             std::size_t hidden_size, hipStream_t stream) {
  const auto& alpha = layer.ssm_alpha;
  const auto& beta = layer.ssm_beta;
  if (time_step_rank == 48 && hidden_size == 5120 &&
      alpha.type == core::GgmlType::kQ8_0 && beta.type == alpha.type &&
      alpha.num_elements == time_step_rank * hidden_size &&
      beta.num_elements == alpha.num_elements) {
    const std::size_t matrix_bytes = alpha.EncodedSizeBytes();
    const std::size_t row_stride = 2 * time_step_rank;
    if (alpha.available_bytes >= 2 * matrix_bytes &&
        static_cast<const std::byte*>(alpha.data) + matrix_bytes == beta.data &&
        scratch.decode.weight_bf16.size_bytes() >=
            batch_size * row_stride * sizeof(float)) {
      // Adjacent control matrices share one launch. Recurrence already accepts
      // a row stride, so it can consume the interleaved output directly.
      auto* const packed =
          reinterpret_cast<float*>(scratch.decode.weight_bf16.data());
      LaunchProjection(alpha, scratch.decode.normed.data(), packed, batch_size,
                       row_stride, hidden_size, stream);
      return {packed, packed + time_step_rank, row_stride};
    }
  }
  LaunchProjection(alpha, scratch.decode.normed.data(), scratch.ssm.alpha.data(),
                   batch_size, time_step_rank, hidden_size, stream);
  LaunchProjection(beta, scratch.decode.normed.data(), scratch.ssm.beta.data(),
                   batch_size, time_step_rank, hidden_size, stream);
  return {scratch.ssm.alpha.data(), scratch.ssm.beta.data(), time_step_rank};
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

  bool capture_replay = false;
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (executor.replaying_ssm_state_) {
      executor.replaying_ssm_state_ = false;
      executor.arena_.DisableSsmReplayCapture();
    }
    // Replay and capture-flag updates may still be queued on a session's
    // stream. The coordinator takes ownership only after they finish.
    if (&executor != coordinator) {
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    }
    capture_replay |= executor.arena_.IsSsmReplayCaptureActive();
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
  std::array<std::uint32_t, kMaxDecodeBatch> host_positions{};
  for (std::size_t row = 0; row < batch_size; ++row) {
    host_tokens[row] = items[row].token_id;
    host_positions[row] = items[row].position;
  }
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.prompt_tokens.data(), host_tokens.data(),
      batch_size * sizeof(std::uint32_t), hipMemcpyHostToDevice, arena.stream));
  LaunchBatchedEmbeddingLookup(weights.token_embd.data, weights.token_embd.type,
                               scratch.decode.prompt_tokens.data(),
                               scratch.decode.hidden.data(), batch_size,
                               hidden_size, arena.stream);
  if (capture_replay) {
    // Embedding has consumed the token IDs. Reuse the buffer for the actual
    // per-session positions until every recurrent layer has captured its row.
    HIP_CHECK(hipMemcpyAsync(
        scratch.decode.prompt_tokens.data(), host_positions.data(),
        batch_size * sizeof(std::uint32_t), hipMemcpyHostToDevice, arena.stream));
  }

  EmitDecodeRouteTelemetry(weights, coordinator->policy_);
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

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

      const std::uint32_t attention_layer =
          layer_index / config.full_attention_interval;
      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
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
      const auto controls = LaunchSsmControls(
          layer, scratch, batch_size, time_step_rank, hidden_size, arena.stream);

      for (std::size_t row = 0; row < batch_size; ++row) {
        auto& state_arena = items[row].executor->arena_;
        auto replay_capture = state_arena.GetSsmReplayCapture();
        replay_capture.position = scratch.decode.prompt_tokens.data() + row;
        LaunchSSMConvRecurrence(
            scratch.ssm.qkv.data() + (row * ssm_qkv_size),
            static_cast<const float*>(layer.ssm_conv1d.data),
            state_arena.d_ssm_conv_state,
            scratch.ssm.conv_out.data() + (row * ssm_qkv_size),
            state_arena.d_ssm_deltanet_state,
            controls.alpha + (row * controls.row_stride),
            controls.beta + (row * controls.row_stride),
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data),
            scratch.ssm.gate.data() + (row * ssm_inner_size),
            scratch.ssm.out.data() + (row * ssm_inner_size), layer_index,
            ssm_qkv_size, config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(), arena.stream,
            replay_capture,
            state_arena.GetRecurrentStateStorage());
      }

      LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       ssm_inner_size, arena.stream);
    }

    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena.stream);
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.ffn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena.stream);

    LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                         hidden_size, arena.stream);
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
  if (candidate_tokens.empty()) {
    h_verification_hidden_.clear();
    h_verification_logits_.clear();
    last_verification_rows_ = 0;
    return {};
  }
  const QwenGpuVerificationItem item{this, candidate_tokens, start_pos,
                                     capture_logits};
  auto predictions = ForwardVerificationBatch(std::span(&item, 1));
  return std::move(predictions.front());
}

std::vector<std::vector<tokenization::TokenId>>
QwenGpuExecutor::ForwardVerificationBatch(
    std::span<const QwenGpuVerificationItem> items) {
  if (items.empty() || items.size() > kMaxDecodeBatch ||
      items.front().executor == nullptr) {
    throw std::invalid_argument(
        "Qwen verification requires one to eight sessions");
  }
  auto& coordinator = *items.front().executor;
  auto& arena_ = coordinator.arena_;
  const auto& weights_ = coordinator.weights_;
  const auto& config = weights_.config;
  const bool capture_prompt_hidden_ = coordinator.capture_prompt_hidden_;
  const auto policy = coordinator.policy_.Fingerprint();
  constexpr std::size_t kMaxRows = kMaxDecodeBatch * kMaxDecodeBatch;
  std::array<std::size_t, kMaxDecodeBatch> offsets{};
  std::size_t batch_size = 0;
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    const auto* executor = item.executor;
    if (executor == nullptr || executor->model_ != coordinator.model_ ||
        executor->policy_.Fingerprint() != policy ||
        executor->capture_prompt_hidden_ != capture_prompt_hidden_ ||
        !std::ranges::equal(executor->arena_.GetTargetLayerCapture(),
                            arena_.GetTargetLayerCapture()) ||
        item.tokens.empty() || item.tokens.size() > kMaxDecodeBatch) {
      throw std::invalid_argument(
          "Qwen verification has incompatible sessions or token counts");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (items[previous].executor == executor)
        throw std::invalid_argument("Qwen verification repeats a session");
    }
    if (item.position > executor->arena_.GetMaxContext() ||
        item.tokens.size() > executor->arena_.GetMaxContext() - item.position) {
      throw std::length_error("Qwen verification exceeds a session context");
    }
    offsets[index] = batch_size;
    batch_size += item.tokens.size();
  }
  if (batch_size > arena_.GetMaxBatch()) {
    throw std::length_error(
        "Qwen verification exceeds coordinator scratch capacity");
  }
  for (const auto& item : items) {
    auto& executor = *item.executor;
    if (&executor != &coordinator)
      HIP_CHECK(hipStreamSynchronize(executor.arena_.stream));
    executor.replaying_ssm_state_ = false;
    executor.last_verification_rows_ = item.tokens.size();
    executor.h_verification_hidden_.clear();
    executor.h_verification_logits_.clear();
  }
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
  const std::size_t feature_width = captured_layer_count * hidden_size;
  const std::size_t feature_elements =
      capture_prompt_hidden_ ? batch_size * feature_width : 0;
  const std::size_t feature_rows =
      feature_elements / vocab_size + (feature_elements % vocab_size != 0);
  coordinator.EnsureVerificationLogits(std::max(batch_size, feature_rows));
  auto* const d_verification_logits_ = coordinator.d_verification_logits_;
  for (const auto& item : items) {
    if (item.executor != &coordinator)
      item.executor->EnsureVerificationLogits(item.tokens.size());
    if (capture_prompt_hidden_)
      item.executor->h_verification_hidden_.resize(
          item.tokens.size() * (feature_width ? feature_width : hidden_size));
  }
  std::array<std::uint32_t, kMaxRows> host_tokens{};
  std::array<std::uint32_t, kMaxRows> host_positions{};
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    for (std::size_t row = 0; row < item.tokens.size(); ++row) {
      const auto position = item.position + static_cast<std::uint32_t>(row);
      host_tokens[offsets[index] + row] = item.tokens[row];
      host_positions[offsets[index] + row] = position;
      item.executor->arena_.MarkSsmReplayPosition(position);
    }
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
  EmitDecodeRouteTelemetry(weights_, coordinator.policy_);
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights_.layers[layer_index];
    const auto route_plan =
        ResolveQwenLayerRoute(coordinator.policy_, QwenExecutionMode::kDecode,
                              layer.is_full_attention);

    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.attn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);

    if (layer.is_full_attention) {
      LaunchProjection(layer.attn_q, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, q_projection_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_k, scratch.decode.normed.data(),
                       scratch.attention.k.data(), batch_size, kv_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_v, scratch.decode.normed.data(),
                       scratch.attention.v.data(), batch_size, kv_size,
                       hidden_size, arena_.stream);
      LaunchBatchedUnpackQG(scratch.ssm.qkv.data(), scratch.attention.q.data(),
                            scratch.ssm.gate.data(), batch_size,
                            config.num_attention_heads, config.head_dim,
                            arena_.stream);

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

      for (std::size_t index = 0; index < items.size(); ++index) {
        const auto& item = items[index];
        auto& state_arena = item.executor->arena_;
        const std::size_t offset = offsets[index];
        const std::size_t sequence_size = item.tokens.size();
        const std::uint32_t start_pos = item.position;
        const std::size_t total_k = state_arena.GetAttentionKvPlaneElements();
        if (fused_qknorm_rope_kv) {
          LaunchBatchedFusedQKNormRoPEKvWrite(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              (scratch.attention.v.data() + offset * kv_size),
              static_cast<const float*>(layer.attn_q_norm.data),
              static_cast<const float*>(layer.attn_k_norm.data),
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.attention.k.data() + offset * kv_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, config.rotary_dim,
              config.rope_theta, 1e-6F, arena_.stream);
          // Exact verification projections do not use weight dequantization
          // scratch. Reuse it for independent split-K rows without allocating
          // another buffer or changing the scalar decode workspace.
          auto attention_scratch = scratch.attention.split_k;
          if (scratch.decode.weight_bf16.size_bytes() >=
              sequence_size * attention_scratch.size_bytes()) {
            attention_scratch = {
                reinterpret_cast<float*>(scratch.decode.weight_bf16.data()),
                sequence_size * attention_scratch.size()};
          }
          LaunchCausalDecodeAttention(
              (scratch.attention.q.data() + offset * attention_size),
              (scratch.ssm.gate.data() + offset * attention_size),
              state_arena.d_kv_cache,
              OffsetIfPresent(state_arena.d_kv_cache, total_k),
              state_arena.d_attention_kv_f16,
              OffsetIfPresent(
                  static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                  total_k),
              (scratch.ssm.out.data() + offset * attention_size),
              attention_layer, start_pos, sequence_size,
              state_arena.GetMaxContext(), config.num_attention_heads,
              config.num_key_value_heads, config.head_dim, arena_.stream,
              attention_scratch);
        } else {
          for (std::size_t row = 0; row < sequence_size; ++row) {
            float* const query =
                (scratch.attention.q.data() + offset * attention_size) +
                (row * attention_size);
            float* const key = (scratch.attention.k.data() + offset * kv_size) +
                               (row * kv_size);
            const std::uint32_t position = host_positions[offset + row];
            LaunchRoPE(query, key, config.num_attention_heads,
                       config.num_key_value_heads, config.head_dim,
                       config.rotary_dim, position, config.rope_theta,
                       arena_.stream);
            LaunchAttention(
                query, key,
                (scratch.attention.v.data() + offset * kv_size) + row * kv_size,
                (scratch.ssm.gate.data() + offset * attention_size) +
                    row * attention_size,
                state_arena.d_kv_cache,
                OffsetIfPresent(state_arena.d_kv_cache, total_k),
                state_arena.d_attention_kv_f16,
                OffsetIfPresent(
                    static_cast<std::uint16_t*>(state_arena.d_attention_kv_f16),
                    total_k),
                (scratch.ssm.out.data() + offset * attention_size) +
                    row * attention_size,
                attention_layer, position, state_arena.GetMaxContext(),
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, arena_.stream,
                state_arena.d_split_k_attention);
          }
        }
      }
      LaunchProjection(layer.attn_output, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       attention_size, arena_.stream);
    } else {
      LaunchProjection(layer.attn_qkv, scratch.decode.normed.data(),
                       scratch.ssm.qkv.data(), batch_size, ssm_qkv_size,
                       hidden_size, arena_.stream);
      LaunchProjection(layer.attn_gate, scratch.decode.normed.data(),
                       scratch.ssm.gate.data(), batch_size, ssm_inner_size,
                       hidden_size, arena_.stream);
      const auto controls = LaunchSsmControls(
          layer, scratch, batch_size, time_step_rank, hidden_size, arena_.stream);
      for (std::size_t index = 0; index < items.size(); ++index) {
        const auto& item = items[index];
        auto& state_arena = item.executor->arena_;
        const std::size_t offset = offsets[index];
        auto replay_capture = state_arena.GetSsmReplayCapture();
        replay_capture.position = scratch.decode.prompt_tokens.data() + offset;
        LaunchSSMConvRecurrenceRows(
            scratch.ssm.qkv.data() + offset * ssm_qkv_size,
            static_cast<const float*>(layer.ssm_conv1d.data),
            state_arena.d_ssm_conv_state,
            scratch.ssm.conv_out.data() + offset * ssm_qkv_size,
            state_arena.d_ssm_deltanet_state,
            controls.alpha + offset * controls.row_stride,
            controls.beta + offset * controls.row_stride,
            static_cast<const float*>(layer.ssm_a.data),
            static_cast<const float*>(layer.ssm_dt.data),
            static_cast<const float*>(layer.ssm_norm.data),
            scratch.ssm.gate.data() + offset * ssm_inner_size,
            scratch.ssm.out.data() + offset * ssm_inner_size, layer_index,
            ssm_qkv_size, config.ssm_group_count, config.ssm_time_step_rank,
            config.ssm_state_size, config.SsmValueSize(),
            static_cast<std::uint32_t>(item.tokens.size()), controls.row_stride,
            ssm_inner_size, arena_.stream, replay_capture,
            state_arena.GetRecurrentStateStorage());
      }

      LaunchProjection(layer.ssm_out, scratch.ssm.out.data(),
                       scratch.attention.output.data(), batch_size, hidden_size,
                       ssm_inner_size, arena_.stream);
    }

    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.attention.output.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                         static_cast<const float*>(layer.ffn_norm.data),
                         scratch.decode.normed.data(), nullptr, batch_size,
                         hidden_size, 1e-6F, arena_.stream);

    LaunchFfnActivation(layer, scratch, batch_size, intermediate_size,
                         hidden_size, arena_.stream);
    LaunchProjection(layer.ffn_down, scratch.ffn.activation.data(),
                     scratch.ffn.out.data(), batch_size, hidden_size,
                     intermediate_size, arena_.stream);
    LaunchBatchedResidualAdd(
        scratch.decode.hidden.data(), scratch.ffn.out.data(),
        scratch.decode.hidden.data(), batch_size, hidden_size, arena_.stream);
    if (capture_prompt_hidden_) {
      if (const auto tap = arena_.GetTargetLayerCaptureIndex(layer_index);
          tap.has_value()) {
        float* const destination =
            d_verification_logits_ + (*tap * hidden_size);
        HIP_CHECK(hipMemcpy2DAsync(
            destination, feature_width * sizeof(float),
            scratch.decode.hidden.data(), hidden_size * sizeof(float),
            hidden_size * sizeof(float), batch_size, hipMemcpyDeviceToDevice,
            arena_.stream));
      }
    }
  }

  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t offset = offsets[index];
    if (capture_prompt_hidden_ && captured_layer_count > 0) {
      HIP_CHECK(
          hipMemcpyAsync(executor.h_verification_hidden_.data(),
                         d_verification_logits_ + offset * feature_width,
                         item.tokens.size() * feature_width * sizeof(float),
                         hipMemcpyDeviceToHost, arena_.stream));
      HIP_CHECK(
          hipMemcpyAsync(executor.arena_.d_target_layer_features,
                         d_verification_logits_ +
                             (offset + item.tokens.size() - 1) * feature_width,
                         feature_width * sizeof(float), hipMemcpyDeviceToDevice,
                         arena_.stream));
    } else if (capture_prompt_hidden_) {
      HIP_CHECK(
          hipMemcpyAsync(executor.h_verification_hidden_.data(),
                         scratch.decode.hidden.data() + offset * hidden_size,
                         item.tokens.size() * hidden_size * sizeof(float),
                         hipMemcpyDeviceToHost, arena_.stream));
    }
    if (&executor == &coordinator) {
      executor.last_hidden_offset_ = (item.tokens.size() - 1) * hidden_size;
    } else {
      HIP_CHECK(hipMemcpyAsync(
          executor.arena_.d_hidden,
          scratch.decode.hidden.data() +
              (offset + item.tokens.size() - 1) * hidden_size,
          hidden_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream));
      executor.last_hidden_offset_ = 0;
    }
  }
  LaunchBatchedRMSNorm(scratch.decode.hidden.data(),
                       static_cast<const float*>(weights_.output_norm.data),
                       scratch.decode.normed.data(), nullptr, batch_size,
                       hidden_size, 1e-6F, arena_.stream);
  LaunchProjection(weights_.output, scratch.decode.normed.data(),
                   d_verification_logits_, batch_size, vocab_size, hidden_size,
                   arena_.stream);
  auto* const d_out_tokens = scratch.decode.prompt_tokens.data();
  LaunchBatchedGPUArgmax(d_verification_logits_, d_out_tokens, batch_size,
                         vocab_size, arena_.stream);
  std::array<tokenization::TokenId, kMaxRows> host_predictions{};
  HIP_CHECK(hipMemcpyAsync(host_predictions.data(), d_out_tokens,
                           batch_size * sizeof(tokenization::TokenId),
                           hipMemcpyDeviceToHost, arena_.stream));
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    auto& executor = *item.executor;
    const std::size_t count = item.tokens.size() * vocab_size;
    const auto* logits = d_verification_logits_ + offsets[index] * vocab_size;
    if (&executor != &coordinator) {
      HIP_CHECK(hipMemcpyAsync(executor.d_verification_logits_, logits,
                               count * sizeof(float), hipMemcpyDeviceToDevice,
                               arena_.stream));
    }
    if (item.capture_logits) {
      executor.h_verification_logits_.resize(count);
      HIP_CHECK(hipMemcpyAsync(executor.h_verification_logits_.data(), logits,
                               count * sizeof(float), hipMemcpyDeviceToHost,
                               arena_.stream));
    }
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  std::vector<std::vector<tokenization::TokenId>> predictions;
  predictions.reserve(items.size());
  for (std::size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    item.executor->arena_.DisableSsmReplayCapture();
    predictions.emplace_back(
        host_predictions.begin() + offsets[index],
        host_predictions.begin() + offsets[index] + item.tokens.size());
  }
  return predictions;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
