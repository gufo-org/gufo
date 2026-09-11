#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip {
namespace {

const QwenGpuModel& RequireModel(
    const std::shared_ptr<const QwenGpuModel>& model) {
  if (model == nullptr) {
    throw std::invalid_argument("Qwen GPU model must not be null");
  }
  return *model;
}

[[nodiscard]] detail::HipGraphCaptureKey BuildGraphCaptureKey(
    const models::QwenModelWeights& weights, const QwenExecutionPolicy& policy,
    std::uint32_t max_context) noexcept {
  const auto& config = weights.config;
  std::uint64_t workload = BeginQwenGraphWorkloadIdentity();
  const std::uint64_t fields[] = {
      config.num_layers,
      config.hidden_size,
      config.intermediate_size,
      config.num_attention_heads,
      config.num_key_value_heads,
      config.head_dim,
      config.vocab_size,
      max_context,
      1U,  // Captured decode always computes logits.
  };
  for (const std::uint64_t field : fields) {
    workload = ExtendQwenGraphWorkloadIdentity(workload, field);
  }
  for (std::uint32_t layer_index = 0; layer_index < config.num_layers;
       ++layer_index) {
    const auto& layer = weights.layers[layer_index];
    const auto resolution = ResolveQwenLayerRouteWithReasons(
        policy, QwenExecutionMode::kDecode, layer.is_full_attention);
    workload = ExtendQwenGraphWorkloadIdentity(workload,
                                               resolution.plan.Fingerprint());
  }
  return {
      .execution_identity = policy.Fingerprint(),
      .workload_identity = workload,
  };
}

}  // namespace

QwenGpuExecutor::QwenGpuExecutor(std::shared_ptr<const QwenGpuModel> model,
                                 std::uint32_t max_context,
                                 QwenExecutionPolicy policy)
    : model_(std::move(model)),
      weights_(RequireModel(model_).GetWeights()),
      tokenizer_(&model_->GetTokenizer()),
      policy_(policy),
      arena_(weights_.config, max_context, policy_),
      graph_key_(BuildGraphCaptureKey(weights_, policy_, max_context)),
      h_logits_(weights_.config.vocab_size, 0.0F) {
  AllocateGpuSamplingWorkspace(&sampling_workspace_, weights_.config.vocab_size,
                               max_context);
  h_penalty_tokens_.reserve(max_context);
  h_penalty_counts_.reserve(max_context);
  detail::EmitQwenExecutionPolicy(policy_.Fingerprint());
}

QwenGpuExecutor::~QwenGpuExecutor() {
  (void)hipStreamSynchronize(arena_.stream);
  if (d_verification_logits_ != nullptr) {
    (void)hipFree(d_verification_logits_);
  }
  FreeGpuSamplingWorkspace(&sampling_workspace_);
}

void QwenGpuExecutor::EnsureVerificationLogits(std::size_t batch_size) {
  if (batch_size <= verification_logits_capacity_) {
    return;
  }
  if (d_verification_logits_ != nullptr) {
    HIP_CHECK(hipFree(d_verification_logits_));
    d_verification_logits_ = nullptr;
  }
  HIP_CHECK(hipMalloc(&d_verification_logits_,
                      batch_size * weights_.config.vocab_size * sizeof(float)));
  verification_logits_capacity_ = batch_size;
}

void QwenGpuExecutor::Reset() noexcept {
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  arena_.Reset();
  graph_executor_.Reset();
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuExecutor::SaveSnapshot(
    std::uint32_t valid_context) {
  return arena_.SaveSnapshot(valid_context);
}

void QwenGpuExecutor::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  arena_.RestoreSnapshot(snapshot);
  graph_executor_.Reset();
}

void QwenGpuExecutor::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  arena_.RestoreCompactSnapshot(payload, expected_valid_context);
  graph_executor_.Reset();
}

void QwenGpuExecutor::SaveState(std::uint32_t valid_context) {
  replaying_ssm_state_ = false;
  arena_.SaveState(valid_context);
  if (arena_.BeginSsmReplayCapture()) {
    graph_executor_.Reset();
  }
}

void QwenGpuExecutor::RestoreState() {
  arena_.RestoreState();
  replaying_ssm_state_ = true;
}

void QwenGpuExecutor::ReplaySsmState(std::uint32_t position,
                                     std::uint32_t count) {
  const auto& config = weights_.config;
  auto scratch = arena_.GetScratchView();
  for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
       ++layer_idx) {
    const auto& layer = weights_.layers[layer_idx];
    if (layer.is_full_attention) {
      continue;
    }

    // Layers have independent recorded inputs and recurrent state. Replay each
    // layer's committed rows together, splitting only at ring/scratch bounds.
    for (std::uint32_t offset = 0; offset < count;) {
      const auto start = position + offset;
      const auto until_wrap = static_cast<std::uint32_t>(
          kSsmReplayCapacity - (start % kSsmReplayCapacity));
      const auto rows =
          std::min({count - offset, until_wrap, arena_.GetMaxBatch()});
      LaunchSSMConvRecurrenceRows(
          arena_.GetReplayQkv(layer_idx, start),
          static_cast<const float*>(layer.ssm_conv1d.data),
          arena_.d_ssm_conv_state, scratch.ssm.conv_out.data(),
          arena_.d_ssm_deltanet_state, arena_.GetReplayAlpha(layer_idx, start),
          arena_.GetReplayBeta(layer_idx, start),
          static_cast<const float*>(layer.ssm_a.data),
          static_cast<const float*>(layer.ssm_dt.data), nullptr, nullptr,
          nullptr, layer_idx, config.SsmQkvSize(), config.ssm_group_count,
          config.ssm_time_step_rank, config.ssm_state_size,
          config.SsmValueSize(), rows, config.ssm_time_step_rank,
          config.ssm_inner_size, arena_.stream, {},
          arena_.GetRecurrentStateStorage());
      offset += rows;
    }
  }
}

std::span<const float> QwenGpuExecutor::CopyLastLogits() {
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(h_logits_.data(), scratch.decode.logits.data(),
                           h_logits_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

tokenization::TokenId QwenGpuExecutor::SampleLastLogits(
    sampling::SamplerState& sampler) {
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = static_cast<float>(sampler.Uniform());
  }

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(
      scratch.decode.logits.data(), d_out_token, weights_.config.vocab_size,
      parameters, h_penalty_tokens_.data(), h_penalty_counts_.data(),
      h_penalty_tokens_.size(), &sampling_workspace_, arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return token;
}

GpuSamplingParameters QwenGpuExecutor::PrepareGpuSamplingParameters(
    const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();
  const auto history = sampler.history();
  h_penalty_tokens_.assign(history.begin(), history.end());
  std::ranges::sort(h_penalty_tokens_);
  h_penalty_counts_.clear();
  std::size_t unique_count = 0;
  for (const std::uint32_t token : h_penalty_tokens_) {
    if (unique_count == 0 || h_penalty_tokens_[unique_count - 1U] != token) {
      h_penalty_tokens_[unique_count++] = token;
      h_penalty_counts_.push_back(1U);
    } else {
      ++h_penalty_counts_.back();
    }
  }
  h_penalty_tokens_.resize(unique_count);
  if (!config.penalties_enabled()) {
    h_penalty_tokens_.clear();
    h_penalty_counts_.clear();
  }

  return GpuSamplingParameters{
      .temperature = config.temperature,
      .top_k = config.top_k,
      .top_p = config.top_p,
      .min_p = config.min_p,
      .min_keep = config.min_keep,
      .repeat_penalty = config.repeat_penalty,
      .frequency_penalty = config.frequency_penalty,
      .presence_penalty = config.presence_penalty,
  };
}

tokenization::TokenId QwenGpuExecutor::SampleVerificationLogits(
    std::size_t row, sampling::SamplerState& sampler) {
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = static_cast<float>(sampler.Uniform());
  }
  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(d_verification_logits_ + (row * weights_.config.vocab_size),
                    d_out_token, weights_.config.vocab_size, parameters,
                    h_penalty_tokens_.data(), h_penalty_counts_.data(),
                    h_penalty_tokens_.size(), &sampling_workspace_,
                    arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return token;
}

QwenSampledVerificationResult QwenGpuExecutor::VerifySampledToken(
    std::size_t row, tokenization::TokenId draft_token,
    std::span<const tokenization::TokenId> draft_candidate_ids,
    std::span<const float> draft_candidate_probabilities,
    double draft_token_probability, sampling::SamplerState& sampler) {
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  if (draft_candidate_ids.empty() ||
      draft_candidate_ids.size() != draft_candidate_probabilities.size() ||
      !std::isfinite(draft_token_probability) ||
      draft_token_probability <= 0.0 || draft_token_probability > 1.0) {
    throw std::invalid_argument(
        "Qwen sampled verification proposal is malformed");
  }

  auto parameters = PrepareGpuSamplingParameters(sampler);
  const float acceptance_uniform = static_cast<float>(sampler.Uniform());
  const std::uint64_t residual_rng_checkpoint = sampler.rng_state();
  const float residual_uniform = static_cast<float>(sampler.Uniform());

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSpeculativeSampling(
      d_verification_logits_ + (row * weights_.config.vocab_size), d_out_token,
      sampling_workspace_.speculative_accepted, weights_.config.vocab_size,
      parameters, draft_token, static_cast<float>(draft_token_probability),
      draft_candidate_ids.data(), draft_candidate_probabilities.data(),
      draft_candidate_ids.size(), acceptance_uniform, residual_uniform,
      h_penalty_tokens_.data(), h_penalty_counts_.data(),
      h_penalty_tokens_.size(), &sampling_workspace_, arena_.stream);

  QwenSampledVerificationResult result;
  std::uint32_t accepted = 0;
  HIP_CHECK(hipMemcpyAsync(&result.token, d_out_token, sizeof(result.token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipMemcpyAsync(&accepted, sampling_workspace_.speculative_accepted,
                           sizeof(accepted), hipMemcpyDeviceToHost,
                           arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  result.accepted = accepted != 0U;
  if (result.accepted) {
    // Lossless rejection sampling consumes a residual draw only when the
    // proposal is rejected. The GPU receives it eagerly so acceptance remains
    // a single synchronization, then restores the exact host RNG state here.
    sampler.SetRngState(residual_rng_checkpoint);
    sampler.Accept(draft_token);
  }
  return result;
}

std::span<const float> QwenGpuExecutor::CopyVerificationLogits(
    std::size_t row) {
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  HIP_CHECK(hipMemcpyAsync(
      h_logits_.data(),
      d_verification_logits_ + (row * weights_.config.vocab_size),
      h_logits_.size() * sizeof(float), hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

void QwenGpuExecutor::SetPromptHiddenCapture(
    bool enabled, std::span<const std::uint32_t> target_layer_ids) {
  capture_prompt_hidden_ = enabled;
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  arena_.SetTargetLayerCapture(enabled ? target_layer_ids
                                       : std::span<const std::uint32_t>{});
}

void QwenGpuExecutor::SetVerificationPolicy(
    QwenVerificationPolicy policy) noexcept {
  if (policy.bf16_from_layer < 0 ||
      policy.bf16_from_layer > static_cast<int>(weights_.config.num_layers)) {
    policy.bf16_from_layer = -1;
  }
  if (policy.fp32_from_layer < 0 ||
      policy.fp32_from_layer > static_cast<int>(weights_.config.num_layers)) {
    policy.fp32_from_layer = -1;
  }
  verification_policy_ = policy;
}

std::span<const float> QwenGpuExecutor::CopyLastHidden() {
  const std::size_t hidden_size = weights_.config.hidden_size;
  const std::size_t target_layer_count = arena_.GetTargetLayerCapture().size();
  if (target_layer_count > 0) {
    h_last_hidden_.resize(target_layer_count * hidden_size);
    HIP_CHECK(hipMemcpyAsync(h_last_hidden_.data(),
                             arena_.d_target_layer_features,
                             h_last_hidden_.size() * sizeof(float),
                             hipMemcpyDeviceToHost, arena_.stream));
  } else {
    h_last_hidden_.resize(hidden_size);
    auto scratch = arena_.GetScratchView(arena_.GetMaxBatch());
    HIP_CHECK(hipMemcpyAsync(h_last_hidden_.data(),
                             scratch.decode.hidden.data() + last_hidden_offset_,
                             hidden_size * sizeof(float), hipMemcpyDeviceToHost,
                             arena_.stream));
  }
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_last_hidden_;
}

std::vector<tokenization::TokenId> QwenGpuExecutor::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  next_token_.reset();
  arena_.Reset();
  return GenerateFromPrefix(prompt_tokens, 0, options, on_token);
}

std::vector<tokenization::TokenId> QwenGpuExecutor::GenerateFromPrefix(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::size_t cached_prefix_tokens, const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }
  if (cached_prefix_tokens > prompt_tokens.size()) {
    throw std::invalid_argument(
        "cached Qwen prefix exceeds the rendered prompt");
  }

  tokenization::TokenId next_token = 0;
  if (cached_prefix_tokens < prompt_tokens.size()) {
    next_token =
        ForwardPromptBatch(prompt_tokens.subspan(cached_prefix_tokens),
                           static_cast<std::uint32_t>(cached_prefix_tokens));
  } else if (next_token_.has_value()) {
    next_token = *next_token_;
  } else {
    throw std::logic_error("Qwen retained prefix has no next-token frontier");
  }

  std::size_t cur_pos = prompt_tokens.size();
  const auto eos_id = tokenizer_->GetEosTokenId();
  sampling::SamplerState sampler(options.sampling, prompt_tokens);
  if (!options.sampling.can_use_unmodified_argmax()) {
    next_token = SampleLastLogits(sampler);
  }

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == eos_id || next_token == 151643U ||
        next_token == 248044U || next_token == 248046U) {
      break;
    }

    output_tokens.push_back(next_token);
    sampler.Accept(next_token);
    if (on_token) {
      const auto piece = tokenizer_->DecodeToken(next_token);
      if (!on_token(next_token, piece)) {
        break;
      }
    }

    next_token = ForwardToken(next_token, static_cast<std::uint32_t>(cur_pos));
    if (!options.sampling.can_use_unmodified_argmax()) {
      next_token = SampleLastLogits(sampler);
    }
    ++cur_pos;
  }

  next_token_ = next_token;
  return output_tokens;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
