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
  verification_logits_capacity_ = 0;
  HIP_CHECK(hipMalloc(&d_verification_logits_,
                      batch_size * weights_.config.vocab_size * sizeof(float)));
  verification_logits_capacity_ = batch_size;
}

QwenGpuMemoryUsage QwenGpuExecutor::EstimateMemoryUsage(
    const core::ModelConfig& config, std::uint32_t max_context,
    QwenExecutionPolicy policy) {
  auto usage = QwenGpuArena::EstimateMemoryUsage(config, max_context, policy);
  usage.temporary_scratch_bytes +=
      EstimateGpuSamplingWorkspaceBytes(config.vocab_size, max_context);
  // Reserve the bounded fallback for eight requests with eight rows each.
  // Large arenas reuse FFN scratch and normally retain fewer rows.
  const std::size_t rows = std::min<std::size_t>(max_context, 64);
  const std::size_t features = rows * 5 * config.hidden_size;
  const std::size_t feature_rows =
      (features + config.vocab_size - 1) / config.vocab_size;
  usage.temporary_scratch_bytes +=
      std::max(rows, feature_rows) * config.vocab_size * sizeof(float);
  return usage;
}

std::size_t QwenGpuExecutor::SnapshotPayloadBytes(
    std::uint32_t valid_context) const {
  return arena_.SnapshotPayloadBytes(valid_context);
}

QwenGpuMemoryUsage QwenGpuExecutor::GetMemoryUsage() const {
  auto usage = arena_.GetMemoryUsage();
  usage.temporary_scratch_bytes += verification_logits_capacity_ *
                                   weights_.config.vocab_size * sizeof(float);
  usage.temporary_scratch_bytes += sampling_workspace_.SizeBytes();
  usage.temporary_scratch_bytes += vision_input_.Bytes();
  return usage;
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
  try {
    arena_.Reset();
    reset_failure_ = nullptr;
  } catch (...) {
    reset_failure_ = std::current_exception();
  }
  graph_executor_.Reset();
}

std::unique_ptr<QwenGpuSnapshot> QwenGpuExecutor::SaveSnapshot(
    std::uint32_t valid_context) {
  CheckReset();
  auto snapshot = arena_.SaveSnapshot(valid_context);
  snapshot->vision_layout_ = vision_input_.layout();
  return snapshot;
}

void QwenGpuExecutor::ConfigureVision(
    std::shared_ptr<const models::qwen::vision::Prompt> prompt,
    std::shared_ptr<models::qwen::vision::Encoder> encoder) {
  CheckReset();
  const auto* previous = vision_input_.rope();
  if (prompt)
    prompt->rope.Validate(arena_.GetMaxContext());
  vision_input_.Configure(std::move(prompt), std::move(encoder), arena_.stream);
  if (previous != vision_input_.rope())
    graph_executor_.Reset();
}

void QwenGpuExecutor::RestoreSnapshot(const QwenGpuSnapshot& snapshot) {
  CheckReset();
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  arena_.RestoreSnapshot(snapshot);
  vision_input_.RestoreLayout(snapshot.vision_layout_, arena_.stream);
  graph_executor_.Reset();
}

void QwenGpuExecutor::RestoreCompactSnapshot(
    std::span<const std::uint8_t> payload,
    std::uint32_t expected_valid_context) {
  CheckReset();
  replaying_ssm_state_ = false;
  next_token_.reset();
  h_prompt_hidden_.clear();
  h_verification_hidden_.clear();
  h_verification_logits_.clear();
  h_last_hidden_.clear();
  last_verification_rows_ = 0;
  last_hidden_offset_ = 0;
  arena_.RestoreCompactSnapshot(payload, expected_valid_context);
  vision_input_.RestoreLayout(QwenGpuSnapshot::ReadRopeLayout(payload),
                              arena_.stream);
  graph_executor_.Reset();
}

void QwenGpuExecutor::SaveState(std::uint32_t valid_context) {
  CheckReset();
  replaying_ssm_state_ = false;
  arena_.SaveState(valid_context);
  if (arena_.BeginSsmReplayCapture()) {
    graph_executor_.Reset();
  }
}

void QwenGpuExecutor::RestoreState() {
  CheckReset();
  if (arena_.IsSsmReplayCaptureActive())
    arena_.DisableSsmReplayCapture();
  arena_.RestoreState();
  replaying_ssm_state_ = true;
}

void QwenGpuExecutor::FinishVerification() {
  CheckReset();
  arena_.DisableSsmReplayCapture();
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
          nullptr, config.SsmLayerIndex(layer_idx), config.SsmQkvSize(),
          config.ssm_group_count, config.ssm_time_step_rank,
          config.ssm_state_size, config.SsmValueSize(), rows,
          config.ssm_time_step_rank, config.ssm_inner_size, arena_.stream, {},
          arena_.GetRecurrentStateStorage());
      offset += rows;
    }
  }
}

std::span<const float> QwenGpuExecutor::CopyLastLogits() {
  CheckReset();
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(h_logits_.data(), scratch.decode.logits.data(),
                           h_logits_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_logits_;
}

tokenization::TokenId QwenGpuExecutor::SampleLastLogits(
    sampling::SamplerState& sampler) {
  CheckReset();
  if (sampler.config().constraint)
    return sampler.Sample(CopyLastLogits());
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = sampler.Uniform();
  }

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(scratch.decode.logits.data(), d_out_token,
                    weights_.config.vocab_size, parameters,
                    sampler.penalties().data(), sampler.penalties().size(),
                    &sampling_workspace_, arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(token, weights_.config.vocab_size);
}

tokenization::TokenId QwenGpuExecutor::SampleCachedLogits(
    std::span<const float> logits, sampling::SamplerState& sampler) {
  CheckReset();
  if (logits.size() != weights_.config.vocab_size)
    throw std::invalid_argument(
        "cached Qwen frontier has the wrong vocabulary");
  if (sampler.config().constraint)
    return sampler.Sample(logits);
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(scratch.decode.logits.data(), logits.data(),
                           logits.size_bytes(), hipMemcpyHostToDevice,
                           arena_.stream));
  return SampleLastLogits(sampler);
}

GpuSamplingParameters QwenGpuExecutor::PrepareGpuSamplingParameters(
    const sampling::SamplerState& sampler) {
  const auto& config = sampler.config();

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
  CheckReset();
  if (row >= last_verification_rows_ || d_verification_logits_ == nullptr) {
    throw std::out_of_range("Qwen verification logit row is unavailable");
  }
  if (sampler.config().constraint)
    return sampler.Sample(CopyVerificationLogits(row));
  auto parameters = PrepareGpuSamplingParameters(sampler);
  if (sampler.config().uses_random_sampling()) {
    parameters.uniform = sampler.Uniform();
  }
  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSampling(d_verification_logits_ + (row * weights_.config.vocab_size),
                    d_out_token, weights_.config.vocab_size, parameters,
                    sampler.penalties().data(), sampler.penalties().size(),
                    &sampling_workspace_, arena_.stream);

  tokenization::TokenId token = 0;
  HIP_CHECK(hipMemcpyAsync(&token, d_out_token, sizeof(token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return CheckedSampleToken(token, weights_.config.vocab_size);
}

QwenSampledVerificationResult QwenGpuExecutor::VerifySampledToken(
    std::size_t row, tokenization::TokenId draft_token,
    std::span<const tokenization::TokenId> draft_candidate_ids,
    std::span<const float> draft_candidate_probabilities,
    double draft_token_probability, sampling::SamplerState& sampler) {
  CheckReset();
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

  if (sampler.config().constraint) {
    const auto target = sampler.Distribution(CopyVerificationLogits(row));
    if (sampler.Uniform() * draft_token_probability <
        target.probability(draft_token)) {
      sampler.Accept(draft_token);
      return {.token = draft_token, .accepted = true};
    }
    return {.token = target.SampleResidual(draft_candidate_ids,
                                           draft_candidate_probabilities,
                                           sampler.mutable_rng_state()),
            .accepted = false};
  }

  auto parameters = PrepareGpuSamplingParameters(sampler);
  const double acceptance_uniform = sampler.Uniform();
  const std::uint64_t residual_rng_checkpoint = sampler.rng_state();
  const double residual_uniform = sampler.Uniform();

  auto scratch = arena_.GetScratchView();
  auto* const d_out_token = scratch.decode.sampled_token.data();
  LaunchGPUSpeculativeSampling(
      d_verification_logits_ + (row * weights_.config.vocab_size), d_out_token,
      sampling_workspace_.speculative_accepted, weights_.config.vocab_size,
      parameters, draft_token, static_cast<float>(draft_token_probability),
      draft_candidate_ids.data(), draft_candidate_probabilities.data(),
      draft_candidate_ids.size(), acceptance_uniform, residual_uniform,
      sampler.penalties().data(), sampler.penalties().size(),
      &sampling_workspace_, arena_.stream);

  QwenSampledVerificationResult result;
  std::uint32_t accepted = 0;
  HIP_CHECK(hipMemcpyAsync(&result.token, d_out_token, sizeof(result.token),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipMemcpyAsync(&accepted, sampling_workspace_.speculative_accepted,
                           sizeof(accepted), hipMemcpyDeviceToHost,
                           arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  if (result.token >= weights_.config.vocab_size || accepted > 1)
    throw std::runtime_error(
        "target logit distribution contains no finite values");
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
  CheckReset();
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

std::span<const float> QwenGpuExecutor::CopyLastHidden() {
  CheckReset();
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
  Reset();
  return GenerateFromPrefix(prompt_tokens, 0, options, on_token);
}

std::vector<tokenization::TokenId> QwenGpuExecutor::GenerateFromPrefix(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::size_t cached_prefix_tokens, const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  CheckReset();
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
  sampling::SamplerState sampler(options.sampling, prompt_tokens);
  if (!options.sampling.can_use_unmodified_argmax()) {
    next_token = SampleLastLogits(sampler);
  }

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (tokenizer_->IsStopToken(next_token)) {
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
