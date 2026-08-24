#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/executor.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"

namespace strix::hip {
namespace {

bool IsSsmReplayEnabled() noexcept {
  const char* value = std::getenv("STRIX_DISABLE_SSM_REPLAY");
  if (value == nullptr) {
    return true;
  }
  const std::string_view setting{value};
  return setting == "0" || setting == "false" || setting == "off";
}

const QwenGpuModel& RequireModel(
    const std::shared_ptr<const QwenGpuModel>& model) {
  if (model == nullptr) {
    throw std::invalid_argument("Qwen GPU model must not be null");
  }
  return *model;
}

[[nodiscard]] detail::HipGraphCaptureKey BuildGraphCaptureKey(
    const models::QwenModelWeights& weights,
    const QwenExecutionPolicy& policy, std::uint32_t max_context) noexcept {
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
    workload = ExtendQwenGraphWorkloadIdentity(
        workload, resolution.plan.Fingerprint());
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
      arena_(weights_.config, max_context),
      graph_key_(BuildGraphCaptureKey(weights_, policy_, max_context)),
      h_logits_(weights_.config.vocab_size, 0.0F) {
  detail::EmitQwenExecutionPolicy(policy_.Fingerprint());
}

QwenGpuExecutor::~QwenGpuExecutor() {
  (void)hipStreamSynchronize(arena_.stream);
}

void QwenGpuExecutor::Reset() noexcept {
  replaying_ssm_state_ = false;
  h_prompt_hidden_.clear();
  h_last_hidden_.clear();
  last_hidden_offset_ = 0;
  arena_.Reset();
  graph_executor_.Reset();
}

void QwenGpuExecutor::SaveState(std::uint32_t valid_context) {
  replaying_ssm_state_ = false;
  arena_.SaveState(valid_context);
  if (IsSsmReplayEnabled() && arena_.BeginSsmReplayCapture()) {
    graph_executor_.Reset();
  }
}

void QwenGpuExecutor::RestoreState() {
  arena_.RestoreState();
  replaying_ssm_state_ = IsSsmReplayEnabled();
}

void QwenGpuExecutor::ReplaySsmState(std::uint32_t position) {
  const auto& config = weights_.config;
  auto scratch = arena_.GetScratchView();
  for (std::uint32_t layer_idx = 0; layer_idx < config.num_layers;
       ++layer_idx) {
    const auto& layer = weights_.layers[layer_idx];
    if (layer.is_full_attention) {
      continue;
    }

    LaunchSSMConvRecurrence(
        arena_.GetReplayQkv(layer_idx, position),
        static_cast<const float*>(layer.ssm_conv1d.data),
        arena_.d_ssm_conv_state, scratch.ssm.conv_out.data(),
        arena_.d_ssm_deltanet_state,
        arena_.GetReplayAlpha(layer_idx, position),
        arena_.GetReplayBeta(layer_idx, position),
        static_cast<const float*>(layer.ssm_a.data),
        static_cast<const float*>(layer.ssm_dt.data), nullptr, nullptr,
        scratch.ssm.out.data(), layer_idx, config.SsmQkvSize(),
        config.ssm_group_count, config.ssm_time_step_rank,
        config.ssm_state_size, config.SsmValueSize(), arena_.stream);
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

void QwenGpuExecutor::SetPromptHiddenCapture(bool enabled) {
  capture_prompt_hidden_ = enabled;
  h_prompt_hidden_.clear();
}

std::span<const float> QwenGpuExecutor::CopyLastHidden() {
  h_last_hidden_.resize(weights_.config.hidden_size);
  auto scratch = arena_.GetScratchView(arena_.GetMaxBatch());
  HIP_CHECK(hipMemcpyAsync(h_last_hidden_.data(),
                           scratch.decode.hidden.data() + last_hidden_offset_,
                           h_last_hidden_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));
  return h_last_hidden_;
}

std::vector<tokenization::TokenId> QwenGpuExecutor::Generate(
    std::span<const tokenization::TokenId> prompt_tokens,
    const models::GenerationOptions& options,
    const std::function<bool(tokenization::TokenId, std::string_view)>&
        on_token) {
  std::vector<tokenization::TokenId> output_tokens;
  if (prompt_tokens.empty()) {
    return output_tokens;
  }

  arena_.Reset();

  // 1. Batched GPU prompt prefill
  tokenization::TokenId next_token = ForwardPromptBatch(prompt_tokens);

  std::size_t cur_pos = prompt_tokens.size();
  const auto eos_id = tokenizer_->GetEosTokenId();

  // 2. Auto-regressive decode generation loop
  while (output_tokens.size() < options.max_new_tokens) {
    if (next_token == eos_id || next_token == 151643U ||
        next_token == 248044U || next_token == 248046U) {
      break;
    }

    output_tokens.push_back(next_token);
    if (on_token) {
      const auto piece = tokenizer_->DecodeToken(next_token);
      if (!on_token(next_token, piece)) {
        break;
      }
    }

    next_token = ForwardToken(next_token, static_cast<std::uint32_t>(cur_pos));
    ++cur_pos;
  }

  return output_tokens;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
