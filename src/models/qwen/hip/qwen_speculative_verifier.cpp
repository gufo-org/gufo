#include "src/models/qwen/hip/qwen_speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
namespace {

using speculative::ISpeculativeTargetExecutor;
using speculative::SampledVerificationResult;
using speculative::TargetVerificationBatch;
using speculative::TargetVerificationItem;
using speculative::VerificationChunkResult;

class QwenGpuSpeculativeTarget final : public ISpeculativeTargetExecutor {
public:
  explicit QwenGpuSpeculativeTarget(hip::QwenGpuExecutor& executor)
      : executor_(executor) {}

  void Reset() noexcept override { executor_.Reset(); }

  sampling::TokenId ForwardPromptBatch(
      std::span<const sampling::TokenId> prompt_tokens) override {
    return executor_.ForwardPromptBatch(prompt_tokens);
  }
  sampling::TokenId ForwardPromptSuffix(
      std::span<const sampling::TokenId> tokens, std::uint32_t position,
      bool compute_logits) override {
    return executor_.ForwardPromptBatch(tokens, position, compute_logits);
  }

  sampling::TokenId ForwardToken(sampling::TokenId token_id, std::uint32_t pos,
                                 bool compute_logits) override {
    return executor_.ForwardToken(token_id, pos, compute_logits);
  }

  void SaveState(std::uint32_t valid_context) override {
    executor_.SaveState(valid_context);
  }

  void RestoreState() override { executor_.RestoreState(); }

  void SetPromptHiddenCapture(
      bool enabled, std::span<const std::uint32_t> target_layer_ids) override {
    executor_.SetPromptHiddenCapture(enabled, target_layer_ids);
  }

  std::span<const float> GetPromptHiddenStates() const noexcept override {
    return executor_.GetPromptHiddenStates();
  }

  std::span<const float> CopyLastHidden() override {
    return executor_.CopyLastHidden();
  }

  std::span<const float> CopyLastLogits() override {
    return executor_.CopyLastLogits();
  }

  std::span<const float> CopyVerificationLogits(std::size_t row) override {
    return executor_.CopyVerificationLogits(row);
  }

  bool SupportsDeviceResidentSampling() const noexcept override { return true; }

  sampling::TokenId SampleLastLogits(sampling::SamplerState& sampler) override {
    return executor_.SampleLastLogits(sampler);
  }

  sampling::TokenId SampleVerificationLogits(
      std::size_t row, sampling::SamplerState& sampler) override {
    return executor_.SampleVerificationLogits(row, sampler);
  }

  SampledVerificationResult VerifySampledToken(
      std::size_t row, sampling::TokenId draft_token,
      std::span<const sampling::TokenId> draft_candidate_ids,
      std::span<const float> draft_candidate_probabilities,
      double draft_token_probability,
      sampling::SamplerState& sampler) override {
    const auto result = executor_.VerifySampledToken(
        row, draft_token, draft_candidate_ids, draft_candidate_probabilities,
        draft_token_probability, sampler);
    return {.token = result.token, .accepted = result.accepted};
  }

  VerificationChunkResult ForwardVerificationChunk(
      std::span<const sampling::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden,
      bool capture_logits) override {
    return CaptureVerification(executor_.ForwardVerificationChunk(
                                   candidate_tokens, start_pos, capture_logits),
                               capture_hidden, capture_logits);
  }

  TargetVerificationBatch ForwardVerificationBatch(
      std::span<const TargetVerificationItem> items) override {
    if (items.size() < 2)
      return ISpeculativeTargetExecutor::ForwardVerificationBatch(items);
    std::vector<QwenGpuSpeculativeTarget*> targets;
    std::vector<hip::QwenGpuVerificationItem> gpu_items;
    targets.reserve(items.size());
    gpu_items.reserve(items.size());
    std::size_t rows = 0;
    for (const auto& item : items) {
      auto* target = dynamic_cast<QwenGpuSpeculativeTarget*>(item.executor);
      if (target == nullptr)
        return ISpeculativeTargetExecutor::ForwardVerificationBatch(items);
      targets.push_back(target);
      gpu_items.push_back({&target->executor_, item.tokens, item.position,
                           item.capture_logits});
      rows += item.tokens.size();
    }
    if (items.size() > 8 ||
        rows > targets.front()->executor_.GetMaxPromptBatch())
      return ISpeculativeTargetExecutor::ForwardVerificationBatch(items);
    auto predictions =
        hip::QwenGpuExecutor::ForwardVerificationBatch(gpu_items);
    TargetVerificationBatch result;
    result.physical_width = items.size();
    result.chunks.reserve(items.size());
    for (std::size_t index = 0; index < items.size(); ++index) {
      result.chunks.push_back(targets[index]->CaptureVerification(
          std::move(predictions[index]), items[index].capture_hidden,
          items[index].capture_logits));
    }
    return result;
  }

  void CommitVerificationChunk(
      std::span<const sampling::TokenId> committed_tokens,
      std::uint32_t start_pos) override {
    executor_.CommitVerificationChunk(committed_tokens, start_pos);
  }

  void FinishVerification() override { executor_.FinishVerification(); }

  sampling::TokenId GetEosTokenId() const noexcept override {
    return executor_.GetTokenizer().GetEosTokenId();
  }
  bool IsStopToken(sampling::TokenId token) const noexcept override {
    return executor_.GetTokenizer().IsStopToken(token);
  }

  std::size_t VocabularySize() const noexcept override {
    return executor_.GetConfig().vocab_size;
  }

  std::string_view DecodeToken(
      sampling::TokenId token_id) const noexcept override {
    return executor_.GetTokenizer().DecodeToken(token_id);
  }

private:
  VerificationChunkResult CaptureVerification(
      std::vector<sampling::TokenId> predictions, bool capture_hidden,
      bool capture_logits) {
    VerificationChunkResult result;
    result.predictions = std::move(predictions);
    if (capture_hidden) {
      const auto hidden = executor_.GetVerificationHiddenStates();
      result.hidden_states.assign(hidden.begin(), hidden.end());
      if (!result.predictions.empty()) {
        if (result.hidden_states.size() % result.predictions.size() != 0) {
          throw std::runtime_error(
              "batched verification hidden-state capture is malformed");
        }
        result.hidden_width =
            result.hidden_states.size() / result.predictions.size();
      }
    }
    if (capture_logits) {
      const auto logits = executor_.GetVerificationLogits();
      result.logits.assign(logits.begin(), logits.end());
      result.vocab_size = executor_.GetModel().GetConfig().vocab_size;
    }
    return result;
  }

  hip::QwenGpuExecutor& executor_;
};

}  // namespace

std::unique_ptr<speculative::SpeculativeVerifier>
CreateQwenGpuSpeculativeVerifier(
    QwenGpuExecutor& target_executor,
    std::unique_ptr<speculative::IDraftBackend> draft_backend,
    speculative::SpeculativeOptions options) {
  return std::make_unique<speculative::SpeculativeVerifier>(
      std::make_unique<QwenGpuSpeculativeTarget>(target_executor),
      std::move(draft_backend), options);
}

std::unique_ptr<speculative::SpeculativeVerifier>
CreateQwenGpuSpeculativeVerifier(QwenGpuExecutor& target_executor,
                                 speculative::IDraftBackend& draft_backend,
                                 speculative::SpeculativeOptions options) {
  return std::make_unique<speculative::SpeculativeVerifier>(
      std::make_unique<QwenGpuSpeculativeTarget>(target_executor),
      draft_backend, options);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)
