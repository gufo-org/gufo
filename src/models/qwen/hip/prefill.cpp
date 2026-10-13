#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>

#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
void QwenGpuExecutor::CheckPrefillCancellation() const {
  if (cancellation_check_ && cancellation_check_())
    throw std::runtime_error("Qwen prefill cancelled");
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  CheckPrefillCancellation();
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.clear();
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    h_prompt_hidden_.reserve(prompt_tokens.size() * captured_layers *
                             weights_.config.hidden_size);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  tokenization::TokenId next_token = 0;
  const std::size_t chunk_capacity =
      std::min<std::size_t>(arena_.GetMaxBatch(), 8);
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += chunk_capacity) {
    CheckPrefillCancellation();
    // A token's persistent state must not depend on whether it arrived in a
    // prompt or was generated. Use the canonical decode arithmetic for both.
    const auto count = std::min(chunk_capacity, prompt_tokens.size() - offset);
    const QwenGpuVerificationItem item{
        this, prompt_tokens.subspan(offset, count),
        start_pos + static_cast<std::uint32_t>(offset), false};
    const auto output = compute_logits && offset + count == prompt_tokens.size()
                            ? QwenGpuChunkOutput::kPrefill
                            : QwenGpuChunkOutput::kPrefillNoLogits;
    const auto predictions =
        ForwardVerificationBatch(std::span(&item, 1), output);
    next_token = predictions.front().back();
    if (capture_prompt_hidden_) {
      const auto hidden = std::span<const float>(h_verification_hidden_);
      h_prompt_hidden_.insert(h_prompt_hidden_.end(), hidden.begin(),
                              hidden.end());
    }
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
