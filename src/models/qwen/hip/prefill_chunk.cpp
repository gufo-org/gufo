#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
std::vector<tokenization::TokenId> QwenGpuExecutor::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_logits) {
  return ForwardDecodeEquivalentVerificationChunk(candidate_tokens, start_pos,
                                                  capture_logits);
}

void QwenGpuExecutor::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  if (committed_tokens.empty()) {
    return;
  }
  CheckReset();
  if (start_pos > arena_.GetMaxContext() ||
      committed_tokens.size() > arena_.GetMaxContext() - start_pos)
    throw std::length_error("committed Qwen chunk exceeds context");
  models::qwen::ContinuationWrite write(
      continuation_hooks_, start_pos,
      start_pos + static_cast<std::uint32_t>(committed_tokens.size()));
  std::size_t replayed = 0;
  if (replaying_ssm_state_) {
    while (replayed < committed_tokens.size() &&
           arena_.CanReplaySsmPosition(start_pos +
                                       static_cast<std::uint32_t>(replayed)))
      ++replayed;
    if (replayed != 0)
      ReplaySsmState(start_pos, static_cast<std::uint32_t>(replayed));
  }
  replaying_ssm_state_ = false;
  if (replayed == committed_tokens.size()) {
    write.Commit(
        start_pos + static_cast<std::uint32_t>(committed_tokens.size()), false);
    return;
  }
  arena_.DisableSsmReplayCapture();

  const bool capture_hidden = capture_prompt_hidden_;
  capture_prompt_hidden_ = false;
  try {
    // Unrecorded rows need a complete evaluation, including their attention
    // and feature state; restoring recorded recurrence alone is insufficient.
    for (std::size_t row = replayed; row < committed_tokens.size(); ++row)
      (void)ForwardToken(committed_tokens[row],
                         start_pos + static_cast<std::uint32_t>(row), false);
  } catch (...) {
    capture_prompt_hidden_ = capture_hidden;
    throw;
  }
  capture_prompt_hidden_ = capture_hidden;
  write.Commit(start_pos + static_cast<std::uint32_t>(committed_tokens.size()),
               false);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
