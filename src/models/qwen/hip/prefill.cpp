#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <stdexcept>

#include "src/models/qwen/hip/executor.hpp"

namespace strix::hip {
tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.clear();
    h_prompt_hidden_.reserve(prompt_tokens.size() *
                             weights_.config.hidden_size);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  tokenization::TokenId next_token = 0;
  for (std::size_t offset = 0; offset < prompt_tokens.size();
       offset += arena_.GetMaxBatch()) {
    const std::size_t chunk_size = std::min<std::size_t>(
        arena_.GetMaxBatch(), prompt_tokens.size() - offset);
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    next_token =
        ForwardPromptChunk(prompt_tokens.subspan(offset, chunk_size),
                           start_pos + static_cast<std::uint32_t>(offset),
                           compute_logits && is_last);
  }
  return next_token;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
