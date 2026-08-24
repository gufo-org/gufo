#include "src/models/qwen/modules/embed.hpp"

#include "src/models/qwen/qwen_forward.hpp"  // ForwardEmbedding

namespace strix::models::qwen {

void EmbedForward(const CpuModuleContext&, std::uint32_t token_id,
                  const QwenTensorRef& token_embd, std::size_t hidden_size,
                  std::span<float> out) noexcept {
  // CPU backend: delegates to the existing embedding lookup (already a small
  // standalone function; no body duplicated).
  ForwardEmbedding(token_id, token_embd, hidden_size, out);
}

}  // namespace strix::models::qwen
