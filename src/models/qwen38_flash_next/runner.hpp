#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {
class Model;

inline constexpr std::uint32_t DecodeConcurrencyForSessions(
    std::size_t session_count) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(session_count, 1, 8));
}

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

/// Model-backed callers keep conservative state estimates: this factory does
/// not own loading or a configured serving session count. A model already
/// attached to a serving admission coordinator is rejected.
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {});
#endif

}  // namespace gufo::models::qwen38_flash_next
