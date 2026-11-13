#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {
class Model;

/// Arrival-time admission is supported only with a loaded vision projector.
inline bool VisionMemoryOnDemand(const ext::TextRunnerOptions& options) {
  const auto* policy = ext::FindOption<std::string>(options.provider_options,
                                                    "--vision-memory-policy");
  if (policy && *policy != "conservative" && *policy != "on-demand")
    throw std::invalid_argument(
        "--vision-memory-policy must be conservative or on-demand");
  const bool on_demand = policy && *policy == "on-demand";
  if (on_demand && options.vision_model_path.empty())
    throw std::invalid_argument(
        "--vision-memory-policy on-demand requires --mmproj");
  return on_demand;
}

inline constexpr std::uint32_t DecodeConcurrencyForSessions(
    std::size_t session_count) {
  return static_cast<std::uint32_t>(
      std::clamp<std::size_t>(session_count, 1, 8));
}

#if defined(ENGINE_ENABLE_HIP)
std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

std::shared_ptr<ext::TextModelRunner> CreateQwenFlashNextTextRunner(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    const ext::TextSpeculativeConfig& speculative_config = {},
    std::string artifact_fingerprint = {},
    std::string draft_artifact_fingerprint = {});
#endif

}  // namespace gufo::models::qwen38_flash_next
