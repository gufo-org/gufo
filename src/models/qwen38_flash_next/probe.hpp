#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);
/// The Gyro-M variant: the same trunk with AgentionAI TQ2_T trellis routed
/// experts, a storage type core cannot size. Requires a reader opened with
/// `core::GgufOpenOptions::allow_unsized_tensors`.
[[nodiscard]] bool ProbeFlashNextGyro(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGyroGguf(const core::GgufReader& reader);
/// The runner descriptor and the provider probe both use these defaults.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults();
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::qwen38_flash_next
