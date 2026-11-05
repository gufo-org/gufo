#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::qwen {

[[nodiscard]] bool ProbeQwen(std::string_view model_root);
[[nodiscard]] bool ProbeQwenGguf(const core::GgufReader& reader);
/// Qwen3.8 defaults for the 27B shape; other shapes leave controls to the
/// caller. The runner descriptor and the provider probe both use these.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
QwenSamplingDefaults(const core::ModelConfig& config);
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
QwenSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::qwen
