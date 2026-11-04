#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::deepseek_v4_flash {

[[nodiscard]] bool ProbeDeepSeek(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeekGguf(const core::GgufReader& reader);
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
DeepSeekSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v4_flash
