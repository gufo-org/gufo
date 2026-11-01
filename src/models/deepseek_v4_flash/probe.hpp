#pragma once

#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::deepseek_v4_flash {

[[nodiscard]] bool ProbeDeepSeek(std::string_view model_root);
[[nodiscard]] bool ProbeDeepSeekGguf(const core::GgufReader& reader);

}  // namespace gufo::models::deepseek_v4_flash
