#pragma once

#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen {

[[nodiscard]] bool ProbeQwen(std::string_view model_root);
[[nodiscard]] bool ProbeQwenGguf(const core::GgufReader& reader);

}  // namespace gufo::models::qwen
