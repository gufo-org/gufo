#pragma once

#include <string_view>

#include "gufo/ext/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);

}  // namespace gufo::models::qwen38_flash_next
