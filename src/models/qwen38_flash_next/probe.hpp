#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);
/// The runner descriptor and the provider probe both use these defaults.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults();
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::qwen38_flash_next
