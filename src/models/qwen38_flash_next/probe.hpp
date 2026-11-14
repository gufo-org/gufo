#pragma once

#include <optional>
#include <string_view>

#include "gufo/ext/gguf_reader.hpp"
#include "gufo/ext/text.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] bool ProbeFlashNext(std::string_view model_root);
[[nodiscard]] bool ProbeFlashNextGguf(const core::GgufReader& reader);
/// Whether `model_root` is an EXL3 checkpoint directory of Flash-Next whose
/// `quantization_config.bits` lies in [min_bits, max_bits).
[[nodiscard]] bool ProbeFlashNextExl3(std::string_view model_root,
                                      double min_bits, double max_bits);
/// The runner descriptor and the provider probe both use these defaults.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults();
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
FlashNextSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::qwen38_flash_next
