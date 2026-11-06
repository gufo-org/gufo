#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "gufo/ext/text.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace gufo::models::qwen38_flash_next {

[[nodiscard]] std::size_t FlashNextBenchContext(
    const ext::TextBenchOptions& options);

[[nodiscard]] bool ResolveFlashNextBenchYarn(
    const ext::TextBenchOptions& options, std::uint32_t native,
    std::optional<RopeScaling>* scaling, std::string* error);

}  // namespace gufo::models::qwen38_flash_next

namespace gufo::cli {
using models::qwen38_flash_next::FlashNextBenchContext;
using models::qwen38_flash_next::ResolveFlashNextBenchYarn;
}  // namespace gufo::cli
