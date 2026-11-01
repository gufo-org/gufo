#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace gufo::models::deepseek_v4_flash {

inline constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";

bool IsSha256Hex(std::string_view value) noexcept;

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens);

}  // namespace gufo::models::deepseek_v4_flash

namespace gufo::server {
using models::deepseek_v4_flash::DeepSeekCompatibilityIdentity;
using models::deepseek_v4_flash::kDeepSeekStateAbi;
}  // namespace gufo::server
