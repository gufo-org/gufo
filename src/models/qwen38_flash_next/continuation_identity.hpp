#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace gufo::models::qwen38_flash_next {

bool IsSha256Hex(std::string_view value) noexcept;

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view position_policy);

std::vector<std::uint8_t> QwenFlashNextProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::uint32_t decode_concurrency);

}  // namespace gufo::models::qwen38_flash_next

namespace gufo::server {
using models::qwen38_flash_next::QwenFlashNextCompatibilityIdentity;
using models::qwen38_flash_next::QwenFlashNextProcessIdentity;
}  // namespace gufo::server
