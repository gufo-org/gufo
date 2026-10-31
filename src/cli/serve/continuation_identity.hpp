#ifndef GUFO_CLI_SERVE_CONTINUATION_IDENTITY_HPP_
#define GUFO_CLI_SERVE_CONTINUATION_IDENTITY_HPP_

#include <cstdint>
#include <string_view>
#include <vector>

#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::speculative {
struct SpeculativeOptions;
}

namespace gufo::server {

// Temporary serving seam for pre-migration byte goldens. Each writer moves with
// its owning model in cards 20–22; this is not a public extension API.
inline constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";
inline constexpr std::uint32_t kQwenComponentPayloadVersion = 2;

inline constexpr std::string_view QwenStateAbi(
    bool speculative, bool fp16_attention_kv,
    bool bf16_recurrent_state) noexcept {
  if (bf16_recurrent_state) {
    if (speculative) {
      return fp16_attention_kv
                 ? "qwen-gfx1151-dflash-state-v4-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-dflash-state-v4-fp32-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-state-v3-fp16-kv-bf16-recurrent"
                             : "qwen-gfx1151-state-v3-fp32-kv-bf16-recurrent";
  }
  if (speculative) {
    return fp16_attention_kv ? "qwen-gfx1151-dflash-state-v3-fp16-kv"
                             : "qwen-gfx1151-dflash-state-v3-fp32-kv";
  }
  return fp16_attention_kv ? "qwen-gfx1151-state-v2-fp16-kv"
                           : "qwen-gfx1151-state-v2-fp32-kv";
}

bool IsSha256Hex(std::string_view value) noexcept;

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens);

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view position_policy);

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy);

std::vector<std::uint8_t> QwenProcessIdentity(const void* model,
                                              const void* draft,
                                              std::uint32_t max_context);
std::vector<std::uint8_t> QwenFlashNextProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::uint32_t decode_concurrency);

}  // namespace gufo::server
#endif  // GUFO_CLI_SERVE_CONTINUATION_IDENTITY_HPP_
