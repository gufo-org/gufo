#include "src/cli/serve/continuation_identity.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "src/core/gguf_identity.hpp"
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/text/chat_template.hpp"
#include "src/models/deepseek_v4_flash/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
namespace gufo::server {
bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> DeepSeekCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=deepseek4\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=joyai-byte-bpe-v1\n"
           << "chat_template=" << models::deepseek_v4_flash::ChatTemplateId()
           << '\n'
           << "chat_template_reference_sha256="
           << models::deepseek_v4_flash::EncoderReferenceSha256() << '\n'
           << "state_abi=" << kDeepSeekStateAbi << '\n'
           << "payload_layout=ds4-rocm-v4-window128-continuation\n"
           // Cached frontiers also depend on the compiled numerical routes.
           << "numerics=ds4-scalar-verifier-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=deepseek4-compiled-v1\n"
           << "adapters=none\n";
  identity << "support_id=" << core::kGgufIdentityScheme << ':'
           << support_fingerprint << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n'
           << "draft_policy=dspark-cost-v5-window128\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::string_view position_policy) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next disk cache requires an artifact fingerprint");
  }
  if (has_mtp && !IsSha256Hex(mtp_fingerprint)) {
    throw std::invalid_argument(
        "Qwen3.8-Flash-Next MTP disk cache requires a draft artifact "
        "fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v"
           << models::qwen38_flash_next::Session::kSnapshotPayloadVersion
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=" << position_policy << '\n'
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen disk cache requires an artifact fingerprint");
  }
  if (speculative && !IsSha256Hex(draft_artifact_fingerprint)) {
    throw std::invalid_argument(
        "Qwen DFlash disk cache requires a draft artifact fingerprint");
  }
  const std::string_view state_abi =
      QwenStateAbi(speculative, execution_policy.UsesFp16AttentionKv(),
                   execution_policy.UsesBf16RecurrentState());
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen3.8\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256="
           << tokenization::QwenChatTemplate::OfficialTemplateSha256() << '\n'
           << "state_abi=" << state_abi << "-component-v2\n"
           << "payload_layout=qwen-component-v2\n"
           // Older learned boundaries could keep logits from an earlier
           // prefix. Their payload has no readiness tag to distinguish them.
           << "checkpoint_frontier=complete-prefix-v1\n"
           << "numerics=qwen-bf16-fp32-prefill-v1\n"
           << "rmsnorm=fused-square-sum-v1\n"
           << "prefill_attention=visible-causal-tail-v1\n"
           << "attention_split_min_context="
           << hip::detail::kSplitKDecodeAttentionMinContext << '\n'
           << "attention_split_count="
           << hip::detail::kSplitKDecodeAttentionMaxSplits << '\n'
           << "kv_storage="
           << (execution_policy.UsesFp16AttentionKv() ? "fp16" : "fp32") << '\n'
           << "recurrent_storage="
           << (execution_policy.UsesBf16RecurrentState() ? "bf16" : "fp32")
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=qwen-gguf-config-v1\n"
           << "adapters=none\n";
  if (speculative) {
    identity
        << "draft_backend=dflash2-gfx1151-v1\n"
        << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
        << draft_artifact_fingerprint << '\n'
        << "draft_state_layout=dflash-window-kv-and-frontier-v4\n"
        << "draft_policy=" << speculative::DFlashDraftPolicyName(draft_policy)
        << "-v3\n"
        << "draft_max_tokens=" << speculative_options.max_draft_tokens << '\n'
        << "draft_min_tokens=" << speculative_options.min_draft_tokens << '\n'
        << "draft_initial_tokens=" << speculative_options.initial_draft_tokens
        << '\n'
        << "draft_rolling_window=" << speculative_options.rolling_window << '\n'
        << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "draft_target_acceptance="
        << speculative_options.target_acceptance_rate << '\n'
        << "draft_adaptive="
        << (speculative_options.enable_adaptive_draft_length ? "true" : "false")
        << '\n'
        << "draft_batched_verification="
        << (speculative_options.use_batched_verification ? "true" : "false")
        << '\n'
        << "draft_target_verification=decode-equivalent-v1\n";
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenProcessIdentity(const void* model,
                                              const void* draft,
                                              std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-qwen-component-v2\n"
           << "model_instance=" << model << '\n'
           << "draft_instance=" << draft << '\n'
           << "context_tokens=" << max_context << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

std::vector<std::uint8_t> QwenFlashNextProcessIdentity(
    const void* model, std::uint32_t max_context, bool use_mtp,
    std::uint32_t max_draft_tokens, std::uint32_t decode_concurrency) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "mtp=" << use_mtp << '\n'
           << "draft_max_tokens=" << max_draft_tokens << '\n'
           << "draft_cost_concurrency=" << decode_concurrency << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

}  // namespace gufo::server
