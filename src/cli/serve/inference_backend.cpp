#include "src/cli/serve/inference_backend.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/gguf_identity.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/generator.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include "src/core/speculative/speculative_verifier.hpp"
#include "src/models/deepseek_v4_flash/dspark_sampler.hpp"
#include "src/models/deepseek_v4_flash/engine.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#endif

namespace gufo::server {
namespace {

using Clock = std::chrono::steady_clock;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

#if defined(ENGINE_ENABLE_HIP)
/// Ordinary host allocations use the host budget, not HIP's device capacity.
std::size_t HostSnapshotBudgetBytes() {
  const long pages = sysconf(_SC_AVPHYS_PAGES);
  const long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return 0;
  std::uint64_t available = std::uint64_t(pages) * page_size;
  std::ifstream meminfo("/proc/meminfo");
  for (std::string line; std::getline(meminfo, line);) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream fields(line.substr(13));
      std::uint64_t kib = 0;
      if (fields >> kib)
        available = kib * 1024;
      break;
    }
  }
  // A cgroup limit can be much smaller than the host's available memory.
  // Walk parents too: a child may say "max" beneath a limited ancestor.
  std::ifstream membership("/proc/self/cgroup");
  for (std::string line; std::getline(membership, line);) {
    if (!line.starts_with("0::/"))
      continue;
    const auto relative = std::filesystem::path(line.substr(4));
    if (std::ranges::any_of(relative,
                            [](const auto& part) { return part == ".."; }))
      continue;
    const std::filesystem::path root("/sys/fs/cgroup");
    for (auto path = root / relative;; path = path.parent_path()) {
      std::ifstream limit_file(path / "memory.max");
      std::ifstream used_file(path / "memory.current");
      std::uint64_t limit = 0, used = 0;
      if ((limit_file >> limit) && (used_file >> used))
        available = std::min(available, limit > used ? limit - used : 0);
      if (path == root)
        break;
    }
    break;
  }
  return static_cast<std::size_t>(std::min<std::uint64_t>(
      available / 2, std::numeric_limits<std::size_t>::max()));
}

struct QwenImageContext final : TextPromptContext {
  std::shared_ptr<const models::qwen::vision::Prompt> prompt;
};

tokenization::ChatTemplateOptions QwenChatOptions(const ChatRequest& request) {
  auto options = tokenization::ResolveQwenChatOptions(request.reasoning,
                                                      request.add_vision_id);
  options.require_tool_call =
      request.tool_choice == ChatRequest::ToolChoice::kRequired;
  return options;
}

std::size_t StablePromptPrefix(std::span<const TextRunnerToken> tokens,
                               std::span<const TextRunnerToken> generation) {
  if (generation.empty() || tokens.size() <= generation.size() ||
      !std::ranges::equal(tokens.last(generation.size()), generation))
    return 0;
  return tokens.size() - generation.size();
}

TextPreparedPrompt PrepareQwenPrompt(
    const ChatRequest& request, const tokenization::QwenTokenizer& tokenizer,
    const std::shared_ptr<models::qwen::vision::Encoder>& encoder,
    std::uint32_t max_context) {
  const bool has_images = std::ranges::any_of(
      request.messages, [](const auto& m) { return !m.images.empty(); });
  const auto options = QwenChatOptions(request);
  auto prompt = std::make_shared<models::qwen::vision::Prompt>(
      models::qwen::vision::Prepare(
          tokenizer, request.messages,
          request.tool_choice == ChatRequest::ToolChoice::kNone
              ? std::span<const tokenization::ChatTool>{}
              : std::span<const tokenization::ChatTool>{request.tools},
          options,
          encoder && has_images ? encoder->identity() : std::string_view{},
          max_context));
  // Agent clients may discard an interrupted assistant entirely and append
  // the next user turn directly after tool results. Preserve a checkpoint
  // before the generation suffix even when reasoning itself is retained.
  std::size_t cache_prefix = 0;
  if (!options.preserve_thinking || !request.tools.empty()) {
    const auto generation = tokenizer.Encode(
        tokenization::GenerationPrompt(options.enable_thinking),
        {.add_bos = false, .add_eos = false, .parse_special_tokens = true});
    cache_prefix = StablePromptPrefix(prompt->tokens, generation);
  }
  if (prompt->images.empty())
    return {std::move(prompt->tokens), {}, cache_prefix};
  if (!encoder)
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  auto context = std::make_shared<QwenImageContext>();
  context->cache_identity = prompt->cache_identity;
  context->prompt = prompt;
  return {prompt->tokens, std::move(context), cache_prefix};
}

std::shared_ptr<const models::qwen::vision::Prompt> QwenPrompt(
    const std::shared_ptr<const TextPromptContext>& context) {
  if (!context)
    return {};
  const auto* image = dynamic_cast<const QwenImageContext*>(context.get());
  if (!image)
    throw std::invalid_argument("invalid Qwen prompt context");
  return image->prompt;
}

constexpr std::string_view kDeepSeekStateAbi =
    "deepseek-v4-flash-gfx1151-state-v4";
constexpr std::array<std::uint8_t, 8> kQwenPersistentSnapshotMagic = {
    'G', 'Q', 'W', 'R', 'U', 'N', '0', '1'};
constexpr std::uint32_t kQwenPersistentPayloadVersion = 3;
constexpr std::size_t kQwenMaxResumeTokens = 9;
constexpr std::size_t kQwenPersistentSnapshotHeaderBytes = 112;
constexpr std::uint32_t kQwenPersistentSpeculativeFlag = 1U << 0U;
constexpr std::uint32_t kQwenPersistentPendingTokenFlag = 1U << 1U;

constexpr std::string_view QwenStateAbi(bool speculative,
                                        bool fp16_attention_kv,
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

bool DiskCacheEnabled(const TextDiskCacheConfig& config) noexcept {
  return !config.directory.empty();
}

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
    std::uint32_t decode_concurrency) {
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
           << "position_policy=absolute-v1\n"
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
    const speculative::SpeculativeOptions& speculative_options) {
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
           << "state_abi=" << state_abi << '\n'
           << "payload_layout=qwen-gfx1151-live-prefix-v3\n"
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
        << "draft_policy="
        << speculative::DFlashDraftPolicyName(speculative_options.dflash_policy)
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

template<typename T>
  requires(std::is_unsigned_v<T>)
void PutLittleEndian(std::span<std::uint8_t> destination, std::size_t offset,
                     T value) {
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    throw std::length_error("Qwen persistent snapshot header is truncated");
  }
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    destination[offset + byte] =
        static_cast<std::uint8_t>(value >> (byte * 8U));
  }
}

template<typename T>
  requires(std::is_unsigned_v<T>)
T GetLittleEndian(std::span<const std::uint8_t> source, std::size_t offset) {
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    throw std::invalid_argument("Qwen persistent snapshot header is truncated");
  }
  T value = 0;
  for (std::size_t byte = 0; byte < sizeof(T); ++byte) {
    value |= static_cast<T>(source[offset + byte]) << (byte * 8U);
  }
  return value;
}

std::size_t CheckedPersistentAdd(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error("Qwen persistent snapshot size overflows");
  }
  return left + right;
}

std::size_t PersistentSizeFromU64(std::uint64_t value) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("Qwen persistent snapshot size overflows");
  }
  return static_cast<std::size_t>(value);
}

bool IsQwenStopToken(const tokenization::QwenTokenizer& tokenizer,
                     TextRunnerToken token) noexcept {
  return token == tokenizer.GetEosTokenId() ||
         token == tokenization::kDefaultQwenEndoftextId || token == 248044U ||
         token == 248046U;
}

class QwenTextRunnerState final : public TextRunnerState {
public:
  QwenTextRunnerState(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model,
      speculative::SpeculativeOptions speculative_options,
      hip::QwenExecutionPolicy execution_policy)
      : model_(std::move(model)) {
    resume_tokens_.reserve(kQwenMaxResumeTokens);
    rollback_tokens_.reserve(kQwenMaxResumeTokens);
    std::string error;
    executor_ = hip::QwenGpuExecutor::Create(model_, &error, max_context,
                                             execution_policy);
    if (executor_ == nullptr) {
      throw std::runtime_error("Failed to create GPU session: " + error);
    }
    if (dflash_model != nullptr) {
      auto draft_backend = hip::QwenDFlashGpuDraftBackend::Create(
          std::move(dflash_model),
          hip::QwenDFlashGpuDraftConfig{
              .max_context = max_context,
              .max_draft_tokens = speculative_options.max_draft_tokens,
              .policy = speculative_options.dflash_policy,
          },
          &error);
      if (draft_backend == nullptr) {
        throw std::runtime_error("Failed to create DFlash session: " + error);
      }
      draft_backend_ = draft_backend.get();
      verifier_ = std::make_unique<speculative::SpeculativeVerifier>(
          *executor_, std::move(draft_backend), speculative_options);
    }
  }

  void Invalidate() noexcept override {
    if (verifier_ != nullptr) {
      verifier_->Reset();
    }
    executor_->Reset();
    sequence_.clear();
    position_ = 0;
    frontier_.reset();
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
    rollback_tokens_.clear();
    rollback_logits_.clear();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    try {
      auto usage = executor_->GetMemoryUsage();
      if (draft_backend_ != nullptr) {
        const auto draft = draft_backend_->GetMemoryUsage();
        usage.request_state_bytes += draft.request_state_bytes;
        usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      }
      return {
          .per_request_state_bytes = usage.request_state_bytes,
          .temporary_scratch_bytes = usage.temporary_scratch_bytes,
      };
    } catch (...) {
      return {};
    }
  }

  [[nodiscard]] bool speculative() const noexcept {
    return verifier_ != nullptr;
  }

  void PrimeSpeculative(std::span<const TextRunnerToken> prompt,
                        bool decode_ready) {
    if (verifier_ == nullptr) {
      throw std::logic_error("Qwen state has no speculative verifier");
    }
    frontier_ = verifier_->Prime(prompt, decode_ready);
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    sequence_.assign(prompt.begin(), prompt.end());
    position_ = prompt.size();
    frontier_published_ = false;
    resume_tokens_.clear();
    rollback_position_.reset();
  }

  void PreparePrefixReuse(std::span<const TextRunnerToken> prefix) {
    if (position_ != prefix.size() || !frontier_.has_value()) {
      throw std::logic_error(
          "Qwen retained prefix does not match its target state");
    }
    if (verifier_ != nullptr) {
      verifier_->BeginRequest();
      sequence_.assign(prefix.begin(), prefix.end());
      // Verification's frontier may be a sampled draw. A new request must
      // select from its distribution again, including when it switches to
      // greedy sampling or omits the previously published pending token.
      if (!resume_tokens_.empty() && !frontier_logits_.empty()) {
        sampling::SamplerState greedy;
        frontier_ = greedy.Sample(frontier_logits_);
      }
    }
    frontier_published_ = false;
    rollback_position_.reset();
  }

  bool ResumeGeneratedToken(TextRunnerToken token, bool decode_ready) {
    if (resume_tokens_.empty())
      return false;
    if (resume_tokens_.front() != token) {
      resume_tokens_.clear();
      return false;
    }
    resume_tokens_.erase(resume_tokens_.begin());
    // This token was already published by the previous request. Consume it
    // with decode arithmetic, exactly as AR did, before chunking the new
    // prompt suffix. Including it in a prefill chunk changes recurrent state.
    const auto position = static_cast<std::uint32_t>(position_);
    frontier_ = verifier_ != nullptr
                    ? verifier_->AdvanceCommittedToken(token, position)
                    : executor_->ForwardToken(token, position);
    if (verifier_ != nullptr)
      sequence_.push_back(token);
    ++position_;
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = executor_->CopyLastLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    return true;
  }

  void ExtendSpeculative(std::span<const TextRunnerToken> suffix,
                         bool decode_ready) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative prefix is not initialized");
    }
    if (sequence_.size() != position_) {
      throw std::logic_error(
          "Qwen speculative sequence does not match retained position");
    }
    frontier_ = verifier_->ExtendPrompt(
        suffix, static_cast<std::uint32_t>(position_), decode_ready);
    sequence_.insert(sequence_.end(), suffix.begin(), suffix.end());
    position_ += suffix.size();
    frontier_logits_.clear();
    if (decode_ready) {
      const auto logits = verifier_->CopyLastTargetLogits();
      frontier_logits_.assign(logits.begin(), logits.end());
    }
    frontier_published_ = false;
  }

  [[nodiscard]] TextRunnerToken SelectFrontier(
      sampling::SamplerState& sampler) const {
    if (!frontier_.has_value()) {
      throw std::logic_error("Qwen state has no next-token frontier");
    }
    if (sampler.config().can_use_unmodified_argmax()) {
      return *frontier_;
    }
    if (frontier_logits_.empty()) {
      return executor_->SampleLastLogits(sampler);
    }
    return executor_->SampleCachedLogits(frontier_logits_, sampler);
  }

  [[nodiscard]] bool PrepareSpeculativeDecode(std::size_t max_tokens,
                                              sampling::SamplerState& sampler,
                                              TextDecodeStep& result) {
    if (verifier_ == nullptr || !frontier_.has_value()) {
      throw std::logic_error("Qwen speculative state has no frontier");
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen speculative decode budget must be at least one token");
    }
    resume_tokens_.clear();
    if (!frontier_published_) {
      if (!sampler.config().can_use_unmodified_argmax())
        frontier_ = SelectFrontier(sampler);
      if (!AppendSpeculativeSelection(*frontier_, sampler, result))
        return false;
      frontier_published_ = true;
    }
    return result.selections.size() < max_tokens;
  }

  [[nodiscard]] speculative::SpeculativeVerifier::StepRequest
  VerificationRequest(std::size_t remaining, sampling::SamplerState& sampler) {
    rollback_position_ = position_;
    rollback_tokens_.assign(1, *frontier_);
    rollback_logits_ = std::move(frontier_logits_);
    return {*verifier_,
            sequence_,
            static_cast<std::uint32_t>(position_),
            *frontier_,
            model_->GetTokenizer().GetEosTokenId(),
            static_cast<std::uint32_t>(std::min<std::size_t>(
                remaining, std::numeric_limits<std::uint32_t>::max())),
            sampler};
  }

  void FinishSpeculativeDecode(
      speculative::SpeculativeVerifier::StepResult verification,
      sampling::SamplerState& sampler, TextDecodeStep& result) {
    result.draft_tokens = verification.draft_count;
    result.draft_accepted_tokens = verification.accepted_count;
    result.execution_plan = {
        .kind = verification.physical_width > 1
                    ? TextExecutionPlanKind::kBatched
                    : TextExecutionPlanKind::kSerial,
        .physical_width = verification.physical_width,
    };
    for (const TextRunnerToken token : verification.emitted_tokens) {
      // Each prediction consumes one input, including a terminal prediction.
      // EOS itself remains the unconsumed frontier and is not published.
      ++position_;
      if (!AppendSpeculativeSelection(token, sampler, result))
        break;
      rollback_tokens_.push_back(token);
    }
    frontier_ = verification.next_token;
    frontier_logits_ = std::move(verification.next_token_logits);
    frontier_published_ = !result.stop;
    set_pending_token(frontier_published_ ? frontier_ : std::nullopt);
    if (verification.draft_count == 0)
      rollback_position_.reset();
  }

  void PrepareCancellation() {
    if (!rollback_position_)
      return;
    if (rollback_tokens_.empty() ||
        rollback_tokens_.size() > kQwenMaxResumeTokens ||
        rollback_logits_.empty())
      throw std::logic_error(
          "Qwen cancelled verification has no exact frontier");
    // Verification already saved the target state. Its target features are
    // still pending in the draft, so rewinding needs no new per-step copy.
    draft_backend_->DiscardPendingTargetContext(
        static_cast<std::uint32_t>(*rollback_position_));
    executor_->RestoreState();
    position_ = *rollback_position_;
    sequence_.resize(position_);
    frontier_logits_ = std::move(rollback_logits_);
    sampling::SamplerState greedy;
    frontier_ = greedy.Sample(frontier_logits_);
    frontier_published_ = false;
    resume_tokens_ = std::move(rollback_tokens_);
    rollback_position_.reset();
  }

  [[nodiscard]] TextDecodeStep DecodeSpeculative(
      std::size_t max_tokens, sampling::SamplerState& sampler) {
    TextDecodeStep result;
    sampling::SamplerState working_sampler = sampler;
    if (PrepareSpeculativeDecode(max_tokens, working_sampler, result)) {
      const auto request = VerificationRequest(
          max_tokens - result.selections.size(), working_sampler);
      auto verification = verifier_->VerifyStep(
          sequence_, request.position, request.current_token, request.eos_id,
          request.max_emitted_tokens, working_sampler);
      FinishSpeculativeDecode(std::move(verification), working_sampler, result);
    }
    sampler.SetRngState(working_sampler.rng_state());
    return result;
  }

  [[nodiscard]] hip::QwenGpuExecutor& executor() const { return *executor_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::optional<TextRunnerToken>& frontier()
      const noexcept {
    return frontier_;
  }
  void set_frontier(TextRunnerToken frontier) noexcept {
    frontier_ = frontier;
    frontier_logits_.clear();
    frontier_published_ = false;
    resume_tokens_.clear();
  }
  [[nodiscard]] std::vector<float> CopyFrontierLogits() const {
    if (!frontier_logits_.empty()) {
      return frontier_logits_;
    }
    const auto logits = executor_->CopyLastLogits();
    return {logits.begin(), logits.end()};
  }
  void RestoreFrontierLogits(std::vector<float> logits) {
    frontier_logits_ = std::move(logits);
  }
  [[nodiscard]] const std::vector<TextRunnerToken>& pending_tokens()
      const noexcept {
    return resume_tokens_;
  }
  void set_pending_token(std::optional<TextRunnerToken> token) {
    resume_tokens_.clear();
    if (token)
      resume_tokens_.push_back(*token);
  }
  void RestorePendingTokens(std::span<const TextRunnerToken> tokens) {
    resume_tokens_.assign(tokens.begin(), tokens.end());
    rollback_position_.reset();
  }
  [[nodiscard]] std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
  SaveVerifierSnapshot() const {
    return verifier_ != nullptr ? verifier_->Snapshot() : nullptr;
  }
  [[nodiscard]] std::size_t SnapshotPayloadBytes() const {
    std::size_t bytes = executor_->SnapshotPayloadBytes(position_);
    const auto checked_add = [&bytes](std::size_t value) {
      if (value > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error("Qwen snapshot size overflows");
      }
      bytes += value;
    };
    const std::size_t logits_count = frontier_logits_.empty()
                                         ? model_->GetConfig().vocab_size
                                         : frontier_logits_.size();
    if (logits_count >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::overflow_error("Qwen snapshot size overflows");
    }
    checked_add(logits_count * sizeof(float));
    checked_add(resume_tokens_.size() * sizeof(TextRunnerToken));
    checked_add(executor_->VisionLayout().images.size() *
                sizeof(models::qwen::vision::ImageGrid));
    if (verifier_ != nullptr) {
      checked_add(verifier_->SnapshotPayloadBytes());
    }
    return bytes;
  }
  void RestoreVerifierSnapshot(
      const speculative::SpeculativeVerifierSnapshot& snapshot) {
    if (verifier_ == nullptr) {
      throw std::invalid_argument(
          "cannot restore speculative state into a plain Qwen session");
    }
    verifier_->RestoreSnapshot(snapshot);
  }
  void RestoreVerifierPersistentSnapshot(
      std::span<const std::uint8_t> payload) {
    if (verifier_ == nullptr) {
      throw std::invalid_argument(
          "cannot restore speculative persistent state into a plain Qwen "
          "session");
    }
    verifier_->RestorePersistentSnapshot(payload);
  }

private:
  bool AppendSpeculativeSelection(TextRunnerToken token,
                                  sampling::SamplerState& sampler,
                                  TextDecodeStep& result) {
    if (IsQwenStopToken(model_->GetTokenizer(), token)) {
      result.stop = true;
      return false;
    }
    result.selections.push_back({
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
    });
    sequence_.push_back(token);
    set_pending_token(token);
    sampler.Accept(token);
    return true;
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::unique_ptr<hip::QwenGpuExecutor> executor_;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  // Owned by verifier_; used only to account the session's draft allocations.
  hip::QwenDFlashGpuDraftBackend* draft_backend_{nullptr};
  std::vector<TextRunnerToken> sequence_;
  std::size_t position_{0};
  std::optional<TextRunnerToken> frontier_;
  std::vector<float> frontier_logits_;
  bool frontier_published_{false};
  std::vector<TextRunnerToken> resume_tokens_;
  std::optional<std::size_t> rollback_position_;
  std::vector<TextRunnerToken> rollback_tokens_;
  std::vector<float> rollback_logits_;
};

class QwenTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenTextRunnerSnapshot(
      std::shared_ptr<const hip::QwenGpuModel> model,
      std::unique_ptr<hip::QwenGpuSnapshot> snapshot, std::size_t position,
      std::optional<TextRunnerToken> frontier,
      std::vector<float> frontier_logits,
      std::vector<TextRunnerToken> pending_tokens,
      std::unique_ptr<speculative::SpeculativeVerifierSnapshot>
          verifier_snapshot)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position),
        frontier(frontier),
        frontier_logits(std::move(frontier_logits)),
        pending_tokens(std::move(pending_tokens)),
        verifier_snapshot(std::move(verifier_snapshot)) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return (snapshot != nullptr ? snapshot->PayloadBytes() : 0) +
           frontier_logits.size() * sizeof(float) +
           pending_tokens.size() * sizeof(TextRunnerToken) +
           (verifier_snapshot != nullptr ? verifier_snapshot->PayloadBytes()
                                         : 0);
  }

  std::shared_ptr<const hip::QwenGpuModel> model;
  std::unique_ptr<hip::QwenGpuSnapshot> snapshot;
  std::size_t position;
  std::optional<TextRunnerToken> frontier;
  std::vector<float> frontier_logits;
  std::vector<TextRunnerToken> pending_tokens;
  std::unique_ptr<speculative::SpeculativeVerifierSnapshot> verifier_snapshot;
};

QwenTextRunnerState& RequireQwenState(TextRunnerState& state) {
  auto* qwen = dynamic_cast<QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

const QwenTextRunnerState& RequireQwenState(const TextRunnerState& state) {
  const auto* qwen = dynamic_cast<const QwenTextRunnerState*>(&state);
  if (qwen == nullptr) {
    throw std::logic_error("text runner state is not Qwen");
  }
  return *qwen;
}

class QwenTextRunner final : public TextModelRunner {
public:
  QwenTextRunner(
      std::shared_ptr<const hip::QwenGpuModel> model, std::uint32_t max_context,
      std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model = nullptr,
      speculative::SpeculativeOptions speculative_options = {},
      std::string artifact_fingerprint = {},
      std::string draft_artifact_fingerprint = {})
      : model_(std::move(model)),
        dflash_model_(std::move(dflash_model)),
        max_context_(max_context),
        speculative_options_(speculative_options),
        execution_policy_(hip::QwenExecutionPolicy::Production()) {
    if (!artifact_fingerprint.empty()) {
      const bool speculative = dflash_model_ != nullptr;
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenCompatibilityIdentity(
              artifact_fingerprint, draft_artifact_fingerprint, max_context_,
              execution_policy_, speculative, speculative_options_),
          .payload_version = kQwenPersistentPayloadVersion,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool speculative_enabled = dflash_model_ != nullptr;
    return {
        .model_id = model_->GetConfig().model_name,
        .state_abi = std::string(QwenStateAbi(
            speculative_enabled, execution_policy_.UsesFp16AttentionKv(),
            execution_policy_.UsesBf16RecurrentState())),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = !speculative_enabled,
                .incremental_text_is_exact = true,
                .multi_token_decode = speculative_enabled,
                .batched_multi_token_decode =
                    speculative_enabled && MaximumDecodeBatchWidth() > 1,
                .batched_multi_token_decode_max_width =
                    speculative_enabled ? MaximumDecodeBatchWidth() : 0,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    auto usage = hip::QwenGpuExecutor::EstimateMemoryUsage(
        model_->GetConfig(), max_context_, execution_policy_);
    std::size_t resident_weights = model_->GetResidentBytes();
    if (dflash_model_ != nullptr) {
      const auto draft = hip::QwenDFlashGpuExecutor::EstimateMemoryUsage(
          *dflash_model_, max_context_, MaximumDecodeBatchWidth());
      usage.request_state_bytes += draft.request_state_bytes;
      usage.temporary_scratch_bytes += draft.temporary_scratch_bytes;
      resident_weights += dflash_model_->GetPackedWeightBytes();
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = resident_weights,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = usage.request_state_bytes,
        .temporary_scratch_bytes = usage.temporary_scratch_bytes,
        .retained_snapshot_capacity_bytes = capacity,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    // Report actual session widths, including C6. Speculation reserves up to
    // eight token rows per session in the coordinator's existing scratch.
    for (std::size_t width = 2; width <= MaximumDecodeBatchWidth(); ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->GetTokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->GetTokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->GetTokenizer(),
                             model_->VisionEncoder(), max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenState(state).executor().ConfigureVision(QwenPrompt(context),
                                                       model_->VisionEncoder());
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->GetTokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenTextRunnerState>(
        model_, max_context_, dflash_model_, speculative_options_,
        execution_policy_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& qwen = RequireQwenState(state);
    qwen.PreparePrefixReuse(prefix);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qwen = RequireQwenState(state);
    if (offset != qwen.position()) {
      throw std::logic_error(
          "Qwen prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("Qwen prefill has no remaining input");
    }
    max_input_tokens = std::min<std::size_t>(
        max_input_tokens, qwen.executor().GetMaxPromptBatch());
    if (max_input_tokens != 0 && offset != 0 &&
        qwen.ResumeGeneratedToken(prompt[offset],
                                  offset + 1 == prompt.size())) {
      return {.consumed_tokens = 1,
              .decode_ready = offset + 1 == prompt.size()};
    }
    if (qwen.speculative()) {
      if (offset == 0) {
        const auto consumed = std::min(max_input_tokens, prompt.size());
        if (consumed == 0) {
          throw std::logic_error(
              "Qwen prefill requires a nonzero token budget");
        }
        qwen.PrimeSpeculative(prompt.first(consumed),
                              consumed == prompt.size());
        return {.consumed_tokens = consumed,
                .decode_ready = consumed == prompt.size()};
      }
      const std::size_t consumed =
          std::min(max_input_tokens, prompt.size() - offset);
      if (consumed == 0) {
        throw std::logic_error(
            "Qwen DFlash retained prefix has no suffix to prefill");
      }
      qwen.ExtendSpeculative(prompt.subspan(offset, consumed),
                             offset + consumed == prompt.size());
      return {
          .consumed_tokens = consumed,
          .decode_ready = offset + consumed == prompt.size(),
      };
    }

    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    const bool decode_ready = offset + consumed == prompt.size();
    const auto frontier = qwen.executor().ForwardPromptBatch(
        prompt.subspan(offset, consumed), static_cast<std::uint32_t>(offset),
        decode_ready);
    qwen.set_position(offset + consumed);
    if (decode_ready) {
      qwen.set_frontier(frontier);
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = decode_ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    const TextRunnerToken token = qwen.SelectFrontier(sampler);
    if (IsQwenStopToken(model_->GetTokenizer(), token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    // Keep the selected token separate from the argmax frontier: sampled
    // cancellation and prompt snapshot restore must not replace that argmax.
    qwen.set_pending_token(token);
    return {
        .stop = false,
        .token = token,
        .piece = std::string(model_->GetTokenizer().DecodeToken(token)),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& qwen = RequireQwenState(state);
    if (qwen.speculative()) {
      throw std::logic_error(
          "Qwen DFlash decoding requires a multi-token decode step");
    }
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen decode state has no retained frontier");
    }
    const auto frontier = qwen.executor().ForwardToken(
        token, static_cast<std::uint32_t>(qwen.position()));
    qwen.set_position(qwen.position() + 1);
    qwen.set_frontier(frontier);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& qwen = RequireQwenState(state);
    if (!qwen.speculative()) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    return qwen.DecodeSpeculative(max_tokens, sampler);
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (dflash_model_ == nullptr || decodes.size() < 2 ||
        decodes.size() > MaximumDecodeBatchWidth()) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    std::vector<TextDecodeStep> steps(decodes.size());
    std::vector<QwenTextRunnerState*> states;
    std::vector<sampling::SamplerState> samplers;
    std::vector<speculative::SpeculativeVerifier::StepRequest> requests;
    std::vector<std::size_t> indices;
    states.reserve(decodes.size());
    samplers.reserve(decodes.size());
    requests.reserve(decodes.size());
    indices.reserve(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      const auto& decode = decodes[index];
      auto& state = RequireQwenState(decode.state.get());
      states.push_back(&state);
      auto& sampler = samplers.emplace_back(decode.sampler.get());
      if (state.PrepareSpeculativeDecode(decode.max_tokens, sampler,
                                         steps[index])) {
        requests.push_back(state.VerificationRequest(
            decode.max_tokens - steps[index].selections.size(), sampler));
        indices.push_back(index);
      }
    }
    auto verified = speculative::SpeculativeVerifier::VerifyBatch(requests);
    for (std::size_t item = 0; item < indices.size(); ++item) {
      const std::size_t index = indices[item];
      states[index]->FinishSpeculativeDecode(std::move(verified[item]),
                                             samplers[index], steps[index]);
    }
    for (std::size_t index = 0; index < decodes.size(); ++index)
      decodes[index].sampler.get().SetRngState(samplers[index].rng_state());
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (dflash_model_ != nullptr) {
      throw std::logic_error(
          "Qwen DFlash sessions do not support batch advance");
    }
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "Qwen batched decode requires two to eight sessions");
    }

    std::vector<hip::QwenGpuBatchItem> items;
    std::vector<QwenTextRunnerState*> states;
    items.reserve(advances.size());
    states.reserve(advances.size());
    for (const auto& advance : advances) {
      auto& qwen = RequireQwenState(advance.state.get());
      if (!qwen.frontier().has_value()) {
        throw std::logic_error("Qwen batched state has no retained frontier");
      }
      states.push_back(&qwen);
      items.push_back({
          .executor = &qwen.executor(),
          .token_id = advance.token,
          .position = static_cast<std::uint32_t>(qwen.position()),
      });
    }

    const auto frontiers = hip::QwenGpuExecutor::ForwardTokenBatch(items);
    if (frontiers.size() != states.size()) {
      throw std::runtime_error(
          "Qwen batched decode returned an invalid frontier count");
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
      states[index]->set_position(states[index]->position() + 1);
      states[index]->set_frontier(frontiers[index]);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).position();
  }
  void PrepareCancellation(TextRunnerState& state) const override {
    RequireQwenState(state).PrepareCancellation();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return RequireQwenState(state).SnapshotPayloadBytes();
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qwen = RequireQwenState(state);
    if (!qwen.frontier().has_value()) {
      throw std::logic_error("Qwen state has no exact frontier to snapshot");
    }
    return std::make_unique<QwenTextRunnerSnapshot>(
        model_,
        qwen.executor().SaveSnapshot(
            static_cast<std::uint32_t>(qwen.position())),
        qwen.position(), qwen.frontier(), qwen.CopyFrontierLogits(),
        qwen.pending_tokens(), qwen.SaveVerifierSnapshot());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    if (qwen_snapshot == nullptr ||
        qwen_snapshot->model.get() != model_.get() ||
        qwen_snapshot->snapshot == nullptr ||
        !qwen_snapshot->frontier.has_value()) {
      throw std::invalid_argument(
          "Qwen snapshot does not belong to this model");
    }
    auto& restored = RequireQwenState(state);
    if (restored.speculative() !=
        (qwen_snapshot->verifier_snapshot != nullptr)) {
      throw std::invalid_argument(
          "Qwen snapshot speculative mode does not match the destination");
    }
    restored.executor().RestoreSnapshot(*qwen_snapshot->snapshot);
    restored.set_position(qwen_snapshot->position);
    restored.set_frontier(*qwen_snapshot->frontier);
    restored.RestoreFrontierLogits(qwen_snapshot->frontier_logits);
    restored.RestorePendingTokens(qwen_snapshot->pending_tokens);
    if (qwen_snapshot->verifier_snapshot != nullptr) {
      restored.RestoreVerifierSnapshot(*qwen_snapshot->verifier_snapshot);
    }
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    const bool speculative = dflash_model_ != nullptr;
    if (qwen_snapshot == nullptr ||
        qwen_snapshot->model.get() != model_.get() ||
        qwen_snapshot->snapshot == nullptr ||
        !qwen_snapshot->frontier.has_value() ||
        (qwen_snapshot->verifier_snapshot != nullptr) != speculative ||
        qwen_snapshot->position != qwen_snapshot->snapshot->ValidContext() ||
        qwen_snapshot->pending_tokens.size() > kQwenMaxResumeTokens ||
        qwen_snapshot->frontier_logits.size() !=
            model_->GetConfig().vocab_size) {
      throw std::invalid_argument(
          "Qwen persistent snapshot does not belong to this runner");
    }
    const std::size_t logits_bytes = CheckedPersistentAdd(
        0, qwen_snapshot->frontier_logits.size() * sizeof(float));
    const std::size_t verifier_payload_bytes =
        qwen_snapshot->verifier_snapshot != nullptr
            ? qwen_snapshot->verifier_snapshot->PersistentPayloadBytes()
            : 0;
    return CheckedPersistentAdd(
        CheckedPersistentAdd(
            CheckedPersistentAdd(
                kQwenPersistentSnapshotHeaderBytes,
                qwen_snapshot->snapshot->CompactPayloadBytes()),
            logits_bytes),
        verifier_payload_bytes);
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* qwen_snapshot =
        dynamic_cast<const QwenTextRunnerSnapshot*>(&snapshot);
    const std::size_t expected_bytes = PersistentSnapshotPayloadBytes(snapshot);
    if (qwen_snapshot == nullptr || destination.size() != expected_bytes) {
      throw std::invalid_argument(
          "Qwen persistent snapshot destination size is invalid");
    }
    const std::size_t gpu_payload_bytes =
        qwen_snapshot->snapshot->CompactPayloadBytes();
    const std::size_t logits_bytes =
        qwen_snapshot->frontier_logits.size() * sizeof(float);
    const std::size_t verifier_payload_bytes =
        qwen_snapshot->verifier_snapshot != nullptr
            ? qwen_snapshot->verifier_snapshot->PersistentPayloadBytes()
            : 0;
    std::fill(destination.begin(), destination.end(), std::uint8_t{0});
    std::copy(kQwenPersistentSnapshotMagic.begin(),
              kQwenPersistentSnapshotMagic.end(), destination.begin());
    PutLittleEndian<std::uint32_t>(destination, 8,
                                   kQwenPersistentPayloadVersion);
    PutLittleEndian<std::uint32_t>(
        destination, 12,
        static_cast<std::uint32_t>(kQwenPersistentSnapshotHeaderBytes));
    PutLittleEndian<std::uint64_t>(
        destination, 16, static_cast<std::uint64_t>(qwen_snapshot->position));
    PutLittleEndian<std::uint32_t>(destination, 24, *qwen_snapshot->frontier);
    PutLittleEndian<std::uint32_t>(destination, 28,
                                   (qwen_snapshot->verifier_snapshot != nullptr
                                        ? kQwenPersistentSpeculativeFlag
                                        : 0U) |
                                       (!qwen_snapshot->pending_tokens.empty()
                                            ? kQwenPersistentPendingTokenFlag
                                            : 0U));
    PutLittleEndian<std::uint64_t>(
        destination, 32,
        static_cast<std::uint64_t>(qwen_snapshot->frontier_logits.size()));
    PutLittleEndian<std::uint64_t>(
        destination, 40, static_cast<std::uint64_t>(gpu_payload_bytes));
    PutLittleEndian<std::uint64_t>(destination, 48,
                                   static_cast<std::uint64_t>(expected_bytes));
    PutLittleEndian<std::uint64_t>(
        destination, 56, static_cast<std::uint64_t>(verifier_payload_bytes));
    PutLittleEndian<std::uint32_t>(
        destination, 64,
        static_cast<std::uint32_t>(qwen_snapshot->pending_tokens.size()));
    for (std::size_t i = 0; i < qwen_snapshot->pending_tokens.size(); ++i)
      PutLittleEndian<std::uint32_t>(destination,
                                     72 + i * sizeof(std::uint32_t),
                                     qwen_snapshot->pending_tokens[i]);
    const std::size_t gpu_offset = kQwenPersistentSnapshotHeaderBytes;
    const std::size_t logits_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t verifier_offset =
        CheckedPersistentAdd(logits_offset, logits_bytes);
    const std::size_t written = qwen_snapshot->snapshot->SerializeCompact(
        destination.subspan(gpu_offset, gpu_payload_bytes));
    if (written != gpu_payload_bytes) {
      throw std::runtime_error(
          "Qwen compact snapshot serializer returned the wrong byte count");
    }
    std::memcpy(destination.data() + logits_offset,
                qwen_snapshot->frontier_logits.data(), logits_bytes);
    if (qwen_snapshot->verifier_snapshot != nullptr) {
      const std::size_t verifier_written =
          qwen_snapshot->verifier_snapshot->SerializePersistent(
              destination.subspan(verifier_offset, verifier_payload_bytes));
      if (verifier_written != verifier_payload_bytes) {
        throw std::runtime_error(
            "Qwen verifier serializer returned the wrong byte count");
      }
    }
    return destination.size();
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireQwenState(state);
    if (payload.size() < kQwenPersistentSnapshotHeaderBytes ||
        !std::equal(kQwenPersistentSnapshotMagic.begin(),
                    kQwenPersistentSnapshotMagic.end(), payload.begin()) ||
        GetLittleEndian<std::uint32_t>(payload, 8) !=
            kQwenPersistentPayloadVersion ||
        GetLittleEndian<std::uint32_t>(payload, 12) !=
            kQwenPersistentSnapshotHeaderBytes) {
      throw std::invalid_argument("Qwen persistent snapshot header is invalid");
    }
    const std::size_t position =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 16));
    const TextRunnerToken frontier =
        GetLittleEndian<std::uint32_t>(payload, 24);
    const std::uint32_t flags = GetLittleEndian<std::uint32_t>(payload, 28);
    const std::size_t logits_count =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 32));
    const std::size_t gpu_payload_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 40));
    const std::size_t total_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 48));
    const std::size_t verifier_payload_bytes =
        PersistentSizeFromU64(GetLittleEndian<std::uint64_t>(payload, 56));
    const bool speculative = (flags & kQwenPersistentSpeculativeFlag) != 0;
    const bool has_pending_token =
        (flags & kQwenPersistentPendingTokenFlag) != 0;
    const std::uint32_t pending_count =
        GetLittleEndian<std::uint32_t>(payload, 64);
    if (position == 0 || position > max_context_ ||
        position > std::numeric_limits<std::uint32_t>::max() ||
        (flags & ~(kQwenPersistentSpeculativeFlag |
                   kQwenPersistentPendingTokenFlag)) != 0 ||
        has_pending_token != (pending_count != 0) ||
        pending_count > kQwenMaxResumeTokens ||
        GetLittleEndian<std::uint32_t>(payload, 68) != 0 ||
        GetLittleEndian<std::uint32_t>(payload, 108) != 0 ||
        speculative != restored.speculative() ||
        frontier >= model_->GetConfig().vocab_size ||
        logits_count != model_->GetConfig().vocab_size ||
        logits_count >
            std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        gpu_payload_bytes == 0 ||
        speculative != (verifier_payload_bytes != 0) ||
        total_bytes != payload.size()) {
      throw std::invalid_argument(
          "Qwen persistent snapshot metadata is invalid");
    }
    std::vector<TextRunnerToken> pending_tokens;
    for (std::size_t i = 0; i < kQwenMaxResumeTokens; ++i) {
      const auto token = GetLittleEndian<std::uint32_t>(
          payload, 72 + i * sizeof(std::uint32_t));
      if (i < pending_count) {
        if (token >= model_->GetConfig().vocab_size)
          throw std::invalid_argument("Qwen snapshot pending token is invalid");
        pending_tokens.push_back(token);
      } else if (token != 0) {
        throw std::invalid_argument(
            "Qwen snapshot has nonzero reserved tokens");
      }
    }
    const std::size_t logits_bytes = logits_count * sizeof(float);
    const std::size_t gpu_offset = kQwenPersistentSnapshotHeaderBytes;
    const std::size_t logits_offset =
        CheckedPersistentAdd(gpu_offset, gpu_payload_bytes);
    const std::size_t verifier_offset =
        CheckedPersistentAdd(logits_offset, logits_bytes);
    if (CheckedPersistentAdd(verifier_offset, verifier_payload_bytes) !=
        payload.size()) {
      throw std::invalid_argument(
          "Qwen persistent snapshot payload size is invalid");
    }

    std::vector<float> frontier_logits(logits_count);
    std::memcpy(frontier_logits.data(), payload.data() + logits_offset,
                logits_bytes);
    restored.executor().RestoreCompactSnapshot(
        payload.subspan(gpu_offset, gpu_payload_bytes),
        static_cast<std::uint32_t>(position));
    if (speculative) {
      restored.RestoreVerifierPersistentSnapshot(
          payload.subspan(verifier_offset, verifier_payload_bytes));
    }
    restored.set_position(position);
    restored.set_frontier(frontier);
    restored.RestoreFrontierLogits(std::move(frontier_logits));
    restored.RestorePendingTokens(pending_tokens);
  }

private:
  [[nodiscard]] std::size_t MaximumDecodeBatchWidth() const noexcept {
    const std::size_t rows_per_session = dflash_model_ != nullptr ? 8 : 1;
    return std::clamp<std::size_t>(max_context_ / rows_per_session, 1, 8);
  }

  std::shared_ptr<const hip::QwenGpuModel> model_;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model_;
  std::uint32_t max_context_;
  speculative::SpeculativeOptions speculative_options_;
  hip::QwenExecutionPolicy execution_policy_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

std::string_view ChatRoleName(tokenization::ChatRole role) {
  switch (role) {
    case tokenization::ChatRole::kSystem:
      return "system";
    case tokenization::ChatRole::kDeveloper:
      return "developer";
    case tokenization::ChatRole::kAssistant:
      return "assistant";
    case tokenization::ChatRole::kTool:
      return "tool";
    case tokenization::ChatRole::kUser:
      return "user";
  }
  return "user";
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(
      const std::shared_ptr<models::deepseek_v4_flash::Model>& model,
      std::uint32_t max_context, bool use_dspark) {
    std::string error;
    session_ = model->CreateSession(
        use_dspark ? gufo::core::SessionMode::kSpeculative
                   : gufo::core::SessionMode::kAutoregressive,
        max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create DeepSeek session: " + error);
    }
  }

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Invalidate();
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] models::deepseek_v4_flash::Session& session() const {
    return *session_;
  }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<models::deepseek_v4_flash::Session> session_;
  std::size_t position_{0};
};

class DeepSeekTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  DeepSeekTextRunnerSnapshot(
      std::shared_ptr<models::deepseek_v4_flash::Model> model,
      std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<models::deepseek_v4_flash::Model> model;
  std::unique_ptr<models::deepseek_v4_flash::SessionSnapshot> snapshot;
  std::size_t position;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

class DeepSeekTextRunner final : public TextModelRunner {
public:
  DeepSeekTextRunner(std::shared_ptr<models::deepseek_v4_flash::Model> model,
                     std::uint32_t max_context, bool use_dspark,
                     std::uint32_t max_draft_tokens,
                     std::string artifact_fingerprint = {},
                     std::string support_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_dspark_(use_dspark),
        max_draft_tokens_(std::max(max_draft_tokens, 1u)) {
    if (!artifact_fingerprint.empty()) {
      if (use_dspark_ && !IsSha256Hex(support_fingerprint)) {
        throw std::invalid_argument(
            "DSpark disk cache requires a support SHA-256 fingerprint");
      }
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = DeepSeekCompatibilityIdentity(
              artifact_fingerprint,
              use_dspark_ ? support_fingerprint : std::string{}, max_context_,
              max_draft_tokens_),
          .payload_version = DS4_SESSION_PAYLOAD_VERSION,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    const bool dspark = use_dspark_;
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kDeepSeekStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = dspark,
                .batched_multi_token_decode = dspark,
                .batched_multi_token_decode_max_width = 8u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<models::deepseek_v4_flash::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      models::deepseek_v4_flash::ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        models::deepseek_v4_flash::ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<models::deepseek_v4_flash::ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    auto tokens = DeepSeekRunnerTokens(model_->EncodeChat(
        messages, tools,
        models::deepseek_v4_flash::ChatTemplateOptions{
            .enable_thinking = request.reasoning.enabled.value_or(false),
            .reasoning_effort =
                request.reasoning.effort.value_or(ReasoningEffort::kLow),
            .preserve_thinking =
                request.reasoning.preserve_thinking.value_or(false),
            .tools_present =
                !request.tools.empty() &&
                request.tool_choice != ChatRequest::ToolChoice::kNone,
            .require_tool_call =
                request.tool_choice == ChatRequest::ToolChoice::kRequired,
        }));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return request.reasoning.enabled.value_or(false)
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // DeepSeek joins adjacent user/tool messages into one user block.
      // A new user turn after a discarded assistant can therefore also change
      // the final BPE token before the generation suffix (e.g. ">\n\n").
      // Find the exact stable frontier using the renderer/tokenizer instead
      // of assuming the suffix alone is the only changed part of the prompt.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<DeepSeekTextRunnerState>(model_, max_context_,
                                                     use_dspark_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() != prefix.size()) {
      throw std::logic_error(
          "DeepSeek reused prefix does not match checkpoint");
    }
    deepseek.session().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }

    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset,
                               deepseek.session().PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    std::string error;
    if (!deepseek.session().Sync(prefix, &error)) {
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.stop = true, .piece = {}};
    // A sampled DSpark cycle may have drawn the next token already.
    int token = deepseek.session().TakePendingDsparkToken();
    if (token < 0) {
      std::string error;
      const auto logits = deepseek.session().CopyLogits(&error);
      if (logits.empty()) {
        throw std::runtime_error("DeepSeek token selection failed: " + error);
      }
      token = static_cast<int>(sampler.Sample(logits));
    }
    if (model_->IsStopToken(token)) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->DecodeToken(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return TextDecodeSelection{.stop = true, .piece = {}};
    std::string error;
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek token preview failed: " + error);
    const auto token = sampler.Sample(logits);
    return TextDecodeSelection{
        .stop = model_->IsStopToken(static_cast<int>(token)),
        .token = token,
        .piece = model_->DecodeToken(static_cast<int>(token)),
    };
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_dspark_) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "DeepSeek DSpark decode budget must be at least one token");
    }

    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto stats_before = deepseek.session().DsparkStatistics();
    std::optional<models::deepseek_v4_flash::DsparkSamplerBridge> bridge;
    if (!sampler.config().can_use_unmodified_argmax()) {
      bridge.emplace(sampler);
    }
    std::vector<int> emitted;
    std::string error;
    if (!deepseek.session().DsparkStep(
            max_tokens, max_draft_tokens_, &emitted, &error,
            bridge ? bridge->hook() : nullptr, true)) {
      throw std::runtime_error("DeepSeek DSpark decode failed: " + error);
    }
    if (emitted.empty()) {
      throw std::runtime_error("DeepSeek DSpark decode produced no tokens");
    }
    if (bridge) {
      sampler.SetRngState(bridge->rng_state());
    }

    TextDecodeStep step;
    deepseek.set_position(deepseek.session().Position());
    step.selections.reserve(emitted.size());
    for (const int token : emitted) {
      if (model_->IsStopToken(token)) {
        step.stop = true;
        break;
      }
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->DecodeToken(token),
      });
    }
    const auto stats_after = deepseek.session().DsparkStatistics();
    step.draft_tokens =
        stats_after.support_drafted - stats_before.support_drafted;
    step.draft_accepted_tokens =
        stats_after.support_accepted - stats_before.support_accepted;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (!use_dspark_ || decodes.size() < 2 || decodes.size() > 8) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    if (std::any_of(decodes.begin(), decodes.end(), [&](const auto& item) {
          return RequireDeepSeekState(item.state.get()).position() >=
                 max_context_;
        })) {
      std::vector<TextRunnerDecode> active;
      std::vector<std::size_t> active_indices;
      std::vector<TextDecodeStep> steps(decodes.size());
      for (std::size_t i = 0; i < decodes.size(); ++i) {
        if (RequireDeepSeekState(decodes[i].state.get()).position() >=
            max_context_) {
          steps[i].stop = true;
        } else {
          active.push_back(decodes[i]);
          active_indices.push_back(i);
        }
      }
      if (!active.empty()) {
        auto active_steps = DecodeBatch(active);
        for (std::size_t i = 0; i < active.size(); ++i)
          steps[active_indices[i]] = std::move(active_steps[i]);
      }
      return steps;
    }
    // Greedy and sampled requests share one DSpark cohort; the runtime keeps
    // the greedy verifier for items without a sampler.
    std::array<models::deepseek_v4_flash::SessionDsparkBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::array<models::deepseek_v4_flash::Session::DsparkStats, 8>
        stats_before{};
    std::array<std::vector<int>, 8> emitted{};
    std::array<std::optional<models::deepseek_v4_flash::DsparkSamplerBridge>, 8>
        bridges{};
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& deepseek = RequireDeepSeekState(decodes[index].state.get());
      states[index] = &deepseek;
      stats_before[index] = deepseek.session().DsparkStatistics();
      auto& sampler = decodes[index].sampler.get();
      if (!sampler.config().can_use_unmodified_argmax()) {
        bridges[index].emplace(sampler);
      }
      items[index] = {
          .session = &deepseek.session(),
          .max_tokens = decodes[index].max_tokens,
          .max_draft_tokens = max_draft_tokens_,
          .emitted = &emitted[index],
          .sampler = bridges[index] ? bridges[index]->hook() : nullptr,
          .stop_at_eos = true,
      };
    }

    std::string error;
    if (!model_->DsparkStepBatch(
            std::span<const models::deepseek_v4_flash::SessionDsparkBatchItem>(
                items.data(), decodes.size()),
            &error)) {
      throw std::runtime_error("DeepSeek DSpark batch decode failed: " + error);
    }

    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      if (emitted[index].empty()) {
        throw std::runtime_error("DeepSeek DSpark batch produced no tokens");
      }
      if (bridges[index]) {
        decodes[index].sampler.get().SetRngState(bridges[index]->rng_state());
      }
      auto& step = steps[index];
      step.execution_plan = {
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = decodes.size(),
      };
      states[index]->set_position(states[index]->session().Position());
      step.selections.reserve(emitted[index].size());
      for (const int token : emitted[index]) {
        if (model_->IsStopToken(token)) {
          step.stop = true;
          break;
        }
        step.selections.push_back({
            .stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->DecodeToken(token),
        });
      }
      const auto stats_after = states[index]->session().DsparkStatistics();
      step.draft_tokens =
          stats_after.support_drafted - stats_before[index].support_drafted;
      step.draft_accepted_tokens =
          stats_after.support_accepted - stats_before[index].support_accepted;
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "DeepSeek batched decode requires two to eight sessions");
    }

    std::array<models::deepseek_v4_flash::SessionBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::size_t item_count = 0;
    for (const auto& advance : advances) {
      if (advance.token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      auto& deepseek = RequireDeepSeekState(advance.state.get());
      states[item_count] = &deepseek;
      items[item_count] = {
          .session = &deepseek.session(),
          .token = static_cast<int>(advance.token),
      };
      ++item_count;
    }

    std::string error;
    if (!model_->EvaluateBatch(
            std::span<const models::deepseek_v4_flash::SessionBatchItem>(
                items.data(), item_count),
            &error)) {
      throw std::runtime_error("DeepSeek batch decode failed: " + error);
    }
    for (std::size_t index = 0; index < item_count; ++index) {
      states[index]->set_position(states[index]->position() + 1);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const auto& session = RequireDeepSeekState(state).session();
    if (use_dspark_ && session.DsparkStatistics().context_tokens !=
                           static_cast<uint32_t>(session.Position())) {
      throw std::runtime_error("DSpark prefix lacks complete support state");
    }
    const std::uint64_t bytes = session.PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error("DeepSeek snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    auto snapshot = deepseek.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("DeepSeek snapshot failed: " + error);
    }
    return std::make_unique<DeepSeekTextRunnerSnapshot>(
        model_, std::move(snapshot), deepseek.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek snapshot does not belong to this model");
    }
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*deepseek_snapshot->snapshot,
                                            &error)) {
      throw std::runtime_error("DeepSeek snapshot restore failed: " + error);
    }
    restored.set_position(deepseek_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot does not belong to this model");
    }
    return deepseek_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* deepseek_snapshot =
        dynamic_cast<const DeepSeekTextRunnerSnapshot*>(&snapshot);
    if (deepseek_snapshot == nullptr ||
        deepseek_snapshot->model.get() != model_.get() ||
        deepseek_snapshot->snapshot == nullptr ||
        destination.size() != deepseek_snapshot->PayloadBytes() ||
        !deepseek_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "DeepSeek persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const DeepSeekTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireDeepSeekState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      throw std::runtime_error("DeepSeek persistent snapshot restore failed: " +
                               error);
    }
    const int position = restored.session().Position();
    if (position <= 0 || static_cast<std::uint64_t>(position) > max_context_) {
      restored.session().Invalidate();
      throw std::runtime_error(
          "DeepSeek persistent snapshot restored an invalid position");
    }
    restored.set_position(static_cast<std::size_t>(position));
  }

private:
  std::shared_ptr<models::deepseek_v4_flash::Model> model_;
  std::uint32_t max_context_;
  bool use_dspark_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};

/// Derives the content identity that keys persistent continuations for one
/// GGUF artifact. Full digests are cached by the open file metadata.
bool FingerprintArtifact(std::string_view label, const core::GgufReader& reader,
                         std::string* fingerprint, std::string* error) {
  const auto start = std::chrono::steady_clock::now();
  Logger::Info("loader", "event=load_phase phase=artifact_identity model=" +
                             std::string(label));
  try {
    *fingerprint = core::GgufIdentityHex(reader);
  } catch (const std::exception& exception) {
    SetError(error, std::string("Failed to fingerprint ") + std::string(label) +
                        " GGUF: " + exception.what());
    return false;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();
  std::ostringstream message;
  message << "Fingerprinted " << label << " artifact ("
          << core::kGgufIdentityScheme << ", " << reader.GetTensorCount()
          << " tensors) in " << std::fixed << std::setprecision(0) << elapsed_ms
          << " ms";
  Logger::Info("engine", message.str());
  return true;
}

bool FingerprintArtifactFile(std::string_view label,
                             const std::filesystem::path& path,
                             std::string* fingerprint, std::string* error) {
  std::string open_error;
  const auto reader = core::GgufReader::OpenFile(path, &open_error);
  if (reader == nullptr) {
    SetError(error, std::string("Failed to open ") + std::string(label) +
                        " GGUF for fingerprinting: " + open_error);
    return false;
  }
  return FingerprintArtifact(label, *reader, fingerprint, error);
}

#endif

#if defined(ENGINE_ENABLE_HIP)
// Each Flash-Next request owns its recurrent/KV state and snapshots. Target
// projections share batches; sampling and rollback remain request-local.
constexpr std::string_view kQwenFlashNextStateAbi =
    "qwen38-flash-next-rocm-session-v1";

using QwenFlashNextModel = models::qwen38_flash_next::Model;
using QwenFlashNextSession = models::qwen38_flash_next::Session;

std::vector<std::int32_t> QwenFlashNextEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class QwenFlashNextTextRunnerState final : public TextRunnerState {
public:
  QwenFlashNextTextRunnerState(const std::shared_ptr<QwenFlashNextModel>& model,
                               std::uint32_t max_context, bool use_mtp) {
    std::string error;
    session_ =
        model->CreateSession(use_mtp ? gufo::core::SessionMode::kSpeculative
                                     : gufo::core::SessionMode::kAutoregressive,
                             max_context, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("Failed to create Qwen3.8-Flash-Next session: " +
                               error);
    }
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    session_->Reset();
    position_ = 0;
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    session_->SetCancellationCheck(check);
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] QwenFlashNextSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
  std::unique_ptr<QwenFlashNextSession> session_;
  std::size_t position_{0};
};

class QwenFlashNextTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  QwenFlashNextTextRunnerSnapshot(
      std::shared_ptr<QwenFlashNextModel> model,
      std::unique_ptr<models::qwen38_flash_next::SessionSnapshot> snapshot,
      std::size_t position)
      : model(std::move(model)),
        snapshot(std::move(snapshot)),
        position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    if (snapshot == nullptr ||
        snapshot->SizeBytes() > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max())) {
      return 0;
    }
    return static_cast<std::size_t>(snapshot->SizeBytes());
  }

  std::shared_ptr<QwenFlashNextModel> model;
  std::unique_ptr<models::qwen38_flash_next::SessionSnapshot> snapshot;
  std::size_t position;
};

QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    TextRunnerState& state) {
  auto* qfn = dynamic_cast<QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

const QwenFlashNextTextRunnerState& RequireQwenFlashNextState(
    const TextRunnerState& state) {
  const auto* qfn = dynamic_cast<const QwenFlashNextTextRunnerState*>(&state);
  if (qfn == nullptr) {
    throw std::logic_error("text runner state is not Qwen3.8-Flash-Next");
  }
  return *qfn;
}

class QwenFlashNextTextRunner final : public TextModelRunner {
public:
  QwenFlashNextTextRunner(std::shared_ptr<QwenFlashNextModel> model,
                          std::uint32_t max_context, bool use_mtp,
                          std::uint32_t max_draft_tokens,
                          std::string artifact_fingerprint = {},
                          std::string mtp_fingerprint = {})
      : model_(std::move(model)),
        max_context_(max_context),
        use_mtp_(use_mtp),
        max_draft_tokens_(max_draft_tokens) {
    if (!artifact_fingerprint.empty()) {
      persistence_ = TextRunnerPersistenceDescriptor{
          .compatibility_identity = QwenFlashNextCompatibilityIdentity(
              artifact_fingerprint, use_mtp_ ? mtp_fingerprint : std::string{},
              use_mtp_, max_context_, max_draft_tokens_,
              model_->DecodeConcurrency()),
          .payload_version =
              models::qwen38_flash_next::Session::kSnapshotPayloadVersion,
      };
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kQwenFlashNextStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = use_mtp_,
                .batched_multi_token_decode = use_mtp_,
                .batched_multi_token_decode_max_width = use_mtp_ ? 8u : 0u,
                .prefix_reuse = true,
            },
        .persistence = persistence_,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      const auto deferred = model_->DeferredScratchBytes();
      capacity = free_bytes > deferred ? free_bytes - deferred : 0;
    }
    // Snapshots live in host memory, not in the device state pool.
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(
            use_mtp_ ? gufo::core::SessionMode::kSpeculative
                     : gufo::core::SessionMode::kAutoregressive,
            max_context_),
        // Runtime scratch is shared and already allocated at model load;
        // reserve its remaining lazy buffers once from aggregate capacity.
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back(
          {.kind = TextExecutionPlanKind::kBatched, .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return tokenization::QwenChatTemplate::RenderAndTokenize(
        model_->tokenizer(), request.messages,
        request.tool_choice == ChatRequest::ToolChoice::kNone
            ? std::span<const tokenization::ChatTool>{}
            : std::span<const tokenization::ChatTool>{request.tools},
        QwenChatOptions(request));
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    return PrepareQwenPrompt(request, model_->tokenizer(),
                             model_->VisionEncoder(), max_context_);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    RequireQwenFlashNextState(state).session().ConfigureVision(
        QwenPrompt(context));
  }

  [[nodiscard]] TextGenerationBackend::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return QwenChatOptions(request).enable_thinking
               ? TextGenerationBackend::InitialOutputState::kReasoning
               : TextGenerationBackend::InitialOutputState::kContent;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    return std::make_unique<QwenFlashNextTextRunnerState>(model_, max_context_,
                                                          use_mtp_);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() != prefix.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next reused prefix does not match checkpoint");
    }
    qfn.session().ResetDraftPolicy();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (offset != qfn.position()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error(
          "Qwen3.8-Flash-Next prefill has no remaining input");
    }
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, model_->PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = QwenFlashNextEngineTokens(prompt.first(next_position));
    std::string error;
    if (!qfn.session().Sync(prefix, &error)) {
      qfn.set_position(0);
      throw std::runtime_error("Qwen3.8-Flash-Next prefill failed: " + error);
    }
    qfn.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = qfn.session().Logits();
    if (logits.empty()) {
      throw std::runtime_error(
          "Qwen3.8-Flash-Next token selection has no logits");
    }
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}};
    }
    return {
        .stop = false,
        .token = static_cast<TextRunnerToken>(token),
        .piece = model_->TokenText(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next token ID exceeds engine range");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    if (!qfn.session().Evaluate(static_cast<std::int32_t>(token), &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next decode failed: " + error);
    }
    qfn.set_position(qfn.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (!use_mtp_ || max_tokens == 1) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next MTP decode budget must be at least one token");
    }
    auto& qfn = RequireQwenFlashNextState(state);
    if (qfn.position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto stats_before = qfn.session().Statistics();
    sampling::SamplerState working_sampler = sampler;
    QwenFlashNextSession::DecodeResult decoded;
    std::string error;
    const auto budget =
        std::min<std::size_t>(max_tokens, std::uint64_t{max_draft_tokens_} + 1);
    if (!qfn.session().DecodeStep(budget, working_sampler, &decoded, &error)) {
      throw std::runtime_error("Qwen3.8-Flash-Next MTP decode failed: " +
                               error);
    }
    // The pool accepts the returned tokens once. Publish the RNG and residual
    // draw so the next batch retains the rejection-conditioned distribution.
    sampler.CopyDrawStateFrom(working_sampler);
    TextDecodeStep step;
    step.stop = decoded.stop;
    step.selections.reserve(decoded.tokens.size());
    for (const std::int32_t token : decoded.tokens) {
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->TokenText(token),
      });
    }
    qfn.set_position(qfn.session().Position());
    const auto stats_after = qfn.session().Statistics();
    step.draft_tokens = stats_after.drafted - stats_before.drafted;
    step.draft_accepted_tokens = stats_after.accepted - stats_before.accepted;
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(advances.size());
    std::vector<QwenFlashNextSession::AdvanceRequest> requests;
    for (const auto& advance : advances) {
      if (advance.token > static_cast<TextRunnerToken>(
                              std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Flash-Next token exceeds engine range");
      }
      requests.push_back(
          {&RequireQwenFlashNextState(advance.state.get()).session(),
           static_cast<std::int32_t>(advance.token),
           &outcomes[requests.size()]});
    }
    std::string error;
    (void)QwenFlashNextSession::EvaluateBatch(requests, &error);
    for (std::size_t i = 0; i < advances.size(); ++i) {
      const auto& advance = advances[i];
      if (!outcomes[i].completed) {
        auto failure = std::make_exception_ptr(std::runtime_error(
            "Flash-Next advance failed: " + outcomes[i].error));
        if (!advance.failure)
          std::rethrow_exception(failure);
        *advance.failure = failure;
        continue;
      }
      auto& state = RequireQwenFlashNextState(advance.state.get());
      state.set_position(state.session().Position());
    }
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !use_mtp_) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    const auto count = decodes.size();
    std::vector<sampling::SamplerState> samplers;
    std::vector<QwenFlashNextSession::DecodeResult> results(count);
    std::vector<QwenFlashNextSession::BatchOutcome> outcomes(count);
    std::vector<QwenFlashNextSession::SpeculativeStats> before;
    std::vector<QwenFlashNextSession::DecodeRequest> requests;
    samplers.reserve(count);
    before.reserve(count);
    requests.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto& session =
          RequireQwenFlashNextState(decodes[i].state.get()).session();
      auto& sampler = samplers.emplace_back(decodes[i].sampler.get());
      before.push_back(session.Statistics());
      requests.push_back(
          {&session,
           std::min<std::size_t>(decodes[i].max_tokens,
                                 std::uint64_t{max_draft_tokens_} + 1),
           &sampler, &results[i], true, &outcomes[i]});
    }
    std::string error;
    (void)QwenFlashNextSession::DecodeBatch(requests, &error);
    const auto active_count = static_cast<std::size_t>(std::count_if(
        results.begin(), results.end(),
        [](const auto& result) { return !result.tokens.empty(); }));
    std::vector<TextDecodeStep> steps(count);
    for (std::size_t i = 0; i < count; ++i) {
      if (!outcomes[i].completed) {
        steps[i].failure = std::make_exception_ptr(
            std::runtime_error("Flash-Next MTP failed: " + outcomes[i].error));
        continue;
      }
      auto& state = RequireQwenFlashNextState(decodes[i].state.get());
      decodes[i].sampler.get().CopyDrawStateFrom(samplers[i]);
      state.set_position(state.session().Position());
      auto& step = steps[i];
      step.stop = results[i].stop;
      if (active_count > 1 && !results[i].tokens.empty()) {
        step.execution_plan = {.kind = TextExecutionPlanKind::kBatched,
                               .physical_width = active_count};
      }
      for (const auto token : results[i].tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      const auto stats = state.session().Statistics();
      step.draft_tokens = stats.drafted - before[i].drafted;
      step.draft_accepted_tokens = stats.accepted - before[i].accepted;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireQwenFlashNextState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    const std::uint64_t bytes =
        RequireQwenFlashNextState(state).session().SnapshotBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      throw std::overflow_error(
          "Qwen3.8-Flash-Next snapshot size is unavailable");
    }
    return static_cast<std::size_t>(bytes);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& qfn = RequireQwenFlashNextState(state);
    std::string error;
    auto snapshot = qfn.session().SaveSnapshot(&error);
    if (snapshot == nullptr) {
      throw std::runtime_error("Qwen3.8-Flash-Next snapshot failed: " + error);
    }
    return std::make_unique<QwenFlashNextTextRunnerSnapshot>(
        model_, std::move(snapshot), qfn.position());
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next snapshot does not belong to this model");
    }
    auto& restored = RequireQwenFlashNextState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(*qfn_snapshot->snapshot, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Qwen3.8-Flash-Next snapshot restore failed: " +
                               error);
    }
    restored.set_position(qfn_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next persistent snapshot does not belong to this "
          "model");
    }
    return qfn_snapshot->PayloadBytes();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* qfn_snapshot =
        dynamic_cast<const QwenFlashNextTextRunnerSnapshot*>(&snapshot);
    if (qfn_snapshot == nullptr || qfn_snapshot->model.get() != model_.get() ||
        qfn_snapshot->snapshot == nullptr ||
        destination.size() != qfn_snapshot->PayloadBytes() ||
        !qfn_snapshot->snapshot->CopyTo(destination)) {
      throw std::invalid_argument(
          "Qwen3.8-Flash-Next persistent snapshot serialization failed");
    }
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const QwenFlashNextTextRunnerSnapshot&>(snapshot)
             .snapshot->bytes());
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireQwenFlashNextState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error(
          "Qwen3.8-Flash-Next persistent snapshot restore failed: " + error);
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(
          "Qwen3.8-Flash-Next persistent snapshot restored an invalid "
          "position");
    }
    restored.set_position(position);
  }

private:
  std::shared_ptr<QwenFlashNextModel> model_;
  std::uint32_t max_context_;
  bool use_mtp_;
  std::uint32_t max_draft_tokens_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};
#endif

}  // namespace

struct InferenceBackend::Impl {
#if defined(ENGINE_ENABLE_HIP)
  struct State {
    std::shared_ptr<TextGenerationScheduler> scheduler;
    std::string model_id;
    SamplingDefaults sampling_defaults;
    ReasoningOptions reasoning_defaults;
  };

  class ScheduledGenerationRequest final : public GenerationRequest {
  public:
    ScheduledGenerationRequest(
        std::shared_ptr<const State> model_state,
        TextGenerationScheduler::Request scheduled_request)
        : state_(std::move(model_state)),
          request_(std::move(scheduled_request)) {}

    Result Wait(const TokenCallback& on_token) override {
      return request_.Wait(on_token);
    }

    void Cancel() noexcept override { request_.Cancel(); }

  private:
    std::shared_ptr<const State> state_;
    TextGenerationScheduler::Request request_;
  };

  [[nodiscard]] std::shared_ptr<const State> Snapshot() const {
    const std::lock_guard<std::mutex> lock(state_mutex);
    return state;
  }

  Result GenerateScheduled(
      std::shared_ptr<const State> current,
      std::vector<TextRunnerToken> prompt_tokens,
      Clock::time_point request_start, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled, const TokenCallback& on_token,
      std::string client_id,
      std::shared_ptr<const TextPromptContext> context = {},
      bool cache_prompt = true, std::size_t cache_prefix_tokens = 0,
      const std::vector<std::string>& stop_sequences = {}) const {
    Result result;
    result.prompt_tokens = prompt_tokens.size();
    result.client_id = client_id.empty() ? "anonymous" : client_id;
    if (current == nullptr || prompt_tokens.empty()) {
      return result;
    }
    if (is_cancelled && is_cancelled()) {
      result.cancelled = true;
      return result;
    }

    auto request = current->scheduler->Submit(
        std::move(prompt_tokens), max_tokens, sampling, is_cancelled,
        static_cast<bool>(on_token),
        TextRequestMetadata{
            .client_id = std::move(client_id),
            .deadline = std::nullopt,
            .request_start = request_start,
            .prompt_context = std::move(context),
            .cache_prompt = cache_prompt,
            .cache_prefix_tokens = cache_prefix_tokens,
            .stop_sequences = stop_sequences,
        });
    result = request.Wait(on_token);

    return result;
  }

  mutable std::mutex state_mutex;
  std::shared_ptr<const State> state;
#endif
};

InferenceBackend::InferenceBackend() : impl_(std::make_unique<Impl>()) {}

InferenceBackend::~InferenceBackend() = default;

bool InferenceBackend::load(const std::string& model_path, std::string* error,
                            std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            const TextSpeculativeConfig& speculative_config,
                            const TextDiskCacheConfig& disk_cache_config,
                            const std::string& vision_model_path) {
#if defined(ENGINE_ENABLE_HIP)
  TextDiskCacheConfig resolved_disk_cache_config = disk_cache_config;
  std::string load_error;
  auto reader_owner = core::GgufReader::OpenFile(model_path, &load_error);
  if (reader_owner == nullptr) {
    SetError(error, "Failed to open GGUF: " + load_error);
    return false;
  }
  const std::shared_ptr<const core::GgufReader> reader(std::move(reader_owner));
  if (reader->GetMetadataString("general.architecture") == "deepseek4") {
    if (!vision_model_path.empty()) {
      SetError(error, "DeepSeek does not support --mmproj");
      return false;
    }
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kDSpark) {
      SetError(error,
               "DeepSeek HTTP models support only DSpark speculative decoding");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        speculative_config.draft_model_path.empty()) {
      SetError(error, "DeepSeek DSpark HTTP decoding requires --dspark-model");
      return false;
    }
    if (!models::deepseek_v4_flash::ValidateGgufTemplate(*reader,
                                                         &load_error)) {
      SetError(error, "Unsupported DeepSeek chat template: " + load_error);
      return false;
    }
    auto model = models::deepseek_v4_flash::Model::Load(
        model_path,
        models::deepseek_v4_flash::ModelOptions{
            .max_context = max_context,
            .dspark_model_path =
                speculative_config.backend == TextSpeculativeBackend::kDSpark
                    ? speculative_config.draft_model_path
                    : std::string{},
        },
        &load_error);
    if (model == nullptr) {
      SetError(error, "Failed to create DeepSeek model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "DeepSeek", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kDSpark &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "DSpark", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config));
  }
  if (reader->GetMetadataString("general.architecture") == "qwen4exp") {
    if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
        speculative_config.backend != TextSpeculativeBackend::kMtp) {
      SetError(error,
               "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
               "decoding (--speculative mtp --mtp-model)");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        speculative_config.draft_model_path.empty()) {
      SetError(error,
               "Qwen3.8-Flash-Next MTP HTTP decoding requires --mtp-model");
      return false;
    }
    if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        (speculative_config.max_draft_tokens == 0 ||
         speculative_config.min_draft_tokens != 1)) {
      SetError(error,
               "Flash-Next MTP requires a positive draft limit and "
               "--min-draft-tokens 1");
      return false;
    }
    // The compiled Qwen3.8 chat template renders through the artifact's
    // own tokenizer (the same vocabulary and pre-tokenizer as Qwen3.8).
    if (!tokenization::QwenChatTemplate::ValidateGgufTemplate(*reader,
                                                              &load_error)) {
      SetError(error,
               "Unsupported Qwen3.8-Flash-Next chat template: " + load_error);
      return false;
    }
    // The model owns prefill geometry for both bulk and scheduled requests.
    auto model = models::qwen38_flash_next::Model::Load(
        model_path,
        models::qwen38_flash_next::ModelOptions{
            .max_context = max_context,
            .mtp_model_path =
                speculative_config.backend == TextSpeculativeBackend::kMtp
                    ? speculative_config.draft_model_path
                    : std::string{},
            .max_draft_tokens = speculative_config.max_draft_tokens,
            .vision_model_path = vision_model_path,
            .decode_concurrency = static_cast<std::uint32_t>(
                std::clamp<std::size_t>(session_count, 1, 8)),
        },
        &load_error);
    if (model == nullptr) {
      SetError(error,
               "Failed to create Qwen3.8-Flash-Next model: " + load_error);
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
        !FingerprintArtifact(
            "Qwen3.8-Flash-Next", *reader,
            &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
      return false;
    }
    if (DiskCacheEnabled(resolved_disk_cache_config) &&
        speculative_config.backend == TextSpeculativeBackend::kMtp &&
        resolved_disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !FingerprintArtifactFile(
            "Qwen3.8-Flash-Next MTP", speculative_config.draft_model_path,
            &resolved_disk_cache_config.draft_model_artifact_fingerprint,
            error)) {
      return false;
    }
    return load(std::move(model), error, max_context, session_count,
                prefill_policy, scheduler_policy, speculative_config,
                std::move(resolved_disk_cache_config));
  }
  std::shared_ptr<models::qwen::vision::Encoder> vision;
  try {
    if (reader->GetMetadataUint64("qwen35.embedding_length") == 5120) {
      vision = models::qwen::vision::Encoder::Open(model_path,
                                                   vision_model_path, 5120);
    } else if (!vision_model_path.empty()) {
      throw std::invalid_argument(
          "image input supports Qwen3.8-27B and Flash-Next");
    }
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
  Logger::Info("loader", "event=load_phase phase=target_weights " +
                             Logger::MemoryStatus());
  auto model =
      hip::QwenGpuModel::CreateFromGguf(reader, &load_error, std::move(vision));
  if (model == nullptr) {
    SetError(error, "Failed to create GPU model: " + load_error);
    return false;
  }
  if (DiskCacheEnabled(resolved_disk_cache_config) &&
      resolved_disk_cache_config.model_artifact_fingerprint.empty() &&
      !FingerprintArtifact(
          "Qwen", *reader,
          &resolved_disk_cache_config.model_artifact_fingerprint, error)) {
    return false;
  }
  return load(std::move(model), error, max_context, session_count,
              prefill_policy, scheduler_policy, speculative_config,
              std::move(resolved_disk_cache_config));
#else
  (void)model_path;
  (void)max_context;
  (void)session_count;
  (void)prefill_policy;
  (void)scheduler_policy;
  (void)speculative_config;
  (void)disk_cache_config;
  (void)vision_model_path;
  SetError(error, "HTTP inference requires the HIP backend");
  return false;
#endif
}

#if defined(ENGINE_ENABLE_HIP)
bool InferenceBackend::load(std::shared_ptr<const hip::QwenGpuModel> model,
                            std::string* error, std::uint32_t max_context,
                            std::size_t session_count,
                            TextPrefillPolicy prefill_policy,
                            TextSchedulerPolicy scheduler_policy,
                            TextSpeculativeConfig speculative_config,
                            TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen GPU model must not be null");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDSpark) {
    SetError(error, "DSpark HTTP decoding requires a DeepSeek model");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp) {
    SetError(error, "MTP HTTP decoding requires a Qwen Flash-Next model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (!disk_cache_config.draft_model_artifact_fingerprint.empty() &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0 ||
       disk_cache_config.staging_capacity_bytes == 0)) {
    SetError(error, "Qwen persistent disk cache configuration is invalid");
    return false;
  }
  if (speculative_config.max_draft_tokens == 0 ||
      speculative_config.min_draft_tokens == 0 ||
      speculative_config.min_draft_tokens >
          speculative_config.max_draft_tokens) {
    SetError(error, "HTTP speculative draft limits are invalid");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kDFlash &&
      speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DFlash2 requires --min-draft-tokens 1; bound blocks with "
             "--draft-tokens");
    return false;
  }

  try {
    std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
    speculative::SpeculativeOptions speculative_options;
    if (speculative_config.backend == TextSpeculativeBackend::kDFlash) {
      if (speculative_config.draft_model_path.empty()) {
        SetError(error, "DFlash HTTP decoding requires --dflash-model");
        return false;
      }
      std::string dflash_error;
      auto dflash_reader_owner = core::GgufReader::OpenFile(
          speculative_config.draft_model_path, &dflash_error);
      if (dflash_reader_owner == nullptr) {
        SetError(error, "Failed to open DFlash GGUF: " + dflash_error);
        return false;
      }
      std::shared_ptr<const core::GgufReader> dflash_reader(
          std::move(dflash_reader_owner));
      if (DiskCacheEnabled(disk_cache_config) &&
          disk_cache_config.draft_model_artifact_fingerprint.empty() &&
          !FingerprintArtifact(
              "DFlash", *dflash_reader,
              &disk_cache_config.draft_model_artifact_fingerprint, error)) {
        return false;
      }
      if (DiskCacheEnabled(disk_cache_config) &&
          !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) {
        SetError(error,
                 "Qwen DFlash persistent disk cache configuration is "
                 "invalid");
        return false;
      }
      Logger::Info("loader", "event=load_phase phase=draft_weights " +
                                 Logger::MemoryStatus());
      dflash_model = hip::QwenDFlashGpuModel::Create(std::move(dflash_reader),
                                                     model, &dflash_error);
      if (dflash_model == nullptr) {
        SetError(error, "Failed to create DFlash model: " + dflash_error);
        return false;
      }

      speculative_options.dflash_policy = speculative_config.dflash_policy;
      speculative_options.max_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.min_draft_tokens =
          speculative_config.min_draft_tokens;
      speculative_options.initial_draft_tokens =
          speculative_config.max_draft_tokens;
      speculative_options.use_batched_verification = true;
      speculative_options.retain_frontier_logits = true;
      speculative_options.enable_adaptive_draft_length = false;
    }

    auto new_state = std::make_shared<Impl::State>();
    auto runner = std::make_shared<QwenTextRunner>(
        std::move(model), max_context, std::move(dflash_model),
        speculative_options, disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    Logger::Info("loader",
                 "event=load_phase phase=sessions " + Logger::MemoryStatus());
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache));
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::deepseek_v4_flash::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "DeepSeek model must not be null");
    return false;
  }
  const bool use_dspark =
      speculative_config.backend == TextSpeculativeBackend::kDSpark;
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (!use_dspark || !model->HasDspark())) {
    SetError(error, "DeepSeek DSpark requires a loaded support model");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context > model->MaxContext()) {
    SetError(error, "HTTP context exceeds the loaded DeepSeek model context");
    return false;
  }
  if (use_dspark && (speculative_config.max_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens == 0 ||
                     speculative_config.min_draft_tokens >
                         speculative_config.max_draft_tokens)) {
    SetError(error, "DeepSeek DSpark draft limits are invalid");
    return false;
  }
  if (use_dspark && speculative_config.min_draft_tokens != 1) {
    SetError(error,
             "DSpark uses model-owned adaptive drafting; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (use_dspark &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0 ||
       disk_cache_config.staging_capacity_bytes == 0)) {
    SetError(error, "DeepSeek persistent disk cache configuration is invalid");
    return false;
  }

  try {
    auto new_state = std::make_shared<Impl::State>();
    auto runner = std::make_shared<DeepSeekTextRunner>(
        std::move(model), max_context, use_dspark,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache));
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}

bool InferenceBackend::load(
    std::shared_ptr<models::qwen38_flash_next::Model> model, std::string* error,
    std::uint32_t max_context, std::size_t session_count,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy,
    TextSpeculativeConfig speculative_config,
    TextDiskCacheConfig disk_cache_config) {
  if (model == nullptr) {
    SetError(error, "Qwen3.8-Flash-Next model must not be null");
    return false;
  }
  if (session_count == 0) {
    SetError(error, "HTTP session count must be at least one");
    return false;
  }
  if (max_context == 0 || max_context > model->MaxContext()) {
    SetError(
        error,
        "HTTP context exceeds the loaded Qwen3.8-Flash-Next model context");
    return false;
  }
  if (speculative_config.backend != TextSpeculativeBackend::kDisabled &&
      (speculative_config.backend != TextSpeculativeBackend::kMtp ||
       !model->HasMtp())) {
    SetError(error,
             "Flash-Next MTP requires a model loaded with its draft sidecar");
    return false;
  }
  if (speculative_config.backend == TextSpeculativeBackend::kMtp &&
      (speculative_config.max_draft_tokens == 0 ||
       speculative_config.min_draft_tokens != 1)) {
    SetError(error,
             "Qwen3.8-Flash-Next MTP drafts a fixed chain; custom draft "
             "floors are unsupported");
    return false;
  }
  if (DiskCacheEnabled(disk_cache_config) &&
      (!IsSha256Hex(disk_cache_config.model_artifact_fingerprint) ||
       (speculative_config.backend == TextSpeculativeBackend::kMtp &&
        !IsSha256Hex(disk_cache_config.draft_model_artifact_fingerprint)) ||
       disk_cache_config.capacity_bytes == 0 ||
       disk_cache_config.staging_capacity_bytes == 0)) {
    SetError(error,
             "Qwen3.8-Flash-Next persistent disk cache configuration is "
             "invalid");
    return false;
  }
  try {
    auto new_state = std::make_shared<Impl::State>();
    auto runner = std::make_shared<QwenFlashNextTextRunner>(
        std::move(model), max_context,
        speculative_config.backend == TextSpeculativeBackend::kMtp,
        speculative_config.max_draft_tokens,
        disk_cache_config.model_artifact_fingerprint,
        disk_cache_config.draft_model_artifact_fingerprint);
    new_state->model_id = runner->Descriptor().model_id;
    std::optional<TextRunnerDiskCacheOptions> runner_disk_cache;
    if (DiskCacheEnabled(disk_cache_config)) {
      runner_disk_cache = TextRunnerDiskCacheOptions{
          .directory = std::move(disk_cache_config.directory),
          .capacity_bytes = disk_cache_config.capacity_bytes,
          .staging_capacity_bytes = disk_cache_config.staging_capacity_bytes,
      };
    }
    auto runner_pool = std::make_shared<TextRunnerPool>(
        std::move(runner), session_count, std::move(runner_disk_cache));
    new_state->scheduler = std::make_shared<TextGenerationScheduler>(
        std::move(runner_pool), prefill_policy, scheduler_policy);
    {
      const std::lock_guard<std::mutex> lock(impl_->state_mutex);
      impl_->state = std::move(new_state);
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, exception.what());
    return false;
  }
}
#endif

std::string InferenceBackend::model_id() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->model_id : "unknown";
#else
  return "unknown";
#endif
}

bool InferenceBackend::ready() const {
#if defined(ENGINE_ENABLE_HIP)
  return impl_->Snapshot() != nullptr;
#else
  return false;
#endif
}

InferenceBackend::SamplingDefaults InferenceBackend::sampling_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->sampling_defaults : SamplingDefaults{};
#else
  return {};
#endif
}

ReasoningOptions InferenceBackend::reasoning_defaults() const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr ? state->reasoning_defaults : ReasoningOptions{};
#else
  return {};
#endif
}

InferenceBackend::InitialOutputState InferenceBackend::initial_output_state(
    const ChatRequest& request) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  return state != nullptr
             ? state->scheduler->runner().InitialOutputState(request)
             : InitialOutputState::kAuto;
#else
  (void)request;
  return InitialOutputState::kAuto;
#endif
}

void InferenceBackend::set_model_id(const std::string& model_id) {
#if defined(ENGINE_ENABLE_HIP)
  if (model_id.empty()) {
    return;
  }
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->model_id = model_id;
  impl_->state = std::move(updated);
#else
  (void)model_id;
#endif
}

void InferenceBackend::set_sampling_defaults(
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config) {
#if defined(ENGINE_ENABLE_HIP)
  sampling_config.Validate();
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->sampling_defaults = {
      .max_tokens = max_tokens,
      .sampling = sampling_config,
  };
  impl_->state = std::move(updated);
#else
  (void)max_tokens;
  (void)sampling_config;
#endif
}

void InferenceBackend::set_reasoning_defaults(
    const ReasoningOptions& reasoning) {
#if defined(ENGINE_ENABLE_HIP)
  const std::lock_guard<std::mutex> lock(impl_->state_mutex);
  if (impl_->state == nullptr) {
    return;
  }
  auto updated = std::make_shared<Impl::State>(*impl_->state);
  updated->reasoning_defaults = reasoning;
  impl_->state = std::move(updated);
#else
  (void)reasoning;
#endif
}

InferenceBackend::Result InferenceBackend::complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token,
    std::string_view client_id,
    const std::vector<std::string>& stop_sequences) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt_tokens = state->scheduler->runner().Tokenize(prompt);
  return impl_->GenerateScheduled(
      state, std::move(prompt_tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, std::string(client_id), {}, true,
      0, stop_sequences);
#else
  (void)client_id;
  (void)stop_sequences;
  (void)prompt;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const ChatRequest& request, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled, const TokenCallback& on_token) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return {};
  }
  auto prompt = state->scheduler->runner().PreparePrompt(request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return {};
  }
  return impl_->GenerateScheduled(
      state, std::move(prompt->tokens), request_start, max_tokens,
      sampling_config, is_cancelled, on_token, request.client_id,
      std::move(prompt->context), request.cache_prompt,
      prompt->cache_prefix_tokens, request.stop_sequences);
#else
  (void)request;
  (void)max_tokens;
  (void)sampling_config;
  (void)is_cancelled;
  (void)on_token;
  return {};
#endif
}

std::shared_ptr<InferenceBackend::GenerationRequest>
InferenceBackend::start_chat(const ChatRequest& request, std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling_config,
                             const CancellationCheck& is_cancelled,
                             bool stream_output) {
#if defined(ENGINE_ENABLE_HIP)
  const auto request_start = Clock::now();
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  auto prompt = state->scheduler->runner().PreparePrompt(request);
  if (!prompt.has_value() || prompt->tokens.empty()) {
    return TextGenerationBackend::start_chat(
        request, max_tokens, sampling_config, is_cancelled, stream_output);
  }

  const std::string client_id =
      request.client_id.empty() ? "anonymous" : request.client_id;
  auto scheduled_request = state->scheduler->Submit(
      std::move(prompt->tokens), max_tokens, sampling_config, is_cancelled,
      stream_output,
      TextRequestMetadata{
          .client_id = client_id,
          .deadline = std::nullopt,
          .request_start = request_start,
          .prompt_context = std::move(prompt->context),
          .cache_prompt = request.cache_prompt,
          .cache_prefix_tokens = prompt->cache_prefix_tokens,
          .stop_sequences = request.stop_sequences,
      });
  return std::make_shared<Impl::ScheduledGenerationRequest>(
      state, std::move(scheduled_request));
#else
  return TextGenerationBackend::start_chat(request, max_tokens, sampling_config,
                                           is_cancelled, stream_output);
#endif
}

InferenceBackend::Result InferenceBackend::chat(
    const std::vector<tokenization::ChatMessage>& messages,
    std::size_t max_tokens, const sampling::SamplingConfig& sampling_config,
    const CancellationCheck& is_cancelled) {
  return chat(ChatRequest{messages}, max_tokens, sampling_config, is_cancelled);
}

std::size_t InferenceBackend::count_tokens(std::string_view text) const {
#if defined(ENGINE_ENABLE_HIP)
  const auto state = impl_->Snapshot();
  if (state == nullptr) {
    return 0;
  }
  return state->scheduler->runner().Tokenize(text).size();
#else
  (void)text;
  return 0;
#endif
}

}  // namespace gufo::server
