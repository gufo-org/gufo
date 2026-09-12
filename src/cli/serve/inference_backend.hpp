#ifndef GUFO_SERVER_INFERENCE_BACKEND_HPP_
#define GUFO_SERVER_INFERENCE_BACKEND_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::hip {
class QwenGpuModel;
}

namespace gufo::models::deepseek_v4_flash {
class Model;
}

namespace gufo::models::qwen38_flash_next {
class Model;
}

namespace gufo::tokenization {
class QwenTokenizer;
}

namespace gufo::server {

enum class TextSpeculativeBackend : std::uint8_t {
  kDisabled,
  kDFlash,
  kDSpark,
  kMtp,  ///< Qwen3.8-Flash-Next's MTP draft block (greedy verification)
};

struct TextSpeculativeConfig {
  TextSpeculativeBackend backend{TextSpeculativeBackend::kDisabled};
  std::string draft_model_path;
  std::uint32_t max_draft_tokens{7};
  std::uint32_t min_draft_tokens{1};
  /// MTP: vocabulary prefix the draft block scores (0 = full vocabulary).
  std::uint32_t draft_vocab{0};
  speculative::DFlashDraftPolicy dflash_policy{
      speculative::DFlashDraftPolicy::kAdaptive};
};

struct TextDiskCacheConfig {
  std::filesystem::path directory;
  std::size_t capacity_bytes{static_cast<std::size_t>(4) * 1024U * 1024U *
                             1024U};
  std::size_t staging_capacity_bytes{static_cast<std::size_t>(512) * 1024U *
                                     1024U};
  std::string model_artifact_fingerprint;
  std::string draft_model_artifact_fingerprint;
};

/// Thread-safe HTTP inference facade over shared immutable GPU model resources
/// and a bounded pool of request-owned executor sessions.
class InferenceBackend final : public TextGenerationBackend {
public:
  using TextGenerationBackend::chat;
  using TextGenerationBackend::complete;
  using TextGenerationBackend::start_chat;

  InferenceBackend();
  ~InferenceBackend() override;

  InferenceBackend(const InferenceBackend&) = delete;
  InferenceBackend& operator=(const InferenceBackend&) = delete;
  InferenceBackend(InferenceBackend&&) = delete;
  InferenceBackend& operator=(InferenceBackend&&) = delete;

  /// Loads weights from a GGUF file. Returns false and sets *error on failure.
  bool load(const std::string& model_path, std::string* error,
            std::uint32_t max_context = 4096, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            const TextSpeculativeConfig& speculative_config = {},
            const TextDiskCacheConfig& disk_cache_config = {});

#if defined(ENGINE_ENABLE_HIP)
  /// Installs a previously loaded model without duplicating mapped weights.
  bool load(std::shared_ptr<const hip::QwenGpuModel> model, std::string* error,
            std::uint32_t max_context = 4096, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded DeepSeek model with request-owned sessions.
  bool load(std::shared_ptr<models::deepseek_v4_flash::Model> model,
            std::string* error, std::uint32_t max_context = 4096,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {},
            TextDiskCacheConfig disk_cache_config = {});

  /// Installs a previously loaded Qwen3.8-Flash-Next model with
  /// request-owned sessions; `tokenizer` is the artifact's tokenizer as the
  /// Qwen chat template renders through it. No continuation snapshots.
  bool load(std::shared_ptr<models::qwen38_flash_next::Model> model,
            std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
            std::string* error, std::uint32_t max_context = 4096,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            TextSpeculativeConfig speculative_config = {});
#endif

  /// Stable model identifier used in API responses.
  [[nodiscard]] std::string model_id() const override;
  [[nodiscard]] bool ready() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  [[nodiscard]] ReasoningOptions reasoning_defaults() const override;
  [[nodiscard]] InitialOutputState initial_output_state(
      const ChatRequest& request) const override;
  void set_model_id(const std::string& model_id);
  void set_sampling_defaults(std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling);
  void set_reasoning_defaults(const ReasoningOptions& reasoning);

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {}) override;

  /// Framed chat conversation; returns the assistant reply.
  Result chat(const ChatRequest& request, std::size_t max_tokens,
              const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) override;

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      bool stream_output = false) override;

  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {});

  /// Token count of raw text (no generation).
  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_INFERENCE_BACKEND_HPP_
