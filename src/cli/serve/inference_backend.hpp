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

#include "gufo/ext/text.hpp"
#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"

namespace gufo::server {

using TextSpeculativeBackend = ext::TextSpeculativeBackend;
using TextDFlashDraftPolicy = ext::TextDFlashDraftPolicy;
using TextSpeculativeConfig = ext::TextSpeculativeConfig;
using TextDiskCacheConfig = ext::TextDiskCacheConfig;

namespace speculative {
using ext::ParseTextDFlashDraftPolicy;
using ext::TextDFlashDraftPolicyName;
[[nodiscard]] inline ext::TextDFlashDraftPolicy ParseDFlashDraftPolicy(
    std::string_view value) {
  return ext::ParseTextDFlashDraftPolicy(value);
}
[[nodiscard]] inline std::string_view DFlashDraftPolicyName(
    ext::TextDFlashDraftPolicy policy) {
  return ext::TextDFlashDraftPolicyName(policy);
}
}  // namespace speculative

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
  /// `enable_yarn` opts Flash-Next into static YaRN, required for (and only
  /// accepted with) a `max_context` above its native context.
  bool load(const std::string& model_path, std::string* error,
            std::uint32_t max_context = 0, std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            const TextSpeculativeConfig& speculative_config = {},
            const TextDiskCacheConfig& disk_cache_config = {},
            const std::string& vision_model_path = {},
            TextRunnerRamCacheOptions ram_cache_config = {},
            bool enable_yarn = false);

  /// Installs a previously prepared runner with request-owned sessions.
  bool load(std::shared_ptr<ext::TextModelRunner> runner, std::string* error,
            std::size_t session_count = 1,
            TextPrefillPolicy prefill_policy = {},
            TextSchedulerPolicy scheduler_policy = {},
            const TextDiskCacheConfig& disk_cache_config = {},
            TextRunnerRamCacheOptions ram_cache_config = {});

  /// Stable model identifier used in API responses.
  [[nodiscard]] std::string model_id() const override;
  [[nodiscard]] std::vector<std::string> model_ids() const override;
  [[nodiscard]] bool ready() const override;
  [[nodiscard]] bool device_lost() const override;
  [[nodiscard]] bool supports_images() const override;
  [[nodiscard]] std::string model_path() const override;
  [[nodiscard]] std::string build_info() const override;
  [[nodiscard]] SamplingDefaults sampling_defaults() const override;
  [[nodiscard]] std::uint32_t max_context() const override;
  [[nodiscard]] std::vector<SessionState> session_states() const override;
  [[nodiscard]] ReasoningOptions reasoning_defaults() const override;
  [[nodiscard]] InitialOutputState initial_output_state(
      const ChatRequest& request) const override;
  /// The first name is primary; the rest are additional served names.
  void set_model_ids(std::vector<std::string> model_ids);
  void set_served_artifact(const std::string& model_path,
                           const std::string& build_info);
  void set_sampling_defaults(std::size_t max_tokens,
                             const sampling::SamplingConfig& sampling,
                             sampling::SamplingOverrides supplied =
                                 sampling::SamplingOverrides::All());
  void set_reasoning_defaults(const ReasoningOptions& reasoning);

  /// Plain text completion (no chat framing).
  Result complete(std::string_view prompt, std::size_t max_tokens,
                  const sampling::SamplingConfig& sampling,
                  const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  std::string_view client_id = "anonymous",
                  const std::vector<std::string>& stop_sequences = {},
                  bool cache_prompt = true, bool cache_persist = true) override;

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

  std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {},
      bool return_progress = false, bool cache_prompt = true,
      bool cache_persist = true) override;

  Result chat(const std::vector<tokenization::ChatMessage>& messages,
              std::size_t max_tokens, const sampling::SamplingConfig& sampling,
              const CancellationCheck& is_cancelled = {});

  /// Token count of raw text (no generation).
  [[nodiscard]] std::size_t count_tokens(std::string_view text) const override;

  /// Prompt tokens of a chat request as generation would prepare them.
  [[nodiscard]] std::optional<std::size_t> count_chat_tokens(
      const ChatRequest& request) const override;

  /// Prompt text of a chat request as generation would render it.
  [[nodiscard]] std::optional<std::string> render_chat_prompt(
      const ChatRequest& request) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_INFERENCE_BACKEND_HPP_
