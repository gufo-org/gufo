#ifndef GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
#define GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/reasoning.hpp"
#include "src/core/sampling.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace gufo::server {

enum class TextGenerationErrorCode : std::uint8_t {
  kQueueFull,
  kClientQueueFull,
  kDeadlineExceeded,
  kOutputLimit,
  kOutputBackpressure,
  kSchedulerStopping,
  kToolChoiceUnsatisfied,
};

class TextGenerationError final : public std::runtime_error {
public:
  TextGenerationError(TextGenerationErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  [[nodiscard]] TextGenerationErrorCode code() const noexcept { return code_; }
  [[nodiscard]] int http_status() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
      case TextGenerationErrorCode::kClientQueueFull:
        return 429;
      case TextGenerationErrorCode::kDeadlineExceeded:
        return 408;
      case TextGenerationErrorCode::kOutputLimit:
      case TextGenerationErrorCode::kOutputBackpressure:
      case TextGenerationErrorCode::kSchedulerStopping:
        return 503;
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return 502;
    }
    return 500;
  }
  [[nodiscard]] const char* stable_code() const noexcept {
    switch (code_) {
      case TextGenerationErrorCode::kQueueFull:
        return "queue_full";
      case TextGenerationErrorCode::kClientQueueFull:
        return "client_queue_full";
      case TextGenerationErrorCode::kDeadlineExceeded:
        return "deadline_exceeded";
      case TextGenerationErrorCode::kOutputLimit:
        return "output_limit";
      case TextGenerationErrorCode::kOutputBackpressure:
        return "output_backpressure";
      case TextGenerationErrorCode::kSchedulerStopping:
        return "scheduler_stopping";
      case TextGenerationErrorCode::kToolChoiceUnsatisfied:
        return "tool_choice_unsatisfied";
    }
    return "generation_error";
  }
  [[nodiscard]] bool retryable() const noexcept {
    return code_ != TextGenerationErrorCode::kOutputLimit &&
           code_ != TextGenerationErrorCode::kToolChoiceUnsatisfied;
  }

private:
  TextGenerationErrorCode code_;
};

struct ChatRequest {
  enum class ToolChoice : std::uint8_t {
    kAuto,
    kNone,
    kRequired,
  };

  ChatRequest() = default;
  explicit ChatRequest(std::vector<tokenization::ChatMessage> chat_messages)
      : messages(std::move(chat_messages)) {}

  std::vector<tokenization::ChatMessage> messages;
  std::vector<tokenization::ChatTool> tools;
  std::string client_id{"anonymous"};
  ToolChoice tool_choice{ToolChoice::kAuto};
  ReasoningOptions reasoning;
  bool add_vision_id{false};
  /// Bypass prompt reuse for this request; its completed state may be retained.
  bool cache_prompt{true};
  std::vector<std::string> stop_sequences;
};

/// Model-agnostic text generation boundary used by the HTTP transport.
///
/// Implementations retain ownership of tokenization, templates, complete
/// continuation state, and execution resources. The transport observes only
/// generated text pieces and aggregate request metrics.
class TextGenerationBackend {
public:
  using CancellationCheck = std::function<bool()>;
  using TokenCallback = std::function<bool(std::string_view)>;

  enum class FinishReason : std::uint8_t {
    kStop,
    kStopSequence,
    kLength,
    kCancelled,
  };

  enum class InitialOutputState : std::uint8_t {
    kAuto,
    kReasoning,
    kContent,
  };

  struct SamplingDefaults {
    /// Zero means generate until EOS or the remaining context is exhausted.
    std::size_t max_tokens{0};
    sampling::SamplingConfig sampling;
  };

  struct Result {
    std::string text;
    std::vector<tokenization::TokenId> tokens;
    std::size_t prompt_tokens{0};
    std::size_t cached_prompt_tokens{0};
    std::size_t cache_restore_bytes{0};
    std::size_t cache_snapshot_bytes{0};
    /// Bytes admitted to background persistence; completion is logged by cache.
    std::size_t cache_disk_queued_bytes{0};
    std::size_t cache_shared_bytes{0};
    /// Shared-prefix snapshots queued while prefilling this request.
    std::size_t cache_shared_prefix_snapshots{0};
    std::size_t cache_shared_prefix_bytes{0};
    std::size_t completion_tokens{0};
    /// Generated reasoning tokens, excluding the closing template delimiter.
    std::size_t reasoning_tokens{0};
    std::size_t draft_tokens{0};
    std::size_t draft_accepted_tokens{0};
    std::size_t prefill_tokens{0};
    std::size_t prefill_chunks{0};
    std::size_t active_decode_prefill_chunks{0};
    std::size_t max_prefill_chunk_tokens{0};
    std::size_t max_consecutive_active_prefill_chunks{0};
    std::size_t configured_active_prefill_tokens{0};
    std::size_t queue_depth_at_submit{0};
    std::size_t client_queue_depth_at_submit{0};
    std::size_t resident_requests_at_admission{0};
    std::size_t requested_logical_concurrency{1};
    std::size_t physical_execution_width{1};
    std::size_t max_buffered_output_bytes{0};
    double queue_ms{0.0};
    double cache_restore_ms{0.0};
    double cache_snapshot_ms{0.0};
    double cache_disk_enqueue_ms{0.0};
    double cache_shared_prefix_ms{0.0};
    double ttft_ms{0.0};
    double mean_inter_token_ms{0.0};
    double max_inter_token_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::string client_id{"anonymous"};
    std::string execution_plan{"serial-c1"};
    std::string prefill_fallback_reason;
    std::string cache_miss_reason;
    std::size_t cache_common_prefix_tokens{0};
    std::size_t cache_checkpoint_tokens{0};
    FinishReason finish_reason{FinishReason::kStop};
    std::string stop_sequence;
    bool incremental_prefill_supported{false};
    bool cache_hit{false};
    bool cache_disk_hit{false};
    bool cancelled{false};
  };

  class GenerationRequest {
  public:
    GenerationRequest() = default;
    virtual ~GenerationRequest() = default;

    GenerationRequest(const GenerationRequest&) = delete;
    GenerationRequest& operator=(const GenerationRequest&) = delete;
    GenerationRequest(GenerationRequest&&) = delete;
    GenerationRequest& operator=(GenerationRequest&&) = delete;

    virtual Result Wait(const TokenCallback& on_token = {}) = 0;
    virtual void Cancel() noexcept = 0;
  };

  TextGenerationBackend() = default;
  virtual ~TextGenerationBackend() = default;

  TextGenerationBackend(const TextGenerationBackend&) = delete;
  TextGenerationBackend& operator=(const TextGenerationBackend&) = delete;
  TextGenerationBackend(TextGenerationBackend&&) = delete;
  TextGenerationBackend& operator=(TextGenerationBackend&&) = delete;

  [[nodiscard]] virtual std::string model_id() const = 0;
  [[nodiscard]] virtual bool ready() const = 0;
  /// Maximum tokens accepted by the loaded model under the configured context.
  /// Zero when no text model is loaded.
  [[nodiscard]] virtual std::uint32_t max_context() const { return 0; }
  [[nodiscard]] virtual SamplingDefaults sampling_defaults() const {
    return {};
  }
  [[nodiscard]] virtual ReasoningOptions reasoning_defaults() const {
    return {};
  }
  [[nodiscard]] virtual InitialOutputState initial_output_state(
      const ChatRequest&) const {
    return InitialOutputState::kAuto;
  }

  virtual Result complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {},
      const TokenCallback& on_token = {},
      std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {}) = 0;

  virtual Result chat(const ChatRequest& request, std::size_t max_tokens,
                      const sampling::SamplingConfig& sampling,
                      const CancellationCheck& is_cancelled = {},
                      const TokenCallback& on_token = {}) = 0;

  Result complete(std::string_view prompt, std::size_t max_tokens,
                  float temperature, const CancellationCheck& is_cancelled = {},
                  const TokenCallback& on_token = {},
                  const std::vector<std::string>& stop_sequences = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return complete(prompt, max_tokens, config, is_cancelled, on_token,
                    "anonymous", stop_sequences);
  }

  Result chat(const ChatRequest& request, std::size_t max_tokens,
              float temperature, const CancellationCheck& is_cancelled = {},
              const TokenCallback& on_token = {}) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return chat(request, max_tokens, config, is_cancelled, on_token);
  }

  /// Reserves admission before a streaming response commits successful headers.
  ///
  /// Backends with an asynchronous scheduler override this. The default
  /// request defers the existing blocking chat call until Wait().
  virtual std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false);

  /// Raw prompt generation with admission reserved before SSE headers.
  virtual std::shared_ptr<GenerationRequest> start_complete(
      std::string_view prompt, std::size_t max_tokens,
      const sampling::SamplingConfig& sampling,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false,
      bool ignore_eos = false, std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {});

  std::shared_ptr<GenerationRequest> start_chat(
      const ChatRequest& request, std::size_t max_tokens, float temperature,
      const CancellationCheck& is_cancelled = {}, bool stream_output = false) {
    sampling::SamplingConfig config;
    config.temperature = temperature;
    return start_chat(request, max_tokens, config, is_cancelled, stream_output);
  }

  [[nodiscard]] virtual std::size_t count_tokens(
      std::string_view text) const = 0;
};

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_complete(
    std::string_view prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool stream_output, bool ignore_eos,
    std::string_view client_id,
    const std::vector<std::string>& stop_sequences) {
  (void)stream_output;
  if (ignore_eos)
    throw std::invalid_argument("backend does not support ignore_eos");
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              std::string prompt, std::size_t max_tokens,
                              sampling::SamplingConfig sampling,
                              CancellationCheck cancellation,
                              std::string client_id,
                              std::vector<std::string> stop_sequences)
        : backend_(backend),
          prompt_(std::move(prompt)),
          max_tokens_(max_tokens),
          sampling_(sampling),
          cancellation_(std::move(cancellation)),
          client_id_(std::move(client_id)),
          stop_sequences_(std::move(stop_sequences)) {}

    Result Wait(const TokenCallback& on_token) override {
      if (waited_.exchange(true, std::memory_order_acq_rel))
        throw std::logic_error("generation request was already consumed");
      return backend_.complete(
          prompt_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (cancellation_ && cancellation_());
          },
          on_token, client_id_, stop_sequences_);
    }
    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    std::string prompt_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck cancellation_;
    std::string client_id_;
    std::vector<std::string> stop_sequences_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };
  return std::make_shared<DeferredGenerationRequest>(
      *this, std::string(prompt), max_tokens, sampling, is_cancelled,
      std::string(client_id), stop_sequences);
}

inline std::shared_ptr<TextGenerationBackend::GenerationRequest>
TextGenerationBackend::start_chat(const ChatRequest& request,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  const CancellationCheck& is_cancelled,
                                  bool stream_output) {
  (void)stream_output;
  class DeferredGenerationRequest final : public GenerationRequest {
  public:
    DeferredGenerationRequest(TextGenerationBackend& backend,
                              ChatRequest chat_request, std::size_t token_limit,
                              sampling::SamplingConfig sampling_config,
                              CancellationCheck external_cancellation)
        : backend_(backend),
          request_(std::move(chat_request)),
          max_tokens_(token_limit),
          sampling_(sampling_config),
          external_cancellation_(std::move(external_cancellation)) {}

    Result Wait(const TokenCallback& on_token) override {
      if (waited_.exchange(true, std::memory_order_acq_rel)) {
        throw std::logic_error("generation request was already consumed");
      }
      return backend_.chat(
          request_, max_tokens_, sampling_,
          [this] {
            return cancelled_.load(std::memory_order_acquire) ||
                   (external_cancellation_ && external_cancellation_());
          },
          on_token);
    }

    void Cancel() noexcept override {
      cancelled_.store(true, std::memory_order_release);
    }

  private:
    TextGenerationBackend& backend_;
    ChatRequest request_;
    std::size_t max_tokens_;
    sampling::SamplingConfig sampling_;
    CancellationCheck external_cancellation_;
    std::atomic<bool> waited_{false};
    std::atomic<bool> cancelled_{false};
  };

  return std::make_shared<DeferredGenerationRequest>(*this, request, max_tokens,
                                                     sampling, is_cancelled);
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_BACKEND_HPP_
