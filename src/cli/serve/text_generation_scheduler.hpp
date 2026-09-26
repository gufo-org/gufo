#ifndef GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
#define GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/cli/serve/text_model_runner.hpp"

namespace gufo::server {

enum class TextRequestPhase : std::uint8_t {
  kQueued,
  kAdmitted,
  kPrefilling,
  kDecodeReady,
  kDecoding,
  kTerminal,
};

inline constexpr std::size_t kDefaultDecodeActivePrefillTokens = 512;
inline constexpr std::size_t kDefaultMaxOutputBytes =
    static_cast<std::size_t>(1024) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytes =
    static_cast<std::size_t>(64) * 1024;
inline constexpr std::size_t kDefaultMaxBufferedOutputBytesTotal =
    static_cast<std::size_t>(256) * 1024;

struct TextPrefillPolicy {
  std::size_t decode_active_tokens{kDefaultDecodeActivePrefillTokens};
};

struct TextSchedulerPolicy {
  std::size_t max_pending_requests{16};
  std::size_t max_pending_requests_per_client{4};
  std::size_t max_output_bytes_per_request{kDefaultMaxOutputBytes};
  std::size_t max_buffered_output_bytes_per_request{
      kDefaultMaxBufferedOutputBytes};
  std::size_t max_buffered_output_bytes_total{
      kDefaultMaxBufferedOutputBytesTotal};
  std::chrono::milliseconds request_timeout{0};
  bool log_progress{false};
};

/// Single-owner scheduler for opaque text-model runner states.
///
/// Submitters never execute model code. One scheduler thread owns admission,
/// runner leases, prefill/decode work units, cancellation, and reclamation.
class TextGenerationScheduler {
public:
  using Clock = std::chrono::steady_clock;
  using Result = TextGenerationBackend::Result;
  using CancellationCheck = TextGenerationBackend::CancellationCheck;
  using TokenCallback = TextGenerationBackend::TokenCallback;

  struct RequestMetadata {
    std::string client_id{"anonymous"};
    std::optional<Clock::time_point> deadline;
    Clock::time_point request_start{Clock::now()};
    std::shared_ptr<const TextPromptContext> prompt_context;
    bool cache_prompt{true};
    std::size_t cache_prefix_tokens{0};
    std::vector<std::string> stop_sequences;
  };

  class Request {
  public:
    Request();
    ~Request();

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) noexcept;
    Request& operator=(Request&&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t id() const noexcept;
    [[nodiscard]] TextRequestPhase phase() const noexcept;

    /// Consumes queued output pieces on the calling thread and waits for the
    /// scheduler-owned request to become terminal.
    Result Wait(const TokenCallback& on_token = {});
    void Cancel() noexcept;

  private:
    friend class TextGenerationScheduler;
    struct Impl;

    explicit Request(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

  explicit TextGenerationScheduler(std::shared_ptr<TextRunnerPool> runner_pool,
                                   TextPrefillPolicy prefill_policy = {},
                                   TextSchedulerPolicy scheduler_policy = {});
  ~TextGenerationScheduler();

  TextGenerationScheduler(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler& operator=(const TextGenerationScheduler&) = delete;
  TextGenerationScheduler(TextGenerationScheduler&&) = delete;
  TextGenerationScheduler& operator=(TextGenerationScheduler&&) = delete;

  [[nodiscard]] const TextModelRunner& runner() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t buffered_output_bytes() const noexcept;
  [[nodiscard]] std::size_t max_buffered_output_bytes() const noexcept;

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens,
                               const sampling::SamplingConfig& sampling,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled = {},
                               bool publish_token_pieces = false);

  [[nodiscard]] Request Submit(std::vector<TextRunnerToken> prompt,
                               std::size_t max_tokens, float temperature,
                               const CancellationCheck& is_cancelled,
                               bool publish_token_pieces,
                               RequestMetadata metadata);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using TextRequestMetadata = TextGenerationScheduler::RequestMetadata;

}  // namespace gufo::server

#endif  // GUFO_SERVER_TEXT_GENERATION_SCHEDULER_HPP_
