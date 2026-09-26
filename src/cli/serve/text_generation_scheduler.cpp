#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

#include "src/cli/serve/stop_sequences.hpp"

namespace gufo::server {
namespace {

struct OutputBudget {
  explicit OutputBudget(std::size_t byte_limit) : limit(byte_limit) {}

  [[nodiscard]] bool TryReserve(std::size_t bytes) noexcept {
    std::size_t current = buffered_bytes.load(std::memory_order_relaxed);
    while (current <= limit && bytes <= limit - current) {
      const std::size_t updated = current + bytes;
      if (buffered_bytes.compare_exchange_weak(current, updated,
                                               std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
        std::size_t high_water =
            max_buffered_bytes.load(std::memory_order_relaxed);
        while (high_water < updated &&
               !max_buffered_bytes.compare_exchange_weak(
                   high_water, updated, std::memory_order_relaxed,
                   std::memory_order_relaxed)) {
        }
        return true;
      }
    }
    return false;
  }

  void Release(std::size_t bytes) noexcept {
    if (bytes > 0) {
      buffered_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
    }
  }

  const std::size_t limit;
  std::atomic<std::size_t> buffered_bytes{0};
  std::atomic<std::size_t> max_buffered_bytes{0};
};

struct ScheduledRequest {
  std::uint64_t id{0};
  std::string client_id{"anonymous"};
  std::vector<TextRunnerToken> prompt;
  std::shared_ptr<const TextPromptContext> prompt_context;
  bool cache_prompt{true};
  std::size_t cache_prefix_tokens{0};
  bool stop_at_eos{true};
  std::size_t token_limit{1};
  sampling::SamplingConfig sampling;
  TextGenerationScheduler::CancellationCheck external_cancellation;
  bool publish_token_pieces{false};
  TextGenerationScheduler::Clock::time_point request_start;
  std::optional<TextGenerationScheduler::Clock::time_point> deadline;
  std::size_t max_output_bytes{0};
  std::size_t max_buffered_output_bytes{0};
  std::shared_ptr<OutputBudget> output_budget;

  std::atomic<bool> cancellation_requested{false};
  std::atomic<TextRequestPhase> phase{TextRequestPhase::kQueued};

  TextRunnerPool::Request runner_request;
  TextGenerationScheduler::Result result;
  std::optional<TextGenerationScheduler::Clock::time_point> previous_token;
  std::chrono::duration<double, std::milli> inter_token_total{0};
  std::size_t inter_token_samples{0};
  bool decode_due{false};
  std::optional<TextRunnerToken> preview_token;
  bool advance_pending{false};
  bool preview_stops{false};
  std::size_t generated_output_bytes{0};
  StopSequenceFilter stop_filter;

  std::mutex output_mutex;
  std::condition_variable output_condition;
  std::deque<std::string> output_pieces;
  std::size_t buffered_output_bytes{0};
  std::exception_ptr failure;
  bool terminal{false};
};

struct PendingClient {
  std::string client_id;
  std::deque<std::shared_ptr<ScheduledRequest>> requests;
};

bool IsTerminal(const std::shared_ptr<ScheduledRequest>& request) noexcept {
  return request->phase.load(std::memory_order_acquire) ==
         TextRequestPhase::kTerminal;
}

void ReleaseBufferedOutputLocked(
    const std::shared_ptr<ScheduledRequest>& request) noexcept {
  request->output_pieces.clear();
  request->output_budget->Release(request->buffered_output_bytes);
  request->buffered_output_bytes = 0;
}

void PublishTerminal(const std::shared_ptr<ScheduledRequest>& request,
                     std::exception_ptr failure = {},
                     bool discard_pending_output = false) noexcept {
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (discard_pending_output) {
      ReleaseBufferedOutputLocked(request);
    }
    request->failure = std::move(failure);
    request->terminal = true;
    request->phase.store(TextRequestPhase::kTerminal,
                         std::memory_order_release);
  }
  request->output_condition.notify_all();
}

[[nodiscard]] bool PublishPiece(
    const std::shared_ptr<ScheduledRequest>& request, std::string piece) {
  if (!request->publish_token_pieces) {
    return true;
  }
  const std::size_t piece_bytes = piece.size();
  {
    const std::lock_guard<std::mutex> lock(request->output_mutex);
    if (request->buffered_output_bytes > request->max_buffered_output_bytes ||
        piece_bytes > request->max_buffered_output_bytes -
                          request->buffered_output_bytes) {
      return false;
    }
    if (!request->output_budget->TryReserve(piece_bytes)) {
      return false;
    }
    try {
      request->buffered_output_bytes += piece_bytes;
      request->result.max_buffered_output_bytes =
          std::max(request->result.max_buffered_output_bytes,
                   request->buffered_output_bytes);
      request->output_pieces.push_back(std::move(piece));
    } catch (...) {
      request->output_budget->Release(piece_bytes);
      request->buffered_output_bytes -= piece_bytes;
      throw;
    }
  }
  request->output_condition.notify_one();
  return true;
}

bool CancellationRequested(const std::shared_ptr<ScheduledRequest>& request) {
  if (request->cancellation_requested.load(std::memory_order_acquire)) {
    return true;
  }
  return request->external_cancellation && request->external_cancellation();
}

bool DeadlineExceeded(const std::shared_ptr<ScheduledRequest>& request) {
  return request->deadline.has_value() &&
         TextGenerationScheduler::Clock::now() >= *request->deadline;
}

}  // namespace

struct TextGenerationScheduler::Request::Impl {
  explicit Impl(std::shared_ptr<ScheduledRequest> scheduled_request)
      : request(std::move(scheduled_request)) {}

  std::shared_ptr<ScheduledRequest> request;
  bool waited{false};
};

struct TextGenerationScheduler::Impl {
  Impl(std::shared_ptr<TextRunnerPool> model_runner_pool,
       TextPrefillPolicy model_prefill_policy,
       TextSchedulerPolicy model_scheduler_policy)
      : runner_pool(std::move(model_runner_pool)),
        prefill_policy(model_prefill_policy),
        scheduler_policy(model_scheduler_policy),
        output_budget(std::make_shared<OutputBudget>(
            model_scheduler_policy.max_buffered_output_bytes_total)) {
    if (runner_pool == nullptr) {
      throw std::invalid_argument(
          "text generation scheduler runner pool must not be null");
    }
    if (prefill_policy.decode_active_tokens == 0) {
      throw std::invalid_argument(
          "active-decode prefill budget must be at least one token");
    }
    if (scheduler_policy.max_pending_requests == 0 ||
        scheduler_policy.max_pending_requests_per_client == 0 ||
        scheduler_policy.max_pending_requests_per_client >
            scheduler_policy.max_pending_requests ||
        scheduler_policy.max_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_per_request == 0 ||
        scheduler_policy.max_buffered_output_bytes_total == 0 ||
        scheduler_policy.request_timeout.count() < 0) {
      throw std::invalid_argument("invalid text scheduler limits");
    }
    incremental_prefill_supported =
        runner_pool->runner().Descriptor().capabilities.incremental_prefill;
    final_token_advance_required =
        runner_pool->runner()
            .Descriptor()
            .capabilities.final_token_advance_required;
    incremental_text_is_exact = runner_pool->runner()
                                    .Descriptor()
                                    .capabilities.incremental_text_is_exact;
    multi_token_decode =
        runner_pool->runner().Descriptor().capabilities.multi_token_decode;
    batched_multi_token_decode = runner_pool->runner()
                                     .Descriptor()
                                     .capabilities.batched_multi_token_decode;
    batched_multi_token_decode_max_width =
        runner_pool->runner()
            .Descriptor()
            .capabilities.batched_multi_token_decode_max_width;
    worker = std::jthread(
        [this](const std::stop_token& stop_token) { Run(stop_token); });
  }

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      stopping = true;
    }
    worker.request_stop();
    queue_condition.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  [[nodiscard]] std::shared_ptr<ScheduledRequest> PopQueued() {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    if (queued_clients.empty()) {
      return {};
    }
    auto client = std::move(queued_clients.front());
    queued_clients.pop_front();
    auto request = std::move(client.requests.front());
    client.requests.pop_front();
    --queued_count;
    if (!client.requests.empty()) {
      queued_clients.push_back(std::move(client));
    }
    return request;
  }

  [[nodiscard]] bool RemoveQueued(
      const std::shared_ptr<ScheduledRequest>& request) {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    for (auto client = queued_clients.begin(); client != queued_clients.end();
         ++client) {
      const auto queued =
          std::find(client->requests.begin(), client->requests.end(), request);
      if (queued == client->requests.end()) {
        continue;
      }
      client->requests.erase(queued);
      --queued_count;
      if (client->requests.empty()) {
        queued_clients.erase(client);
      }
      return true;
    }
    return false;
  }

  [[nodiscard]] std::vector<std::shared_ptr<ScheduledRequest>> QueuedSnapshot()
      const {
    const std::lock_guard<std::mutex> lock(queue_mutex);
    std::vector<std::shared_ptr<ScheduledRequest>> snapshot;
    snapshot.reserve(queued_count);
    for (const auto& client : queued_clients) {
      snapshot.insert(snapshot.end(), client.requests.begin(),
                      client.requests.end());
    }
    return snapshot;
  }

  void FinalizeResult(const std::shared_ptr<ScheduledRequest>& request,
                      TextGenerationBackend::FinishReason finish_reason) {
    request->result.completion_tokens = request->result.tokens.size();
    request->result.finish_reason = finish_reason;
    if (request->inter_token_samples > 0) {
      request->result.mean_inter_token_ms =
          request->inter_token_total.count() /
          static_cast<double>(request->inter_token_samples);
    }
    if (!incremental_text_is_exact) {
      request->result.text =
          runner_pool->runner().Decode(request->result.tokens);
    }
  }

  void CompleteCancelled(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    try {
      if (request->runner_request) {
        const auto retained = request->runner_request.Cancel();
        request->result.cache_snapshot_bytes = retained.snapshot_bytes;
        request->result.cache_snapshot_ms = retained.snapshot_ms;
        request->result.cache_disk_queued_bytes = retained.disk_queued_bytes;
        request->result.cache_disk_enqueue_ms = retained.disk_enqueue_ms;
      }
      request->result.cancelled = true;
      FinalizeResult(request, TextGenerationBackend::FinishReason::kCancelled);
      PublishTerminal(request, {}, true);
    } catch (...) {
      PublishTerminal(request, std::current_exception(), true);
    }
  }

  void CompleteFailure(const std::shared_ptr<ScheduledRequest>& request,
                       std::exception_ptr failure) noexcept {
    if (request->runner_request) {
      request->runner_request.Invalidate();
    }
    request->result.completion_tokens = request->result.tokens.size();
    PublishTerminal(request, std::move(failure), true);
  }

  void CompleteDeadline(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    CompleteFailure(request, std::make_exception_ptr(TextGenerationError(
                                 TextGenerationErrorCode::kDeadlineExceeded,
                                 "text generation request deadline exceeded")));
  }

  [[nodiscard]] bool CompleteIfStopped(
      const std::shared_ptr<ScheduledRequest>& request) noexcept {
    try {
      if (DeadlineExceeded(request)) {
        CompleteDeadline(request);
        return true;
      }
      if (CancellationRequested(request)) {
        CompleteCancelled(request);
        return true;
      }
      return false;
    } catch (...) {
      CompleteFailure(request, std::current_exception());
      return true;
    }
  }

  void CompleteSuccess(const std::shared_ptr<ScheduledRequest>& request,
                       TextGenerationBackend::FinishReason finish_reason) {
    if (request->stop_filter.enabled()) {
      auto tail = request->stop_filter.Finish();
      request->result.text += tail;
      if (!tail.empty() && !PublishPiece(request, std::move(tail)))
        throw TextGenerationError(
            TextGenerationErrorCode::kOutputBackpressure,
            "text generation buffered output limit exceeded");
    }
    // A stop may land on a preview, an unadvanced AR token, or inside an
    // accepted speculative block. Retain only correctly labelled executed
    // state, using the same cache reconciliation as an interrupted request.
    const auto cache_commit =
        finish_reason == TextGenerationBackend::FinishReason::kStopSequence
            ? request->runner_request.Cancel()
            : request->runner_request.Commit();
    request->result.cache_snapshot_bytes = cache_commit.snapshot_bytes;
    request->result.cache_snapshot_ms = cache_commit.snapshot_ms;
    request->result.cache_disk_queued_bytes = cache_commit.disk_queued_bytes;
    request->result.cache_disk_enqueue_ms = cache_commit.disk_enqueue_ms;
    request->result.cache_shared_prefix_snapshots =
        cache_commit.shared_prefix_snapshots;
    request->result.cache_shared_prefix_bytes =
        cache_commit.shared_prefix_bytes;
    request->result.cache_shared_prefix_ms = cache_commit.shared_prefix_ms;
    FinalizeResult(request, finish_reason);
    PublishTerminal(request);
  }

  void ProcessQueuedCancellations() {
    for (const auto& request : QueuedSnapshot()) {
      try {
        const bool deadline_exceeded = DeadlineExceeded(request);
        const bool cancelled =
            !deadline_exceeded && CancellationRequested(request);
        if (!(deadline_exceeded || cancelled) || !RemoveQueued(request)) {
          continue;
        }
        if (deadline_exceeded) {
          CompleteDeadline(request);
        } else {
          CompleteCancelled(request);
        }
      } catch (...) {
        if (RemoveQueued(request)) {
          CompleteFailure(request, std::current_exception());
        }
      }
    }
  }

  void Admit(std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
             std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
             std::size_t capturing, const std::stop_token& stop_token) {
    // Captures retain their runner lease until decoding resumes.
    while (!stop_token.stop_requested() &&
           prefilling.size() + decoding.size() + capturing <
               runner_pool->capacity()) {
      auto request = PopQueued();
      if (request == nullptr) {
        return;
      }
      try {
        if (CompleteIfStopped(request)) {
          continue;
        }

        const std::weak_ptr<ScheduledRequest> weak_request = request;
        request->runner_request = runner_pool->Acquire(
            std::move(request->prompt), request->sampling,
            [weak_request, stop_token] {
              const auto request = weak_request.lock();
              return stop_token.stop_requested() || request == nullptr ||
                     CancellationRequested(request) ||
                     DeadlineExceeded(request);
            },
            std::move(request->prompt_context), request->cache_prompt,
            request->cache_prefix_tokens, request->stop_at_eos);
        if (!request->runner_request) {
          CompleteCancelled(request);
          continue;
        }

        request->result.cache_hit = request->runner_request.cache_hit();
        const auto lookup = request->runner_request.cache_lookup();
        request->result.cache_miss_reason = lookup.miss_reason;
        request->result.cache_common_prefix_tokens =
            lookup.common_prefix_tokens;
        request->result.cache_checkpoint_tokens = lookup.checkpoint_tokens;
        request->result.cached_prompt_tokens =
            request->runner_request.cached_prompt_tokens();
        request->result.cache_restore_bytes =
            request->runner_request.cache_restore_bytes();
        request->result.cache_restore_ms =
            request->runner_request.cache_restore_ms();
        request->result.cache_disk_hit =
            request->runner_request.cache_disk_hit();
        request->result.incremental_prefill_supported =
            incremental_prefill_supported;
        request->result.queue_ms = std::chrono::duration<double, std::milli>(
                                       Clock::now() - request->request_start)
                                       .count();
        request->result.resident_requests_at_admission =
            prefilling.size() + decoding.size() + capturing + 1;
        request->phase.store(TextRequestPhase::kAdmitted,
                             std::memory_order_release);
        request->phase.store(request->runner_request.prefill_complete()
                                 ? TextRequestPhase::kDecodeReady
                                 : TextRequestPhase::kPrefilling,
                             std::memory_order_release);
        if (request->runner_request.prefill_complete()) {
          request->decode_due = true;
          decoding.push_back(std::move(request));
        } else {
          prefilling.push_back(std::move(request));
        }
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
  }

  void StepPrefill(const std::shared_ptr<ScheduledRequest>& request,
                   bool decoder_runnable, bool snapshot_pending = false) {
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kPrefilling,
                           std::memory_order_release);
      // A published first token is also latency-sensitive while its frozen
      // prompt is being captured. Bound other prefill work so we can poll
      // that capture promptly. Spare slots alone do not change lone prefill.
      const bool bounded_prefill = incremental_prefill_supported &&
                                   (decoder_runnable || snapshot_pending);
      const std::size_t budget = bounded_prefill
                                     ? prefill_policy.decode_active_tokens
                                     : request->runner_request.prompt_tokens();
      if ((decoder_runnable || runner_pool->capacity() > 1) &&
          !incremental_prefill_supported) {
        request->result.prefill_fallback_reason =
            "incremental_prefill_unavailable";
      }
      const auto start = Clock::now();
      const auto step = request->runner_request.Prefill(budget);
      request->result.prefill_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - start)
              .count();
      request->result.prefill_tokens += step.consumed_tokens;
      ++request->result.prefill_chunks;
      request->result.max_prefill_chunk_tokens = std::max(
          request->result.max_prefill_chunk_tokens, step.consumed_tokens);

      if (decoder_runnable) {
        ++request->result.active_decode_prefill_chunks;
        ++consecutive_active_prefill_chunks;
        request->result.max_consecutive_active_prefill_chunks =
            std::max(request->result.max_consecutive_active_prefill_chunks,
                     consecutive_active_prefill_chunks);
      } else {
        consecutive_active_prefill_chunks = 0;
      }

      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(step.decode_ready ? TextRequestPhase::kDecodeReady
                                             : TextRequestPhase::kPrefilling,
                           std::memory_order_release);
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  [[nodiscard]] std::optional<Clock::time_point> PrepareDecode(
      const std::shared_ptr<ScheduledRequest>& request) {
    if (CompleteIfStopped(request)) {
      return std::nullopt;
    }

    if (request->advance_pending) {
      if (!request->runner_request.PreparePromptSnapshot())
        return std::nullopt;
      request->advance_pending = false;
      if (!final_token_advance_required &&
          request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
        return std::nullopt;
      }
      // Snapshot capture and work for other requests are outside this step.
      return Clock::now();
    }
    request->phase.store(TextRequestPhase::kDecoding,
                         std::memory_order_release);
    const auto decode_start = Clock::now();
    const auto selection = request->runner_request.SelectNext();
    if (selection.stop) {
      request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
              .count();
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return std::nullopt;
    }

    if (!PublishSelection(request, selection)) {
      return std::nullopt;
    }
    request->result.decode_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count();
    if (request->stop_filter.stopped()) {
      CompleteSuccess(request,
                      TextGenerationBackend::FinishReason::kStopSequence);
      return std::nullopt;
    }
    request->advance_pending = true;
    return PrepareDecode(request);
  }

  [[nodiscard]] bool PublishSelection(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeSelection& selection) {
    const auto now = Clock::now();
    if (!request->previous_token.has_value()) {
      request->result.ttft_ms = std::chrono::duration<double, std::milli>(
                                    now - request->request_start)
                                    .count();
    } else {
      const auto inter_token = now - *request->previous_token;
      request->inter_token_total += inter_token;
      request->result.max_inter_token_ms = std::max(
          request->result.max_inter_token_ms,
          std::chrono::duration<double, std::milli>(inter_token).count());
      ++request->inter_token_samples;
    }
    request->previous_token = now;
    if (selection.piece.size() >
        request->max_output_bytes - request->generated_output_bytes) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputLimit,
                          "text generation output byte limit exceeded")));
      return false;
    }
    request->generated_output_bytes += selection.piece.size();
    request->result.tokens.push_back(selection.token);
    const auto piece = request->stop_filter.enabled()
                           ? request->stop_filter.Push(selection.piece)
                           : selection.piece;
    if ((!piece.empty() || !request->stop_filter.enabled()) &&
        !PublishPiece(request, piece)) {
      CompleteFailure(request,
                      std::make_exception_ptr(TextGenerationError(
                          TextGenerationErrorCode::kOutputBackpressure,
                          "text generation buffered output limit exceeded")));
      return false;
    }
    if (incremental_text_is_exact) {
      request->result.text += piece;
    }
    if (request->stop_filter.stopped())
      request->result.stop_sequence = request->stop_filter.matched_sequence();
    if (CompleteIfStopped(request)) {
      return false;
    }
    return true;
  }

  void FinishAdvanced(const std::shared_ptr<ScheduledRequest>& request,
                      Clock::time_point decode_start) {
    request->result.decode_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count();
    if (CompleteIfStopped(request)) {
      return;
    }
    if (request->result.tokens.size() >= request->token_limit) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
    }
  }

  void StepDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    if (multi_token_decode) {
      StepMultiTokenDecode(request);
      return;
    }
    try {
      const auto decode_start = PrepareDecode(request);
      if (!decode_start.has_value()) {
        return;
      }
      request->runner_request.Advance();
      FinishAdvanced(request, *decode_start);
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  void StepMultiTokenDecode(const std::shared_ptr<ScheduledRequest>& request) {
    consecutive_active_prefill_chunks = 0;
    try {
      if (CompleteIfStopped(request)) {
        return;
      }

      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      if (!PrepareFirstSnapshot(request))
        return;
      const auto decode_start = Clock::now();
      const std::size_t remaining =
          request->token_limit - request->result.tokens.size() +
          (request->preview_token.has_value() ? 1 : 0);
      const auto step = request->runner_request.DecodeStep(remaining);
      request->result.draft_tokens += step.draft_tokens;
      request->result.draft_accepted_tokens += step.draft_accepted_tokens;

      CheckPreviewResult(request, step);
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(request, selection)) {
          return;
        }
        if (request->stop_filter.stopped())
          break;
      }

      request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
              .count();
      if (request->stop_filter.stopped()) {
        CompleteSuccess(request,
                        TextGenerationBackend::FinishReason::kStopSequence);
        return;
      }
      if (step.stop) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
        return;
      }
      if (request->result.tokens.size() >= request->token_limit) {
        CompleteSuccess(request, TextGenerationBackend::FinishReason::kLength);
      }
    } catch (...) {
      const auto failure = std::current_exception();
      if (!CompleteIfStopped(request)) {
        CompleteFailure(request, failure);
      }
    }
  }

  bool PrepareFirstSnapshot(const std::shared_ptr<ScheduledRequest>& request) {
    if (request->result.tokens.empty() && !request->preview_stops) {
      const auto preview_start = Clock::now();
      const auto preview = request->runner_request.PreviewFirstToken();
      if (preview && !preview->stop) {
        if (!PublishSelection(request, *preview))
          return false;
        request->preview_token = preview->token;
      }
      request->preview_stops = preview && preview->stop;
      request->result.decode_ms += std::chrono::duration<double, std::milli>(
                                       Clock::now() - preview_start)
                                       .count();
    }
    if (request->stop_filter.stopped()) {
      CompleteSuccess(request,
                      TextGenerationBackend::FinishReason::kStopSequence);
      return false;
    }
    if (!request->runner_request.PreparePromptSnapshot())
      return false;
    if (request->preview_stops) {
      CompleteSuccess(request, TextGenerationBackend::FinishReason::kStop);
      return false;
    }
    return !CompleteIfStopped(request);
  }

  static void CheckPreviewResult(
      const std::shared_ptr<ScheduledRequest>& request,
      const TextDecodeStep& step) {
    if (request->preview_token &&
        (step.selections.empty() ||
         step.selections.front().token != *request->preview_token)) {
      throw std::runtime_error("first-token preview disagrees with decoding");
    }
  }

  bool PublishDecodedSelection(const std::shared_ptr<ScheduledRequest>& request,
                               const TextDecodeSelection& selection) {
    if (request->preview_token) {
      request->preview_token.reset();
      return true;
    }
    return PublishSelection(request, selection);
  }

  void StepDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    const auto selected_plan = runner_pool->SelectDecodePlan(requests.size());
    if (batched_multi_token_decode &&
        (batched_multi_token_decode_max_width == 0 ||
         selected_plan.physical_width <=
             batched_multi_token_decode_max_width)) {
      StepMultiTokenDecodeBatch(requests);
      return;
    }
    if (requests.size() == 1) {
      StepDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
      Clock::time_point decode_start;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      try {
        const auto decode_start = PrepareDecode(request);
        if (decode_start.has_value()) {
          prepared.push_back({
              .request = request,
              .decode_start = *decode_start,
          });
        }
      } catch (...) {
        CompleteFailure(request, std::current_exception());
      }
    }
    if (prepared.empty()) {
      return;
    }
    if (prepared.size() == 1) {
      try {
        prepared.front().request->runner_request.Advance();
        FinishAdvanced(prepared.front().request, prepared.front().decode_start);
      } catch (...) {
        CompleteFailure(prepared.front().request, std::current_exception());
      }
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        try {
          item.request->runner_request.Advance();
          FinishAdvanced(item.request, item.decode_start);
        } catch (...) {
          CompleteFailure(item.request, std::current_exception());
        }
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    runner_requests.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
    }
    std::vector<std::exception_ptr> failures;
    try {
      failures = runner_pool->AdvanceBatch(runner_requests, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& item : prepared) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    const std::string execution_plan =
        "batched-w" + std::to_string(plan.physical_width);
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      const auto& item = prepared[i];
      if (failures[i]) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failures[i]);
        continue;
      }
      item.request->result.physical_execution_width = std::max(
          item.request->result.physical_execution_width, plan.physical_width);
      item.request->result.execution_plan = execution_plan;
      FinishAdvanced(item.request, item.decode_start);
    }
  }

  void StepMultiTokenDecodeBatch(
      const std::vector<std::shared_ptr<ScheduledRequest>>& requests) {
    if (requests.size() == 1) {
      StepMultiTokenDecode(requests.front());
      return;
    }

    consecutive_active_prefill_chunks = 0;
    struct PreparedRequest {
      std::shared_ptr<ScheduledRequest> request;
      Clock::time_point decode_start;
      std::size_t remaining;
    };
    std::vector<PreparedRequest> prepared;
    prepared.reserve(requests.size());
    for (const auto& request : requests) {
      if (CompleteIfStopped(request)) {
        continue;
      }
      request->phase.store(TextRequestPhase::kDecoding,
                           std::memory_order_release);
      try {
        if (!PrepareFirstSnapshot(request))
          continue;
      } catch (...) {
        CompleteFailure(request, std::current_exception());
        continue;
      }
      prepared.push_back({
          .request = request,
          .decode_start = Clock::now(),
          .remaining = request->token_limit - request->result.tokens.size() +
                       (request->preview_token.has_value() ? 1 : 0),
      });
    }
    if (prepared.empty()) {
      return;
    }
    if (prepared.size() == 1) {
      StepMultiTokenDecode(prepared.front().request);
      return;
    }

    const auto plan = runner_pool->SelectDecodePlan(prepared.size());
    if (plan.kind != TextExecutionPlanKind::kBatched) {
      for (const auto& item : prepared) {
        StepMultiTokenDecode(item.request);
      }
      return;
    }

    std::vector<TextRunnerPool::Request*> runner_requests;
    std::vector<std::size_t> max_tokens;
    runner_requests.reserve(prepared.size());
    max_tokens.reserve(prepared.size());
    for (const auto& item : prepared) {
      runner_requests.push_back(&item.request->runner_request);
      max_tokens.push_back(item.remaining);
    }

    std::vector<TextDecodeStep> steps;
    try {
      steps = runner_pool->DecodeBatch(runner_requests, max_tokens, plan);
    } catch (...) {
      const auto failure = std::current_exception();
      for (const auto& item : prepared) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, failure);
      }
      return;
    }

    for (std::size_t index = 0; index < prepared.size(); ++index) {
      const auto& item = prepared[index];
      const auto& step = steps[index];
      if (step.failure) {
        if (!CompleteIfStopped(item.request))
          CompleteFailure(item.request, step.failure);
        continue;
      }
      if (step.execution_plan.physical_width >=
          item.request->result.physical_execution_width) {
        item.request->result.physical_execution_width =
            step.execution_plan.physical_width;
        item.request->result.execution_plan =
            step.execution_plan.kind == TextExecutionPlanKind::kBatched
                ? "batched-w" +
                      std::to_string(step.execution_plan.physical_width)
                : (runner_pool->capacity() == 1 ? "serial-c1"
                                                : "serial-fallback");
      }
      item.request->result.draft_tokens += step.draft_tokens;
      item.request->result.draft_accepted_tokens += step.draft_accepted_tokens;

      bool published = true;
      try {
        CheckPreviewResult(item.request, step);
      } catch (...) {
        CompleteFailure(item.request, std::current_exception());
        continue;
      }
      for (const auto& selection : step.selections) {
        if (!PublishDecodedSelection(item.request, selection)) {
          published = false;
          break;
        }
        if (item.request->stop_filter.stopped())
          break;
      }
      item.request->result.decode_ms +=
          std::chrono::duration<double, std::milli>(Clock::now() -
                                                    item.decode_start)
              .count();
      if (!published || IsTerminal(item.request)) {
        continue;
      }
      try {
        if (item.request->stop_filter.stopped()) {
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kStopSequence);
        } else if (step.stop) {
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kStop);
        } else if (item.request->result.tokens.size() >=
                   item.request->token_limit) {
          CompleteSuccess(item.request,
                          TextGenerationBackend::FinishReason::kLength);
        }
      } catch (...) {
        CompleteFailure(item.request, std::current_exception());
      }
    }
  }

  [[nodiscard]] static bool HasDueDecoder(
      const std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    return std::any_of(decoding.begin(), decoding.end(),
                       [](const auto& request) { return request->decode_due; });
  }

  static void MarkAllDecodersDue(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding) {
    for (const auto& request : decoding) {
      request->decode_due = true;
    }
  }

  [[nodiscard]] static std::shared_ptr<ScheduledRequest> PopDecoder(
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding,
      bool require_due) {
    const std::size_t candidates = decoding.size();
    for (std::size_t index = 0; index < candidates; ++index) {
      auto request = std::move(decoding.front());
      decoding.pop_front();
      if (!require_due || request->decode_due) {
        return request;
      }
      decoding.push_back(std::move(request));
    }
    auto request = std::move(decoding.front());
    decoding.pop_front();
    return request;
  }

  void CancelRemaining(
      std::deque<std::shared_ptr<ScheduledRequest>>& prefilling,
      std::deque<std::shared_ptr<ScheduledRequest>>& decoding) noexcept {
    std::vector<std::shared_ptr<ScheduledRequest>> remaining_queued;
    {
      const std::lock_guard<std::mutex> lock(queue_mutex);
      remaining_queued.reserve(queued_count);
      for (auto& client : queued_clients) {
        std::move(client.requests.begin(), client.requests.end(),
                  std::back_inserter(remaining_queued));
      }
      queued_clients.clear();
      queued_count = 0;
    }
    for (const auto& request : remaining_queued) {
      CompleteCancelled(request);
    }
    for (const auto& request : prefilling) {
      CompleteCancelled(request);
    }
    for (const auto& request : decoding) {
      CompleteCancelled(request);
    }
    prefilling.clear();
    decoding.clear();
  }

  void Run(const std::stop_token& stop_token) noexcept {
    std::deque<std::shared_ptr<ScheduledRequest>> prefilling;
    std::deque<std::shared_ptr<ScheduledRequest>> decoding;
    std::deque<std::shared_ptr<ScheduledRequest>> capturing;
    while (!stop_token.stop_requested()) {
      ProcessQueuedCancellations();

      for (std::size_t count = capturing.size(); count != 0; --count) {
        auto request = std::move(capturing.front());
        capturing.pop_front();
        if (request->runner_request.SnapshotPending()) {
          capturing.push_back(std::move(request));
        } else if (!CompleteIfStopped(request)) {
          if (request->runner_request.prefill_complete())
            decoding.push_back(std::move(request));
          else
            prefilling.push_back(std::move(request));
        }
      }
      Admit(prefilling, decoding, capturing.size(), stop_token);
      if (prefilling.empty() && decoding.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        const auto wake = [&] {
          return stop_token.stop_requested() || stopping ||
                 (queued_count != 0 &&
                  capturing.size() < runner_pool->capacity());
        };
        if (capturing.empty())
          queue_condition.wait(lock, wake);
        else
          queue_condition.wait_for(lock, std::chrono::milliseconds(1), wake);
        continue;
      }

      const bool due_decoder = HasDueDecoder(decoding);
      const std::size_t resident_count = prefilling.size() + decoding.size();
      const bool preparing_multi_token_batch =
          multi_token_decode && resident_count > 1 && !prefilling.empty() &&
          runner_pool->SelectDecodePlan(resident_count).kind ==
              TextExecutionPlanKind::kBatched;
      if (preparing_multi_token_batch) {
        for (const auto& pending : prefilling) {
          pending->runner_request.PrepareBatchExecution();
        }
        for (const auto& ready : decoding) {
          ready->runner_request.PrepareBatchExecution();
        }
      }
      // Give simultaneous new requests one bounded chunk to form their first
      // batch. Once decoding starts, every due decoder runs before more
      // prefill.
      const bool assemble_initial_batch =
          preparing_multi_token_batch &&
          consecutive_active_prefill_chunks == 0 &&
          std::all_of(decoding.begin(), decoding.end(),
                      [](const auto& request) {
                        return request->result.tokens.empty();
                      });
      if (!prefilling.empty() && !decoding.empty() &&
          (!due_decoder || assemble_initial_batch)) {
        auto request = std::move(prefilling.front());
        prefilling.pop_front();
        StepPrefill(request, true);
        if (!IsTerminal(request)) {
          if (request->runner_request.SnapshotPending()) {
            capturing.push_back(std::move(request));
          } else if (request->runner_request.prefill_complete()) {
            decoding.push_back(std::move(request));
          } else {
            prefilling.push_back(std::move(request));
          }
        }
        MarkAllDecodersDue(decoding);
        continue;
      }

      if (!decoding.empty()) {
        const std::size_t candidate_count =
            due_decoder
                ? static_cast<std::size_t>(std::count_if(
                      decoding.begin(), decoding.end(),
                      [](const auto& request) { return request->decode_due; }))
                : decoding.size();
        const auto plan = runner_pool->SelectDecodePlan(candidate_count);
        const std::size_t batch_size =
            plan.kind == TextExecutionPlanKind::kBatched
                ? std::min(candidate_count, plan.physical_width)
                : 1;
        std::vector<std::shared_ptr<ScheduledRequest>> batch;
        batch.reserve(batch_size);
        for (std::size_t index = 0; index < batch_size; ++index) {
          auto request = PopDecoder(decoding, due_decoder);
          request->decode_due = false;
          batch.push_back(std::move(request));
        }
        StepDecodeBatch(batch);
        for (auto& request : batch) {
          if (!IsTerminal(request)) {
            if (request->runner_request.SnapshotPending())
              capturing.push_back(std::move(request));
            else
              decoding.push_back(std::move(request));
          }
        }
        continue;
      }

      auto request = std::move(prefilling.front());
      prefilling.pop_front();
      StepPrefill(request, false, !capturing.empty());
      if (!IsTerminal(request)) {
        if (request->runner_request.SnapshotPending()) {
          capturing.push_back(std::move(request));
        } else if (request->runner_request.prefill_complete()) {
          request->decode_due = true;
          decoding.push_back(std::move(request));
        } else {
          prefilling.push_back(std::move(request));
        }
      }
    }
    for (auto& request : capturing)
      decoding.push_back(std::move(request));
    CancelRemaining(prefilling, decoding);
  }

  std::shared_ptr<TextRunnerPool> runner_pool;
  TextPrefillPolicy prefill_policy;
  TextSchedulerPolicy scheduler_policy;
  std::shared_ptr<OutputBudget> output_budget;
  bool incremental_prefill_supported{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool batched_multi_token_decode{false};
  std::size_t batched_multi_token_decode_max_width{0};
  mutable std::mutex queue_mutex;
  std::condition_variable queue_condition;
  std::deque<PendingClient> queued_clients;
  std::size_t queued_count{0};
  bool stopping{false};
  std::size_t consecutive_active_prefill_chunks{0};
  std::atomic<std::uint64_t> next_request_id{1};
  std::jthread worker;
};

TextGenerationScheduler::Request::Request() = default;

TextGenerationScheduler::Request::Request(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TextGenerationScheduler::Request::~Request() {
  Cancel();
}

TextGenerationScheduler::Request::Request(Request&&) noexcept = default;

TextGenerationScheduler::Request& TextGenerationScheduler::Request::operator=(
    Request&& other) noexcept {
  if (this != &other) {
    Cancel();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

TextGenerationScheduler::Request::operator bool() const noexcept {
  return impl_ != nullptr && impl_->request != nullptr;
}

std::uint64_t TextGenerationScheduler::Request::id() const noexcept {
  return *this ? impl_->request->id : 0;
}

TextRequestPhase TextGenerationScheduler::Request::phase() const noexcept {
  return *this ? impl_->request->phase.load(std::memory_order_acquire)
               : TextRequestPhase::kTerminal;
}

TextGenerationScheduler::Result TextGenerationScheduler::Request::Wait(
    const TokenCallback& on_token) {
  if (!*this) {
    throw std::logic_error("text scheduler request is empty");
  }
  if (impl_->waited) {
    throw std::logic_error("text scheduler request was already consumed");
  }
  impl_->waited = true;

  bool deliver_pieces = true;
  bool consumer_cancelled = false;
  std::exception_ptr callback_failure;
  Result result;
  std::exception_ptr scheduler_failure;

  while (true) {
    std::string piece;
    bool has_piece = false;
    bool terminal = false;
    {
      std::unique_lock<std::mutex> lock(impl_->request->output_mutex);
      impl_->request->output_condition.wait(lock, [&] {
        return impl_->request->terminal ||
               !impl_->request->output_pieces.empty();
      });
      if (!impl_->request->output_pieces.empty()) {
        has_piece = true;
        const std::size_t piece_bytes =
            impl_->request->output_pieces.front().size();
        piece = std::move(impl_->request->output_pieces.front());
        impl_->request->output_pieces.pop_front();
        impl_->request->buffered_output_bytes -= piece_bytes;
        impl_->request->output_budget->Release(piece_bytes);
      } else if (impl_->request->terminal) {
        result = impl_->request->result;
        scheduler_failure = impl_->request->failure;
        terminal = true;
      }
    }

    if (has_piece && deliver_pieces && on_token) {
      try {
        if (!on_token(piece)) {
          consumer_cancelled = true;
          deliver_pieces = false;
          Cancel();
        }
      } catch (...) {
        callback_failure = std::current_exception();
        deliver_pieces = false;
        Cancel();
      }
    }
    if (terminal) {
      break;
    }
  }

  if (callback_failure != nullptr) {
    std::rethrow_exception(callback_failure);
  }
  if (scheduler_failure != nullptr) {
    std::rethrow_exception(scheduler_failure);
  }
  if (consumer_cancelled) {
    result.cancelled = true;
    result.finish_reason = TextGenerationBackend::FinishReason::kCancelled;
  }
  return result;
}

void TextGenerationScheduler::Request::Cancel() noexcept {
  if (*this && !IsTerminal(impl_->request)) {
    impl_->request->cancellation_requested.store(true,
                                                 std::memory_order_release);
  }
}

TextGenerationScheduler::TextGenerationScheduler(
    std::shared_ptr<TextRunnerPool> runner_pool,
    TextPrefillPolicy prefill_policy, TextSchedulerPolicy scheduler_policy)
    : impl_(std::make_unique<Impl>(std::move(runner_pool), prefill_policy,
                                   scheduler_policy)) {}

TextGenerationScheduler::~TextGenerationScheduler() = default;

const TextModelRunner& TextGenerationScheduler::runner() const noexcept {
  return impl_->runner_pool->runner();
}

std::size_t TextGenerationScheduler::capacity() const noexcept {
  return impl_->runner_pool->capacity();
}

std::size_t TextGenerationScheduler::buffered_output_bytes() const noexcept {
  return impl_->output_budget->buffered_bytes.load(std::memory_order_relaxed);
}

std::size_t TextGenerationScheduler::max_buffered_output_bytes()
    const noexcept {
  return impl_->output_budget->max_buffered_bytes.load(
      std::memory_order_relaxed);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces) {
  return Submit(std::move(prompt), max_tokens, sampling, is_cancelled,
                publish_token_pieces, RequestMetadata{});
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    const sampling::SamplingConfig& sampling,
    const CancellationCheck& is_cancelled, bool publish_token_pieces,
    RequestMetadata metadata) {
  if (prompt.empty()) {
    throw std::invalid_argument("text scheduler prompt must not be empty");
  }
  const auto context = impl_->runner_pool->runner().Descriptor().max_context;
  if (prompt.size() >= context) {
    throw std::length_error("prompt has " + std::to_string(prompt.size()) +
                            " tokens but the context is " +
                            std::to_string(context) +
                            "; increase --context or shorten the conversation");
  }
  const std::size_t available = context - prompt.size();
  sampling.Validate();
  ValidateStopSequences(metadata.stop_sequences);
  if (!metadata.stop_sequences.empty() && !impl_->incremental_text_is_exact)
    throw std::invalid_argument(
        "stop sequences require exact incremental token decoding");

  auto request = std::make_shared<ScheduledRequest>();
  request->stop_filter = StopSequenceFilter(std::move(metadata.stop_sequences));
  request->id = impl_->next_request_id.fetch_add(1, std::memory_order_relaxed);
  request->client_id =
      metadata.client_id.empty() ? "anonymous" : std::move(metadata.client_id);
  request->result.prompt_tokens = prompt.size();
  request->result.client_id = request->client_id;
  request->result.configured_active_prefill_tokens =
      impl_->prefill_policy.decode_active_tokens;
  request->result.requested_logical_concurrency =
      impl_->runner_pool->capacity();
  request->result.execution_plan =
      impl_->runner_pool->capacity() == 1 ? "serial-c1" : "serial-fallback";
  request->prompt = std::move(prompt);
  request->prompt_context = std::move(metadata.prompt_context);
  request->cache_prompt = metadata.cache_prompt;
  request->cache_prefix_tokens = metadata.cache_prefix_tokens;
  request->stop_at_eos = metadata.stop_at_eos;
  request->token_limit =
      max_tokens > 0 ? std::min(max_tokens, available) : available;
  request->sampling = sampling;
  request->external_cancellation = is_cancelled;
  request->publish_token_pieces = publish_token_pieces;
  request->request_start = metadata.request_start;
  request->deadline = metadata.deadline;
  if (!request->deadline.has_value() &&
      impl_->scheduler_policy.request_timeout.count() > 0) {
    request->deadline =
        request->request_start + impl_->scheduler_policy.request_timeout;
  }
  request->max_output_bytes =
      impl_->scheduler_policy.max_output_bytes_per_request;
  request->max_buffered_output_bytes =
      impl_->scheduler_policy.max_buffered_output_bytes_per_request;
  request->output_budget = impl_->output_budget;

  {
    const std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    if (impl_->stopping) {
      throw TextGenerationError(TextGenerationErrorCode::kSchedulerStopping,
                                "text generation scheduler is stopping");
    }
    if (impl_->queued_count >= impl_->scheduler_policy.max_pending_requests) {
      throw TextGenerationError(TextGenerationErrorCode::kQueueFull,
                                "text generation pending queue is full");
    }
    auto client =
        std::find_if(impl_->queued_clients.begin(), impl_->queued_clients.end(),
                     [&](const PendingClient& pending) {
                       return pending.client_id == request->client_id;
                     });
    if (client != impl_->queued_clients.end() &&
        client->requests.size() >=
            impl_->scheduler_policy.max_pending_requests_per_client) {
      throw TextGenerationError(TextGenerationErrorCode::kClientQueueFull,
                                "text generation client pending queue is full");
    }
    if (client == impl_->queued_clients.end()) {
      impl_->queued_clients.push_back({
          .client_id = request->client_id,
          .requests = {},
      });
      client = std::prev(impl_->queued_clients.end());
    }
    request->result.queue_depth_at_submit = impl_->queued_count + 1;
    request->result.client_queue_depth_at_submit = client->requests.size() + 1;
    client->requests.push_back(request);
    ++impl_->queued_count;
  }
  impl_->queue_condition.notify_one();
  return Request(std::make_unique<Request::Impl>(std::move(request)));
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces);
}

TextGenerationScheduler::Request TextGenerationScheduler::Submit(
    std::vector<TextRunnerToken> prompt, std::size_t max_tokens,
    float temperature, const CancellationCheck& is_cancelled,
    bool publish_token_pieces, RequestMetadata metadata) {
  sampling::SamplingConfig config;
  config.temperature = temperature;
  return Submit(std::move(prompt), max_tokens, config, is_cancelled,
                publish_token_pieces, std::move(metadata));
}

}  // namespace gufo::server
