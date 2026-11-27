#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"
#include "src/core/metrics.hpp"

namespace gufo::server {

/// Schedulers add their own deltas to the request gauges, so a draining
/// scheduler and its replacement can both contribute during a model reload.
struct TextMetrics {
  TextMetrics();

  core::metrics::Registry registry;
  core::metrics::Counter device_lost;
  core::metrics::Counter prompt_tokens;
  core::metrics::Counter cached_prompt_tokens;
  core::metrics::FloatCounter prompt_seconds;
  core::metrics::Counter gen_tokens;
  core::metrics::FloatCounter gen_seconds;
  core::metrics::MaxCounter max_sequence_tokens;
  core::metrics::Counter draft_rounds;
  core::metrics::Counter draft_tokens;
  core::metrics::Counter draft_accepted_tokens;
  core::metrics::Counter lookup_tokens;
  core::metrics::Counter lookup_accepted_tokens;
  core::metrics::Gauge last_prompt_speed;
  core::metrics::Gauge last_gen_speed;
  core::metrics::IntGauge requests_processing;
  core::metrics::IntGauge requests_deferred;
  core::metrics::Gauge kv_cache_usage_ratio;
  core::metrics::Histogram queue_seconds;
  core::metrics::Histogram first_token_seconds;

  struct SharedState {
    core::metrics::UintGauge bytes;
    core::metrics::UintGauge charged_bytes;
    core::metrics::UintGauge reclaimable_bytes;
    core::metrics::UintGauge max_prompt_tokens;
    core::metrics::Counter waits;
    core::metrics::Counter pauses;
    core::metrics::Counter preemptions;
    std::vector<core::metrics::Visibility> families;

    void Show(bool visible) const {
      for (const auto& family : families)
        family.Set(visible);
    }
  } shared_state;
  struct PromptState {
    core::metrics::UintGauge budget_bytes;
    core::metrics::UintGauge reserved_bytes;
    core::metrics::UintGauge available_bytes;
    std::vector<core::metrics::Visibility> families;

    void Show(bool visible) const {
      for (const auto& family : families)
        family.Set(visible);
    }
  } prompt_state;
};

inline TextMetrics::TextMetrics() {
  using namespace core::metrics;
  device_lost = registry.Add<Counter>("gufo_device_lost_total",
                                      "Confirmed GPU context losses");
  prompt_tokens = registry.Add<Counter>(
      "llamacpp:prompt_tokens_total",
      "Total prompt tokens processed, excluding cache hits");
  cached_prompt_tokens =
      registry.Add<Counter>("llamacpp:prompt_tokens_cached_total",
                            "Total prompt tokens reused from cache");
  prompt_seconds = registry.Add<FloatCounter>("llamacpp:prompt_seconds_total",
                                              "Prompt process time");
  gen_tokens = registry.Add<Counter>("llamacpp:tokens_predicted_total",
                                     "Total tokens generated");
  gen_seconds = registry.Add<FloatCounter>(
      "llamacpp:tokens_predicted_seconds_total", "Predict process time");
  max_sequence_tokens = registry.Add<MaxCounter>("llamacpp:n_tokens_max",
                                                 "Largest observed n_tokens.");
  draft_rounds = registry.Add<Counter>("llamacpp:spec_decode_num_drafts_total",
                                       "Total speculative verification rounds");
  draft_tokens =
      registry.Add<Counter>("llamacpp:spec_decode_num_draft_tokens_total",
                            "Total draft tokens proposed");
  draft_accepted_tokens =
      registry.Add<Counter>("llamacpp:spec_decode_num_accepted_tokens_total",
                            "Total draft tokens accepted");
  lookup_tokens = registry.Add<Counter>("gufo_spec_lookup_tokens_total",
                                        "Total prompt-lookup copies proposed");
  lookup_accepted_tokens =
      registry.Add<Counter>("gufo_spec_lookup_accepted_tokens_total",
                            "Total prompt-lookup copies accepted");
  last_prompt_speed =
      registry.Add<Gauge>("llamacpp:prompt_tokens_seconds",
                          "Prompt processing speed in tokens per second");
  last_gen_speed = registry.Add<Gauge>("llamacpp:predicted_tokens_seconds",
                                       "Generation speed in tokens per second");
  requests_processing = registry.Add<IntGauge>(
      "llamacpp:requests_processing",
      "Number of admitted requests, including cache preparation");
  requests_deferred = registry.Add<IntGauge>(
      "llamacpp:requests_deferred", "Number of requests waiting for a session");
  kv_cache_usage_ratio = registry.Add<Gauge>(
      "llamacpp:kv_cache_usage_ratio",
      "In-flight prompt and generated tokens over sessions times context; "
      "excludes retained cache");
  // Prompts can wait or prefill for minutes, so the range reaches 20 minutes.
  static constexpr std::array<double, 16> kRequestLatencySeconds = {
      0.001, 0.01, 0.05, 0.1,  0.25,  0.5,   1.0,   2.5,
      5.0,   10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0};
  queue_seconds = registry.Add<Histogram>(
      "gufo_request_queue_seconds",
      "Time from request arrival until a session admits it, including cache "
      "preparation; requests that end before admission are not counted",
      kRequestLatencySeconds);
  first_token_seconds = registry.Add<Histogram>(
      "gufo_time_to_first_token_seconds",
      "Time from request arrival until the first generated token; requests "
      "that end before one are not counted",
      kRequestLatencySeconds);

  const auto optional = [this](std::vector<Visibility>& group, auto& metric,
                               std::string_view name, std::string_view help) {
    using Metric = std::remove_reference_t<decltype(metric)>;
    const auto family = registry.AddFamily<Metric>(name, help);
    metric = family.Add();
    group.push_back(family.visibility());
  };
  optional(shared_state.families, shared_state.bytes, "gufo_shared_state_bytes",
           "Session state the configured sessions share when they do not all "
           "fit at full context");
  optional(shared_state.families, shared_state.charged_bytes,
           "gufo_shared_state_charged_bytes",
           "Shared session state charged to sessions");
  optional(shared_state.families, shared_state.reclaimable_bytes,
           "gufo_shared_state_reclaimable_bytes",
           "Charged session state held by idle sessions or mapped past a "
           "session's need");
  optional(shared_state.families, shared_state.max_prompt_tokens,
           "gufo_shared_state_max_prompt_tokens",
           "Longest prompt the shared session state holds");
  optional(shared_state.families, shared_state.waits,
           "gufo_shared_state_waits_total",
           "Requests that waited for shared session state");
  optional(shared_state.families, shared_state.pauses,
           "gufo_shared_state_pauses_total",
           "Decoding requests paused until shared session state freed");
  optional(shared_state.families, shared_state.preemptions,
           "gufo_shared_state_preemptions_total",
           "Requests stopped with memory_pressure so others could continue");
  optional(prompt_state.families, prompt_state.budget_bytes,
           "gufo_prompt_state_budget_bytes",
           "Host RAM prompt state may claim before admission refuses");
  optional(prompt_state.families, prompt_state.reserved_bytes,
           "gufo_prompt_state_reserved_bytes",
           "Host prompt state charged to in-flight requests");
  optional(prompt_state.families, prompt_state.available_bytes,
           "gufo_prompt_state_available_bytes",
           "Remaining host prompt state budget");
  shared_state.Show(false);
  prompt_state.Show(false);
}

/// Never destroyed, so late updates during process shutdown stay valid.
inline TextMetrics& Metrics() {
  static TextMetrics* const metrics = new TextMetrics;
  return *metrics;
}

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
inline void RecordRequestMetrics(const TextGenerationBackend::Result& result) {
  TextMetrics& metrics = Metrics();
  metrics.cached_prompt_tokens.Add(
      std::min(result.cached_prompt_tokens, result.prompt_tokens));
  metrics.prompt_seconds.Add(result.prefill_ms / 1000.0);
  metrics.gen_seconds.Add(result.decode_ms / 1000.0);
  metrics.max_sequence_tokens.Observe(result.prompt_tokens +
                                      result.completion_tokens);
  metrics.draft_rounds.Add(result.draft_rounds);
  metrics.draft_tokens.Add(result.draft_tokens);
  metrics.draft_accepted_tokens.Add(result.draft_accepted_tokens);
  metrics.lookup_tokens.Add(result.lookup_tokens);
  metrics.lookup_accepted_tokens.Add(result.lookup_accepted_tokens);
  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  if (prompt_per_second > 0.0)
    metrics.last_prompt_speed.Set(prompt_per_second);
  if (tok_per_sec > 0.0)
    metrics.last_gen_speed.Set(tok_per_sec);
}

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
          : result.finish_reason ==
                  TextGenerationBackend::FinishReason::kStopSequence
              ? "stop_sequence"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " queue_ms=" << result.queue_ms
      << " shared_prefix_wait_ms=" << result.shared_prefix_wait_ms
      << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens
      << " draft_rounds=" << result.draft_rounds;
  if (result.component_cache)
    out << " cache_schema=component-cache-v1 cache_reuse="
        << result.component_cache->reuse
        << " cache_captures=" << result.component_cache->captures
        << " cache_captured_capacity_bytes="
        << result.component_cache->captured_bytes
        << " cache_global_ledger_bytes="
        << result.component_cache->ledger.total_bytes;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
  if (result.lookup_tokens > 0)
    out << " lookup_accepted=" << result.lookup_accepted_tokens
        << " lookup_proposed=" << result.lookup_tokens;
  if (result.ngram_rows > 0)
    out << " ngram_rows=" << result.ngram_rows
        << " ngram_dedup_rows=" << result.ngram_dedup_rows
        << " ngram_cache_hits=" << result.ngram_cache_hits
        << " ngram_disk_reads=" << result.ngram_disk_reads;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_rounds"] = result.draft_rounds;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  if (result.component_cache)
    timings["component_cache"] = ComponentCacheJson(*result.component_cache);
  if (result.lookup_tokens > 0) {
    timings["lookup_n"] = result.lookup_tokens;
    timings["lookup_n_accepted"] = result.lookup_accepted_tokens;
  }
  return timings;
}

/// llama-server `prompt_progress` object for streamed `return_progress`.
inline json::Value PromptProgressJson(
    const TextGenerationBackend::PromptProgress& progress) {
  json::Value value = json::Value::object();
  value["total"] = progress.total;
  value["cache"] = progress.cache;
  value["processed"] = progress.processed;
  value["time_ms"] = progress.time_ms;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_
