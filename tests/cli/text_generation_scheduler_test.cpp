#include "src/cli/serve/text_generation_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextDecodeStep;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextGenerationError;
using gufo::server::TextGenerationErrorCode;
using gufo::server::TextGenerationScheduler;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillPolicy;
using gufo::server::TextPrefillStep;
using gufo::server::TextRequestMetadata;
using gufo::server::TextRequestPhase;
using gufo::server::TextRunnerAdvance;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;
using gufo::server::TextSchedulerPolicy;

constexpr auto kTestTimeout = std::chrono::seconds{5};

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

enum class EventKind : std::uint8_t {
  kPrefill,
  kAdvance,
};

struct Event {
  EventKind kind;
  TextRunnerToken label;
  std::size_t index;
  std::size_t count;
};

struct FakeControl {
  void Log(Event event) {
    const std::lock_guard<std::mutex> lock(mutex);
    events.push_back(event);
    condition.notify_all();
  }

  void WaitForAdvance(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return advance_gate_entered && advance_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked decode advance");
  }

  void WaitForPrefill(TextRunnerToken label) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool entered = condition.wait_for(lock, kTestTimeout, [&] {
      return prefill_gate_entered && prefill_gate_label == label;
    });
    Expect(entered, "timed out waiting for blocked prefill");
  }

  void ReleaseAdvance() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_advance = true;
    }
    condition.notify_all();
  }

  void ReleasePrefill() {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      release_prefill = true;
    }
    condition.notify_all();
  }

  void RecordInvalidation() {
    invalidations.fetch_add(1, std::memory_order_relaxed);
    condition.notify_all();
  }

  void WaitForInvalidations(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex);
    const bool reached = condition.wait_for(lock, kTestTimeout, [&] {
      return invalidations.load(std::memory_order_relaxed) >= count;
    });
    Expect(reached, "timed out waiting for state invalidation");
  }

  [[nodiscard]] std::vector<Event> Events() const {
    const std::lock_guard<std::mutex> lock(mutex);
    return events;
  }

  [[nodiscard]] std::vector<std::vector<TextRunnerToken>> AdvanceBatches()
      const {
    const std::lock_guard<std::mutex> lock(mutex);
    return advance_batches;
  }

  mutable std::mutex mutex;
  std::condition_variable condition;
  std::vector<Event> events;
  std::vector<std::vector<TextRunnerToken>> advance_batches;
  std::optional<TextRunnerToken> block_advance_label;
  std::optional<TextRunnerToken> block_prefill_label;
  std::optional<TextRunnerToken> throw_advance_label;
  TextRunnerToken advance_gate_label{0};
  TextRunnerToken prefill_gate_label{0};
  bool advance_gate_entered{false};
  bool prefill_gate_entered{false};
  bool release_advance{false};
  bool release_prefill{false};
  std::atomic<std::size_t> invalidations{0};
  std::atomic<std::size_t> states_created{0};
  std::atomic<std::size_t> decode_calls{0};
  bool incremental_prefill{true};
  bool supports_batched_advance{false};
  bool final_token_advance_required{true};
  bool incremental_text_is_exact{false};
  bool multi_token_decode{false};
  bool prefix_reuse{true};
};

class FakeState final : public TextRunnerState {
public:
  explicit FakeState(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  void Invalidate() noexcept override {
    control_->RecordInvalidation();
    label = 0;
    position = 0;
    decode_count = 0;
    frontier.reset();
  }

  std::shared_ptr<FakeControl> control_;
  TextRunnerToken label{0};
  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected scheduler fake state");
  }
  return *fake;
}

class FakeRunner final : public TextModelRunner {
public:
  explicit FakeRunner(std::shared_ptr<FakeControl> control)
      : control_(std::move(control)) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "scheduler-fake",
        .state_abi = "scheduler-fake-v1",
        .max_context = 128,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = control_->incremental_prefill,
                .final_token_advance_required =
                    control_->final_token_advance_required,
                .incremental_text_is_exact =
                    control_->incremental_text_is_exact,
                .multi_token_decode = control_->multi_token_decode,
                .prefix_reuse = control_->prefix_reuse,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = 0,
        .state_capacity_bytes = 8 * 64,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = 0,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{
        .kind = TextExecutionPlanKind::kSerial,
        .physical_width = 1,
    }};
    if (control_->supports_batched_advance) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 2,
      });
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = 4,
      });
    }
    return plans;
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return {static_cast<TextRunnerToken>(text.size())};
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::nullopt;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    control_->decode_calls.fetch_add(1, std::memory_order_relaxed);
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (!text.empty()) {
        text.push_back(',');
      }
      text += std::to_string(token);
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    control_->states_created.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<FakeState>(control_);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& fake = RequireFakeState(state);
    if (offset != fake.position || prompt.empty()) {
      throw std::logic_error("invalid scheduler fake prefill");
    }
    fake.label = prompt.front();
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      control_->events.push_back({
          .kind = EventKind::kPrefill,
          .label = fake.label,
          .index = fake.position,
          .count = consumed,
      });
      if (control_->block_prefill_label == fake.label &&
          !control_->prefill_gate_entered) {
        control_->prefill_gate_entered = true;
        control_->prefill_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_prefill; });
        if (!released) {
          throw std::runtime_error("prefill gate timed out");
        }
      }
    }

    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = fake.label * 100;
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, gufo::sampling::SamplerState&) const override {
    const auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no frontier");
    }
    return {
        .stop = false,
        .token = *fake.frontier,
        .piece = std::to_string(*fake.frontier),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value() || token != *fake.frontier) {
      throw std::logic_error("scheduler fake frontier mismatch");
    }

    {
      std::unique_lock<std::mutex> lock(control_->mutex);
      if (control_->block_advance_label == fake.label &&
          fake.decode_count == 0 && !control_->advance_gate_entered) {
        control_->advance_gate_entered = true;
        control_->advance_gate_label = fake.label;
        control_->condition.notify_all();
        const bool released = control_->condition.wait_for(
            lock, kTestTimeout, [&] { return control_->release_advance; });
        if (!released) {
          throw std::runtime_error("advance gate timed out");
        }
      }
      if (control_->throw_advance_label == fake.label) {
        throw std::runtime_error("injected scheduler runner failure");
      }
      control_->events.push_back({
          .kind = EventKind::kAdvance,
          .label = fake.label,
          .index = fake.decode_count,
          .count = 1,
      });
    }

    ++fake.position;
    ++fake.decode_count;
    fake.frontier = token + 1;
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      gufo::sampling::SamplerState& sampler) const override {
    if (!control_->multi_token_decode) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("scheduler fake has no multi-token frontier");
    }
    const std::size_t count = std::min<std::size_t>(max_tokens, 3);
    TextDecodeStep step;
    step.selections.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      const TextRunnerToken token =
          *fake.frontier + static_cast<TextRunnerToken>(index);
      step.selections.push_back({
          .stop = false,
          .token = token,
          .piece = std::to_string(token),
      });
    }
    step.draft_tokens = count + 1;
    step.draft_accepted_tokens = count;
    fake.position += count;
    fake.decode_count += count;
    fake.frontier = *fake.frontier + static_cast<TextRunnerToken>(count);
    return step;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    std::vector<TextRunnerToken> labels;
    labels.reserve(advances.size());
    for (const auto& advance : advances) {
      labels.push_back(RequireFakeState(advance.state.get()).label);
    }
    {
      const std::lock_guard<std::mutex> lock(control_->mutex);
      control_->advance_batches.push_back(std::move(labels));
    }
    for (const auto& advance : advances) {
      Advance(advance.state.get(), advance.token);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

private:
  std::shared_ptr<FakeControl> control_;
};

std::unique_ptr<TextGenerationScheduler> MakeScheduler(
    const std::shared_ptr<FakeControl>& control, std::size_t capacity,
    TextPrefillPolicy prefill_policy = {},
    TextSchedulerPolicy scheduler_policy = {}) {
  auto runner = std::make_shared<FakeRunner>(control);
  auto pool = std::make_shared<TextRunnerPool>(std::move(runner), capacity);
  return std::make_unique<TextGenerationScheduler>(
      std::move(pool), prefill_policy, scheduler_policy);
}

std::size_t EventIndex(std::span<const Event> events, EventKind kind,
                       TextRunnerToken label, std::size_t occurrence = 0) {
  std::size_t seen = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind == kind && events[index].label == label) {
      if (seen == occurrence) {
        return index;
      }
      ++seen;
    }
  }
  return events.size();
}

std::vector<TextRunnerToken> ExpectedTokens(TextRunnerToken label,
                                            std::size_t count) {
  std::vector<TextRunnerToken> tokens;
  tokens.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens.push_back(label * 100 + static_cast<TextRunnerToken>(index));
  }
  return tokens;
}

TextRequestMetadata ClientMetadata(std::string client_id) {
  return {
      .client_id = std::move(client_id),
      .deadline = std::nullopt,
      .request_start = TextGenerationScheduler::Clock::now(),
  };
}

void TestIdlePrefillUsesBulkWorkUnit() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 1, {.decode_active_tokens = 2});

  const auto result = scheduler->Submit({7, 70, 71, 72, 73}, 2, 0.0F).Wait();

  const auto events = control->Events();
  Expect(events.front().kind == EventKind::kPrefill &&
             events.front().label == 7 && events.front().count == 5,
         "decode-idle prefill consumes the complete prompt");
  Expect(result.prefill_chunks == 1 && result.prefill_tokens == 5,
         "decode-idle prefill metrics report one bulk work unit");
  Expect(result.active_decode_prefill_chunks == 0,
         "decode-idle prefill is not counted as active-decode work");
  Expect(result.requested_logical_concurrency == 1 &&
             result.physical_execution_width == 1 &&
             result.execution_plan == "serial-c1",
         "C=1 telemetry reports immediate serial dispatch");
}

void TestRunnerCanSkipUnusedFinalAdvance() {
  auto control = std::make_shared<FakeControl>();
  control->final_token_advance_required = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.tokens == ExpectedTokens(7, 2),
         "skipped final advance preserves emitted tokens");

  std::size_t advances = 0;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kAdvance && event.label == 7) {
      ++advances;
    }
  }
  Expect(advances == 1,
         "runner skips the unused frontier computation after the final token");
}

void TestRunnerCanReuseExactIncrementalText() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_text_is_exact = true;
  auto scheduler = MakeScheduler(control, 1);

  const auto result = scheduler->Submit({7}, 2, 0.0F).Wait();
  Expect(result.text == "700701",
         "exact incremental pieces form the final response text");
  Expect(control->decode_calls.load(std::memory_order_relaxed) == 0,
         "exact incremental text avoids duplicate final decoding");
}

void TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->multi_token_decode = true;
  control->prefix_reuse = false;
  auto scheduler = MakeScheduler(control, 1);

  const auto first = scheduler->Submit({7}, 5, 0.0F).Wait();
  Expect(first.tokens == ExpectedTokens(7, 5),
         "multi-token decode preserves the generated trajectory");
  Expect(first.draft_tokens == 7 && first.draft_accepted_tokens == 5,
         "multi-token decode reports accumulated draft statistics");
  Expect(!first.cache_hit,
         "multi-token state starts without continuation reuse");
  control->WaitForInvalidations(1);

  const auto second = scheduler->Submit({7, 70}, 2, 0.0F).Wait();
  Expect(!second.cache_hit,
         "runner-disabled prefix reuse cannot retain speculative state");
}

void TestMultiResidentPrefillUsesBoundedWorkUnits() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  const auto result =
      scheduler->Submit({7, 70, 71, 72, 73, 74, 75}, 2, 0.0F).Wait();

  std::size_t chunks = 0;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kPrefill && event.label == 7) {
      Expect(event.count <= 2,
             "multi-resident prefill respects the fairness budget");
      ++chunks;
    }
  }
  Expect(chunks == 4 && result.prefill_chunks == 4 &&
             result.active_decode_prefill_chunks == 0,
         "multi-resident prefill yields between bounded work units");
}

void TestDecodeActivePrefillIsBounded() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  auto request_a = scheduler->Submit({1}, 10, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23, 24, 25}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 10),
         "active decoder preserves its isolated trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "chunked prefill preserves the new request trajectory");

  const auto events = control->Events();
  std::size_t previous_b_prefill = events.size();
  std::size_t b_prefill_chunks = 0;
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 2) {
      continue;
    }
    Expect(events[index].count <= 2,
           "active-decode prefill respects its token budget");
    if (previous_b_prefill != events.size()) {
      bool a_advanced = false;
      for (std::size_t between = previous_b_prefill + 1; between < index;
           ++between) {
        a_advanced =
            a_advanced || (events[between].kind == EventKind::kAdvance &&
                           events[between].label == 1);
      }
      Expect(a_advanced, "an active decoder advances between prefill chunks");
    }
    previous_b_prefill = index;
    ++b_prefill_chunks;
  }
  Expect(b_prefill_chunks == 4,
         "long active-decode prompt is split into bounded chunks");
  Expect(result_b.prefill_chunks == 4 && result_b.prefill_tokens == 7 &&
             result_b.active_decode_prefill_chunks == 4 &&
             result_b.max_prefill_chunk_tokens == 2 &&
             result_b.max_consecutive_active_prefill_chunks == 1,
         "chunk metrics capture the selected active-decode policy");
}

void TestPrefillYieldsToEveryDueDecoder() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 2;
  auto scheduler = MakeScheduler(control, 3, {.decode_active_tokens = 2});

  auto request_a = scheduler->Submit({1}, 10, 0.0F);
  auto request_b = scheduler->Submit({2}, 10, 0.0F);
  control->WaitForAdvance(2);
  auto request_c = scheduler->Submit({3, 30, 31, 32, 33, 34, 35}, 2, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 10) &&
             result_b.tokens == ExpectedTokens(2, 10) &&
             result_c.tokens == ExpectedTokens(3, 2),
         "all mixed prefill/decode trajectories remain isolated");

  const auto events = control->Events();
  std::size_t previous_c_prefill = events.size();
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].kind != EventKind::kPrefill || events[index].label != 3) {
      continue;
    }
    if (previous_c_prefill != events.size()) {
      bool a_advanced = false;
      bool b_advanced = false;
      for (std::size_t between = previous_c_prefill + 1; between < index;
           ++between) {
        if (events[between].kind == EventKind::kAdvance) {
          a_advanced = a_advanced || events[between].label == 1;
          b_advanced = b_advanced || events[between].label == 2;
        }
      }
      Expect(a_advanced && b_advanced,
             "every due decoder advances before another prefill chunk");
    }
    previous_c_prefill = index;
  }
  Expect(result_c.max_consecutive_active_prefill_chunks == 1,
         "scheduler never runs consecutive chunks while decode is due");
}

void TestNonIncrementalRunnerFallsBackSafely() {
  auto control = std::make_shared<FakeControl>();
  control->incremental_prefill = false;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20, 21, 22, 23}, 2, 0.0F);
  control->ReleaseAdvance();

  Expect(request_a.Wait().tokens == ExpectedTokens(1, 8),
         "fallback preserves the active decoder trajectory");
  const auto result_b = request_b.Wait();
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "fallback preserves the admitted request trajectory");
  Expect(
      !result_b.incremental_prefill_supported && result_b.prefill_chunks == 1 &&
          result_b.max_prefill_chunk_tokens == 5 &&
          result_b.prefill_fallback_reason == "incremental_prefill_unavailable",
      "non-incremental runners report their full-prefill fallback");
}

void TestPendingLimitsRejectBeforeStateAdmission() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 1,
                                     .max_pending_requests_per_client = 1,
                                 });

  auto active = scheduler->Submit({1}, 4, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto queued = scheduler->Submit({2}, 1, 0.0F, {}, false,
                                  ClientMetadata("queued-client"));

  bool rejected = false;
  try {
    (void)scheduler->Submit({3}, 1, 0.0F, {}, false,
                            ClientMetadata("third-client"));
  } catch (const TextGenerationError& error) {
    rejected = error.code() == TextGenerationErrorCode::kQueueFull;
  }
  Expect(rejected, "full pending queue rejects before admission");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "rejected request cannot allocate another runner state");

  control->ReleaseAdvance();
  Expect(active.Wait().tokens == ExpectedTokens(1, 4) &&
             queued.Wait().tokens == ExpectedTokens(2, 1),
         "accepted work survives a queue rejection");
}

void TestPendingClientsAreRoundRobinAndIndividuallyBounded() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_pending_requests = 4,
                                     .max_pending_requests_per_client = 2,
                                 });

  auto active = scheduler->Submit({1}, 3, 0.0F, {}, false,
                                  ClientMetadata("active-client"));
  control->WaitForAdvance(1);
  auto client_a_first =
      scheduler->Submit({2}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_a_second =
      scheduler->Submit({3}, 1, 0.0F, {}, false, ClientMetadata("client-a"));
  auto client_b =
      scheduler->Submit({4}, 1, 0.0F, {}, false, ClientMetadata("client-b"));

  bool client_rejected = false;
  try {
    (void)scheduler->Submit({5}, 1, 0.0F, {}, false,
                            ClientMetadata("client-a"));
  } catch (const TextGenerationError& error) {
    client_rejected = error.code() == TextGenerationErrorCode::kClientQueueFull;
  }
  Expect(client_rejected, "one client cannot monopolize the pending queue");

  control->ReleaseAdvance();
  (void)active.Wait();
  (void)client_a_first.Wait();
  (void)client_a_second.Wait();
  (void)client_b.Wait();

  const auto events = control->Events();
  const std::size_t a_first = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_second = EventIndex(events, EventKind::kPrefill, 3);
  const std::size_t b_first = EventIndex(events, EventKind::kPrefill, 4);
  Expect(a_first < b_first && b_first < a_second,
         "pending clients rotate before one client receives another admission");
}

void TestExpiredQueuedRequestNeverConsumesState() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto active = scheduler->Submit({1}, 3, 0.0F);
  control->WaitForAdvance(1);
  auto expired = scheduler->Submit(
      {2}, 1, 0.0F, {}, false,
      TextRequestMetadata{
          .client_id = "expired-client",
          .deadline = TextGenerationScheduler::Clock::now() -
                      std::chrono::milliseconds{1},
          .request_start = TextGenerationScheduler::Clock::now(),
      });
  control->ReleaseAdvance();

  bool deadline_reported = false;
  try {
    (void)expired.Wait();
  } catch (const TextGenerationError& error) {
    deadline_reported =
        error.code() == TextGenerationErrorCode::kDeadlineExceeded;
  }
  Expect(deadline_reported, "expired queued work reports a stable deadline");
  Expect(active.Wait().tokens == ExpectedTokens(1, 3),
         "expired queued work does not disturb active generation");
  Expect(control->states_created.load(std::memory_order_relaxed) == 1,
         "expired queued work never creates or acquires another state");
}

void TestSlowConsumerOutputIsBoundedAndReclaimed() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_buffered_output_bytes_per_request = 4,
                                     .max_buffered_output_bytes_total = 4,
                                 });

  auto request = scheduler->Submit({5}, 5, 0.0F, {}, true);
  std::mutex callback_mutex;
  std::condition_variable callback_condition;
  bool callback_entered = false;
  bool release_callback = false;
  bool backpressure_reported = false;
  std::jthread consumer([&] {
    try {
      (void)request.Wait([&](std::string_view) {
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_entered = true;
        control->ReleaseAdvance();
        callback_condition.notify_all();
        callback_condition.wait(lock, [&] { return release_callback; });
        return true;
      });
    } catch (const TextGenerationError& error) {
      backpressure_reported =
          error.code() == TextGenerationErrorCode::kOutputBackpressure;
    }
  });

  {
    std::unique_lock<std::mutex> lock(callback_mutex);
    const bool entered = callback_condition.wait_for(
        lock, kTestTimeout, [&] { return callback_entered; });
    Expect(entered, "slow consumer receives its first output piece");
  }
  control->WaitForInvalidations(1);
  {
    const std::lock_guard<std::mutex> lock(callback_mutex);
    release_callback = true;
  }
  callback_condition.notify_all();
  consumer.join();

  Expect(backpressure_reported,
         "bounded output queue fails a persistently slow consumer");
  Expect(scheduler->buffered_output_bytes() == 0,
         "failed slow-consumer output is fully reclaimed");
  Expect(scheduler->max_buffered_output_bytes() <= 4,
         "server-wide buffered output never exceeds its declared limit");

  const auto replacement = scheduler->Submit({6}, 2, 0.0F).Wait();
  Expect(replacement.tokens == ExpectedTokens(6, 2),
         "state is reusable after output backpressure cancellation");
}

void TestGeneratedOutputLimitAppliesWithoutStreaming() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 1, {},
                                 {
                                     .max_output_bytes_per_request = 4,
                                 });

  bool output_limit_reported = false;
  try {
    (void)scheduler->Submit({7}, 3, 0.0F).Wait();
  } catch (const TextGenerationError& error) {
    output_limit_reported =
        error.code() == TextGenerationErrorCode::kOutputLimit;
  }
  Expect(output_limit_reported,
         "non-streaming generation obeys its output byte limit");
}

void TestMidGenerationAdmissionAndIsolatedTrajectories() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1, 10}, 4, 0.0F, {}, true);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 2, 0.0F, {}, true);
  control->ReleaseAdvance();

  std::vector<std::string> pieces_a;
  const auto result_a = request_a.Wait([&](std::string_view piece) {
    pieces_a.emplace_back(piece);
    return true;
  });
  std::vector<std::string> pieces_b;
  const auto result_b = request_b.Wait([&](std::string_view piece) {
    pieces_b.emplace_back(piece);
    return true;
  });

  Expect(result_a.tokens == ExpectedTokens(1, 4),
         "request A follows its isolated greedy trajectory");
  Expect(result_b.tokens == ExpectedTokens(2, 2),
         "request B follows its isolated greedy trajectory");
  Expect(pieces_a.size() == 4 && pieces_b.size() == 2,
         "each client receives only its own token pieces");

  const auto events = control->Events();
  const std::size_t b_prefill = EventIndex(events, EventKind::kPrefill, 2);
  const std::size_t a_last_advance =
      EventIndex(events, EventKind::kAdvance, 1, 3);
  Expect(b_prefill < a_last_advance,
         "request B begins prefill before request A completes");
}

void TestMidGenerationRequestJoinsNextDecodeBatch() {
  auto control = std::make_shared<FakeControl>();
  control->supports_batched_advance = true;
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 2);

  auto request_a = scheduler->Submit({1}, 8, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2}, 4, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  Expect(result_a.tokens == ExpectedTokens(1, 8) &&
             result_b.tokens == ExpectedTokens(2, 4),
         "dynamic batching preserves both isolated trajectories");

  bool joined = false;
  for (const auto& labels : control->AdvanceBatches()) {
    joined = joined || labels == std::vector<TextRunnerToken>({1, 2}) ||
             labels == std::vector<TextRunnerToken>({2, 1});
  }
  Expect(joined,
         "request B joins request A at a decode boundary after prefill");
  Expect(result_a.physical_execution_width == 2 &&
             result_b.physical_execution_width == 2 &&
             result_a.execution_plan == "batched-w2" &&
             result_b.execution_plan == "batched-w2",
         "joined requests report the real W=2 execution plan");
}

void TestFifoReplacementAdmissionWithOneSlot() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 1;
  auto scheduler = MakeScheduler(control, 1);

  auto request_a = scheduler->Submit({1, 10}, 2, 0.0F);
  control->WaitForAdvance(1);
  auto request_b = scheduler->Submit({2, 20}, 1, 0.0F);
  auto request_c = scheduler->Submit({3, 30}, 1, 0.0F);
  control->ReleaseAdvance();

  const auto result_a = request_a.Wait();
  const auto result_b = request_b.Wait();
  const auto result_c = request_c.Wait();
  Expect(!result_a.cancelled && !result_b.cancelled && !result_c.cancelled,
         "all FIFO requests complete");

  std::vector<TextRunnerToken> prefill_order;
  for (const auto& event : control->Events()) {
    if (event.kind == EventKind::kPrefill) {
      prefill_order.push_back(event.label);
    }
  }
  Expect(prefill_order == std::vector<TextRunnerToken>({1, 2, 3}),
         "replacement admission preserves FIFO order");
}

void TestQueuedAndPrefillCancellation() {
  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    auto scheduler = MakeScheduler(control, 1);

    auto active = scheduler->Submit({1, 10}, 2, 0.0F);
    control->WaitForAdvance(1);
    auto queued = scheduler->Submit({2, 20}, 1, 0.0F);
    queued.Cancel();
    control->ReleaseAdvance();

    Expect(!active.Wait().cancelled, "active request completes normally");
    Expect(queued.Wait().cancelled, "queued request cancellation is reported");
    const auto events = control->Events();
    Expect(EventIndex(events, EventKind::kPrefill, 2) == events.size(),
           "cancelled queued request never acquires a model state");
  }

  {
    auto control = std::make_shared<FakeControl>();
    control->block_advance_label = 1;
    control->block_prefill_label = 4;
    auto scheduler = MakeScheduler(control, 2, {.decode_active_tokens = 2});

    auto active = scheduler->Submit({1}, 4, 0.0F);
    control->WaitForAdvance(1);
    auto request = scheduler->Submit({4, 40, 41, 42}, 2, 0.0F);
    control->ReleaseAdvance();
    control->WaitForPrefill(4);
    request.Cancel();
    control->ReleasePrefill();

    const auto result = request.Wait();
    Expect(!active.Wait().cancelled,
           "active decoder survives another request cancellation");
    Expect(result.cancelled, "active-decode prefill cancellation is reported");
    Expect(EventIndex(control->Events(), EventKind::kAdvance, 4) ==
               control->Events().size(),
           "cancelled prefill never advances decode");
  }
}

void TestDecodeCancellationAndStateReclamation() {
  auto control = std::make_shared<FakeControl>();
  control->block_advance_label = 5;
  auto scheduler = MakeScheduler(control, 1);

  auto cancelled = scheduler->Submit({5, 50}, 5, 0.0F, {}, true);
  std::size_t delivered_pieces = 0;
  const auto cancelled_result = cancelled.Wait([&](std::string_view) {
    ++delivered_pieces;
    control->ReleaseAdvance();
    return false;
  });
  Expect(cancelled_result.cancelled,
         "callback cancellation reaches the scheduler");
  Expect(delivered_pieces == 1,
         "no token is published after callback cancellation");

  auto replacement = scheduler->Submit({6, 60}, 2, 0.0F);
  const auto replacement_result = replacement.Wait();
  Expect(replacement_result.tokens == ExpectedTokens(6, 2),
         "cancelled state slot is immediately reusable");
  Expect(control->invalidations.load(std::memory_order_relaxed) >= 1,
         "decode cancellation invalidates partial model state");
}

void TestFourResidentRequestsMakeProgress() {
  auto control = std::make_shared<FakeControl>();
  auto scheduler = MakeScheduler(control, 4);

  std::vector<TextGenerationScheduler::Request> requests;
  for (TextRunnerToken label = 1; label <= 4; ++label) {
    requests.push_back(scheduler->Submit({label, label + 10}, 3, 0.0F));
  }
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto result = requests[index].Wait();
    Expect(result.tokens ==
               ExpectedTokens(static_cast<TextRunnerToken>(index + 1), 3),
           "C=4 serial fallback preserves isolated output");
    Expect(result.requested_logical_concurrency == 4 &&
               result.physical_execution_width == 1 &&
               result.execution_plan == "serial-fallback",
           "C=4 telemetry exposes the physical serial fallback");
  }
}

void TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement() {
  auto control = std::make_shared<FakeControl>();
  control->throw_advance_label = 9;
  auto scheduler = MakeScheduler(control, 1);

  bool failed = false;
  try {
    auto request = scheduler->Submit({9, 90}, 2, 0.0F);
    (void)request.Wait();
  } catch (const std::runtime_error& exception) {
    failed = std::string_view(exception.what()) ==
             "injected scheduler runner failure";
  }
  Expect(failed, "runner failure reaches the submitting client");

  control->throw_advance_label.reset();
  auto replacement = scheduler->Submit({8, 80}, 2, 0.0F);
  Expect(replacement.Wait().tokens == ExpectedTokens(8, 2),
         "replacement request succeeds after runner failure");
}

}  // namespace

int main() {
  TestIdlePrefillUsesBulkWorkUnit();
  TestRunnerCanSkipUnusedFinalAdvance();
  TestRunnerCanReuseExactIncrementalText();
  TestMultiTokenDecodePublishesDraftMetricsAndDisablesPrefixReuse();
  TestMultiResidentPrefillUsesBoundedWorkUnits();
  TestDecodeActivePrefillIsBounded();
  TestPrefillYieldsToEveryDueDecoder();
  TestNonIncrementalRunnerFallsBackSafely();
  TestPendingLimitsRejectBeforeStateAdmission();
  TestPendingClientsAreRoundRobinAndIndividuallyBounded();
  TestExpiredQueuedRequestNeverConsumesState();
  TestSlowConsumerOutputIsBoundedAndReclaimed();
  TestGeneratedOutputLimitAppliesWithoutStreaming();
  TestMidGenerationAdmissionAndIsolatedTrajectories();
  TestMidGenerationRequestJoinsNextDecodeBatch();
  TestFifoReplacementAdmissionWithOneSlot();
  TestQueuedAndPrefillCancellation();
  TestDecodeCancellationAndStateReclamation();
  TestFourResidentRequestsMakeProgress();
  TestRunnerFailureInvalidatesAndDoesNotPoisonReplacement();
  std::cout << "All text generation scheduler tests passed\n";
  return 0;
}
