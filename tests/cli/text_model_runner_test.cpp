#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using strix::server::ChatRequest;
using strix::server::TextDecodeSelection;
using strix::server::TextExecutionPlan;
using strix::server::TextExecutionPlanKind;
using strix::server::TextModelRunner;
using strix::server::TextPrefillStep;
using strix::server::TextRunnerAdvance;
using strix::server::TextRunnerCapabilities;
using strix::server::TextRunnerDescriptor;
using strix::server::TextRunnerMeasuredResources;
using strix::server::TextRunnerPool;
using strix::server::TextRunnerResourceClaim;
using strix::server::TextRunnerSnapshot;
using strix::server::TextRunnerState;
using strix::server::TextRunnerToken;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeStats {
  std::size_t states_created{0};
  std::size_t invalidations{0};
  std::size_t cancellation_bindings{0};
  std::size_t cancellation_clears{0};
  std::vector<std::size_t> prefill_spans;
  std::vector<TextRunnerToken> advanced_tokens;
  std::vector<std::vector<TextRunnerToken>> advanced_batches;
};

class FakeState final : public TextRunnerState {
public:
  FakeState(std::shared_ptr<FakeStats> stats, std::size_t measured_bytes)
      : stats_(std::move(stats)), measured_bytes_(measured_bytes) {}

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    if (is_cancelled) {
      ++stats_->cancellation_bindings;
    } else {
      ++stats_->cancellation_clears;
    }
  }

  void Invalidate() noexcept override {
    ++stats_->invalidations;
    position = 0;
    decode_count = 0;
    frontier.reset();
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {
        .per_request_state_bytes = measured_bytes_,
        .temporary_scratch_bytes = 16,
    };
  }

  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;

private:
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
};

FakeState& RequireFakeState(TextRunnerState& state) {
  auto* fake = dynamic_cast<FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected fake runner state");
  }
  return *fake;
}

const FakeState& RequireFakeState(const TextRunnerState& state) {
  const auto* fake = dynamic_cast<const FakeState*>(&state);
  if (fake == nullptr) {
    throw std::logic_error("unexpected fake runner state");
  }
  return *fake;
}

class FakeRunner : public TextModelRunner {
public:
  FakeRunner(std::shared_ptr<FakeStats> stats, std::size_t measured_bytes = 64,
             std::size_t state_capacity_bytes = 256)
      : stats_(std::move(stats)),
        measured_bytes_(measured_bytes),
        state_capacity_bytes_(state_capacity_bytes) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "fake-model",
        .state_abi = "fake-state-v1",
        .max_context = 64,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
            },
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = state_capacity_bytes_,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 16,
        .requires_device_runtime_lock = false,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 2,
        },
        {
            .kind = TextExecutionPlanKind::kBatched,
            .physical_width = 4,
        },
    };
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(text.size());
    for (const char value : text) {
      tokens.push_back(static_cast<unsigned char>(value));
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    if (request.messages.empty()) {
      return std::nullopt;
    }
    return Tokenize(request.messages.front().content);
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      text += std::to_string(token);
    }
    return text;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    ++stats_->states_created;
    return std::make_unique<FakeState>(stats_, measured_bytes_);
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& fake = RequireFakeState(state);
    if (offset != fake.position || offset >= prompt.size()) {
      throw std::logic_error("invalid fake prefill position");
    }
    const std::size_t consumed =
        std::min(max_input_tokens, prompt.size() - offset);
    stats_->prefill_spans.push_back(consumed);
    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = 90;
    }
    return {
        .consumed_tokens = consumed,
        .decode_ready = ready,
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(TextRunnerState& state, float,
                                               std::uint64_t*) const override {
    auto& fake = RequireFakeState(state);
    if (!fake.frontier.has_value()) {
      throw std::logic_error("fake state has no decode frontier");
    }
    if (fake.decode_count == 2) {
      return {
          .stop = true,
          .token = 0,
          .piece = {},
      };
    }
    const TextRunnerToken token =
        *fake.frontier + static_cast<TextRunnerToken>(fake.decode_count);
    return {
        .token = token,
        .piece = std::to_string(token),
    };
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& fake = RequireFakeState(state);
    stats_->advanced_tokens.push_back(token);
    ++fake.position;
    ++fake.decode_count;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    std::vector<TextRunnerToken> tokens;
    tokens.reserve(advances.size());
    for (const auto& advance : advances) {
      tokens.push_back(advance.token);
    }
    stats_->advanced_batches.push_back(std::move(tokens));
    for (const auto& advance : advances) {
      Advance(advance.state.get(), advance.token);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireFakeState(state).position;
  }

private:
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
  std::size_t state_capacity_bytes_;
};

void TestBoundedPrefillDecodeAndPrefixReuse() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  Expect(pool.runner().Descriptor().model_id == "fake-model",
         "pool exposes the validated runner");
  Expect(pool.capacity() == 1, "pool exposes its bounded capacity");

  {
    auto request = pool.Acquire({1, 2, 3, 4});
    Expect(static_cast<bool>(request), "cold request acquires state");
    Expect(!request.cache_hit(), "first request is a miss");
    Expect(request.cached_prompt_tokens() == 0,
           "cold request starts at token zero");

    const auto first = request.Prefill(2);
    Expect(first.consumed_tokens == 2 && !first.decode_ready,
           "first prefill obeys its input budget");
    const auto second = request.Prefill(8);
    Expect(second.consumed_tokens == 2 && second.decode_ready,
           "second prefill reaches the decode frontier");

    const auto token_0 = request.SelectNext(0.0F);
    Expect(!token_0.stop && token_0.token == 90,
           "first frontier token is selected");
    request.Advance();
    const auto token_1 = request.SelectNext(0.0F);
    Expect(!token_1.stop && token_1.token == 91,
           "second frontier token is selected");
    request.Advance();
    Expect(request.SelectNext(0.0F).stop,
           "runner reports an explicit stop boundary");
    request.Commit();
  }

  {
    auto extension = pool.Acquire({1, 2, 3, 4, 90, 91, 7, 8});
    Expect(extension.cache_hit(), "exact extension reuses opaque state");
    Expect(extension.cached_prompt_tokens() == 6,
           "cache boundary includes advanced decode tokens");
    const auto suffix = extension.Prefill(16);
    Expect(suffix.consumed_tokens == 2 && suffix.decode_ready,
           "only the uncached suffix is prefetched");
    extension.Invalidate();
  }

  Expect(stats->prefill_spans == std::vector<std::size_t>({2, 2, 2}),
         "runner receives deterministic bounded prefill work units");
  Expect(stats->advanced_tokens == std::vector<TextRunnerToken>({90, 91}),
         "runner receives one decode advance per emitted token");
}

void TestAbandonedRequestRollsBackState() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto request = pool.Acquire({4, 5, 6});
    (void)request.Prefill(3);
  }
  Expect(stats->invalidations == 1,
         "abandoned request invalidates partially executed state");

  auto retry = pool.Acquire({4, 5, 6, 7});
  Expect(!retry.cache_hit(), "rolled-back state is not reusable");
  retry.Invalidate();
}

void TestRequestBindsAndClearsCancellation() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 1);

  auto request = pool.Acquire({4, 5, 6}, [] { return false; });
  Expect(stats->cancellation_bindings == 1,
         "request binds its cancellation check to opaque state");
  request.Invalidate();
  Expect(stats->cancellation_clears == 1,
         "request clears its cancellation check before releasing state");
}

void TestBatchedAdvancePreservesIndependentRequests() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FakeRunner>(stats);
  TextRunnerPool pool(runner, 2);

  auto first = pool.Acquire({1});
  auto second = pool.Acquire({2});
  Expect(first.Prefill(1).decode_ready && second.Prefill(1).decode_ready,
         "independent requests reach their decode frontiers");
  Expect(
      first.SelectNext(0.0F).token == 90 && second.SelectNext(0.0F).token == 90,
      "independent requests select their pending tokens");

  const auto plan = pool.SelectDecodePlan(2);
  Expect(
      plan.kind == TextExecutionPlanKind::kBatched && plan.physical_width == 2,
      "two ready requests select W=2");

  std::array<TextRunnerPool::Request*, 2> requests{&first, &second};
  pool.AdvanceBatch(requests, plan);
  Expect(stats->advanced_batches ==
             std::vector<std::vector<TextRunnerToken>>{{90, 90}},
         "one batched runner call receives both request tokens");

  Expect(
      first.SelectNext(0.0F).token == 91 && second.SelectNext(0.0F).token == 91,
      "batched advance independently updates both request states");
  first.Invalidate();
  second.Invalidate();
}

void TestResourceClaimsAreValidatedBeforeAllocation() {
  auto stats = std::make_shared<FakeStats>();
  bool rejected = false;
  try {
    auto runner = std::make_shared<FakeRunner>(stats, 64, 159);
    TextRunnerPool pool(runner, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Expect(rejected, "aggregate request-state claim must fit capacity");
  Expect(stats->states_created == 0,
         "invalid resource claim is rejected before state allocation");
}

class FakeSnapshot final : public TextRunnerSnapshot {
public:
  explicit FakeSnapshot(std::size_t position) : position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return sizeof(position);
  }

  std::size_t position;
};

class SnapshotRunner final : public FakeRunner {
public:
  using FakeRunner::FakeRunner;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = FakeRunner::Descriptor();
    descriptor.capabilities.snapshot = true;
    descriptor.capabilities.fork = true;
    return descriptor;
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    return std::make_unique<FakeSnapshot>(RequireFakeState(state).position);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> RestoreOrFork(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
    if (fake == nullptr) {
      throw std::invalid_argument("snapshot type mismatch");
    }
    auto state = CreateState();
    RequireFakeState(*state).position = fake->position;
    return state;
  }
};

void TestSnapshotForkAndUnsupportedCapabilities() {
  auto stats = std::make_shared<FakeStats>();
  SnapshotRunner runner(stats);
  auto state = runner.CreateState();
  RequireFakeState(*state).position = 7;
  auto snapshot = runner.Snapshot(*state);
  Expect(snapshot != nullptr && snapshot->PayloadBytes() == sizeof(std::size_t),
         "snapshot reports its opaque payload bytes");
  auto fork = runner.RestoreOrFork(*snapshot);
  Expect(RequireFakeState(*fork).position == 7,
         "fork restores the exact runner-owned boundary");

  FakeRunner unsupported(stats);
  bool snapshot_rejected = false;
  try {
    (void)unsupported.Snapshot(*state);
  } catch (const std::logic_error&) {
    snapshot_rejected = true;
  }
  Expect(snapshot_rejected, "unsupported snapshots fail explicitly");
}

void TestMeasuredStateIsReconciledWithClaim() {
  auto stats = std::make_shared<FakeStats>();
  bool rejected = false;
  try {
    auto runner = std::make_shared<FakeRunner>(stats, 65, 256);
    TextRunnerPool pool(runner, 1);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected, "measured state cannot exceed its proposed allocation");
  Expect(stats->states_created == 1,
         "measured resource check runs immediately after allocation");
}

}  // namespace

int main() {
  TestBoundedPrefillDecodeAndPrefixReuse();
  TestAbandonedRequestRollsBackState();
  TestRequestBindsAndClearsCancellation();
  TestBatchedAdvancePreservesIndependentRequests();
  TestResourceClaimsAreValidatedBeforeAllocation();
  TestSnapshotForkAndUnsupportedCapabilities();
  TestMeasuredStateIsReconciledWithClaim();
  std::cout << "All text model runner tests passed\n";
  return 0;
}
