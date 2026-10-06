#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen/vision/prompt.hpp"

namespace {

using gufo::server::ChatRequest;
using gufo::server::TextDecodeSelection;
using gufo::server::TextExecutionPlan;
using gufo::server::TextExecutionPlanKind;
using gufo::server::TextModelRunner;
using gufo::server::TextPrefillStep;
using gufo::server::TextRunnerAdvance;
using gufo::server::TextRunnerCapabilities;
using gufo::server::TextRunnerDescriptor;
using gufo::server::TextRunnerDiskCacheOptions;
using gufo::server::TextRunnerMeasuredResources;
using gufo::server::TextRunnerPool;
using gufo::server::TextRunnerResourceClaim;
using gufo::server::TextRunnerSnapshot;
using gufo::server::TextRunnerState;
using gufo::server::TextRunnerToken;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeStats {
  bool fail_after_advance{false};
  std::size_t states_created{0};
  std::size_t invalidations{0};
  std::size_t snapshot_restores{0};
  std::size_t snapshot_size_queries{0};
  std::size_t snapshot_captures{0};
  std::size_t cancellation_bindings{0};
  std::size_t cancellation_clears{0};
  std::vector<std::vector<TextRunnerToken>> prepared_prefixes;
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
             std::size_t state_capacity_bytes = 256,
             std::size_t retained_snapshot_capacity_bytes = 256)
      : stats_(std::move(stats)),
        measured_bytes_(measured_bytes),
        state_capacity_bytes_(state_capacity_bytes),
        retained_snapshot_capacity_bytes_(retained_snapshot_capacity_bytes) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "fake-model",
        .state_abi = "fake-state-v1",
        .max_context = 64,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = state_capacity_bytes_,
        .per_request_state_bytes = 64,
        .temporary_scratch_bytes = 16,
        .retained_snapshot_capacity_bytes = retained_snapshot_capacity_bytes_,
        .retained_snapshot_ceiling_bytes = retained_snapshot_ceiling_bytes,
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

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    const auto& fake = RequireFakeState(state);
    if (fake.position != prefix.size()) {
      throw std::logic_error("fake retained prefix position mismatch");
    }
    stats_->prepared_prefixes.emplace_back(prefix.begin(), prefix.end());
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

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, gufo::sampling::SamplerState&) const override {
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
    if (stats_->fail_after_advance)
      throw std::runtime_error("injected failure after state mutation");
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

protected:
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
  std::size_t state_capacity_bytes_;
  std::size_t retained_snapshot_capacity_bytes_;

public:
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;

private:
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

    const auto token_0 = request.SelectNext();
    Expect(!token_0.stop && token_0.token == 90,
           "first frontier token is selected");
    request.Advance();
    const auto token_1 = request.SelectNext();
    Expect(!token_1.stop && token_1.token == 91,
           "second frontier token is selected");
    request.Advance();
    Expect(request.SelectNext().stop,
           "runner reports an explicit stop boundary");
    request.Commit();
  }

  {
    auto extension = pool.Acquire({1, 2, 3, 4, 90, 91, 7, 8});
    Expect(extension.cache_hit(), "exact extension reuses opaque state");
    Expect(extension.cached_prompt_tokens() == 6,
           "cache boundary includes advanced decode tokens");
    Expect(
        stats->prepared_prefixes ==
            std::vector<std::vector<TextRunnerToken>>({{1, 2, 3, 4, 90, 91}}),
        "runner prepares retained metadata before suffix prefill");
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
  Expect(first.SelectNext().token == 90 && second.SelectNext().token == 90,
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

  Expect(first.SelectNext().token == 91 && second.SelectNext().token == 91,
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
  FakeSnapshot(std::size_t position, std::size_t decode_count,
               std::optional<TextRunnerToken> frontier)
      : position(position), decode_count(decode_count), frontier(frontier) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return sizeof(FakeSnapshot);
  }

  std::size_t position;
  std::size_t decode_count;
  std::optional<TextRunnerToken> frontier;
};

class SnapshotRunner : public FakeRunner {
public:
  using FakeRunner::FakeRunner;

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = FakeRunner::Descriptor();
    descriptor.capabilities.snapshot = true;
    descriptor.capabilities.fork = true;
    return descriptor;
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState&) const override {
    ++stats_->snapshot_size_queries;
    return sizeof(FakeSnapshot);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    ++stats_->snapshot_captures;
    const auto& fake = RequireFakeState(state);
    return std::make_unique<FakeSnapshot>(fake.position, fake.decode_count,
                                          fake.frontier);
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
    if (fake == nullptr) {
      throw std::invalid_argument("snapshot type mismatch");
    }
    auto& restored = RequireFakeState(state);
    restored.position = fake->position;
    restored.decode_count = fake->decode_count;
    restored.frontier = fake->frontier;
    ++stats_->snapshot_restores;
  }
};

class InPassRunner final : public SnapshotRunner {
public:
  using SnapshotRunner::SnapshotRunner;
  bool fail_after_prefill{false};
  TextRunnerDescriptor Descriptor() const override {
    auto descriptor = SnapshotRunner::Descriptor();
    descriptor.capabilities.in_pass_checkpoint = true;
    return descriptor;
  }
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    if (boundary <= offset ||
        boundary >= std::min(prompt.size(), offset + budget))
      return std::nullopt;
    return sizeof(FakeSnapshot);
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint) const override {
    ++stats_->snapshot_captures;
    *checkpoint = std::make_unique<FakeSnapshot>(boundary, 0, 90);
    auto step = FakeRunner::Prefill(state, prompt, offset, budget);
    if (fail_after_prefill) {
      auto& fake = RequireFakeState(state);
      fake.decode_count = 7;
      fake.frontier = 777;
      throw std::runtime_error("injected in-pass failure after state mutation");
    }
    step.checkpoint_ms = 2;
    return step;
  }
};

void TestInPassStableCheckpoint() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<InPassRunner>(stats), 1);
  auto first = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  const auto step = first.Prefill(64);
  Expect(step.consumed_tokens == 5 && step.decode_ready,
         "in-pass checkpoint completes framing in the same forward");
  Expect(stats->snapshot_captures == 1 &&
             stats->prefill_spans == std::vector<std::size_t>{5},
         "one prefill captures the stable boundary before mutation");
  const auto commit = first.Commit();
  Expect(commit.snapshot_ms >= 2,
         "in-pass capture contributes to snapshot phase timing");
  auto second = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(second.cached_prompt_tokens() == 3,
         "rewritten framing restores the in-pass boundary");
  Expect(second.Prefill(64).decode_ready,
         "warm rewritten turn captures its next boundary in one pass");
  second.Commit();
  auto third = pool.Acquire({1, 2, 3, 50, 51, 7, 60, 61}, {}, {}, {}, true, 6);
  Expect(third.cached_prompt_tokens() == 6,
         "the stable checkpoint advances on warm turns");
  third.Invalidate();
}

void TestInPassFailureRetainsOnlyCompletedCheckpoints() {
  auto stats = std::make_shared<FakeStats>();
  // Both the old fallback and the next stable boundary fit, with no spare
  // reservation. A leaked reservation would force the old fallback out.
  auto runner =
      std::make_shared<InPassRunner>(stats, 64, 256, 2 * sizeof(FakeSnapshot));
  TextRunnerPool pool(runner, 1);
  auto seed = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(seed.Prefill(64).decode_ready, "seed reaches its decode frontier");
  seed.Cancel();  // Retain the stable checkpoint without a full-prompt copy.

  auto failed = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(failed.cached_prompt_tokens() == 3,
         "rewritten framing starts from the immutable fallback");
  runner->fail_after_prefill = true;
  bool saw_failure = false;
  try {
    (void)failed.Prefill(64);
  } catch (const std::runtime_error& error) {
    saw_failure = std::string_view(error.what()) ==
                  "injected in-pass failure after state mutation";
  }
  Expect(saw_failure, "in-pass prefill fails after changing the fake state");
  failed.Cancel();
  runner->fail_after_prefill = false;

  auto retry = pool.Acquire({1, 2, 3, 50, 51, 7, 40, 41}, {}, {}, {}, true, 6);
  Expect(retry.cached_prompt_tokens() == 3 &&
             retry.cache_restore_bytes() == sizeof(FakeSnapshot),
         "cancel restores the immutable fallback, not the failed frontier");
  const auto step = retry.Prefill(64);
  Expect(step.consumed_tokens == 5 && step.decode_ready,
         "retry recomputes all work from the failed in-pass operation");
  Expect(retry.SelectNext().token == 90,
         "retry does not inherit the failed operation's decode state");
  retry.Advance();
  retry.Commit();

  auto old_branch = pool.Acquire({1, 2, 3, 70, 71}, {}, {}, {}, true, 3);
  Expect(old_branch.cached_prompt_tokens() == 3 &&
             old_branch.cache_restore_bytes() == sizeof(FakeSnapshot),
         "released reservation lets the old fallback survive the healthy turn");
  old_branch.Invalidate();
  auto new_branch =
      pool.Acquire({1, 2, 3, 50, 51, 7, 60, 61}, {}, {}, {}, true, 6);
  Expect(new_branch.cached_prompt_tokens() == 6 &&
             new_branch.cache_restore_bytes() == sizeof(FakeSnapshot),
         "healthy retry also retains its newly completed stable boundary");
  new_branch.Invalidate();
}

class PersistentSnapshotRunner final : public SnapshotRunner {
public:
  std::function<void()> before_serialize;
  PersistentSnapshotRunner(std::shared_ptr<FakeStats> stats,
                           std::string identity,
                           std::size_t retained_snapshot_capacity_bytes = 256,
                           std::size_t max_context = 64)
      : SnapshotRunner(std::move(stats), 64, 256,
                       retained_snapshot_capacity_bytes),
        identity_(identity.begin(), identity.end()),
        max_context_(max_context) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = SnapshotRunner::Descriptor();
    descriptor.max_context = max_context_;
    descriptor.persistence = gufo::server::TextRunnerPersistenceDescriptor{
        .compatibility_identity = identity_,
        .payload_version = 1,
    };
    return descriptor;
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    (void)RequireSnapshot(snapshot);
    return 4 * sizeof(std::uint64_t);
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    if (before_serialize)
      before_serialize();
    const auto& saved = RequireSnapshot(snapshot);
    if (destination.size() != 4 * sizeof(std::uint64_t)) {
      throw std::invalid_argument("fake persistent payload size mismatch");
    }
    const std::array<std::uint64_t, 4> fields = {
        saved.position,
        saved.decode_count,
        saved.frontier.has_value() ? 1U : 0U,
        saved.frontier.value_or(0),
    };
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        destination[field * sizeof(std::uint64_t) + byte] =
            static_cast<std::uint8_t>(fields[field] >> (byte * 8U));
      }
    }
    return destination.size();
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    if (payload.size() != 4 * sizeof(std::uint64_t)) {
      throw std::invalid_argument("fake persistent payload size mismatch");
    }
    std::array<std::uint64_t, 4> fields{};
    for (std::size_t field = 0; field < fields.size(); ++field) {
      for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte) {
        fields[field] |= static_cast<std::uint64_t>(
                             payload[field * sizeof(std::uint64_t) + byte])
                         << (byte * 8U);
      }
    }
    auto& restored = RequireFakeState(state);
    restored.position = fields[0];
    restored.decode_count = fields[1];
    restored.frontier = fields[2] != 0
                            ? std::optional<TextRunnerToken>(
                                  static_cast<TextRunnerToken>(fields[3]))
                            : std::nullopt;
    ++stats_->snapshot_restores;
  }

private:
  static const FakeSnapshot& RequireSnapshot(
      const TextRunnerSnapshot& snapshot) {
    const auto* fake = dynamic_cast<const FakeSnapshot*>(&snapshot);
    if (fake == nullptr) {
      throw std::invalid_argument("snapshot type mismatch");
    }
    return *fake;
  }

  std::vector<std::uint8_t> identity_;
  std::size_t max_context_;
};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "gufo-runner-disk-cache-XXXXXX")
                              .string();
    const char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error("failed to create runner cache directory");
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

void TestSnapshotForkAndUnsupportedCapabilities() {
  auto stats = std::make_shared<FakeStats>();
  SnapshotRunner runner(stats);
  auto state = runner.CreateState();
  RequireFakeState(*state).position = 7;
  RequireFakeState(*state).frontier = 90;
  auto snapshot = runner.Snapshot(*state);
  Expect(
      snapshot != nullptr && snapshot->PayloadBytes() == sizeof(FakeSnapshot),
      "snapshot reports its opaque payload bytes");
  auto fork = runner.CreateState();
  runner.RestoreOrFork(*fork, *snapshot);
  Expect(RequireFakeState(*fork).position == 7 &&
             RequireFakeState(*fork).frontier == 90,
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

void TestSnapshotCacheBranchesOnePrefixIntoIndependentStates() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 2);

  {
    auto root = pool.Acquire({1, 2, 3, 4});
    Expect(root.Prefill(4).decode_ready,
           "root prefix reaches its snapshot boundary");
    const auto commit = root.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot) &&
               commit.snapshot_ms >= 0.0,
           "root commit reports retained full-copy snapshot cost");
  }

  auto first = pool.Acquire({1, 2, 3, 4, 5});
  auto second = pool.Acquire({1, 2, 3, 4, 6});
  Expect(first.cache_hit() && second.cache_hit(),
         "two simultaneous requests restore one retained snapshot");
  Expect(
      first.cached_prompt_tokens() == 4 && second.cached_prompt_tokens() == 4,
      "both branches report the same immutable root prefix");
  Expect(
      first.cache_restore_bytes() == 0 &&
          second.cache_restore_bytes() == sizeof(FakeSnapshot) &&
          first.cache_restore_ms() >= 0.0 && second.cache_restore_ms() >= 0.0,
      "one branch uses the live frontier and the other restores the snapshot");
  Expect(stats->snapshot_restores == 1,
         "only the simultaneous branch needs a snapshot copy");
  Expect(stats->states_created == 2,
         "snapshot branching reuses preallocated request states");
  Expect(stats->snapshot_size_queries == 1 && stats->snapshot_captures == 1,
         "root snapshot reserves its exact payload before capture");

  Expect(first.Prefill(1).decode_ready && second.Prefill(1).decode_ready,
         "each branch prefills only its divergent suffix");
  Expect(first.SelectNext().token == 90 && second.SelectNext().token == 90,
         "both restored branches retain an exact frontier");
  first.Advance();
  second.Advance();
  first.Commit();
  second.Commit();

  auto third = pool.Acquire({1, 2, 3, 4, 7});
  Expect(third.cache_hit() && third.cached_prompt_tokens() == 4,
         "branch commits preserve the shared root while capacity permits");
  third.Invalidate();
}

void TestSnapshotRetentionUsesPromptBoundary() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto root = pool.Acquire({1, 2, 3, 4});
    Expect(root.Prefill(4).decode_ready,
           "root prompt reaches its decode frontier");
    Expect(stats->snapshot_captures == 0,
           "prefill returns before optional snapshot capture");
    Expect(root.SelectNext().token == 90,
           "root selects its prompt-boundary frontier");
    root.Advance();
    Expect(stats->snapshot_captures == 1,
           "snapshot is captured before generated tokens mutate the state");
    Expect(root.SelectNext().token == 91,
           "root advances beyond the reusable prompt boundary");
    root.Advance();
    const auto commit = root.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot),
           "root commit publishes the held prompt snapshot");
  }

  {
    auto continuation = pool.Acquire({1, 2, 3, 4, 90, 91, 7});
    Expect(continuation.cache_hit() &&
               continuation.cached_prompt_tokens() == 6 &&
               continuation.cache_restore_bytes() == 0,
           "conversation reuses all executed assistant tokens without "
           "restoration");
    continuation.Invalidate();
  }

  {
    auto repeated = pool.Acquire({1, 2, 3, 4});
    Expect(repeated.cache_hit() && repeated.cached_prompt_tokens() == 4,
           "an identical prompt restores the prompt-boundary snapshot");
    Expect(repeated.prefill_complete(),
           "an exact snapshot hit retains its decode frontier");
    Expect(stats->snapshot_captures == 1,
           "an exact in-memory hit avoids another snapshot copy");
    Expect(repeated.SelectNext().token == 90,
           "exact reuse starts from the prompt rather than post-generation");
    repeated.Invalidate();
  }

  {
    auto extension = pool.Acquire({1, 2, 3, 4, 7, 8});
    Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 4,
           "a prompt that omits generated tokens still reuses the root");
    const auto suffix = extension.Prefill(8);
    Expect(suffix.consumed_tokens == 2 && suffix.decode_ready,
           "only the extension after the stable prompt is prefetched");
    extension.CapturePromptSnapshot();
    Expect(stats->snapshot_captures == 2,
           "the extended prompt captures its own reusable boundary");
    Expect(extension.SelectNext().token == 90,
           "extended reuse preserves the rebuilt decode frontier");
    extension.Invalidate();
  }
}

void TestGeneratedFrontierForksBeforeMutation() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
  TextRunnerPool pool(runner, 2);
  {
    auto root = pool.Acquire({1, 2, 3, 4});
    root.Prefill(4);
    Expect(root.SelectNext().token == 90, "generated root token");
    root.Advance();
    root.Commit();
  }
  auto first = pool.Acquire({1, 2, 3, 4, 90, 5});
  auto second = pool.Acquire({1, 2, 3, 4, 90, 6});
  Expect(
      first.cached_prompt_tokens() == 5 && second.cached_prompt_tokens() == 5,
      "concurrent forks must reuse the generated frontier, not re-prefill "
      "its output");
  Expect(first.cache_restore_bytes() == 0 &&
             second.cache_restore_bytes() == sizeof(FakeSnapshot),
         "one live frontier is frozen for its peer");
  Expect(stats->snapshot_captures == 2 && stats->snapshot_restores == 1,
         "the generated frontier is copied once while the prompt is retained");
  auto blocked = pool.Acquire({9}, [] { return true; });
  Expect(!blocked, "publishing a snapshot must not release either live lease");
  Expect(first.Prefill(64).consumed_tokens == 1 &&
             second.Prefill(64).consumed_tokens == 1,
         "neither fork puts generated history into a prefill chunk");
  first.Invalidate();
  second.Invalidate();
  auto root_branch = pool.Acquire({1, 2, 3, 4, 7});
  Expect(root_branch.cached_prompt_tokens() == 4,
         "freezing a generated frontier preserves the original prompt branch");
}

void TestGeneratedFrontierPersistsForForks() {
  // Persist every frontier; spacing is checked below and by the disk store.
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk{.directory = directory.path(),
                                        .capacity_bytes = 4096,
                                        .staging_capacity_bytes = 4096,
                                        .min_checkpoint_step_tokens = 0};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool pool(runner, 2, disk);
    auto root = pool.Acquire({1, 2, 3});
    root.Prefill(3);
    (void)root.SelectNext();
    root.Advance();
    root.Commit();
    auto continuation = pool.Acquire({1, 2, 3, 90, 4});
    Expect(continuation.cached_prompt_tokens() == 4,
           "the live frontier includes generated output");
    continuation.Invalidate();
  }
  TextRunnerPool restarted(runner, 1, disk);
  auto restored = restarted.Acquire({1, 2, 3, 90, 5});
  Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 4,
         "disk forks restore the generated checkpoint before suffix prefill");
  Expect(restored.Prefill(16).consumed_tokens == 1,
         "disk restoration does not re-prefill known generated tokens");

  // With the default step, the generated frontier one token past the prompt
  // is not written; a restart resumes from the prompt and re-prefills it.
  TemporaryDirectory spaced_directory;
  const TextRunnerDiskCacheOptions spaced{.directory = spaced_directory.path(),
                                          .capacity_bytes = 4096,
                                          .staging_capacity_bytes = 4096};
  {
    TextRunnerPool pool(runner, 2, spaced);
    auto root = pool.Acquire({1, 2, 3});
    root.Prefill(3);
    (void)root.SelectNext();
    root.Advance();
    root.Commit();
  }
  TextRunnerPool spaced_restart(runner, 1, spaced);
  auto resumed = spaced_restart.Acquire({1, 2, 3, 90, 5});
  Expect(resumed.cache_disk_hit() && resumed.cached_prompt_tokens() == 3 &&
             resumed.Prefill(16).consumed_tokens == 2,
         "spaced disk checkpoints resume from the nearest stored prefix");
}

void TestCancellationRetainsOnlyCompletedWork() {
  for (const bool pending : {false, true}) {
    auto stats = std::make_shared<FakeStats>();
    TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
    auto request = pool.Acquire({1, 2});
    request.Prefill(2);
    Expect(request.SelectNext().token == 90, "first cancellation token");
    request.Advance();
    if (pending)
      Expect(request.SelectNext().token == 91, "selected but unexecuted token");
    request.Cancel();
    Expect(
        stats->advanced_tokens.size() == 1 && stats->snapshot_captures == 1,
        "cancellation neither advances a pending token nor starts a snapshot");
    auto continuation = pool.Acquire({1, 2, 90, 91, 7});
    Expect(continuation.cached_prompt_tokens() == 3 &&
               continuation.cache_restore_bytes() == 0,
           "cancelled conversation retains the exact completed live frontier");
    continuation.Invalidate();
    auto branch = pool.Acquire({1, 2, 8});
    Expect(branch.cached_prompt_tokens() == 2,
           "the original prompt remains available for branching after "
           "cancellation");
    branch.Invalidate();
  }

  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  auto request = pool.Acquire({1, 2});
  request.Prefill(2);
  (void)request.SelectNext();
  stats->fail_after_advance = true;
  try {
    request.Advance();
    Expect(false, "mutating failure must throw");
  } catch (const std::runtime_error&) {
  }
  request.Cancel();
  auto continuation = pool.Acquire({1, 2, 90, 7});
  Expect(
      continuation.cached_prompt_tokens() == 2 &&
          continuation.cache_restore_bytes() > 0,
      "a failed operation restores the immutable prompt, never mutated state");
  continuation.Invalidate();
}

void TestPersistentSnapshotRestoresAcrossPools() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{
      .directory = directory.path(),
      .capacity_bytes = 4096,
      .staging_capacity_bytes = 4096,
  };

  {
    auto writer_stats = std::make_shared<FakeStats>();
    auto writer = std::make_shared<PersistentSnapshotRunner>(
        writer_stats, "artifact-A", sizeof(FakeSnapshot) - 1);
    TextRunnerPool pool(writer, 1, disk_cache);
    auto request = pool.Acquire({1, 2, 3});
    Expect(request.Prefill(3).decode_ready,
           "writer reaches persistent checkpoint");
    const auto commit = request.Commit();
    Expect(commit.snapshot_bytes == 0 && commit.disk_queued_bytes > 0 &&
               commit.disk_enqueue_ms >= 0.0,
           "disk publication is independent of RAM snapshot admission");
  }

  auto reader_stats = std::make_shared<FakeStats>();
  auto reader =
      std::make_shared<PersistentSnapshotRunner>(reader_stats, "artifact-A");
  TextRunnerPool restarted(reader, 1, disk_cache);
  {
    auto cold = restarted.Acquire({1, 2, 3, 4, 5}, {}, {}, {}, false);
    Expect(!cold.cache_hit() && !cold.cache_disk_hit() &&
               cold.cached_prompt_tokens() == 0 &&
               reader_stats->snapshot_restores == 0,
           "cache_prompt=false bypasses disk restoration");
    cold.Invalidate();
  }
  auto extension = restarted.Acquire({1, 2, 3, 4, 5});
  Expect(extension.cache_hit() && extension.cache_disk_hit(),
         "clean pool restores compatible prefix from disk");
  Expect(extension.cached_prompt_tokens() == 3 &&
             extension.cache_restore_bytes() > 0 &&
             extension.cache_restore_ms() >= 0.0,
         "disk restore reports exact prefix and actual read metrics");
  Expect(extension.Prefill(8).consumed_tokens == 2,
         "restarted request prefills only its unmatched suffix");
  extension.Invalidate();

  auto incompatible_stats = std::make_shared<FakeStats>();
  auto incompatible = std::make_shared<PersistentSnapshotRunner>(
      incompatible_stats, "artifact-B");
  TextRunnerPool incompatible_pool(incompatible, 1, disk_cache);
  auto miss = incompatible_pool.Acquire({1, 2, 3, 4});
  Expect(!miss.cache_hit() && !miss.cache_disk_hit(),
         "changed compatibility identity is a cold miss");
  miss.Invalidate();
}

void TestPromptReuseCanBeDisabledPerRequest() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  {
    auto initial = pool.Acquire({1, 2});
    initial.Prefill(2);
    (void)initial.SelectNext();
    initial.Advance();
    initial.Commit();
  }
  const auto restores = stats->snapshot_restores;
  auto cold = pool.Acquire({1, 2, 90, 7}, {}, {}, {}, false);
  Expect(!cold.cache_hit() && cold.cached_prompt_tokens() == 0 &&
             stats->snapshot_restores == restores,
         "cache_prompt=false bypasses both live and immutable RAM frontiers");
  Expect(cold.Prefill(8).consumed_tokens == 4,
         "disabled reuse processes every prompt token");
  cold.Commit();
  auto retained = pool.Acquire({1, 2, 90, 7, 8});
  Expect(retained.cached_prompt_tokens() == 4,
         "a no-reuse request can populate the cache for later requests");
  retained.Invalidate();
}

void TestStableChatPrefixSurvivesInterruptedFraming() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 4096};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    // {1,2,3} is stable history; {40,41} opens assistant reasoning.
    auto initial = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    auto step = initial.Prefill(64);
    Expect(step.consumed_tokens == 3 && !step.decode_ready,
           "prefill stops before mutable assistant framing");
    Expect(initial.Prefill(64).decode_ready,
           "assistant suffix completes prefill");
    (void)initial.SelectNext();
    initial.Advance();
    initial.Cancel();
    // Client omits the unfinished reasoning and closes the assistant turn.
    auto resumed = pool.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
    Expect(resumed.cached_prompt_tokens() == 3,
           "changed assistant framing retains all stable prompt tokens");
    resumed.Invalidate();
  }
  TextRunnerPool restarted(runner, 1, disk_cache);
  auto restored = restarted.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(restored.cache_disk_hit() && restored.cached_prompt_tokens() == 3,
         "interrupted stable chat prefix survives server restart");
  restored.Invalidate();

  // Full-prompt entries written by older servers must not prevent creating
  // the shorter, stable checkpoint on the next request.
  TextRunnerPool legacy(runner, 1);
  auto old = legacy.Acquire({1, 2, 3, 40, 41});
  old.Prefill(64);
  old.Commit();
  auto migrated = legacy.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!migrated.cache_hit(),
         "legacy mutable-suffix snapshot cannot bypass the stable boundary");
  Expect(migrated.Prefill(64).consumed_tokens == 3,
         "legacy entry is replaced by a stable chat checkpoint");
  migrated.Cancel();
  auto next = legacy.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(next.cached_prompt_tokens() == 3,
         "migrated checkpoint survives changed assistant framing");
  next.Invalidate();
}

void TestWarmChatCheckpointsStopAtTheStableBoundary() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);
  Expect(pool.capacity() == 1 && stats->states_created == 1,
         "extra checkpoint entries do not allocate execution sessions");
  auto root = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(root.Prefill(64).consumed_tokens == 3,
         "cold chat retains the stable fallback before assistant framing");
  Expect(root.Prefill(64).decode_ready, "cold chat completes the suffix");
  const auto first = root.SelectNext().token;
  root.Advance();
  root.Commit();

  auto continuation =
      pool.Acquire({1, 2, 3, 40, 41, first, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(continuation.cached_prompt_tokens() == 6 &&
             continuation.cache_restore_bytes() == 0,
         "retained reasoning reuses generated tokens without a restore");
  // The warm turn stops once at its own stable boundary so that boundary is
  // checkpointed. Without that stop the conversation keeps falling back to
  // this turn's frontier for every later turn that rewrites the assistant.
  const auto step = continuation.Prefill(64);
  Expect(step.consumed_tokens == 1 && !step.decode_ready,
         "warm prefill stops at the stable boundary before assistant framing");
  const auto framing = continuation.Prefill(64);
  Expect(framing.consumed_tokens == 2 && framing.decode_ready,
         "assistant framing completes in the following pass");
  const auto second = continuation.SelectNext().token;
  continuation.Advance();
  continuation.Commit();

  auto repeated =
      pool.Acquire({1, 2, 3, 40, 41, first, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(repeated.cached_prompt_tokens() == 9 && repeated.prefill_complete(),
         "identical thinking prompt reuses the full prompt checkpoint");
  Expect(repeated.SelectNext().token == second,
         "the full checkpoint preserves greedy selection");
  repeated.Advance();
  repeated.Commit();

  // Empty reasoning changes 40,41 into 50,51 in the replayed assistant.
  auto dropped = pool.Acquire({1, 2, 3, 40, 41, first, 7, 50, 51, 8, 40, 41},
                              {}, {}, {}, true, 10);
  Expect(dropped.cached_prompt_tokens() == 7,
         "omitted reasoning resumes from the previous turn's stable boundary "
         "rather than the frontier before it");
  Expect(!dropped.Prefill(64).decode_ready,
         "the rewritten turn stops at its own boundary in turn");
  Expect(dropped.Prefill(64).decode_ready,
         "omitted reasoning prefills the remaining suffix");
  dropped.Cancel();
}

void TestHistoryEditsRestoreIntermediateCheckpoints() {
  class LongSnapshotRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = SnapshotRunner::Descriptor();
      descriptor.max_context = 32768;
      return descriptor;
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<LongSnapshotRunner>(stats, 64, 256, 4096);
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(10000, 1);
  auto original = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  while (!original.prefill_complete())
    (void)original.Prefill(32768);
  original.Commit();
  const auto captures = stats->snapshot_captures;

  auto unchanged = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(unchanged.prefill_complete() &&
             unchanged.cached_prompt_tokens() == prompt.size(),
         "complete prompt retries still reuse every token");
  unchanged.Commit();
  Expect(stats->snapshot_captures == captures,
         "an unchanged retry does not recapture intermediate state");

  prompt[9000] = 2;
  auto late_edit = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(late_edit.cache_hit() && late_edit.cached_prompt_tokens() == 8192,
         "late edits resume from the nearest earlier intermediate checkpoint");
  while (!late_edit.prefill_complete())
    (void)late_edit.Prefill(32768);
  late_edit.Commit();

  prompt[4500] = 3;
  auto earlier_edit = pool.Acquire(prompt, {}, {}, {}, true, 9900);
  Expect(
      earlier_edit.cache_hit() && earlier_edit.cached_prompt_tokens() == 4096,
      "committing a late edit preserves earlier usable checkpoints");
  earlier_edit.Invalidate();
  Expect(stats->states_created == 1,
         "intermediate checkpoint entries allocate no extra execution states");

  auto limited_stats = std::make_shared<FakeStats>();
  TextRunnerPool limited_pool(
      std::make_shared<LongSnapshotRunner>(limited_stats, 64, 256,
                                           3 * sizeof(FakeSnapshot)),
      1);
  std::vector<TextRunnerToken> root_prompt(1000, 1);
  auto root = limited_pool.Acquire(root_prompt);
  (void)root.Prefill(32768);
  root.Commit();
  std::vector<TextRunnerToken> extension(10000, 1);
  auto continued = limited_pool.Acquire(extension, {}, {}, {}, true, 9900);
  while (!continued.prefill_complete())
    (void)continued.Prefill(32768);
  continued.Commit();
  extension[1500] = 2;
  auto branched = limited_pool.Acquire(extension, {}, {}, {}, true, 9900);
  Expect(branched.cached_prompt_tokens() == root_prompt.size(),
         "intermediate admission never evicts the reused branching fallback");
  branched.Invalidate();

  auto cancel_stats = std::make_shared<FakeStats>();
  TextRunnerPool cancel_pool(
      std::make_shared<LongSnapshotRunner>(cancel_stats, 64, 256, 4096), 1);
  auto interrupted =
      cancel_pool.Acquire(std::vector<TextRunnerToken>(10000, 1));
  (void)interrupted.Prefill(32768);
  (void)interrupted.Prefill(1024);
  interrupted.Cancel();
  auto resumed = cancel_pool.Acquire(std::vector<TextRunnerToken>(10000, 1));
  Expect(resumed.cached_prompt_tokens() == 2048,
         "cancellation retains completed intermediate state, not partial work");
  resumed.Invalidate();

  auto short_stats = std::make_shared<FakeStats>();
  TextRunnerPool short_pool(
      std::make_shared<LongSnapshotRunner>(short_stats, 64, 256, 4096), 1);
  std::vector<TextRunnerToken> short_prompt(7900, 1);
  auto short_root = short_pool.Acquire(short_prompt);
  while (!short_root.prefill_complete())
    (void)short_root.Prefill(32768);
  short_root.Commit();
  const auto before_short = short_stats->snapshot_captures;
  short_stats->prefill_spans.clear();
  short_prompt.resize(8400, 1);
  auto short_turn = short_pool.Acquire(short_prompt);
  Expect(short_turn.cached_prompt_tokens() == 7900,
         "short continuation starts from the previous frontier");
  Expect(short_turn.Prefill(32768).decode_ready,
         "crossing a nearby grid point does not split a short continuation");
  short_turn.Commit();
  Expect(short_stats->prefill_spans == std::vector<std::size_t>{500} &&
             short_stats->snapshot_captures == before_short + 1,
         "short continuation captures only its completed prompt");
  short_prompt[7800] = 2;
  auto short_edit = short_pool.Acquire(short_prompt);
  Expect(short_edit.cached_prompt_tokens() == 6144,
         "skipping a redundant warm checkpoint retains earlier edit recovery");
  short_edit.Invalidate();

  auto aligned_stats = std::make_shared<FakeStats>();
  TextRunnerPool aligned_pool(
      std::make_shared<LongSnapshotRunner>(aligned_stats, 64, 256, 4096), 1);
  auto aligned_root =
      aligned_pool.Acquire(std::vector<TextRunnerToken>(1000, 1));
  (void)aligned_root.Prefill(32768);
  aligned_root.Commit();
  std::vector<TextRunnerToken> aligned_prompt(5000, 1);
  auto aligned = aligned_pool.Acquire(aligned_prompt, {}, {}, {}, true, 4096);
  while (!aligned.prefill_complete())
    (void)aligned.Prefill(32768);
  aligned.Commit();
  Expect(aligned_stats->snapshot_captures == 3,
         "a nearby grid point is skipped and the stable boundary is captured "
         "only once");
  aligned_prompt[4096] = 2;
  auto aligned_next =
      aligned_pool.Acquire(aligned_prompt, {}, {}, {}, true, 4096);
  Expect(aligned_next.cached_prompt_tokens() == 4096,
         "an aligned stable boundary remains reusable after assistant changes");
  aligned_next.Invalidate();

  auto tail_stats = std::make_shared<FakeStats>();
  TextRunnerPool tail_pool(
      std::make_shared<LongSnapshotRunner>(tail_stats, 64, 256, 4096), 1);
  auto near_end = tail_pool.Acquire(std::vector<TextRunnerToken>(2100, 1));
  Expect(near_end.Prefill(32768).decode_ready &&
             tail_stats->prefill_spans == std::vector<std::size_t>{2100},
         "a grid point near the prompt end does not split the final prefill");
  near_end.Invalidate();
  tail_stats->prefill_spans.clear();
  auto past_tail = tail_pool.Acquire(std::vector<TextRunnerToken>(2300, 2));
  while (!past_tail.prefill_complete())
    (void)past_tail.Prefill(32768);
  Expect(tail_stats->prefill_spans == std::vector<std::size_t>{2048, 252},
         "a grid point farther from the prompt end is still retained");
  past_tail.Invalidate();
}

/// A client that rewrites the assistant turn, as one that drops reasoning
/// does, diverges after the previous turn's stable boundary. Each warm turn
/// must therefore checkpoint its own boundary: if only the reused frontier is
/// retained, every later turn falls back to the same early position and the
/// re-prefilled tail grows for the rest of the conversation.
void TestWarmHitAdvancesTheStableCheckpoint() {
  auto stats = std::make_shared<FakeStats>();
  TextRunnerPool pool(std::make_shared<SnapshotRunner>(stats), 1);

  // Cold turn. The stable boundary is 3; 40,41 is the assistant framing.
  auto first = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!first.Prefill(64).decode_ready, "cold chat stops at its boundary");
  Expect(first.Prefill(64).decode_ready, "cold chat completes its suffix");
  const auto reply = first.SelectNext().token;
  first.Advance();
  first.Commit();

  // Warm turn replaying the assistant verbatim, then a new user turn. Its
  // stable boundary is 7, past the frontier of 6 that it reuses.
  auto second =
      pool.Acquire({1, 2, 3, 40, 41, reply, 7, 40, 41}, {}, {}, {}, true, 7);
  Expect(second.cached_prompt_tokens() == 6,
         "the warm turn resumes from the generated frontier");
  Expect(!second.Prefill(64).decode_ready,
         "the warm turn stops to checkpoint its stable boundary");
  Expect(second.Prefill(64).decode_ready, "the warm turn prefills its suffix");
  (void)second.SelectNext();
  second.Advance();
  second.Commit();

  // The client now drops the reasoning from that assistant turn, so 40,41
  // becomes 50,51 and the prompt diverges at token 7 -- exactly the previous
  // turn's stable boundary.
  auto third = pool.Acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 40, 41}, {},
                            {}, {}, true, 10);
  Expect(third.cached_prompt_tokens() == 7,
         "a warm turn checkpoints its own stable boundary, not only the "
         "frontier it reused");
  Expect(!third.Prefill(64).decode_ready,
         "the rewritten turn stops at its own boundary in turn");
  Expect(third.Prefill(64).decode_ready, "the rewritten turn prefills");
  (void)third.SelectNext();
  third.Advance();
  third.Commit();

  // Dropping reasoning again must keep advancing rather than falling back to
  // the original frontier for a third time.
  auto fourth =
      pool.Acquire({1, 2, 3, 40, 41, reply, 7, 50, 51, 8, 60, 61, 11, 40, 41},
                   {}, {}, {}, true, 13);
  Expect(fourth.cached_prompt_tokens() == 10,
         "successive rewritten turns resume from the latest boundary");
  fourth.Cancel();
}

void TestChatFallbackSurvivesSnapshotBudgetPressure() {
  auto stats = std::make_shared<FakeStats>();
  auto runner =
      std::make_shared<SnapshotRunner>(stats, 64, 256, sizeof(FakeSnapshot));
  TextRunnerPool pool(runner, 1);
  auto root = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!root.Prefill(64).decode_ready, "cold chat stops at its fallback");
  Expect(root.Prefill(64).decode_ready, "cold chat completes prefill");
  root.CapturePromptSnapshot();
  Expect(stats->snapshot_captures == 1,
         "a full prompt is not copied when only the fallback fits");
  const auto committed = root.Commit();
  Expect(committed.snapshot_bytes == sizeof(FakeSnapshot),
         "checkpoint retention respects the original one-snapshot budget");
  auto dropped = pool.Acquire({1, 2, 3, 50, 51, 9}, {}, {}, {}, true, 5);
  Expect(dropped.cached_prompt_tokens() == 3,
         "the optional full checkpoint never evicts the required fallback");
  dropped.Invalidate();
}

void TestFullChatCheckpointRestoresWithoutSuffixPrefill() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{.directory = directory.path(),
                                              .capacity_bytes = 8192,
                                              .staging_capacity_bytes = 4096,
                                              .min_checkpoint_step_tokens = 0};
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(stats, "artifact-A");
  {
    TextRunnerPool writer(runner, 1, disk_cache);
    auto prompt = writer.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    Expect(!prompt.Prefill(64).decode_ready, "writer saves the stable prefix");
    Expect(prompt.Prefill(64).decode_ready, "writer saves the complete prompt");
    (void)prompt.SelectNext();
    prompt.Advance();
    prompt.Commit();
  }
  TextRunnerPool reader(runner, 1, disk_cache);
  auto exact = reader.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(exact.cache_disk_hit() && exact.cached_prompt_tokens() == 5 &&
             exact.prefill_complete(),
         "disk restores the exact prompt instead of recomputing its suffix");
  const auto token = exact.SelectNext().token;
  exact.Advance();
  exact.Commit();
  auto repeated = reader.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!repeated.cache_disk_hit() && repeated.prefill_complete(),
         "a disk-restored full prompt is retained for cheap RAM retries");
  Expect(repeated.SelectNext().token == token,
         "disk and RAM checkpoints select the same token");
  repeated.Advance();
  repeated.Commit();
  auto changed = reader.Acquire({1, 2, 3, 50, 51, 60}, {}, {}, {}, true, 5);
  Expect(changed.cache_disk_hit() && changed.cached_prompt_tokens() == 3,
         "the earlier disk fallback survives exact full-prompt restoration");
  changed.Invalidate();

  TemporaryDirectory legacy_directory;
  const TextRunnerDiskCacheOptions legacy_disk{
      .directory = legacy_directory.path(),
      .capacity_bytes = 8192,
      .staging_capacity_bytes = 4096,
      .min_checkpoint_step_tokens = 0};
  {
    TextRunnerPool writer(runner, 1, legacy_disk);
    auto old = writer.Acquire({1, 2, 3, 40, 41});
    Expect(old.Prefill(64).decode_ready,
           "legacy writer saves only the full prompt");
    old.Commit();
  }
  TextRunnerPool migrated(runner, 1, legacy_disk);
  auto old = migrated.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
  Expect(!old.cache_hit(),
         "a legacy full-prompt file without a fallback still migrates safely");
  old.Invalidate();
}

void TestNewImageGetsAStableCheckpoint() {
  class ContextRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;
    void SetPromptContext(
        TextRunnerState&,
        std::shared_ptr<const gufo::server::TextPromptContext>) const override {
    }
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ContextRunner>(stats, 64, 256, 1024);
  TextRunnerPool pool(runner, 1);
  {
    auto text = pool.Acquire({1, 2, 3, 40, 41}, {}, {}, {}, true, 3);
    while (!text.prefill_complete())
      (void)text.Prefill(64);
    text.Commit();
  }
  auto image = std::make_shared<gufo::server::TextPromptContext>();
  image->cache_identity = {10};
  image->cache_prefixes = {{5, {}}};
  const std::vector<TextRunnerToken> prompt{1, 2, 3, 50, 51, 60, 61, 40, 41};
  {
    auto appended = pool.Acquire(prompt, {}, {}, image, true, 7);
    Expect(appended.cached_prompt_tokens() == 3,
           "new image restores the preceding text checkpoint");
    Expect(!appended.Prefill(64).decode_ready,
           "new image stops before mutable assistant framing");
    Expect(appended.Prefill(64).decode_ready,
           "assistant framing completes after saving the image checkpoint");
    appended.Commit();
  }
  {
    auto retry = pool.Acquire(prompt, {}, {}, image, true, 7);
    Expect(retry.prefill_complete() && retry.cached_prompt_tokens() == 9,
           "exact image retry keeps its complete prompt checkpoint");
    retry.Commit();
  }
  {
    auto followup = pool.Acquire({1, 2, 3, 50, 51, 60, 61, 52, 53, 40, 41}, {},
                                 {}, image, true, 9);
    Expect(followup.cached_prompt_tokens() == 7,
           "rewritten assistant framing does not reprocess the earlier image");
    Expect(!followup.Prefill(64).decode_ready,
           "the warm turn stops to checkpoint its own stable boundary");
    Expect(followup.Prefill(64).decode_ready,
           "assistant framing completes after the boundary is saved");
    followup.Commit();
  }
}

void TestSharedPrefixIsLearnedAndRestoredAcrossConversations() {
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk_cache{
      .directory = directory.path(),
      .capacity_bytes = 8192,
      .staging_capacity_bytes = 4096,
      .shared_prefix_min_tokens = 2,
      .shared_prefix_max_boundaries = 4,
  };
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  // Conversation A: system prefix {7, 7, 7} plus its own turn.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto request = pool.Acquire({7, 7, 7, 1, 2});
    Expect(!request.cache_hit(), "first conversation is cold");
    Expect(request.Prefill(8).consumed_tokens == 5,
           "nothing is shared yet, so prefill runs uninterrupted");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 0,
           "no shared prefix exists after one conversation");
  }

  // Conversation B shares only the system prefix. Prefill stops there so the
  // prefix is persisted for the next conversation.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto request = pool.Acquire({7, 7, 7, 3, 4, 5});
    Expect(!request.cache_hit(), "second conversation still has no prefix");
    const auto first = request.Prefill(8);
    Expect(first.consumed_tokens == 3 && !first.decode_ready,
           "prefill stops at the learned shared prefix");
    const auto second = request.Prefill(8);
    Expect(second.consumed_tokens == 3 && second.decode_ready,
           "prefill resumes after the boundary");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 1 &&
               commit.shared_prefix_bytes > 0 && commit.shared_prefix_ms >= 0.0,
           "the shared prefix was written once during prefill");
  }

  // Conversation C restores the shared prefix and prefills only its turn.
  {
    TextRunnerPool pool(runner, 1, disk_cache);
    auto exact = pool.Acquire({7, 7, 7});
    Expect(exact.cache_disk_hit() && exact.prefill_complete() &&
               exact.SelectNext().token == 90,
           "an exact shared-prefix restore has a current decode frontier");
    exact.Invalidate();
    auto request = pool.Acquire({7, 7, 7, 9});
    Expect(request.cache_hit() && request.cache_disk_hit() &&
               request.cached_prompt_tokens() == 3,
           "third conversation restores the shared prefix from disk");
    const auto step = request.Prefill(8);
    Expect(step.consumed_tokens == 1 && step.decode_ready,
           "only the conversation's own turn is prefilled");
    const auto commit = request.Commit();
    Expect(commit.shared_prefix_snapshots == 0,
           "a restored prefix is not written again");
  }
}

void TestRamLearnsDivergenceBoundaries() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "ram-boundary", std::size_t{1} << 20, 4096);
  TextRunnerPool pool(runner, 1);
  // Conversations share a system prompt, then each has its own tail. Below
  // 2048 tokens no grid checkpoint exists, so only learning can help.
  const auto conversation = [](std::size_t shared, TextRunnerToken tail) {
    std::vector<TextRunnerToken> prompt;
    for (std::size_t index = 0; index < shared; ++index)
      prompt.push_back(static_cast<TextRunnerToken>(100 + index % 50));
    for (TextRunnerToken index = 0; index < 600; ++index)
      prompt.push_back(tail + index);
    return prompt;
  };
  const auto run = [&](const std::vector<TextRunnerToken>& prompt) {
    auto request = pool.Acquire(prompt);
    const auto cached = request.cached_prompt_tokens();
    std::vector<std::size_t> steps;
    while (!request.prefill_complete())
      steps.push_back(request.Prefill(4096).consumed_tokens);
    (void)request.Commit();
    return std::pair{cached, steps};
  };

  const auto [first_cached, first_steps] = run(conversation(1000, 10000));
  Expect(first_cached == 0 && first_steps == std::vector<std::size_t>{1600},
         "the first conversation has nothing to share");
  const auto [second_cached, second_steps] = run(conversation(1000, 20000));
  Expect(second_cached == 0 &&
             second_steps == std::vector<std::size_t>({1000, 600}),
         "the second conversation stops at the divergence point to retain it");
  const auto [third_cached, third_steps] = run(conversation(1000, 30000));
  Expect(third_cached == 1000 && third_steps == std::vector<std::size_t>{600},
         "later conversations restore the learned boundary exactly");
  const auto [short_cached, short_steps] = run(conversation(300, 40000));
  Expect(short_cached == 0 && short_steps == std::vector<std::size_t>{900},
         "a short shared prefix is not worth an extra checkpoint");
  // An edit near the end diverges inside this request's own final tokens;
  // its stable checkpoint covers that, so no extra copy is taken.
  auto edited = conversation(1000, 30000);
  edited.back() += 1;
  auto request = pool.Acquire(edited);
  Expect(request.cached_prompt_tokens() == 1000 &&
             request.Prefill(4096).consumed_tokens == 600,
         "a late divergence does not stop prefill for another checkpoint");
  request.Invalidate();
}

void TestCoincidentCacheBoundariesShareOneCopy() {
  for (const bool stable : {false, true}) {
    TemporaryDirectory directory;
    const TextRunnerDiskCacheOptions disk_cache{
        .directory = directory.path(),
        .capacity_bytes = 1024 * 1024,
        .staging_capacity_bytes = 64 * 1024,
        .shared_prefix_min_tokens = 2048,
    };
    auto stats = std::make_shared<FakeStats>();
    auto runner = std::make_shared<PersistentSnapshotRunner>(
        stats, "artifact-A", 1024, 4096);
    std::vector<TextRunnerToken> prompt(2050, 7);
    prompt[2048] = 1;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto source = pool.Acquire(prompt);
      while (!source.prefill_complete())
        (void)source.Prefill(4096);
      source.Commit();
    }
    prompt[2048] = 2;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto branch = pool.Acquire(prompt, {}, {}, {}, true, stable ? 2048 : 0);
      const auto before = stats->snapshot_captures;
      Expect(branch.Prefill(4096).consumed_tokens == 2048,
             "disk boundary coincides with a history or stable checkpoint");
      Expect(branch.Prefill(4096).consumed_tokens == 2 &&
                 stats->snapshot_captures == before + 1,
             "one immutable copy serves both coincident boundaries");
      branch.Commit();
    }
    prompt[2048] = 3;
    {
      TextRunnerPool pool(runner, 1, disk_cache);
      auto restored = pool.Acquire(prompt);
      Expect(restored.cache_disk_hit() &&
                 restored.cached_prompt_tokens() == 2048 &&
                 restored.Prefill(4096).consumed_tokens == 2,
             "the shared copy survives a restart with its exact position");
      restored.Commit();
    }
  }
}

void TestDiskOnlyCaptureReservesBudgetBeforeCommit() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  std::binary_semaphore entered(0), release(0);
  runner->before_serialize = [&] {
    entered.release();
    release.acquire();
  };
  TextRunnerPool pool(runner, 2,
                      TextRunnerDiskCacheOptions{
                          .directory = directory.path(),
                          .capacity_bytes = 4096,
                          .staging_capacity_bytes = 192,
                      });
  auto first = pool.Acquire({1, 2, 3});
  (void)first.Prefill(3);
  first.CapturePromptSnapshot();
  Expect(entered.try_acquire_for(std::chrono::seconds(2)),
         "disk-only prompt is queued immediately after capture");
  auto second = pool.Acquire({4, 5, 6});
  (void)second.Prefill(3);
  second.CapturePromptSnapshot();
  const auto second_commit = second.Commit();
  Expect(stats->snapshot_captures == 1 && second_commit.disk_queued_bytes == 0,
         "pending disk-only state prevents another unbudgeted capture");
  release.release();
  Expect(first.Commit().disk_queued_bytes != 0,
         "request reports its earlier background persistence admission");
}

void TestDiskPreflightAvoidsUnusableCapture() {
  TemporaryDirectory directory;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<PersistentSnapshotRunner>(
      stats, "artifact-A", sizeof(FakeSnapshot) - 1);
  TextRunnerPool pool(runner, 1,
                      TextRunnerDiskCacheOptions{
                          .directory = directory.path(),
                          .capacity_bytes = 4096,
                          .staging_capacity_bytes = 96,
                      });
  auto request = pool.Acquire({1, 2, 3});
  (void)request.Prefill(3);
  const auto commit = request.Commit();
  Expect(stats->snapshot_captures == 0 && commit.disk_queued_bytes == 0,
         "disk staging refusal happens before snapshot capture");
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

void TestSnapshotBudgetRefusalDoesNotFailCompletedRequest() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats, 64, 256,
                                                 sizeof(FakeSnapshot) - 1);
  TextRunnerPool pool(runner, 1);

  auto request = pool.Acquire({1, 2, 3});
  Expect(request.Prefill(3).decode_ready,
         "request completes before optional snapshot retention");
  const auto commit = request.Commit();
  Expect(commit.snapshot_bytes == 0 && commit.snapshot_ms == 0.0,
         "budget refusal succeeds without reporting a retained snapshot");
  Expect(stats->snapshot_size_queries == 1 && stats->snapshot_captures == 0,
         "cache admission happens before snapshot allocation");

  auto extension = pool.Acquire({1, 2, 3, 4});
  Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 3 &&
             extension.cache_restore_bytes() == 0,
         "live state remains reusable when snapshot admission is refused");
  extension.Invalidate();
}

class FlakySnapshotRunner final : public SnapshotRunner {
public:
  using SnapshotRunner::SnapshotRunner;

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    ++stats_->snapshot_captures;
    if (stats_->snapshot_captures == 1) {
      throw std::runtime_error("synthetic snapshot capture failure");
    }
    const auto& fake = RequireFakeState(state);
    return std::make_unique<FakeSnapshot>(fake.position, fake.decode_count,
                                          fake.frontier);
  }
};

void TestSnapshotCaptureFailureReleasesReservationAndKeepsRequestSuccessful() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<FlakySnapshotRunner>(stats);
  TextRunnerPool pool(runner, 1);

  {
    auto first = pool.Acquire({1, 2, 3});
    Expect(first.Prefill(3).decode_ready, "first request reaches checkpoint");
    const auto commit = first.Commit();
    Expect(commit.snapshot_bytes == 0 && commit.snapshot_ms >= 0.0,
           "snapshot exception does not fail the completed request");
  }

  {
    auto second = pool.Acquire({4, 5});
    Expect(!second.cache_hit(), "failed capture retained no partial entry");
    Expect(second.Prefill(2).decode_ready,
           "second request executes normally after capture failure");
    const auto commit = second.Commit();
    Expect(commit.snapshot_bytes == sizeof(FakeSnapshot),
           "released reservation admits a later successful snapshot");
  }

  auto extension = pool.Acquire({4, 5, 6});
  Expect(extension.cache_hit() && extension.cached_prompt_tokens() == 2,
         "later retained snapshot restores after the failed attempt");
  extension.Invalidate();
}

/// Evicting a retained prefix because the entry table is full is not routine:
/// it means the server is configured below its workload and is doing avoidable
/// full re-prefills. It must be visible, unlike exact replacement.
void TestEntryCapacityEvictionIsLogged() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats, 64, 256, 8192);
  // Checkpoint records are bounded independently of execution sessions.
  TextRunnerPool pool(runner, 1);

  for (std::size_t i = 0;
       i <= gufo::server::TextRunnerRamCacheOptions::kMaxEntries; ++i) {
    const std::vector<TextRunnerToken> prefix{
        static_cast<TextRunnerToken>(i + 1), 0};
    auto request = pool.Acquire(prefix);
    Expect(request.Prefill(prefix.size()).decode_ready,
           "each distinct prefix reaches its snapshot boundary");
    (void)request.SelectNext();
    request.Advance();
    Expect(request.Commit().snapshot_bytes == sizeof(FakeSnapshot),
           "each distinct prefix retains a snapshot");
  }

  // The first prefix was evicted, so it can no longer be reused.
  auto evicted = pool.Acquire({1, 0});
  Expect(!evicted.cache_hit(),
         "the oldest retained prefix is gone once the entries are full");
  evicted.Invalidate();
}

void TestSnapshotCacheCapacityIsReportedAtStartup() {
  using gufo::server::TextRunnerRamCacheOptions;
  // Automatic sizing takes the model budget; an explicit limit may use the
  // larger ceiling the model reports, or the budget when it reports none.
  for (const std::optional<std::size_t> ceiling :
       {std::optional<std::size_t>{}, std::optional{std::size_t{128} << 30}}) {
    for (const std::size_t budget :
         {std::size_t{0}, std::size_t{256}, std::size_t{64} << 30}) {
      for (const std::size_t requested :
           {std::size_t{0}, std::size_t{64}, std::size_t{48} << 30,
            std::size_t{96} << 30}) {
        for (const std::size_t sessions : {1U, 2U}) {
          auto stats = std::make_shared<FakeStats>();
          auto runner =
              std::make_shared<SnapshotRunner>(stats, 64, 256, budget);
          runner->retained_snapshot_ceiling_bytes = ceiling;
          std::ostringstream startup_log;
          auto* previous = std::clog.rdbuf(startup_log.rdbuf());
          {
            TextRunnerPool pool(runner, sessions, std::nullopt,
                                {.capacity_bytes = requested});
          }
          std::clog.rdbuf(previous);
          const auto output = startup_log.str();
          const auto automatic =
              std::min(budget, TextRunnerRamCacheOptions::kAutomaticMaxBytes);
          const auto maximum = ceiling.value_or(budget);
          const auto capacity =
              requested == 0 ? automatic : std::min(requested, maximum);
          const std::string expected =
              "event=snapshot_cache_configured sessions=" +
              std::to_string(sessions) +
              " snapshot_entries=128 capacity_bytes=" +
              std::to_string(capacity) +
              " automatic_bytes=" + std::to_string(automatic) +
              " max_bytes=" + std::to_string(maximum) + "\n";
          const auto position = output.find(expected);
          Expect(position != std::string::npos &&
                     output.find(expected, position + expected.size()) ==
                         std::string::npos,
                 "startup reports actual session, entry and byte limits once");
          Expect(output.find("retained_conversations") == std::string::npos,
                 "startup does not present session count as conversation "
                 "capacity");
          Expect(stats->states_created == sessions,
                 "checkpoint record capacity never creates extra execution "
                 "states");
        }
      }
    }
  }
}

void TestSnapshotStartupReportsSelectedLimits() {
  class ChangingHeadroomRunner final : public SnapshotRunner {
  public:
    using SnapshotRunner::SnapshotRunner;

    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = SnapshotRunner::ResourceClaim();
      constexpr std::size_t gib = std::size_t{1} << 30;
      // Synthetic headroom falls during state allocation, then again after
      // cache sizing. No large buffers are allocated for these resource claims.
      // Each claim follows Flash-Next's half-free / free-minus-4-GiB policy.
      const auto available = stats_->states_created == 0 ? 14 * gib
                             : post_state_claims++ == 0  ? 12 * gib
                                                         : 11 * gib;
      claim.retained_snapshot_capacity_bytes = available / 2;
      claim.retained_snapshot_ceiling_bytes =
          available - gufo::server::kHostSnapshotHeadroomBytes;
      return claim;
    }

    mutable std::size_t post_state_claims{0};
  };

  constexpr std::size_t gib = std::size_t{1} << 30;
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<ChangingHeadroomRunner>(stats);
  std::ostringstream startup_log;
  auto* previous = std::clog.rdbuf(startup_log.rdbuf());
  {
    TextRunnerPool pool(runner, 1, std::nullopt, {.capacity_bytes = 8 * gib});
  }
  std::clog.rdbuf(previous);

  const auto output = startup_log.str();
  const auto event = output.find("event=snapshot_cache_configured ");
  Expect(event != std::string::npos, "startup reports snapshot limits");
  const auto line = output.substr(event, output.find('\n', event) - event);
  const auto field = [&](std::string_view name) {
    const auto start = line.find(name);
    Expect(start != std::string::npos, "startup reports the requested field");
    std::istringstream value(line.substr(start + name.size()));
    std::size_t bytes = 0;
    Expect(static_cast<bool>(value >> bytes), "startup byte field is numeric");
    return bytes;
  };
  const auto capacity = field("capacity_bytes=");
  const auto automatic = field("automatic_bytes=");
  const auto maximum = field("max_bytes=");
  Expect(stats->states_created == 1, "only the requested state was created");
  Expect(capacity == 8 * gib,
         "cache capacity uses the first post-allocation memory observation");
  Expect(capacity <= maximum,
         "startup maximum must not be below the selected cache capacity");
  Expect(automatic == 6 * gib && maximum == 8 * gib,
         "startup limits describe the claim that selected cache capacity");
}

void TestServerInstructionsAreFraming() {
  using namespace gufo::tokenization;
  class ConstraintRunner final : public FakeRunner {
  public:
    explicit ConstraintRunner(gufo::sampling::JsonConstraint::ToolFormat format)
        : FakeRunner(std::make_shared<FakeStats>()), format_(format) {}
    gufo::sampling::JsonConstraint::ToolFormat ToolFormat() const override {
      return format_;
    }
    std::shared_ptr<const gufo::sampling::ConstraintVocabulary>
    BuildConstraintVocabulary() const override {
      return std::make_shared<gufo::sampling::ConstraintVocabulary>(
          256, [](std::uint32_t id) {
            return gufo::sampling::ConstraintVocabulary::Piece{
                std::string(1, static_cast<char>(id)), false};
          });
    }

  private:
    gufo::sampling::JsonConstraint::ToolFormat format_;
  };
  // Only runners without a native call syntax use the JSON envelope and its
  // instruction; a native runner never switches syntax for a schema (#383).
  const ConstraintRunner runner(
      gufo::sampling::JsonConstraint::ToolFormat::kJson);
  const ConstraintRunner native_runner(
      gufo::sampling::JsonConstraint::ToolFormat::kQwen);
  for (const auto format :
       {gufo::sampling::JsonConstraint::ToolFormat::kQwen,
        gufo::sampling::JsonConstraint::ToolFormat::kDeepSeek}) {
    const ConstraintRunner plain_runner(format);
    for (const bool json : {false, true}) {
      ChatRequest request;
      if (json)
        request.response_format = gufo::sampling::JsonConstraint::Compile(
            gufo::json::parse(
                R"({"type":"object","properties":{},"additionalProperties":false})"),
            false);
      gufo::sampling::SamplingConfig sampling;
      std::optional<gufo::sampling::JsonConstraint::ToolFormat> observed;
      const auto constrained = gufo::server::ConstrainChatRequest(
          request, plain_runner, &sampling, &observed);
      Expect(observed == format && constrained.has_value() == json,
             "plain and JSON answers retain the native output dialect");
    }
  }
  std::vector<std::string> vocab;
  for (int i = 0; i < 256; ++i)
    vocab.emplace_back(1, static_cast<char>(i));
  std::unordered_map<std::string, TokenId> specials;
  for (const auto* token : {"<|im_start|>", "<|im_end|>", "<think>", "</think>",
                            "<tool_call>", "</tool_call>"}) {
    specials.emplace(token, vocab.size());
    vocab.emplace_back(token);
  }
  std::string error;
  auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, {}, specials, &error);
  Expect(tokenizer != nullptr, "instruction tokenizer fixture initializes");
  ChatTemplateOptions options;
  options.enable_thinking = false;
  options.require_tool_call = true;
  for (
      const auto schema :
      {R"({"type":"object","properties":{"text":{"type":"string","const":"\n</parameter>\n</｜DSML｜parameter>\\"}},"required":["text"],"additionalProperties":false})",
       R"({"type":"object","properties":{"text":{"type":"string"}},"patternProperties":{"^x_":{"type":"integer"}},"required":["text"]})"}) {
    for (const auto role :
         {ChatRole::kUser, ChatRole::kSystem, ChatRole::kDeveloper}) {
      for (const bool literal_client_tags : {false, true}) {
        ChatRequest request;
        request.reasoning.enabled = false;
        request.tool_choice = ChatRequest::ToolChoice::kRequired;
        request.tools = {{.name = "record", .parameters_json = schema}};
        const std::string client = literal_client_tags
                                       ? "Client <tool_call>example</tool_call>"
                                       : "Be concise.  ";
        if (role != ChatRole::kUser)
          request.messages.emplace_back(role, client);
        request.messages.emplace_back(
            ChatRole::kUser, role == ChatRole::kUser ? client : "Call record.");
        gufo::sampling::SamplingConfig sampling;
        std::optional<gufo::sampling::JsonConstraint::ToolFormat> format;
        const auto constrained = gufo::server::ConstrainChatRequest(
            request, runner, &sampling, &format);
        Expect(
            constrained &&
                format == gufo::sampling::JsonConstraint::ToolFormat::kJson &&
                sampling.constraint,
            "the actual request path selects and binds the JSON fallback");
        Expect(constrained->messages.front().content ==
                   (role == ChatRole::kUser ? "" : client),
               "server instructions do not mutate client system/developer "
               "content");
        Expect(request.messages.front().content == client,
               "constraint construction leaves the original request untouched");
        const auto& instruction = constrained->messages.front().framing_suffix;
        Expect(instruction.find("<tool_call>") != std::string::npos &&
                   instruction.find("</tool_call>") != std::string::npos,
               "JSON fallback uses the server framing field");
        const auto prepared = gufo::models::qwen::vision::Prepare(
            *tokenizer, constrained->messages, constrained->tools, options, {},
            8192);
        auto without = constrained->messages;
        without.front().framing_suffix.clear();
        const auto baseline = QwenChatTemplate::RenderAndTokenize(
            *tokenizer, without, constrained->tools, options);
        Expect(baseline.has_value(), "client-only template encodes");
        for (const auto* tag : {"<tool_call>", "</tool_call>"}) {
          const auto id = *tokenizer->FindSpecialToken(tag);
          Expect(
              std::count(prepared.tokens.begin(), prepared.tokens.end(), id) ==
                  std::count(baseline->begin(), baseline->end(), id) + 1,
              "each server delimiter is exactly one special token, including "
              "beside literal client tags");
        }
        const auto client_only = QwenChatTemplate::RenderAndTokenize(
            *tokenizer, request.messages, options);
        Expect(client_only.has_value(), "client tag control encodes");
        for (const auto* tag : {"<tool_call>", "</tool_call>"})
          Expect(std::count(client_only->begin(), client_only->end(),
                            *tokenizer->FindSpecialToken(tag)) == 0,
                 "client-supplied delimiter spellings remain ordinary text");
        if (!literal_client_tags) {
          auto legacy = constrained->messages;
          legacy.front().content += legacy.front().framing_suffix;
          legacy.front().framing_suffix.clear();
          const auto rendered =
              QwenChatTemplate::Render(legacy, constrained->tools, options);
          TokenizerOptions framing;
          framing.parse_special_tokens = true;
          Expect(rendered &&
                     prepared.tokens == tokenizer->Encode(*rendered, framing),
                 "the full fallback prompt exactly matches main's legacy "
                 "instruction tokenization");
        }
      }
    }
  }
  // JSON-object prompts, schema prompts and their optional descriptions use
  // the same framing field. Native tools add no instruction or prompt work.
  for (const bool schema : {false, true}) {
    ChatRequest request(
        {{ChatRole::kSystem, "Client <tool_call>literal</tool_call>"},
         {ChatRole::kUser, "Return JSON."}});
    request.reasoning.enabled = false;
    request.response_format =
        schema
            ? gufo::sampling::JsonConstraint::Compile(
                  gufo::json::parse(
                      R"({"type":"object","properties":{"text":{"type":"string","const":"<tool_call>"}},"required":["text"],"additionalProperties":false})"),
                  true)
            : gufo::sampling::JsonConstraint::Object();
    request.response_format_description = "Describe <tool_call> in the schema.";
    gufo::sampling::SamplingConfig sampling;
    auto constrained =
        gufo::server::ConstrainChatRequest(request, runner, &sampling);
    Expect(constrained &&
               constrained->messages.front().content ==
                   request.messages.front().content &&
               constrained->messages.front().framing_suffix.find(
                   request.response_format_description + "\n\n" +
                   request.response_format->prompt()) != std::string::npos,
           "response formats and descriptions are separated from client "
           "message content");
    const auto tokens = QwenChatTemplate::RenderAndTokenize(
        *tokenizer, constrained->messages, options);
    Expect(tokens && std::count(tokens->begin(), tokens->end(),
                                *tokenizer->FindSpecialToken("<tool_call>")) ==
                         (schema ? 2 : 1),
           "generated description/schema instruction tags retain framing token "
           "identity");
  }
  ChatRequest native({{ChatRole::kUser, "Call record."}});
  native.reasoning.enabled = false;
  native.tools = {
      {.name = "record",
       .parameters_json =
           R"({"type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false})"}};
  gufo::sampling::SamplingConfig sampling;
  const auto constrained =
      gufo::server::ConstrainChatRequest(native, native_runner, &sampling);
  Expect(constrained && constrained->messages.front().framing_suffix.empty(),
         "native constraints add no instruction or change to the prompt");
  // Schemas native tags cannot enforce exactly, beside an ordinary neighbor,
  // under every tool choice: the request stays native as in llama.cpp, so the
  // prompt is the client's own and needs no extra prefill.
  for (
      const auto* schema :
      {R"({"type":"object","properties":{"text":{"type":"string","const":"\n</parameter>\n"}},"required":["text"],"additionalProperties":false})",
       R"({"type":"object","properties":{"text":{"type":"string"}},"patternProperties":{"^x_":{"type":"integer"}},"required":["text"]})",
       R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"text":{"type":"string"}},"required":["text"]})",
       R"({"type":"object","properties":{"date":{"type":"string","pattern":"^[0-9]{4}$"}}})",
       R"({"type":"object","properties":{"v":{"oneOf":[{"type":"string"},{"type":"integer"}]}}})",
       R"({"type":"object","properties":{"v":{"allOf":[{"type":"string"},{"minLength":1}]}}})",
       R"({"type":"object","properties":{"v":{"not":{"type":"null"}}}})",
       R"({"type":"object","properties":{"v":{"type":"string"}},"additionalProperties":true})",
       R"({"type":"object","properties":{"o":{"type":"object","properties":{"x":{"type":"integer"}}}}})"}) {
    for (const auto choice :
         {ChatRequest::ToolChoice::kAuto, ChatRequest::ToolChoice::kRequired}) {
      for (const bool strict : {false, true}) {
        ChatRequest request({{ChatRole::kSystem, "Be concise."},
                             {ChatRole::kUser, "Call record."}});
        request.reasoning.enabled = false;
        request.tool_choice = choice;
        const std::string definition =
            std::string(
                R"({"type":"function","function":{"name":"record","strict":)") +
            (strict ? "true" : "false") + R"(,"parameters":)" + schema + "}}";
        request.tools = {
            {.name = "bash",
             .parameters_json =
                 R"({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]})"},
            {.name = "record",
             .parameters_json = schema,
             .definition_json = definition}};
        gufo::sampling::SamplingConfig sampled;
        std::optional<gufo::sampling::JsonConstraint::ToolFormat> format;
        std::optional<ChatRequest> result;
        try {
          result = gufo::server::ConstrainChatRequest(request, native_runner,
                                                      &sampled, &format);
        } catch (const std::invalid_argument&) {
          // An impossible strict schema is rejected before generation.
          Expect(strict, "only strict schemas may be rejected");
          continue;
        }
        Expect(
            result &&
                format == gufo::sampling::JsonConstraint::ToolFormat::kQwen &&
                sampled.constraint,
            "a native runner keeps native calls for every schema");
        Expect(result->messages.front().framing_suffix.empty() &&
                   result->messages.front().content == "Be concise.",
               "a native runner adds no tool instruction to the prompt");
      }
    }
  }
}

}  // namespace

int main() {
  TestServerInstructionsAreFraming();
  // The cache warning assertion in this binary matches the plain "[WARN]
  // [cache]" text captured from a redirected sink; a TTY stderr tints it.
  ::setenv("NO_COLOR", "1", 1);
  TestInPassStableCheckpoint();
  TestInPassFailureRetainsOnlyCompletedCheckpoints();
  TestNewImageGetsAStableCheckpoint();
  TestGeneratedFrontierForksBeforeMutation();
  TestGeneratedFrontierPersistsForForks();
  TestCancellationRetainsOnlyCompletedWork();
  TestPromptReuseCanBeDisabledPerRequest();
  TestStableChatPrefixSurvivesInterruptedFraming();
  TestWarmChatCheckpointsStopAtTheStableBoundary();
  TestWarmHitAdvancesTheStableCheckpoint();
  TestHistoryEditsRestoreIntermediateCheckpoints();
  TestChatFallbackSurvivesSnapshotBudgetPressure();
  TestFullChatCheckpointRestoresWithoutSuffixPrefill();
  TestDiskOnlyCaptureReservesBudgetBeforeCommit();
  TestDiskPreflightAvoidsUnusableCapture();
  TestBoundedPrefillDecodeAndPrefixReuse();
  TestAbandonedRequestRollsBackState();
  TestRequestBindsAndClearsCancellation();
  TestBatchedAdvancePreservesIndependentRequests();
  TestResourceClaimsAreValidatedBeforeAllocation();
  TestSnapshotForkAndUnsupportedCapabilities();
  TestSnapshotCacheBranchesOnePrefixIntoIndependentStates();
  std::ostringstream normal_log;
  auto* previous = std::clog.rdbuf(normal_log.rdbuf());
  TestSnapshotRetentionUsesPromptBoundary();
  std::clog.rdbuf(previous);
  Expect(normal_log.str().find("action=removed") == std::string::npos &&
             normal_log.str().find("action=skipped") == std::string::npos,
         "routine cache replacement stays quiet");

  std::ostringstream eviction_log;
  previous = std::clog.rdbuf(eviction_log.rdbuf());
  TestEntryCapacityEvictionIsLogged();
  std::clog.rdbuf(previous);
  Expect(
      eviction_log.str().find("action=removed") != std::string::npos &&
          eviction_log.str().find("reason=entry_capacity") != std::string::npos,
      "evicting a retained prefix for entry capacity is reported");

  TestSnapshotCacheCapacityIsReportedAtStartup();
  TestSnapshotStartupReportsSelectedLimits();
  TestPersistentSnapshotRestoresAcrossPools();
  TestSharedPrefixIsLearnedAndRestoredAcrossConversations();
  TestRamLearnsDivergenceBoundaries();
  TestCoincidentCacheBoundariesShareOneCopy();
  TestMeasuredStateIsReconciledWithClaim();
  TestSnapshotBudgetRefusalDoesNotFailCompletedRequest();
  std::ostringstream failure_log;
  previous = std::clog.rdbuf(failure_log.rdbuf());
  TestSnapshotCaptureFailureReleasesReservationAndKeepsRequestSuccessful();
  std::clog.rdbuf(previous);
  Expect(
      failure_log.str().find("[WARN] [cache]") != std::string::npos &&
          failure_log.str().find("reason=capture_failure") != std::string::npos,
      "cache fallback retains an actionable warning");
  std::cout << "All text model runner tests passed\n";
  return 0;
}
