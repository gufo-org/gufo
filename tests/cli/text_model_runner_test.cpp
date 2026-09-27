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
#include <utility>
#include <vector>

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

class PersistentSnapshotRunner final : public SnapshotRunner {
public:
  std::function<void()> before_serialize;
  PersistentSnapshotRunner(std::shared_ptr<FakeStats> stats,
                           std::string identity,
                           std::size_t retained_snapshot_capacity_bytes = 256)
      : SnapshotRunner(std::move(stats), 64, 256,
                       retained_snapshot_capacity_bytes),
        identity_(identity.begin(), identity.end()) {}

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    auto descriptor = SnapshotRunner::Descriptor();
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
  TemporaryDirectory directory;
  const TextRunnerDiskCacheOptions disk{.directory = directory.path(),
                                        .capacity_bytes = 4096,
                                        .staging_capacity_bytes = 4096};
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

}  // namespace

int main() {
  TestGeneratedFrontierForksBeforeMutation();
  TestGeneratedFrontierPersistsForForks();
  TestCancellationRetainsOnlyCompletedWork();
  TestPromptReuseCanBeDisabledPerRequest();
  TestStableChatPrefixSurvivesInterruptedFraming();
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
  Expect(normal_log.str().empty(), "routine cache replacement stays quiet");
  TestPersistentSnapshotRestoresAcrossPools();
  TestSharedPrefixIsLearnedAndRestoredAcrossConversations();
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
