#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
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
    std::array<char, 96> pattern{};
    const std::string path = (std::filesystem::temp_directory_path() /
                              "gufo-runner-disk-cache-XXXXXX")
                                 .string();
    Expect(path.size() + 1 <= pattern.size(),
           "temporary path fits fixed buffer");
    std::copy(path.begin(), path.end(), pattern.begin());
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
  Expect(first.cache_restore_bytes() == sizeof(FakeSnapshot) &&
             second.cache_restore_bytes() == sizeof(FakeSnapshot) &&
             first.cache_restore_ms() >= 0.0 &&
             second.cache_restore_ms() >= 0.0,
         "both branches report full-copy restore bytes and latency");
  Expect(stats->snapshot_restores == 2,
         "the root snapshot is copied into two mutable states");
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
    Expect(commit.snapshot_bytes == 0 && commit.disk_write_bytes > 0 &&
               commit.disk_write_ms >= 0.0,
           "disk publication is independent of RAM snapshot admission");
  }

  auto reader_stats = std::make_shared<FakeStats>();
  auto reader =
      std::make_shared<PersistentSnapshotRunner>(reader_stats, "artifact-A");
  TextRunnerPool restarted(reader, 1, disk_cache);
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
  Expect(!extension.cache_hit(),
         "a refused snapshot becomes an ordinary deterministic miss");
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
  TestBoundedPrefillDecodeAndPrefixReuse();
  TestAbandonedRequestRollsBackState();
  TestRequestBindsAndClearsCancellation();
  TestBatchedAdvancePreservesIndependentRequests();
  TestResourceClaimsAreValidatedBeforeAllocation();
  TestSnapshotForkAndUnsupportedCapabilities();
  TestSnapshotCacheBranchesOnePrefixIntoIndependentStates();
  TestPersistentSnapshotRestoresAcrossPools();
  TestMeasuredStateIsReconciledWithClaim();
  TestSnapshotBudgetRefusalDoesNotFailCompletedRequest();
  TestSnapshotCaptureFailureReleasesReservationAndKeepsRequestSuccessful();
  std::cout << "All text model runner tests passed\n";
  return 0;
}
