#include "src/cli/serve/text_model_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#include "src/core/platform/env.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/families/qwen/control_tokens/control_tokens.hpp"
#include "src/families/qwen/vision/prompt.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/cli/fake_cache_resources.hpp"
#include "tests/support/temporary_directory.hpp"

using gufo::tokenization::kImEnd;
using gufo::tokenization::kImStart;

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

// What a checkpoint restores besides the adapter's rows.
struct FakeFrontier {
  std::size_t position{0};
  std::size_t decode_count{0};
  std::optional<TextRunnerToken> frontier;
};

class ObservedAdapter final : public gufo::cache::testing::FakeAdapter {
public:
  explicit ObservedAdapter(std::shared_ptr<FakeStats> stats)
      : FakeAdapter(16), stats_(std::move(stats)) {}
  std::unique_ptr<gufo::cache::Slot> CreateSlot(
      gufo::cache::MutationGuard& guard) override {
    ++stats_->states_created;
    return FakeAdapter::CreateSlot(guard);
  }
  void BeginRestore(
      gufo::cache::Slot& slot,
      std::span<const gufo::cache::ComponentPosition> positions) override {
    ++stats_->snapshot_restores;
    FakeAdapter::BeginRestore(slot, positions);
  }
  gufo::cache::Completion CapturePrivatePiece(
      const gufo::cache::Slot& slot, gufo::cache::ComponentId id,
      std::size_t offset, std::span<std::byte> bytes,
      gufo::cache::Stream& stream) override {
    if (offset == 0)
      ++stats_->snapshot_captures;
    return FakeAdapter::CapturePrivatePiece(slot, id, offset, bytes, stream);
  }
  bool Invalidate(gufo::cache::Slot& slot) noexcept override {
    ++stats_->invalidations;
    return FakeAdapter::Invalidate(slot);
  }

private:
  std::shared_ptr<FakeStats> stats_;
};

class FakeState final : public TextRunnerState, public FakeFrontier {
public:
  FakeState(std::shared_ptr<FakeStats> stats, std::size_t measured_bytes,
            gufo::cache::testing::FakeAdapter& adapter, gufo::cache::Slot& slot)
      : adapter(adapter),
        slot(slot),
        stats_(std::move(stats)),
        measured_bytes_(measured_bytes) {}

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    cancellation_armed = static_cast<bool>(is_cancelled);
    if (is_cancelled) {
      ++stats_->cancellation_bindings;
    } else {
      ++stats_->cancellation_clears;
    }
  }

  bool cancellation_armed{false};

  void Invalidate() noexcept override {
    (void)adapter.Invalidate(slot);
    static_cast<FakeFrontier&>(*this) = {};
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {
        .per_request_state_bytes = measured_bytes_,
        .temporary_scratch_bytes = 16,
    };
  }

  gufo::cache::testing::FakeAdapter& adapter;
  gufo::cache::Slot& slot;

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
             std::size_t retained_snapshot_capacity_bytes = 1U << 25)
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
    throw std::logic_error("slots are owned by the component cache");
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }
  [[nodiscard]] std::unique_ptr<gufo::server::ComponentCacheResources>
  CreateComponentCacheResources(gufo::cache::ResourceLedger& ledger,
                                std::size_t) const override {
    return gufo::server::testing::MakeFakeCacheResources(
        ledger, std::make_unique<ObservedAdapter>(stats_),
        {.private_blocks = 512,
         .row_backing_bytes = 64,
         .total_row_bytes = 1U << 20});
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      gufo::cache::Adapter& adapter, gufo::cache::Slot& slot) const override {
    auto state = std::make_unique<FakeState>(
        stats_, measured_bytes_,
        dynamic_cast<gufo::cache::testing::FakeAdapter&>(adapter), slot);
    Recall(*state);
    return state;
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t) const override {
    Recall(RequireFakeState(state));
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
    const auto rows = prompt.subspan(offset, consumed);
    fake.adapter.Append(fake.slot, rows, rows);
    fake.position += consumed;
    const bool ready = fake.position == prompt.size();
    if (ready) {
      fake.frontier = 90;
    }
    Remember(fake);
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
    fake.adapter.Append(fake.slot, {&token, 1}, {&token, 1});
    ++fake.position;
    ++fake.decode_count;
    Remember(fake);
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
  void Remember(const FakeState& state) const {
    frontiers_[state.adapter.RecurrentHash(state.slot)] = state;
  }
  void Recall(FakeState& state) const {
    const auto found = frontiers_.find(state.adapter.RecurrentHash(state.slot));
    static_cast<FakeFrontier&>(state) =
        found == frontiers_.end() ? FakeFrontier{} : found->second;
  }
  std::shared_ptr<FakeStats> stats_;
  std::size_t measured_bytes_;
  std::size_t state_capacity_bytes_;
  std::size_t retained_snapshot_capacity_bytes_;
  mutable std::unordered_map<std::uint64_t, FakeFrontier> frontiers_;

public:
  std::optional<std::size_t> retained_snapshot_ceiling_bytes;

private:
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
};

void TestPromptCountingValidatesCapabilitiesBeforeInference() {
  const auto stats = std::make_shared<FakeStats>();
  FakeRunner runner(stats);
  ChatRequest request;
  Expect(!runner.CountPromptTokens(request),
         "unsupported template has no count");
  gufo::tokenization::ChatMessage message;
  message.role = gufo::tokenization::ChatRole::kUser;
  message.content = std::string(runner.Descriptor().max_context + 1, 'x');
  request.messages.push_back(message);
  Expect(runner.CountPromptTokens(request) == message.content.size(),
         "counting accepts text beyond the execution context");
  // A text renderer can ignore images. Counting must reject them first,
  // including images carried by a tool result.
  for (const auto role : {gufo::tokenization::ChatRole::kUser,
                          gufo::tokenization::ChatRole::kTool}) {
    auto& input = request.messages.front();
    input.role = role;
    input.images = {{0, std::make_shared<const std::vector<std::uint8_t>>(
                            std::initializer_list<std::uint8_t>{1})}};
    bool rejected = false;
    try {
      (void)runner.CountPromptTokens(request);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Expect(rejected,
           "text-only counting rejects images instead of dropping them");
  }
  Expect(stats->states_created == 0 && stats->prefill_spans.empty() &&
             stats->advanced_tokens.empty(),
         "counting never creates execution state or runs inference");
}

void TestBoundedPrefillDecodeAndPrefixReuse() {
  auto stats = std::make_shared<FakeStats>();
  auto runner = std::make_shared<SnapshotRunner>(stats);
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

  std::size_t before_release = 0;
  {
    auto request = pool.Acquire({4, 5, 6});
    (void)request.Prefill(3);
    before_release = stats->invalidations;
  }
  Expect(stats->invalidations > before_release,
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
  Expect(stats->cancellation_clears >= 1,
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
  std::string message;
  try {
    auto runner = std::make_shared<FakeRunner>(stats, 64, 159);
    TextRunnerPool pool(runner, 2);
  } catch (const std::invalid_argument& error) {
    message = error.what();
  }
  Expect(!message.empty(), "aggregate request-state claim must fit capacity");
  Expect(message.find("2 sessions at 64-token context") != std::string::npos &&
             message.find("need 80 bytes") != std::string::npos &&
             message.find("only 159 bytes") != std::string::npos &&
             message.find("at most 1 session fits") != std::string::npos,
         "capacity rejection reports the claim and the sessions that fit");
  Expect(stats->states_created == 0,
         "invalid resource claim is rejected before state allocation");
}

// A runner written against the single-boundary PrefillThrough keeps working
// through the default boundaries form, one checkpoint per call.
std::size_t RegularFiles(const std::filesystem::path& directory) {
  std::size_t files = 0;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(directory))
    files += entry.is_regular_file();
  return files;
}

/// A client that rewrites the assistant turn, as one that drops reasoning
/// does, diverges after the previous turn's stable boundary. Each warm turn
/// must therefore checkpoint its own boundary: if only the reused frontier is
/// retained, every later turn falls back to the same early position and the
/// re-prefilled tail grows for the rest of the conversation.
/// Evicting a retained prefix because the entry table is full is not routine:
/// it means the server is configured below its workload and is doing avoidable
/// full re-prefills. It must be visible, unlike exact replacement.
void TestRamCacheOptionParsing() {
  using gufo::server::ParseRamCacheOption;
  std::string error;
  const auto off = ParseRamCacheOption("off", &error);
  Expect(off && !off->enabled, "off disables the RAM cache");
  const auto automatic = ParseRamCacheOption("0", &error);
  Expect(automatic && automatic->enabled && automatic->capacity_bytes == 0,
         "0 still selects the automatic budget");
  const auto explicit_bytes = ParseRamCacheOption("17179869184", &error);
  Expect(explicit_bytes && explicit_bytes->enabled &&
             explicit_bytes->capacity_bytes == std::size_t{16} << 30,
         "a byte count is an explicit budget");
  for (const std::string_view bad :
       {"", "OFF", "Off", " off", "off ", "none", "-1", "+5", "1.5", "16G",
        "0x10", "99999999999999999999999"}) {
    error.clear();
    Expect(!ParseRamCacheOption(bad, &error),
           "an unclear value is rejected: " + std::string(bad));
    Expect(error.find("off") != std::string::npos &&
               error.find("--cache-ram-bytes") != std::string::npos,
           "the error names the option and the off value");
  }
  Expect(!ParseRamCacheOption("nope", nullptr),
         "a caller may ignore the error text");
}

/// Off keeps no snapshot at all: nothing is captured or refused, the log says
/// so, and a session still reuses its own live state.
void TestMeminfoAvailableExcludesFreeCma() {
  using gufo::server::MeminfoAvailableBytes;
  constexpr std::uint64_t kib = 1024;
  // Ubuntu 26.04's KHO scratch reports free CMA pages with CmaTotal 0.
  Expect(MeminfoAvailableBytes("MemTotal:       131072000 kB\n"
                               "MemAvailable:   100000000 kB\n"
                               "CmaTotal:               0 kB\n"
                               "CmaFree:         13034628 kB\n") ==
             (100000000 - 13034628) * kib,
         "free CMA pages are not available for snapshots");
  Expect(MeminfoAvailableBytes("MemAvailable:   8000000 kB\n") == 8000000 * kib,
         "MemAvailable is used unchanged without a CmaFree line");
  Expect(
      MeminfoAvailableBytes("CmaFree:  9000 kB\nMemAvailable:  4000 kB\n") == 0,
      "free CMA beyond MemAvailable saturates at zero");
  Expect(!MeminfoAvailableBytes("MemTotal:  4000 kB\nCmaFree:  10 kB\n"),
         "missing MemAvailable keeps the caller's fallback");
}

void TestCgroupAvailableCountsReclaimableCache() {
  namespace fs = std::filesystem;
  using gufo::server::CgroupAvailableBytes;
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  const auto root =
      fs::temp_directory_path() /
      ("gufo_cgroup_test_" + std::to_string(gufo::platform::ProcessId()));
  fs::remove_all(root);
  const auto write = [&](const fs::path& dir, const char* name,
                         const std::string& text) {
    fs::create_directories(root / dir);
    std::ofstream(root / dir / name) << text << "\n";
  };
  const auto level = [&](const fs::path& dir, const std::string& max,
                         std::uint64_t current,
                         std::optional<std::uint64_t> inactive_file) {
    write(dir, "memory.max", max);
    write(dir, "memory.current", std::to_string(current));
    fs::remove(root / dir / "memory.stat");
    if (inactive_file)
      write(dir, "memory.stat",
            "anon 1\ninactive_file " + std::to_string(*inactive_file) +
                "\nactive_file 5");
  };
  const std::string member = "0::/pod/ctr\n";
  const auto host = 500 * gib;

  level("pod/ctr", std::to_string(80 * gib), 79 * gib, 32 * gib);
  Expect(CgroupAvailableBytes(root, member, host) == 33 * gib,
         "inactive file pages count as available");
  Expect(CgroupAvailableBytes(root, member, 10 * gib) == 10 * gib,
         "host MemAvailable still bounds the result");

  level("pod/ctr", std::to_string(80 * gib), 79 * gib, std::nullopt);
  Expect(CgroupAvailableBytes(root, member, host) == 1 * gib,
         "missing memory.stat keeps memory.current as used");
  const auto stat = [&](const std::string& text) {
    write("pod/ctr", "memory.stat", text);
  };
  level("pod/ctr", std::to_string(80 * gib), 79 * gib, 32 * gib);
  stat("anon 1\nactive_file 5\n");
  Expect(CgroupAvailableBytes(root, member, host) == 1 * gib,
         "memory.stat without inactive_file keeps memory.current as used");
  stat("inactive_file " + std::to_string(32 * gib) + "\nfile_dirty lots\n");
  Expect(CgroupAvailableBytes(root, member, host) == 1 * gib,
         "a malformed memory.stat value is ignored");
  stat("inactive_file notanumber\n");
  Expect(CgroupAvailableBytes(root, member, host) == 1 * gib,
         "a malformed inactive_file value is ignored");
  stat("inactive_file " + std::to_string(32 * gib) + "\nfile_dirty " +
       std::to_string(3 * gib) + "\nfile_writeback " + std::to_string(1 * gib) +
       "\n");
  Expect(CgroupAvailableBytes(root, member, host) == 29 * gib,
         "dirty and writeback pages stay charged");
  stat("inactive_file " + std::to_string(2 * gib) + "\nfile_dirty " +
       std::to_string(5 * gib) + "\n");
  Expect(CgroupAvailableBytes(root, member, host) == 1 * gib,
         "dirty pages beyond inactive_file reclaim nothing");

  level("pod/ctr", std::to_string(80 * gib), 10 * gib, 32 * gib);
  Expect(CgroupAvailableBytes(root, member, host) == 80 * gib,
         "inactive file larger than current saturates");

  level("pod/ctr", std::to_string(80 * gib), 79 * gib, 32 * gib);
  level("pod", std::to_string(40 * gib), 30 * gib, 5 * gib);
  Expect(CgroupAvailableBytes(root, member, host) == 15 * gib,
         "a tighter parent limit wins");

  level("pod/ctr", "max", 79 * gib, 32 * gib);
  Expect(CgroupAvailableBytes(root, member, host) == 15 * gib,
         "an unlimited level defers to its ancestors");
  fs::remove_all(root / "pod");
  level("", "max", 79 * gib, 32 * gib);
  level("pod/ctr", "max", 79 * gib, 32 * gib);
  Expect(CgroupAvailableBytes(root, member, host) == host,
         "unlimited cgroups leave host availability unchanged");
  fs::remove_all(root);
}

void TestPreparedConstraintVocabularyKeepsGrammarBehavior() {
  // Startup preparation changes when the vocabulary is built, not
  // what it allows: one build, reused by every binding, with the same masks
  // and accepted states as a runner that builds it for its first request.
  class VocabularyRunner final : public FakeRunner {
  public:
    VocabularyRunner() : FakeRunner(std::make_shared<FakeStats>()) {}
    std::shared_ptr<const gufo::sampling::ConstraintVocabulary>
    BuildConstraintVocabulary() const override {
      ++builds;
      // Single bytes, a few multi-byte pieces and a stop token.
      static const std::vector<std::string> extra{"{\"", "\":", "\",\"",
                                                  "ab",  "12",  "\"}"};
      return std::make_shared<gufo::sampling::ConstraintVocabulary>(
          128 + extra.size() + 1, [](std::uint32_t id) {
            if (id < 128)
              return gufo::sampling::ConstraintVocabulary::Piece{
                  std::string(1, static_cast<char>(id)), false};
            if (id < 128 + extra.size())
              return gufo::sampling::ConstraintVocabulary::Piece{
                  extra[id - 128], false};
            return gufo::sampling::ConstraintVocabulary::Piece{"", true};
          });
    }
    mutable std::size_t builds{0};
  };
  const VocabularyRunner prepared;
  const VocabularyRunner lazy;
  const auto size = prepared.PrepareConstraintVocabulary();
  Expect(size == 135 && prepared.PrepareConstraintVocabulary() == size &&
             prepared.builds == 1,
         "preparation builds the vocabulary once");
  const auto grammar = gufo::sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{"keyword":{"type":"string"},)"
          R"("n":{"type":"integer"}},"required":["keyword","n"],)"
          R"("additionalProperties":false})"),
      false);
  const auto other = gufo::sampling::JsonConstraint::Compile(
      gufo::json::parse(
          R"({"type":"object","properties":{},"additionalProperties":false})"),
      false);
  const auto binding = prepared.BindConstraint(grammar);
  Expect(prepared.BindConstraint(other)->vocabulary == binding->vocabulary &&
             prepared.builds == 1,
         "constrained requests reuse the prepared vocabulary");
  const auto reference = lazy.BindConstraint(grammar);
  Expect(lazy.builds == 1, "without preparation the first binding builds it");
  // {"keyword":"ab","n":12} in mixed single- and multi-byte pieces.
  const std::vector<std::uint32_t> tokens{128, 'k', 'e', 'y', 'w', 'o',
                                          'r', 'd', 129, '"', 131, 130,
                                          'n', 129, 132, '}'};
  auto state = grammar->Start();
  auto expected = grammar->Start();
  for (const auto token : tokens) {
    const auto mask = binding->Allowed(state);
    Expect(*mask == *reference->Allowed(expected) && (*mask)[token],
           "a prepared vocabulary allows the same tokens");
    state = binding->vocabulary->Accept(*grammar, state, token);
    expected = reference->vocabulary->Accept(*grammar, expected, token);
  }
  Expect(grammar->Complete(state) && grammar->Complete(expected) &&
             *binding->Allowed(state) == *reference->Allowed(expected),
         "both bindings complete the same output");
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
  for (std::string_view token : std::initializer_list<std::string_view>{
           kImStart, kImEnd, "<think>", "</think>", "<tool_call>",
           "</tool_call>"}) {
    specials.emplace(token, vocab.size());
    vocab.emplace_back(token);
  }
  std::string error;
  auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocab, {}, specials, &error);
  Expect(tokenizer != nullptr, "instruction tokenizer fixture initializes");
  ChatTemplateOptions options;
  options.enable_thinking = false;
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
  TestPromptCountingValidatesCapabilitiesBeforeInference();
  TestServerInstructionsAreFraming();
  TestPreparedConstraintVocabularyKeepsGrammarBehavior();
  TestBoundedPrefillDecodeAndPrefixReuse();
  TestAbandonedRequestRollsBackState();
  TestRequestBindsAndClearsCancellation();
  TestBatchedAdvancePreservesIndependentRequests();
  TestResourceClaimsAreValidatedBeforeAllocation();
  TestRamCacheOptionParsing();
  TestMeminfoAvailableExcludesFreeCma();
  TestCgroupAvailableCountsReclaimableCache();
  std::cout << "All text model runner tests passed\n";
  return 0;
}
