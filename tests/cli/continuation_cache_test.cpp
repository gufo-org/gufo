#include "src/cli/serve/continuation_cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include "src/cli/serve/logging.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

struct FakeState final : gufo::server::ContinuationState {
  FakeState(std::size_t state_id, std::vector<std::size_t>* invalidations)
      : id(state_id), invalidation_counts(invalidations) {}

  void Invalidate() noexcept override { ++invalidation_counts->at(id); }

  std::size_t id;
  std::size_t value{0};
  std::vector<std::size_t>* invalidation_counts;
};

struct FakeSnapshot final : gufo::server::ContinuationSnapshot {
  explicit FakeSnapshot(std::size_t value,
                        std::size_t payload_bytes = sizeof(std::size_t))
      : value(value), payload_bytes(payload_bytes) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return payload_bytes;
  }

  std::size_t value;
  std::size_t payload_bytes;
};

void TestColdMissThenExactExtensionHit() {
  std::vector<std::size_t> invalidations(1);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(1, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  const std::vector<gufo::server::ContinuationToken> first_prompt{1, 2, 3};
  {
    auto lease = cache.Acquire(first_prompt);
    Expect(static_cast<bool>(lease), "cold request acquires the slot");
    Expect(!lease.cache_hit(), "first request is a cache miss");
    Expect(lease.lookup().miss_reason == "no_checkpoint",
           "cold miss is distinguished from changed input");
    Expect(lease.cached_tokens() == 0, "cold request reuses no tokens");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "lease exposes the opaque model state");
    lease.Commit({1, 2, 3, 4});
  }

  const std::vector<gufo::server::ContinuationToken> extension{1, 2, 3,
                                                               4, 5, 6};
  {
    auto lease = cache.Acquire(extension);
    Expect(lease.cache_hit(), "exact extension reuses the slot");
    Expect(lease.lookup().miss_reason.empty(), "hits carry no miss reason");
    Expect(lease.cached_tokens() == 4,
           "hit reports the complete retained prefix");
    Expect(dynamic_cast<FakeState&>(lease.state()).id == 0,
           "hit returns the same opaque state");
    Expect(invalidations[0] == 0, "hit does not invalidate retained state");
    lease.Commit(extension);
  }
}

void TestDivergenceInvalidatesOldState() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    lease.Commit({1, 2, 3});
  }
  {
    auto lease =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 9});
    Expect(!lease.cache_hit(), "divergent request is a miss");
    Expect(lease.lookup().miss_reason == "prefix_changed" &&
               lease.lookup().common_prefix_tokens == 1 &&
               lease.lookup().checkpoint_tokens == 3,
           "miss identifies the first changed token without exposing it");
    Expect(invalidations[0] == 1,
           "divergence invalidates the previous model state");
    lease.Commit({1, 9});
  }
}

void TestUncommittedLeaseIsInvalidated() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });

  {
    auto lease = cache.Acquire(std::vector<gufo::server::ContinuationToken>{7});
    Expect(static_cast<bool>(lease), "request acquires the slot");
  }
  Expect(invalidations[0] == 1,
         "abandoned request invalidates partially computed state");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{7, 8});
  Expect(!retry.cache_hit(), "abandoned state is never reused");
  retry.Commit({7, 8});
}

void TestLongestAvailablePrefixWins() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(2, [&] {
    return std::make_unique<FakeState>(next_id++, &invalidations);
  });

  {
    auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    first.Commit({1, 2});
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    second.Commit({9, 8, 7});
  }

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7, 6});
  Expect(lease.cache_hit(), "one available entry matches");
  Expect(lease.cached_tokens() == 3, "longest exact prefix is selected");
  Expect(dynamic_cast<FakeState&>(lease.state()).id == 1,
         "matching state is selected rather than an arbitrary slot");
  lease.Commit({9, 8, 7, 6});
}

void TestWaitingAcquireCanBeCancelled() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });
  auto held =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  std::atomic<bool> cancelled{true};

  auto cancelled_lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4},
                    [&] { return cancelled.load(); });
  Expect(!cancelled_lease, "cancelled waiter does not acquire a state");
  held.Commit({1, 2, 3});
}

void TestSnapshotCanBranchIntoTwoIndependentStateSlots() {
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                auto& fake = dynamic_cast<FakeState&>(state);
                const auto& saved = dynamic_cast<const FakeSnapshot&>(snapshot);
                fake.value = saved.value;
              },
          .capacity_bytes = [] { return 1024; },
          .on_event = {},
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    dynamic_cast<FakeState&>(root.state()).value = 7;
    Expect(root.TryReserveSnapshot(sizeof(std::size_t), 3),
           "root snapshot reserves aggregate capacity before allocation");
    root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7));
  }

  auto first =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4});
  auto second =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 5});
  Expect(first.cache_hit() && second.cache_hit(),
         "one snapshot can satisfy two simultaneous leases");
  auto& first_state = dynamic_cast<FakeState&>(first.state());
  auto& second_state = dynamic_cast<FakeState&>(second.state());
  Expect(first_state.id != second_state.id,
         "snapshot branches use different mutable state slots");
  Expect(first_state.value == 7 && second_state.value == 7,
         "both mutable states restore the root payload");

  first_state.value = 8;
  second_state.value = 9;
  Expect(first.TryReserveSnapshot(sizeof(std::size_t), 4),
         "first branch reserves snapshot capacity");
  Expect(second.TryReserveSnapshot(sizeof(std::size_t), 4),
         "second branch reserves snapshot capacity");
  first.Commit({1, 2, 3, 4}, std::make_unique<FakeSnapshot>(8));
  second.Commit({1, 2, 3, 5}, std::make_unique<FakeSnapshot>(9));

  auto root_again =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 6});
  Expect(root_again.cache_hit() && root_again.cached_tokens() == 3,
         "branch commits preserve the shared root snapshot");
  Expect(dynamic_cast<FakeState&>(root_again.state()).value == 7,
         "branch mutation never changes the immutable root payload");
  root_again.Invalidate();
}

void TestByteCapacityEvictsBeforeSnapshotAllocation() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    Expect(root.TryReserveSnapshot(8, 3),
           "initial snapshot fits the byte budget");
    Expect(root.Commit({1, 2, 3}, std::make_unique<FakeSnapshot>(7, 8)) == 8,
           "initial snapshot is retained");
  }
  Expect(cache.retained_snapshot_bytes() == 8,
         "retained bytes account the initial snapshot");

  auto replacement =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
  Expect(replacement.TryReserveSnapshot(12, 3),
         "reservation evicts stale bytes before snapshot allocation");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 12,
         "eviction transfers budget from retained to reserved bytes");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 3,
         "byte-pressure removal reports sanitized reason and dimensions");
  Expect(replacement.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(9, 12)) ==
             12,
         "reserved replacement is retained");
  Expect(cache.retained_snapshot_bytes() == 12 &&
             cache.reserved_snapshot_bytes() == 0,
         "commit converts the reservation into exact retained bytes");
}

void TestConcurrentReservationsCannotOvercommitBudget() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 12; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto first = cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
  auto second = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(first.TryReserveSnapshot(8, 1),
         "first in-flight snapshot reserves bytes");
  Expect(!second.TryReserveSnapshot(8, 1),
         "second reservation is rejected instead of overcommitting");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 8,
         "only the admitted in-flight reservation is accounted");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 1,
         "capacity refusal emits a sanitized skip event");

  second.Commit({2});
  first.Commit({1}, std::make_unique<FakeSnapshot>(1, 8));
  Expect(cache.retained_snapshot_bytes() == 8 &&
             cache.reserved_snapshot_bytes() == 0,
         "completed requests leave no leaked reservation");
}

void TestImpossibleReservationPreservesRetainedEntries() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore =
              [](gufo::server::ContinuationState& state,
                 const gufo::server::ContinuationSnapshot& snapshot) {
                dynamic_cast<FakeState&>(state).value =
                    dynamic_cast<const FakeSnapshot&>(snapshot).value;
              },
          .capacity_bytes = [] { return 8; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto root =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(root.TryReserveSnapshot(8, 2), "root fills the byte budget");
    root.Commit({1, 2}, std::make_unique<FakeSnapshot>(7, 8));
  }
  {
    auto oversized =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9});
    Expect(!oversized.TryReserveSnapshot(9, 1),
           "snapshot larger than the total budget is rejected");
    oversized.Commit({9});
  }

  Expect(cache.retained_snapshot_bytes() == 8,
         "impossible admission does not evict a useful retained entry");
  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kSkipped &&
             events.front().reason == SnapshotEventReason::kByteCapacity &&
             events.front().snapshot_bytes == 9,
         "oversized refusal emits one skip and no removal");
  auto extension =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
  Expect(extension.cache_hit() && extension.cached_tokens() == 2,
         "retained root remains reusable after oversized refusal");
  extension.Invalidate();
}

void TestAbandonedReservationIsReleased() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 8; },
          .on_event = {},
      });

  {
    auto abandoned =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1});
    Expect(abandoned.TryReserveSnapshot(8, 1),
           "abandoned request owns the whole reservation");
  }
  Expect(cache.reserved_snapshot_bytes() == 0,
         "lease destruction releases its in-flight reservation");

  auto retry = cache.Acquire(std::vector<gufo::server::ContinuationToken>{2});
  Expect(retry.TryReserveSnapshot(8, 1),
         "released bytes are immediately available to another request");
  retry.Commit({2}, std::make_unique<FakeSnapshot>(2, 8));
}

void TestReservationMismatchSkipsRetentionWithoutFailingCommit() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 16; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  auto lease =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5});
  Expect(lease.TryReserveSnapshot(4, 2),
         "snapshot estimate reserves before allocation");
  Expect(lease.Commit({4, 5}, std::make_unique<FakeSnapshot>(1, 8)) == 0,
         "an underestimated snapshot is skipped without failing commit");
  Expect(cache.retained_snapshot_bytes() == 0 &&
             cache.reserved_snapshot_bytes() == 0,
         "mismatch releases the complete reservation");
  Expect(
      events.size() == 1 &&
          events.front().action == SnapshotEventAction::kSkipped &&
          events.front().reason == SnapshotEventReason::kReservationMismatch &&
          events.front().snapshot_bytes == 8 && events.front().token_count == 2,
      "reservation mismatch is observable without prompt content");

  auto retry =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{4, 5, 6});
  Expect(!retry.cache_hit(), "skipped snapshot is a deterministic cache miss");
  retry.Invalidate();
}

void TestEntryReplacementLogsRemovedSnapshot() {
  using gufo::server::SnapshotEventAction;
  using gufo::server::SnapshotEventReason;

  std::vector<std::size_t> invalidations(1);
  std::vector<gufo::server::SnapshotEvent> events;
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); },
      {
          .restore = [](gufo::server::ContinuationState&,
                        const gufo::server::ContinuationSnapshot&) {},
          .capacity_bytes = [] { return 32; },
          .on_event =
              [&](const gufo::server::SnapshotEvent& event) {
                events.push_back(event);
              },
      });

  {
    auto first =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2});
    Expect(first.TryReserveSnapshot(8, 2), "first entry reserves bytes");
    first.Commit({1, 2}, std::make_unique<FakeSnapshot>(1, 8));
  }
  {
    auto second =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{9, 8, 7});
    Expect(second.TryReserveSnapshot(8, 3), "replacement reserves bytes");
    second.Commit({9, 8, 7}, std::make_unique<FakeSnapshot>(2, 8));
  }

  Expect(events.size() == 1 &&
             events.front().action == SnapshotEventAction::kRemoved &&
             events.front().reason == SnapshotEventReason::kEntryCapacity &&
             events.front().snapshot_bytes == 8 &&
             events.front().token_count == 2,
         "entry replacement logs the removed snapshot dimensions");
  Expect(cache.retained_snapshot_bytes() == 8,
         "entry replacement keeps exact aggregate accounting");
}

// The summarised cache lines stay at INFO/WARN; which candidate was rejected,
// and by which guard, is only worth the bytes when the operator asked for it.
void TestCacheCandidateDetailIsDebugTierOnly() {
  const auto run = [](gufo::server::LogLevel level) {
    std::vector<std::size_t> invalidations(1);
    std::size_t next_id = 0;
    const auto previous_level = gufo::server::Logger::Level();
    gufo::server::Logger::SetLevel(level);
    std::ostringstream output;
    auto* previous_sink = std::clog.rdbuf(output.rdbuf());
    {
      gufo::server::ContinuationCache cache(1, [&] {
        return std::make_unique<FakeState>(next_id++, &invalidations);
      });
      const std::vector<gufo::server::ContinuationToken> retained{1, 2, 3};
      const std::vector<gufo::server::ContinuationToken> diverged{9, 8, 7};
      {
        auto lease = cache.Acquire(retained);
        lease.Commit(retained);
      }
      auto other = cache.Acquire(diverged);
      Expect(!other.cache_hit(), "a different prompt cannot reuse the slot");
    }
    std::clog.rdbuf(previous_sink);
    gufo::server::Logger::SetLevel(previous_level);
    return output.str();
  };

  const auto quiet = run(gufo::server::LogLevel::kInfo);
  Expect(quiet.find("event=candidate_skip") == std::string::npos,
         "candidate detail is silent at the default level");
  Expect(quiet.find("event=capture_evicts") == std::string::npos,
         "eviction detail is silent at the default level");

  const auto verbose = run(gufo::server::LogLevel::kDebug);
  Expect(verbose.find("event=candidate_skip index=0 tokens=3 prompt_tokens=3 "
                      "reason=token_prefix") != std::string::npos,
         "debug tier names the rejected candidate and its failing guard");
  Expect(verbose.find("event=capture_evicts index=0 tokens=3") !=
             std::string::npos,
         "debug tier names the checkpoint a capture overwrites");
}

// A waiter that finds every slot busy still reports why each candidate was
// rejected: contention is exactly when that question gets asked, and the
// records used to die with the poll iteration that collected them.
void TestWaitingAcquireReportsCandidateSkips() {
  std::vector<std::size_t> invalidations(1);
  gufo::server::ContinuationCache cache(
      1, [&] { return std::make_unique<FakeState>(0, &invalidations); });
  {
    auto seed =
        cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3});
    seed.Commit({1, 2, 3});
  }
  auto busy =
      cache.Acquire(std::vector<gufo::server::ContinuationToken>{1, 2, 3, 4});
  Expect(busy.cache_hit(), "the sole slot is leased to a matching prompt");
  // The lease clears the saved tokens in the live (non-snapshot) mode, so the
  // waiter records the busy slot with zero checkpoint tokens.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds{200};
  const auto previous_level = gufo::server::Logger::Level();
  gufo::server::Logger::SetLevel(gufo::server::LogLevel::kDebug);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    auto waiter = cache.Acquire(
        std::vector<gufo::server::ContinuationToken>{9, 8, 7},
        [&] { return std::chrono::steady_clock::now() >= deadline; });
    Expect(!waiter, "the cancelled waiter never acquires the busy slot");
  }
  std::clog.rdbuf(previous_sink);
  gufo::server::Logger::SetLevel(previous_level);
  const std::string log = output.str();
  Expect(log.find("event=candidate_skip index=0 tokens=0 prompt_tokens=3 "
                  "reason=unavailable") != std::string::npos,
         "a waiter names the guard that rejected the busy candidate");
  std::size_t bursts = 0;
  for (std::size_t at = log.find("event=candidate_skip");
       at != std::string::npos; at = log.find("event=candidate_skip", at + 1)) {
    ++bursts;
  }
  Expect(bursts == 1, "a long wait reports its first record set once");
}

}  // namespace

void TestImageIdentityIsolation() {
  for (const bool snapshot_mode : {false, true}) {
    std::vector<std::size_t> invalidations(2);
    std::size_t next_id = 0;
    gufo::server::ContinuationCache::SnapshotSupport support;
    if (snapshot_mode) {
      support.capacity_bytes = [] { return std::size_t{1024}; };
      support.restore = [](auto& state, const auto& snapshot) {
        dynamic_cast<FakeState&>(state).value =
            dynamic_cast<const FakeSnapshot&>(snapshot).value;
      };
    }
    gufo::server::ContinuationCache cache(
        2,
        [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
        std::move(support));
    const std::vector<gufo::server::ContinuationToken> prompt{1, 248056, 3};
    const std::vector<std::uint8_t> a{1, 2}, b{1, 3};
    const auto save = [&](auto identity, std::size_t value) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(!lease.cache_hit(), "different image is a cold input");
      dynamic_cast<FakeState&>(lease.state()).value = value;
      if (snapshot_mode) {
        Expect(lease.TryReserveSnapshot(sizeof(std::size_t), prompt.size()),
               "snapshot admitted");
        lease.Commit(prompt, std::make_unique<FakeSnapshot>(value));
      } else {
        lease.Commit(prompt);
      }
    };
    save(a, 10);
    save(b, 20);
    for (const auto& [identity, value] :
         std::vector<std::pair<std::vector<std::uint8_t>, std::size_t>>{
             {a, 10}, {b, 20}}) {
      auto lease = cache.Acquire(prompt, {}, identity);
      Expect(lease.cache_hit(), "same image reuses its own prefix");
      Expect(dynamic_cast<FakeState&>(lease.state()).value == value,
             "image state cannot cross requests");
      if (!snapshot_mode)
        lease.Commit(prompt);
    }
    auto text = cache.Acquire(prompt);
    Expect(!text.cache_hit(),
           "literal image-pad text cannot reuse image state");
  }
}

void TestAppendedImagesReuseOnlyCompatiblePrefixes() {
  using namespace gufo::server;
  std::vector<std::size_t> invalidations(2);
  std::size_t next_id = 0;
  ContinuationCache cache(
      2, [&] { return std::make_unique<FakeState>(next_id++, &invalidations); },
      {.restore =
           [](auto& state, const auto& snapshot) {
             dynamic_cast<FakeState&>(state).value =
                 dynamic_cast<const FakeSnapshot&>(snapshot).value;
           },
       .capacity_bytes = [] { return 1024; },
       .on_event = {}},
      4);
  const std::vector<ContinuationToken> text{1, 2, 3};
  const std::vector<ContinuationToken> image{1, 2, 3, 248056, 4};
  const std::vector<ContinuationToken> appended{1, 2, 3, 248056, 4, 248056, 5};
  const std::vector<std::uint8_t> a{10}, b{20}, ab{30};
  const std::vector<ContinuationInputPrefix> first{{3, {}}};
  const std::vector<ContinuationInputPrefix> second{{3, {}}, {5, a}};
  {
    auto lease = cache.Acquire(image, {}, a, {}, true, 0, first);
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), text.size()),
           "image request admits a checkpoint before its first image");
    lease.PublishSnapshot(text, std::make_unique<FakeSnapshot>(7));
    Expect(lease.HasSnapshotFor(text),
           "checkpoint lookup uses the identity at the saved position");
    dynamic_cast<FakeState&>(lease.state()).value = 70;
    lease.Commit({}, {}, image);
  }
  {
    auto lease = cache.Acquire(appended, {}, ab, {}, true, 0, second);
    Expect(lease.cache_hit() && lease.cached_tokens() == image.size() &&
               dynamic_cast<FakeState&>(lease.state()).value == 70,
           "new image reuses the live first-image state");
    Expect(lease.TryReserveSnapshot(sizeof(std::size_t), image.size()),
           "first-image snapshot admitted");
    lease.Commit(image, std::make_unique<FakeSnapshot>(70));
  }
  {
    int attachments = 0;
    auto attach = [&](ContinuationState&) { ++attachments; };
    auto matching = cache.Acquire(appended, {}, ab, attach, true, 0, second);
    auto changed = cache.Acquire(image, {}, b, {}, true, 0, first);
    Expect(matching.cached_tokens() == image.size() &&
               changed.cached_tokens() == text.size(),
           "simultaneous different images choose independent safe frontiers");
    Expect(attachments == 2, "request input is reattached after restoration");
    Expect(dynamic_cast<FakeState&>(matching.state()).value == 70 &&
               dynamic_cast<FakeState&>(changed.state()).value == 7 &&
               dynamic_cast<FakeState&>(matching.state()).id !=
                   dynamic_cast<FakeState&>(changed.state()).id,
           "branches restore separate payloads into separate sessions");
  }
  {
    auto removed = cache.Acquire(image);
    Expect(removed.cached_tokens() == text.size(),
           "removing an image cannot restore its computed embedding state");
  }
  Expect(cache.retained_snapshot_bytes() == 2 * sizeof(std::size_t),
         "prefix identity does not duplicate snapshot payloads");

  // Without an earlier checkpoint, expose token agreement even when image
  // identity makes the available state unusable.
  std::vector<std::size_t> direct_invalidations(1);
  ContinuationCache direct(
      1, [&] { return std::make_unique<FakeState>(0, &direct_invalidations); });
  auto original = direct.Acquire(image, {}, a, {}, true, 0, first);
  original.Commit(image);
  auto changed = direct.Acquire(image, {}, b, {}, true, 0, first);
  Expect(!changed.cache_hit() &&
             changed.lookup().miss_reason == "input_changed" &&
             changed.lookup().common_prefix_tokens == image.size() &&
             changed.lookup().checkpoint_tokens == image.size(),
         "image mismatch diagnostics retain the actual token agreement");
}

int main() {
  TestAppendedImagesReuseOnlyCompatiblePrefixes();
  TestImageIdentityIsolation();
  TestColdMissThenExactExtensionHit();
  TestDivergenceInvalidatesOldState();
  TestUncommittedLeaseIsInvalidated();
  TestLongestAvailablePrefixWins();
  TestWaitingAcquireCanBeCancelled();
  TestSnapshotCanBranchIntoTwoIndependentStateSlots();
  TestByteCapacityEvictsBeforeSnapshotAllocation();
  TestConcurrentReservationsCannotOvercommitBudget();
  TestImpossibleReservationPreservesRetainedEntries();
  TestAbandonedReservationIsReleased();
  TestReservationMismatchSkipsRetentionWithoutFailingCommit();
  TestEntryReplacementLogsRemovedSnapshot();
  TestCacheCandidateDetailIsDebugTierOnly();
  TestWaitingAcquireReportsCandidateSkips();
  std::cout << "All continuation cache tests passed\n";
  return 0;
}
