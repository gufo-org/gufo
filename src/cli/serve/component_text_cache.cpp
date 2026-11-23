#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "src/cache/disk_catalog.hpp"
#include "src/cache/retention.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/mem/ledger.hpp"
#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/expert_stream/store.hpp"
#endif

namespace gufo::server {
namespace {
using Clock = std::chrono::steady_clock;
long long MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                               start)
      .count();
}
std::size_t SaturatingAdd(std::size_t a, std::size_t b) {
  return a + std::min(b, std::numeric_limits<std::size_t>::max() - a);
}
class SettledSignal final : public cache::CompletionSignal {
public:
  bool Ready() const noexcept override { return true; }
  cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
cache::InputIdentity Inputs(std::size_t count,
                            const TextPromptContext* context) {
  std::vector<cache::InputPrefix> prefixes;
  if (context)
    for (const auto& p : context->cache_prefixes)
      prefixes.push_back({p.token_count, p.identity});
  return cache::InputIdentity(
      count, context ? context->cache_identity : cache::Identity{},
      std::move(prefixes));
}
cache::Identity At(const cache::InputIdentity& input, std::size_t boundary) {
  const auto bytes = input.At(boundary);
  return {bytes.begin(), bytes.end()};
}
// How long a restore of `bytes` stops the thread that runs it: RAM restores
// copy ~55 GB/s, disk restores read ~13 GB/s from the page cache (Flash-Next
// and Qwen 27B, November 2026).
std::chrono::microseconds RestoreEstimate(std::size_t bytes, bool disk) {
  return std::chrono::microseconds(bytes / (disk ? 13'000 : 55'000));
}

std::size_t PayloadBytes(const cache::DiskManifest& manifest) {
  std::size_t bytes = 0;
  for (const auto& component : manifest.components) {
    for (const auto& chunk : component.chunks)
      bytes += chunk.bytes;
    if (component.tail)
      bytes += component.tail->bytes;
    if (component.private_state)
      bytes += component.private_state->bytes;
  }
  return bytes;
}
// Disk eviction during admission decides which conversations survive a
// restart; name every retired entry so retention losses can be traced.
bool LogDiskEviction(const cache::DiskEviction& eviction) {
  if (!eviction)
    return false;
  const auto& manifest = eviction.description->Manifest();
  Logger::Info(
      "cache",
      "schema=component-cache-v1 event=disk_entry_evicted reason=admission "
      "tokens=" +
          std::to_string(manifest.tokens.size()) + " checkpoint_id=" +
          std::to_string(manifest.checkpoint.value) + " purpose=" +
          std::to_string(static_cast<unsigned>(manifest.purpose)) +
          " superseded=" + std::to_string(eviction.superseded) +
          " reused=" + std::to_string(eviction.reused) +
          " payload_bytes=" + std::to_string(PayloadBytes(manifest)));
  return true;
}
class FrontierHeadroomRefused final : public std::runtime_error {
public:
  FrontierHeadroomRefused() : std::runtime_error("frontier capture headroom") {}
};
struct DurableCapture {
  cache::ResourceCharge metadata;
  std::shared_ptr<const void> branch;
  cache::DiskReadPin ancestor_pin;
  cache::DiskManifest manifest;
  std::vector<cache::PersistenceSource> sources;
};
class ComponentSnapshot final : public ContinuationSnapshot {
public:
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::shared_ptr<DurableCapture> durable;
  std::size_t bytes{};
  std::size_t PayloadBytes() const noexcept override { return bytes; }
};
cache::RetentionPurpose Purpose(SnapshotPurpose purpose) {
  switch (purpose) {
    case SnapshotPurpose::kRetry:
      return cache::RetentionPurpose::kRetry;
    case SnapshotPurpose::kHistory:
      return cache::RetentionPurpose::kHistory;
    case SnapshotPurpose::kBranchPoint:
      return cache::RetentionPurpose::kBranchPoint;
    case SnapshotPurpose::kContinuation:
      return cache::RetentionPurpose::kContinuation;
  }
  throw std::logic_error("unknown retention purpose");
}
void Check(cache::Completion completion) {
  if (completion.Wait() != cache::TransferResult::kSucceeded)
    throw std::runtime_error("component cache transfer failed");
}
// A published checkpoint that is still resident restores from RAM, so its
// candidate carries no disk description; its availability still marks every
// component durable.
bool Published(const cache::PrefixCandidate& candidate) {
  if (candidate.durable)
    return true;
  return candidate.availability &&
         !candidate.availability->components.empty() &&
         std::ranges::all_of(candidate.availability->components,
                             &cache::ComponentAvailability::durable);
}
// Only prompt checkpoints space later disk writes on the same prefix.
bool Spaced(cache::CheckpointPurpose purpose) {
  return purpose == cache::CheckpointPurpose::kPrompt;
}
bool SpacedCheckpoint(const cache::PrefixCandidate& candidate) {
  if (candidate.checkpoint && !Spaced(candidate.checkpoint->Purpose()))
    return false;
  if (candidate.durable && !Spaced(candidate.durable->Manifest().purpose))
    return false;
  return true;
}
bool SameOwner(const std::shared_ptr<const void>& a,
               const std::shared_ptr<const void>& b) {
  return !a.owner_before(b) && !b.owner_before(a);
}
class TransferBatch {
public:
  explicit TransferBatch(std::size_t capacity) : capacity_(capacity) {
    if (!capacity || capacity > jobs_.size())
      throw std::invalid_argument("invalid cache transfer batch size");
  }
  // Acquire returns null when the shared pool is momentarily exhausted. Settle
  // this batch's own leases before retrying, so a wide batch never fails a
  // transfer that a narrower one would have completed.
  template<class Acquire, class Submit>
  void Add(Acquire acquire, Submit submit) {
    if (size_ == capacity_)
      Finish();
    auto stream = acquire();
    if (!stream && size_) {
      Finish();
      stream = acquire();
    }
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    auto completion = submit(*stream);
    jobs_[size_++].emplace(Job{std::move(stream), std::move(completion)});
  }
  void Finish() {
    // Jobs keep their streams and completions together. On any exception,
    // destruction drains every remaining transfer before buffers unwind.
    for (std::size_t i = 0; i < size_; ++i) {
      Check(std::move(jobs_[i]->completion));
      jobs_[i].reset();
    }
    size_ = 0;
  }

private:
  struct Job {
    std::unique_ptr<cache::Stream> stream;
    cache::Completion completion;
  };
  std::array<std::optional<Job>, 16> jobs_;
  std::size_t capacity_, size_{};
};
class Events final : public cache::RetentionEventSink {
public:
  void Emit(const cache::RetentionEvent& e) noexcept override {
    try {
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=retention action=" +
              std::to_string(static_cast<unsigned>(e.action)) +
              " reason=" + std::to_string(static_cast<unsigned>(e.reason)) +
              " tokens=" + std::to_string(e.boundary) +
              " purpose=" + std::to_string(static_cast<unsigned>(e.purpose)) +
              " rank=" + std::to_string(e.rank) +
              " unique_bytes_freed=" + std::to_string(e.unique_bytes_freed) +
              " ledger_bytes=" + std::to_string(e.ledger_bytes));
    } catch (...) {
    }
  }
};

std::string HostReserveMessage(std::size_t sessions, std::size_t requested,
                               std::size_t available) {
  constexpr std::size_t kMiB = std::size_t{1} << 20;
  return std::to_string(sessions) +
         (sessions == 1 ? " session leaves " : " sessions leave ") +
         std::to_string(available / kMiB) +
         " MiB for the RAM cache above the " +
         std::to_string(kHostSnapshotHeadroomBytes / kMiB) +
         " MiB host reserve" +
         (requested
              ? " (" + std::to_string(requested / kMiB) + " MiB requested)"
              : std::string()) +
         ". Use fewer --sessions, a smaller --context, or --cache-ram-bytes "
         "off to serve without RAM checkpoints";
}
}  // namespace
struct ComponentTextCache::Impl {
  struct Entry {
    // Guard fallbacks outlive their slots. Idle pieces and capture/load work
    // obtain fresh one-completion leases from the shared transfer pool.
    std::unique_ptr<cache::Stream> preservation;
    std::unique_ptr<cache::LeasedSlot> slot;
    std::optional<cache::IndexEntryId> live;
    std::unique_ptr<cache::ExecutionHistory> history;
    // Identity of uninterrupted native execution; token equality alone does
    // not establish that a durable chunk belongs to this execution.
    std::shared_ptr<const void> branch;
    std::shared_ptr<const cache::DiskDescription> durable;
    // The live frontier's prompt, to tell whether any checkpoint covers it.
    std::vector<cache::Token> live_tokens;
    cache::Identity live_input;
    // Edges of replies within live_tokens, where a chat template trims
    // whitespace from message content: where a reply begins, and where it
    // ends or its final token begins. Kept across live continuations.
    std::vector<std::size_t> reply_starts, reply_ends;
    // Acquisition order; cold assignment replaces the oldest live frontier.
    std::uint64_t used{};
    bool busy{};
    bool frontier_active{};
    std::thread frontier_worker;
    // The session's shared-state account, when sessions share their
    // demand-mapped state (#715).
    std::shared_ptr<mem::StateAccount> account;
    // Disk writes that read this session's rows in place (Observe): the
    // tokens each covers and when it was captured. Until a write finishes,
    // reclaim keeps those rows mapped and drops the frontier last.
    struct ObservedWrite {
      std::weak_ptr<cache::BorrowedRows> rows;
      std::size_t tokens{};
      Clock::time_point captured;
    };
    std::vector<ObservedWrite> observed;
  };
  // Disk writes still reading an entry's rows in place: the deepest one's
  // tokens and when the oldest was captured. Forgets finished and retired
  // writes. Requires mutex.
  struct PendingWrites {
    std::size_t tokens{};
    std::optional<Clock::time_point> since;
  };
  static PendingWrites ObservedWrites(Entry& e) {
    PendingWrites writes;
    std::erase_if(e.observed, [&](const Entry::ObservedWrite& write) {
      const auto rows = write.rows.lock();
      if (!rows || !rows->IsValid())
        return true;
      writes.tokens = std::max(writes.tokens, write.tokens);
      writes.since = writes.since ? std::min(*writes.since, write.captured)
                                  : write.captured;
      return false;
    });
    return writes;
  }
  // A frontier far beyond every checkpoint of its prompt is the only copy of
  // that work. Requires mutex.
  static constexpr std::size_t kUncoveredTokens = 4096;
  bool Uncovered(const Entry& e) const {
    if (!e.live || e.live_tokens.size() <= kUncoveredTokens)
      return false;
    return !index.HasCheckpoint(
        {identity, e.live_tokens,
         cache::InputIdentity(e.live_tokens.size(), e.live_input)},
        e.live_tokens.size() - kUncoveredTokens);
  }
  // Order in which idle slots give up what they hold, cheapest first: no
  // uncovered frontier, completed preservation, no live frontier, a shallow
  // one, then the least recently used. Requires mutex.
  using Rank = std::tuple<bool, bool, bool, bool, std::uint64_t>;
  Rank ColdRank(const Entry& e) const {
    return Rank{Uncovered(e), !e.slot->PreservationComplete(), bool(e.live),
                e.live_tokens.size() >= kUncoveredTokens, e.used};
  }
  std::shared_ptr<TextModelRunner> runner;
  cache::ResourceLedger ledger;
  std::unique_ptr<ComponentCacheResources> resources;
  cache::Identity identity;
  cache::CompatibilityDigest digest;
  cache::DiskFileId nonce{};
  Events events;
  // Durable descriptions in the index retain a Store facade until index and
  // retention teardown have finished using its publication pins.
  std::unique_ptr<cache::DiskStore> disk;
  cache::PrefixIndex index;
  cache::RetentionPolicy retention;
  mutable std::mutex mutex;
  std::condition_variable frontier_ready;
  std::vector<std::unique_ptr<Entry>> entries;
  std::uint64_t clock{};
  std::unique_ptr<cache::DiskCatalog> catalog;
  std::unique_ptr<cache::StreamedStore> streaming;
  std::unique_ptr<cache::PersistenceQueue> queue;
  // Disk checkpoints read into the page cache ahead of their restore
  // (WarmDisk). Stopped and joined first on teardown.
  struct Warmup {
    cache::CheckpointId id;
    std::uint64_t bytes{};
    std::shared_ptr<std::atomic<bool>> done;
    std::jthread thread;
  };
  std::mutex warm_mutex;
  std::vector<std::unique_ptr<Warmup>> warmups;
  // Payload bytes of disk restores admitted but not captured yet. Their
  // captures share no RAM chunks, so each needs backing for its whole
  // checkpoint, and a new restore must prepare for all of them.
  std::atomic<std::size_t> pending_capture_bytes{};
  static constexpr std::size_t kPersistenceDepth = 8;
  // Backing that disk captures copy through when the RAM cache is off. It
  // equals the small explicit budget used before off existed, and it retains
  // nothing.
  static constexpr std::size_t kDiskOnlyBackingBytes = std::size_t{512} << 20;
  // How long a disk-only capture waits for queued writes to release that
  // backing. Each write releases it within seconds; this bounds a stall.
  static constexpr std::chrono::seconds kStagingWait{20};
  // Signalled when a disk write completes, so waiting captures retry.
  std::mutex staging_mutex;
  std::condition_variable staging_released;
  // Queue ownership plus transient submissions before superseded callbacks
  // release their serving handles. No callback runs under the serving mutex.
  std::array<std::shared_ptr<const cache::Checkpoint>, 2 * kPersistenceDepth>
      pending;
  std::array<std::shared_ptr<DurableCapture>, 2 * kPersistenceDepth>
      pending_durable;
  // At most one pressure demotion may hold payload beyond logical removal.
  // A full queue or this occupied slot drops further victims immediately.
  std::shared_ptr<const cache::Checkpoint> demotion;
  cache::CheckpointId demotion_id{};
  std::size_t disk_capacity{}, disk_step{};
  // The cache's own RAM budget. The ledger limit adds what the slots pinned.
  // Zero when the RAM cache is off.
  std::size_t budget{};
  // RAM backing committed only to copy captures on their way to disk.
  std::size_t disk_backing{};
  bool ram_enabled{true};
  // This cache's own share of the state pages it may hold (see budget).
  std::size_t lender_grant{};
  // Checkpoints evicted so sessions could borrow their pages back.
  std::atomic<std::size_t> state_sheds{};
  // A host-pressure episode was logged; cleared when the shortfall ends.
  bool host_pressure_reported{};
  cache::ResourceCharge metadata;

  // Sizes a streamed model's expert cache once the sessions exist and the
  // prompt cache committed its backing: what the cache holds plus the host
  // memory available now, less the runner's headroom and the prompt-cache
  // budget still to commit (`backing_bytes` beyond what was prepared). The
  // backing may hold less than it prepares: the log reports what it
  // committed.
  void FitExpertCache(const TextRunnerExpertCacheClaim& claim,
                      std::size_t backing_bytes) const {
    if (!claim.store)
      return;
#if defined(ENGINE_ENABLE_HIP)
    // Commit the whole budget before the grant. Where registered pages are
    // charged to the device heap, later growth must also leave the heap's
    // dispatch reserve, which the filled expert cache does not leave: a
    // budget kept back for that growth would stay unused by either cache.
    const bool whole = resources->commit_backing &&
                       resources->prepare_capture &&
                       backing_bytes > cache::kPreparedBackingBytes;
    if (whole) {
      resources->prepare_capture(backing_bytes);
      if (resources->await_capture)
        resources->await_capture();
    }
    std::size_t prepared = 0;
    if (whole)
      prepared = backing_bytes;
    else if (resources->commit_backing)
      prepared = std::min(backing_bytes, cache::kPreparedBackingBytes);
    const std::uint64_t pending = backing_bytes - prepared;
    const std::uint64_t committed = resources->committed_backing_bytes
                                        ? resources->committed_backing_bytes()
                                        : prepared;
    const std::uint64_t cached = claim.store->GetStats().cache_bytes;
    const std::uint64_t available = ext::HostAvailableBytes();
    const std::uint64_t keep = claim.headroom_bytes + pending;
    const std::uint64_t grant =
        available > keep ? cached + (available - keep)
                         : cached - std::min(cached, keep - available);
    Logger::Info(
        "cache",
        "schema=component-cache-v1 event=expert_cache_grant "
        "cached_bytes=" +
            std::to_string(cached) +
            " available_bytes=" + std::to_string(available) +
            " headroom_bytes=" + std::to_string(claim.headroom_bytes) +
            " prompt_cache_budget_bytes=" + std::to_string(budget) +
            " prompt_cache_committed_bytes=" + std::to_string(committed) +
            " prompt_cache_pending_bytes=" + std::to_string(pending) +
            " grant_bytes=" + std::to_string(grant));
    if (!claim.store->Fit(grant))
      Logger::Warn("cache",
                   "schema=component-cache-v1 event=expert_cache_fit_failed "
                   "grant_bytes=" +
                       std::to_string(grant));
#else
    (void)backing_bytes;
#endif
  }

  Impl(std::shared_ptr<TextModelRunner> runner, std::size_t count,
       std::size_t ram, std::size_t slot_ceiling,
       const std::optional<TextRunnerDiskCacheOptions>& options,
       bool automatic_ram, bool ram_on, TextRunnerExpertCacheClaim experts)
      : runner(std::move(runner)),
        // Disk transfers lease one fixed pinned staging buffer: one piece for
        // writes, whose speed a larger piece does not change (#562), and
        // kDiskRestoreReaders pieces that restores read in parallel.
        // Off disables checkpoint retention, not metadata accounting. Disk
        // recovery and publication must admit their bookkeeping from startup.
        // Persistence pins remain bounded by the disk-capture backing pool.
        ledger([&]() -> cache::ResourceLimits {
          const auto staging = options ? cache::kDefaultDiskStagingBytes : 0;
          if (ram_on)
            return {slot_ceiling + staging, slot_ceiling, staging,
                    slot_ceiling};
          const auto unlimited = std::numeric_limits<std::size_t>::max();
          return {unlimited, unlimited, staging, kDiskOnlyBackingBytes};
        }()),
        resources(this->runner->CreateComponentCacheResourcesForSessions(
            ledger, ram, count)),
        identity(resources->adapter->CompatibilityIdentity()),
        index(ledger),
        retention(ledger, index, &events),
        ram_enabled(ram_on) {
    metadata = ledger.Reserve(cache::ResourceCategory::kMetadata, sizeof(Impl))
                   .Convert();
    crypto::Sha256Hasher hash;
    hash.Update(identity);
    digest = hash.Finish();
    if (!platform::FillRandomBytes(nonce))
      throw std::runtime_error("cannot obtain cache publication nonce");
    index.Register(identity, resources->adapter->Components(), digest);
    if (options) {
      const auto row_bytes =
          cache::MinimumStagingBytes(resources->adapter->Components());
      // Writes move one staging piece at a time, whole rows per piece.
      if (std::min(ledger.Limits().staging_bytes,
                   cache::kDiskStagingPieceBytes) < row_bytes)
        throw std::logic_error("a " + std::to_string(row_bytes) +
                               "-byte cache row exceeds the disk staging "
                               "piece");
      if (options->staging_capacity_bytes)
        Logger::Warn("cache",
                     "schema=component-cache-v1 event=option_ignored "
                     "option=--cache-disk-staging-bytes requested_bytes=" +
                         std::to_string(options->staging_capacity_bytes) +
                         " staging_capacity_bytes=" +
                         std::to_string(ledger.Limits().staging_bytes));
      // Take directory ownership before slots and backing commit, so a
      // second server fails without claiming its RAM budget first.
      disk = std::make_unique<cache::DiskStore>(ledger, options->directory,
                                                options->capacity_bytes);
    }
    for (std::size_t i = 0; i < count; ++i) {
      auto entry = std::make_unique<Entry>();
      entry->preservation = Stream();
      entry->slot = std::make_unique<cache::LeasedSlot>(
          ledger, *resources->adapter, *entry->preservation,
          cache::SlotId{i + 1},
          cache::IdleSpillConfig{1ULL << 20, resources->stream,
                                 [this] {
                                   const std::lock_guard lock(mutex);
                                   return std::ranges::none_of(
                                       entries,
                                       [](const auto& e) { return e->busy; });
                                 },
                                 resources->reassignment_batch_size});
      entries.push_back(std::move(entry));
    }
    // Sessions that share their demand-mapped state (#715): learn each
    // session's account now, so admission can size and reclaim state before
    // a slot is chosen. Leasing an unused slot only creates it.
    if (mem::StateLedger::Process().PromiseCap() != 0)
      for (auto& entry : entries) {
        auto lease = entry->slot->Acquire();
        entry->account =
            this->runner
                ->BindComponentState(*resources->adapter, lease.Execution())
                ->state_account();
      }
    std::size_t automatic_bytes{}, max_bytes{}, slot_pinned_bytes{};
    std::size_t owned_manifest_bytes{};
    // RAM cache limit for a runner that pins its slots; computed before the
    // branch below so a short host refuses the load before anything commits.
    std::size_t pinned{}, limit{}, borrowable{}, own_grant{};
    if (ram_enabled && resources->commit_backing) {
      const auto claim = this->runner->ResourceClaim();
      // This observation is after slots and disk recovery allocated. Pinned
      // slots sit above the cache budget; recovered manifest vectors are
      // already inside it. Compare only the budget's remaining obligations
      // with remaining host headroom. Neither ledger reservations nor its
      // converted overhead estimates establish materialized storage.
      pinned = resources->pinned_bytes ? resources->pinned_bytes() : 0;
      owned_manifest_bytes = disk ? disk->OwnedManifestBytes() : 0;
      const auto measured = AutomaticRetainedBytes(claim, count);
      const auto owned = SaturatingAdd(pinned, owned_manifest_bytes);
      // Promised but unwritten session state the cache may borrow; its
      // pages go back to the sessions on demand. Zero without lending.
      borrowable = mem::StateLedger::Process().BorrowableBytes();
      const auto automatic =
          SaturatingAdd(SaturatingAdd(measured, owned / 2), borrowable);
      const auto host_ceiling =
          claim.retained_snapshot_ceiling_bytes.value_or(measured);
      const auto ceiling =
          SaturatingAdd(SaturatingAdd(host_ceiling, owned), borrowable);
      automatic_bytes =
          std::min(TextRunnerRamCacheOptions::kAutomaticMaxBytes, automatic);
      // Staging is pinned host memory too, so the host keeps its reserve
      // only if both fit under the ceiling.
      const auto staging = ledger.Limits().staging_bytes;
      const auto requested = ram;
      max_bytes = ceiling > staging ? ceiling - staging : 0;
      ram = std::min(ram, max_bytes);
      if (automatic_ram)
        ram = std::min(ram, automatic);
      // Within the host ceiling and the limit the slots allocated under. Other
      // charges above the budget are refused here, as without pins.
      limit = std::min(
          {SaturatingAdd(ram, pinned), max_bytes, ledger.Limits().ram_bytes});
      const auto charged = ledger.Snapshot().ram_bytes;
      const auto granted = std::min(ram, limit > pinned ? limit - pinned : 0);
      // The same grant without borrowing: the cache's own share.
      if (borrowable) {
        const auto own_max =
            max_bytes > borrowable ? max_bytes - borrowable : 0;
        auto own = std::min(requested, own_max);
        if (automatic_ram)
          own = std::min(own, automatic - borrowable);
        const auto own_limit = std::min(
            {SaturatingAdd(own, pinned), own_max, ledger.Limits().ram_bytes});
        own_grant = std::min(own, own_limit > pinned ? own_limit - pinned : 0);
      }
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=ram_admission "
          "requested_bytes=" +
              std::to_string(requested) +
              " host_ceiling_bytes=" + std::to_string(host_ceiling) +
              " owned_manifest_bytes=" + std::to_string(owned_manifest_bytes) +
              " slot_pinned_bytes=" + std::to_string(pinned) +
              " staging_capacity_bytes=" + std::to_string(staging) +
              " ledger_ram_bytes=" + std::to_string(charged) +
              " borrowable_bytes=" + std::to_string(borrowable) +
              " proposed_grant_bytes=" + std::to_string(granted) +
              " additional_bytes=" +
              std::to_string(SaturatingAdd(
                  granted - std::min(granted, owned_manifest_bytes), staging)));
      // The ceiling is measured after the slots allocated. When they leave
      // no RAM for the cache above the host reserve, refuse the load with
      // the remedies instead of serving without the cache the user expects
      // (#1053). An explicit budget below the other charges is the
      // operator's setting, not the host's state.
      const bool host_bound = automatic_ram || ram < requested;
      if (max_bytes < charged || !granted || (limit < charged && host_bound)) {
        Logger::Warn("cache",
                     "schema=component-cache-v1 event=ram_cache_refused "
                     "sessions=" +
                         std::to_string(count) + " " +
                         ext::DescribeHostMemory(ext::SampleHostMemory()));
        throw std::runtime_error(HostReserveMessage(
            count, requested, max_bytes > pinned ? max_bytes - pinned : 0));
      }
      if (limit < charged)
        throw std::runtime_error(
            "--cache-ram-bytes " + std::to_string(requested) +
            " is below the " + std::to_string(charged) +
            " bytes the cache already holds for its sessions and metadata");
    }
    if (!ram_enabled) {
      // Only disk captures copy through backing, so without a disk cache no
      // pool exists.
      ram = options ? kDiskOnlyBackingBytes : 0;
      disk_backing = ram;
      slot_pinned_bytes =
          resources->pinned_bytes ? resources->pinned_bytes() : 0;
      if (resources->commit_backing) {
        if (ram) {
          // The staging backing is all the cache's own: no part of it stands
          // for promised session state, and it lends no pages (#1260).
          lender_grant = ram;
          mem::StateLedger::Process().AddLenderGrant(lender_grant);
          resources->commit_backing(ram);
        } else {
          // No pool exists, so a payload allocation must fail, not crash.
          resources->prepare_prefill = nullptr;
          resources->prepare_capture = nullptr;
          resources->await_capture = nullptr;
          resources->can_capture = nullptr;
          resources->allocate =
              [](const cache::PayloadRequest&) -> ComponentCaptureBuffer {
            throw cache::ResourceExhausted();
          };
        }
      }
    } else if (resources->commit_backing) {
      const auto staging = ledger.Limits().staging_bytes;
      ledger.ConstrainLimits({limit + staging, limit, staging, limit});
      ram = std::min(ram, limit > pinned ? limit - pinned : 0);
      // Cache pages up to the grant it would have had without borrowing are
      // its own; only those beyond it stand for promised session state. The
      // backing reads it while committing.
      lender_grant = borrowable ? std::min(own_grant, ram) : ram;
      mem::StateLedger::Process().AddLenderGrant(lender_grant);
      resources->commit_backing(ram);
      slot_pinned_bytes = pinned;
    } else {
      // Slots allocated under the host ceiling; return to the budget. This
      // ignores pinned_bytes: a runner that pins must commit backing.
      const auto staging = ledger.Limits().staging_bytes;
      ledger.ConstrainLimits({ram + staging, ram, staging, ram});
    }
    budget = ram_enabled ? ram : 0;
    FitExpertCache(experts, ram);
    if (options) {
      catalog = std::make_unique<cache::DiskCatalog>(ledger, *disk, index,
                                                     options->capacity_bytes);
      for (const auto& durable : disk->Entries()) {
        cache::ExecutionHistory::ObserveDurableId(
            durable.Manifest().checkpoint);
        if (durable.Manifest().compatibility == digest)
          (void)index.Insert(identity,
                             catalog->Find(durable.Manifest().checkpoint));
      }
      streaming = std::make_unique<cache::StreamedStore>(
          ledger, *disk, ledger.Limits().staging_bytes, resources->staging,
          resources->stream);
      streaming->SetStreamClasses(resources->background_stream,
                                  resources->restore_stream);
      disk_capacity = options->capacity_bytes;
      disk_step = options->min_checkpoint_step_tokens;
      disk->ReclaimOrphans();
      if (disk->Stats().managed_bytes > disk_capacity) {
        // A lowered --cache-disk-bytes keeps its meaning: reclaim orphans and
        // temporaries first, then evict the oldest checkpoints by creation
        // (startup has no use history) to fit.
        disk->WaitForReclamation();
        std::size_t evicted{};
        while (disk->Stats().managed_bytes > disk_capacity &&
               catalog->EvictOne())
          ++evicted;
        const auto managed = disk->Stats().managed_bytes;
        const auto fields = " managed_bytes=" + std::to_string(managed) +
                            " capacity_bytes=" + std::to_string(disk_capacity);
        if (evicted)
          Logger::Info("cache",
                       "schema=component-cache-v1 event=disk_evicted "
                       "reason=budget entries=" +
                           std::to_string(evicted) + fields);
        if (managed > disk_capacity)
          Logger::Warn(
              "cache",
              "schema=component-cache-v1 event=disk_over_budget" + fields);
      }
      queue = std::make_unique<cache::PersistenceQueue>(
          ledger, *streaming, kPersistenceDepth, ram,
          [this] {
            const std::lock_guard lock(mutex);
            return std::ranges::none_of(
                entries, [](const auto& entry) { return entry->busy; });
          },
          [this](cache::CheckpointId id, bool success) {
            // The job releases its staged backing as it completes; waiting
            // captures also poll, so this only shortens their wait.
            staging_released.notify_all();
            const std::lock_guard lock(mutex);
            const bool was_demoted = id == demotion_id;
            if (was_demoted)
              demotion_id = {};
            for (auto& checkpoint : pending)
              if (checkpoint && checkpoint->Id() == id)
                checkpoint.reset();
            std::shared_ptr<DurableCapture> completed;
            for (auto& capture : pending_durable)
              if (capture && capture->manifest.checkpoint == id) {
                completed = capture;
                capture.reset();
              }
            if (!success) {
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=persistence_incomplete "
                  "checkpoint_id=" +
                      std::to_string(id.value));
              return;
            }
            catalog->Track(id);
            if (const auto description = catalog->Find(id)) {
              if (completed)
                for (auto& entry : entries)
                  if (entry->branch == completed->branch &&
                      (!entry->durable ||
                       entry->durable->Manifest().tokens.size() <
                           description->Manifest().tokens.size()))
                    entry->durable = description;
              auto found = index.Lookup(
                  {identity, description->Manifest().tokens,
                   cache::InputIdentity(description->Manifest().tokens.size(),
                                        description->Manifest().input)});
              bool attached = false;
              for (const auto& candidate : found.candidates) {
                if (candidate.checkpoint && candidate.checkpoint->Id() == id) {
                  index.AttachDurable(candidate.entry, description);
                  attached = true;
                }
              }
              if (!attached)
                (void)index.Insert(identity, description);
              std::size_t prior = 0;
              for (const auto& candidate : found.candidates)
                if (candidate.boundary <
                        description->Manifest().tokens.size() &&
                    Published(candidate) && SpacedCheckpoint(candidate))
                  prior = std::max(
                      prior, static_cast<std::size_t>(candidate.boundary));
              const bool spacing_exempt =
                  description->Manifest().purpose ==
                      cache::CheckpointPurpose::kGenerated ||
                  description->Manifest().purpose ==
                      cache::CheckpointPurpose::kLearned ||
                  (was_demoted && prior &&
                   description->Manifest().tokens.size() - prior < disk_step);
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=published tokens=" +
                      std::to_string(description->Manifest().tokens.size()) +
                      " checkpoint_id=" + std::to_string(id.value) +
                      " purpose=" +
                      std::to_string(static_cast<unsigned>(
                          description->Manifest().purpose)) +
                      " spacing_exempt=" + std::to_string(spacing_exempt) +
                      " payload_bytes=" +
                      std::to_string(PayloadBytes(description->Manifest())));
            }
          },
          [this](const cache::DiskManifest& manifest) {
            // A checkpoint larger than the entire disk budget cannot be
            // admitted by deleting other conversations.
            const auto bytes = PayloadBytes(manifest);
            if (bytes > disk_capacity)
              return false;
            // Startup orphan reclamation holds the store lock asynchronously
            // and eviction only tries that lock, so a write racing it would
            // be dropped. Only this worker joins it once the queue exists.
            try {
              disk->WaitForReclamation();
            } catch (const std::exception& error) {
              Logger::Warn("cache", std::string("schema=component-cache-v1 "
                                                "event=disk_reclamation_failed "
                                                "reason=") +
                                        error.what());
            }
            const std::lock_guard lock(mutex);
            return LogDiskEviction(catalog->EvictOne());
          });
    }
    if (queue && ram_enabled)
      retention.SetDemotion([this](auto checkpoint) {
        if (!demotion_id.value &&
            checkpoint->Purpose() != cache::CheckpointPurpose::kGrid) {
          demotion_id = checkpoint->Id();
          demotion = std::move(checkpoint);
        }
      });
    Logger::Info(
        "cache",
        "schema=component-cache-v1 event=configured sessions=" +
            std::to_string(count) + " snapshot_entries=128 capacity_bytes=" +
            std::to_string(budget) + " staging_capacity_bytes=" +
            std::to_string(ledger.Limits().staging_bytes) +
            " automatic_bytes=" + std::to_string(automatic_bytes) +
            " max_bytes=" + std::to_string(max_bytes) +
            " disk_capacity_bytes=" + std::to_string(disk_capacity) +
            " slot_pinned_bytes=" + std::to_string(slot_pinned_bytes) +
            " owned_manifest_bytes=" + std::to_string(owned_manifest_bytes) +
            " ram_cache=" + (ram_enabled ? "on" : "off") +
            " disk_backing_bytes=" + std::to_string(disk_backing) +
            " borrowable_bytes=" + std::to_string(borrowable) +
            " own_grant_bytes=" + std::to_string(lender_grant));
    // Sessions that need pages back while the host is at its floor evict
    // checkpoints here, as capture pressure does: the state pool's thread
    // then maps the freed pages into the session (no new memory).
    if (ram_enabled && resources->set_shed)
      resources->set_shed([this] {
        DiscardUnusedFrontierWrites();
        bool reclaimed;
        {
          const std::lock_guard lock(mutex);
          reclaimed = retention.Reclaim(cache::RetentionRequest{});
        }
        DrainDemotion();
        if (reclaimed)
          ++state_sheds;
        return reclaimed;
      });
  }
  ~Impl() {
    {
      decltype(warmups) stopped;
      {
        const std::lock_guard lock(warm_mutex);
        stopped.swap(warmups);
      }
    }
    if (resources->set_shed) {
      resources->set_shed(nullptr);
      Logger::Info("cache",
                   "schema=component-cache-v1 event=state_lending sheds=" +
                       std::to_string(state_sheds.load()));
    }
    if (lender_grant)
      mem::StateLedger::Process().RemoveLenderGrant(lender_grant);
    for (auto& entry : entries)
      if (entry->frontier_worker.joinable())
        entry->frontier_worker.join();
    retention.SetDemotion({});
    DrainDemotion();
    // Settle immutable-source readers before slots/resources/index unwind.
    if (queue)
      queue->Drain();
    for (auto& entry : entries)
      entry->slot->StopIdleSpill();
  }
  std::unique_ptr<cache::Stream> Stream() {
    auto stream = resources->stream();
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    return stream;
  }
  cache::DiskFileId File(std::uint64_t kind, std::uint64_t id) const {
    crypto::Sha256Hasher hash;
    hash.Update(nonce);
    std::array<std::uint8_t, 16> suffix{};
    for (std::size_t i = 0; i < 8; ++i) {
      suffix[i] = static_cast<std::uint8_t>(kind >> (8 * i));
      suffix[8 + i] = static_cast<std::uint8_t>(id >> (8 * i));
    }
    hash.Update(suffix);
    const auto bytes = hash.Finish();
    cache::DiskFileId file;
    std::copy_n(bytes.begin(), file.size(), file.begin());
    return file;
  }
  std::size_t Persist(std::shared_ptr<const cache::Checkpoint> checkpoint,
                      bool demoting = false, bool grid = false) {
    // Grid captures support RAM edits during prefill. Persist semantic prompt
    // and learned boundaries, plus the one grid a lease selects to keep a
    // long document's prefix on disk (Lease::PersistGrid). Grids never count
    // as spacing priors, so they cannot consume the allowance of a turn.
    if (!queue ||
        (checkpoint->Purpose() == cache::CheckpointPurpose::kGrid && !grid))
      return 0;
    {
      const std::lock_guard lock(mutex);
      auto compatible_prefix = [&](const auto& tokens, const auto& input) {
        return input == checkpoint->Input() &&
               tokens.size() <= checkpoint->Boundary() &&
               std::equal(tokens.begin(), tokens.end(),
                          checkpoint->Tokens().begin());
      };
      if (!demoting &&
          checkpoint->Purpose() != cache::CheckpointPurpose::kLearned &&
          checkpoint->Purpose() != cache::CheckpointPurpose::kGenerated) {
        auto found = index.Lookup({identity, checkpoint->Tokens(),
                                   cache::InputIdentity(checkpoint->Boundary(),
                                                        checkpoint->Input())});
        std::size_t prior = 0;
        for (const auto& candidate : found.candidates)
          if (Published(candidate) && SpacedCheckpoint(candidate))
            prior =
                std::max(prior, static_cast<std::size_t>(candidate.boundary));
        for (const auto& queued : pending)
          if (queued && compatible_prefix(queued->Tokens(), queued->Input()) &&
              Spaced(queued->Purpose()))
            prior =
                std::max(prior, static_cast<std::size_t>(queued->Boundary()));
        for (const auto& queued : pending_durable)
          if (queued &&
              compatible_prefix(queued->manifest.tokens,
                                queued->manifest.input) &&
              Spaced(queued->manifest.purpose))
            prior = std::max(prior, queued->manifest.tokens.size());
        if (prior && checkpoint->Boundary() - prior < disk_step) {
          Logger::Info(
              "cache",
              "schema=component-cache-v1 event=persistence_refused "
              "reason=min_step tokens=" +
                  std::to_string(checkpoint->Boundary()) +
                  " checkpoint_id=" + std::to_string(checkpoint->Id().value));
          return 0;
        }
      }
    }
    cache::DiskManifest manifest{
        .checkpoint = checkpoint->Id(),
        .lineage = checkpoint->Lineage(),
        .compatibility = digest,
        .input = checkpoint->Input(),
        .tokens = {checkpoint->Tokens().begin(), checkpoint->Tokens().end()},
        .purpose = checkpoint->Purpose(),
        .rank = checkpoint->Rank()};
    std::vector<cache::PersistenceSource> sources;
    std::uint64_t private_id = checkpoint->Id().value;
    auto append = [&](const cache::Payload& payload, std::size_t bytes,
                      bool private_file, cache::DiskFileId file) {
      if (auto rows = payload.BorrowedStorage()) {
        // Read rows still in a slot from there: idle preservation only runs
        // while no request is busy, and a waiting job keeps its pins.
        sources.emplace_back(private_file, file, bytes, 1, std::move(rows),
                             resources->copy_observed_rows);
        return cache::DiskPayload{file, bytes, 0};
      }
      auto owner = payload.Owner();
      if (!owner)
        throw std::logic_error("persistence source is not immutable");
      if (resources->device_only && resources->copy_observed_rows &&
          resources->device_only(owner.get())) {
        // Shared session pages have no host view: the provider copies them
        // out on the writer's own thread where it can, outside the write's
        // stream, else through that stream as rows still in a slot.
        sources.emplace_back(
            private_file, file, bytes, 1, payload.CommittedCharge(), owner,
            [owner, checkpoint, direct = resources->copy_device_only,
             copy = resources->copy_observed_rows](
                std::uint64_t offset, std::span<std::byte> destination,
                cache::Stream& stream) {
              const std::span<const std::byte> source(
                  static_cast<const std::byte*>(owner.get()) + offset,
                  destination.size());
              if (direct && direct(source, destination))
                return cache::Completion(std::make_unique<SettledSignal>());
              return copy(source, destination, stream);
            });
        return cache::DiskPayload{file, bytes, 0};
      }
      sources.emplace_back(
          private_file, file, bytes, 1, payload.CommittedCharge(), owner,
          [owner, checkpoint](std::uint64_t offset,
                              std::span<std::byte> destination,
                              cache::Stream&) {
            std::memcpy(destination.data(),
                        static_cast<const std::byte*>(owner.get()) + offset,
                        destination.size());
            return cache::Completion(std::make_unique<SettledSignal>());
          });
      return cache::DiskPayload{file, bytes, 0};
    };
    for (const auto& c : checkpoint->Components()) {
      cache::DiskComponent component{.descriptor = c.descriptor,
                                     .position = c.position};
      if (c.private_state)
        component.private_state =
            append(*c.private_state, c.descriptor.state_bytes, true,
                   File(2 + c.descriptor.id.value, private_id));
      for (const auto& chunk : c.chunks)
        component.chunks.push_back(
            append(chunk.Storage(),
                   (chunk.End() - chunk.First()) * c.descriptor.row_bytes,
                   false, File(0, chunk.Id().value)));
      if (c.tail)
        component.tail = append(
            *c.tail,
            (c.position.valid_rows % c.descriptor.rows_per_chunk) *
                c.descriptor.row_bytes,
            true, File(0x100000000ULL + c.descriptor.id.value, private_id));
      manifest.components.push_back(std::move(component));
    }
    std::size_t bytes = 0;
    for (const auto& c : manifest.components) {
      for (const auto& chunk : c.chunks)
        bytes += chunk.bytes;
      if (c.tail)
        bytes += c.tail->bytes;
      if (c.private_state)
        bytes += c.private_state->bytes;
    }
    auto job = cache::PersistenceJob(ledger, File(1, private_id), manifest,
                                     std::move(sources));
    std::size_t pending_slot;
    {
      const std::lock_guard lock(mutex);
      auto position = std::find(pending.begin(), pending.end(), nullptr);
      if (position == pending.end())
        return 0;
      pending_slot = position - pending.begin();
      *position = checkpoint;
    }
    if (!queue->TrySubmit(std::move(job))) {
      const std::lock_guard lock(mutex);
      if (pending[pending_slot] == checkpoint)
        pending[pending_slot].reset();
      return 0;
    }
    return bytes;
  }
  // Backing bytes that captures hold. Without the RAM cache only captures on
  // their way to disk hold any.
  std::size_t StagedBytes() const {
    const auto snapshot = ledger.Snapshot();
    std::size_t held = 0;
    for (const auto category : {cache::ResourceCategory::kBackingAssigned,
                                cache::ResourceCategory::kBackingMaterialized,
                                cache::ResourceCategory::kPrivateState,
                                cache::ResourceCategory::kPrivateTail})
      held += snapshot.bytes[static_cast<std::size_t>(category)];
    return held;
  }
  // Waits up to one short poll for a queued write to release staged backing.
  // False when no capture holds any, or at the deadline.
  bool AwaitStaging(Clock::time_point deadline) {
    const auto now = Clock::now();
    if (now >= deadline || !StagedBytes())
      return false;
    std::unique_lock lock(staging_mutex);
    staging_released.wait_until(
        lock, std::min(deadline, now + std::chrono::milliseconds(20)));
    return true;
  }
  void DiscardUnusedFrontierWrites() {
    if (!queue)
      return;
    // Select under serving serialization, but discard/callback outside it.
    // An active write is left alone; foreground work never drains disk I/O.
    const auto discard = [&](const auto& jobs, const auto& select) {
      for (std::size_t i = 0; i < jobs.size(); ++i) {
        cache::CheckpointId id{};
        {
          const std::lock_guard lock(mutex);
          if (jobs[i])
            id = select(*jobs[i]);
        }
        if (id.value)
          (void)queue->DiscardPending(id);
      }
    };
    discard(pending, [&](const auto& checkpoint) {
      return retention.IsUnusedFrontier(checkpoint.Id())
                 ? checkpoint.Id()
                 : cache::CheckpointId{};
    });
    discard(pending_durable, [](const auto& capture) {
      return capture.manifest.purpose == cache::CheckpointPurpose::kGenerated
                 ? capture.manifest.checkpoint
                 : cache::CheckpointId{};
    });
  }

  void DrainDemotion() noexcept {
    std::shared_ptr<const cache::Checkpoint> checkpoint;
    {
      const std::lock_guard lock(mutex);
      checkpoint = std::move(demotion);
    }
    if (!checkpoint)
      return;
    bool queued = false;
    try {
      queued = Persist(checkpoint, true) != 0;
    } catch (...) {
    }
    if (!queued) {
      const std::lock_guard lock(mutex);
      if (demotion_id == checkpoint->Id())
        demotion_id = {};
    }
  }
  bool WithinDiskStep(std::span<const cache::Token> tokens,
                      const cache::Identity& input) {
    const std::lock_guard lock(mutex);
    auto found = index.Lookup(
        {identity, tokens, cache::InputIdentity(tokens.size(), input)});
    std::size_t prior = 0;
    for (const auto& candidate : found.candidates)
      if (Published(candidate) && SpacedCheckpoint(candidate))
        prior = std::max(prior, static_cast<std::size_t>(candidate.boundary));
    const auto compatible = [&](auto prefix, const auto& prefix_input) {
      return prefix_input == input && prefix.size() <= tokens.size() &&
             std::equal(prefix.begin(), prefix.end(), tokens.begin());
    };
    for (const auto& checkpoint : pending)
      if (checkpoint && compatible(checkpoint->Tokens(), checkpoint->Input()) &&
          Spaced(checkpoint->Purpose()))
        prior =
            std::max(prior, static_cast<std::size_t>(checkpoint->Boundary()));
    for (const auto& capture : pending_durable)
      if (capture &&
          compatible(capture->manifest.tokens, capture->manifest.input) &&
          Spaced(capture->manifest.purpose))
        prior = std::max(prior, capture->manifest.tokens.size());
    return prior && tokens.size() - prior < disk_step;
  }
  std::size_t Persist(std::shared_ptr<DurableCapture> capture) {
    if (capture->sources.empty())
      return 0;
    // A refused capture lets go of its sources at once: rows it observes
    // in place would otherwise read as a pending write.
    if (WithinDiskStep(capture->manifest.tokens, capture->manifest.input) &&
        capture->manifest.purpose != cache::CheckpointPurpose::kLearned &&
        capture->manifest.purpose != cache::CheckpointPurpose::kGenerated) {
      capture->sources.clear();
      return 0;
    }
    // The job's metadata is optional too: a full ledger refuses the write,
    // not the request that captured it.
    std::optional<cache::PersistenceJob> job;
    try {
      job.emplace(ledger, File(1, capture->manifest.checkpoint.value),
                  capture->manifest, std::move(capture->sources));
    } catch (const std::bad_alloc&) {
      capture->sources.clear();
      Logger::Info("cache",
                   "schema=component-cache-v1 event=persistence_refused "
                   "reason=allocation tokens=" +
                       std::to_string(capture->manifest.tokens.size()));
      return 0;
    }
    std::size_t pending_slot;
    {
      const std::lock_guard lock(mutex);
      const auto position =
          std::find(pending_durable.begin(), pending_durable.end(), nullptr);
      if (position == pending_durable.end()) {
        capture->sources.clear();
        return 0;
      }
      pending_slot = position - pending_durable.begin();
      *position = capture;
    }
    if (!queue->TrySubmit(std::move(*job))) {
      const std::lock_guard lock(mutex);
      if (pending_durable[pending_slot] == capture)
        pending_durable[pending_slot].reset();
      return 0;
    }
    return PayloadBytes(capture->manifest);
  }
};

struct ComponentTextCache::Lease::Impl {
  ComponentTextCache::Impl* cache;
  ComponentTextCache::Impl::Entry* entry;
  cache::SlotLease slot;
  std::unique_ptr<TextRunnerState> state;
  cache::InputIdentity input;
  std::vector<cache::Token> prompt;
  std::size_t cached{}, restored{}, stable{}, capture_boundary{};
  // Restored bytes that took no copy: rows whose pages the slot now shares.
  std::size_t remapped{};
  // The planned grid boundary this lease also persists (0: none).
  std::size_t disk_grid{};
  double restore_time{};
  bool disk_hit{}, active{true}, preserve_source{};
  // False keeps every checkpoint this lease captures in RAM only.
  bool persist{true};
  bool generated_frontier{false};
  cache::CheckpointId source{};
  std::array<cache::CheckpointId, cache::RetentionPolicy::kRecordLimit>
      edit_boundaries{};
  std::size_t edit_boundary_count{};
  cache::ResourceCharge edit_metadata;
  SnapshotPurpose purpose{SnapshotPurpose::kContinuation};
  std::vector<cache::Token> capture_tokens;
  // Reply edges of the live frontier this lease continues.
  std::vector<std::size_t> reply_starts, reply_ends;
  ContinuationLookup lookup_result;
  ComponentCacheMetrics metrics;
  // The session's shared-state account; its need returns to the live
  // frontier (or nothing) when the lease ends.
  std::shared_ptr<mem::StateAccount> account;
  // Bytes of a disk restore whose first capture has not run yet, counted in
  // the cache's pending_capture_bytes until then.
  std::size_t pending_capture{};
  ~Impl() {
    SettleCapture();
    if (active && account)
      (void)account->SetNeed(0);
  }
  void SettleCapture() noexcept {
    if (pending_capture)
      cache->pending_capture_bytes.fetch_sub(pending_capture);
    pending_capture = 0;
  }
  Impl(ComponentTextCache::Impl* cache, ComponentTextCache::Impl::Entry* entry,
       cache::SlotLease&& slot, std::span<const cache::Token> prompt,
       cache::InputIdentity input, std::size_t stable)
      : cache(cache),
        entry(entry),
        input(std::move(input)),
        prompt(prompt.begin(), prompt.end()),
        stable(stable),
        edit_metadata(cache->ledger
                          .Reserve(cache::ResourceCategory::kMetadata,
                                   sizeof(edit_boundaries))
                          .Convert()) {
    // Leave the caller's lease intact until all fallible setup is complete.
    state = cache->runner->BindComponentState(*cache->resources->adapter,
                                              slot.Execution());
    this->slot = std::move(slot);
  }
  cache::RetentionRequest Retain(
      std::span<const cache::Token> tokens,
      std::optional<std::size_t> new_payload_bytes = {}) const {
    return {
        .tokens = tokens,
        .compatibility = cache->identity,
        .input = At(input, std::min(tokens.size(), prompt.size())),
        .purpose = Purpose(purpose),
        .stable = std::min(stable, tokens.size()),
        .source = source,
        .preserve_source = preserve_source,
        .new_payload_bytes =
            new_payload_bytes
                ? *new_payload_bytes
                : entry->history->NewPayloadBytes(
                      cache->resources->adapter->Positions(slot.Execution())),
        .protected_checkpoints =
            std::span(edit_boundaries).first(edit_boundary_count),
        .reused = cached != 0,
        .reused_tokens = cached,
        .turn = entry->used,
        .persistable = persist,
        .optional_frontier = generated_frontier};
  }
  [[nodiscard]] bool DiskGrid(std::size_t boundary) const noexcept {
    return disk_grid && boundary == disk_grid;
  }
  std::unique_ptr<ContinuationSnapshot> CaptureDurable();
};

std::unique_ptr<ContinuationSnapshot>
ComponentTextCache::Lease::Impl::CaptureDurable() {
  const bool grid = purpose == SnapshotPurpose::kHistory;
  if (!persist || !cache->queue || !cache->resources->copy_observed_rows ||
      (grid && !DiskGrid(capture_tokens.size())))
    throw cache::ResourceExhausted();
  const auto capture_input =
      At(input, std::min(capture_tokens.size(), prompt.size()));
  if (!generated_frontier && purpose != SnapshotPurpose::kBranchPoint &&
      cache->WithinDiskStep(capture_tokens, capture_input))
    return {};
  const auto positions = cache->resources->adapter->Positions(slot.Execution());
  const auto descriptors = cache->resources->adapter->Components();
  struct Copy {
    cache::PayloadRequest request;
    ComponentCaptureBuffer buffer;
    cache::DiskFileId file;
    bool private_file;
  };
  std::size_t pieces = 0;
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    if (p == positions.end())
      throw std::logic_error("incomplete durable capture positions");
    pieces += d.kind == cache::ComponentKind::kPrivateState
                  ? 1
                  : (p->valid_rows - cache::FirstChunkRow(d, p->valid_rows)) /
                            d.rows_per_chunk +
                        bool(p->valid_rows % d.rows_per_chunk);
  }
  // Admit metadata before vectors/owners. No append payload allocation is
  // admitted for the native rows: observations may be retired at any guard.
  auto admission = cache->ledger.Reserve(
      cache::ResourceCategory::kMetadata,
      sizeof(DurableCapture) + sizeof(ComponentSnapshot) +
          capture_tokens.size() * sizeof(cache::Token) + capture_input.size() +
          descriptors.size() * sizeof(cache::DiskComponent) +
          pieces * (sizeof(cache::DiskPayload) +
                    sizeof(cache::PersistenceSource) + sizeof(Copy) + 256));
  auto durable = std::make_shared<DurableCapture>();
  durable->metadata = admission.Convert();
  std::shared_ptr<const cache::DiskDescription> ancestor;
  {
    const std::lock_guard lock(cache->mutex);
    durable->branch = entry->branch;
    ancestor = entry->durable;
  }
  if (ancestor) {
    const auto& previous = ancestor->Manifest();
    if (previous.tokens.size() > capture_tokens.size() ||
        !std::equal(previous.tokens.begin(), previous.tokens.end(),
                    capture_tokens.begin()) ||
        previous.input != At(input, previous.tokens.size()))
      ancestor.reset();
    else {
      durable->ancestor_pin = ancestor->Pin();
      if (!durable->ancestor_pin)
        ancestor.reset();
    }
  }
  auto& manifest = durable->manifest;
  manifest.checkpoint = cache::ExecutionHistory::AllocateCheckpointId();
  manifest.lineage =
      ancestor ? ancestor->Manifest().lineage : entry->history->Lineage();
  manifest.compatibility = cache->digest;
  manifest.input = capture_input;
  manifest.tokens = capture_tokens;
  manifest.purpose = generated_frontier ? cache::CheckpointPurpose::kGenerated
                     : purpose == SnapshotPurpose::kBranchPoint
                         ? cache::CheckpointPurpose::kLearned
                     : grid ? cache::CheckpointPurpose::kGrid
                            : cache::CheckpointPurpose::kPrompt;
  manifest.components.reserve(descriptors.size());
  durable->sources.reserve(pieces);
  std::vector<Copy> copies;
  copies.reserve(pieces);
  std::vector<std::shared_ptr<cache::BorrowedRows>> observed;
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    cache::DiskComponent component{.descriptor = d, .position = *p};
    const auto completed =
        cache::HasRows(d.kind)
            ? p->valid_rows / d.rows_per_chunk * d.rows_per_chunk
            : 0;
    // Window components start at their first retained chunk, not row 0.
    const auto start = cache::FirstChunkRow(d, p->valid_rows);
    cache::Rows inherited = start;
    if (completed > start && ancestor) {
      const auto& components = ancestor->Manifest().components;
      const auto previous = std::ranges::find_if(
          components, [&](const auto& c) { return c.descriptor.id == d.id; });
      if (previous != components.end()) {
        // The ancestor's chunk j covers row previous_first + j rows_per_chunk;
        // its window never starts past this capture's.
        const auto previous_first =
            cache::FirstChunkRow(d, previous->position.valid_rows);
        const auto skip =
            std::min<std::size_t>(previous->chunks.size(),
                                  (start - previous_first) / d.rows_per_chunk);
        const auto chunks =
            std::min<std::size_t>(previous->chunks.size() - skip,
                                  (completed - start) / d.rows_per_chunk);
        component.chunks.assign(previous->chunks.begin() + skip,
                                previous->chunks.begin() + skip + chunks);
        inherited = start + chunks * d.rows_per_chunk;
      }
    }
    std::shared_ptr<cache::BorrowedRows> rows;
    if (completed > start) {
      if (inherited != completed) {
        auto view = cache->resources->adapter->BorrowRowsView(
            slot.Execution(), d.id, inherited, completed);
        if (!view.empty()) {
          rows = slot.Observe(d.id, inherited, completed, view);
          observed.push_back(rows);
        }
      }
      component.chunks.reserve((completed - start) / d.rows_per_chunk);
    }
    const auto capture = [&](cache::Rows first, cache::Rows end,
                             bool private_file) {
      const auto bytes = d.kind == cache::ComponentKind::kPrivateState
                             ? d.state_bytes
                             : (end - first) * d.row_bytes;
      const auto file =
          cache->File(0, cache::ExecutionHistory::AllocateCheckpointId().value);
      if (rows && !private_file) {
        durable->sources.emplace_back(false, file, bytes, d.row_bytes, rows,
                                      (first - inherited) * d.row_bytes,
                                      cache->resources->copy_observed_rows);
      } else {
        const cache::PayloadRequest request{
            d.id,
            d.kind == cache::ComponentKind::kPrivateState
                ? cache::ResourceCategory::kPrivateState
            : private_file ? cache::ResourceCategory::kPrivateTail
                           : cache::ResourceCategory::kBackingAssigned,
            first, end, bytes};
        copies.push_back({request, {}, file, private_file});
      }
      return cache::DiskPayload{file, bytes, 0};
    };
    if (d.kind == cache::ComponentKind::kPrivateState)
      component.private_state = capture(0, 0, true);
    else {
      for (cache::Rows first = inherited; first < completed;
           first += d.rows_per_chunk)
        component.chunks.push_back(
            capture(first, first + d.rows_per_chunk, false));
      if (completed != p->valid_rows)
        component.tail = capture(completed, p->valid_rows, true);
    }
    manifest.components.push_back(std::move(component));
  }
  // Private states, tails, and rows the slot cannot lend are staged in RAM
  // backing. Allocate them largest first, as Capture does. Reclaim for the
  // staged bytes only, because Reclaim refuses a request larger than RAM. The
  // disk checkpoint does not replace a retained one, so keep the source.
  std::vector<cache::PayloadRequest> requests;
  requests.reserve(copies.size());
  std::size_t staged = 0;
  for (const auto& copy : copies) {
    requests.push_back(copy.request);
    staged += copy.request.bytes;
  }
  std::vector<std::size_t> allocation_order(copies.size());
  std::iota(allocation_order.begin(), allocation_order.end(), 0);
  std::ranges::sort(allocation_order, [&](auto a, auto b) {
    return requests[a].bytes != requests[b].bytes
               ? requests[a].bytes > requests[b].bytes
               : a < b;
  });
  auto retain = Retain(capture_tokens, staged);
  retain.preserve_source = true;
  // Nearby retry copies may replace their source even with preserve_source.
  // Disk staging publishes no RAM replacement, so protect it explicitly along
  // with this turn's edit boundaries.
  std::array<cache::CheckpointId, cache::RetentionPolicy::kRecordLimit + 1>
      protected_checkpoints{};
  auto protected_count = retain.protected_checkpoints.size();
  std::ranges::copy(retain.protected_checkpoints,
                    protected_checkpoints.begin());
  if (retain.source.value)
    protected_checkpoints[protected_count++] = retain.source;
  retain.protected_checkpoints =
      std::span(protected_checkpoints).first(protected_count);
  // Without the RAM cache the backing retains nothing: it is full only while
  // other captures wait to be written, and each is queued as it is staged
  // (below). Wait for those writes rather than drop this checkpoint (#1260).
  // Exact frontiers stay optional, as in the persistence queue.
  const bool await_staging = !cache->ram_enabled && !generated_frontier;
  const auto wait_start = Clock::now();
  bool waited = false;
  while (!copies.empty()) {
    try {
      if (cache->resources->can_capture &&
          !cache->resources->can_capture(requests))
        throw cache::ResourceExhausted();
      for (const auto i : allocation_order)
        copies[i].buffer = cache->resources->allocate(requests[i]);
      break;
    } catch (const cache::ResourceExhausted&) {
      for (auto& copy : copies)
        copy.buffer = {};
      cache->DiscardUnusedFrontierWrites();
      bool reclaimed;
      {
        const std::lock_guard lock(cache->mutex);
        reclaimed = cache->retention.Reclaim(retain);
      }
      cache->DrainDemotion();
      if (reclaimed)
        continue;
      if (await_staging &&
          cache->AwaitStaging(wait_start +
                              ComponentTextCache::Impl::kStagingWait)) {
        waited = true;
        continue;
      }
      if (!cache->ram_enabled)
        Logger::Info(
            "cache",
            "schema=component-cache-v1 event=staging_exhausted tokens=" +
                std::to_string(capture_tokens.size()) +
                " staged_bytes=" + std::to_string(staged) +
                " held_bytes=" + std::to_string(cache->StagedBytes()) +
                " backing_bytes=" + std::to_string(cache->disk_backing) +
                " waited_ms=" + std::to_string(MillisecondsSince(wait_start)) +
                " frontier=" + (generated_frontier ? "1" : "0"));
      throw;
    }
  }
  if (waited)
    Logger::Info("cache",
                 "schema=component-cache-v1 event=staging_waited tokens=" +
                     std::to_string(capture_tokens.size()) +
                     " staged_bytes=" + std::to_string(staged) + " waited_ms=" +
                     std::to_string(MillisecondsSince(wait_start)));
  TransferBatch transfers(cache->resources->transfer_batch_size);
  for (auto& copy : copies) {
    const auto& r = copy.request;
    transfers.Add([&] { return cache->resources->stream(); },
                  [&](cache::Stream& stream) {
                    return r.category == cache::ResourceCategory::kPrivateState
                               ? cache->resources->adapter->CapturePrivate(
                                     slot.Execution(), r.component,
                                     copy.buffer.bytes.first(r.bytes), stream)
                               : cache->resources->adapter->CopyRowsOut(
                                     slot.Execution(), r.component, r.first,
                                     r.end, copy.buffer.bytes.first(r.bytes),
                                     stream);
                  });
  }
  transfers.Finish();
  for (auto& copy : copies) {
    auto payload = copy.buffer.finish();
    auto owner = payload.Owner();
    metrics.captured_bytes += payload.Bytes();
    metrics.copied_payload_bytes += copy.request.bytes;
    durable->sources.emplace_back(
        copy.private_file, copy.file, copy.request.bytes,
        copy.request.category == cache::ResourceCategory::kPrivateState
            ? 1
            : std::ranges::find(descriptors, copy.request.component,
                                &cache::ComponentDescriptor::id)
                  ->row_bytes,
        payload.CommittedCharge(), owner,
        [owner](auto offset, auto destination, cache::Stream&) {
          std::memcpy(destination.data(),
                      static_cast<const std::byte*>(owner.get()) + offset,
                      destination.size());
          return cache::Completion(std::make_unique<SettledSignal>());
        });
  }
  ++metrics.captures;
  metrics.captured_checkpoint_ids.push_back(durable->manifest.checkpoint.value);
  if (!observed.empty()) {
    const std::lock_guard lock(cache->mutex);
    (void)ComponentTextCache::Impl::ObservedWrites(*entry);
    for (auto& rows : observed)
      entry->observed.push_back(
          {std::move(rows), capture_tokens.size(), Clock::now()});
  }
  // Without the RAM cache nothing reads this capture but its disk write, and
  // the request may hold it until it ends. Queue it now, so its staged
  // backing returns while other captures wait for it. Publication then finds
  // no sources left to queue.
  if (!cache->ram_enabled) {
    const auto started = Clock::now();
    metrics.disk_queued_bytes += cache->Persist(durable);
    metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
  }
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->bytes = entry->history->NewPayloadBytes(positions);
  snapshot->durable = std::move(durable);
  return snapshot;
}

ComponentTextCache::ComponentTextCache(
    std::shared_ptr<TextModelRunner> runner, std::size_t count, std::size_t ram,
    const std::optional<TextRunnerDiskCacheOptions>& disk, bool automatic_ram,
    std::size_t ceiling, bool ram_enabled, TextRunnerExpertCacheClaim experts)
    : impl_(std::make_unique<Impl>(std::move(runner), count, ram,
                                   std::max(ram, ceiling), disk, automatic_ram,
                                   ram_enabled, experts)) {}
ComponentTextCache::~ComponentTextCache() = default;
std::size_t ComponentTextCache::capacity() const noexcept {
  return impl_->entries.size();
}
std::size_t ComponentTextCache::ram_budget() const noexcept {
  return impl_->budget;
}

std::unique_ptr<ComponentTextCache::Lease> ComponentTextCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const std::function<bool()>& cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse,
    std::size_t stable, bool stop_at_eos, bool persist,
    std::size_t state_tokens, mem::StateReservation* reservation,
    const RestoreRunner& run_copies) {
  if (cancelled && cancelled())
    return {};
  auto input = Inputs(prompt.size(), context.get());
  cache::PrefixLookup found;
  Impl::Entry* entry = nullptr;
  std::optional<cache::BorrowedLocation> live;
  std::vector<std::size_t> reply_starts, reply_ends;
  std::size_t common = 0;
  {
    std::unique_lock lock(impl_->mutex);
    impl_->frontier_ready.wait(lock, [&] {
      return std::ranges::none_of(
          impl_->entries, [](const auto& e) { return e->frontier_active; });
    });
    if (cancelled && cancelled())
      return {};
    if (impl_->catalog)
      impl_->catalog->Reconcile();
    found =
        impl_->index.Lookup({impl_->identity, prompt, input, stable, reuse});
    // A miss reports how far a stored prompt agreed, so an empty cache can
    // be told apart from one holding only incompatible boundaries.
    if (reuse && !found.selected)
      common =
          impl_->index.CommonPrefixTokens({impl_->identity, prompt, input});
    if (found.selected && found.selected->live) {
      const auto location = *found.selected->live;
      auto& selected_entry = *impl_->entries.at(location.slot.value - 1);
      if (!selected_entry.busy) {
        entry = &selected_entry;
        live = location;
      }
    }
    if (!entry) {
      // A frontier far beyond every checkpoint of its prompt is the only
      // copy of that work, as for a conversation larger than the RAM budget
      // whose captures are refused: take it last. Then completed
      // preservation first: it needs no foreground copy. Among those, a live
      // frontier holds generated tokens no checkpoint covers, such as the
      // tool call the next turn replays. Take a slot without one, then a
      // shallow frontier before a deep one, then the least recently used.
      // Faster idle preservation must not make short side requests evict a
      // deep conversation merely because its backing copy finished earlier.
      constexpr std::size_t kUncoveredTokens = Impl::kUncoveredTokens;
      using Rank = Impl::Rank;
      const auto rank = [&](const Impl::Entry& e) {
        return impl_->ColdRank(e);
      };
      std::optional<Rank> selected_rank;
      for (auto& candidate : impl_->entries) {
        if (candidate->busy)
          continue;
        if (!entry) {
          entry = candidate.get();
          continue;
        }
        if (!selected_rank)
          selected_rank = rank(*entry);
        const auto candidate_rank = rank(*candidate);
        if (candidate_rank < *selected_rank) {
          entry = candidate.get();
          selected_rank = candidate_rank;
        }
      }
      // Keep ordinary cold selection unchanged. Only a cancelled executor
      // with outstanding work needs an alternative; do not query every
      // stream or look up exact checkpoint coverage on normal admissions.
      if (entry && !entry->slot->ExecutionComplete()) {
        if (!selected_rank)
          selected_rank = rank(*entry);
        Impl::Entry* ready = nullptr;
        std::optional<Rank> ready_rank;
        for (auto& candidate : impl_->entries) {
          if (candidate.get() == entry || candidate->busy ||
              !candidate->slot->ExecutionComplete())
            continue;
          const auto candidate_rank = rank(*candidate);
          // A ready slot must not sacrifice uncovered work or introduce
          // foreground preservation. Keep a deep generated tail unless an
          // exact checkpoint or another idle live copy still covers it.
          const auto live_copy = [&] {
            return candidate->history &&
                   std::ranges::any_of(impl_->entries, [&](const auto& other) {
                     return other != candidate && !other->busy && other->live &&
                            other->history &&
                            other->history->Lineage() ==
                                candidate->history->Lineage() &&
                            other->live_input == candidate->live_input &&
                            other->live_tokens == candidate->live_tokens;
                   });
          };
          if (std::tie(std::get<0>(candidate_rank),
                       std::get<1>(candidate_rank)) >
                  std::tie(std::get<0>(*selected_rank),
                           std::get<1>(*selected_rank)) ||
              (candidate->live &&
               candidate->live_tokens.size() >= kUncoveredTokens &&
               !impl_->index.HasCheckpoint(
                   {impl_->identity, candidate->live_tokens,
                    cache::InputIdentity(candidate->live_tokens.size(),
                                         candidate->live_input)},
                   candidate->live_tokens.size()) &&
               !live_copy()))
            continue;
          if (!ready || candidate_rank < *ready_rank) {
            ready = candidate.get();
            ready_rank = candidate_rank;
          }
        }
        if (ready)
          entry = ready;
      }
    }
    if (!entry)
      throw cache::SlotBusy();
    entry->busy = true;
    entry->used = ++impl_->clock;
    if (!live) {
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
    }
    if (entry->live) {
      impl_->index.Erase(*entry->live);
      entry->live.reset();
      entry->live_tokens.clear();
    }
    if (live) {
      reply_starts = std::move(entry->reply_starts);
      reply_ends = std::move(entry->reply_ends);
    }
    entry->reply_starts.clear();
    entry->reply_ends.clear();
  }
  std::unique_ptr<Lease> lease;
  try {
    if (entry->frontier_worker.joinable())
      entry->frontier_worker.join();
    auto slot = entry->slot->Acquire(live);
    lease = std::make_unique<Lease>(std::make_unique<Lease::Impl>(
        impl_.get(), entry, std::move(slot), prompt, input, stable));
    auto& request = *lease->impl_;
    request.persist = persist;
    // Claim the request's tokens before a restore or prefill writes them.
    // An older frontier's need falls to them; its pages past them stay
    // charged until reclaimed.
    if (state_tokens != 0)
      if (auto account = request.state->state_account()) {
        request.account = account;
        if (!account->Claim(state_tokens, reservation))
          throw StateMemoryShort();
        (void)account->SetNeed(state_tokens);
      }
    request.state->SetCancellationCheck(cancelled);
    request.state->SetStopAtEos(stop_at_eos);
    impl_->runner->SetPromptContext(*request.state, context);
    const auto started = Clock::now();
    // Cold Acquire has already reset the slot. Only an attempted restore (or
    // an unusable live frontier) can require another reset before cold use.
    bool clean = !live;
    // A frontier captured between prefill passes cannot select the next token
    // of a prompt that ends there, and a runner may refuse other prompts; serve
    // those from a shallower boundary.
    const auto decodable = [&](std::size_t boundary) {
      return (boundary < prompt.size() ||
              impl_->runner->HasDecodeFrontier(*request.state)) &&
             impl_->runner->CanContinue(*request.state, prompt.size());
    };
    // Prefill must not resume inside an indivisible input of this prompt,
    // which a checkpoint from another prompt with the same tokens can do.
    const auto whole_inputs = [&](std::size_t boundary) {
      return !context || context->PrefillStop(boundary, false) == boundary;
    };
    if (live && (!decodable(found.selected->boundary) ||
                 !whole_inputs(found.selected->boundary))) {
      live.reset();
      const std::lock_guard lock(impl_->mutex);
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
    }
    bool restored = bool(live);
    if (live) {
      request.cached = found.selected->boundary;
      request.reply_starts = std::move(reply_starts);
      request.reply_ends = std::move(reply_ends);
    }
    // Prefill continues a checkpoint exactly only from an aligned position.
    // The complete prompt stays off the alignment for an unchanged retry,
    // which continues it without prefill; a longer prompt takes an aligned
    // boundary instead (TextRunnerCapabilities::prefill_alignment).
    const std::size_t alignment = std::max<std::size_t>(
        1, impl_->runner->Descriptor().capabilities.prefill_alignment);
    for (const auto& candidate : found.candidates) {
      if (restored || (cancelled && cancelled()))
        break;
      const bool generated =
          (candidate.checkpoint && candidate.checkpoint->Purpose() ==
                                       cache::CheckpointPurpose::kGenerated) ||
          (candidate.durable && candidate.durable->Manifest().purpose ==
                                    cache::CheckpointPurpose::kGenerated);
      if (candidate.live || !whole_inputs(candidate.boundary) ||
          (!generated && candidate.boundary % alignment != 0 &&
           candidate.boundary < prompt.size()))
        continue;
      try {
        if (candidate.checkpoint) {
          const auto& checkpoint = *candidate.checkpoint;
          std::vector<cache::ComponentPosition> positions;
          for (const auto& c : checkpoint.Components())
            positions.push_back(c.position);
          clean = false;
          // Chunks whose storage the adapter may share take no load: say
          // where they are. Rows still borrowed from a slot are left out;
          // that slot's guard may yet copy them.
          std::vector<cache::RestoreSource> sources;
          std::vector<cache::RowPin> source_pins;
          cache::ResourceReservation source_metadata;
          if (impl_->resources->share &&
              impl_->resources->adapter->GetCapabilities().shared_restore) {
            std::size_t count = 0;
            for (const auto& c : checkpoint.Components())
              if (c.descriptor.kind == cache::ComponentKind::kAppendRows)
                count += c.chunks.size();
            // A checkpoint shorter than one chunk has none: every row is in
            // its tails, and the ledger reserves no empty allocation.
            if (count != 0)
              source_metadata =
                  impl_->ledger.Reserve(cache::ResourceCategory::kMetadata,
                                        count * (sizeof(cache::RowPin) +
                                                 sizeof(cache::RestoreSource)));
            sources.reserve(count);
            source_pins.reserve(count);
            for (const auto& c : checkpoint.Components()) {
              if (c.descriptor.kind != cache::ComponentKind::kAppendRows)
                continue;
              for (const auto& chunk : c.chunks) {
                const auto& storage = chunk.Storage();
                if (storage.BorrowedFrom())
                  continue;
                auto pin = storage.PinRows();
                const auto bytes =
                    (chunk.End() - chunk.First()) * c.descriptor.row_bytes;
                if (pin.Location() || pin.Bytes().size() < bytes)
                  continue;
                sources.push_back({c.descriptor.id, chunk.First(), chunk.End(),
                                   pin.Bytes().first(bytes)});
                source_pins.push_back(std::move(pin));
              }
            }
          }
          // Rows the adapter took in place, ascending per component.
          const auto taken = impl_->resources->adapter->BeginSharedRestore(
              request.slot.Execution(), positions, sources);
          std::size_t taken_bytes = 0;
          for (const auto& rows : taken)
            for (const auto& c : checkpoint.Components())
              if (c.descriptor.id == rows.component)
                taken_bytes += (rows.end - rows.first) * c.descriptor.row_bytes;
          request.remapped += taken_bytes;
          // Every pin outlives all submitted loads, including failure drain.
          // Borrowed views name immutable native rows; materialized views name
          // assigned host backing. No source slot execution lease is needed.
          std::size_t pin_count = 0;
          for (const auto& c : checkpoint.Components())
            pin_count += c.chunks.size() + bool(c.tail) + bool(c.private_state);
          auto pin_metadata =
              impl_->ledger.Reserve(cache::ResourceCategory::kMetadata,
                                    pin_count * sizeof(cache::RowPin));
          std::vector<cache::RowPin> pins;
          pins.reserve(pin_count);
          for (const auto& c : checkpoint.Components()) {
            if (c.private_state)
              pins.push_back(c.private_state->PinRows());
            for (const auto& chunk : c.chunks)
              pins.push_back(chunk.Storage().PinRows());
            if (c.tail)
              pins.push_back(c.tail->PinRows());
          }
          // Restore streams only: this may run beside decode rounds.
          const auto copies = [&] {
            TransferBatch transfers(impl_->resources->transfer_batch_size);
            std::size_t base = 0;
            for (const auto& c : checkpoint.Components()) {
              auto load_piece = [&](const cache::RowPin& pin, cache::Rows base,
                                    cache::Rows first, cache::Rows end) {
                const auto bytes =
                    c.descriptor.kind == cache::ComponentKind::kPrivateState
                        ? c.descriptor.state_bytes
                        : (end - first) * c.descriptor.row_bytes;
                if (pin.Bytes().empty())
                  throw std::logic_error("RAM source has no readable view");
                const std::span<const std::byte> buffer(
                    pin.Bytes().data() +
                        (first - base) * c.descriptor.row_bytes,
                    bytes);
                transfers.Add(
                    [&] {
                      // Restores take their own streams; a busy class falls
                      // back to the shared pool rather than wait.
                      std::unique_ptr<cache::Stream> stream;
                      if (impl_->resources->restore_stream)
                        stream = impl_->resources->restore_stream();
                      return stream ? std::move(stream)
                                    : impl_->resources->stream();
                    },
                    [&](cache::Stream& stream) {
                      return c.descriptor.kind ==
                                     cache::ComponentKind::kPrivateState
                                 ? impl_->resources->adapter->LoadPrivate(
                                       request.slot.Execution(),
                                       c.descriptor.id, buffer, stream)
                                 : impl_->resources->adapter->CopyRowsIn(
                                       request.slot.Execution(),
                                       c.descriptor.id, first, end, buffer,
                                       stream);
                    });
                request.restored += bytes;
              };
              // Loads rows [first, end), whose bytes start at the pin's,
              // except those the adapter already took in place.
              auto load = [&](const cache::RowPin& pin, cache::Rows first,
                              cache::Rows end) {
                auto at = first;
                if (c.descriptor.kind != cache::ComponentKind::kPrivateState)
                  for (const auto& rows : taken) {
                    if (rows.component != c.descriptor.id || rows.end <= at)
                      continue;
                    if (rows.first >= end)
                      break;
                    if (rows.first > at)
                      load_piece(pin, first, at, rows.first);
                    const auto skip = std::min(rows.end, end);
                    request.restored += (skip - std::max(rows.first, at)) *
                                        c.descriptor.row_bytes;
                    at = skip;
                  }
                if (at < end || first == end)
                  load_piece(pin, first, at, end);
              };
              if (c.private_state)
                load(pins[base++], 0, 0);
              auto same_source = [](const cache::RowPin& a,
                                    const cache::RowPin& b) {
                if (a.Location() || b.Location())
                  return a.Location() == b.Location();
                return SameOwner(a.Owner(), b.Owner());
              };
              bool loaded_tail = false;
              for (std::size_t i = 0; i < c.chunks.size();) {
                const auto& chunk = c.chunks[i];
                auto end = chunk.End();
                const auto& pin = pins[base + i];
                const auto* data = pin.Bytes().data();
                auto next = i + 1;
                while (data && next < c.chunks.size() &&
                       c.chunks[next].First() == end &&
                       same_source(pin, pins[base + next]) &&
                       pins[base + next].Bytes().data() ==
                           data +
                               (end - chunk.First()) * c.descriptor.row_bytes) {
                  end = c.chunks[next].End();
                  ++next;
                }
                if (data && next == c.chunks.size() && c.tail &&
                    same_source(pin, pins[base + c.chunks.size()]) &&
                    end == c.position.valid_rows / c.descriptor.rows_per_chunk *
                               c.descriptor.rows_per_chunk &&
                    pins[base + c.chunks.size()].Bytes().data() ==
                        data + (end - chunk.First()) * c.descriptor.row_bytes) {
                  end = c.position.valid_rows;
                  loaded_tail = true;
                }
                load(pin, chunk.First(), end);
                i = next;
              }
              base += c.chunks.size();
              if (c.tail) {
                if (!loaded_tail)
                  load(pins[base],
                       c.position.valid_rows / c.descriptor.rows_per_chunk *
                           c.descriptor.rows_per_chunk,
                       c.position.valid_rows);
                ++base;
              }
            }
            transfers.Finish();
          };
          std::size_t restore_bytes = 0;
          for (const auto& c : checkpoint.Components())
            restore_bytes +=
                c.descriptor.kind == cache::ComponentKind::kPrivateState
                    ? c.descriptor.state_bytes
                    : c.position.valid_rows * c.descriptor.row_bytes;
          restore_bytes -= std::min(restore_bytes, taken_bytes);
          if (run_copies)
            run_copies(RestoreEstimate(restore_bytes, false), copies);
          else
            copies();
          if (!impl_->resources->adapter->Validate(request.slot.Execution(),
                                                   positions))
            throw std::runtime_error("component RAM restore validation failed");
          entry->history = std::make_unique<cache::ExecutionHistory>(
              cache::ExecutionHistory::Restored(impl_->ledger, checkpoint));
          request.source = checkpoint.Id();
          {
            const std::lock_guard lock(impl_->mutex);
            impl_->retention.Touch(checkpoint.Id());
          }
          restored = true;
        } else if (candidate.durable && impl_->streaming) {
          clean = false;
          cache::TransferTiming timing;
          const auto id = candidate.durable->Manifest().checkpoint;
          // The restored history shares no RAM chunks, so its next capture
          // needs backing for the whole checkpoint: commit it off this
          // thread while the restore reads, along with the backing of
          // restores not captured yet.
          request.SettleCapture();  // An earlier candidate that failed.
          request.pending_capture = PayloadBytes(candidate.durable->Manifest());
          const auto pending =
              impl_->pending_capture_bytes.fetch_add(request.pending_capture) +
              request.pending_capture;
          if (impl_->resources->prepare_capture)
            impl_->resources->prepare_capture(pending);
          restored = impl_->streaming->Restore(
              id, impl_->digest, *impl_->resources->adapter,
              request.slot.Execution(), timing, {}, {},
              [this](bool private_file, cache::DiskFileId file) {
                const std::lock_guard lock(impl_->mutex);
                impl_->catalog->Invalidate(private_file, file);
              },
              run_copies
                  ? cache::RestoreCopyRunner(
                        [&, estimate = RestoreEstimate(
                                PayloadBytes(candidate.durable->Manifest()),
                                true)](const auto& copies) {
                          // Beside decoders the restore can outrun the
                          // backing its capture needs: wait for it here,
                          // not in the capture on the serving thread.
                          run_copies(estimate, [&] {
                            copies();
                            if (cache::CopyingBesideDecode() &&
                                impl_->resources->await_capture)
                              impl_->resources->await_capture();
                          });
                        })
                  : cache::RestoreCopyRunner{});
          if (restored) {
            request.disk_hit = true;
            request.restored = timing.bytes;
            Logger::Info(
                "cache",
                "schema=component-cache-v1 event=disk_restore tokens=" +
                    std::to_string(candidate.boundary) +
                    " bytes=" + std::to_string(timing.bytes) + " pieces=" +
                    std::to_string(timing.pieces) + " stream_acquire_ms=" +
                    std::to_string(timing.stream_acquire_ns / 1e6) +
                    " stream_wait_ms=" +
                    std::to_string(timing.stream_wait_ns / 1e6) +
                    " filesystem_ms=" +
                    std::to_string(timing.filesystem_ns / 1e6) +
                    " checksum_ms=" + std::to_string(timing.checksum_ns / 1e6) +
                    " staging_wait_ms=" +
                    std::to_string(timing.staging_wait_ns / 1e6));
            entry->history = std::make_unique<cache::ExecutionHistory>(
                cache::ExecutionHistory::RestoredUnshared(
                    impl_->ledger, impl_->resources->adapter->Components(),
                    impl_->identity));
            const std::lock_guard lock(impl_->mutex);
            impl_->catalog->Touch(id);
          }
        }
        if (restored && !decodable(candidate.boundary)) {
          if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
            throw std::runtime_error("component restore reset failed");
          entry->history.reset();
          request.SettleCapture();
          request.source = {};
          request.restored = 0;
          request.remapped = 0;
          request.disk_hit = false;
          restored = false;
          continue;
        }
        if (restored) {
          request.cached = candidate.boundary;
          const std::lock_guard lock(impl_->mutex);
          entry->durable = candidate.durable;
        }
      } catch (...) {
        request.SettleCapture();
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        clean = true;
        request.restored = 0;
        request.remapped = 0;
      }
    }
    if (!restored) {
      request.SettleCapture();
      if (!clean &&
          !impl_->resources->adapter->Invalidate(request.slot.Execution()))
        throw std::runtime_error("component cold initialization failed");
      entry->history = std::make_unique<cache::ExecutionHistory>(
          cache::ExecutionHistory::Cold(impl_->ledger,
                                        impl_->resources->adapter->Components(),
                                        impl_->identity));
    }
    // Restore/reset may clear the attachment; install this request again.
    impl_->runner->SetPromptContext(*request.state, context);
    impl_->runner->ReconcileComponentState(*request.state, request.cached);
    request.restore_time =
        request.restored
            ? std::chrono::duration<double, std::milli>(Clock::now() - started)
                  .count()
            : 0;
    request.metrics.reuse = live               ? "live"
                            : request.disk_hit ? "disk"
                            : request.cached   ? "ram"
                                               : "none";
    request.metrics.selection_reason = static_cast<std::uint32_t>(found.reason);
    request.metrics.selected = request.cached;
    request.metrics.restored_bytes = request.restored;
    request.metrics.remapped_bytes = request.remapped;
    for (const auto& candidate : found.candidates) {
      auto& deepest = candidate.live         ? request.metrics.deepest_live
                      : candidate.checkpoint ? request.metrics.deepest_ram
                                             : request.metrics.deepest_disk;
      deepest = std::max(deepest, static_cast<std::size_t>(candidate.boundary));
    }
    std::size_t nearest = 0;
    for (const auto& candidate : found.candidates)
      nearest = std::max(nearest, static_cast<std::size_t>(candidate.boundary));
    request.lookup_result = {request.cached      ? ""
                             : !reuse            ? "disabled"
                             : common || nearest ? "no_compatible_boundary"
                                                 : "no_checkpoint",
                             request.cached ? 0 : common,
                             request.cached ? request.cached : nearest};
    Logger::Info("cache",
                 "schema=component-cache-v1 event=acquire reuse=" +
                     std::string(live               ? "live"
                                 : request.disk_hit ? "disk"
                                 : request.cached   ? "ram"
                                                    : "none") +
                     " selected_tokens=" + std::to_string(request.cached) +
                     " restored_bytes=" + std::to_string(request.restored) +
                     " remapped_bytes=" + std::to_string(request.remapped));
    if (const auto estimate = impl_->resources->adapter->Cost())
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=cost_estimate from=" +
              std::to_string(request.cached) +
              " to=" + std::to_string(prompt.size()) + " prefill_ms=" +
              std::to_string(1000 * estimate->PrefillSeconds(request.cached,
                                                             prompt.size())) +
              " restore_ms=" +
              std::to_string(1000 * estimate->RestoreSeconds(
                                        request.restored, request.disk_hit)));
    return lease;
  } catch (...) {
    if (lease)
      lease->Invalidate();
    else {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
    }
    throw;
  }
}

std::optional<TextPendingPublication> ComponentTextCache::PendingPublication(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    std::uint64_t checkpoint_id) const {
  if (!impl_->queue)
    return std::nullopt;
  const auto input = Inputs(prompt.size(), context);
  const std::size_t alignment = std::max<std::size_t>(
      1, impl_->runner->Descriptor().capabilities.prefill_alignment);
  const std::lock_guard lock(impl_->mutex);
  const auto found =
      impl_->index.Lookup({impl_->identity, prompt, input, 0, true});
  const std::size_t best = found.selected ? found.selected->boundary : 0;
  std::optional<TextPendingPublication> selected;
  for (const auto& capture : impl_->pending_durable) {
    if (!capture)
      continue;
    const auto id = capture->manifest.checkpoint.value;
    if (checkpoint_id && id != checkpoint_id)
      continue;
    const auto& tokens = capture->manifest.tokens;
    // Acquire skips an unaligned checkpoint below the prompt end.
    if (tokens.size() > prompt.size() ||
        (capture->manifest.purpose != cache::CheckpointPurpose::kGenerated &&
         tokens.size() % alignment != 0 && tokens.size() < prompt.size()) ||
        capture->manifest.input != At(input, tokens.size()) ||
        !std::equal(tokens.begin(), tokens.end(), prompt.begin()))
      continue;
    TextPendingPublication candidate{id, best, tokens.size()};
    if (!checkpoint_id &&
        candidate.Gain() <= TextPendingPublication::kMinimumGainTokens)
      continue;
    if (!selected || candidate.pending_tokens > selected->pending_tokens ||
        (candidate.pending_tokens == selected->pending_tokens &&
         id < selected->checkpoint_id))
      selected = candidate;
  }
  return selected;
}

std::shared_ptr<const std::atomic<bool>> ComponentTextCache::WarmDisk(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    std::size_t stable, std::chrono::microseconds beside_after) {
  if (!impl_->disk || !impl_->streaming)
    return nullptr;
  const auto input = Inputs(prompt.size(), context);
  cache::CheckpointId id{};
  std::size_t boundary{};
  std::uint64_t bytes{};
  {
    const std::lock_guard lock(impl_->mutex);
    const auto found =
        impl_->index.Lookup({impl_->identity, prompt, input, stable, true});
    const auto& selected = found.selected;
    if (!selected || selected->live || selected->checkpoint ||
        !selected->durable)
      return nullptr;
    const auto& manifest = selected->durable->Manifest();
    id = manifest.checkpoint;
    boundary = selected->boundary;
    bytes = PayloadBytes(manifest);
  }
  if (RestoreEstimate(bytes, true) >= beside_after) {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=disk_warm skipped=beside "
                 "tokens=" +
                     std::to_string(boundary) +
                     " bytes=" + std::to_string(bytes));
    return nullptr;
  }
  const std::lock_guard lock(impl_->warm_mutex);
  std::erase_if(impl_->warmups, [](const auto& warmup) {
    return warmup->done->load(std::memory_order_acquire);
  });
  std::uint64_t reading{};
  for (const auto& warmup : impl_->warmups) {
    if (warmup->id == id)
      return warmup->done;
    reading += warmup->bytes;
  }
  // Page cache read into the free pages the host keeps would make the kernel
  // reclaim and compact beside running work (RFC §4.4).
  const auto free = mem::HostFreeBytes();
  const auto floor = mem::HostFreeFloorBytes();
  if (free < floor || free - floor < reading + bytes) {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=disk_warm "
                 "skipped=floor tokens=" +
                     std::to_string(boundary) +
                     " bytes=" + std::to_string(bytes) +
                     " free_bytes=" + std::to_string(free) +
                     " floor_bytes=" + std::to_string(floor));
    return nullptr;
  }
  // The restore after this read is captured in full: commit its backing
  // while the request waits, not in its capture beside running decodes.
  // Restores still reading, and restored ones not captured yet, are captured
  // too, so ask for all of them.
  if (impl_->resources->prepare_capture)
    impl_->resources->prepare_capture(impl_->pending_capture_bytes.load() +
                                      reading + bytes);
  auto warmup = std::make_unique<Impl::Warmup>();
  warmup->id = id;
  warmup->bytes = bytes;
  warmup->done = std::make_shared<std::atomic<bool>>(false);
  warmup->thread = std::jthread([disk = impl_->disk.get(), id, boundary,
                                 done = warmup->done](std::stop_token stop) {
    const auto start = Clock::now();
    std::atomic<std::uint64_t> read{};
    try {
      if (const auto snapshot = disk->Open(id)) {
        std::vector<std::pair<bool, const cache::DiskPayload*>> files;
        for (const auto& c : snapshot->Manifest().components) {
          for (const auto& chunk : c.chunks)
            files.emplace_back(false, &chunk);
          if (c.tail)
            files.emplace_back(true, &*c.tail);
          if (c.private_state)
            files.emplace_back(true, &*c.private_state);
        }
        std::atomic<std::size_t> next{};
        std::vector<std::jthread> readers;
        for (std::size_t i = 0;
             i < std::min(cache::kDiskReadsInFlight, files.size()); ++i)
          readers.emplace_back([&] {
            std::vector<std::uint8_t> buffer(cache::kDiskStagingPieceBytes);
            for (std::size_t f; !stop.stop_requested() &&
                                (f = next.fetch_add(1)) < files.size();) {
              try {
                read += disk->WarmPayload(*snapshot, files[f].first,
                                          *files[f].second, buffer, stop);
              } catch (...) {
                // A hint only; the restore validates every file it reads.
              }
            }
          });
      }
    } catch (...) {
    }
    Logger::Info("cache",
                 "schema=component-cache-v1 event=disk_warm tokens=" +
                     std::to_string(boundary) +
                     " bytes=" + std::to_string(read.load()) + " ms=" +
                     std::to_string(std::chrono::duration<double, std::milli>(
                                        Clock::now() - start)
                                        .count()));
    done->store(true, std::memory_order_release);
  });
  auto done = warmup->done;
  impl_->warmups.push_back(std::move(warmup));
  return done;
}

std::size_t ComponentTextCache::CachedPrefixTokens(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool common) const {
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex);
  const cache::PrefixQuery query{impl_->identity, prompt, input};
  return common ? impl_->index.CommonPrefixTokens(query)
                : impl_->index.CachedPrefixTokens(query);
}

namespace {
// Unicode White_Space and the separators Python's str.strip also removes:
// what chat templates trim from the ends of message content.
bool TrimmedWhitespace(std::string_view text) {
  if (text.empty())
    return false;
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    char32_t cp = lead;
    std::size_t length = 1;
    if (lead >= 0x80) {
      // No whitespace needs four bytes.
      if ((lead & 0xE0) == 0xC0) {
        cp = lead & 0x1F;
        length = 2;
      } else if ((lead & 0xF0) == 0xE0) {
        cp = lead & 0x0F;
        length = 3;
      } else {
        return false;
      }
      if (i + length > text.size())
        return false;
      for (std::size_t k = 1; k < length; ++k) {
        const auto next = static_cast<unsigned char>(text[i + k]);
        if ((next & 0xC0) != 0x80)
          return false;
        cp = (cp << 6) | (next & 0x3F);
      }
    }
    if (!((cp >= 0x09 && cp <= 0x0D) || cp == 0x20 ||
          (cp >= 0x1C && cp <= 0x1F) || cp == 0x85 || cp == 0xA0 ||
          cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
          cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000))
      return false;
    i += length;
  }
  return true;
}

struct FrontierEnd {
  // Prompt tokens the frontier's text covers.
  std::size_t prompt{};
  // Bytes of the last of them past the frontier's end, when it ends inside
  // one, as a reply cut off by the token limit can.
  std::string partial;
  // Frontier tokens whose text the prompt covers: all of them, or fewer when
  // the whole prompt is byte-identical text ending at a frontier token
  // boundary (an unchanged retry of a prompt that was itself aligned).
  std::size_t frontier{};
};

// Where a chat template may have trimmed whitespace from a reply in
// `frontier`: at the start of a reply, or at its end or before its final
// (end-of-turn) token. Without a template nothing was trimmed.
struct TrimEdges {
  std::span<const std::size_t> starts, ends;
};

// Where the text of all of `frontier` ends in `prompt`, or nothing when the
// texts differ first. A prompt whose whole text exactly equals a longer prefix
// of the frontier, ending at one of its token boundaries, covers that prefix.
// Only tokens past an ID mismatch are decoded; equal IDs at a shared boundary
// are skipped as they are. The template trims whitespace
// around message content, so frontier whitespace the prompt lacks matches
// only at the edges of a reply, where the frontier ends or both continue with
// the same token. Whitespace a client removed elsewhere is an edit.
std::optional<FrontierEnd> FrontierTextEnd(
    std::span<const ContinuationToken> frontier,
    std::span<const ContinuationToken> prompt, std::size_t common,
    const TextModelRunner& runner, TrimEdges edges) {
  const auto piece = [&](ContinuationToken token) {
    return runner.Decode(std::span<const ContinuationToken>(&token, 1));
  };
  std::size_t f = common, p = common;
  // Bytes one side has consumed past the other's last token boundary.
  std::string ahead;
  bool frontier_ahead = false;
  // Whitespace was just dropped at a reply edge. A prompt ending there would
  // gain whitespace it never had, unless later text follows the edge.
  bool trimmed = false;
  for (;;) {
    if (ahead.empty()) {
      if (f == frontier.size())
        return FrontierEnd{p, {}, f};
      if (p == prompt.size()) {
        if (trimmed || f == common)
          return std::nullopt;
        return FrontierEnd{p, {}, f};
      }
      if (frontier[f] == prompt[p]) {
        ++f;
        ++p;
        trimmed = false;
        continue;
      }
      trimmed = false;
      ahead = piece(frontier[f++]);
      frontier_ahead = true;
      if (ahead.empty())
        return std::nullopt;
    }
    const auto lagging = frontier_ahead ? prompt : frontier;
    auto& at = frontier_ahead ? p : f;
    if (at == lagging.size()) {
      if (frontier_ahead)
        return std::nullopt;
      return FrontierEnd{p, std::move(ahead), f};
    }
    const auto next = piece(lagging[at++]);
    const auto n = std::min(ahead.size(), next.size());
    if (next.empty() || ahead.compare(0, n, next, 0, n) != 0) {
      --at;
      if (!frontier_ahead || !TrimmedWhitespace(ahead))
        return std::nullopt;
      // `ahead` is the rest of frontier token f - 1; all of it when the
      // whitespace starts at that token.
      const auto first = f - 1;
      const bool whole = ahead.size() == piece(frontier[first]).size();
      while (f < frontier.size() && frontier[f] != prompt[p] &&
             TrimmedWhitespace(piece(frontier[f])))
        ++f;
      if (f < frontier.size() && frontier[f] != prompt[p])
        return std::nullopt;
      if (std::ranges::find(edges.ends, f) == edges.ends.end() &&
          (!whole ||
           std::ranges::find(edges.starts, first) == edges.starts.end()))
        return std::nullopt;
      trimmed = true;
      ahead.clear();
      continue;
    }
    if (next.size() <= ahead.size()) {
      ahead.erase(0, n);
    } else {
      ahead = next.substr(n);
      frontier_ahead = !frontier_ahead;
    }
  }
}
}  // namespace

bool ComponentTextCache::AlignToLiveFrontier(
    std::vector<ContinuationToken>& prompt, std::size_t& stable,
    std::size_t max_tokens, bool trimmed) const {
  const std::lock_guard lock(impl_->mutex);
  std::span<const ContinuationToken> selected;
  std::size_t seam = 0, exact_depth = 0, aligned_stable = 0;
  // The prompt's own tokens already begin a frontier, as an unchanged retry
  // does. Rewriting it to an older frontier's tokens would only change what
  // the identical request samples.
  bool covered = false;
  // Tokens for the rest of a prompt token the frontier ends inside.
  std::vector<ContinuationToken> rest;
  const auto consider = [&](std::span<const ContinuationToken> frontier,
                            TrimEdges edges) {
    const auto common = static_cast<std::size_t>(
        std::ranges::mismatch(frontier, prompt).in1 - frontier.begin());
    if (common == prompt.size()) {
      covered = true;
      return;
    }
    if (common == frontier.size()) {
      exact_depth = std::max(exact_depth, frontier.size());
      return;
    }
    if (common == 0 ||
        (!selected.empty() && frontier.size() <= selected.size()) ||
        frontier.size() <= exact_depth)
      return;
    auto end = FrontierTextEnd(frontier, prompt, common, *impl_->runner, edges);
    if (!end)
      return;
    std::size_t mapped = 0;
    if (end->frontier < frontier.size()) {
      // A prompt ending inside the frontier gains only from a checkpoint
      // exactly at that end: an unchanged retry of an aligned prompt, whose
      // fresh tokenization would otherwise diverge from the retained one.
      frontier = frontier.first(end->frontier);
      const cache::PrefixQuery query{impl_->identity, frontier,
                                     Inputs(frontier.size(), nullptr)};
      if (impl_->index.CachedPrefixTokens(query) != frontier.size())
        return;
      // The whole prompt is rewritten, so its stable boundary moves to the
      // frontier token boundary ending the same text, if there is one.
      mapped = stable;
      if (stable > common) {
        const auto at =
            FrontierTextEnd(frontier, std::span(prompt).first(stable), common,
                            *impl_->runner, edges);
        if (!at || !at->partial.empty() || at->prompt != stable)
          return;
        mapped = at->frontier;
      }
      if ((!selected.empty() && frontier.size() <= selected.size()) ||
          frontier.size() <= exact_depth)
        return;
    } else if (stable != 0 && stable < end->prompt) {
      return;
    } else if (stable != 0) {
      mapped = stable - end->prompt + frontier.size();
    }
    std::vector<ContinuationToken> tail;
    if (!end->partial.empty()) {
      // A fragment that does not tokenize back to its own bytes (a split
      // UTF-8 sequence, say) cannot follow the frontier.
      try {
        tail = impl_->runner->Tokenize(end->partial);
      } catch (const std::exception&) {
        return;
      }
      if (tail.empty() || impl_->runner->Decode(tail) != end->partial)
        return;
    }
    if (frontier.size() + tail.size() + (prompt.size() - end->prompt) >=
        max_tokens)
      return;
    selected = frontier;
    seam = end->prompt;
    aligned_stable = stable != 0 ? mapped + tail.size() : 0;
    rest = std::move(tail);
  };
  for (const auto& entry : impl_->entries) {
    const auto& frontier = entry->live_tokens;
    // Positions of image inputs would move with the rewrite.
    if (!entry->live || entry->busy || frontier.empty() ||
        !entry->live_input.empty())
      continue;
    consider(frontier, trimmed
                           ? TrimEdges{entry->reply_starts, entry->reply_ends}
                           : TrimEdges{});
  }
  impl_->index.VisitGenerated(impl_->identity, [&](auto frontier) {
    // Retained checkpoints have no template whitespace-edge metadata. Only
    // byte-identical text may be rewritten after a cold assignment/restart.
    consider(frontier, {});
  });
  if (covered || selected.empty() || exact_depth >= selected.size())
    return false;
  if (Logger::Enabled(LogLevel::kDebug))
    Logger::Debug("cache",
                  "event=frontier_aligned frontier_tokens=" +
                      std::to_string(selected.size()) +
                      " prompt_seam=" + std::to_string(seam) +
                      " split_tokens=" + std::to_string(rest.size()) +
                      " prompt_tokens=" + std::to_string(prompt.size()));
  stable = aligned_stable;
  std::vector<ContinuationToken> aligned;
  aligned.reserve(selected.size() + rest.size() + prompt.size() - seam);
  aligned.assign(selected.begin(), selected.end());
  aligned.insert(aligned.end(), rest.begin(), rest.end());
  aligned.insert(aligned.end(), prompt.begin() + seam, prompt.end());
  prompt = std::move(aligned);
  return true;
}
ComponentTextCache::Lease::Lease(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ComponentTextCache::Lease::~Lease() {
  Invalidate();
}
ContinuationState& ComponentTextCache::Lease::state() const {
  return *impl_->state;
}
bool ComponentTextCache::Lease::cache_hit() const noexcept {
  return impl_->cached != 0;
}
std::size_t ComponentTextCache::Lease::cached_tokens() const noexcept {
  return impl_->cached;
}
std::size_t ComponentTextCache::Lease::restored_snapshot_bytes()
    const noexcept {
  return impl_->restored;
}
double ComponentTextCache::Lease::restore_ms() const noexcept {
  return impl_->restore_time;
}
bool ComponentTextCache::Lease::restored_from_disk() const noexcept {
  return impl_->disk_hit;
}
ContinuationLookup ComponentTextCache::Lease::lookup() const noexcept {
  return impl_->lookup_result;
}
bool ComponentTextCache::Lease::HasSnapshotFor(
    std::span<const ContinuationToken> tokens) const {
  const std::lock_guard lock(impl_->cache->mutex);
  return impl_->cache->index.HasCheckpoint(
      {impl_->cache->identity, tokens,
       cache::InputIdentity(tokens.size(), At(impl_->input, tokens.size()))},
      tokens.size());
}
bool ComponentTextCache::Lease::TryReserveSnapshot(
    std::size_t bytes, std::size_t boundary, bool preserve,
    SnapshotPurpose purpose, std::span<const ContinuationToken> tokens) {
  // Grids stay in RAM, except the one this lease persists.
  const bool ram_only =
      purpose == SnapshotPurpose::kHistory && !impl_->DiskGrid(boundary);
  if (!impl_->cache->ram_enabled && (!impl_->cache->queue || ram_only))
    return false;
  if (!boundary || boundary > impl_->prompt.size() ||
      (bytes > impl_->cache->budget && (!impl_->cache->queue || ram_only)))
    return false;
  if (tokens.empty())
    tokens = std::span<const cache::Token>(impl_->prompt).first(boundary);
  // A capture within the disk step has nowhere to go when the RAM cache is
  // off. Refuse it here so callers do not log it as a failed capture.
  if (!impl_->cache->ram_enabled && purpose != SnapshotPurpose::kBranchPoint &&
      impl_->cache->WithinDiskStep(tokens, At(impl_->input, tokens.size())))
    return false;
  impl_->capture_boundary = boundary;
  impl_->capture_tokens.assign(tokens.begin(), tokens.end());
  impl_->preserve_source = preserve;
  impl_->purpose = purpose;
  return true;
}
void ComponentTextCache::Lease::SkipSnapshot(SnapshotEventReason reason,
                                             std::size_t bytes,
                                             std::size_t boundary) noexcept {
  try {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=snapshot_skipped reason=" +
                     std::to_string(static_cast<unsigned>(reason)) +
                     " tokens=" + std::to_string(boundary) +
                     " bytes=" + std::to_string(bytes));
  } catch (...) {
  }
}
void ComponentTextCache::Lease::PreparePrefill(std::size_t tokens) {
  // Between this thread's launches: nothing of the model's is in flight.
  if (impl_->cache->resources->settle)
    impl_->cache->resources->settle();
  if (impl_->cache->resources->prepare_prefill)
    impl_->cache->resources->prepare_prefill(tokens);
}

std::size_t ComponentTextCache::Lease::CaptureBytes() const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution()));
}
std::optional<std::size_t> ComponentTextCache::Lease::CaptureBytes(
    std::size_t boundary) const {
  try {
    return impl_->entry->history->NewPayloadBytes(
        impl_->cache->resources->adapter->PrefillCheckpointPositions(
            impl_->slot.Execution(), boundary));
  } catch (const std::invalid_argument& error) {
    // An in-pass checkpoint is an optimisation; the adapter refusing one
    // here must not fail the request.
    Logger::Info("cache",
                 "schema=component-cache-v1 event=in_pass_unsupported tokens=" +
                     std::to_string(boundary) + " cached_tokens=" +
                     std::to_string(impl_->cached) + " reason=" + error.what());
    return std::nullopt;
  }
}
std::unique_ptr<ContinuationSnapshot> ComponentTextCache::Lease::Capture() {
  auto& request = *impl_;
  auto& cache = *request.cache;
  // After this capture, a restored history shares its chunks like any other.
  struct Settle {
    Impl& request;
    ~Settle() { request.SettleCapture(); }
  } settle{request};
  auto positions =
      cache.resources->adapter->Positions(request.slot.Execution());
  const auto bytes = request.entry->history->NewPayloadBytes(positions);
  const auto retain = request.Retain(request.capture_tokens, bytes);
  const auto refused = [&] {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=capture_refused "
                 "reason=byte_capacity tokens=" +
                     std::to_string(request.capture_tokens.size()));
  };
  const auto failed = [](const std::exception& error) {
    Logger::Info(
        "cache",
        std::string("schema=component-cache-v1 event=capture_failed reason=") +
            error.what());
  };
  if (!cache.ram_enabled || bytes > cache.budget) {
    try {
      return request.CaptureDurable();
    } catch (const cache::ResourceExhausted&) {
      refused();
      throw;
    } catch (const std::exception& error) {
      failed(error);
      throw;
    }
  }
  const auto reclaim = [&] {
    cache.DiscardUnusedFrontierWrites();
    bool reclaimed;
    {
      const std::lock_guard lock(cache.mutex);
      reclaimed = cache.retention.Reclaim(retain);
    }
    cache.DrainDemotion();
    return reclaimed;
  };
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::size_t captured_capacity = 0;
  std::size_t borrowed_capacity = 0, copied_payload = 0, shared_capacity = 0;
  while (true) {
    try {
      captured_capacity = 0;
      borrowed_capacity = copied_payload = shared_capacity = 0;
      cache::ResourceReservation buffer_metadata;
      std::vector<ComponentCaptureBuffer> buffers;
      std::vector<std::span<const std::byte>> borrowed;
      // Payloads that share their rows' storage; they need no buffer.
      std::vector<std::optional<cache::Payload>> shared;
      std::size_t next_buffer = 0;
      checkpoint = request.entry->history->Capture(
          {request.capture_tokens,
           request.input.Through(request.capture_tokens.size()), positions,
           request.generated_frontier ? cache::CheckpointPurpose::kGenerated
           : request.purpose == SnapshotPurpose::kHistory
               ? cache::CheckpointPurpose::kGrid
           : request.purpose == SnapshotPurpose::kBranchPoint
               ? cache::CheckpointPurpose::kLearned
               : cache::CheckpointPurpose::kPrompt,
           0},
          [&](const cache::PayloadRequest&) {
            const auto i = next_buffer++;
            auto& buffer = buffers.at(i);
            const bool taken = bool(shared[i]);
            auto storage = taken ? std::move(*shared[i])
                           : borrowed[i].empty()
                               ? buffer.finish()
                               : buffer.borrow(request.slot, borrowed[i]);
            captured_capacity += storage.Bytes();
            if (taken || !borrowed[i].empty())
              borrowed_capacity += storage.Bytes();
            if (taken)
              shared_capacity += storage.Bytes();
            return storage;
          },
          [&](std::span<const cache::PayloadRequest> payloads) {
            buffer_metadata = cache.ledger.Reserve(
                cache::ResourceCategory::kMetadata,
                payloads.size() *
                    (sizeof(ComponentCaptureBuffer) +
                     sizeof(std::span<const std::byte>) + sizeof(std::size_t) +
                     sizeof(std::optional<cache::Payload>)));
            buffers.reserve(payloads.size());
            borrowed.reserve(payloads.size());
            shared.reserve(payloads.size());
            // Small row/tail allocations must not split the remaining runs
            // that larger private states require. Keep equal-size requests
            // in their original order; transfers still follow component order.
            std::vector<std::size_t> allocation_order(payloads.size());
            std::iota(allocation_order.begin(), allocation_order.end(), 0);
            std::ranges::sort(allocation_order, [&](auto a, auto b) {
              return payloads[a].bytes != payloads[b].bytes
                         ? payloads[a].bytes > payloads[b].bytes
                         : a < b;
            });
            const bool share_rows =
                cache.resources->share &&
                cache.resources->adapter->GetCapabilities().shared_restore;
            while (true) {
              try {
                if (request.generated_frontier &&
                    cache.resources->can_capture) {
                  auto metadata = cache.ledger.Reserve(
                      cache::ResourceCategory::kMetadata,
                      2 * payloads.size() * sizeof(cache::PayloadRequest));
                  std::vector<cache::PayloadRequest> with_headroom(
                      payloads.begin(), payloads.end());
                  with_headroom.insert(with_headroom.end(), payloads.begin(),
                                       payloads.end());
                  if (!cache.resources->can_capture(with_headroom)) {
                    if (reclaim())
                      continue;
                    throw FrontierHeadroomRefused();
                  }
                }
                if (cache.resources->can_capture &&
                    !cache.resources->can_capture(payloads))
                  throw cache::ResourceExhausted();
                buffers.resize(payloads.size());
                borrowed.resize(payloads.size());
                shared.resize(payloads.size());
                for (const auto i : allocation_order) {
                  const auto& payload = payloads[i];
                  const bool rows = payload.category ==
                                    cache::ResourceCategory::kBackingAssigned;
                  // Rows whose pages the cache can share cost no backing
                  // and no copy, now or when the slot moves on.
                  // Only for an adapter that keeps its writes off shared
                  // pages and restores them (Capabilities::shared_restore).
                  std::optional<std::span<const std::byte>> view;
                  if (rows && share_rows) {
                    view = cache.resources->adapter->BorrowRowsView(
                        request.slot.Execution(), payload.component,
                        payload.first, payload.end);
                    if (!view->empty() &&
                        (shared[i] = cache.resources->share(payload, *view)))
                      continue;
                  }
                  buffers[i] = cache.resources->allocate(payload);
                  borrowed[i] =
                      buffers[i].borrow && rows
                          ? view ? *view
                                 : cache.resources->adapter->BorrowRowsView(
                                       request.slot.Execution(),
                                       payload.component, payload.first,
                                       payload.end)
                          : std::span<const std::byte>{};
                }
                break;
              } catch (const cache::ResourceExhausted&) {
                // No transfer has started. Release partial allocations before
                // eviction, but keep the plan's shared chunks alive. Rebuilding
                // the whole checkpoint can otherwise lose those predecessors
                // and allocate their rows again after their last owner leaves.
                borrowed.clear();
                buffers.clear();
                shared.clear();
                if (!reclaim())
                  throw;
              }
            }
            TransferBatch transfers(cache.resources->transfer_batch_size);
            for (std::size_t i = 0; i < payloads.size();) {
              if (shared[i] || !borrowed[i].empty()) {
                ++i;
                continue;
              }
              const auto& payload = payloads[i];
              auto end = payload.end;
              auto bytes = payload.bytes;
              auto next = i + 1;
              if (payload.category != cache::ResourceCategory::kPrivateState)
                while (buffers[i].allocation && next < payloads.size() &&
                       !shared[next] && borrowed[next].empty() &&
                       buffers[next].allocation == buffers[i].allocation &&
                       payloads[next].component == payload.component &&
                       payloads[next].category !=
                           cache::ResourceCategory::kPrivateState &&
                       payloads[next].first == end &&
                       buffers[next].bytes.data() ==
                           buffers[i].bytes.data() + bytes) {
                  end = payloads[next].end;
                  bytes += payloads[next].bytes;
                  ++next;
                }
              const std::span<std::byte> buffer(buffers[i].bytes.data(), bytes);
              transfers.Add(
                  [&] { return cache.resources->stream(); },
                  [&](cache::Stream& stream) {
                    return payload.category ==
                                   cache::ResourceCategory::kPrivateState
                               ? cache.resources->adapter->CapturePrivate(
                                     request.slot.Execution(),
                                     payload.component, buffer, stream)
                               : cache.resources->adapter->CopyRowsOut(
                                     request.slot.Execution(),
                                     payload.component, payload.first, end,
                                     buffer, stream);
                  });
              copied_payload += bytes;
              i = next;
            }
            transfers.Finish();
          });
      break;
    } catch (const FrontierHeadroomRefused&) {
      // Do not bypass the foreground reserve through optional disk staging.
      // The live frontier remains usable until its eventual reassignment.
      Logger::Info("cache", "event=frontier_capture_refused reason=headroom");
      return {};
    } catch (const cache::ResourceExhausted&) {
      if (!reclaim()) {
        if (cache.queue && (request.purpose != SnapshotPurpose::kHistory ||
                            request.DiskGrid(request.capture_tokens.size()))) {
          try {
            if (auto durable = request.CaptureDurable())
              return durable;
          } catch (const cache::ResourceExhausted&) {
            // Staging for disk did not fit either. Refuse below.
          } catch (const std::exception& error) {
            failed(error);
            throw;
          }
        }
        // RAM refused the copy; the disk fallback, when there is one, already
        // holds a checkpoint within its step of this boundary.
        refused();
        throw;
      }
    } catch (const std::exception& error) {
      failed(error);
      throw;
    }
  }
  ++request.metrics.captures;
  request.metrics.captured_checkpoint_ids.push_back(checkpoint->Id().value);
  request.metrics.captured_bytes += captured_capacity;
  request.metrics.borrowed_capacity_bytes += borrowed_capacity;
  request.metrics.copied_payload_bytes += copied_payload;
  request.metrics.shared_payload_bytes += shared_capacity;
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->checkpoint = std::move(checkpoint);
  snapshot->bytes = bytes;
  return snapshot;
}
std::size_t ComponentTextCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot, bool preserve) {
  const auto* component =
      dynamic_cast<const ComponentSnapshot*>(snapshot.get());
  if (component && component->durable) {
    const auto started = Clock::now();
    impl_->metrics.disk_queued_bytes +=
        impl_->cache->Persist(component->durable);
    impl_->metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return 0;
  }
  if (!component || !component->checkpoint)
    return 0;
  auto retain = impl_->Retain(tokens);
  retain.preserve_source = preserve;
  bool admitted;
  {
    const std::lock_guard lock(impl_->cache->mutex);
    admitted = impl_->cache->retention.Admit(
        retain, [&] { return component->checkpoint; });
  }
  impl_->cache->DrainDemotion();
  if (!admitted)
    return 0;
  if (retain.purpose == cache::RetentionPurpose::kHistory) {
    if (impl_->edit_boundary_count == impl_->edit_boundaries.size())
      throw std::logic_error("too many edit boundaries in one turn");
    impl_->edit_boundaries[impl_->edit_boundary_count++] =
        component->checkpoint->Id();
  }
  if (!impl_->persist)
    return component->bytes;
  try {
    const auto started = Clock::now();
    impl_->metrics.disk_queued_bytes += impl_->cache->Persist(
        component->checkpoint, false,
        impl_->DiskGrid(component->checkpoint->Boundary()));
    impl_->metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
  } catch (const std::exception& error) {
    Logger::Info("cache", std::string("event=persistence_refused reason=") +
                              error.what());
  }
  return component->bytes;
}
std::size_t ComponentTextCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<ContinuationToken> live) {
  const auto bytes =
      snapshot ? PublishSnapshot(std::move(tokens), std::move(snapshot), true)
               : 0;
  if (live.empty()) {
    Invalidate();
    return bytes;
  }
  // Optional frontier capture runs after commit when the device is idle.
  // Concurrent foreground leases keep the existing exclusive capture path;
  // an idle task must never race their execution or a new slot admission.
  bool async_frontier = false;
  const bool capture_frontier =
      live.size() > impl_->prompt.size() &&
      (impl_->cache->ram_enabled || (impl_->persist && impl_->cache->queue));
  // Clear admission's barrier even if commit bookkeeping fails before launch.
  struct FrontierBarrier {
    ComponentTextCache::Impl* cache;
    ComponentTextCache::Impl::Entry* entry;
    bool armed{};
    ~FrontierBarrier() {
      if (armed) {
        const std::lock_guard lock(cache->mutex);
        entry->frontier_active = false;
        cache->frontier_ready.notify_all();
      }
    }
  } barrier{impl_->cache, impl_->entry};
  if (capture_frontier) {
    const std::lock_guard lock(impl_->cache->mutex);
    async_frontier =
        std::ranges::none_of(impl_->cache->entries, [&](const auto& e) {
          return e.get() != impl_->entry && (e->busy || e->frontier_active);
        });
    if (async_frontier) {
      impl_->entry->frontier_active = true;
      barrier.armed = true;
    }
  }
  if (capture_frontier && !async_frontier) {
    try {
      impl_->generated_frontier = true;
      impl_->capture_tokens = live;
      impl_->capture_boundary = live.size();
      impl_->purpose = SnapshotPurpose::kContinuation;
      impl_->preserve_source = false;
      if (auto frontier = Capture())
        (void)PublishSnapshot(live, std::move(frontier), false);
    } catch (const std::exception& error) {
      Logger::Info(
          "cache",
          std::string("event=frontier_capture_refused reason=") + error.what());
    }
  }
  auto positions =
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution());
  const auto location = impl_->slot.Location();
  impl_->slot.Commit();
  // The live frontier's index record is optional: a session without one
  // restores from a retained checkpoint, if any. A failed allocation, such as
  // a full ledger, reclaims retained records until the record fits, then
  // leaves the frontier unindexed rather than failing the reply.
  const auto index_live = [&] {
    try {
      impl_->entry->live = impl_->cache->index.Insert(cache::LiveFrontier{
          location, impl_->entry->live_tokens, impl_->cache->identity,
          impl_->entry->live_input, positions});
      return true;
    } catch (const std::bad_alloc&) {
      return false;
    }
  };
  bool indexed;
  {
    const std::lock_guard lock(impl_->cache->mutex);
    auto& starts = impl_->entry->reply_starts;
    auto& ends = impl_->entry->reply_ends;
    starts = std::move(impl_->reply_starts);
    ends = std::move(impl_->reply_ends);
    if (live.size() > impl_->prompt.size()) {
      starts.push_back(impl_->prompt.size());
      ends.push_back(live.size() - 1);
      ends.push_back(live.size());
    }
    impl_->entry->live_tokens = live;
    // The idle session keeps its frontier's pages.
    if (impl_->account)
      (void)impl_->account->SetNeed(live.size());
    impl_->entry->live_input = At(impl_->input, impl_->prompt.size());
    indexed = index_live();
    if (indexed)
      impl_->entry->busy = false;
  }
  if (!indexed) {
    auto& cache = *impl_->cache;
    // Retry under the reclaim's lock, before draining its demotion: the
    // demotion's disk job charges the metadata the reclaim just freed. Queued
    // disk writes are kept, since one may be this turn's frontier checkpoint.
    for (bool reclaimed = true; !indexed && reclaimed;) {
      try {
        const std::lock_guard lock(cache.mutex);
        reclaimed = cache.retention.Reclaim(cache::RetentionRequest{});
        indexed = reclaimed && index_live();
      } catch (const std::exception&) {
        // The reply stands; only the frontier's index record is lost.
        reclaimed = false;
      }
      cache.DrainDemotion();
    }
    {
      const std::lock_guard lock(cache.mutex);
      impl_->entry->busy = false;
    }
    if (!indexed)
      Logger::Info("cache",
                   "schema=component-cache-v1 event=live_frontier_refused "
                   "reason=allocation tokens=" +
                       std::to_string(live.size()));
  }
  impl_->active = false;
  if (async_frontier) {
    auto* cache = impl_->cache;
    auto* entry = impl_->entry;
    try {
      // Publish the joinable handle before completion can wake admission.
      const std::lock_guard launch_lock(cache->mutex);
      // All task inputs are independent of the completed request facade.
      // Teardown joins the bounded per-slot task before draining disk jobs.
      entry->frontier_worker =
          std::thread([cache, entry, location, input = impl_->input,
                       prompt = impl_->prompt, stable = impl_->stable,
                       cached = impl_->cached, source = impl_->source,
                       persist = impl_->persist, live = entry->live_tokens] {
            try {
              auto slot = entry->slot->Acquire(location);
              std::unique_ptr<Impl> request;
              try {
                request = std::make_unique<Impl>(cache, entry, std::move(slot),
                                                 prompt, input, stable);
              } catch (...) {
                slot.Commit();
                throw;
              }
              Lease task(std::move(request));
              task.impl_->active = false;
              try {
                task.impl_->persist = persist;
                task.impl_->cached = cached;
                task.impl_->source = source;
                task.impl_->generated_frontier = true;
                task.impl_->capture_tokens = live;
                task.impl_->capture_boundary = live.size();
                if (auto frontier = task.Capture())
                  (void)task.PublishSnapshot(live, std::move(frontier), false);
              } catch (...) {
                task.impl_->slot.Commit();
                throw;
              }
              task.impl_->slot.Commit();
            } catch (const std::exception& error) {
              Logger::Info(
                  "cache",
                  std::string("event=frontier_capture_refused reason=") +
                      error.what());
            }
            {
              const std::lock_guard lock(cache->mutex);
              entry->frontier_active = false;
            }
            cache->frontier_ready.notify_all();
          });
      barrier.armed = false;
    } catch (...) {
      // The barrier releases admission; the live frontier stays reusable.
    }
  }
  return bytes;
}
void ComponentTextCache::Lease::Invalidate() noexcept {
  if (!impl_ || !impl_->active)
    return;
  impl_->SettleCapture();
  try {
    impl_->state->SetCancellationCheck({});
    impl_->slot = {};
    if (impl_->account)
      (void)impl_->account->SetNeed(0);
    impl_->entry->history.reset();
    const std::lock_guard lock(impl_->cache->mutex);
    impl_->entry->branch.reset();
    impl_->entry->durable.reset();
    impl_->entry->busy = false;
  } catch (...) {
  }
  impl_->active = false;
}

std::vector<std::size_t> ComponentTextCache::Lease::PlannedCaptures(
    std::size_t stable) const {
  // Off with no disk cache keeps no checkpoint, so skip the planning.
  if (!impl_->cache->ram_enabled && !impl_->cache->queue)
    return {};
  const auto& adapter = *impl_->cache->resources->adapter;
  // Each grid checkpoint keeps its private state and at most a window plus a
  // partial leading chunk. Checkpoints closer than a window share those rows,
  // so this bound can only plan fewer captures than RAM holds.
  std::size_t private_bytes = 0;
  for (const auto& d : adapter.Components())
    if (d.kind == cache::ComponentKind::kPrivateState)
      private_bytes += d.state_bytes;
    else if (d.kind == cache::ComponentKind::kWindowRows)
      private_bytes += (d.window_rows + d.rows_per_chunk - 1) * d.row_bytes;
  const auto plan = cache::PlanCaptures(
      adapter, impl_->slot.Execution(),
      {impl_->prompt.size(),
       impl_->cached,
       stable,
       0,
       true,
       false,
       {},
       cache::GridLimit(impl_->cache->budget, private_bytes)});
  std::vector<std::size_t> boundaries;
  for (const auto& capture : plan)
    if (!capture.required &&
        !HasSnapshotFor(std::span<const cache::Token>(impl_->prompt)
                            .first(capture.boundary)))
      boundaries.push_back(capture.boundary);
  return boundaries;
}
void ComponentTextCache::Lease::PersistGrid(
    std::span<const std::size_t> planned, std::size_t boundary) {
  // A later question about the same long document diverges inside the
  // prompt this lease persists at `boundary`, so that checkpoint cannot
  // serve it. Persist the deepest planned grid clear of the question too.
  // The gap leaves the next prompt checkpoint its disk step above the grid
  // even when that question is up to an eighth of the prompt long.
  impl_->disk_grid = 0;
  if (!impl_->persist || !impl_->cache->queue)
    return;
  const auto step = impl_->cache->disk_step;
  const auto gap = std::max(step, boundary / 8) + step;
  for (const auto position : planned)
    if (position < boundary && boundary - position >= gap)
      impl_->disk_grid = std::max(impl_->disk_grid, position);
}
ComponentCacheMetrics ComponentTextCache::Lease::Metrics() const {
  auto report = impl_->metrics;
  report.ledger = impl_->cache->ledger.Snapshot();
  if (impl_->cache->queue)
    report.persistence = impl_->cache->queue->Stats();
  for (const auto& entry : impl_->cache->entries) {
    const auto spill = entry->slot->Metrics();
    report.spill.idle_bytes += spill.idle_bytes;
    report.spill.foreground_bytes += spill.foreground_bytes;
    report.spill.idle_copy_ns += spill.idle_copy_ns;
    report.spill.foreground_copy_ns += spill.foreground_copy_ns;
    report.spill.stream_wait_ns += spill.stream_wait_ns;
    report.spill.residual_wait_ns += spill.residual_wait_ns;
    report.spill.reassignments += spill.reassignments;
    report.spill.idle_failures += spill.idle_failures;
  }
  return report;
}

bool ComponentTextCache::restores_beside_decode() const noexcept {
  return impl_->resources->restore_beside_decode;
}

bool ComponentTextCache::shared_state() const noexcept {
  return mem::StateLedger::Process().PromiseCap() != 0 &&
         std::ranges::any_of(impl_->entries,
                             [](const auto& e) { return bool(e->account); });
}

std::size_t ComponentTextCache::StateBytesFor(std::size_t tokens) const {
  if (!shared_state())
    return 0;
  for (const auto& entry : impl_->entries)
    if (entry->account)
      return entry->account->BytesFor(tokens);
  return 0;
}

std::size_t ComponentTextCache::ReclaimableStateBytes() const {
  if (!shared_state())
    return 0;
  const std::lock_guard lock(impl_->mutex);
  std::size_t bytes = 0;
  for (const auto& entry : impl_->entries)
    if (entry->account)
      bytes += entry->busy ? entry->account->tail_bytes()
                           : entry->account->charged_bytes();
  return bytes;
}

std::size_t ComponentTextCache::RelieveHostPressure(
    const ext::HostMemorySample& sample, std::optional<std::uint64_t> reserve) {
  auto& impl = *impl_;
  // Idle: nothing is in flight, so what waits for the device costs nothing.
  if (impl.resources->settle)
    impl.resources->settle();
  const auto shortfall =
      reserve ? (sample.available < *reserve ? *reserve - sample.available : 0)
              : ext::HostReserveShortfallBytes(sample);
  if (!shortfall) {
    impl.host_pressure_reported = false;
    return 0;
  }
  if (!impl.ram_enabled || !impl.resources->release_host)
    return 0;
  const auto want = static_cast<std::size_t>(
      std::min<std::uint64_t>(shortfall + kHostPressureReliefBytes,
                              std::numeric_limits<std::size_t>::max()));
  const auto start = Clock::now();
  std::size_t released = 0, evicted = 0;
  for (;;) {
    released += impl.resources->release_host(want - released);
    if (released >= want)
      break;
    // Free whole units by evicting the least valuable checkpoint, as when
    // sessions borrow the cache's pages back.
    impl.DiscardUnusedFrontierWrites();
    bool reclaimed;
    {
      const std::lock_guard lock(impl.mutex);
      reclaimed = impl.retention.Reclaim(cache::RetentionRequest{});
    }
    impl.DrainDemotion();
    if (!reclaimed)
      break;
    ++evicted;
  }
  // Once per pressure episode when nothing more can go; each release.
  if (released || evicted || !impl.host_pressure_reported) {
    impl.host_pressure_reported = true;
    Logger::Warn("cache",
                 "schema=component-cache-v1 event=host_pressure_release "
                 "shortfall_bytes=" +
                     std::to_string(shortfall) +
                     " wanted_bytes=" + std::to_string(want) +
                     " released_bytes=" + std::to_string(released) +
                     " evicted=" + std::to_string(evicted) + " ms=" +
                     std::to_string(std::chrono::duration<double, std::milli>(
                                        Clock::now() - start)
                                        .count()) +
                     " " + ext::DescribeHostMemory(sample));
  }
  return released;
}

bool ComponentTextCache::ReclaimState(std::size_t bytes,
                                      std::span<const ContinuationToken> prompt,
                                      const TextPromptContext* context,
                                      std::chrono::milliseconds write_wait) {
  auto& ledger = mem::StateLedger::Process();
  if (!shared_state() || ledger.ChargeableBytes() >= bytes)
    return true;
  const auto start = Clock::now();
  std::size_t tail_bytes = 0, frontier_bytes = 0, frontiers = 0;
  // Frontiers kept for a disk write still reading them, and those dropped
  // with their write after `write_wait`.
  std::size_t deferred = 0, dropped_writes = 0;
  // Where the time goes: unmapping waits for the device; reassigning a slot
  // copies rows checkpoints still borrow from it.
  double unmap_ms = 0, reassign_ms = 0;
  const auto since = [](Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  };
  const auto log = [&](bool enough) {
    // Dropping a frontier with its disk write loses that checkpoint.
    (dropped_writes != 0 ? Logger::Warn : Logger::Info)(
        "memory",
        "event=state_reclaimed needed_bytes=" + std::to_string(bytes) +
            " tail_bytes=" + std::to_string(tail_bytes) +
            " frontier_bytes=" + std::to_string(frontier_bytes) +
            " frontiers=" + std::to_string(frontiers) +
            " deferred_writes=" + std::to_string(deferred) +
            " dropped_writes=" + std::to_string(dropped_writes) +
            " reassign_ms=" + std::to_string(reassign_ms) +
            " unmap_ms=" + std::to_string(unmap_ms) +
            " chargeable_bytes=" + std::to_string(ledger.ChargeableBytes()) +
            " enough=" + (enough ? "true" : "false") + " ms=" +
            std::to_string(
                std::chrono::duration<double, std::milli>(Clock::now() - start)
                    .count()));
  };
  // 1. Pages mapped past each session's need: ahead mappings and an older,
  // longer conversation's rows. No state is lost. Busy sessions have no
  // work in flight here either, and their need covers what they write. A
  // session without a live frontier may still lend rows past its need to
  // checkpoints that have not copied them (an uncommitted release keeps
  // them borrowed): those go through reassignment below, which preserves
  // them first. A disk write that reads rows in place keeps them mapped
  // until it finishes: unmapped, its later pieces would read pages mapped
  // again, zeroed.
  std::vector<std::pair<Impl::Entry*, std::size_t>> tails;
  {
    const std::lock_guard lock(impl_->mutex);
    for (auto& entry : impl_->entries)
      if (entry->account && (entry->busy || entry->live))
        tails.emplace_back(entry.get(), Impl::ObservedWrites(*entry).tokens);
  }
  for (const auto& [entry, written] : tails) {
    if (entry->account->tail_bytes() == 0)
      continue;
    const auto trim_start = Clock::now();
    tail_bytes += entry->account->TrimTo(
        std::max(entry->account->need_tokens(), written));
    unmap_ms += since(trim_start);
    if (ledger.ChargeableBytes() >= bytes) {
      log(true);
      return true;
    }
  }
  // 2. Idle sessions' live frontiers, cheapest to lose first (the order
  // cold admission takes slots in), never the frontier this prompt would
  // continue. Reassigning the slot preserves checkpoints that borrow its
  // rows before they go. A frontier a disk write still reads goes last:
  // reassignment would fail that write, and with it the checkpoint that
  // lets the conversation come back. Within `write_wait` of its capture it
  // stays: the caller retries once the write has finished.
  std::optional<std::size_t> keep;
  struct Candidate {
    Impl::Entry* entry;
    std::optional<Clock::time_point> writing;
  };
  std::vector<Candidate> order;
  {
    const std::lock_guard lock(impl_->mutex);
    const auto input = Inputs(prompt.size(), context);
    const auto found =
        impl_->index.Lookup({impl_->identity, prompt, input, prompt.size(),
                             /*reuse=*/true});
    if (found.selected && found.selected->live)
      keep = found.selected->live->slot.value - 1;
    for (std::size_t i = 0; i < impl_->entries.size(); ++i) {
      auto& entry = *impl_->entries[i];
      if (entry.busy || !entry.account || (keep && *keep == i) ||
          entry.account->charged_bytes() == 0)
        continue;
      order.push_back({&entry, Impl::ObservedWrites(entry).since});
    }
    std::ranges::sort(order, [&](const auto& a, const auto& b) {
      return std::pair(a.writing.has_value(), impl_->ColdRank(*a.entry)) <
             std::pair(b.writing.has_value(), impl_->ColdRank(*b.entry));
    });
  }
  const auto now = Clock::now();
  for (const auto& [entry, writing] : order) {
    if (writing && now - *writing < write_wait) {
      ++deferred;
      continue;
    }
    {
      const std::lock_guard lock(impl_->mutex);
      if (entry->busy)
        continue;
      if (writing && Impl::ObservedWrites(*entry).since)
        ++dropped_writes;
      entry->busy = true;
      if (entry->live) {
        impl_->index.Erase(*entry->live);
        entry->live.reset();
      }
      entry->live_tokens.clear();
      entry->reply_starts.clear();
      entry->reply_ends.clear();
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
    }
    try {
      if (entry->frontier_worker.joinable())
        entry->frontier_worker.join();
      const std::size_t charged = entry->account->charged_bytes();
      {
        const auto reassign_start = Clock::now();
        auto lease = entry->slot->Acquire();
        reassign_ms += since(reassign_start);
        const auto trim_start = Clock::now();
        entry->account->TrimTo(0);
        unmap_ms += since(trim_start);
      }
      frontier_bytes += charged - entry->account->charged_bytes();
      ++frontiers;
    } catch (const std::exception& error) {
      Logger::Warn("memory", std::string("event=state_reclaim_failed reason=") +
                                 error.what());
    }
    {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
    }
    impl_->frontier_ready.notify_all();
    if (ledger.ChargeableBytes() >= bytes) {
      log(true);
      return true;
    }
  }
  // Nothing to free is the common case while requests wait: stay quiet.
  if (tail_bytes != 0 || frontiers != 0)
    log(false);
  return false;
}

}  // namespace gufo::server
