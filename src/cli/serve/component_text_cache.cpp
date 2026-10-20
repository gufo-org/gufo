#include "src/cli/serve/component_text_cache.hpp"

#include <sys/random.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

#include "src/cache/disk_catalog.hpp"
#include "src/cache/retention.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
namespace {
using Clock = std::chrono::steady_clock;
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
struct DurableCapture {
  cache::ResourceCharge metadata;
  std::shared_ptr<const void> branch;
  cache::DiskReadPin ancestor_pin;
  cache::DiskManifest manifest;
  std::vector<cache::PersistenceSource> sources;
};
class ComponentSnapshot final : public TextRunnerSnapshot {
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
              " rank=" + std::to_string(e.rank) +
              " unique_bytes_freed=" + std::to_string(e.unique_bytes_freed) +
              " ledger_bytes=" + std::to_string(e.ledger_bytes));
    } catch (...) {
    }
  }
};
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
    // Acquisition order; cold assignment replaces the oldest live frontier.
    std::uint64_t used{};
    bool busy{};
  };
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
  std::vector<std::unique_ptr<Entry>> entries;
  std::uint64_t clock{};
  std::unique_ptr<cache::DiskCatalog> catalog;
  std::unique_ptr<cache::StreamedStore> streaming;
  std::unique_ptr<cache::PersistenceQueue> queue;
  static constexpr std::size_t kPersistenceDepth = 8;
  // Queue ownership plus transient submissions before superseded callbacks
  // release their serving handles. No callback runs under the serving mutex.
  std::array<std::shared_ptr<const cache::Checkpoint>, 2 * kPersistenceDepth>
      pending;
  std::array<std::shared_ptr<DurableCapture>, 2 * kPersistenceDepth>
      pending_durable;
  std::size_t disk_capacity{}, disk_step{};
  cache::ResourceCharge metadata;

  Impl(std::shared_ptr<TextModelRunner> runner, std::size_t count,
       std::size_t ram,
       const std::optional<TextRunnerDiskCacheOptions>& options,
       bool automatic_ram)
      : runner(std::move(runner)),
        // Disk transfers lease one fixed pinned staging buffer. Its size does
        // not change restore or write speed (#562), and a larger one only
        // takes pinned RAM and device heap from the cache and the sessions.
        ledger({ram + (options ? cache::kDefaultDiskStagingBytes : 0), ram,
                options ? cache::kDefaultDiskStagingBytes : 0, ram}),
        resources(this->runner->CreateComponentCacheResources(ledger, ram)),
        identity(resources->adapter->CompatibilityIdentity()),
        index(ledger),
        retention(ledger, index, &events) {
    metadata = ledger.Reserve(cache::ResourceCategory::kMetadata, sizeof(Impl))
                   .Convert();
    crypto::Sha256Hasher hash;
    hash.Update(identity);
    digest = hash.Finish();
    if (getrandom(nonce.data(), nonce.size(), 0) !=
        static_cast<ssize_t>(nonce.size()))
      throw std::runtime_error("cannot obtain cache publication nonce");
    index.Register(identity, resources->adapter->Components(), digest);
    if (options) {
      const auto row_bytes =
          cache::MinimumStagingBytes(resources->adapter->Components());
      if (ledger.Limits().staging_bytes < row_bytes)
        throw std::logic_error("a " + std::to_string(row_bytes) +
                               "-byte cache row exceeds the disk staging "
                               "buffer");
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
          cache::IdleSpillConfig{1ULL << 20, resources->stream, [this] {
                                   const std::lock_guard lock(mutex);
                                   return std::ranges::none_of(
                                       entries,
                                       [](const auto& e) { return e->busy; });
                                 }});
      entries.push_back(std::move(entry));
    }
    std::size_t automatic_bytes{}, max_bytes{};
    if (resources->commit_backing) {
      const auto claim = this->runner->ResourceClaim();
      const auto automatic = claim.retained_snapshot_capacity_bytes.value_or(0);
      const auto ceiling =
          claim.retained_snapshot_ceiling_bytes.value_or(automatic);
      automatic_bytes =
          std::min(TextRunnerRamCacheOptions::kAutomaticMaxBytes, automatic);
      // Staging is pinned host memory too, so the host keeps its reserve
      // only if both fit under the ceiling.
      const auto staging = ledger.Limits().staging_bytes;
      max_bytes = ceiling > staging ? ceiling - staging : 0;
      ram = std::min(ram, max_bytes);
      if (automatic_ram)
        ram = std::min(ram, automatic);
      ledger.ConstrainLimits({ram + staging, ram, staging, ram});
      resources->commit_backing(ram);
    }
    if (options) {
      catalog = std::make_unique<cache::DiskCatalog>(ledger, *disk, index);
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
            const std::lock_guard lock(mutex);
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
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=published tokens=" +
                      std::to_string(description->Manifest().tokens.size()) +
                      " checkpoint_id=" + std::to_string(id.value) +
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
            return catalog->EvictOne();
          });
    }
    Logger::Info("cache",
                 "schema=component-cache-v1 event=configured sessions=" +
                     std::to_string(count) +
                     " snapshot_entries=128 capacity_bytes=" +
                     std::to_string(ram) + " staging_capacity_bytes=" +
                     std::to_string(ledger.Limits().staging_bytes) +
                     " automatic_bytes=" + std::to_string(automatic_bytes) +
                     " max_bytes=" + std::to_string(max_bytes) +
                     " disk_capacity_bytes=" + std::to_string(disk_capacity));
  }
  ~Impl() {
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
  std::size_t Persist(std::shared_ptr<const cache::Checkpoint> checkpoint) {
    // Grid captures support RAM edits during prefill. Persist semantic prompt
    // and learned boundaries: an early grid must not consume the spacing
    // allowance of the first complete turn.
    if (!queue || checkpoint->Purpose() == cache::CheckpointPurpose::kGrid)
      return 0;
    {
      const std::lock_guard lock(mutex);
      auto compatible_prefix = [&](const auto& tokens, const auto& input) {
        return input == checkpoint->Input() &&
               tokens.size() <= checkpoint->Boundary() &&
               std::equal(tokens.begin(), tokens.end(),
                          checkpoint->Tokens().begin());
      };
      if (checkpoint->Purpose() != cache::CheckpointPurpose::kLearned) {
        auto found = index.Lookup({identity, checkpoint->Tokens(),
                                   cache::InputIdentity(checkpoint->Boundary(),
                                                        checkpoint->Input())});
        std::size_t prior = 0;
        for (const auto& candidate : found.candidates)
          if (Published(candidate))
            prior =
                std::max(prior, static_cast<std::size_t>(candidate.boundary));
        for (const auto& queued : pending)
          if (queued && compatible_prefix(queued->Tokens(), queued->Input()))
            prior =
                std::max(prior, static_cast<std::size_t>(queued->Boundary()));
        for (const auto& queued : pending_durable)
          if (queued && compatible_prefix(queued->manifest.tokens,
                                          queued->manifest.input))
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
  bool WithinDiskStep(std::span<const cache::Token> tokens,
                      const cache::Identity& input) {
    const std::lock_guard lock(mutex);
    auto found = index.Lookup(
        {identity, tokens, cache::InputIdentity(tokens.size(), input)});
    std::size_t prior = 0;
    for (const auto& candidate : found.candidates)
      if (Published(candidate))
        prior = std::max(prior, static_cast<std::size_t>(candidate.boundary));
    const auto compatible = [&](auto prefix, const auto& prefix_input) {
      return prefix_input == input && prefix.size() <= tokens.size() &&
             std::equal(prefix.begin(), prefix.end(), tokens.begin());
    };
    for (const auto& checkpoint : pending)
      if (checkpoint && compatible(checkpoint->Tokens(), checkpoint->Input()))
        prior =
            std::max(prior, static_cast<std::size_t>(checkpoint->Boundary()));
    for (const auto& capture : pending_durable)
      if (capture &&
          compatible(capture->manifest.tokens, capture->manifest.input))
        prior = std::max(prior, capture->manifest.tokens.size());
    return prior && tokens.size() - prior < disk_step;
  }
  std::size_t Persist(std::shared_ptr<DurableCapture> capture) {
    if (capture->sources.empty())
      return 0;
    if (WithinDiskStep(capture->manifest.tokens, capture->manifest.input) &&
        capture->manifest.purpose != cache::CheckpointPurpose::kLearned)
      return 0;
    auto job = cache::PersistenceJob(
        ledger, File(1, capture->manifest.checkpoint.value), capture->manifest,
        std::move(capture->sources));
    std::size_t pending_slot;
    {
      const std::lock_guard lock(mutex);
      const auto position =
          std::find(pending_durable.begin(), pending_durable.end(), nullptr);
      if (position == pending_durable.end())
        return 0;
      pending_slot = position - pending_durable.begin();
      *position = capture;
    }
    if (!queue->TrySubmit(std::move(job))) {
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
  double restore_time{};
  bool disk_hit{}, active{true}, preserve_source{};
  cache::CheckpointId source{};
  std::array<cache::CheckpointId, cache::RetentionPolicy::kRecordLimit>
      edit_boundaries{};
  std::size_t edit_boundary_count{};
  cache::ResourceCharge edit_metadata;
  SnapshotPurpose purpose{SnapshotPurpose::kContinuation};
  std::vector<cache::Token> capture_tokens;
  ContinuationLookup lookup_result;
  ComponentCacheMetrics metrics;
  Impl(ComponentTextCache::Impl* cache, ComponentTextCache::Impl::Entry* entry,
       cache::SlotLease slot, std::span<const cache::Token> prompt,
       cache::InputIdentity input, std::size_t stable)
      : cache(cache),
        entry(entry),
        slot(std::move(slot)),
        state(cache->runner->BindComponentState(*cache->resources->adapter,
                                                this->slot.Execution())),
        input(std::move(input)),
        prompt(prompt.begin(), prompt.end()),
        stable(stable),
        edit_metadata(cache->ledger
                          .Reserve(cache::ResourceCategory::kMetadata,
                                   sizeof(edit_boundaries))
                          .Convert()) {}
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
        .reused_tokens = cached};
  }
  std::unique_ptr<TextRunnerSnapshot> CaptureDurable();
};

std::unique_ptr<TextRunnerSnapshot>
ComponentTextCache::Lease::Impl::CaptureDurable() {
  if (!cache->queue || !cache->resources->copy_observed_rows ||
      purpose == SnapshotPurpose::kHistory)
    throw cache::ResourceExhausted();
  const auto capture_input = At(input, capture_tokens.size());
  if (purpose != SnapshotPurpose::kBranchPoint &&
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
                  : p->valid_rows / d.rows_per_chunk +
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
  manifest.purpose = purpose == SnapshotPurpose::kBranchPoint
                         ? cache::CheckpointPurpose::kLearned
                         : cache::CheckpointPurpose::kPrompt;
  manifest.components.reserve(descriptors.size());
  durable->sources.reserve(pieces);
  std::vector<Copy> copies;
  copies.reserve(pieces);
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    cache::DiskComponent component{.descriptor = d, .position = *p};
    const auto completed =
        d.kind == cache::ComponentKind::kAppendRows
            ? p->valid_rows / d.rows_per_chunk * d.rows_per_chunk
            : 0;
    cache::Rows inherited = 0;
    if (completed && ancestor) {
      const auto& components = ancestor->Manifest().components;
      const auto previous = std::ranges::find_if(
          components, [&](const auto& c) { return c.descriptor.id == d.id; });
      if (previous != components.end()) {
        const auto chunks = std::min<std::size_t>(previous->chunks.size(),
                                                  completed / d.rows_per_chunk);
        component.chunks.assign(previous->chunks.begin(),
                                previous->chunks.begin() + chunks);
        inherited = chunks * d.rows_per_chunk;
      }
    }
    std::shared_ptr<cache::BorrowedRows> rows;
    if (completed) {
      if (inherited != completed) {
        auto view = cache->resources->adapter->BorrowRowsView(
            slot.Execution(), d.id, inherited, completed);
        if (!view.empty())
          rows = slot.Observe(d.id, inherited, completed, view);
      }
      component.chunks.reserve(completed / d.rows_per_chunk);
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
        copies.push_back(
            {request, cache->resources->allocate(request), file, private_file});
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
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->bytes = entry->history->NewPayloadBytes(positions);
  snapshot->durable = std::move(durable);
  return snapshot;
}

ComponentTextCache::ComponentTextCache(
    std::shared_ptr<TextModelRunner> runner, std::size_t count, std::size_t ram,
    const std::optional<TextRunnerDiskCacheOptions>& disk, bool automatic_ram)
    : impl_(std::make_unique<Impl>(std::move(runner), count, ram, disk,
                                   automatic_ram)) {}
ComponentTextCache::~ComponentTextCache() = default;
std::size_t ComponentTextCache::capacity() const noexcept {
  return impl_->entries.size();
}

std::unique_ptr<ComponentTextCache::Lease> ComponentTextCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const ContinuationCache::CancellationCheck& cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse,
    std::size_t stable, bool stop_at_eos) {
  if (cancelled && cancelled())
    return {};
  auto input = Inputs(prompt.size(), context.get());
  cache::PrefixLookup found;
  Impl::Entry* entry = nullptr;
  std::optional<cache::BorrowedLocation> live;
  std::size_t common = 0;
  {
    const std::lock_guard lock(impl_->mutex);
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
      constexpr std::size_t kUncoveredTokens = 4096;
      const auto uncovered = [&](const Impl::Entry& e) {
        if (!e.live || e.live_tokens.size() <= kUncoveredTokens)
          return false;
        return !impl_->index.HasCheckpoint(
            {impl_->identity, e.live_tokens,
             cache::InputIdentity(e.live_tokens.size(), e.live_input)},
            e.live_tokens.size() - kUncoveredTokens);
      };
      using Rank = std::tuple<bool, bool, bool, bool, std::uint64_t>;
      const auto rank = [&](const Impl::Entry& e) {
        return Rank{uncovered(e), !e.slot->PreservationComplete(), bool(e.live),
                    e.live_tokens.size() >= kUncoveredTokens, e.used};
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
  }
  std::unique_ptr<Lease> lease;
  try {
    auto slot = entry->slot->Acquire(live);
    lease = std::make_unique<Lease>(std::make_unique<Lease::Impl>(
        impl_.get(), entry, std::move(slot), prompt, input, stable));
    auto& request = *lease->impl_;
    request.state->SetCancellationCheck(cancelled);
    request.state->SetStopAtEos(stop_at_eos);
    impl_->runner->SetPromptContext(*request.state, context);
    const auto started = Clock::now();
    // Cold Acquire has already reset the slot. Only an attempted restore (or
    // an unusable live frontier) can require another reset before cold use.
    bool clean = !live;
    bool restored = bool(live);
    if (live)
      request.cached = found.selected->boundary;
    for (const auto& candidate : found.candidates) {
      if (restored || (cancelled && cancelled()))
        break;
      if (candidate.live)
        continue;
      try {
        if (candidate.checkpoint) {
          const auto& checkpoint = *candidate.checkpoint;
          std::vector<cache::ComponentPosition> positions;
          for (const auto& c : checkpoint.Components())
            positions.push_back(c.position);
          clean = false;
          impl_->resources->adapter->BeginRestore(request.slot.Execution(),
                                                  positions);
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
          TransferBatch transfers(impl_->resources->transfer_batch_size);
          std::size_t base = 0;
          for (const auto& c : checkpoint.Components()) {
            auto load = [&](const cache::RowPin& pin, cache::Rows first,
                            cache::Rows end) {
              const auto bytes =
                  c.descriptor.kind == cache::ComponentKind::kPrivateState
                      ? c.descriptor.state_bytes
                      : (end - first) * c.descriptor.row_bytes;
              if (pin.Bytes().empty())
                throw std::logic_error("RAM source has no readable view");
              const std::span<const std::byte> buffer(pin.Bytes().data(),
                                                      bytes);
              transfers.Add(
                  [&] { return impl_->resources->stream(); },
                  [&](cache::Stream& stream) {
                    return c.descriptor.kind ==
                                   cache::ComponentKind::kPrivateState
                               ? impl_->resources->adapter->LoadPrivate(
                                     request.slot.Execution(), c.descriptor.id,
                                     buffer, stream)
                               : impl_->resources->adapter->CopyRowsIn(
                                     request.slot.Execution(), c.descriptor.id,
                                     first, end, buffer, stream);
                  });
              request.restored += bytes;
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
          restored = impl_->streaming->Restore(
              id, impl_->digest, *impl_->resources->adapter,
              request.slot.Execution(), timing, {}, {},
              [this](bool private_file, cache::DiskFileId file) {
                const std::lock_guard lock(impl_->mutex);
                impl_->catalog->Invalidate(private_file, file);
              });
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
        if (restored) {
          request.cached = candidate.boundary;
          const std::lock_guard lock(impl_->mutex);
          entry->durable = candidate.durable;
        }
      } catch (...) {
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        clean = true;
        request.restored = 0;
      }
    }
    if (!restored) {
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
                     " restored_bytes=" + std::to_string(request.restored));
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
    if (tokens.size() > prompt.size() ||
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

std::size_t ComponentTextCache::CachedPrefixTokens(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool common) const {
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex);
  const cache::PrefixQuery query{impl_->identity, prompt, input};
  return common ? impl_->index.CommonPrefixTokens(query)
                : impl_->index.CachedPrefixTokens(query);
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
  if (!boundary || boundary > impl_->prompt.size() ||
      (bytes > impl_->cache->ledger.Limits().ram_bytes &&
       (!impl_->cache->queue || purpose == SnapshotPurpose::kHistory)))
    return false;
  impl_->capture_boundary = boundary;
  if (tokens.empty())
    tokens = std::span<const cache::Token>(impl_->prompt).first(boundary);
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
  if (impl_->cache->resources->prepare_prefill)
    impl_->cache->resources->prepare_prefill(tokens);
}

std::size_t ComponentTextCache::Lease::CaptureBytes() const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution()));
}
std::size_t ComponentTextCache::Lease::CaptureBytes(
    std::size_t boundary) const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->PrefillCheckpointPositions(
          impl_->slot.Execution(), boundary));
}
std::unique_ptr<TextRunnerSnapshot> ComponentTextCache::Lease::Capture() {
  auto& request = *impl_;
  auto& cache = *request.cache;
  auto positions =
      cache.resources->adapter->Positions(request.slot.Execution());
  const auto bytes = request.entry->history->NewPayloadBytes(positions);
  const auto retain = request.Retain(request.capture_tokens, bytes);
  if (bytes > cache.ledger.Limits().ram_bytes)
    return request.CaptureDurable();
  const auto reclaim = [&] {
    const std::lock_guard lock(cache.mutex);
    return cache.retention.Reclaim(retain);
  };
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::size_t captured_capacity = 0;
  std::size_t borrowed_capacity = 0, copied_payload = 0;
  while (true) {
    try {
      captured_capacity = 0;
      borrowed_capacity = copied_payload = 0;
      cache::ResourceReservation buffer_metadata;
      std::vector<ComponentCaptureBuffer> buffers;
      std::vector<std::span<const std::byte>> borrowed;
      std::size_t next_buffer = 0;
      checkpoint = request.entry->history->Capture(
          {request.capture_tokens,
           request.input.Through(request.capture_tokens.size()), positions,
           request.purpose == SnapshotPurpose::kHistory
               ? cache::CheckpointPurpose::kGrid
           : request.purpose == SnapshotPurpose::kBranchPoint
               ? cache::CheckpointPurpose::kLearned
               : cache::CheckpointPurpose::kPrompt,
           0},
          [&](const cache::PayloadRequest&) {
            const auto i = next_buffer++;
            auto& buffer = buffers.at(i);
            auto storage = borrowed[i].empty()
                               ? buffer.finish()
                               : buffer.borrow(request.slot, borrowed[i]);
            captured_capacity += storage.Bytes();
            if (!borrowed[i].empty())
              borrowed_capacity += storage.Bytes();
            return storage;
          },
          [&](std::span<const cache::PayloadRequest> payloads) {
            buffer_metadata = cache.ledger.Reserve(
                cache::ResourceCategory::kMetadata,
                payloads.size() *
                    (sizeof(ComponentCaptureBuffer) +
                     sizeof(std::span<const std::byte>) + sizeof(std::size_t)));
            buffers.reserve(payloads.size());
            borrowed.reserve(payloads.size());
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
            while (true) {
              try {
                if (cache.resources->can_capture &&
                    !cache.resources->can_capture(payloads))
                  throw cache::ResourceExhausted();
                buffers.resize(payloads.size());
                borrowed.resize(payloads.size());
                for (const auto i : allocation_order) {
                  const auto& payload = payloads[i];
                  buffers[i] = cache.resources->allocate(payload);
                  borrowed[i] =
                      buffers[i].borrow &&
                              payload.category ==
                                  cache::ResourceCategory::kBackingAssigned
                          ? cache.resources->adapter->BorrowRowsView(
                                request.slot.Execution(), payload.component,
                                payload.first, payload.end)
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
                if (!reclaim())
                  throw;
              }
            }
            TransferBatch transfers(cache.resources->transfer_batch_size);
            for (std::size_t i = 0; i < payloads.size();) {
              if (!borrowed[i].empty()) {
                ++i;
                continue;
              }
              const auto& payload = payloads[i];
              auto end = payload.end;
              auto bytes = payload.bytes;
              auto next = i + 1;
              if (payload.category != cache::ResourceCategory::kPrivateState)
                while (buffers[i].allocation && next < payloads.size() &&
                       borrowed[next].empty() &&
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
    } catch (const cache::ResourceExhausted&) {
      if (!reclaim()) {
        if (cache.queue && request.purpose != SnapshotPurpose::kHistory)
          if (auto durable = request.CaptureDurable())
            return durable;
        // RAM refused the copy; the disk fallback, when there is one, already
        // holds a checkpoint within its step of this boundary.
        Logger::Info("cache",
                     "schema=component-cache-v1 event=capture_refused "
                     "reason=byte_capacity tokens=" +
                         std::to_string(request.capture_tokens.size()));
        throw;
      }
    } catch (const std::exception& error) {
      Logger::Info(
          "cache",
          std::string(
              "schema=component-cache-v1 event=capture_failed reason=") +
              error.what());
      throw;
    }
  }
  ++request.metrics.captures;
  request.metrics.captured_checkpoint_ids.push_back(checkpoint->Id().value);
  request.metrics.captured_bytes += captured_capacity;
  request.metrics.borrowed_capacity_bytes += borrowed_capacity;
  request.metrics.copied_payload_bytes += copied_payload;
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
  if (!admitted)
    return 0;
  if (retain.purpose == cache::RetentionPurpose::kHistory) {
    if (impl_->edit_boundary_count == impl_->edit_boundaries.size())
      throw std::logic_error("too many edit boundaries in one turn");
    impl_->edit_boundaries[impl_->edit_boundary_count++] =
        component->checkpoint->Id();
  }
  try {
    const auto started = Clock::now();
    impl_->metrics.disk_queued_bytes +=
        impl_->cache->Persist(component->checkpoint);
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
  auto positions =
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution());
  const auto location = impl_->slot.Location();
  impl_->slot.Commit();
  {
    const std::lock_guard lock(impl_->cache->mutex);
    impl_->entry->live_tokens = live;
    impl_->entry->live_input = At(impl_->input, impl_->prompt.size());
    impl_->entry->live = impl_->cache->index.Insert(cache::LiveFrontier{
        location, std::move(live), impl_->cache->identity,
        At(impl_->input, impl_->prompt.size()), std::move(positions)});
    impl_->entry->busy = false;
  }
  impl_->active = false;
  return bytes;
}
void ComponentTextCache::Lease::Invalidate() noexcept {
  if (!impl_ || !impl_->active)
    return;
  try {
    impl_->state->SetCancellationCheck({});
    impl_->slot = {};
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
  const auto& adapter = *impl_->cache->resources->adapter;
  std::size_t private_bytes = 0;
  for (const auto& d : adapter.Components())
    if (d.kind == cache::ComponentKind::kPrivateState)
      private_bytes += d.state_bytes;
  const auto plan = cache::PlanCaptures(
      adapter, impl_->slot.Execution(),
      {impl_->prompt.size(),
       impl_->cached,
       stable,
       0,
       true,
       false,
       {},
       cache::GridLimit(impl_->cache->ledger.Limits().ram_bytes,
                        private_bytes)});
  std::vector<std::size_t> boundaries;
  for (const auto& capture : plan)
    if (!capture.required &&
        !HasSnapshotFor(std::span<const cache::Token>(impl_->prompt)
                            .first(capture.boundary)))
      boundaries.push_back(capture.boundary);
  return boundaries;
}
std::vector<std::size_t> ServingCacheLease::PlannedCaptures(
    std::size_t stable) const {
  return component_->PlannedCaptures(stable);
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
std::optional<ComponentCacheMetrics> ServingCacheLease::Metrics() const {
  return component_ ? std::optional(component_->Metrics()) : std::nullopt;
}

#define FORWARD_RESULT(type, name, qualifiers, args, call)         \
  type ServingCacheLease::name args qualifiers {                   \
    return component_ ? component_->name call : legacy_.name call; \
  }
FORWARD_RESULT(ContinuationState&, state, const, (), ())
FORWARD_RESULT(bool, cache_hit, const noexcept, (), ())
FORWARD_RESULT(std::size_t, cached_tokens, const noexcept, (), ())
FORWARD_RESULT(std::size_t, restored_snapshot_bytes, const noexcept, (), ())
FORWARD_RESULT(double, restore_ms, const noexcept, (), ())
FORWARD_RESULT(bool, restored_from_disk, const noexcept, (), ())
FORWARD_RESULT(ContinuationLookup, lookup, const noexcept, (), ())
FORWARD_RESULT(bool, HasSnapshotFor, const,
               (std::span<const ContinuationToken> tokens), (tokens))
FORWARD_RESULT(bool, TryReserveSnapshot, ,
               (std::size_t bytes, std::size_t boundary, bool preserve,
                SnapshotPurpose purpose,
                std::span<const ContinuationToken> tokens),
               (bytes, boundary, preserve, purpose, tokens))
FORWARD_RESULT(void, SkipSnapshot, noexcept,
               (SnapshotEventReason reason, std::size_t bytes,
                std::size_t boundary),
               (reason, bytes, boundary))
FORWARD_RESULT(std::size_t, PublishSnapshot, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                bool preserve),
               (std::move(tokens), std::move(snapshot), preserve))
FORWARD_RESULT(std::size_t, Commit, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                std::vector<ContinuationToken> live),
               (std::move(tokens), std::move(snapshot), std::move(live)))
FORWARD_RESULT(void, Invalidate, noexcept, (), ())
#undef FORWARD_RESULT
void ServingCacheLease::AdoptRestoredPrefix(std::size_t tokens,
                                            std::size_t bytes, double ms) {
  if (component_)
    throw std::logic_error("component restore is owned by the component cache");
  legacy_.AdoptRestoredPrefix(tokens, bytes, ms);
}
void ServingCacheLease::PreparePrefill(std::size_t tokens) {
  if (component_)
    component_->PreparePrefill(tokens);
}

std::size_t ServingCacheLease::CaptureBytes() const {
  return component_->CaptureBytes();
}
std::size_t ServingCacheLease::CaptureBytes(std::size_t boundary) const {
  return component_->CaptureBytes(boundary);
}
std::unique_ptr<TextRunnerSnapshot> ServingCacheLease::Capture() {
  return component_->Capture();
}
}  // namespace gufo::server
