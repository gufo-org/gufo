#include "src/cli/serve/component_text_cache.hpp"

#include <sys/random.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
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
class ComponentSnapshot final : public TextRunnerSnapshot {
public:
  std::shared_ptr<const cache::Checkpoint> checkpoint;
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
    // Streams are only guard fallbacks. Committed captures do not borrow rows;
    // every capture/load itself obtains a fresh one-completion stream lease.
    std::unique_ptr<cache::Stream> preservation;
    std::unique_ptr<cache::LeasedSlot> slot;
    std::optional<cache::IndexEntryId> live;
    std::unique_ptr<cache::ExecutionHistory> history;
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
  std::unique_ptr<cache::DiskCatalog> catalog;
  std::unique_ptr<cache::StreamedStore> streaming;
  std::unique_ptr<cache::PersistenceQueue> queue;
  std::array<std::shared_ptr<const cache::Checkpoint>, 4> pending;
  std::size_t disk_capacity{}, disk_step{};
  cache::ResourceCharge metadata;

  Impl(std::shared_ptr<TextModelRunner> runner, std::size_t count,
       std::size_t ram,
       const std::optional<TextRunnerDiskCacheOptions>& options)
      : runner(std::move(runner)),
        ledger({ram + (options ? (options->staging_capacity_bytes
                                      ? options->staging_capacity_bytes
                                      : cache::kDefaultDiskStagingBytes)
                               : 0),
                ram,
                options ? (options->staging_capacity_bytes
                               ? options->staging_capacity_bytes
                               : cache::kDefaultDiskStagingBytes)
                        : 0,
                ram}),
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
    for (std::size_t i = 0; i < count; ++i) {
      auto entry = std::make_unique<Entry>();
      entry->preservation = Stream();
      entry->slot = std::make_unique<cache::LeasedSlot>(
          ledger, *resources->adapter, *entry->preservation,
          cache::SlotId{i + 1});
      entries.push_back(std::move(entry));
    }
    if (options) {
      disk = std::make_unique<cache::DiskStore>(ledger, options->directory,
                                                options->capacity_bytes);
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
      queue = std::make_unique<cache::PersistenceQueue>(
          ledger, *streaming, pending.size(), ram, [] { return true; },
          [this](cache::CheckpointId id, bool success) {
            const std::lock_guard lock(mutex);
            for (auto& checkpoint : pending)
              if (checkpoint && checkpoint->Id() == id)
                checkpoint.reset();
            if (!success) {
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=persistence_failed");
              return;
            }
            catalog->Track(id);
            if (const auto description = catalog->Find(id)) {
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
                      std::to_string(description->Manifest().tokens.size()));
            }
          },
          [this](const cache::DiskManifest& manifest) {
            // A checkpoint larger than the entire disk budget cannot be
            // admitted by deleting other conversations.
            std::size_t bytes = 0;
            for (const auto& c : manifest.components) {
              for (const auto& chunk : c.chunks)
                bytes += chunk.bytes;
              if (c.tail)
                bytes += c.tail->bytes;
              if (c.private_state)
                bytes += c.private_state->bytes;
            }
            if (bytes > disk_capacity)
              return false;
            const std::lock_guard lock(mutex);
            return catalog->EvictOne();
          });
    }
    Logger::Info("cache",
                 "schema=component-cache-v1 event=configured sessions=" +
                     std::to_string(count) +
                     " snapshot_entries=128 capacity_bytes=" +
                     std::to_string(ram) + " staging_capacity_bytes=" +
                     std::to_string(ledger.Limits().staging_bytes));
  }
  ~Impl() {
    // Settle immutable-source readers before slots/resources/index unwind.
    if (queue)
      queue->Drain();
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
    if (!queue)
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
          if (candidate.durable)
            prior =
                std::max(prior, static_cast<std::size_t>(candidate.boundary));
        for (const auto& queued : pending)
          if (queued && compatible_prefix(queued->Tokens(), queued->Input()))
            prior =
                std::max(prior, static_cast<std::size_t>(queued->Boundary()));
        if (prior && checkpoint->Boundary() - prior < disk_step) {
          Logger::Info("cache",
                       "schema=component-cache-v1 event=persistence_refused "
                       "reason=min_step tokens=" +
                           std::to_string(checkpoint->Boundary()));
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
    const std::lock_guard lock(mutex);
    auto position = std::find(pending.begin(), pending.end(), nullptr);
    if (position == pending.end())
      return 0;
    *position = checkpoint;
    if (!queue->TrySubmit(std::move(job))) {
      position->reset();
      return 0;
    }
    return bytes;
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
        stable(stable) {}
  cache::RetentionRequest Retain(std::span<const cache::Token> tokens) const {
    return {.tokens = tokens,
            .compatibility = cache->identity,
            .input = At(input, std::min(tokens.size(), prompt.size())),
            .purpose = Purpose(purpose),
            .stable = std::min(stable, tokens.size()),
            .source = source,
            .preserve_source = preserve_source,
            .new_payload_bytes = entry->history->NewPayloadBytes(
                cache->resources->adapter->Positions(slot.Execution()))};
  }
};

ComponentTextCache::ComponentTextCache(
    std::shared_ptr<TextModelRunner> runner, std::size_t count, std::size_t ram,
    const std::optional<TextRunnerDiskCacheOptions>& disk)
    : impl_(std::make_unique<Impl>(std::move(runner), count, ram, disk)) {}
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
  {
    const std::lock_guard lock(impl_->mutex);
    if (impl_->catalog)
      impl_->catalog->Reconcile();
    found =
        impl_->index.Lookup({impl_->identity, prompt, input, stable, reuse});
    if (found.selected && found.selected->live) {
      const auto location = *found.selected->live;
      auto& selected_entry = *impl_->entries.at(location.slot.value - 1);
      if (!selected_entry.busy) {
        entry = &selected_entry;
        live = location;
      }
    }
    if (!entry)
      for (auto& candidate : impl_->entries)
        if (!candidate->busy) {
          entry = candidate.get();
          break;
        }
    if (!entry)
      throw cache::SlotBusy();
    entry->busy = true;
    if (entry->live) {
      impl_->index.Erase(*entry->live);
      entry->live.reset();
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
          impl_->resources->adapter->BeginRestore(request.slot.Execution(),
                                                  positions);
          for (const auto& c : checkpoint.Components()) {
            auto load = [&](const cache::Payload& payload, cache::Rows first,
                            cache::Rows end) {
              const auto pin = payload.PinRows();
              if (pin.Location())
                throw std::logic_error("unpreserved RAM source");
              auto owner = pin.Owner() ? pin.Owner() : payload.Owner();
              auto stream = impl_->Stream();
              const auto bytes =
                  c.descriptor.kind == cache::ComponentKind::kPrivateState
                      ? c.descriptor.state_bytes
                      : (end - first) * c.descriptor.row_bytes;
              const std::span<const std::byte> buffer(
                  static_cast<const std::byte*>(owner.get()), bytes);
              if (c.descriptor.kind == cache::ComponentKind::kPrivateState)
                Check(impl_->resources->adapter->LoadPrivate(
                    request.slot.Execution(), c.descriptor.id, buffer,
                    *stream));
              else
                Check(impl_->resources->adapter->CopyRowsIn(
                    request.slot.Execution(), c.descriptor.id, first, end,
                    buffer, *stream));
              request.restored += bytes;
            };
            if (c.private_state)
              load(*c.private_state, 0, 0);
            for (const auto& chunk : c.chunks)
              load(chunk.Storage(), chunk.First(), chunk.End());
            if (c.tail)
              load(*c.tail,
                   c.position.valid_rows / c.descriptor.rows_per_chunk *
                       c.descriptor.rows_per_chunk,
                   c.position.valid_rows);
          }
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
            entry->history = std::make_unique<cache::ExecutionHistory>(
                cache::ExecutionHistory::RestoredUnshared(
                    impl_->ledger, impl_->resources->adapter->Components(),
                    impl_->identity));
            const std::lock_guard lock(impl_->mutex);
            impl_->catalog->Touch(id);
          }
        }
        if (restored)
          request.cached = candidate.boundary;
      } catch (...) {
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        request.restored = 0;
      }
    }
    if (!restored) {
      if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
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
    request.lookup_result = {
        request.cached ? "" : (reuse ? "no_compatible_boundary" : "disabled"),
        0, request.cached};
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
  auto found = impl_->cache->index.Lookup(
      {impl_->cache->identity, tokens,
       cache::InputIdentity(tokens.size(), At(impl_->input, tokens.size()))});
  return std::any_of(
      found.candidates.begin(), found.candidates.end(),
      [&](const auto& c) { return !c.live && c.boundary == tokens.size(); });
}
bool ComponentTextCache::Lease::TryReserveSnapshot(
    std::size_t bytes, std::size_t boundary, bool preserve,
    SnapshotPurpose purpose, std::span<const ContinuationToken> tokens) {
  if (!boundary || boundary > impl_->prompt.size() ||
      bytes > impl_->cache->ledger.Limits().ram_bytes)
    return false;
  impl_->capture_boundary = boundary;
  if (tokens.empty())
    tokens = std::span<const cache::Token>(impl_->prompt).first(boundary);
  impl_->capture_tokens.assign(tokens.begin(), tokens.end());
  impl_->preserve_source = preserve;
  impl_->purpose = purpose;
  return true;
}
void ComponentTextCache::Lease::SkipSnapshot(SnapshotEventReason, std::size_t,
                                             std::size_t) noexcept {}
std::size_t ComponentTextCache::Lease::CaptureBytes() const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution()));
}
std::unique_ptr<TextRunnerSnapshot> ComponentTextCache::Lease::Capture() {
  auto& request = *impl_;
  auto& cache = *request.cache;
  auto positions =
      cache.resources->adapter->Positions(request.slot.Execution());
  const auto bytes = CaptureBytes();
  const auto retain = request.Retain(request.capture_tokens);
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::size_t captured_capacity = 0;
  while (true) {
    try {
      captured_capacity = 0;
      checkpoint = request.entry->history->Capture(
          {request.capture_tokens,
           cache::InputIdentity(request.capture_tokens.size(), retain.input),
           positions,
           request.purpose == SnapshotPurpose::kHistory
               ? cache::CheckpointPurpose::kGrid
           : request.purpose == SnapshotPurpose::kBranchPoint
               ? cache::CheckpointPurpose::kLearned
               : cache::CheckpointPurpose::kPrompt,
           0},
          [&](const cache::PayloadRequest& payload) {
            auto buffer = cache.resources->allocate(payload);
            auto stream = cache.Stream();
            if (payload.category == cache::ResourceCategory::kPrivateState)
              Check(cache.resources->adapter->CapturePrivate(
                  request.slot.Execution(), payload.component,
                  buffer.bytes.first(payload.bytes), *stream));
            else
              Check(cache.resources->adapter->CopyRowsOut(
                  request.slot.Execution(), payload.component, payload.first,
                  payload.end, buffer.bytes.first(payload.bytes), *stream));
            auto storage = buffer.finish();
            captured_capacity += storage.Bytes();
            return storage;
          });
      break;
    } catch (const cache::ResourceExhausted&) {
      const std::lock_guard lock(cache.mutex);
      if (!cache.retention.Reclaim(retain)) {
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
  request.metrics.captured_bytes += captured_capacity;
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
    impl_->entry->busy = false;
  } catch (...) {
  }
  impl_->active = false;
}

std::vector<std::size_t> ComponentTextCache::Lease::PlannedCaptures(
    std::size_t stable) const {
  const auto plan = cache::PlanCaptures(
      *impl_->cache->resources->adapter, impl_->slot.Execution(),
      {impl_->prompt.size(), impl_->cached, stable, 0, true, false, {}});
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
std::size_t ServingCacheLease::CaptureBytes() const {
  return component_->CaptureBytes();
}
std::unique_ptr<TextRunnerSnapshot> ServingCacheLease::Capture() {
  return component_->Capture();
}
}  // namespace gufo::server
