#include "src/cache/streaming.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace gufo::cache {
namespace {
template<class F>
struct Finally {
  F action;
  ~Finally() { action(); }
};
using Clock = std::chrono::steady_clock;
std::uint64_t Elapsed(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                              start)
      .count();
}
std::size_t Add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::invalid_argument("transfer size overflow");
  return a + b;
}
std::size_t Multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("transfer size overflow");
  return a * b;
}
std::size_t ManifestCapacity(const DiskManifest& manifest) {
  auto bytes = Add(sizeof(DiskManifest), manifest.input.size());
  bytes = Add(bytes, Multiply(manifest.tokens.size(), sizeof(Token)));
  bytes =
      Add(bytes, Multiply(manifest.components.size(), sizeof(DiskComponent)));
  for (const auto& component : manifest.components)
    bytes = Add(bytes, Multiply(component.chunks.size(), sizeof(DiskPayload)));
  return bytes;
}
class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("cache transfer cancelled") {}
};
void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled();
}
void Settle(Completion completion, TransferTiming& timing) {
  const auto start = Clock::now();
  const auto result = completion.Wait();
  timing.stream_wait_ns += Elapsed(start);
  if (result != TransferResult::kSucceeded)
    throw std::runtime_error("cache transfer failed");
}
void Accumulate(TransferTiming& destination, const TransferTiming& source) {
#define ADD_COST(field) destination.field += source.field
  ADD_COST(allocator_ns);
  ADD_COST(stream_acquire_ns);
  ADD_COST(stream_wait_ns);
  ADD_COST(metadata_lock_ns);
  ADD_COST(filesystem_ns);
  ADD_COST(fsync_ns);
  ADD_COST(checksum_ns);
  ADD_COST(yield_ns);
  ADD_COST(bytes);
  ADD_COST(pieces);
  ADD_COST(staging_wait_ns);
  ADD_COST(io_lock_ns);
#undef ADD_COST
}
bool SameJob(const DiskManifest& a, const DiskManifest& b) {
  if (a.checkpoint != b.checkpoint || a.lineage != b.lineage ||
      a.compatibility != b.compatibility || a.input != b.input ||
      a.tokens != b.tokens || a.components.size() != b.components.size())
    return false;
  for (std::size_t i = 0; i < a.components.size(); ++i) {
    const auto& x = a.components[i];
    const auto& y = b.components[i];
    if (x.position != y.position || x.descriptor.id != y.descriptor.id ||
        x.descriptor.layout_version != y.descriptor.layout_version ||
        x.descriptor.kind != y.descriptor.kind ||
        x.descriptor.row_bytes != y.descriptor.row_bytes ||
        x.descriptor.rows_per_chunk != y.descriptor.rows_per_chunk ||
        x.descriptor.state_bytes != y.descriptor.state_bytes)
      return false;
  }
  return true;
}
}  // namespace

PersistenceSource::PersistenceSource(bool private_file, DiskFileId file,
                                     std::uint64_t bytes, std::size_t alignment,
                                     ResourceCharge charge,
                                     std::shared_ptr<const void> owner,
                                     Copy copy)
    : charge_(std::move(charge)),
      owner_(std::move(owner)),
      copy_(std::move(copy)),
      private_file_(private_file),
      file_(file),
      bytes_(bytes),
      alignment_(alignment) {
  if (!charge_ || !owner_ || !copy_ || !alignment_ || bytes_ % alignment_)
    throw std::invalid_argument("invalid committed persistence source");
  const auto info = charge_.Info();
  if (info.reserved || bytes_ > info.bytes ||
      (info.pool_backing && !info.assigned_backing))
    throw std::invalid_argument(
        "persistence source does not own its allocation");
  const auto category = info.category;
  if (category != ResourceCategory::kBackingMaterialized &&
      category != ResourceCategory::kPrivateState &&
      category != ResourceCategory::kPrivateTail)
    throw std::invalid_argument("persistence source is not committed payload");
}
struct StreamedStore::Impl {
  ResourceLedger* ledger;
  DiskStore* disk;
  ResourceCharge metadata, staging_charge;
  StagingAllocation staging;
  StreamLease lease;
  std::timed_mutex mutex;
  std::uint64_t allocation_ns{};

  Impl(ResourceLedger& resources, DiskStore& store, std::size_t bytes,
       StagingAllocator allocator, StreamLease provider,
       ResourceReservation reservation)
      : ledger(&resources),
        disk(&store),
        metadata(reservation.Convert()),
        lease(std::move(provider)) {
    if (!bytes || !allocator || !lease)
      throw std::invalid_argument("invalid streaming configuration");
    const auto start = Clock::now();
    auto admission =
        resources.Reserve(ResourceCategory::kTransferStaging, bytes);
    try {
      staging = allocator(bytes);
      if (!staging.owner || staging.bytes.size() != bytes)
        throw std::invalid_argument(
            "staging allocator returned wrong capacity");
      staging_charge = admission.Convert();
    } catch (...) {
      staging = {};
      throw;
    }
    allocation_ns = Elapsed(start);
  }
  std::unique_lock<std::timed_mutex> Lock(std::stop_token stop,
                                          TransferTiming& timing) {
    const auto start = Clock::now();
    Finally wait_observation{[&] { timing.staging_wait_ns += Elapsed(start); }};
    std::unique_lock guard(mutex, std::defer_lock);
    while (!guard.try_lock_for(std::chrono::milliseconds(2)))
      CheckStop(stop);
    CheckStop(stop);
    return guard;
  }
  std::unique_ptr<Stream> Lease(TransferTiming& timing) {
    const auto start = Clock::now();
    auto stream = lease();
    timing.stream_acquire_ns += Elapsed(start);
    if (!stream)
      throw ResourceExhausted();
    return stream;
  }
};
StreamedStore::StreamedStore(ResourceLedger& ledger, DiskStore& disk,
                             std::size_t bytes, StagingAllocator allocator,
                             StreamLease lease)
    : impl_(std::make_unique<Impl>(
          ledger, disk, bytes, std::move(allocator), std::move(lease),
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)))) {}
StreamedStore::~StreamedStore() = default;
StreamedStore::StreamedStore(ResourceLedger& ledger, DiskStore& disk,
                             StagingAllocator allocator, StreamLease lease)
    : StreamedStore(ledger, disk, kDefaultDiskStagingBytes,
                    std::move(allocator), std::move(lease)) {}
std::size_t StreamedStore::StagingBytes() const {
  return impl_->staging.bytes.size();
}
std::uint64_t StreamedStore::StartupAllocatorNanoseconds() const {
  return impl_->allocation_ns;
}
void StreamedStore::Write(DiskFileId file, const DiskManifest& manifest,
                          std::span<PersistenceSource> sources,
                          TransferTiming& timing, std::stop_token stop,
                          const std::function<bool()>& model_idle) {
  CheckStop(stop);
  const auto start = Clock::now();
  bool allocation_recorded{};
  Finally allocation_observation{[&] {
    if (!allocation_recorded)
      timing.allocator_ns += Elapsed(start);
  }};
  auto reservation = impl_->ledger->Reserve(
      ResourceCategory::kMetadata,
      Add(128, Multiply(sources.size(), sizeof(DiskWriteSource) +
                                            sizeof(PersistencePin) + 64)));
  std::optional<std::unique_lock<std::timed_mutex>> piece_lock;
  DiskStagingAccess access{
      [&] {
        CheckStop(stop);
        {
          const auto start = Clock::now();
          Finally yield_observation{[&] { timing.yield_ns += Elapsed(start); }};
          if (model_idle) {
            std::mutex wait_mutex;
            std::condition_variable_any wake;
            std::unique_lock wait_guard(wait_mutex);
            while (!model_idle()) {
              CheckStop(stop);
              wake.wait_for(wait_guard, stop, std::chrono::milliseconds(2),
                            [] { return false; });
            }
          }
        }
        piece_lock.emplace(impl_->Lock(stop, timing));
      },
      [&] { piece_lock.reset(); }};
  std::vector<PersistencePin> pins;
  pins.reserve(sources.size());
  for (auto& source : sources)
    pins.push_back(source.charge_.PinPersistence());
  std::vector<DiskWriteSource> writes;
  writes.reserve(sources.size());
  for (auto& source : sources) {
    writes.push_back({source.private_file_, source.file_, source.bytes_,
                      [&, source_ptr = &source](auto offset, auto piece) {
                        CheckStop(stop);
                        auto stream = impl_->Lease(timing);
                        Settle(
                            source_ptr->copy_(
                                offset, std::as_writable_bytes(piece), *stream),
                            timing);
                        CheckStop(stop);
                        timing.bytes += piece.size();
                        ++timing.pieces;
                      },
                      source.alignment_});
  }
  timing.allocator_ns += Elapsed(start);
  allocation_recorded = true;
  DiskPublicationStats costs;
  Finally observe{[&] {
    timing.filesystem_ns += costs.filesystem_ns;
    timing.fsync_ns += costs.fsync_ns;
    timing.checksum_ns += costs.checksum_ns;
    timing.metadata_lock_ns += costs.metadata_lock_ns;
    timing.io_lock_ns += costs.io_lock_ns;
  }};
  (void)impl_->disk->PublishStream(file, manifest, writes, impl_->staging.bytes,
                                   &costs, &access, stop);
}
bool StreamedStore::Restore(
    CheckpointId id, CompatibilityDigest compatibility, Adapter& adapter,
    Slot& slot, TransferTiming& timing, std::stop_token stop,
    const std::function<void()>& fallback,
    const std::function<void(bool, DiskFileId)>& dependency_failed) {
  std::optional<std::pair<bool, DiskFileId>> failed;
  try {
    auto guard = impl_->Lock(stop, timing);
    auto start = Clock::now();
    auto snapshot = impl_->disk->Open(id);
    timing.metadata_lock_ns += Elapsed(start);
    if (!snapshot)
      throw std::runtime_error("missing disk checkpoint");
    const auto& manifest = snapshot->Manifest();
    if (manifest.compatibility != compatibility ||
        !adapter.GetCapabilities().continuation ||
        !adapter.GetCapabilities().persistent_encoding ||
        adapter.Components().size() != manifest.components.size())
      throw std::invalid_argument("unsupported checkpoint adapter");
    auto metadata = impl_->ledger->Reserve(
        ResourceCategory::kMetadata,
        Multiply(manifest.components.size(), sizeof(ComponentPosition)));
    std::vector<ComponentPosition> positions;
    positions.reserve(manifest.components.size());
    for (const auto& c : manifest.components) {
      auto descriptor =
          std::find_if(adapter.Components().begin(), adapter.Components().end(),
                       [&](auto d) { return d.id == c.descriptor.id; });
      if (descriptor == adapter.Components().end() ||
          descriptor->layout_version != c.descriptor.layout_version ||
          descriptor->kind != c.descriptor.kind ||
          descriptor->row_bytes != c.descriptor.row_bytes ||
          descriptor->rows_per_chunk != c.descriptor.rows_per_chunk ||
          descriptor->state_bytes != c.descriptor.state_bytes ||
          (descriptor->kind == ComponentKind::kAppendRows &&
           descriptor->row_bytes > impl_->staging.bytes.size()))
        throw std::invalid_argument("checkpoint component layout mismatch");
      positions.push_back(c.position);
    }
    CheckStop(stop);
    adapter.BeginRestore(slot, positions);
    for (const auto& c : manifest.components) {
      Rows first{};
      auto read = [&](bool private_file, const DiskPayload& payload,
                      bool private_state) {
        auto piece = impl_->staging.bytes;
        if (!private_state)
          piece = piece.first(piece.size() / c.descriptor.row_bytes *
                              c.descriptor.row_bytes);
        DiskReadStats costs;
        Finally observe{[&] {
          timing.filesystem_ns += costs.filesystem_ns;
          timing.checksum_ns += costs.checksum_ns;
        }};
        try {
          (void)impl_->disk->ReadPayload(
              *snapshot, private_file, payload, piece,
              [&](std::uint64_t offset, auto bytes) {
                CheckStop(stop);
                auto stream = impl_->Lease(timing);
                auto completion =
                    private_state ? adapter.LoadPrivatePiece(
                                        slot, c.descriptor.id, offset,
                                        std::as_bytes(bytes), *stream)
                                  : adapter.CopyRowsIn(
                                        slot, c.descriptor.id,
                                        first + offset / c.descriptor.row_bytes,
                                        first + (offset + bytes.size()) /
                                                    c.descriptor.row_bytes,
                                        std::as_bytes(bytes), *stream);
                Settle(std::move(completion), timing);
                ++timing.pieces;
                timing.bytes += bytes.size();
                CheckStop(stop);
              },
              &costs);
        } catch (const DiskDependencyError& error) {
          failed = {error.private_file, error.file};
          throw;
        }
        if (!private_state)
          first += payload.bytes / c.descriptor.row_bytes;
      };
      for (const auto& chunk : c.chunks)
        read(false, chunk, false);
      if (c.tail)
        read(true, *c.tail, false);
      if (c.private_state)
        read(true, *c.private_state, true);
    }
    CheckStop(stop);
    if (!adapter.Validate(slot, positions))
      throw std::runtime_error("checkpoint position validation failed");
    return true;
  } catch (...) {
    // Every completion was settled before unwinding. A failed BeginRestore may
    // also have disabled execution, so invalidate even when began is false.
    if (!adapter.Invalidate(slot))
      throw std::runtime_error("cannot invalidate failed disk restore");
    if (failed && dependency_failed)
      dependency_failed(failed->first, failed->second);
    if (!stop.stop_requested() && fallback)
      fallback();
    return false;
  }
}
PersistenceJob::PersistenceJob(ResourceLedger& ledger, DiskFileId file,
                               const DiskManifest& manifest,
                               std::vector<PersistenceSource> sources)
    : file_(file) {
  auto reservation = ledger.Reserve(
      ResourceCategory::kMetadata,
      Add(Add(sizeof(PersistenceJob), ManifestCapacity(manifest)),
          Multiply(sources.capacity(), sizeof(PersistenceSource))));
  try {
    manifest_ = manifest;
    sources_ = std::move(sources);
    for (const auto& source : sources_)
      pinned_bytes_ = Add(pinned_bytes_, source.charge_.Info().bytes);
    metadata_ = reservation.Convert();
  } catch (...) {
    manifest_ = {};
    std::vector<PersistenceSource>().swap(sources_);
    throw;
  }
}
struct PersistenceQueue::Impl {
  ResourceCharge metadata;
  StreamedStore* store;
  std::size_t max_bytes;
  std::function<bool()> idle;
  mutable std::mutex mutex;
  std::mutex stop_mutex;
  std::condition_variable wake;
  std::vector<std::unique_ptr<PersistenceJob>> pending;
  std::unique_ptr<PersistenceJob> active;
  PersistenceQueueStats stats;
  bool stopped{};
  std::jthread worker;
  Impl(ResourceReservation reservation, StreamedStore& streams,
       std::size_t depth, std::size_t bytes, std::function<bool()> model_idle)
      : metadata(reservation.Convert()),
        store(&streams),
        max_bytes(bytes),
        idle(std::move(model_idle)),
        pending(depth) {
    if (!depth || !bytes || !idle)
      throw std::invalid_argument("invalid persistence queue limits");
    worker = std::jthread([this](std::stop_token stop) {
      for (;;) {
        {
          std::unique_lock guard(mutex);
          wake.wait(guard, [&] { return stopped || stats.pending != 0; });
          if (stopped)
            return;
          auto it = std::find_if(pending.begin(), pending.end(),
                                 [](const auto& job) { return bool(job); });
          active = std::move(*it);
        }
        bool succeeded{};
        TransferTiming timing;
        try {
          store->Write(active->file_, active->manifest_, active->sources_,
                       timing, stop, idle);
          succeeded = true;
        } catch (...) {
          // Optional persistence failures remain observable and never enable a
          // checkpoint. Disk recovery retains partial bytes conservatively.
        }
        {
          std::lock_guard guard(mutex);
          succeeded ? ++stats.completed : ++stats.failed;
          Accumulate(stats.timing, timing);
          stats.pinned_bytes -= active->pinned_bytes_;
          --stats.pending;
          // Free the job before its admission/pin accounting can be reused.
          active.reset();
        }
        wake.notify_all();
      }
    });
  }
};
PersistenceQueue::PersistenceQueue(ResourceLedger& ledger, StreamedStore& store,
                                   std::size_t depth, std::size_t bytes,
                                   std::function<bool()> model_idle)
    : impl_(std::make_unique<Impl>(
          ledger.Reserve(ResourceCategory::kMetadata,
                         Add(sizeof(Impl), Multiply(depth, sizeof(void*)))),
          store, depth, bytes, std::move(model_idle))) {}
PersistenceQueue::~PersistenceQueue() {
  Stop();
}
bool PersistenceQueue::TrySubmit(PersistenceJob job) {
  std::lock_guard guard(impl_->mutex);
  if (impl_->stopped) {
    ++impl_->stats.skipped;
    return false;
  }
  if ((impl_->active && SameJob(impl_->active->manifest_, job.manifest_)) ||
      std::any_of(impl_->pending.begin(), impl_->pending.end(),
                  [&](auto& other) {
                    return other && SameJob(other->manifest_, job.manifest_);
                  })) {
    ++impl_->stats.coalesced;
    return false;
  }
  if (impl_->stats.pending == impl_->pending.size() ||
      job.pinned_bytes_ > impl_->max_bytes - impl_->stats.pinned_bytes) {
    ++impl_->stats.skipped;
    return false;
  }
  auto slot = std::find_if(impl_->pending.begin(), impl_->pending.end(),
                           [](auto& item) { return !item; });
  try {
    for (auto& source : job.sources_)
      source.pin_ = source.charge_.PinPersistence();
    *slot = std::make_unique<PersistenceJob>(std::move(job));
  } catch (const std::bad_alloc&) {
    ++impl_->stats.skipped;
    return false;
  }
  ++impl_->stats.accepted;
  ++impl_->stats.pending;
  impl_->stats.pinned_bytes += (*slot)->pinned_bytes_;
  impl_->stats.peak_pending =
      std::max(impl_->stats.peak_pending, impl_->stats.pending);
  impl_->stats.peak_pinned_bytes =
      std::max(impl_->stats.peak_pinned_bytes, impl_->stats.pinned_bytes);
  impl_->wake.notify_all();
  return true;
}
void PersistenceQueue::Drain() {
  std::unique_lock guard(impl_->mutex);
  impl_->wake.wait(guard, [&] { return impl_->stats.pending == 0; });
}
void PersistenceQueue::Stop() {
  // Joining a jthread is not concurrently safe. Keep lifecycle serialization
  // separate from state: the worker needs the state mutex while settling.
  std::lock_guard stop_guard(impl_->stop_mutex);
  {
    std::lock_guard guard(impl_->mutex);
    impl_->stopped = true;
    impl_->worker.request_stop();
    for (auto& job : impl_->pending) {
      if (!job)
        continue;
      --impl_->stats.pending;
      impl_->stats.pinned_bytes -= job->pinned_bytes_;
      ++impl_->stats.skipped;
      job.reset();
    }
  }
  impl_->wake.notify_all();
  if (impl_->worker.joinable())
    impl_->worker.join();
}
PersistenceQueueStats PersistenceQueue::Stats() const {
  std::lock_guard guard(impl_->mutex);
  return impl_->stats;
}
}  // namespace gufo::cache
