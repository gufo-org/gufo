#include "src/cache/streaming.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/cache/slot.hpp"

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
class SettledSignal final : public CompletionSignal {
public:
  bool Ready() const noexcept override { return true; }
  TransferResult Wait() noexcept override { return TransferResult::kSucceeded; }
};
class PinnedRowSignal final : public CompletionSignal {
public:
  PinnedRowSignal(RowPin pin, Completion completion)
      : pin_(std::move(pin)), completion_(std::move(completion)) {}
  bool Ready() const noexcept override { return completion_.Ready(); }
  TransferResult Wait() noexcept override { return completion_.Wait(); }

private:
  RowPin pin_;
  Completion completion_;
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
// Cold checkpoints are thousands of small files, and one thread hinting them
// in turn waits on each: 0.4 GB/s on gfx1151's NVMe, against 3.2 GB/s for four
// helpers (measured). Helpers hint whole files up to a bounded byte window
// ahead of the reader, which still reads, checks and copies every file in
// order.
class Prefetcher {
public:
  struct Item {
    bool private_file;
    const DiskPayload* payload;
  };
  Prefetcher(const DiskStore& disk, const DiskSnapshot& snapshot,
             std::vector<Item> items)
      : disk_(disk), snapshot_(snapshot), items_(std::move(items)) {
    offsets_.reserve(items_.size() + 1);
    offsets_.push_back(0);
    for (const auto& item : items_)
      offsets_.push_back(offsets_.back() + item.payload->bytes);
    for (auto& thread : threads_)
      thread = std::jthread([this](std::stop_token stop) { Run(stop); });
  }
  // The reader is about to read item `index`.
  void Reading(std::size_t index) {
    {
      const std::lock_guard lock(mutex_);
      reader_ = index;
    }
    wake_.notify_all();
  }

private:
  static constexpr std::uint64_t kWindowBytes = 32 << 20;
  void Run(std::stop_token stop) {
    while (true) {
      std::size_t i;
      {
        std::unique_lock lock(mutex_);
        if (!wake_.wait(lock, stop, [&] {
              return next_ < items_.size() &&
                     offsets_[next_] - offsets_[reader_] < kWindowBytes;
            }))
          return;
        i = next_++;
      }
      try {
        disk_.PrefetchPayload(snapshot_, items_[i].private_file,
                              *items_[i].payload, items_[i].payload->bytes);
      } catch (...) {
        // A hint only; the reader validates every dependency it reads.
      }
    }
  }
  const DiskStore& disk_;
  const DiskSnapshot& snapshot_;
  std::vector<Item> items_;
  std::vector<std::uint64_t> offsets_;
  std::mutex mutex_;
  std::condition_variable_any wake_;
  std::size_t next_{}, reader_{};
  std::array<std::jthread, 4> threads_;
};
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
        x.descriptor.state_bytes != y.descriptor.state_bytes ||
        x.descriptor.window_rows != y.descriptor.window_rows)
      return false;
  }
  return true;
}
bool Extends(const DiskManifest& next, const DiskManifest& prior) {
  return next.compatibility == prior.compatibility &&
         next.input == prior.input &&
         prior.tokens.size() < next.tokens.size() &&
         std::equal(prior.tokens.begin(), prior.tokens.end(),
                    next.tokens.begin());
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
PersistenceSource::PersistenceSource(bool private_file, DiskFileId file,
                                     std::uint64_t bytes, std::size_t alignment,
                                     std::shared_ptr<BorrowedRows> rows,
                                     CopyRows read_live)
    : rows_(std::move(rows)),
      live_reads_(read_live && rows_ && rows_->SourceViewBytes() >= bytes),
      private_file_(private_file),
      file_(file),
      bytes_(bytes),
      alignment_(alignment) {
  if (!rows_ || !rows_->IsValid() || !rows_->HasBacking() || !alignment_ ||
      bytes_ % alignment_ || bytes_ > rows_->Bytes())
    throw std::invalid_argument("invalid borrowed persistence source");
  if (live_reads_) {
    copy_ = [rows = rows_, copy = std::move(read_live)](
                std::uint64_t offset, std::span<std::byte> destination,
                Stream& stream) {
      // Preserved rows are read from their backing, the others from the
      // slot. Either way the pin keeps the bytes until the copy completes.
      auto pin = rows->Pin();
      const auto source = pin.Bytes().subspan(offset, destination.size());
      if (!pin.Location()) {
        std::memcpy(destination.data(), source.data(), destination.size());
        return Completion(std::make_unique<SettledSignal>());
      }
      auto completion = copy(source, destination, stream);
      return Completion(std::make_unique<PinnedRowSignal>(
          std::move(pin), std::move(completion)));
    };
    return;
  }
  copy_ = [rows = rows_](std::uint64_t offset, std::span<std::byte> destination,
                         Stream&) {
    // Write waits for preservation before entering disk publication. Owner
    // refuses retired or still-live rows, so this never reads a live slot.
    const auto owner = rows->Owner();
    if (!owner)
      throw std::runtime_error("borrowed persistence source retired");
    std::memcpy(destination.data(),
                static_cast<const std::byte*>(owner.get()) + offset,
                destination.size());
    return Completion(std::make_unique<SettledSignal>());
  };
}
PersistenceSource::PersistenceSource(bool private_file, DiskFileId file,
                                     std::uint64_t bytes, std::size_t alignment,
                                     std::shared_ptr<BorrowedRows> rows,
                                     std::size_t first_byte, CopyRows copy)
    : rows_(std::move(rows)),
      private_file_(private_file),
      file_(file),
      bytes_(bytes),
      alignment_(alignment) {
  if (!rows_ || !rows_->IsValid() || rows_->HasBacking() || !copy ||
      !alignment || bytes % alignment || first_byte > rows_->Bytes() ||
      bytes > rows_->Bytes() - first_byte)
    throw std::invalid_argument("invalid observed persistence source");
  copy_ = [rows = rows_, first_byte, bytes, copy = std::move(copy)](
              std::uint64_t offset, std::span<std::byte> destination,
              Stream& stream) {
    if (offset > bytes || destination.size() > bytes - offset)
      throw std::invalid_argument("invalid observed persistence piece");
    auto pin = rows->Pin();
    const auto source =
        pin.Bytes().subspan(first_byte + offset, destination.size());
    auto completion = copy(source, destination, stream);
    return Completion(std::make_unique<PinnedRowSignal>(std::move(pin),
                                                        std::move(completion)));
  };
}
PersistencePin PersistenceSource::Pin() const {
  return rows_ ? rows_->PinPersistence() : charge_.PinPersistence();
}
std::size_t PersistenceSource::Capacity() const {
  return rows_ ? (rows_->HasBacking() ? rows_->Bytes() : 0)
               : charge_.Info().bytes;
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
  // Preservation runs independently through slot guards/idle workers. Never
  // hold staging, a transfer stream, filesystem publication locks or a live
  // source pin while waiting. Foreground restore can therefore preempt us.
  const auto preservation_start = Clock::now();
  for (const auto& source : sources) {
    if (!source.rows_ || !source.rows_->HasBacking() || source.live_reads_)
      continue;
    std::mutex wait_mutex;
    std::condition_variable_any wake;
    std::unique_lock wait_guard(wait_mutex);
    while (true) {
      CheckStop(stop);
      if (!source.rows_->IsValid())
        throw std::runtime_error("borrowed persistence source retired");
      if (source.rows_->PreservationFailed())
        throw std::runtime_error("borrowed persistence source spill failed");
      if (!source.rows_->Location())
        break;
      wake.wait_for(wait_guard, stop, std::chrono::milliseconds(1),
                    [] { return false; });
    }
  }
  timing.yield_ns += Elapsed(preservation_start);
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
    pins.push_back(source.Pin());
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
          descriptor->window_rows != c.descriptor.window_rows ||
          (HasRows(descriptor->kind) &&
           descriptor->row_bytes > impl_->staging.bytes.size()))
        throw std::invalid_argument("checkpoint component layout mismatch");
      positions.push_back(c.position);
    }
    CheckStop(stop);
    adapter.BeginRestore(slot, positions);
    std::vector<Prefetcher::Item> items;
    for (const auto& c : manifest.components) {
      for (const auto& chunk : c.chunks)
        items.push_back({false, &chunk});
      if (c.tail)
        items.push_back({true, &*c.tail});
      if (c.private_state)
        items.push_back({true, &*c.private_state});
    }
    Prefetcher prefetch(*impl_->disk, *snapshot, std::move(items));
    std::size_t reading{};
    for (const auto& c : manifest.components) {
      Rows first = FirstChunkRow(c.descriptor, c.position.valid_rows);
      std::size_t staged{};
      auto rows = impl_->staging.bytes;
      if (HasRows(c.descriptor.kind))
        rows = rows.first(rows.size() / c.descriptor.row_bytes *
                          c.descriptor.row_bytes);
      const auto flush = [&] {
        if (!staged)
          return;
        CheckStop(stop);
        auto stream = impl_->Lease(timing);
        Settle(
            adapter.CopyRowsIn(slot, c.descriptor.id,
                               first - staged / c.descriptor.row_bytes, first,
                               std::as_bytes(rows.first(staged)), *stream),
            timing);
        ++timing.pieces;
        timing.bytes += staged;
        staged = 0;
        CheckStop(stop);
      };
      auto read = [&](bool private_file, const DiskPayload& payload,
                      bool private_state) {
        if (private_state || payload.bytes > rows.size() - staged)
          flush();
        const bool collect = !private_state && payload.bytes <= rows.size();
        // Validate each file independently, while gathering adjacent native
        // rows in the already admitted staging allocation. Large files keep
        // the bounded streaming path. Never reuse staging before its transfer
        // completes, or execute state before all dependencies validate.
        auto piece = private_state ? impl_->staging.bytes
                     : collect     ? rows.subspan(staged, payload.bytes)
                                   : rows;
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
                if (collect)
                  return;
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
        if (!private_state) {
          if (collect)
            staged += payload.bytes;
          first += payload.bytes / c.descriptor.row_bytes;
          if (staged == rows.size())
            flush();
        }
      };
      for (const auto& chunk : c.chunks) {
        CheckStop(stop);
        prefetch.Reading(reading++);
        read(false, chunk, false);
      }
      if (c.tail) {
        prefetch.Reading(reading++);
        read(true, *c.tail, false);
      }
      if (c.private_state) {
        prefetch.Reading(reading++);
        read(true, *c.private_state, true);
      }
      flush();
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
      pinned_bytes_ = Add(pinned_bytes_, source.Capacity());
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
  std::function<void(CheckpointId, bool)> completed;
  std::function<bool(const DiskManifest&)> reclaim;
  std::chrono::milliseconds yield_limit;
  mutable std::mutex mutex;
  std::mutex stop_mutex;
  std::condition_variable wake;
  std::vector<std::unique_ptr<PersistenceJob>> pending;
  std::unique_ptr<PersistenceJob> active;
  PersistenceQueueStats stats;
  bool stopped{};
  std::uint64_t sequence{};
  std::jthread worker;
  Impl(ResourceReservation reservation, StreamedStore& streams,
       std::size_t depth, std::size_t bytes, std::function<bool()> model_idle,
       std::function<void(CheckpointId, bool)> on_completed,
       std::function<bool(const DiskManifest&)> on_reclaim,
       std::chrono::milliseconds limit)
      : metadata(reservation.Convert()),
        store(&streams),
        max_bytes(bytes),
        idle(std::move(model_idle)),
        completed(std::move(on_completed)),
        reclaim(std::move(on_reclaim)),
        yield_limit(limit),
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
          auto it = std::min_element(
              pending.begin(), pending.end(), [](const auto& a, const auto& b) {
                if (!a)
                  return false;
                return !b || a->submitted_ < b->submitted_;
              });
          active = std::move(*it);
        }
        bool succeeded{};
        TransferTiming timing;
        // Yield to the model only for a bounded time. Under continuous load
        // the model is never idle, and a job that never starts keeps its
        // sources pinned, so later captures fail for want of RAM.
        const auto deadline = Clock::now() + yield_limit;
        const std::function<bool()> idle_or_late = [&] {
          return Clock::now() >= deadline || idle();
        };
        while (!stop.stop_requested()) {
          try {
            store->Write(active->file_, active->manifest_, active->sources_,
                         timing, stop, idle_or_late);
            succeeded = true;
            break;
          } catch (const DiskCapacityExhausted&) {
            try {
              if (reclaim && reclaim(active->manifest_))
                continue;
            } catch (...) {
            }
            break;
          } catch (...) {
            // Non-capacity failures must never evict another checkpoint.
            break;
          }
        }
        if (completed) {
          try {
            completed(active->manifest_.checkpoint, succeeded);
          } catch (...) {
            // Publication remains durable and discoverable at restart even
            // if optional serving metadata cannot be admitted now.
          }
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
PersistenceQueue::PersistenceQueue(
    ResourceLedger& ledger, StreamedStore& store, std::size_t depth,
    std::size_t bytes, std::function<bool()> model_idle,
    std::function<void(CheckpointId, bool)> completed,
    std::function<bool(const DiskManifest&)> reclaim,
    std::chrono::milliseconds yield_limit)
    : impl_(std::make_unique<Impl>(
          ledger.Reserve(ResourceCategory::kMetadata,
                         Add(sizeof(Impl), Multiply(depth, sizeof(void*)))),
          store, depth, bytes, std::move(model_idle), std::move(completed),
          std::move(reclaim), yield_limit)) {}
PersistenceQueue::~PersistenceQueue() {
  Stop();
}
bool PersistenceQueue::TrySubmit(PersistenceJob job) {
  std::unique_lock guard(impl_->mutex);
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
  auto slot = std::find_if(impl_->pending.begin(), impl_->pending.end(),
                           [](auto& item) { return !item; });
  auto replacement = std::find_if(
      impl_->pending.begin(), impl_->pending.end(), [&](const auto& other) {
        if (!other || other->manifest_.purpose == CheckpointPurpose::kLearned ||
            !Extends(job.manifest_, other->manifest_))
          return false;
        // Keep the first pending boundary until its write starts. A busy
        // writer may belong to another conversation; replacing this anchor
        // would leave only a deep checkpoint after graceful shutdown.
        return (impl_->active &&
                Extends(other->manifest_, impl_->active->manifest_)) ||
               std::any_of(impl_->pending.begin(), impl_->pending.end(),
                           [&](const auto& earlier) {
                             return earlier && earlier.get() != other.get() &&
                                    Extends(other->manifest_,
                                            earlier->manifest_);
                           });
      });
  const bool redundant = replacement != impl_->pending.end();
  if (!redundant &&
      (impl_->stats.pending == impl_->pending.size() ||
       job.pinned_bytes_ > impl_->max_bytes - impl_->stats.pinned_bytes)) {
    for (auto it = impl_->pending.begin(); it != impl_->pending.end(); ++it) {
      if (!*it || ((*it)->manifest_.purpose == CheckpointPurpose::kLearned &&
                   job.manifest_.purpose != CheckpointPurpose::kLearned))
        continue;
      if (replacement == impl_->pending.end() ||
          (*it)->submitted_ < (*replacement)->submitted_)
        replacement = it;
    }
  }
  const auto released =
      replacement == impl_->pending.end() ? 0 : (*replacement)->pinned_bytes_;
  if ((impl_->stats.pending == impl_->pending.size() && !released) ||
      job.pinned_bytes_ >
          impl_->max_bytes - (impl_->stats.pinned_bytes - released)) {
    ++impl_->stats.skipped;
    return false;
  }
  std::optional<CheckpointId> superseded;
  bool accepted = false;
  try {
    // Allocate job metadata before discarding anything. Release the old pins
    // before admitting new pins, including on a subsequent admission failure.
    auto next = std::make_unique<PersistenceJob>(std::move(job));
    if (replacement != impl_->pending.end()) {
      superseded = (*replacement)->manifest_.checkpoint;
      impl_->stats.pinned_bytes -= released;
      --impl_->stats.pending;
      ++impl_->stats.superseded;
      if (redundant)
        ++impl_->stats.coalesced;
      replacement->reset();
      slot = replacement;
    }
    for (auto& source : next->sources_)
      source.pin_ = source.Pin();
    next->submitted_ = ++impl_->sequence;
    *slot = std::move(next);
    ++impl_->stats.accepted;
    ++impl_->stats.pending;
    impl_->stats.pinned_bytes += (*slot)->pinned_bytes_;
    impl_->stats.peak_pending =
        std::max(impl_->stats.peak_pending, impl_->stats.pending);
    impl_->stats.peak_pinned_bytes =
        std::max(impl_->stats.peak_pinned_bytes, impl_->stats.pinned_bytes);
    accepted = true;
  } catch (const std::bad_alloc&) {
    ++impl_->stats.skipped;
  }
  guard.unlock();
  if (superseded && impl_->completed) {
    try {
      impl_->completed(*superseded, false);
    } catch (...) {
    }
  }
  impl_->wake.notify_all();
  return accepted;
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
