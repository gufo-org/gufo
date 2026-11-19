#include "src/cache/streaming.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/cache/prefetch_cursor.hpp"
#include "src/cache/slot.hpp"
#include "src/core/platform/fs.hpp"

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
// helpers and about 6 GB/s for kDiskReadsInFlight (measured). Helpers hint
// whole files up to a bounded byte window ahead of the readers, which still
// read and check every file.
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
    // Without an OS hint (Windows), helpers would only open and stat files.
    if constexpr (platform::kAdviseReachesOs)
      for (auto& thread : threads_)
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
  }
  // The reader is about to read item `index`.
  void Reading(std::size_t index) {
    {
      const std::lock_guard lock(mutex_);
      cursor_.Reading(index);
    }
    wake_.notify_all();
  }

private:
  static constexpr std::uint64_t kWindowBytes = 256 << 20;
  void Run(std::stop_token stop) {
    while (true) {
      std::size_t i;
      {
        std::unique_lock lock(mutex_);
        if (!wake_.wait(lock, stop,
                        [&] { return cursor_.Ready(offsets_, kWindowBytes); }))
          return;
        i = cursor_.Take();
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
  detail::PrefetchCursor cursor_;
  std::array<std::jthread, kDiskReadsInFlight> threads_;
};
// How a restore divides staging: `lanes` pieces of `bytes` that readers fill
// in turn. Staging too small for two pieces, or rows wider than one, keeps a
// single piece read and copied one at a time on the restoring thread.
struct StagingPieces {
  std::size_t bytes{}, lanes{1}, readers{};
  [[nodiscard]] std::span<std::uint8_t> Lane(std::span<std::uint8_t> staging,
                                             std::size_t unit) const {
    return staging.subspan(unit % lanes * bytes, bytes);
  }
};
StagingPieces DivideStaging(std::size_t staging, std::size_t piece,
                            std::span<const ComponentDescriptor> components) {
  if (staging < 2 * piece || MinimumStagingBytes(components) > piece)
    return {staging, 1, 0};
  const auto lanes = staging / piece;
  return {piece, lanes, std::min(lanes, kDiskRestoreReaders)};
}
// A restore copies units of at most one staging piece: a group of adjacent
// small row files, or one piece of a larger file. Readers read items (a
// group, or a whole larger file, piece by piece) into lanes, unit u into lane
// u % lanes once the copy out of that lane for unit u - lanes has settled, so
// a piece is never rewritten while a transfer reads it. Only the restoring
// thread calls the adapter or waits for copies; readers touch files and
// staging alone. Without readers, the restoring thread reads every item
// itself and submits each unit as soon as it is read, one piece at a time.
class RestorePipeline {
public:
  using ReadItem = std::function<void(std::size_t item)>;
  using SubmitUnit = std::function<void(std::size_t unit)>;
  RestorePipeline(std::size_t units, std::size_t items,
                  const StagingPieces& pieces, TransferTiming& timing)
      : state_(units),
        next_lane_(pieces.lanes),
        copies_(pieces.lanes),
        items_(items),
        threads_(std::min(pieces.readers, items)),
        timing_(timing) {
    for (std::size_t i = 0; i < next_lane_.size(); ++i)
      next_lane_[i] = i;
  }
  ~RestorePipeline() { Stop(); }
  RestorePipeline(const RestorePipeline&) = delete;
  RestorePipeline& operator=(const RestorePipeline&) = delete;
  [[nodiscard]] bool Threaded() const noexcept { return threads_ != 0; }
  // Starts the readers, or with none reads every item here, submitting each
  // unit as it is read.
  void Run(ReadItem read, SubmitUnit submit) {
    read_ = std::move(read);
    submit_ = std::move(submit);
    if (!Threaded()) {
      for (std::size_t item = 0; item < items_; ++item)
        read_(item);
      return;
    }
    workers_.reserve(threads_);
    try {
      for (std::size_t i = 0; i < threads_; ++i)
        workers_.emplace_back([this] {
          while (true) {
            std::size_t item;
            {
              const std::lock_guard lock(mutex_);
              if (abort_ || next_item_ >= items_)
                return;
              item = next_item_++;
            }
            try {
              read_(item);
            } catch (...) {
              {
                const std::lock_guard lock(mutex_);
                if (!error_)
                  error_ = std::current_exception();
                abort_ = true;
              }
              wake_.notify_all();
              return;
            }
          }
        });
    } catch (...) {
      Stop();
      throw;
    }
  }
  // Reader side: blocks until unit's lane is free.
  void WaitLane(std::size_t unit) {
    const auto lane = unit % next_lane_.size();
    if (!Threaded()) {
      while (next_lane_[lane] != unit)
        SettleOldest();
      return;
    }
    std::unique_lock lock(mutex_);
    wake_.wait(lock, [&] { return abort_ || next_lane_[lane] == unit; });
    if (abort_)
      throw Cancelled();
  }
  // Reader side: unit's lane holds its bytes.
  void Read(std::size_t unit) {
    if (!Threaded()) {
      submit_(unit);
      return;
    }
    {
      const std::lock_guard lock(mutex_);
      state_[unit] = 1;
    }
    wake_.notify_all();
  }
  // Restoring thread: settles finished copies, freeing their lanes for
  // readers, until `unit` is read. Returns the first error when it cannot be.
  [[nodiscard]] std::exception_ptr WaitRead(std::size_t unit) {
    while (true) {
      {
        std::unique_lock lock(mutex_);
        if (outstanding_.empty())
          wake_.wait(lock, [&] { return state_[unit] || abort_; });
        if (state_[unit])
          return nullptr;
        if (abort_)
          return error_ ? error_ : std::make_exception_ptr(Cancelled());
      }
      // Copies finish on their own; waiting for the oldest always advances.
      SettleOldest();
    }
  }
  // Restoring thread: unit's copy reads its lane until it settles.
  void Submitted(std::size_t unit, std::unique_ptr<Stream> stream,
                 Completion copy) {
    auto& lane = copies_[unit % copies_.size()];
    lane.stream = std::move(stream);
    lane.copy.emplace(std::move(copy));
    outstanding_.push_back(unit);
  }
  void Observe(const DiskReadStats& costs) noexcept {
    filesystem_ns_ += costs.filesystem_ns;
    checksum_ns_ += costs.checksum_ns;
  }
  void Failed(bool private_file, DiskFileId file) {
    const std::lock_guard lock(mutex_);
    if (!failed_)
      failed_.emplace(private_file, file);
  }
  [[nodiscard]] std::optional<std::pair<bool, DiskFileId>> FailedFile() {
    const std::lock_guard lock(mutex_);
    return failed_;
  }
  // Success: every unit was submitted. Joins the readers and settles the
  // last copies, which may still fail.
  void Finish() {
    Stop();
    while (!outstanding_.empty())
      SettleOldest();
  }
  // Every exit, including failures: no reader runs and no copy reads
  // staging once this returns. Adds the readers' costs to the timing.
  void Drain() noexcept {
    Stop();
    for (const auto unit : outstanding_) {
      auto& lane = copies_[unit % copies_.size()];
      const auto start = Clock::now();
      (void)lane.copy->Wait();
      timing_.stream_wait_ns += Elapsed(start);
      lane.copy.reset();
      lane.stream.reset();
    }
    outstanding_.clear();
    timing_.filesystem_ns += filesystem_ns_.exchange(0);
    timing_.checksum_ns += checksum_ns_.exchange(0);
  }

private:
  struct Copy {
    std::unique_ptr<Stream> stream;
    std::optional<Completion> copy;
  };
  void SettleOldest() {
    if (outstanding_.empty())
      throw std::logic_error("restore lane has no copy to settle");
    const auto unit = outstanding_.front();
    outstanding_.pop_front();
    auto& lane = copies_[unit % copies_.size()];
    auto copy = std::move(*lane.copy);
    lane.copy.reset();
    // The stream returns to its pool only once the copy has settled.
    auto stream = std::move(lane.stream);
    Settle(std::move(copy), timing_);
    {
      const std::lock_guard lock(mutex_);
      next_lane_[unit % next_lane_.size()] = unit + next_lane_.size();
    }
    wake_.notify_all();
  }
  void Stop() noexcept {
    {
      const std::lock_guard lock(mutex_);
      abort_ = true;
    }
    wake_.notify_all();
    for (auto& worker : workers_)
      if (worker.joinable())
        worker.join();
  }
  std::mutex mutex_;
  std::condition_variable wake_;
  // Guarded by mutex_ when readers run: per-unit read state, the next unit
  // each lane may take, the item cursor and the first failure.
  std::vector<std::uint8_t> state_;
  std::vector<std::size_t> next_lane_;
  std::size_t next_item_{};
  bool abort_{};
  std::exception_ptr error_;
  std::optional<std::pair<bool, DiskFileId>> failed_;
  // Restoring thread only.
  std::vector<Copy> copies_;
  std::deque<std::size_t> outstanding_;
  std::size_t items_, threads_;
  TransferTiming& timing_;
  ReadItem read_;
  SubmitUnit submit_;
  std::vector<std::thread> workers_;
  std::atomic<std::uint64_t> filesystem_ns_{}, checksum_ns_{};
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
         (next.input == prior.input || next.lineage == prior.lineage) &&
         prior.tokens.size() < next.tokens.size() &&
         std::equal(prior.tokens.begin(), prior.tokens.end(),
                    next.tokens.begin());
}
thread_local bool copying_beside_decode = false;
}  // namespace
BesideDecodeScope::BesideDecodeScope() noexcept
    : previous_(copying_beside_decode) {
  copying_beside_decode = true;
}
BesideDecodeScope::~BesideDecodeScope() {
  copying_beside_decode = previous_;
}
bool CopyingBesideDecode() noexcept {
  return copying_beside_decode;
}

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
  StreamLease lease, write_lease, restore_lease;
  std::timed_mutex mutex;
  std::uint64_t allocation_ns{};
  std::size_t piece_bytes;

  Impl(ResourceLedger& resources, DiskStore& store, std::size_t bytes,
       StagingAllocator allocator, StreamLease provider,
       ResourceReservation reservation, std::size_t piece)
      : ledger(&resources),
        disk(&store),
        metadata(reservation.Convert()),
        lease(std::move(provider)),
        piece_bytes(piece) {
    if (!bytes || !piece || !allocator || !lease)
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
  std::unique_ptr<Stream> Lease(TransferTiming& timing,
                                const StreamLease* preferred = nullptr) {
    const auto start = Clock::now();
    // A busy stream class falls back to the shared pool rather than wait.
    auto stream = preferred && *preferred ? (*preferred)() : nullptr;
    if (!stream)
      stream = lease();
    timing.stream_acquire_ns += Elapsed(start);
    if (!stream)
      throw ResourceExhausted();
    return stream;
  }
};
StreamedStore::StreamedStore(ResourceLedger& ledger, DiskStore& disk,
                             std::size_t bytes, StagingAllocator allocator,
                             StreamLease lease, std::size_t piece_bytes)
    : impl_(std::make_unique<Impl>(
          ledger, disk, bytes, std::move(allocator), std::move(lease),
          ledger.Reserve(ResourceCategory::kMetadata, sizeof(Impl)),
          piece_bytes)) {}
StreamedStore::~StreamedStore() = default;
void StreamedStore::SetStreamClasses(StreamLease write, StreamLease restore) {
  impl_->write_lease = std::move(write);
  impl_->restore_lease = std::move(restore);
}
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
                        auto stream = impl_->Lease(timing, &impl_->write_lease);
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
  // Writes take one piece at a time, as before restores had several.
  (void)impl_->disk->PublishStream(
      file, manifest, writes,
      impl_->staging.bytes.first(
          std::min(impl_->staging.bytes.size(), impl_->piece_bytes)),
      &costs, &access, stop);
}
bool StreamedStore::Restore(
    CheckpointId id, CompatibilityDigest compatibility, Adapter& adapter,
    Slot& slot, TransferTiming& timing, std::stop_token stop,
    const std::function<void()>& fallback,
    const std::function<void(bool, DiskFileId)>& dependency_failed,
    const RestoreCopyRunner& run_copies) {
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
    // Plan the copies first, exactly as one staging piece would gather them:
    // adjacent row files that fit a piece form one unit; a larger row file or
    // private state is copied in units of one piece.
    const auto pieces = DivideStaging(impl_->staging.bytes.size(),
                                      impl_->piece_bytes, adapter.Components());
    struct File {
      bool private_file;
      const DiskPayload* payload;
      std::size_t prefetch;
    };
    struct Item {
      std::vector<File> files;
      std::size_t unit{}, capacity{};
      // One file read in pieces of `capacity`, one unit each.
      bool pieces{};
    };
    struct Unit {
      std::size_t component;
      Rows first;
      std::uint64_t offset{};
      std::size_t bytes{};
      bool private_state{};
    };
    std::vector<Item> plan;
    std::vector<Unit> units;
    {
      std::size_t prefetch_index{};
      for (std::size_t ci = 0; ci < manifest.components.size(); ++ci) {
        const auto& c = manifest.components[ci];
        Rows first = FirstChunkRow(c.descriptor, c.position.valid_rows);
        const std::size_t capacity =
            HasRows(c.descriptor.kind)
                ? pieces.bytes / c.descriptor.row_bytes * c.descriptor.row_bytes
                : pieces.bytes;
        std::optional<std::size_t> open;
        const auto add = [&](bool private_file, const DiskPayload& payload,
                             bool private_state) {
          if (open &&
              (private_state ||
               payload.bytes > capacity - units[plan[*open].unit].bytes))
            open.reset();
          if (!private_state && payload.bytes <= capacity) {
            if (!open) {
              open = plan.size();
              plan.push_back({{}, units.size(), capacity, false});
              units.push_back({ci, first, 0, 0, false});
            }
            plan[*open].files.push_back(
                {private_file, &payload, prefetch_index});
            units[plan[*open].unit].bytes += payload.bytes;
            if (units[plan[*open].unit].bytes == capacity)
              open.reset();
          } else {
            const auto piece = private_state ? pieces.bytes : capacity;
            plan.push_back({{{private_file, &payload, prefetch_index}},
                            units.size(),
                            piece,
                            true});
            for (std::uint64_t offset = 0; offset < payload.bytes;
                 offset += piece)
              units.push_back({ci, first, offset,
                               static_cast<std::size_t>(std::min<std::uint64_t>(
                                   piece, payload.bytes - offset)),
                               private_state});
          }
          if (!private_state)
            first += payload.bytes / c.descriptor.row_bytes;
          ++prefetch_index;
        };
        for (const auto& chunk : c.chunks)
          add(false, chunk, false);
        if (c.tail)
          add(true, *c.tail, false);
        if (c.private_state)
          add(true, *c.private_state, true);
      }
    }
    // Readers read and check items into their own pieces while this thread
    // submits the copies in order: one reader read, checked and copied a
    // piece at a time, so a 2.47 GB restore spent 474 ms reading, 135 ms
    // checking and 161 ms waiting for copies one after another.
    const auto copies = [&] {
      RestorePipeline pipeline(units.size(), plan.size(), pieces, timing);
      Finally drain{[&] { pipeline.Drain(); }};
      const auto read_item = [&](std::size_t index) {
        const auto& item = plan[index];
        const auto observe = [&](const DiskReadStats& costs) {
          if (pipeline.Threaded()) {
            pipeline.Observe(costs);
          } else {
            timing.filesystem_ns += costs.filesystem_ns;
            timing.checksum_ns += costs.checksum_ns;
          }
        };
        std::size_t offset{};
        if (!item.pieces)
          pipeline.WaitLane(item.unit);
        for (const auto& file : item.files) {
          CheckStop(stop);
          prefetch.Reading(file.prefetch);
          DiskReadStats costs;
          Finally record{[&] { observe(costs); }};
          try {
            if (item.pieces) {
              (void)impl_->disk->ReadPayload(
                  *snapshot, file.private_file, *file.payload,
                  [&](std::uint64_t at) {
                    const auto unit = item.unit + at / item.capacity;
                    pipeline.WaitLane(unit);
                    return pieces.Lane(impl_->staging.bytes, unit)
                        .first(item.capacity);
                  },
                  [&](std::uint64_t at, auto) {
                    CheckStop(stop);
                    pipeline.Read(item.unit + at / item.capacity);
                  },
                  &costs);
            } else {
              (void)impl_->disk->ReadPayload(
                  *snapshot, file.private_file, *file.payload,
                  pieces.Lane(impl_->staging.bytes, item.unit)
                      .subspan(offset, file.payload->bytes),
                  [](std::uint64_t, auto) {}, &costs);
            }
          } catch (const DiskDependencyError& error) {
            pipeline.Failed(error.private_file, error.file);
            throw;
          }
          offset += file.payload->bytes;
        }
        if (!item.pieces)
          pipeline.Read(item.unit);
      };
      const auto submit = [&](std::size_t index) {
        const auto& unit = units[index];
        const auto& c = manifest.components[unit.component];
        const auto bytes = std::as_bytes(
            pieces.Lane(impl_->staging.bytes, index).first(unit.bytes));
        auto stream = impl_->Lease(timing, &impl_->restore_lease);
        auto completion =
            unit.private_state
                ? adapter.LoadPrivatePiece(slot, c.descriptor.id, unit.offset,
                                           bytes, *stream)
                : adapter.CopyRowsIn(
                      slot, c.descriptor.id,
                      unit.first + unit.offset / c.descriptor.row_bytes,
                      unit.first +
                          (unit.offset + unit.bytes) / c.descriptor.row_bytes,
                      bytes, *stream);
        pipeline.Submitted(index, std::move(stream), std::move(completion));
        ++timing.pieces;
        timing.bytes += unit.bytes;
      };
      try {
        pipeline.Run(read_item, submit);
      } catch (...) {
        if (const auto file = pipeline.FailedFile())
          failed = file;
        throw;
      }
      if (pipeline.Threaded())
        for (std::size_t index = 0; index < units.size(); ++index) {
          CheckStop(stop);
          if (const auto error = pipeline.WaitRead(index)) {
            failed = pipeline.FailedFile();
            std::rethrow_exception(error);
          }
          submit(index);
        }
      pipeline.Finish();
    };
    if (run_copies)
      run_copies(copies);
    else
      copies();
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
  // TrySubmit scratch, sized with pending so submission allocates nothing.
  std::vector<bool> chosen;
  std::unique_ptr<PersistenceJob> active;
  PersistenceQueueStats stats;
  bool stopped{};
  std::size_t discard_callbacks{};
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
        pending(depth),
        chosen(depth) {
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
                         Add(sizeof(Impl), Multiply(depth, sizeof(void*) + 1))),
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
  // Exact frontiers are optional write-through work. They must not displace
  // an unrelated conversation's pending prompt anchor when the queue is full,
  // nor may they consume more than half of the queue's pinned byte budget.
  const std::size_t frontier_budget = impl_->max_bytes / 2;
  const bool full =
      impl_->stats.pending == impl_->pending.size() ||
      job.pinned_bytes_ > impl_->max_bytes - impl_->stats.pinned_bytes ||
      (job.manifest_.purpose == CheckpointPurpose::kGenerated &&
       job.pinned_bytes_ >
           frontier_budget -
               std::min(frontier_budget, impl_->stats.pinned_bytes));
  const bool generated = job.manifest_.purpose == CheckpointPurpose::kGenerated;
  // Slots this job coalesces or displaces. A larger turn end can need more
  // than one covered turn end to make room, so a full queue keeps coalescing
  // until it fits instead of skipping the newest turn end.
  auto& chosen = impl_->chosen;
  std::fill(chosen.begin(), chosen.end(), false);
  const auto present = [&](const auto& item) {
    return item && !chosen[&item - impl_->pending.data()];
  };
  const auto covered = [&](const auto& other) {
    if (!present(other) ||
        other->manifest_.purpose == CheckpointPurpose::kLearned)
      return false;
    const bool extended = Extends(job.manifest_, other->manifest_);
    // A generated frontier must not coalesce away the prompt checkpoint
    // that an edit or a retry of this request needs. Likewise preserve
    // two recent generated frontiers before treating them as redundant.
    if (other->manifest_.purpose != CheckpointPurpose::kGenerated &&
        (!extended || job.manifest_.purpose == CheckpointPurpose::kGenerated))
      return false;
    // Each generated frontier is a turn end that an edit or a branch
    // restores, which no descendant covers. Coalesce one only when the
    // queue has no room for this job (#1250), so whether a turn end
    // reaches disk does not depend on how far behind the writer is. Any
    // conversation's turn end with two newer queued ones qualifies, so
    // one conversation's backlog cannot crowd out another's newest one.
    if (other->manifest_.purpose == CheckpointPurpose::kGenerated) {
      if (!full)
        return false;
      std::size_t newer = extended && generated;
      for (const auto& peer : impl_->pending)
        newer += present(peer) && &peer != &other &&
                 peer->manifest_.purpose == CheckpointPurpose::kGenerated &&
                 Extends(peer->manifest_, other->manifest_);
      newer +=
          impl_->active &&
          impl_->active->manifest_.purpose == CheckpointPurpose::kGenerated &&
          Extends(impl_->active->manifest_, other->manifest_);
      if (newer < 2)
        return false;
    }
    // Keep the first pending boundary until its write starts. A busy
    // writer may belong to another conversation; replacing this anchor
    // would leave only a deep checkpoint after graceful shutdown.
    return (impl_->active &&
            Extends(other->manifest_, impl_->active->manifest_)) ||
           std::any_of(impl_->pending.begin(), impl_->pending.end(),
                       [&](const auto& earlier) {
                         return present(earlier) && &earlier != &other &&
                                Extends(other->manifest_, earlier->manifest_);
                       });
  };
  std::size_t released = 0, displaced = 0;
  const auto fits = [&] {
    const auto pinned = impl_->stats.pinned_bytes - released;
    return impl_->stats.pending - displaced < impl_->pending.size() &&
           job.pinned_bytes_ <= impl_->max_bytes - pinned &&
           (!generated ||
            job.pinned_bytes_ <=
                frontier_budget - std::min(frontier_budget, pinned));
  };
  auto replacement = impl_->pending.end();
  for (auto it =
           std::find_if(impl_->pending.begin(), impl_->pending.end(), covered);
       it != impl_->pending.end();
       it = std::find_if(impl_->pending.begin(), impl_->pending.end(),
                         covered)) {
    if (replacement == impl_->pending.end())
      replacement = it;
    chosen[it - impl_->pending.begin()] = true;
    released += (*it)->pinned_bytes_;
    ++displaced;
    if (!generated || !full || fits())
      break;
  }
  const bool redundant = replacement != impl_->pending.end();
  if (!redundant && full && generated) {
    ++impl_->stats.skipped;
    return false;
  }
  if (!redundant &&
      (impl_->stats.pending == impl_->pending.size() ||
       job.pinned_bytes_ > impl_->max_bytes - impl_->stats.pinned_bytes)) {
    for (auto it = impl_->pending.begin(); it != impl_->pending.end(); ++it) {
      if (!*it || ((*it)->manifest_.purpose == CheckpointPurpose::kLearned &&
                   job.manifest_.purpose != CheckpointPurpose::kLearned))
        continue;
      if (replacement == impl_->pending.end() ||
          ((*it)->manifest_.purpose == CheckpointPurpose::kGenerated &&
           (*replacement)->manifest_.purpose !=
               CheckpointPurpose::kGenerated) ||
          (((*it)->manifest_.purpose == CheckpointPurpose::kGenerated) ==
               ((*replacement)->manifest_.purpose ==
                CheckpointPurpose::kGenerated) &&
           (*it)->submitted_ < (*replacement)->submitted_))
        replacement = it;
    }
    if (replacement != impl_->pending.end()) {
      chosen[replacement - impl_->pending.begin()] = true;
      released = (*replacement)->pinned_bytes_;
      displaced = 1;
    }
  }
  if (!fits()) {
    ++impl_->stats.skipped;
    return false;
  }
  std::vector<CheckpointId> superseded;
  bool accepted = false;
  try {
    // Allocate job metadata before discarding anything. Release the old pins
    // before admitting new pins, including on a subsequent admission failure.
    auto next = std::make_unique<PersistenceJob>(std::move(job));
    superseded.reserve(displaced);
    for (std::size_t i = 0; i < chosen.size(); ++i) {
      auto& item = impl_->pending[i];
      if (!chosen[i] || !item)
        continue;
      superseded.push_back(item->manifest_.checkpoint);
      impl_->stats.pinned_bytes -= item->pinned_bytes_;
      --impl_->stats.pending;
      ++impl_->stats.superseded;
      if (redundant)
        ++impl_->stats.coalesced;
      item.reset();
    }
    if (replacement != impl_->pending.end())
      slot = replacement;
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
  if (impl_->completed)
    for (const auto id : superseded) {
      try {
        impl_->completed(id, false);
      } catch (...) {
      }
    }
  impl_->wake.notify_all();
  return accepted;
}
bool PersistenceQueue::DiscardPending(CheckpointId id) {
  std::unique_ptr<PersistenceJob> discarded;
  {
    std::lock_guard guard(impl_->mutex);
    for (auto& job : impl_->pending) {
      if (!job || job->manifest_.checkpoint != id)
        continue;
      impl_->stats.pinned_bytes -= job->pinned_bytes_;
      --impl_->stats.pending;
      ++impl_->stats.superseded;
      ++impl_->discard_callbacks;
      discarded = std::move(job);
      break;
    }
  }
  if (!discarded)
    return false;
  discarded.reset();
  if (impl_->completed) {
    try {
      impl_->completed(id, false);
    } catch (...) {
    }
  }
  {
    std::lock_guard guard(impl_->mutex);
    --impl_->discard_callbacks;
  }
  impl_->wake.notify_all();
  return true;
}
void PersistenceQueue::Drain() {
  std::unique_lock guard(impl_->mutex);
  impl_->wake.wait(guard, [&] {
    return impl_->stats.pending == 0 && impl_->discard_callbacks == 0;
  });
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
