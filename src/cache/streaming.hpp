#ifndef GUFO_CACHE_STREAMING_HPP_
#define GUFO_CACHE_STREAMING_HPP_

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <stop_token>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/cache/disk.hpp"

namespace gufo::cache {
class BorrowedRows;
inline constexpr std::size_t kDefaultDiskStagingBytes = 1ULL << 20;
inline constexpr std::chrono::milliseconds kPersistenceYieldLimit{500};
// Restores and observed-row captures move whole rows per staging piece; a
// smaller staging budget can never restore a checkpoint of these components.
[[nodiscard]] inline std::size_t MinimumStagingBytes(
    std::span<const ComponentDescriptor> components) {
  std::size_t bytes = 1;
  for (const auto& d : components)
    if (HasRows(d.kind))
      bytes = std::max(bytes, d.row_bytes);
  return bytes;
}
struct StagingAllocation {
  std::shared_ptr<void> owner;
  std::span<std::uint8_t> bytes;
};
using StagingAllocator = std::function<StagingAllocation(std::size_t)>;
using StreamLease = std::function<std::unique_ptr<Stream>()>;

struct TransferTiming {
  std::uint64_t allocator_ns{}, stream_acquire_ns{}, stream_wait_ns{};
  std::uint64_t metadata_lock_ns{}, filesystem_ns{}, fsync_ns{}, checksum_ns{};
  std::uint64_t yield_ns{}, bytes{}, pieces{};
  std::uint64_t staging_wait_ns{}, io_lock_ns{};
};
// Sources retain immutable storage, assigned spill backing or discardable
// observations. Backed borrowers wait for preservation before disk work
// unless they read live pieces; live reads and observations pin one native
// piece until its leased transfer completes.
// Capture supplies private state/tails before submitting a job to the queue.
class PersistenceSource {
public:
  using Copy =
      std::function<Completion(std::uint64_t, std::span<std::byte>, Stream&)>;
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, ResourceCharge,
                    std::shared_ptr<const void> owner, Copy);
  using CopyRows = std::function<Completion(std::span<const std::byte>,
                                            std::span<std::byte>, Stream&)>;
  // Backed borrowed rows. Without read_live, the writer waits until slot
  // preservation has copied them to their backing. With it, a piece still in
  // the slot is read from there under a reader pin held through completion,
  // so a busy or reused slot cannot hold the job (and its pins) back.
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, std::shared_ptr<BorrowedRows>,
                    CopyRows read_live = {});
  // Discardable live rows, for histories exceeding RAM. Reads pin exactly
  // one staging piece through completion; mutation retires unread sources.
  // Private state/tails remain immutable committed copies. No publication
  // completes with unread retired rows. No full-history backing is assigned.
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, std::shared_ptr<BorrowedRows>,
                    std::size_t first_byte, CopyRows);
  PersistenceSource(PersistenceSource&&) noexcept = default;
  PersistenceSource& operator=(PersistenceSource&&) = delete;
  PersistenceSource(const PersistenceSource&) = delete;

private:
  friend class StreamedStore;
  friend class PersistenceJob;
  friend class PersistenceQueue;
  [[nodiscard]] PersistencePin Pin() const;
  [[nodiscard]] std::size_t Capacity() const;
  ResourceCharge charge_;
  PersistencePin pin_;
  std::shared_ptr<const void> owner_;
  std::shared_ptr<BorrowedRows> rows_;
  Copy copy_;
  bool live_reads_{};
  bool private_file_;
  DiskFileId file_;
  std::uint64_t bytes_;
  std::size_t alignment_;
};
// One initialization-time allocation, leased by piece to writers and held by
// foreground restores. Optional yielding never holds staging or a stream. A HIP
// allocator supplies committed pinned storage outside this CPU-only package.
// The disk store, ledger and stream provider must outlive this object.
class StreamedStore {
public:
  StreamedStore(ResourceLedger&, DiskStore&, std::size_t staging_bytes,
                StagingAllocator, StreamLease);
  StreamedStore(ResourceLedger&, DiskStore&, StagingAllocator, StreamLease);
  ~StreamedStore();
  StreamedStore(const StreamedStore&) = delete;
  StreamedStore& operator=(const StreamedStore&) = delete;
  void Write(DiskFileId, const DiskManifest&, std::span<PersistenceSource>,
             TransferTiming&, std::stop_token = {},
             const std::function<bool()>& model_idle = {});
  // Invalidation happens before fallback. Cancellation also invalidates, but
  // does not run fallback. All loads and disk pins settle before either path.
  // dependency_failed runs after invalidation and before fallback, only for
  // filesystem/checksum failures; it can quarantine reverse dependencies.
  [[nodiscard]] bool Restore(
      CheckpointId, CompatibilityDigest expected_compatibility, Adapter&, Slot&,
      TransferTiming&, std::stop_token = {},
      const std::function<void()>& fallback = {},
      const std::function<void(bool, DiskFileId)>& dependency_failed = {});
  [[nodiscard]] std::size_t StagingBytes() const;
  [[nodiscard]] std::uint64_t StartupAllocatorNanoseconds() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class PersistenceJob {
public:
  PersistenceJob(ResourceLedger&, DiskFileId, const DiskManifest&,
                 std::vector<PersistenceSource>);
  PersistenceJob(PersistenceJob&&) noexcept = default;
  PersistenceJob& operator=(PersistenceJob&&) = delete;
  PersistenceJob(const PersistenceJob&) = delete;

private:
  friend class PersistenceQueue;
  ResourceCharge metadata_;
  DiskFileId file_;
  DiskManifest manifest_;
  std::vector<PersistenceSource> sources_;
  std::size_t pinned_bytes_{};
  std::uint64_t submitted_{};
};
struct PersistenceQueueStats {
  std::uint64_t accepted{}, skipped{}, coalesced{}, superseded{}, completed{},
      failed{};
  std::size_t pending{}, pinned_bytes{}, peak_pending{}, peak_pinned_bytes{};
  TransferTiming timing;
};
// One optional worker. Capacity counts the active job as well as pending jobs;
// TrySubmit never waits for capacity. Stop cancels active/pending work and
// joins. Callbacks must be thread-safe, nonblocking and outlive the queue.
// Model idle is checked between pieces, without holding metadata or
// device-source locks. A job yields for at most yield_limit after its write
// starts, so continuous load cannot hold its pins indefinitely.
// Keep the first pending boundary and coalesce later pending ancestors into
// their newest descendant. Under pressure, recent jobs
// may replace the oldest pending non-learned job; active writes are untouched.
// Completion reports false for superseded work, outside the queue lock.
class PersistenceQueue {
public:
  PersistenceQueue(
      ResourceLedger&, StreamedStore&, std::size_t max_depth,
      std::size_t max_pinned_bytes, std::function<bool()> model_idle,
      std::function<void(CheckpointId, bool)> completed = {},
      std::function<bool(const DiskManifest&)> reclaim = {},
      std::chrono::milliseconds yield_limit = kPersistenceYieldLimit);
  ~PersistenceQueue();
  PersistenceQueue(const PersistenceQueue&) = delete;
  PersistenceQueue& operator=(const PersistenceQueue&) = delete;
  bool TrySubmit(PersistenceJob);
  void Drain();
  void Stop();
  [[nodiscard]] PersistenceQueueStats Stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gufo::cache
#endif
