#ifndef GUFO_CACHE_STREAMING_HPP_
#define GUFO_CACHE_STREAMING_HPP_

#include <functional>
#include <memory>
#include <span>
#include <stop_token>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/cache/disk.hpp"

namespace gufo::cache {
inline constexpr std::size_t kDefaultDiskStagingBytes = 1ULL << 20;
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
// The owner and charge describe committed immutable source storage, not a live
// execution slot. A borrowed source must have completed preservation first.
// Copy settles via Completion on a leased stream. Its callback and owner must
// retain all source-specific metadata; no capture occurs in the queue worker.
class PersistenceSource {
public:
  using Copy =
      std::function<Completion(std::uint64_t, std::span<std::byte>, Stream&)>;
  PersistenceSource(bool private_file, DiskFileId, std::uint64_t bytes,
                    std::size_t alignment, ResourceCharge,
                    std::shared_ptr<const void> owner, Copy);
  PersistenceSource(PersistenceSource&&) noexcept = default;
  PersistenceSource& operator=(PersistenceSource&&) = delete;
  PersistenceSource(const PersistenceSource&) = delete;

private:
  friend class StreamedStore;
  friend class PersistenceJob;
  friend class PersistenceQueue;
  ResourceCharge charge_;
  PersistencePin pin_;
  std::shared_ptr<const void> owner_;
  Copy copy_;
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
  [[nodiscard]] bool Restore(CheckpointId,
                             CompatibilityDigest expected_compatibility,
                             Adapter&, Slot&, TransferTiming&,
                             std::stop_token = {},
                             const std::function<void()>& fallback = {});
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
};
struct PersistenceQueueStats {
  std::uint64_t accepted{}, skipped{}, coalesced{}, completed{}, failed{};
  std::size_t pending{}, pinned_bytes{}, peak_pending{}, peak_pinned_bytes{};
  TransferTiming timing;
};
// One optional worker. Capacity counts the active job as well as pending jobs;
// TrySubmit never waits for capacity. Stop cancels active/pending work and
// joins. Callbacks must be thread-safe, nonblocking and outlive the queue.
// Model idle is checked between pieces, without holding metadata or
// device-source locks.
class PersistenceQueue {
public:
  PersistenceQueue(ResourceLedger&, StreamedStore&, std::size_t max_depth,
                   std::size_t max_pinned_bytes,
                   std::function<bool()> model_idle);
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
