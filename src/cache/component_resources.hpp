#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <span>

#include "src/cache/adapter.hpp"
#include "src/cache/checkpoint.hpp"
#include "src/cache/streaming.hpp"

namespace gufo::cache {

// Neutral provider contract for one component-cache capture. The model supplies
// physical storage and binds its ordinary execution facade to adapter-owned
// slots; queueing, retention and admission policy stay in the serving layer.
struct ComponentCaptureBuffer {
  std::span<std::byte> bytes;
  std::function<Payload()> finish;
  // Identity of one underlying allocation, for joining adjacent byte spans.
  // Null providers always transfer each buffer separately.
  const void* allocation{};
  // Consume assigned backing without copying completed append rows. The
  // source view is backend-native and kept immutable by the slot guard.
  std::function<Payload(SlotLease&, std::span<const std::byte>)> borrow;
};
struct ComponentCacheResources {
  std::shared_ptr<void> owner;
  std::unique_ptr<Adapter> adapter;
  StreamLease stream;
  std::function<ComponentCaptureBuffer(const PayloadRequest&)> allocate;
  StagingAllocator staging;
  // Prepare on-demand backing immediately before the next prefill chunk.
  std::function<void(std::size_t)> prepare_prefill;
  // Optional capacity bound before assigning a complete capture. False takes
  // the ordinary reclamation path; true still requires every allocation.
  std::function<bool(std::span<const PayloadRequest>)> can_capture;
  // Optional initialization step after slots allocate. The provider leaves
  // payload backing uncommitted until serving selects the post-slot budget.
  std::function<void(std::size_t)> commit_backing;
  // Optional host bytes the slots pinned and charged to the ledger as they
  // allocated. Serving charges them on top of the cache's budget, and adds
  // them back to the runner's claim, which measures the host after them.
  std::function<std::size_t()> pinned_bytes;
  PersistenceSource::CopyRows copy_observed_rows;
  // At most this many transfers of one capture/restore are in flight before
  // waiting (16 at most). When the shared pool is momentarily exhausted, the
  // batch settles its own leases and retries. One keeps generic providers
  // usable without a pool-capacity contract.
  std::size_t transfer_batch_size{1};
  std::size_t reassignment_batch_size{1};
};
}  // namespace gufo::cache
