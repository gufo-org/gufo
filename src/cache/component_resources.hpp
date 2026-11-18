#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include "src/cache/adapter.hpp"
#include "src/cache/checkpoint.hpp"
#include "src/cache/streaming.hpp"

namespace gufo::cache {

// Backing ComponentCacheResources::commit_backing commits before readiness, at
// most: the budget beyond it commits as captures need it.
inline constexpr std::size_t kPreparedBackingBytes = std::size_t{2} << 30;

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
  // Optional stream classes; empty ones fall back to `stream`. Foreground
  // restores never queue behind captures or persistence on a shared FIFO
  // stream, and disk writes never queue in front of a restore.
  StreamLease restore_stream;
  StreamLease background_stream;
  std::function<ComponentCaptureBuffer(const PayloadRequest&)> allocate;
  // Optional: a payload for completed append rows that shares their storage
  // (`view`, from Adapter::BorrowRowsView) instead of copying it, charged to
  // the ledger like a copy. The rows stay valid when the slot is rewound,
  // reused or destroyed, so they need no borrow, guard or preservation.
  // Nullopt when this storage cannot be shared; the caller then allocates.
  std::function<std::optional<Payload>(const PayloadRequest&,
                                       std::span<const std::byte> view)>
      share;
  // Optional: whether payload bytes at this address are readable by the
  // device only. A reader on the host copies them through a stream
  // (copy_observed_rows) instead of reading the address.
  std::function<bool(const void*)> device_only;
  // Optional: copies device-only payload bytes to host memory the device
  // can address (staging), on the caller's thread and outside every stream,
  // and waits for the copy. False when it cannot: the reader then copies
  // through its stream, behind whatever that stream's queue is running.
  std::function<bool(std::span<const std::byte> source,
                     std::span<std::byte> destination)>
      copy_device_only;
  // Optional housekeeping that waits for the device (releasing addresses of
  // shared storage that is gone). The serving layer calls it on the thread
  // that runs the model, between its launches, and while idle.
  std::function<void()> settle;
  StagingAllocator staging;
  // Prepare on-demand backing immediately before the next prefill chunk.
  std::function<void(std::size_t)> prepare_prefill;
  // Optional: prepare free backing for captures of about this many payload
  // bytes off the calling thread. A disk restore asks before it starts for
  // its checkpoint's bytes plus those of restores not captured yet, since
  // the capture after each one needs its whole checkpoint.
  std::function<void(std::size_t)> prepare_capture;
  // Optional: returns once the backing `prepare_capture` asked for is free,
  // or will not be. A disk restore whose copies run beside decoders
  // (RestoreCopyRunner) calls it after them, on their thread: a restore can
  // outrun the backing, and its capture would then wait for the rest on the
  // serving thread, with every decoder. Any thread may call it while others
  // capture.
  std::function<void()> await_capture;
  // Optional capacity bound before assigning a complete capture. False takes
  // the ordinary reclamation path; true still requires every allocation.
  std::function<bool(std::span<const PayloadRequest>)> can_capture;
  // Optional initialization step after slots allocate. The provider leaves
  // payload backing uncommitted until serving selects the post-slot budget,
  // then commits up to kPreparedBackingBytes of it before returning.
  std::function<void(std::size_t)> commit_backing;
  // Optional: the payload backing committed so far, in bytes. Growth beyond
  // it can be refused (host or device-heap reserves), so serving logs it
  // beside the budget.
  std::function<std::size_t()> committed_backing_bytes;
  // Optional host bytes the slots pinned and charged to the ledger as they
  // allocated. Serving charges them on top of the cache's budget, and adds
  // them back to the runner's claim, which measures the host after them.
  std::function<std::size_t()> pinned_bytes;
  // Optional: installs how the provider evicts one retained checkpoint when
  // sessions need its pages back (nullptr removes it; a call in progress
  // finishes first). The function returns false when nothing could go. Only
  // a provider whose backing sessions can borrow sets this.
  std::function<void(std::function<bool()>)> set_shed;
  PersistenceSource::CopyRows copy_observed_rows;
  // At most this many transfers of one capture/restore are in flight before
  // waiting (16 at most). When the shared pool is momentarily exhausted, the
  // batch settles its own leases and retries. One keeps generic providers
  // usable without a pool-capacity contract.
  std::size_t transfer_batch_size{1};
  std::size_t reassignment_batch_size{1};
  // Optional: gives up to about the bytes asked for of committed backing
  // whose payloads are all gone back to the host, and returns the bytes it
  // released; nothing is evicted. The serving layer calls it only while no
  // work runs, since releasing device-mapped pages waits for the device.
  // Released backing comes back as captures need it, under the provider's
  // usual growth rules.
  std::function<std::size_t(std::size_t)> release_host;
  // Inside a BesideDecodeScope, restore_stream leases streams whose copies
  // never wait behind the model's work (the HIP provider's copy engine): a
  // restore may copy beside decode (RestoreCopyRunner). On streams that share
  // a hardware queue with decode, a 15K-token DeepSeek restore beside decode
  // waited ~1 s behind it instead of finishing in ~0.1 s.
  bool restore_beside_decode{false};
};
}  // namespace gufo::cache
