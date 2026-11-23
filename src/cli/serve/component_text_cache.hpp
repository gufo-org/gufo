#ifndef GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_
#define GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>

#include "gufo/ext/text.hpp"
#include "src/cache/checkpoint.hpp"
#include "src/cache/component_resources.hpp"
#include "src/cache/streaming.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/continuation_types.hpp"
#include "src/cli/serve/text_publication_wait.hpp"
#include "src/ext/host_budget.hpp"
#include "src/mem/account.hpp"

namespace gufo::server {
using ext::TextModelRunner;
using ext::TextPromptContext;
using ext::TextRunnerDiskCacheOptions;
using ext::TextRunnerExpertCacheClaim;
using ext::TextRunnerState;

// The capture/backing contracts are cache-owned (RFC step 6); these aliases
// keep the serving spelling.
using cache::ComponentCacheResources;
using cache::ComponentCaptureBuffer;

// Model-independent serving orchestration; the model supplies physical storage
// and binds its ordinary execution facade to adapter-owned slots.
class ComponentTextCache {
public:
  struct Impl;
  class Lease {
  public:
    struct Impl;
    explicit Lease(std::unique_ptr<Impl>);
    ~Lease();
    Lease(const Lease&) = delete;
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_tokens() const noexcept;
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
    [[nodiscard]] double restore_ms() const noexcept;
    [[nodiscard]] bool restored_from_disk() const noexcept;
    [[nodiscard]] ContinuationLookup lookup() const noexcept;
    [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
    [[nodiscard]] bool TryReserveSnapshot(
        std::size_t, std::size_t, bool = false,
        SnapshotPurpose = SnapshotPurpose::kContinuation,
        std::span<const ContinuationToken> = {});
    void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
    std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                                std::shared_ptr<const ContinuationSnapshot>,
                                bool = false);
    std::size_t Commit(std::vector<ContinuationToken>,
                       std::shared_ptr<const ContinuationSnapshot> = {},
                       std::vector<ContinuationToken> = {});
    void Invalidate() noexcept;
    [[nodiscard]] std::size_t CaptureBytes() const;
    // The same for a checkpoint taken at `boundary` inside a prefill pass.
    // Nullopt when the adapter cannot take one there: the caller captures
    // at a pass boundary instead. Never a reason to fail the request (#1240).
    [[nodiscard]] std::optional<std::size_t> CaptureBytes(
        std::size_t boundary) const;
    [[nodiscard]] std::unique_ptr<ContinuationSnapshot> Capture();
    void PreparePrefill(std::size_t tokens);
    [[nodiscard]] ComponentCacheMetrics Metrics() const;
    [[nodiscard]] std::vector<std::size_t> PlannedCaptures(
        std::size_t stable) const;
    /// Also persists one of the `planned` grid captures: the deepest far
    /// enough below the prompt checkpoint at `boundary` to serve another
    /// question about the same prefix after RAM has lost it.
    void PersistGrid(std::span<const std::size_t> planned,
                     std::size_t boundary);

  private:
    friend class ComponentTextCache;
    std::unique_ptr<Impl> impl_;
  };
  // Ledger room for index and request metadata when the RAM cache is off. It
  // caps accounting only; the memory is not reserved. A caller sizing the
  // ledger from host RAM keeps at least this much, so a host short of its
  // reserve still loads the model with the cache off.
  static constexpr std::size_t kOffMetadataBytes = std::size_t{64} << 20;
  // Slots allocate under the larger of ram_bytes and ceiling_bytes, so a
  // smaller budget does not refuse what they pin. The RAM limit then becomes
  // the budget plus those pins, within the host ceiling. With `ram_enabled`
  // false, ram_bytes is ignored: no checkpoint stays in RAM and no backing is
  // committed, unless a disk cache needs some to stage captures. A streamed
  // model's expert cache (`experts`, from the load-time claim) is sized once
  // the prompt cache's backing exists.
  /// Runs a restore's copies (cache::RestoreCopyRunner), told how long they
  /// would stop the calling thread.
  using RestoreRunner =
      std::function<void(std::chrono::microseconds inline_estimate,
                         const std::function<void()>& copies)>;
  ComponentTextCache(std::shared_ptr<TextModelRunner>, std::size_t,
                     std::size_t ram_bytes,
                     const std::optional<TextRunnerDiskCacheOptions>&,
                     bool automatic_ram = false, std::size_t ceiling_bytes = 0,
                     bool ram_enabled = true,
                     TextRunnerExpertCacheClaim experts = {});
  ~ComponentTextCache();
  // With `state_tokens`, the session the prompt lands on claims them from
  // the shared state budget (StateAccount::Claim), drawing on
  // `reservation` first. Throws StateMemoryShort when the budget cannot
  // cover them; the scheduler reserves first, so it does not.
  [[nodiscard]] std::unique_ptr<Lease> Acquire(
      std::span<const ContinuationToken>, const std::function<bool()>&,
      std::shared_ptr<const TextPromptContext>, bool, std::size_t, bool,
      bool persist = true, std::size_t state_tokens = 0,
      mem::StateReservation* reservation = nullptr,
      const RestoreRunner& run_copies = {});
  // Whether sessions share their demand-mapped state (#715).
  [[nodiscard]] bool shared_state() const noexcept;
  // Whether restores copy where decode work cannot hold them up, and so may
  // run beside decode (ComponentCacheResources::restore_beside_decode).
  [[nodiscard]] bool restores_beside_decode() const noexcept;
  // Bytes a session that holds nothing would charge for `tokens`; zero
  // without shared state.
  [[nodiscard]] std::size_t StateBytesFor(std::size_t tokens) const;
  // Frees state pages of sessions until `bytes` can be charged to the
  // shared budget: first pages mapped past each session's need (no state
  // lost), then the live frontiers of idle sessions, the cheapest to lose
  // first, never the one `prompt` would continue. Unmapping waits for the
  // device: call it where no model work is in flight (the scheduler, between
  // work units). Returns whether `bytes` can be charged now.
  //
  // A frontier that a disk write still reads in place goes last, and stays
  // while the write is younger than `write_wait`: dropping it would fail
  // the write. Admission waits for such writes (kStateWriteWait); a running
  // request's decode does not wait.
  [[nodiscard]] bool ReclaimState(std::size_t bytes,
                                  std::span<const ContinuationToken> prompt,
                                  const TextPromptContext* context,
                                  std::chrono::milliseconds write_wait = {});
  // How long admission keeps an idle frontier for a disk write that reads
  // it, from the write's capture. Writes queue one after another and share
  // the disk with restores: three queued 1.9 GB Flash-Next frontiers beside
  // five disk restores took 15 s. Losing one costs its whole prefill when
  // the conversation returns (40 s at 66K tokens).
  static constexpr std::chrono::milliseconds kStateWriteWait{30000};
  // Host memory pressure from other processes (#932). When `sample` leaves
  // MemAvailable short of the host reserve (ext::HostReserveShortfallBytes),
  // gives
  // RAM-cache memory back to the host until the shortfall plus
  // kHostPressureReliefBytes is covered, evicting the least valuable
  // checkpoints as needed (disk copies stay). Returns the bytes released.
  // Releasing device-mapped pages waits for the device: call it only while
  // no work runs (the idle scheduler).
  // `reserve` replaces the host reserve when given (a streamed model's
  // expert-cache floor, TextRunnerPool::Idle).
  std::size_t RelieveHostPressure(
      const ext::HostMemorySample& sample,
      std::optional<std::uint64_t> reserve = std::nullopt);
  // Released beyond the shortfall, so that pressure hovering at the reserve
  // does not release a little on every idle tick.
  static constexpr std::size_t kHostPressureReliefBytes = std::size_t{2} << 30;
  // Charged bytes that ReclaimState could free: tails of all sessions and
  // idle sessions' frontiers.
  [[nodiscard]] std::size_t ReclaimableStateBytes() const;
  // A disk job is writing a prefix of the prompt much deeper than any
  // restorable boundary; admission may wait for it instead of prefilling.
  // A nonzero checkpoint ID follows that one job, even if another deeper job
  // is submitted later. Its current gain may shrink as usable cache improves.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const ContinuationToken> prompt,
      const TextPromptContext* context, std::uint64_t checkpoint_id = 0) const;
  // When the checkpoint Acquire would select for `prompt` is only on disk,
  // starts reading its files into the page cache on a worker and returns a
  // flag set when that finishes. Null when there is nothing to read, when
  // the read would take free pages the host keeps (mem::HostFreeFloorBytes),
  // or when the restore would take `beside_after` or longer: the caller then
  // copies it beside running work (RestoreRunner), which reads it there.
  [[nodiscard]] std::shared_ptr<const std::atomic<bool>> WarmDisk(
      std::span<const ContinuationToken> prompt,
      const TextPromptContext* context, std::size_t stable,
      std::chrono::microseconds beside_after =
          std::chrono::microseconds::max());
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const ContinuationToken>, const TextPromptContext*,
      bool common = false) const;
  // Re-tokenizing a conversation can split a previous answer differently from
  // the tokens the model sampled, and a recurrent frontier cannot rewind to
  // the first difference (#1154). When the prompt repeats the whole text of an
  // idle live frontier, replace that part with the frontier's own tokens.
  // With `trimmed` (a chat template rendered the prompt), whitespace it trimmed
  // from the edges of a reply stays as the model generated it; whitespace
  // missing anywhere else is an edit. A frontier ending inside a prompt token
  // continues with the rest of that token. `stable` follows the rewrite, and
  // the prompt stays below `max_tokens`. Returns whether the prompt changed.
  bool AlignToLiveFrontier(std::vector<ContinuationToken>& prompt,
                           std::size_t& stable, std::size_t max_tokens,
                           bool trimmed) const;
  [[nodiscard]] std::size_t capacity() const noexcept;
  // RAM the cache may retain after host limits; zero with the RAM cache off.
  [[nodiscard]] std::size_t ram_budget() const noexcept;

private:
  std::unique_ptr<Impl> impl_;
};

// The shared state budget cannot cover a request's tokens.
class StateMemoryShort final : public std::runtime_error {
public:
  StateMemoryShort() : std::runtime_error("shared session state exhausted") {}
};

}  // namespace gufo::server
#endif
