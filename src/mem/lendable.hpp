#ifndef GUFO_MEM_LENDABLE_HPP_
#define GUFO_MEM_LENDABLE_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace gufo::mem {

/// A fixed range of whole state pages that RAM-cache payloads live in. The
/// CPU and the GPU use the same address, and pages are committed as the
/// cache grows. When a session needs a page and the host is at its floor,
/// the state pool asks the owner for pages it no longer uses and maps them
/// into the session as they are: a map, never an unmap or a new page, so
/// reclaiming the cache costs no device drain and no compaction. A lent page
/// stays mapped here, unused, until it comes back.
///
/// HIP-free so the cache backing builds without the state backend; the
/// backend installs the factory when its pool can lend.
class LendableRange {
public:
  /// Appends up to `pages` page indices the owner gave up, evicting what
  /// they held if it must. It must not use them until Restore().
  using Select =
      std::function<std::size_t(std::size_t pages, std::vector<std::size_t>&)>;
  /// A lent page is back, still mapped, holding a session's old bytes.
  using Restore = std::function<void(std::size_t page)>;

  virtual ~LendableRange() = default;
  [[nodiscard]] virtual std::byte* data() const noexcept = 0;
  [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
  [[nodiscard]] virtual std::size_t page_bytes() const noexcept = 0;
  /// Backs pages [first, first + count). New pages come from free host
  /// memory above the floor, or, with `may_reclaim` (while loading, before
  /// serving: a compaction stall then costs nothing), from MemAvailable.
  /// False leaves the rest unbacked: the host is near its floor, or the
  /// driver refused.
  [[nodiscard]] virtual bool Commit(std::size_t first, std::size_t count,
                                    bool may_reclaim) = 0;
  /// Nothing is lent until the owner is set. No lock is held while the
  /// functions run, and a call may still be in progress after clearing, so
  /// they must stay safe on their own (hold weak references). Lent pages
  /// stay with the sessions that map them.
  virtual void SetOwner(Select select, Restore restore) = 0;
  virtual void ClearOwner() = 0;
  /// Pages Decommit() gave back, and pages it could not unmap and retired:
  /// those have no CPU view any more and must never be used or committed
  /// again. Every other page passed in stays committed.
  struct Decommitted {
    std::vector<std::size_t> released;
    std::vector<std::size_t> retired;
  };
  /// Gives committed pages back to the driver, and so to the host; Commit()
  /// can back released pages again. The owner must not use them meanwhile.
  /// Unmapping waits until no GPU work is in flight, so call it only while
  /// the device is idle: beside running work it waited 0.05-15 s per page.
  [[nodiscard]] virtual Decommitted Decommit(
      std::span<const std::size_t> pages) = 0;
};

using LendableRangeFactory = std::shared_ptr<LendableRange> (*)(std::size_t);

/// Installed by the state backend for the process; null without one.
void SetLendableRangeFactory(LendableRangeFactory factory) noexcept;
/// A range of at least `bytes`, or null when pages cannot be lent here
/// (no backend, no virtual memory management, or no CPU view of device
/// pages, as on Windows today).
[[nodiscard]] std::shared_ptr<LendableRange> CreateLendableRange(
    std::size_t bytes);

}  // namespace gufo::mem

#endif  // GUFO_MEM_LENDABLE_HPP_
