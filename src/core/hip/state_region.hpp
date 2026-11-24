#ifndef GUFO_CORE_HIP_STATE_REGION_HPP_
#define GUFO_CORE_HIP_STATE_REGION_HPP_

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "src/mem/account.hpp"

namespace gufo::mem {

class PagePool;
class SharedPage;

/// Whether state regions on the current device take pages as rows are
/// written, rather than their whole size at creation (false without virtual
/// memory management, or where device pages are not host RAM).
[[nodiscard]] bool DemandMapped();

/// One buffer at a fixed address for its whole size, backed by physical
/// pages from the front of each segment as rows are written. The address
/// never changes, so captured graphs and kernel arguments stay valid, and
/// growth copies nothing.
///
/// Every writer backs its rows first: Ensure() before the write, at one choke
/// point per model, including cache restores. Reads past the backed rows
/// fault, so a model maps a tile's worth of rows past its frontier.
///
/// Without virtual memory management the region allocates its full size,
/// as before, and Ensure() only checks bounds.
class Region : public std::enable_shared_from_this<Region> {
public:
  /// With `account`, the region's pages are charged to that session
  /// (StateAccount), and its rows are `tokens_per_row` tokens each.
  [[nodiscard]] static std::shared_ptr<Region> Create(
      const RegionShape& shape, std::string* error,
      const std::shared_ptr<StateAccount>& account = nullptr,
      std::size_t tokens_per_row = 1);
  ~Region();
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

  [[nodiscard]] void* data() const noexcept { return base_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] const RegionShape& shape() const noexcept { return shape_; }
  [[nodiscard]] bool demand_mapped() const noexcept { return pool_ != nullptr; }
  /// Rows backed in every segment; lock-free.
  [[nodiscard]] std::size_t backed_rows() const noexcept {
    return backed_rows_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return mapped_bytes_.load(std::memory_order_relaxed);
  }
  /// Backs rows [0, rows) of every segment now. Rows past the shape clamp.
  /// Lock-free when they are already backed.
  [[nodiscard]] bool Ensure(std::size_t rows, std::string* error = nullptr);
  /// Ensure() for map-ahead: backs at most AheadLimit(rows), read under the
  /// region's lock, so a session trimmed meanwhile (TrimTo lowers the need
  /// first) is not mapped again.
  [[nodiscard]] bool EnsureAhead(std::size_t rows);
  /// Asks the pool thread to back rows [0, rows). Returns at once. While
  /// sessions share their state, only rows the account's need covers are
  /// mapped ahead.
  void MapAhead(std::size_t rows);
  /// `rows` clamped to what map-ahead may back: the session's need, which
  /// may have fallen since a job was queued (a trimmed session).
  [[nodiscard]] std::size_t AheadLimit(std::size_t rows) const;
  /// Unmaps every page that rows [0, rows) of no segment use. Unmapping
  /// waits for all in-flight GPU work on the device: call it only where
  /// nothing reads or writes this region. Pages return to the pool.
  void TrimTo(std::size_t rows);

  // Shared pages. Rows are append-only, so a page of written rows can serve
  // several users at once: this region, the RAM cache that retains the rows
  // for a checkpoint, and regions that restore the checkpoint by mapping the
  // page instead of copying it. A user that must write shared rows calls
  // PrepareWrite() first. Only single-segment regions share pages, and only
  // where the cache may hold device pages (StateLedger::LendingEnabled()).

  /// The demand-mapped region whose range holds `address`, or null.
  [[nodiscard]] static std::shared_ptr<Region> Find(const void* address);
  /// Page `page` as a shared page, for a user of bytes below `end_byte` of
  /// the region. Null when the region cannot share it; the user then copies
  /// the rows. The region keeps writing rows at or past `end_byte` in place.
  [[nodiscard]] std::shared_ptr<SharedPage> Share(std::size_t page,
                                                  std::size_t end_byte);
  /// Before writing rows [first_row, end_row): every page of those rows
  /// that others share becomes private to the region. Rows below
  /// `kept_rows` hold data the caller still needs (when appending, the rows
  /// before `first_row`; when rewriting rows in place, all of them): a page
  /// that holds any is copied, the others are replaced by cleared pages.
  /// Lock-free when no page of the region can be shared. Replacing a page
  /// unmaps it, which waits for in-flight GPU work: call it on the thread
  /// that runs the model, between its launches. A replaced page reaches the
  /// device, and gets its copy or is cleared, only in SettleRemaps(): call
  /// that once after PrepareWrite() on every region of the write and before
  /// launching it.
  [[nodiscard]] bool PrepareWrite(std::size_t first_row, std::size_t end_row,
                                  std::size_t kept_rows,
                                  std::string* error = nullptr);
  /// Before a restore brings rows [0, rows): page p maps `plan[p]` where
  /// that is set (rows restored without a copy), and every other page of
  /// those rows is private to the region or unmapped, so a load may write
  /// it. Pages past the rows stay as they are. Returns, per page of `plan`,
  /// whether the region now maps the planned page; a page that could not be
  /// mapped is left for a copy. Rows are backed afterwards only as far as
  /// backed_rows() says: Ensure() before loading. Same thread rule as
  /// PrepareWrite(), and SettleRemaps() once after Restore() on every region
  /// of the restore, before anything reads, loads or maps the rows ahead. An
  /// empty plan only releases what others share.
  [[nodiscard]] std::vector<bool> Restore(
      std::span<const std::shared_ptr<SharedPage>> plan, std::size_t rows);
  /// After PrepareWrite() or Restore() remapped pages: lets the device reach
  /// them (mem::RemapBarrier, about 0.5 ms with the device idle), then fills
  /// or clears the replaced pages. One call serves every region of the
  /// device; without remaps it costs two atomic reads. False when a page
  /// copy failed. Same thread rule as PrepareWrite().
  [[nodiscard]] bool SettleRemaps(std::string* error = nullptr);
  /// Pages this region maps that another user shares.
  [[nodiscard]] std::size_t shared_pages() const;
  /// A session given to another conversation keeps the pages a checkpoint
  /// shares mapped past what it writes, as it keeps its own: if the first
  /// conversation comes back to it, Restore() finds them in place.

private:
  Region() = default;
  /// `ahead`: on the pool's thread, which never maps an address that needs
  /// the remap barrier.
  [[nodiscard]] bool MapPages(std::size_t rows, std::string* error, bool ahead);
  /// Unmaps pages [first, end), all mapped, in one call; requires mutex_.
  void UnmapRun(std::size_t first, std::size_t end);
  /// Unmaps every page marked in `drop`, one call per run; requires mutex_.
  void UnmapMarked(const std::vector<bool>& drop);
  /// Gives up page `p` after its address was unmapped; requires mutex_.
  void ReleasePage(std::size_t p);
  [[nodiscard]] bool SharedWithOthers(std::size_t p) const noexcept;
  /// Whether a write starting at `first_byte` must first replace page `p`:
  /// others use the page, and the write is not this region appending past
  /// the rows they share. Requires mutex_.
  [[nodiscard]] bool MustReplace(std::size_t p, std::size_t first_byte);
  /// Rows of segment 0 that the mapped pages cover from its start.
  [[nodiscard]] std::size_t MappedPrefixRows() const noexcept;
  /// Lowers shared_end_ to the end of the last page others still share.
  void SettleSharedEnd() noexcept;
  /// Bytes from its segment's start at which the last page others share
  /// ends, over all segments; 0 without one. Requires mutex_.
  [[nodiscard]] std::size_t SharedEnd() const noexcept;
  /// First byte of the segment page `p` lies in, in a region whose pages
  /// belong to one segment each.
  [[nodiscard]] std::size_t SegmentStart(std::size_t p) const noexcept;
  /// Replaces pages [first, first + count), all shared with others, by pages
  /// of the region's own: copies of them (`copy`) or cleared pages.
  /// Requires mutex_.
  [[nodiscard]] bool ReplaceRun(std::size_t first, std::size_t count, bool copy,
                                std::string* error);
  [[nodiscard]] std::size_t CoveredRows(std::size_t pages_end_bytes,
                                        std::size_t segment) const noexcept;

  std::shared_ptr<PagePool> pool_;
  std::shared_ptr<StateAccount> account_;
  std::size_t tokens_per_row_{1};
  RegionShape shape_{};
  std::size_t stride_{0};
  std::size_t bytes_{0};
  std::size_t reserved_bytes_{0};
  std::uint8_t* base_{nullptr};
  mutable std::mutex mutex_;
  std::vector<hipMemGenericAllocationHandle_t> pages_;
  std::vector<bool> mapped_;
  /// Set for a page that has, or had, another user; null for a page only
  /// this region ever used.
  std::vector<std::shared_ptr<SharedPage>> shared_;
  /// For a page this region wrote itself: the byte, from the region's
  /// start, up to which others share its rows; the region goes on writing
  /// past it in place. A page mapped from a checkpoint (Restore) is shared
  /// whole and `written_here_` is false: any write replaces it.
  std::vector<std::size_t> shared_until_;
  std::vector<bool> written_here_;
  /// RemapBarrier::Stamp() of the moment each page's address last lost a
  /// mapping; 0 for an address that never had one.
  std::vector<std::uint64_t> unmapped_at_;
  /// Bytes from a segment's start below which a page may be shared.
  std::atomic<std::size_t> shared_end_{0};
  /// Pages belong to one segment each: one segment, or segments that start
  /// on a page. Only such a region shares pages.
  bool shareable_{false};
  std::size_t requested_rows_{0};
  std::atomic<std::size_t> backed_rows_{0};
  std::atomic<std::size_t> mapped_bytes_{0};
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_REGION_HPP_
