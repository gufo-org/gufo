#ifndef GUFO_CORE_HIP_STATE_REGION_HPP_
#define GUFO_CORE_HIP_STATE_REGION_HPP_

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "src/mem/account.hpp"

namespace gufo::mem {

class PagePool;

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

private:
  Region() = default;
  [[nodiscard]] bool MapPages(std::size_t rows, std::string* error);
  /// Unmaps page `p`; requires mutex_.
  void UnmapPage(std::size_t p);
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
  std::mutex mutex_;
  std::vector<hipMemGenericAllocationHandle_t> pages_;
  std::vector<bool> mapped_;
  std::size_t requested_rows_{0};
  std::atomic<std::size_t> backed_rows_{0};
  std::atomic<std::size_t> mapped_bytes_{0};
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_STATE_REGION_HPP_
