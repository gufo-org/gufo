#ifndef GUFO_CORE_HIP_STATE_REGION_HPP_
#define GUFO_CORE_HIP_STATE_REGION_HPP_

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gufo::mem {

class PagePool;

/// Rows of state laid out in `segments` equal runs, each starting
/// `segment_stride` bytes after the previous one: one segment for a K/V
/// buffer per layer, or one per layer and plane for a layer-major buffer
/// such as Qwen 27B's [plane][layer][position][width]. Rows grow from the
/// start of every segment.
struct RegionShape {
  std::size_t row_bytes{0};
  std::size_t rows{0};
  std::size_t segments{1};
  /// Zero means rows * row_bytes: segments back to back.
  std::size_t segment_stride{0};
};

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
  [[nodiscard]] static std::shared_ptr<Region> Create(const RegionShape& shape,
                                                      std::string* error);
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
  /// Asks the pool thread to back rows [0, rows). Returns at once.
  void MapAhead(std::size_t rows);

private:
  Region() = default;
  [[nodiscard]] bool MapPages(std::size_t rows, std::string* error);
  [[nodiscard]] std::size_t CoveredRows(std::size_t pages_end_bytes,
                                        std::size_t segment) const noexcept;

  std::shared_ptr<PagePool> pool_;
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
