#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_MEMORY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_MEMORY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::qwen38_flash_next::rocm {

/// Physical memory for K/V rows that grow with the tokens a session holds,
/// shared by one executor's sessions under a byte budget. Rows live in
/// virtual ranges reserved for the full context and backed chunk by chunk
/// (HIP virtual memory management), so a buffer never moves: captured
/// graphs and borrowed row views keep their addresses while it grows.
class KvPool {
public:
  /// Physical memory is mapped in chunks of this size, or of the whole
  /// buffer rounded up to kGranuleBytes when it is smaller.
  static constexpr std::size_t kChunkBytes = std::size_t{2} << 20;
  /// A multiple of the device's allocation granularity.
  static constexpr std::size_t kGranuleBytes = std::size_t{64} << 10;

  /// `budget` bytes of chunks may be mapped at once; zero means no limit
  /// beyond the device heap.
  explicit KvPool(std::size_t budget) : budget_(budget) {}
  KvPool(const KvPool&) = delete;
  KvPool& operator=(const KvPool&) = delete;
  /// Frees every virtual range; no buffer may outlive the pool.
  ~KvPool();

  /// Checks that this device can back growable rows.
  [[nodiscard]] static bool Supported(std::string* error);

  [[nodiscard]] std::size_t budget() const noexcept { return budget_; }
  [[nodiscard]] std::size_t mapped() const;
  [[nodiscard]] std::size_t Available() const;
  /// Takes `bytes` of the budget, or nothing when they do not fit.
  [[nodiscard]] bool Charge(std::size_t bytes);
  void Refund(std::size_t bytes);
  /// A virtual range of `bytes`: a released one of that size, or a new
  /// reservation. Null when the device has no address space left.
  [[nodiscard]] void* ReserveRange(std::size_t bytes, hipError_t* status);
  /// Keeps an unmapped range for ReserveRange. Ranges go back to HIP only
  /// when the pool is destroyed: on ROCm 7.2.4, chunks mapped into an
  /// address range that an earlier hipMemAddressFree or hipFree released
  /// intermittently fault on first access.
  void ReleaseRange(void* base, std::size_t bytes);

private:
  mutable std::mutex mutex_;
  std::size_t budget_;
  std::size_t mapped_{0};
  std::vector<std::pair<void*, std::size_t>> ranges_;  ///< every reservation
  std::vector<std::pair<void*, std::size_t>> free_ranges_;
};

/// One K, V or pooled-key buffer of a session: rows of `row_bytes` up to
/// `max_rows`, of which a prefix is backed by mapped chunks. Rows past the
/// backed prefix must not be touched. New chunks read as zeros.
class GrowableRows {
public:
  GrowableRows() = default;
  GrowableRows(const GrowableRows&) = delete;
  GrowableRows& operator=(const GrowableRows&) = delete;
  GrowableRows(GrowableRows&& other) noexcept;
  GrowableRows& operator=(GrowableRows&&) = delete;
  ~GrowableRows();

  /// Reserves the full virtual range; nothing is backed yet.
  [[nodiscard]] bool Reserve(KvPool* pool, std::size_t row_bytes,
                             std::size_t max_rows, std::string* error);
  /// Bytes a buffer of `max_rows` rows maps once every row is backed.
  [[nodiscard]] static std::size_t FullBytes(std::size_t row_bytes,
                                             std::size_t max_rows) noexcept;
  [[nodiscard]] std::uint8_t* data() const noexcept { return base_; }
  /// Rows currently backed (capped at the reserved rows).
  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t mapped_bytes() const noexcept {
    return chunks_.size() * chunk_;
  }
  /// Bytes Grow(rows) would add.
  [[nodiscard]] std::size_t GrowthBytes(std::size_t rows) const noexcept;
  /// Backs rows [0, rows), charging the pool. New chunks are zeroed on
  /// `stream` before any later work on it. On failure nothing changes.
  [[nodiscard]] bool Grow(std::size_t rows, hipStream_t stream,
                          std::string* error);
  /// Releases the chunks past those holding rows [0, rows). The caller has
  /// drained every reader of those rows. Returns the bytes released.
  std::size_t Shrink(std::size_t rows) noexcept;

private:
  [[nodiscard]] std::size_t ChunksFor(std::size_t rows) const noexcept;
  KvPool* pool_{nullptr};
  std::uint8_t* base_{nullptr};
  std::size_t reserved_bytes_{0};
  std::size_t chunk_{0};
  std::size_t row_bytes_{0};
  std::size_t max_rows_{0};
  std::vector<hipMemGenericAllocationHandle_t> chunks_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_KV_MEMORY_HPP_
