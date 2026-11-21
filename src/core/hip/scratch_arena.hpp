#ifndef GUFO_CORE_HIP_SCRATCH_ARENA_HPP_
#define GUFO_CORE_HIP_SCRATCH_ARENA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gufo::mem {

/// Pass scratch at fixed addresses whose rows past a resident width take
/// memory only while a wider pass needs them (#952). Buffers are carved at
/// their widest size, so kernel arguments and captured graphs never change.
/// Resident ranges (what narrow passes touch) are backed for the arena's
/// life; MapTransient() backs every other page, and ReleaseTransient()
/// returns those pages to the driver, not to the state page pool, so the
/// memory is free again for the RAM cache and the rest of the host.
///
/// Unlike state regions, the arena promises nothing to the state ledger: a
/// caller maps transient pages only when the host has them to spare.
class ScratchArena {
public:
  /// Reserves `capacity` bytes of address space. Null where device memory
  /// is not demand-mapped (no virtual memory management, or device pages
  /// that are not host RAM): callers then allocate their narrow scratch
  /// whole, as before.
  [[nodiscard]] static std::unique_ptr<ScratchArena> Reserve(
      std::size_t capacity, std::string* error);
  ~ScratchArena();
  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;

  /// Carves `bytes` at a 2 MiB boundary and backs its first `resident`
  /// bytes now. Null when the reservation is exhausted or a page cannot be
  /// made.
  [[nodiscard]] void* Take(std::size_t bytes, std::size_t resident,
                           std::string* error);
  /// Backs [pointer, pointer + bytes) for the arena's life, for buffers
  /// carved together (sub-buffers of one Take).
  [[nodiscard]] bool MapResident(const void* pointer, std::size_t bytes,
                                 std::string* error);

  /// Bytes MapTransient() maps: the carved pages that are not resident.
  [[nodiscard]] std::size_t transient_bytes() const noexcept;
  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return resident_pages_ * kPage;
  }
  [[nodiscard]] bool transient_mapped() const noexcept {
    return transient_mapped_;
  }
  /// Backs every carved page. On failure nothing stays mapped beyond the
  /// resident pages.
  [[nodiscard]] bool MapTransient(std::string* error);
  /// Unmaps and frees the transient pages. Unmapping waits for all in-flight
  /// GPU work on the device: call it only where no pass uses them.
  void ReleaseTransient() noexcept;

  static constexpr std::size_t kPage = std::size_t{2} << 20;

private:
  ScratchArena() = default;
  struct Run {
    std::size_t first{0};
    std::size_t pages{0};
    hipMemGenericAllocationHandle_t handle{};
  };
  [[nodiscard]] bool MapRun(std::size_t first, std::size_t pages, Run* run,
                            std::string* error);
  void UnmapRun(const Run& run) noexcept;

  hipMemAllocationProp prop_{};
  hipMemAccessDesc access_{};
  std::uint8_t* base_{nullptr};
  std::size_t capacity_{0};
  std::size_t used_pages_{0};
  /// One flag per page of the reservation.
  std::vector<bool> resident_;
  std::size_t resident_pages_{0};
  std::vector<Run> resident_runs_;
  std::vector<Run> transient_runs_;
  bool transient_mapped_{false};
  /// RemapBarrier::Stamp() of the last ReleaseTransient() that unmapped
  /// rows; 0 before the first.
  std::uint64_t released_at_{0};
};

}  // namespace gufo::mem

#endif  // GUFO_CORE_HIP_SCRATCH_ARENA_HPP_
