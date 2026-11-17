#ifndef GUFO_CORE_PLATFORM_PAGES_HPP_
#define GUFO_CORE_PLATFORM_PAGES_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace gufo::platform {

struct SharedTempMapping {
  void* data = nullptr;
  std::size_t size = 0;
  std::intptr_t handle = -1;
  std::string unavailable_reason{};
};

/// Reserves and commits `bytes` of zero-filled anonymous memory, opting out
/// of transparent huge pages and touching every page before returning, so
/// later promotions cannot invalidate device mappings. Returns nullptr on
/// failure. Pair with ReleaseCommittedPages(); registration with a device
/// runtime stays with the caller.
void* CommitZeroedPages(std::size_t bytes) noexcept;

/// Releases memory obtained from CommitZeroedPages() or
/// CloneMappedWeightsToAnonymousPages(). `bytes` must match the commit call.
void ReleaseCommittedPages(void* base, std::size_t bytes) noexcept;

/// Clones a read-only file mapping range into anonymous pages populated in
/// parallel and drops the source page-cache pages as chunks complete. Returns
/// nullptr when anonymous cloning is unsupported on the host OS or when
/// allocation fails; throws std::system_error if page population fails.
void* CloneMappedWeightsToAnonymousPages(const void* source_data,
                                         std::size_t bytes);

/// True when VisitMappedChunks can read a file mapping without keeping its
/// pages in the process: Linux drops each chunk's page-table entries after
/// the visit, while Windows would keep the whole range in the working set.
bool CanVisitMappedChunks() noexcept;

/// Visits a read-only file mapping range in `chunk_bytes` pieces from up to
/// `workers` threads. Each chunk is populated from the page cache or disk
/// before `visit(worker, offset, chunk, chunk_size)` runs and its page-table
/// entries are dropped afterwards; the page cache keeps the file data. A
/// `false` from `visit` stops the remaining chunks. Requires
/// CanVisitMappedChunks(); throws std::system_error if population fails.
void VisitMappedChunks(
    const void* source_data, std::size_t bytes, std::size_t chunk_bytes,
    std::size_t workers,
    const std::function<bool(std::size_t worker, std::size_t offset,
                             const void* chunk, std::size_t chunk_size)>&
        visit);

/// Populates writable anonymous pages across up to `worker_limit` threads and
/// requests transparent huge pages where supported.
void PopulateWritablePages(void* data, std::size_t bytes,
                           std::size_t worker_limit = 4) noexcept;

/// Returns a human-readable summary of resident pages in `[data, data +
/// bytes)`.
std::string DescribeResidentPages(void* data, std::size_t bytes);

/// Creates an anonymous shared memory-backed mapping for benchmarking, or sets
/// `unavailable_reason` if the OS does not support `memfd_create`.
SharedTempMapping CreateSharedTempMapping(std::size_t bytes) noexcept;

/// Releases a mapping created by CreateSharedTempMapping().
void ReleaseSharedTempMapping(SharedTempMapping& mapping) noexcept;

/// True when device pages are made from host RAM, so state promised to
/// sessions but not mapped yet is host memory the host budgets must hold
/// back. False on Windows, where they come from the dedicated carve-out.
bool DevicePagesUseHostMemory() noexcept;

/// True when an exported device-memory file (a Linux dma-buf) can be mapped
/// for the CPU with MapDeviceFileAt().
bool CanMapDeviceFiles() noexcept;

/// True when every byte of [address, address + bytes) is already mapped in
/// this process (reserved by the device runtime, for example), so a fixed
/// mapping over it cannot replace anything else.
bool IsAddressRangeReserved(const void* address, std::size_t bytes);

/// Maps `bytes` of the device-memory file `handle` read-write at `address`,
/// replacing the reservation there, then closes the handle, and faults the
/// range in so later CPU reads take no page faults. The device runtime
/// reserves the same range for the GPU, so the CPU and the GPU use one
/// pointer. Returns false (the handle still closed) on failure.
bool MapDeviceFileAt(void* address, std::size_t bytes,
                     std::intptr_t handle) noexcept;

/// Closes a device-memory file handle that was not mapped.
void CloseDeviceFile(std::intptr_t handle) noexcept;

/// Replaces a MapDeviceFileAt() mapping with an inaccessible reservation,
/// as the device runtime left the range.
void UnmapDeviceFileAt(void* address, std::size_t bytes) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PAGES_HPP_
