#ifndef GUFO_CORE_PLATFORM_PAGES_HPP_
#define GUFO_CORE_PLATFORM_PAGES_HPP_

#include <cstddef>
#include <cstdint>
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

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PAGES_HPP_
