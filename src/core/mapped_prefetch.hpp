#ifndef GUFO_CORE_MAPPED_PREFETCH_HPP_
#define GUFO_CORE_MAPPED_PREFETCH_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

namespace gufo::core {

/// Populate an existing readable mapping before HIP registration or upload.
/// Parallel faults expose disk queue depth without making another weight copy.
/// The mapping must remain alive until this call returns, including on failure.
inline void PrefaultMappedRange(const void* data, std::size_t bytes) {
  if (bytes == 0)
    return;
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (!data || page_size <= 0)
    throw std::invalid_argument("invalid mapped weight range");
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    throw std::length_error("mapped weight address overflows");
  const auto skip = address % static_cast<std::size_t>(page_size);
  if (bytes > std::numeric_limits<std::size_t>::max() - skip)
    throw std::length_error("mapped weight range overflows");
  const auto begin = address - skip;
  bytes += skip;
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  constexpr std::size_t kReaders = 16;
  const auto chunks = bytes / kChunkBytes + (bytes % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
  auto populate = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      if (::madvise(reinterpret_cast<void*>(begin + offset),
                    std::min(kChunkBytes, bytes - offset),
                    MADV_POPULATE_READ) != 0)
        failure.store(errno, std::memory_order_relaxed);
    }
  };
  {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < std::min(kReaders, chunks); ++i)
      readers.emplace_back(populate);
    populate();
    // Join before a caller can register, upload, or unmap any of these pages.
  }
  if (const int code = failure.load(); code != 0)
    throw std::system_error(code, std::generic_category(),
                            "cannot read mapped weights");
}

}  // namespace gufo::core

#endif  // GUFO_CORE_MAPPED_PREFETCH_HPP_
