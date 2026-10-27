#ifndef GUFO_CORE_PLATFORM_PREFETCH_HPP_
#define GUFO_CORE_PLATFORM_PREFETCH_HPP_

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

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace gufo::platform {
#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)
namespace detail {

inline int PrefaultExceptionFilter(EXCEPTION_POINTERS* exception,
                                   std::uintptr_t begin, std::size_t bytes,
                                   DWORD* error) noexcept {
  const auto* record = exception->ExceptionRecord;
  if ((record->ExceptionCode != EXCEPTION_IN_PAGE_ERROR &&
       record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) ||
      record->NumberParameters < 2)
    return EXCEPTION_CONTINUE_SEARCH;
  const auto address = record->ExceptionInformation[1];
  if (address < begin || address - begin >= bytes)
    return EXCEPTION_CONTINUE_SEARCH;
  *error = record->ExceptionCode == EXCEPTION_IN_PAGE_ERROR ? ERROR_READ_FAULT
                                                            : ERROR_NOACCESS;
  return EXCEPTION_EXECUTE_HANDLER;
}

// Keep SEH in a helper with no C++ objects requiring stack unwinding. Faults
// must be caught on the worker that touches the mapping, before joining it.
inline DWORD TouchMappedPages(std::uintptr_t begin, std::size_t bytes,
                              std::size_t page_size) noexcept {
  DWORD error = ERROR_SUCCESS;
  __try {
    const auto* touch = reinterpret_cast<volatile const std::uint8_t*>(begin);
    for (std::size_t i = 0; i < bytes; i += page_size)
      static_cast<void>(touch[i]);
  } __except (PrefaultExceptionFilter(GetExceptionInformation(), begin, bytes,
                                      &error)) {
  }
  return error;
}

}  // namespace detail
#endif

/// Populate an existing readable mapping before HIP registration or upload.
/// Parallel faults expose disk queue depth without making another weight copy.
/// The mapping must remain alive until this call returns, including on failure.
///
/// Linux faults pages with MADV_POPULATE_READ from several threads; Windows
/// asks the memory manager to prefetch the range into the standby list and
/// then touches pages in parallel so failures surface before registration.
inline void PrefaultMappedRange(const void* data, std::size_t bytes) {
#if defined(__HIP_DEVICE_COMPILE__)
  // Host-only population never runs in a device pass of a .hip translation
  // unit; the device pass only needs this declaration to parse.
  static_cast<void>(data);
  static_cast<void>(bytes);
  return;
#else
  if (bytes == 0)
    return;
#if defined(_WIN32)
  const std::size_t page_size = [] {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                                : static_cast<std::size_t>(4096);
  }();
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
#endif
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
#if defined(_WIN32)
  // Windows 8+; returns immediately, prefetching into the standby list.
  WIN32_MEMORY_RANGE_ENTRY entry{reinterpret_cast<void*>(begin), bytes};
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#endif
  auto populate = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto span = std::min(kChunkBytes, bytes - offset);
#if defined(_WIN32)
      if (const DWORD error =
              detail::TouchMappedPages(begin + offset, span, page_size);
          error != ERROR_SUCCESS)
        failure.store(static_cast<int>(error), std::memory_order_relaxed);
#else
      if (::madvise(reinterpret_cast<void*>(begin + offset), span,
                    MADV_POPULATE_READ) != 0)
        failure.store(errno, std::memory_order_relaxed);
#endif
    }
  };
  {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < std::min(kReaders, chunks); ++i)
      readers.emplace_back(populate);
    populate();
    // Join before a caller can register, upload, or unmap any of these pages.
  }
  if (const int code = failure.load(); code != 0) {
#if defined(_WIN32)
    throw std::system_error(code, std::system_category(),
                            "cannot read mapped weights");
#else
    throw std::system_error(code, std::generic_category(),
                            "cannot read mapped weights");
#endif
  }
#endif  // !__HIP_DEVICE_COMPILE__
}

/// Drop this process's resident pages of a read-only file mapping once its
/// contents have been copied elsewhere (device upload done and synchronized).
/// The mapping stays valid: a later read refaults the page from the file, so
/// this is safe even if a host read remains. Never call it on a range that is
/// hipHostRegister'ed, where the mapping is the GPU-visible weight memory.
/// Best effort: failures are ignored, and it is a no-op on Windows.
inline void ReleaseMappedPages(const void* data, std::size_t bytes) noexcept {
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(_WIN32)
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (!data || bytes == 0 || page_size <= 0)
    return;
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  const auto skip = address % static_cast<std::size_t>(page_size);
  (void)::madvise(reinterpret_cast<void*>(address - skip), bytes + skip,
                  MADV_DONTNEED);
#else
  static_cast<void>(data);
  static_cast<void>(bytes);
#endif
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PREFETCH_HPP_
