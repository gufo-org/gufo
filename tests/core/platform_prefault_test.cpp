#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>

#include "src/core/platform/prefault.hpp"

#if !defined(_WIN32)
// mincore reports page-cache residency, which MADV_DONTNEED deliberately
// preserves. smaps measures this process's resident file-mapping pages.
std::size_t ResidentKiB(const void* memory) {
  std::ifstream smaps("/proc/self/smaps");
  assert(smaps);
  std::string line;
  bool in_mapping = false;
  while (std::getline(smaps, line)) {
    unsigned long begin = 0, end = 0;
    if (std::sscanf(line.c_str(), "%lx-%lx", &begin, &end) == 2) {
      const auto address = reinterpret_cast<std::uintptr_t>(memory);
      in_mapping = begin <= address && address < end;
    } else if (in_mapping && line.starts_with("Rss:")) {
      std::istringstream fields(line.substr(4));
      std::size_t kib = 0;
      fields >> kib;
      assert(fields);
      return kib;
    }
  }
  assert(false && "mapping missing from smaps");
  return 0;
}

void CheckReleaseMappedPages() {
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  assert(page_size > 0);
  const auto bytes = static_cast<std::size_t>(page_size) * 4;
  std::FILE* file = std::tmpfile();
  assert(file);
  for (std::size_t i = 0; i < bytes; ++i)
    assert(std::fputc(static_cast<int>(i % 251), file) != EOF);
  assert(std::fflush(file) == 0);
  const auto* memory = static_cast<const unsigned char*>(
      ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, ::fileno(file), 0));
  assert(memory != MAP_FAILED);
  gufo::platform::PrefaultMappedRange(memory, bytes);
  assert(ResidentKiB(memory) == bytes / 1024);
  // Include the first partial page and the final byte of the mapping.
  gufo::platform::ReleaseMappedPages(memory + 1, bytes - 1);
  assert(ResidentKiB(memory) == 0);
  for (std::size_t i = 0; i < bytes; ++i)
    assert(memory[i] == i % 251);
  assert(ResidentKiB(memory) == bytes / 1024);
  assert(::munmap(const_cast<unsigned char*>(memory), bytes) == 0);
  assert(std::fclose(file) == 0);
}
#endif

int main() {
  gufo::platform::ReleaseMappedPages(nullptr, 0);
#if !defined(_WIN32)
  CheckReleaseMappedPages();
#endif
  constexpr std::size_t bytes = 32ULL << 20;
#if defined(_WIN32)
  auto* memory = static_cast<char*>(
      ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  assert(memory != nullptr);
  SYSTEM_INFO info{};
  ::GetSystemInfo(&info);
  const auto page_size = info.dwPageSize;
#else
  auto* memory =
      static_cast<char*>(::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  assert(memory != MAP_FAILED);
  const auto page_size = ::sysconf(_SC_PAGESIZE);
  assert(page_size > 0);
#endif
  gufo::platform::PrefaultMappedRange(memory + 1, bytes - 1);
  // A fault in a later chunk must be reported after all workers have joined.
  auto* protected_page = memory + (16ULL << 20) + page_size;
#if defined(_WIN32)
  DWORD old_protection = 0;
  assert(::VirtualProtect(protected_page, page_size, PAGE_NOACCESS,
                          &old_protection));
#else
  assert(::mprotect(protected_page, page_size, PROT_NONE) == 0);
#endif
  bool failed = false;
  try {
    gufo::platform::PrefaultMappedRange(memory, bytes);
  } catch (const std::system_error& error) {
    failed = true;
#if defined(_WIN32)
    assert(error.code() ==
           std::error_code(ERROR_NOACCESS, std::system_category()));
#endif
  }
  assert(failed);
#if defined(_WIN32)
  assert(::VirtualFree(memory, 0, MEM_RELEASE));
#else
  assert(::munmap(memory, bytes) == 0);
#endif
}
