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
#else
#include <psapi.h>

#include <cstdlib>
#include <filesystem>
#include <vector>

// Survives NDEBUG: the release builds run these tests too.
#define REQUIRE(condition)                                                 \
  do {                                                                     \
    if (!(condition)) {                                                    \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
      std::abort();                                                        \
    }                                                                      \
  } while (false)

// Pages of [memory, memory + bytes) in this process's working set.
std::size_t ResidentPages(const void* memory, std::size_t bytes,
                          std::size_t page_size) {
  const std::size_t pages = (bytes + page_size - 1) / page_size;
  std::vector<PSAPI_WORKING_SET_EX_INFORMATION> info(pages);
  for (std::size_t i = 0; i < pages; ++i)
    info[i].VirtualAddress =
        const_cast<char*>(static_cast<const char*>(memory)) + i * page_size;
  REQUIRE(QueryWorkingSetEx(GetCurrentProcess(), info.data(),
                            static_cast<DWORD>(pages * sizeof(info[0]))));
  std::size_t resident = 0;
  for (const auto& entry : info)
    resident += entry.VirtualAttributes.Valid ? 1 : 0;
  return resident;
}

// Windows keeps touched file-view pages in the working set (#829);
// ReleaseMappedPages must trim them while the view keeps its contents.
void CheckReleaseMappedPages(std::size_t page_size) {
  const std::size_t bytes = page_size * 64;
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo_prefault_test_" + std::to_string(GetCurrentProcessId()));
  {
    std::ofstream out(path, std::ios::binary);
    for (std::size_t i = 0; i < bytes; ++i)
      out.put(static_cast<char>(i % 251));
    REQUIRE(out.good());
  }
  HANDLE file =
      CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  REQUIRE(file != INVALID_HANDLE_VALUE);
  HANDLE section =
      CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  REQUIRE(section != nullptr);
  const auto* memory = static_cast<const unsigned char*>(
      MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0));
  REQUIRE(memory != nullptr);
  gufo::platform::PrefaultMappedRange(memory, bytes);
  REQUIRE(ResidentPages(memory, bytes, page_size) == bytes / page_size);
  // Host memory outside the view is never trimmed.
  std::vector<unsigned char> heap(bytes, 1);
  gufo::platform::ReleaseMappedRange(memory, bytes, heap.data(), heap.size());
  REQUIRE(ResidentPages(memory, bytes, page_size) == bytes / page_size);
  // A range clipped to the view: the first partial page up to the end.
  gufo::platform::ReleaseMappedRange(memory, bytes, memory + 1, 2 * bytes);
  REQUIRE(ResidentPages(memory, bytes, page_size) == 0);
  for (std::size_t i = 0; i < bytes; ++i)
    REQUIRE(memory[i] == i % 251);
  REQUIRE(ResidentPages(memory, bytes, page_size) == bytes / page_size);
  gufo::platform::ReleaseMappedPages(memory + 2 * page_size, page_size);
  REQUIRE(ResidentPages(memory, bytes, page_size) == bytes / page_size - 1);
  // A range starting before the view: only its part inside the view.
  const auto* before = reinterpret_cast<const void*>(
      reinterpret_cast<std::uintptr_t>(memory) - page_size);
  gufo::platform::ReleaseMappedRange(memory, bytes, before, 2 * page_size);
  REQUIRE(ResidentPages(memory, bytes, page_size) == bytes / page_size - 2);
  REQUIRE(UnmapViewOfFile(memory));
  REQUIRE(CloseHandle(section));
  REQUIRE(CloseHandle(file));
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
#endif

int main() {
  gufo::platform::ReleaseMappedPages(nullptr, 0);
#if !defined(_WIN32)
  CheckReleaseMappedPages();
#else
  {
    SYSTEM_INFO system{};
    ::GetSystemInfo(&system);
    CheckReleaseMappedPages(system.dwPageSize);
  }
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
#if defined(_WIN32)
  // The fault-injection check below has never run on Windows (the test was
  // POSIX-only). With clang-cl it ends in an access violation instead of
  // ERROR_NOACCESS (07.11.2026, gfx1151 box), so it stays POSIX-only until
  // PrefaultMappedRange's SEH path is looked at separately.
  ::VirtualFree(memory, 0, MEM_RELEASE);
  return 0;
#endif
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
