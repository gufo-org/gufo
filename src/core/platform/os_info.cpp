#include "src/core/platform/os_info.hpp"

#include <cstddef>
#include <string>
#include <vector>

#if defined(_WIN32)
// clang-format off: psapi.h and bcrypt.h need the windows.h types above them,
// and the console and CRT isatty calls need the io and stdio headers.
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <io.h>
#include <stdio.h>
// clang-format on
#else
#include <sys/random.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace gufo::platform {

std::size_t PageSizeBytes() {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                              : static_cast<std::size_t>(4096);
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::size_t>(page_size)
                       : static_cast<std::size_t>(4096);
#endif
}

std::uint64_t AvailableMemoryBytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return 0;
  return status.ullAvailPhys;
#else
  const long pages = ::sysconf(_SC_AVPHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return (pages > 0 && page_size > 0)
             ? static_cast<std::uint64_t>(pages) * page_size
             : 0;
#endif
}

int ProcessId() noexcept {
#if defined(_WIN32)
  return static_cast<int>(GetCurrentProcessId());
#else
  return ::getpid();
#endif
}

std::optional<HostIdentity> QueryHostIdentity() {
#if defined(_WIN32)
  // RtlGetVersion reports the real NT version; GetVersionEx is capped by the
  // application manifest.
  using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
  const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
  const auto get_version = ntdll == nullptr
                               ? nullptr
                               : reinterpret_cast<RtlGetVersionFn>(
                                     GetProcAddress(ntdll, "RtlGetVersion"));
  OSVERSIONINFOW version{};
  version.dwOSVersionInfoSize = sizeof(version);
  if (get_version == nullptr || get_version(&version) != 0)
    return std::nullopt;
  HostIdentity identity;
  identity.sysname = "Windows";
  char name[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD length = sizeof(name);
  if (GetComputerNameA(name, &length))
    identity.nodename.assign(name, length);
  identity.release = std::to_string(version.dwMajorVersion) + "." +
                     std::to_string(version.dwMinorVersion) + "." +
                     std::to_string(version.dwBuildNumber);
  identity.version = "build " + std::to_string(version.dwBuildNumber);
  identity.machine = "x86_64";
  return identity;
#else
  struct utsname uts{};
  if (::uname(&uts) != 0)
    return std::nullopt;
  return HostIdentity{uts.sysname, uts.nodename, uts.release, uts.version,
                      uts.machine};
#endif
}

const char* HostOsName() noexcept {
#if defined(_WIN32)
  return "Windows";
#else
  return "Linux";
#endif
}

std::optional<HostCpu> QueryHostCpu() {
#if defined(_WIN32)
  HostCpu cpu;
  cpu.logical_cores = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (bytes != 0) {
    std::vector<std::byte> buffer(bytes);
    auto* const first =
        reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffer.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, first,
                                         &bytes)) {
      for (DWORD offset = 0; offset < bytes;) {
        const auto* const entry =
            reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                buffer.data() + offset);
        ++cpu.physical_cores;
        offset += entry->Size;
      }
    }
  }
  constexpr const char* kProcessorKey =
      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
  const auto read_string = [&](const char* value) {
    char text[256]{};
    DWORD size = sizeof(text);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, kProcessorKey, value, RRF_RT_REG_SZ,
                     nullptr, text, &size) != ERROR_SUCCESS)
      return std::string();
    std::string result(text);
    const auto end = result.find_last_not_of(' ');
    result.erase(end == std::string::npos ? 0 : end + 1);
    return result;
  };
  cpu.model_name = read_string("ProcessorNameString");
  cpu.vendor = read_string("VendorIdentifier");
  if (cpu.logical_cores == 0 && cpu.physical_cores == 0)
    return std::nullopt;
  return cpu;
#else
  const long online = ::sysconf(_SC_NPROCESSORS_ONLN);
  if (online <= 0)
    return std::nullopt;
  HostCpu cpu;
  cpu.logical_cores = static_cast<std::uint32_t>(online);
  return cpu;
#endif
}

std::optional<HostMemory> QueryHostMemory() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status))
    return std::nullopt;
  return HostMemory{status.ullTotalPhys, status.ullAvailPhys};
#else
  const long pages = ::sysconf(_SC_PHYS_PAGES);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0)
    return std::nullopt;
  return HostMemory{static_cast<std::uint64_t>(pages) * page_size,
                    AvailableMemoryBytes()};
#endif
}

bool LocalTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::localtime_s(out, &seconds) == 0;
#else
  return ::localtime_r(&seconds, out) != nullptr;
#endif
}

bool UtcTime(std::time_t seconds, std::tm* out) noexcept {
#if defined(_WIN32)
  return ::gmtime_s(out, &seconds) == 0;
#else
  return ::gmtime_r(&seconds, out) != nullptr;
#endif
}

bool StderrIsTerminal() noexcept {
#if defined(_WIN32)
  return ::_isatty(::_fileno(stderr)) != 0;
#else
  return ::isatty(STDERR_FILENO) != 0;
#endif
}

PageFaults ReadPageFaults() noexcept {
#if defined(_WIN32)
  // The Windows counter totals soft and hard faults; the split is not
  // reported, so the count lands in the minor field.
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return {};
  return {counters.PageFaultCount, 0};
#else
  struct rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0)
    return {};
  return {static_cast<std::uint64_t>(usage.ru_minflt),
          static_cast<std::uint64_t>(usage.ru_majflt)};
#endif
}

bool FillRandomBytes(std::span<std::uint8_t> out) noexcept {
  if (out.empty())
    return true;
#if defined(_WIN32)
  return BCryptGenRandom(nullptr, out.data(), static_cast<ULONG>(out.size()),
                         BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
  return ::getrandom(out.data(), out.size(), 0) ==
         static_cast<ssize_t>(out.size());
#endif
}

}  // namespace gufo::platform
