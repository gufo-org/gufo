#ifndef GUFO_CORE_PLATFORM_OS_INFO_HPP_
#define GUFO_CORE_PLATFORM_OS_INFO_HPP_

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <optional>
#include <span>
#include <string>

namespace gufo::platform {

/// OS page size in bytes; never zero.
std::size_t PageSizeBytes();

/// Best-effort physically available memory in bytes. Linux reports
/// sysconf(_SC_AVPHYS_PAGES) times the page size; Windows reports
/// GlobalMemoryStatusEx available physical memory. Callers layer their own
/// refinements (MemAvailable parsing, cgroup limits) on top. Returns 0
/// when the OS cannot answer.
std::uint64_t AvailableMemoryBytes();

/// Id of the running process. Used to make temporary output names
/// collision-free between concurrent processes.
int ProcessId() noexcept;

/// The host's uname(2) identity. Windows reports "Windows", the computer
/// name, the NT version (major.minor.build) and the architecture.
struct HostIdentity {
  std::string sysname;
  std::string nodename;
  std::string release;
  std::string version;
  std::string machine;
};

/// Returns nullopt when the OS cannot describe itself.
std::optional<HostIdentity> QueryHostIdentity();

/// The OS Gufo was built for: "Linux" or "Windows".
const char* HostOsName() noexcept;

/// The host's processors. Windows reads the processor name from the
/// registry and the core counts from GetLogicalProcessorInformationEx;
/// Linux reports only the online logical processors (the diagnostics parse
/// /proc/cpuinfo for the rest). Zero or empty fields are unknown.
struct HostCpu {
  std::string model_name;
  std::string vendor;
  std::uint32_t logical_cores{0};
  std::uint32_t physical_cores{0};
};

/// Returns nullopt when the OS cannot describe its processors.
std::optional<HostCpu> QueryHostCpu();

/// Physical memory in bytes. Windows reports GlobalMemoryStatusEx; Linux
/// reports sysconf pages times the page size.
struct HostMemory {
  std::uint64_t total_bytes{0};
  std::uint64_t available_bytes{0};
};

/// Returns nullopt when the OS cannot report its memory.
std::optional<HostMemory> QueryHostMemory();

/// Fills `out` with the process's local calendar time for `seconds`.
/// Returns false when the platform cannot convert the time.
bool LocalTime(std::time_t seconds, std::tm* out) noexcept;

/// Fills `out` with the UTC calendar time for `seconds`. Returns false when
/// the platform cannot convert the time.
bool UtcTime(std::time_t seconds, std::tm* out) noexcept;

/// Whether standard error is attached to a terminal. Log and trace sinks
/// use it to decide between plain and colored output.
bool StderrIsTerminal() noexcept;

/// Minor and major page faults of this process since it started.
struct PageFaults {
  std::uint64_t minor{0};
  std::uint64_t major{0};
};

/// Reads the process page-fault counters. Returns zeros when the
/// platform cannot report them; Windows reports every fault as minor.
PageFaults ReadPageFaults() noexcept;

/// Fills `out` with OS-provided cryptographically random bytes.
bool FillRandomBytes(std::span<std::uint8_t> out) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_OS_INFO_HPP_
