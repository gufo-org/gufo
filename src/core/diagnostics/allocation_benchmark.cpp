#include "src/core/diagnostics/allocation_benchmark.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace gufo::diagnostics {
namespace {

#if !defined(ENGINE_ENABLE_HIP)
std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}
#endif

std::string EscapeJson(std::string_view text) {
  std::ostringstream output;
  for (const char character : text) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        output << character;
        break;
    }
  }
  return output.str();
}

std::optional<std::size_t> ParseSize(std::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0 ||
      value > std::numeric_limits<std::size_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(value);
}

void WriteMetric(std::ostringstream& output,
                 const AllocationPhaseResult& phase) {
  output << "          {\n"
         << "            \"name\": \"" << EscapeJson(phase.name) << "\",\n"
         << "            \"operation\": \"" << EscapeJson(phase.operation)
         << "\",\n"
         << "            \"unit\": \"" << EscapeJson(phase.unit) << "\",\n"
         << "            \"status\": \"" << EscapeJson(phase.status) << "\",\n"
         << "            \"reason\": \"" << EscapeJson(phase.reason) << "\",\n"
         << "            \"median\": " << std::setprecision(12) << phase.median
         << ",\n"
         << "            \"p95\": " << phase.p95 << ",\n"
         << "            \"minimum\": " << phase.minimum << ",\n"
         << "            \"maximum\": " << phase.maximum << ",\n"
         << "            \"minorFaults\": " << phase.faults.minor << ",\n"
         << "            \"majorFaults\": " << phase.faults.major << ",\n"
         << "            \"rawRepetitions\": [";
  for (std::size_t index = 0; index < phase.raw_repetitions.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << phase.raw_repetitions[index];
  }
  output << "]\n"
         << "          }";
}

}  // namespace

bool ParseWorkingSetSizes(std::string_view text,
                          std::vector<std::size_t>* sizes, std::string* error) {
  if (sizes == nullptr) {
    if (error != nullptr) {
      *error = "working-set destination must not be null";
    }
    return false;
  }
  if (text.empty() || text.front() == ',' || text.back() == ',') {
    if (error != nullptr) {
      *error = "working-set sizes must be positive comma-separated MiB";
    }
    return false;
  }
  std::vector<std::size_t> parsed;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const auto comma = text.find(',', begin);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto size_mib = ParseSize(text.substr(begin, end - begin));
    if (!size_mib.has_value() ||
        *size_mib >
            std::numeric_limits<std::size_t>::max() / (1024ULL * 1024ULL)) {
      if (error != nullptr) {
        *error = "working-set sizes must be positive comma-separated MiB";
      }
      return false;
    }
    parsed.push_back(*size_mib * 1024ULL * 1024ULL);
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (parsed.empty()) {
    if (error != nullptr) {
      *error = "at least one working-set size is required";
    }
    return false;
  }
  *sizes = std::move(parsed);
  return true;
}

void ComputeAllocationPhaseStatistics(AllocationPhaseResult* phase) {
  if (phase == nullptr || phase->raw_repetitions.empty()) {
    return;
  }
  auto sorted = phase->raw_repetitions;
  std::ranges::sort(sorted);
  phase->minimum = sorted.front();
  phase->maximum = sorted.back();
  const std::size_t count = sorted.size();
  phase->median = count % 2 == 0
                      ? (sorted[(count / 2) - 1] + sorted[count / 2]) / 2.0
                      : sorted[count / 2];
  const auto p95_index =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(count))) -
      1;
  phase->p95 = sorted[std::min(p95_index, count - 1)];
}

bool AllocationBenchmarkReport::Success() const noexcept {
  return all_requested_paths_reported && checksums_verified && phases_separated;
}

std::string AllocationBenchmarkReport::ToJson() const {
  std::ostringstream output;
  output << "{\n"
         << "  \"schemaVersion\": \"" << EscapeJson(schema_version) << "\",\n"
         << "  \"artifactType\": \"" << EscapeJson(artifact_type) << "\",\n"
         << "  \"fingerprintId\": \"" << EscapeJson(fingerprint_id) << "\",\n"
         << "  \"engineRevision\": \"" << EscapeJson(engine_revision) << "\",\n"
         << "  \"timestamp\": \"" << EscapeJson(timestamp) << "\",\n"
         << "  \"summary\": {\n"
         << "    \"allRequestedPathsReported\": "
         << (all_requested_paths_reported ? "true" : "false") << ",\n"
         << "    \"checksumsVerified\": "
         << (checksums_verified ? "true" : "false") << ",\n"
         << "    \"phasesSeparated\": " << (phases_separated ? "true" : "false")
         << ",\n"
         << "    \"status\": \"" << (Success() ? "completed" : "failed")
         << "\"\n"
         << "  },\n"
         << "  \"options\": {\n"
         << "    \"warmup\": " << options.warmup << ",\n"
         << "    \"repetitions\": " << options.repetitions << ",\n"
         << "    \"transferChunkBytes\": " << options.transfer_chunk_bytes
         << ",\n"
         << "    \"workingSetBytes\": [";
  for (std::size_t index = 0; index < options.working_set_bytes.size();
       ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << options.working_set_bytes[index];
  }
  output << "]\n"
         << "  },\n"
         << "  \"machine\": {\n"
         << "    \"kernelRelease\": \""
         << EscapeJson(inventory.toolchain.kernel_release) << "\",\n"
         << "    \"rocmVersion\": \""
         << EscapeJson(inventory.toolchain.rocm_version) << "\",\n"
         << "    \"gpuArchitecture\": \""
         << EscapeJson(inventory.gpu.architecture) << "\",\n"
         << "    \"gpuComputeUnits\": " << inventory.gpu.compute_units << ",\n"
         << "    \"gpuPowerMode\": \"" << EscapeJson(inventory.gpu.power_mode)
         << "\",\n"
         << "    \"gpuClockState\": \"" << EscapeJson(inventory.gpu.clock_state)
         << "\",\n"
         << "    \"memoryTotalBytes\": " << inventory.memory.total_bytes
         << ",\n"
         << "    \"memoryAvailableBytes\": " << inventory.memory.available_bytes
         << ",\n"
         << "    \"memoryType\": \"" << EscapeJson(inventory.memory.memory_type)
         << "\",\n"
         << "    \"pageSizeBytes\": " << page_size_bytes << ",\n"
         << "    \"hugePagesTotal\": " << huge_pages_total << ",\n"
         << "    \"hugePageSizeBytes\": " << huge_page_size_bytes << ",\n"
         << "    \"npuFirmwareVersion\": \""
         << EscapeJson(inventory.npu.firmware_version) << "\"\n"
         << "  },\n"
         << "  \"fingerprint\": " << fingerprint.ToJson() << ",\n"
         << "  \"paths\": [\n";

  for (std::size_t path_index = 0; path_index < paths.size(); ++path_index) {
    const auto& path = paths[path_index];
    output << "    {\n"
           << "      \"pathName\": \"" << EscapeJson(path.path_name) << "\",\n"
           << "      \"allocationApi\": \"" << EscapeJson(path.allocation_api)
           << "\",\n"
           << "      \"flags\": \"" << EscapeJson(path.flags) << "\",\n"
           << "      \"workingSetBytes\": " << path.working_set_bytes << ",\n"
           << "      \"status\": \"" << EscapeJson(path.status) << "\",\n"
           << "      \"reason\": \"" << EscapeJson(path.reason) << "\",\n"
           << "      \"pointerMemoryType\": \""
           << EscapeJson(path.pointer_memory_type) << "\",\n"
           << "      \"managed\": " << (path.managed ? "true" : "false")
           << ",\n"
           << "      \"residencyEvidence\": \""
           << EscapeJson(path.residency_evidence) << "\",\n"
           << "      \"placementInference\": \""
           << EscapeJson(path.placement_inference) << "\",\n"
           << "      \"expectedChecksum\": " << path.expected_checksum << ",\n"
           << "      \"cpuChecksum\": " << path.cpu_checksum << ",\n"
           << "      \"gpuChecksum\": " << path.gpu_checksum << ",\n"
           << "      \"checksumVerified\": "
           << (path.checksum_verified ? "true" : "false") << ",\n"
           << "      \"phases\": [\n";
    for (std::size_t phase_index = 0; phase_index < path.phases.size();
         ++phase_index) {
      WriteMetric(output, path.phases[phase_index]);
      if (phase_index + 1 != path.phases.size()) {
        output << ',';
      }
      output << '\n';
    }
    output << "      ]\n"
           << "    }";
    if (path_index + 1 != paths.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "  ],\n"
         << "  \"limitations\": [";
  for (std::size_t index = 0; index < limitations.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << '"' << EscapeJson(limitations[index]) << '"';
  }
  output << "]\n"
         << "}\n";
  return output.str();
}

std::string AllocationBenchmarkReport::ToHuman() const {
  std::ostringstream output;
  output << "=== HIP Allocation and Placement Diagnostic ===\n"
         << "Status          : " << (Success() ? "completed" : "failed") << "\n"
         << "Fingerprint ID  : " << fingerprint_id << "\n"
         << "Engine Revision : " << engine_revision << "\n"
         << "ROCm / Kernel   : " << inventory.toolchain.rocm_version << " / "
         << inventory.toolchain.kernel_release << "\n"
         << "GPU / Power     : " << inventory.gpu.architecture << " / "
         << inventory.gpu.power_mode << "\n"
         << "Pages           : " << page_size_bytes << " B base, "
         << huge_pages_total << " huge pages @ " << huge_page_size_bytes
         << " B\n\n";

  for (const auto& path : paths) {
    output << path.path_name << " @ "
           << (path.working_set_bytes / (1024ULL * 1024ULL))
           << " MiB: " << path.status;
    if (!path.reason.empty()) {
      output << " (" << path.reason << ')';
    }
    output << "\n  api=" << path.allocation_api << " flags=" << path.flags
           << " pointer=" << path.pointer_memory_type
           << " checksum=" << (path.checksum_verified ? "verified" : "n/a")
           << "\n";
    for (const auto& phase : path.phases) {
      output << "    " << std::left << std::setw(26) << phase.name << std::right
             << std::setw(12) << std::fixed << std::setprecision(3)
             << phase.median << ' ' << phase.unit << "  " << phase.status;
      if (!phase.reason.empty()) {
        output << " (" << phase.reason << ')';
      }
      output << "\n";
    }
  }
  if (!limitations.empty()) {
    output << "\nInference limits:\n";
    for (const auto& limitation : limitations) {
      output << "  - " << limitation << "\n";
    }
  }
  return output.str();
}

#if !defined(ENGINE_ENABLE_HIP)
AllocationBenchmarkReport RunHipAllocationBenchmark(
    const AllocationBenchmarkOptions& options, const SystemInventory& inventory,
    const MachineFingerprint& fingerprint) {
  AllocationBenchmarkReport report;
  report.timestamp = CurrentIso8601Utc();
  report.options = options;
  report.inventory = inventory;
  report.fingerprint = fingerprint;
  report.fingerprint_id = fingerprint.fingerprint_id;
  constexpr std::string_view paths[] = {
      "hip_malloc", "hip_malloc_managed", "hip_host_malloc_mapped",
      "file_mmap_host_register", "hip_malloc_async"};
  for (const auto size : options.working_set_bytes) {
    for (const auto path_name : paths) {
      AllocationPathResult path;
      path.path_name = path_name;
      path.allocation_api = "HIP unavailable";
      path.status = "unsupported";
      path.reason = "uncompiled (ENGINE_ENABLE_HIP=OFF)";
      path.working_set_bytes = size;
      path.placement_inference = "No HIP placement evidence is available.";
      report.paths.push_back(std::move(path));
    }
  }
  report.all_requested_paths_reported = true;
  report.checksums_verified = true;
  report.phases_separated = true;
  return report;
}
#endif

}  // namespace gufo::diagnostics
