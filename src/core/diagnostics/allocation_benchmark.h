#ifndef GUFO_CORE_DIAGNOSTICS_ALLOCATION_BENCHMARK_H_
#define GUFO_CORE_DIAGNOSTICS_ALLOCATION_BENCHMARK_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"

namespace gufo::diagnostics {

struct AllocationBenchmarkOptions {
  std::vector<std::size_t> working_set_bytes{64ULL * 1024ULL * 1024ULL};
  std::uint32_t warmup{1};
  std::uint32_t repetitions{5};
  std::size_t transfer_chunk_bytes{64ULL * 1024ULL * 1024ULL};
};

struct AllocationFaultDelta {
  std::uint64_t minor{0};
  std::uint64_t major{0};
};

struct AllocationPhaseResult {
  std::string name;
  std::string operation;
  std::string unit;
  std::string status{"completed"};
  std::string reason;
  std::vector<double> raw_repetitions;
  double median{0.0};
  double p95{0.0};
  double minimum{0.0};
  double maximum{0.0};
  AllocationFaultDelta faults;
};

struct AllocationPathResult {
  std::string path_name;
  std::string allocation_api;
  std::string flags;
  std::string status{"completed"};
  std::string reason;
  std::string pointer_memory_type;
  std::string residency_evidence;
  std::string placement_inference;
  std::size_t working_set_bytes{0};
  std::uint64_t expected_checksum{0};
  std::uint64_t cpu_checksum{0};
  std::uint64_t gpu_checksum{0};
  bool checksum_verified{false};
  bool managed{false};
  std::vector<AllocationPhaseResult> phases;
};

struct AllocationBenchmarkReport {
  std::string schema_version{"1.0.0"};
  std::string artifact_type{"hipAllocation"};
  std::string fingerprint_id;
  std::string engine_revision{"development"};
  std::string timestamp;
  AllocationBenchmarkOptions options;
  SystemInventory inventory;
  MachineFingerprint fingerprint;
  std::uint64_t page_size_bytes{0};
  std::uint64_t huge_pages_total{0};
  std::uint64_t huge_page_size_bytes{0};
  std::vector<AllocationPathResult> paths;
  std::vector<std::string> limitations;
  bool all_requested_paths_reported{false};
  bool checksums_verified{false};
  bool phases_separated{false};

  [[nodiscard]] bool Success() const noexcept;
  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] bool ParseWorkingSetSizes(std::string_view text,
                                        std::vector<std::size_t>* sizes,
                                        std::string* error = nullptr);

void ComputeAllocationPhaseStatistics(AllocationPhaseResult* phase);

[[nodiscard]] AllocationBenchmarkReport RunHipAllocationBenchmark(
    const AllocationBenchmarkOptions& options, const SystemInventory& inventory,
    const MachineFingerprint& fingerprint);

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_ALLOCATION_BENCHMARK_H_
