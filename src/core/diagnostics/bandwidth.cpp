#include "src/core/diagnostics/bandwidth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include "src/core/diagnostics/bandwidth_run.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::diagnostics {

namespace {

std::string CurrentIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::tm tm_utc{};
  gufo::platform::UtcTime(time_t_now, &tm_utc);

  std::ostringstream oss;
  oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
  return oss.str();
}

std::string EscapeJsonString(std::string_view str) {
  std::ostringstream oss;
  for (const char character : str) {
    if (character == '"') {
      oss << "\\\"";
    } else if (character == '\\') {
      oss << "\\\\";
    } else if (character == '\n') {
      oss << "\\n";
    } else if (character == '\t') {
      oss << "\\t";
    } else {
      oss << character;
    }
  }
  return oss.str();
}

void ComputeBandwidthStatistics(BandwidthPathResult& result) {
  if (result.raw_repetitions_gbps.empty()) {
    return;
  }
  std::vector<double> sorted = result.raw_repetitions_gbps;
  std::ranges::sort(sorted);

  result.min_gbps = sorted.front();
  result.max_gbps = sorted.back();

  // Median
  const std::size_t n = sorted.size();
  if (n % 2 == 0) {
    result.median_gbps = (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  } else {
    result.median_gbps = sorted[n / 2];
  }

  // P95
  const std::size_t p95_idx =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
  result.p95_gbps = sorted[std::min(p95_idx, n - 1)];
}

}  // namespace

std::vector<BandwidthPathResult> MeasureCpuBandwidth(
    const BandwidthOptions& options) {
  std::vector<BandwidthPathResult> results;
  const std::size_t size = options.working_set_bytes;

  std::vector<std::uint8_t> src(size);
  std::vector<std::uint8_t> dst(size);

  // Initialize source buffer with deterministic pseudorandom pattern
  for (std::size_t i = 0; i < size; ++i) {
    src[i] = static_cast<std::uint8_t>((i * 17 + 31) & 0xFF);
  }

  // 1. CPU Copy Path
  {
    BandwidthPathResult path;
    path.backend = "cpu";
    path.path_name = "cpu_copy";
    path.allocation_type = "host_pageable";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    // Warmup
    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      std::memcpy(dst.data(), src.data(), size);
    }

    // Repetitions
    bool sentinel_ok = true;
    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      std::memset(dst.data(), 0, size);
      const auto start = std::chrono::steady_clock::now();
      std::memcpy(dst.data(), src.data(), size);
      const auto end = std::chrono::steady_clock::now();

      const double elapsed_sec =
          std::chrono::duration<double>(end - start).count();
      // Copy moves size bytes read + size bytes written = 2 * size
      run.Record(elapsed_sec, static_cast<double>(size) * 2);

      if (dst[0] != src[0] || dst[size / 2] != src[size / 2] ||
          dst[size - 1] != src[size - 1]) {
        sentinel_ok = false;
      }
    }

    path.sentinel_verified = sentinel_ok;
    ComputeBandwidthStatistics(path);
    results.push_back(std::move(path));
  }

  // 2. CPU Read (Reduction) Path
  {
    BandwidthPathResult path;
    path.backend = "cpu";
    path.path_name = "cpu_read";
    path.allocation_type = "host_pageable";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    // Warmup
    volatile std::uint64_t sum = 0;
    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      std::uint64_t local_sum = 0;
      for (std::size_t i = 0; i < size; ++i) {
        local_sum += src[i];
      }
      sum = local_sum;
    }

    // Repetitions
    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      const auto start = std::chrono::steady_clock::now();
      std::uint64_t local_sum = 0;
      for (std::size_t i = 0; i < size; ++i) {
        local_sum += src[i];
      }
      sum = local_sum;
      const auto end = std::chrono::steady_clock::now();

      const double elapsed_sec =
          std::chrono::duration<double>(end - start).count();
      run.Record(elapsed_sec, static_cast<double>(size));
    }

    path.sentinel_verified = (sum != 0);
    ComputeBandwidthStatistics(path);
    results.push_back(std::move(path));
  }

  // 3. CPU Write (Fill) Path
  {
    BandwidthPathResult path;
    path.backend = "cpu";
    path.path_name = "cpu_write";
    path.allocation_type = "host_pageable";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    // Warmup
    for (std::uint32_t w = 0; w < options.warmup; ++w) {
      std::memset(dst.data(), static_cast<int>(w & 0xFF), size);
    }

    // Repetitions
    bool sentinel_ok = true;
    for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
      const int val = static_cast<int>((path.repetitions + 0x5A) & 0xFF);
      const auto start = std::chrono::steady_clock::now();
      std::memset(dst.data(), val, size);
      const auto end = std::chrono::steady_clock::now();

      const double elapsed_sec =
          std::chrono::duration<double>(end - start).count();
      run.Record(elapsed_sec, static_cast<double>(size));

      if (dst[0] != static_cast<std::uint8_t>(val) ||
          dst[size - 1] != static_cast<std::uint8_t>(val)) {
        sentinel_ok = false;
      }
    }

    path.sentinel_verified = sentinel_ok;
    ComputeBandwidthStatistics(path);
    results.push_back(std::move(path));
  }

  return results;
}

std::string BandwidthReport::ToJson() const {
  std::ostringstream oss;
  oss << "{\n";
  oss << "  \"schemaVersion\": \"" << EscapeJsonString(schema_version)
      << "\",\n";
  oss << "  \"fingerprintId\": \"" << EscapeJsonString(fingerprint_id)
      << "\",\n";
  oss << "  \"engineRevision\": \"" << EscapeJsonString(engine_revision)
      << "\",\n";
  oss << "  \"timestamp\": \"" << EscapeJsonString(timestamp) << "\",\n";

  oss << "  \"options\": {\n";
  oss << "    \"warmup\": " << options.warmup << ",\n";
  oss << "    \"repetitions\": " << options.repetitions << ",\n";
  oss << "    \"durationMs\": " << options.duration_ms << ",\n";
  oss << "    \"workingSetBytes\": " << options.working_set_bytes << "\n";
  oss << "  },\n";

  oss << "  \"paths\": [\n";
  for (std::size_t i = 0; i < paths.size(); ++i) {
    const auto& p = paths[i];
    oss << "    {\n";
    oss << "      \"backend\": \"" << EscapeJsonString(p.backend) << "\",\n";
    oss << "      \"pathName\": \"" << EscapeJsonString(p.path_name) << "\",\n";
    oss << "      \"allocationType\": \"" << EscapeJsonString(p.allocation_type)
        << "\",\n";
    oss << "      \"workingSetBytes\": " << p.working_set_bytes << ",\n";
    oss << "      \"warmupRuns\": " << p.warmup_runs << ",\n";
    oss << "      \"repetitions\": " << p.repetitions << ",\n";
    oss << "      \"elapsedMs\": " << p.elapsed_ms << ",\n";
    oss << "      \"medianGbps\": " << std::fixed << std::setprecision(2)
        << p.median_gbps << ",\n";
    oss << "      \"p95Gbps\": " << std::fixed << std::setprecision(2)
        << p.p95_gbps << ",\n";
    oss << "      \"minGbps\": " << std::fixed << std::setprecision(2)
        << p.min_gbps << ",\n";
    oss << "      \"maxGbps\": " << std::fixed << std::setprecision(2)
        << p.max_gbps << ",\n";
    oss << "      \"sentinelVerified\": "
        << (p.sentinel_verified ? "true" : "false") << ",\n";
    oss << "      \"status\": \"" << EscapeJsonString(p.status) << "\",\n";
    oss << "      \"reason\": \"" << EscapeJsonString(p.reason) << "\",\n";

    oss << "      \"rawRepetitionsGbps\": [";
    for (std::size_t j = 0; j < p.raw_repetitions_gbps.size(); ++j) {
      oss << std::fixed << std::setprecision(2) << p.raw_repetitions_gbps[j];
      if (j + 1 < p.raw_repetitions_gbps.size()) {
        oss << ", ";
      }
    }
    oss << "]\n";
    oss << "    }";
    if (i + 1 < paths.size()) {
      oss << ",";
    }
    oss << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

std::string BandwidthReport::ToHuman() const {
  std::ostringstream oss;
  oss << "=== Sustained Memory Bandwidth Benchmark ===\n";
  oss << "Schema Version  : " << schema_version << "\n";
  oss << "Fingerprint ID  : " << fingerprint_id << "\n";
  oss << "Engine Revision : " << engine_revision << "\n";
  oss << "Timestamp       : " << timestamp << "\n";
  oss << "Working Set     : "
      << (options.working_set_bytes / (1024ULL * 1024ULL)) << " MiB\n\n";

  oss << std::left << std::setw(8) << "Backend" << std::setw(26) << "Path"
      << std::setw(14) << "Median GB/s" << std::setw(12) << "P95 GB/s"
      << std::setw(12) << "Max GB/s" << std::setw(12) << "Status" << "\n";
  oss << std::string(84, '-') << "\n";

  for (const auto& p : paths) {
    oss << std::left << std::setw(8) << p.backend << std::setw(26)
        << p.path_name << std::setw(14)
        << (std::to_string(static_cast<int>(p.median_gbps)) + "." +
            std::to_string(static_cast<int>(p.median_gbps * 100) % 100))
        << std::setw(12)
        << (std::to_string(static_cast<int>(p.p95_gbps)) + "." +
            std::to_string(static_cast<int>(p.p95_gbps * 100) % 100))
        << std::setw(12)
        << (std::to_string(static_cast<int>(p.max_gbps)) + "." +
            std::to_string(static_cast<int>(p.max_gbps * 100) % 100))
        << std::setw(12) << p.status;
    if (!p.reason.empty()) {
      oss << " (" << p.reason << ")";
    }
    oss << "\n";
  }

  return oss.str();
}

BandwidthReport RunBandwidthBenchmark(const BandwidthOptions& options,
                                      const MachineFingerprint& fingerprint) {
  BandwidthReport report;
  report.fingerprint_id = fingerprint.fingerprint_id;
  report.timestamp = CurrentIso8601Utc();
  report.options = options;

  for (const auto& backend : options.backends) {
    if (backend == "cpu") {
      auto cpu_paths = MeasureCpuBandwidth(options);
      report.paths.insert(report.paths.end(), cpu_paths.begin(),
                          cpu_paths.end());
    } else if (backend == "hip") {
      auto hip_paths = MeasureHipBandwidth(options);
      report.paths.insert(report.paths.end(), hip_paths.begin(),
                          hip_paths.end());
    }
  }

  return report;
}

}  // namespace gufo::diagnostics
