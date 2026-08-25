#include "src/cli/bench/kernel_bench.hpp"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace strix::bench {
namespace {

double Percentile(const std::vector<double>& sorted, double quantile) {
  if (sorted.empty()) {
    throw std::invalid_argument("kernel benchmark samples are empty");
  }
  const double position =
      quantile * static_cast<double>(sorted.size() - std::size_t{1});
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction);
}

std::string EscapeJson(std::string_view value) {
  std::ostringstream output;
  for (const char character : value) {
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

void WriteResourceMetrics(std::ostringstream& output,
                          const KernelResourceMetrics& resources) {
  output << "      \"resources\": {\n";
  output << "        \"source\": \"" << EscapeJson(resources.source) << "\",\n";
  if (resources.populated) {
    output << "        \"occupancyPercent\": " << resources.occupancy_percent
           << ",\n";
    output << "        \"vgprCount\": " << resources.vgpr_count << ",\n";
    output << "        \"ldsBytes\": " << resources.lds_bytes << ",\n";
    output << "        \"scratchBytesPerThread\": "
           << resources.scratch_bytes_per_thread << "\n";
  } else {
    output << "        \"occupancyPercent\": null,\n";
    output << "        \"vgprCount\": null,\n";
    output << "        \"ldsBytes\": null,\n";
    output << "        \"scratchBytesPerThread\": null\n";
  }
  output << "      }\n";
}

void WriteStringArray(std::ostringstream& output,
                      const std::vector<std::string>& values) {
  output << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    output << "\"" << EscapeJson(values[index]) << "\"";
    if (index + 1 < values.size()) {
      output << ", ";
    }
  }
  output << "]";
}

void WriteDispatchTelemetry(std::ostringstream& output,
                            const KernelDispatchTelemetry& dispatch) {
  output << "      \"dispatch\": {\n";
  output << "        \"selectedAttentionBackend\": ";
  if (dispatch.selected_attention_backend.empty()) {
    output << "null,\n";
  } else {
    output << "\"" << EscapeJson(dispatch.selected_attention_backend)
           << "\",\n";
  }
  output << "        \"gemvStrategy\": ";
  if (dispatch.gemv_strategy.empty()) {
    output << "null,\n";
  } else {
    output << "\"" << EscapeJson(dispatch.gemv_strategy) << "\",\n";
  }
  output << "        \"rejectedFastPaths\": ";
  WriteStringArray(output, dispatch.rejected_fast_paths);
  output << ",\n";
  output << "        \"hipblasltAlgorithmId\": ";
  if (dispatch.hipblaslt_algorithm_id < 0) {
    output << "null,\n";
  } else {
    output << dispatch.hipblaslt_algorithm_id << ",\n";
  }
  output << "        \"hipblasltSolutionName\": ";
  if (dispatch.hipblaslt_solution_name.empty()) {
    output << "null,\n";
  } else {
    output << "\"" << EscapeJson(dispatch.hipblaslt_solution_name) << "\",\n";
  }
  output << "        \"hipblasltKernelName\": ";
  if (dispatch.hipblaslt_kernel_name.empty()) {
    output << "null,\n";
  } else {
    output << "\"" << EscapeJson(dispatch.hipblaslt_kernel_name) << "\",\n";
  }
  output << "        \"hipblasltPlanSource\": ";
  if (dispatch.hipblaslt_plan_source.empty()) {
    output << "null,\n";
  } else {
    output << "\"" << EscapeJson(dispatch.hipblaslt_plan_source) << "\",\n";
  }
  output << "        \"hipblasltWorkspaceBytes\": "
         << dispatch.hipblaslt_workspace_bytes << ",\n";
  output << "        \"hipblasltPlanResolutionUs\": "
         << dispatch.hipblaslt_plan_resolution_us << ",\n";
  output << "        \"planCacheStatus\": \""
         << EscapeJson(dispatch.plan_cache_status) << "\",\n";
  output << "        \"persistentPlanCacheStatus\": \""
         << EscapeJson(dispatch.persistent_plan_cache_status) << "\",\n";
  output << "        \"graphCacheStatus\": \""
         << EscapeJson(dispatch.graph_cache_status) << "\"\n";
  output << "      },\n";
}

std::string FormatShape(const KernelBenchResult& result) {
  if (result.m != 0 && result.k != 0) {
    std::ostringstream shape;
    if (result.batch_size != 0) {
      shape << result.batch_size << "x";
    }
    shape << result.m << "x" << result.k;
    return shape.str();
  }
  if (result.context_tokens != 0) {
    return std::to_string(result.context_tokens) + " tok";
  }
  return std::to_string(result.elements) + " elem";
}

}  // namespace

KernelBenchStatistics ComputeKernelBenchStatistics(
    const std::vector<double>& samples_us) {
  if (samples_us.empty()) {
    throw std::invalid_argument("kernel benchmark samples are empty");
  }

  std::vector<double> sorted = samples_us;
  std::ranges::sort(sorted);
  const double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

  return {
      .mean_us = sum / static_cast<double>(sorted.size()),
      .median_us = Percentile(sorted, 0.50),
      .p10_us = Percentile(sorted, 0.10),
      .p90_us = Percentile(sorted, 0.90),
      .min_us = sorted.front(),
      .max_us = sorted.back(),
  };
}

std::string KernelBenchReport::ToJson() const {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6);
  output << "{\n";
  output << "  \"schemaVersion\": \"" << EscapeJson(schema_version) << "\",\n";
  output << "  \"fingerprintId\": \"" << EscapeJson(fingerprint_id) << "\",\n";
  output << "  \"engineRevision\": \"" << EscapeJson(engine_revision)
         << "\",\n";
  output << "  \"hardware\": {\n";
  output << "    \"deviceName\": \"" << EscapeJson(device_name) << "\",\n";
  output << "    \"gpuArchitecture\": \"" << EscapeJson(gpu_architecture)
         << "\",\n";
  output << "    \"computeUnits\": " << compute_units << ",\n";
  output << "    \"totalMemoryBytes\": " << total_memory_bytes << "\n";
  output << "  },\n";
  output << "  \"options\": {\n";
  output << "    \"warmup\": " << options.warmup << ",\n";
  output << "    \"repetitions\": " << options.repetitions << "\n";
  output << "  },\n";
  output << "  \"results\": [\n";
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    output << "    {\n";
    output << "      \"kernel\": \"" << EscapeJson(result.kernel) << "\",\n";
    output << "      \"backend\": \"" << EscapeJson(result.backend) << "\",\n";
    output << "      \"marker\": \"" << EscapeJson(result.marker) << "\",\n";
    output << "      \"dataType\": \"" << EscapeJson(result.data_type)
           << "\",\n";
    output << "      \"layout\": \"" << EscapeJson(result.layout) << "\",\n";
    output << "      \"batchSize\": " << result.batch_size << ",\n";
    output << "      \"contextTokens\": " << result.context_tokens << ",\n";
    output << "      \"m\": " << result.m << ",\n";
    output << "      \"n\": " << result.n << ",\n";
    output << "      \"k\": " << result.k << ",\n";
    output << "      \"elements\": " << result.elements << ",\n";
    output << "      \"tokensPerIteration\": " << result.tokens_per_iteration
           << ",\n";
    output << "      \"numHeads\": " << result.num_heads << ",\n";
    output << "      \"numKvHeads\": " << result.num_kv_heads << ",\n";
    output << "      \"headDim\": " << result.head_dim << ",\n";
    output << "      \"estimatedBytesPerIteration\": "
           << result.estimated_bytes_per_iteration << ",\n";
    output << "      \"meanUs\": " << result.timing.mean_us << ",\n";
    output << "      \"medianUs\": " << result.timing.median_us << ",\n";
    output << "      \"p10Us\": " << result.timing.p10_us << ",\n";
    output << "      \"p90Us\": " << result.timing.p90_us << ",\n";
    output << "      \"minUs\": " << result.timing.min_us << ",\n";
    output << "      \"maxUs\": " << result.timing.max_us << ",\n";
    output << "      \"tokensPerSecond\": " << result.tokens_per_second
           << ",\n";
    output << "      \"effectiveGbps\": " << result.effective_gbps << ",\n";
    output << "      \"correctnessVerified\": "
           << (result.correctness_verified ? "true" : "false") << ",\n";
    output << "      \"rawMicroseconds\": [";
    for (std::size_t sample = 0; sample < result.raw_microseconds.size();
         ++sample) {
      output << result.raw_microseconds[sample];
      if (sample + 1 < result.raw_microseconds.size()) {
        output << ", ";
      }
    }
    output << "],\n";
    WriteDispatchTelemetry(output, result.dispatch);
    WriteResourceMetrics(output, result.resources);
    output << "    }";
    if (index + 1 < results.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << "  ]\n";
  output << "}\n";
  return output.str();
}

std::string KernelBenchReport::ToHuman() const {
  std::ostringstream output;
  output << "GPU Kernel Benchmark\n";
  output << "Fingerprint: " << fingerprint_id << "\n";
  output << "Device: " << device_name << " (" << gpu_architecture << ", "
         << compute_units << " CUs)\n";
  output << std::left << std::setw(22) << "Kernel" << std::setw(20) << "Backend"
         << std::right << std::setw(16) << "Shape" << std::setw(14)
         << "Median us" << std::setw(14) << "p10 us" << std::setw(14)
         << "p90 us" << std::setw(14) << "GB/s" << "\n";
  output << std::string(114, '-') << "\n";
  output << std::fixed << std::setprecision(2);
  for (const auto& result : results) {
    output << std::left << std::setw(22) << result.kernel << std::setw(20)
           << result.backend << std::right << std::setw(16)
           << FormatShape(result) << std::setw(14) << result.timing.median_us
           << std::setw(14) << result.timing.p10_us << std::setw(14)
           << result.timing.p90_us << std::setw(14) << result.effective_gbps
           << "\n";
  }
  return output.str();
}

}  // namespace strix::bench
