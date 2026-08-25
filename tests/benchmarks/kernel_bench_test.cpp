#include "src/cli/bench/kernel_bench.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

void TestStatistics() {
  const auto statistics =
      strix::bench::ComputeKernelBenchStatistics({50, 10, 40, 20, 30});
  Expect(std::abs(statistics.mean_us - 30.0) < 1e-9, "mean");
  Expect(std::abs(statistics.median_us - 30.0) < 1e-9, "median");
  Expect(std::abs(statistics.p10_us - 14.0) < 1e-9, "p10");
  Expect(std::abs(statistics.p90_us - 46.0) < 1e-9, "p90");
  Expect(statistics.min_us == 10.0, "minimum");
  Expect(statistics.max_us == 50.0, "maximum");
}

void TestEmptyStatisticsFail() {
  bool threw = false;
  try {
    (void)strix::bench::ComputeKernelBenchStatistics({});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "empty samples must fail");
}

void TestJsonReport() {
  strix::bench::KernelBenchReport report;
  report.fingerprint_id = std::string(64, 'a');
  report.engine_revision = "test";
  report.device_name = "AMD Radeon 8060S Graphics";
  report.gpu_architecture = "gfx1151";
  report.compute_units = 40;
  report.total_memory_bytes = 1024;
  report.options = {.warmup = 2, .repetitions = 3};
  strix::bench::KernelBenchResult result;
  result.kernel = "decode_attention";
  result.backend = "online_fp32";
  result.marker = "decode_attention/context=4096";
  result.data_type = "f32";
  result.layout = "head_major";
  result.batch_size = 1;
  result.context_tokens = 4096;
  result.elements = 6144;
  result.tokens_per_iteration = 1;
  result.num_heads = 24;
  result.num_kv_heads = 4;
  result.head_dim = 256;
  result.estimated_bytes_per_iteration = 1024;
  result.timing = {
      .mean_us = 10,
      .median_us = 9,
      .p10_us = 8,
      .p90_us = 12,
      .min_us = 7,
      .max_us = 13,
  };
  result.tokens_per_second = 111111;
  result.effective_gbps = 100;
  result.correctness_verified = true;
  result.raw_microseconds = {8, 9, 10};
  result.dispatch.selected_attention_backend = "decode_online_fp32";
  result.dispatch.rejected_fast_paths = {"baseline: unsupported"};
  result.dispatch.hipblaslt_plan_source = "persistent";
  result.dispatch.hipblaslt_workspace_bytes = 4096;
  result.dispatch.hipblaslt_plan_resolution_us = 12.5;
  result.dispatch.plan_cache_status = "not_applicable";
  result.dispatch.persistent_plan_cache_status = "hit";
  result.dispatch.graph_cache_status = "hit";
  report.results.push_back(std::move(result));

  const std::string json = report.ToJson();
  Expect(json.find("\"schemaVersion\": \"1.1.0\"") != std::string::npos,
         "schema version");
  Expect(json.find("\"gpuArchitecture\": \"gfx1151\"") != std::string::npos,
         "GPU architecture");
  Expect(json.find("\"medianUs\": 9.000000") != std::string::npos,
         "median timing");
  Expect(json.find("\"batchSize\": 1") != std::string::npos, "batch size");
  Expect(json.find("\"dataType\": \"f32\"") != std::string::npos, "data type");
  Expect(json.find("\"selectedAttentionBackend\": "
                   "\"decode_online_fp32\"") != std::string::npos,
         "attention dispatch");
  Expect(json.find("\"rejectedFastPaths\": "
                   "[\"baseline: unsupported\"]") != std::string::npos,
         "rejected fast paths");
  Expect(json.find("\"graphCacheStatus\": \"hit\"") != std::string::npos,
         "graph cache status");
  Expect(
      json.find("\"hipblasltPlanSource\": \"persistent\"") != std::string::npos,
      "hipBLASLt plan source");
  Expect(json.find("\"hipblasltWorkspaceBytes\": 4096") != std::string::npos,
         "hipBLASLt workspace");
  Expect(
      json.find("\"persistentPlanCacheStatus\": \"hit\"") != std::string::npos,
      "persistent plan cache status");
  Expect(json.find("\"occupancyPercent\": null") != std::string::npos,
         "profiler resource placeholder");
  Expect(json.find("\"correctnessVerified\": true") != std::string::npos,
         "correctness sentinel");
}

}  // namespace

int main() {
  TestStatistics();
  TestEmptyStatisticsFail();
  TestJsonReport();
  std::cout << "All kernel benchmark support tests passed.\n";
  return 0;
}
