#ifndef STRIX_BENCH_KERNEL_BENCH_HPP_
#define STRIX_BENCH_KERNEL_BENCH_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::bench {

struct KernelBenchOptions {
  std::uint32_t warmup{3};
  std::uint32_t repetitions{10};
};

struct KernelBenchStatistics {
  double mean_us{0.0};
  double median_us{0.0};
  double p10_us{0.0};
  double p90_us{0.0};
  double min_us{0.0};
  double max_us{0.0};
};

struct KernelResourceMetrics {
  std::string source{"rocprofv3"};
  bool populated{false};
  double occupancy_percent{0.0};
  std::uint32_t vgpr_count{0};
  std::uint32_t lds_bytes{0};
  std::uint32_t scratch_bytes_per_thread{0};
};

struct KernelDispatchTelemetry {
  std::string selected_attention_backend;
  std::string gemv_strategy;
  std::vector<std::string> rejected_fast_paths;
  std::int64_t hipblaslt_algorithm_id{-1};
  std::string hipblaslt_solution_name;
  std::string hipblaslt_kernel_name;
  std::string hipblaslt_plan_source;
  std::uint64_t hipblaslt_workspace_bytes{0};
  double hipblaslt_plan_resolution_us{0.0};
  std::string plan_cache_status{"not_applicable"};
  std::string persistent_plan_cache_status{"not_applicable"};
  std::string graph_cache_status{"not_applicable"};
};

struct KernelBenchResult {
  std::string kernel;
  std::string backend;
  std::string marker;
  std::string data_type;
  std::string layout;
  std::size_t batch_size{0};
  std::size_t context_tokens{0};
  std::size_t m{0};
  std::size_t n{0};
  std::size_t k{0};
  std::uint64_t elements{0};
  std::uint64_t tokens_per_iteration{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint64_t estimated_bytes_per_iteration{0};
  KernelBenchStatistics timing;
  double tokens_per_second{0.0};
  double effective_gbps{0.0};
  bool correctness_verified{false};
  std::vector<double> raw_microseconds;
  KernelDispatchTelemetry dispatch;
  KernelResourceMetrics resources;
};

struct KernelBenchReport {
  std::string schema_version{"1.1.0"};
  std::string fingerprint_id;
  std::string engine_revision{"development"};
  std::string device_name;
  std::string gpu_architecture;
  std::uint32_t compute_units{0};
  std::uint64_t total_memory_bytes{0};
  KernelBenchOptions options;
  std::vector<KernelBenchResult> results;

  [[nodiscard]] std::string ToJson() const;
  [[nodiscard]] std::string ToHuman() const;
};

[[nodiscard]] KernelBenchStatistics ComputeKernelBenchStatistics(
    const std::vector<double>& samples_us);

}  // namespace strix::bench

#endif  // STRIX_BENCH_KERNEL_BENCH_HPP_
