#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/bench/kernel_bench.hpp"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/hip/detail/gemv_dispatcher.hpp"
#include "src/models/qwen/hip/detail/qwen_attention_policy.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/qwen_gpu_ops.hpp"
#include "src/core/model_config.hpp"

#ifndef STRIX_VERSION
#define STRIX_VERSION "development"
#endif

namespace {

constexpr std::uint32_t kAttentionHeads = 24;
constexpr std::uint32_t kAttentionKvHeads = 4;
constexpr std::uint32_t kAttentionHeadDim = 256;
constexpr std::uint32_t kFloatPatternBits = 0x3c3c3c3cU;
constexpr std::uint16_t kBf16PatternBits = 0x3f3fU;

struct CommandLineOptions {
  std::vector<std::string> kernels{"decode-attention"};
  std::vector<std::size_t> contexts = {128,  512,  1024,  2048,
                                       4096, 8192, 16384, 32768};
  std::size_t m{4096};
  std::size_t k{4096};
  std::string data_type{"bf16"};
  std::uint32_t warmup{3};
  std::uint32_t repetitions{10};
  std::string output_path;
  bool json{false};
};

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    if (count_ == 0) {
      throw std::invalid_argument("HIP buffer element count must be positive");
    }
    HIP_CHECK(hipMalloc(&data_, count_ * sizeof(T)));
  }

  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* Get() noexcept { return data_; }
  [[nodiscard]] const T* Get() const noexcept { return data_; }
  [[nodiscard]] std::size_t Count() const noexcept { return count_; }
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

class HipStream {
public:
  HipStream() { HIP_CHECK(hipStreamCreate(&stream_)); }
  ~HipStream() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }

  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;

  [[nodiscard]] hipStream_t Get() const noexcept { return stream_; }

private:
  hipStream_t stream_{nullptr};
};

class HipEvent {
public:
  HipEvent() { HIP_CHECK(hipEventCreate(&event_)); }
  ~HipEvent() {
    if (event_ != nullptr) {
      (void)hipEventDestroy(event_);
    }
  }

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  [[nodiscard]] hipEvent_t Get() const noexcept { return event_; }

private:
  hipEvent_t event_{nullptr};
};

class HipblasHandle {
public:
  HipblasHandle() { HIPBLAS_CHECK(hipblasCreate(&handle_)); }
  ~HipblasHandle() {
    if (handle_ != nullptr) {
      (void)hipblasDestroy(handle_);
    }
  }

  HipblasHandle(const HipblasHandle&) = delete;
  HipblasHandle& operator=(const HipblasHandle&) = delete;

  [[nodiscard]] hipblasHandle_t Get() const noexcept { return handle_; }

private:
  hipblasHandle_t handle_{nullptr};
};

void PrintHelp(std::string_view program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Benchmark existing gfx1151 GPU kernels with unprofiled HIP-event "
         "timing.\n\n"
      << "Options:\n"
      << "  --kernel <LIST>       Comma-separated cases: decode-attention, "
         "gemv,\n"
      << "                        gemm, batched-attention, deltanet, "
         "elementwise,\n"
      << "                        or all (default: decode-attention)\n"
      << "  --context <n,n,...>   Context/batch matrix (default: standard "
         "128..32768)\n"
      << "  --m <N>               GEMV/GEMM output rows (default: 4096)\n"
      << "  --k <N>               GEMV/GEMM input columns and element width\n"
      << "                        (default: 4096)\n"
      << "  --type <f32|bf16>     GEMV/GEMM weight/input type (default: bf16)\n"
      << "  --warmup <N>          Warmup launches per case (default: 3)\n"
      << "  --repetitions <N>     Measured launches per case (default: 10)\n"
      << "  --json                Print the structured JSON report\n"
      << "  --output <PATH>       Write the JSON report to PATH\n"
      << "  -h, --help            Print help\n";
}

template<typename Integer>
Integer ParsePositiveInteger(std::string_view value,
                             std::string_view option_name) {
  Integer parsed = 0;
  const auto [ptr, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || ptr != value.data() + value.size() ||
      parsed == 0) {
    throw std::invalid_argument("invalid value for " +
                                std::string(option_name));
  }
  return parsed;
}

std::vector<std::string> ParseStringList(std::string_view input,
                                         std::string_view option_name) {
  std::vector<std::string> values;
  std::size_t start = 0;
  while (start < input.size()) {
    const auto separator = input.find(',', start);
    const auto end =
        separator == std::string_view::npos ? input.size() : separator;
    const auto value = input.substr(start, end - start);
    if (value.empty()) {
      throw std::invalid_argument("invalid value for " +
                                  std::string(option_name));
    }
    values.emplace_back(value);
    start = end + 1;
  }
  if (values.empty()) {
    throw std::invalid_argument("empty value for " + std::string(option_name));
  }
  return values;
}

std::vector<std::size_t> ParseContexts(std::string_view input) {
  std::vector<std::size_t> contexts;
  for (const auto& value : ParseStringList(input, "--context")) {
    contexts.push_back(ParsePositiveInteger<std::size_t>(value, "--context"));
  }
  return contexts;
}

std::vector<std::string> ExpandKernels(std::vector<std::string> kernels) {
  constexpr std::array<std::string_view, 6> supported = {
      "decode-attention",  "gemv",     "gemm",
      "batched-attention", "deltanet", "elementwise"};
  if (std::ranges::find(kernels, "all") != kernels.end()) {
    if (kernels.size() != 1) {
      throw std::invalid_argument("--kernel all cannot be combined");
    }
    kernels.clear();
    for (const std::string_view kernel : supported) {
      kernels.emplace_back(kernel);
    }
    return kernels;
  }

  for (const auto& kernel : kernels) {
    if (std::ranges::find(supported, kernel) == supported.end()) {
      throw std::invalid_argument("unsupported kernel: " + kernel);
    }
  }
  std::ranges::sort(kernels);
  kernels.erase(std::unique(kernels.begin(), kernels.end()), kernels.end());
  return kernels;
}

CommandLineOptions ParseOptions(std::span<const char* const> args) {
  CommandLineOptions options;
  for (std::size_t index = 1; index < args.size(); ++index) {
    const std::string_view argument = args[index];
    if (argument == "-h" || argument == "--help") {
      PrintHelp(args.front());
      std::exit(0);
    }
    if (argument == "--json") {
      options.json = true;
      continue;
    }
    if (index + 1 >= args.size()) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value = args[++index];
    if (argument == "--kernel") {
      options.kernels = ParseStringList(value, argument);
    } else if (argument == "--context") {
      options.contexts = ParseContexts(value);
    } else if (argument == "--m") {
      options.m = ParsePositiveInteger<std::size_t>(value, argument);
    } else if (argument == "--k") {
      options.k = ParsePositiveInteger<std::size_t>(value, argument);
    } else if (argument == "--type") {
      options.data_type = value;
    } else if (argument == "--warmup") {
      options.warmup = ParsePositiveInteger<std::uint32_t>(value, argument);
    } else if (argument == "--repetitions") {
      options.repetitions =
          ParsePositiveInteger<std::uint32_t>(value, argument);
    } else if (argument == "--output") {
      options.output_path = value;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  options.kernels = ExpandKernels(std::move(options.kernels));
  if (options.data_type != "f32" && options.data_type != "bf16") {
    throw std::invalid_argument("--type must be f32 or bf16");
  }
  return options;
}

float FloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float Bf16FromBits(std::uint16_t bits) {
  return FloatFromBits(static_cast<std::uint32_t>(bits) << 16U);
}

template<typename T>
void FillBytes(HipBuffer<T>& buffer, int byte_value, hipStream_t stream) {
  HIP_CHECK(
      hipMemsetAsync(buffer.Get(), byte_value, buffer.SizeBytes(), stream));
}

template<typename T>
void CopyToDevice(HipBuffer<T>& buffer, std::span<const T> values,
                  hipStream_t stream) {
  if (values.size() != buffer.Count()) {
    throw std::invalid_argument("host/device buffer sizes do not match");
  }
  HIP_CHECK(hipMemcpyAsync(buffer.Get(), values.data(), buffer.SizeBytes(),
                           hipMemcpyHostToDevice, stream));
}

bool NearlyEqual(float actual, float expected, float relative_tolerance,
                 float absolute_tolerance) {
  const float difference = std::abs(actual - expected);
  return difference <=
         std::max(absolute_tolerance,
                  relative_tolerance * std::max(std::abs(expected), 1.0F));
}

void VerifyUniformOutput(const HipBuffer<float>& output, float expected,
                         float relative_tolerance = 1e-3F,
                         float absolute_tolerance = 1e-3F) {
  std::array<float, 2> samples{};
  HIP_CHECK(hipMemcpy(&samples[0], output.Get(), sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&samples[1], output.Get() + output.Count() - 1,
                      sizeof(float), hipMemcpyDeviceToHost));
  for (const float sample : samples) {
    if (!std::isfinite(sample) ||
        !NearlyEqual(sample, expected, relative_tolerance,
                     absolute_tolerance)) {
      throw std::runtime_error("kernel correctness sentinel mismatch");
    }
  }
}

void VerifyFiniteNonzero(const HipBuffer<float>& output) {
  const std::size_t sample_count = std::min<std::size_t>(output.Count(), 256);
  std::vector<float> samples(sample_count);
  HIP_CHECK(hipMemcpy(samples.data(), output.Get(),
                      sample_count * sizeof(float), hipMemcpyDeviceToHost));
  const bool finite = std::ranges::all_of(
      samples, [](float value) { return std::isfinite(value); });
  const bool nonzero = std::ranges::any_of(
      samples, [](float value) { return std::abs(value) > 1e-12F; });
  if (!finite || !nonzero) {
    throw std::runtime_error("kernel correctness sentinel failed");
  }
}

template<typename Launcher>
std::vector<double> MeasureKernel(const CommandLineOptions& options,
                                  std::string_view marker, hipStream_t stream,
                                  Launcher&& launch) {
  for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
    launch();
    HIP_CHECK(hipGetLastError());
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HipEvent start;
  HipEvent stop;
  std::vector<double> samples_us;
  samples_us.reserve(options.repetitions);
  const std::string marker_text{marker};
  roctxProfilerResume(0);
  for (std::uint32_t repetition = 0; repetition < options.repetitions;
       ++repetition) {
    HIP_CHECK(hipEventRecord(start.Get(), stream));
    (void)roctxRangePushA(marker_text.c_str());
    launch();
    (void)roctxRangePop();
    HIP_CHECK(hipEventRecord(stop.Get(), stream));
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventSynchronize(stop.Get()));
    float elapsed_ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start.Get(), stop.Get()));
    samples_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
  }
  roctxProfilerPause(0);
  return samples_us;
}

strix::bench::KernelBenchResult FinalizeResult(
    strix::bench::KernelBenchResult result, std::vector<double> samples_us) {
  result.timing = strix::bench::ComputeKernelBenchStatistics(samples_us);
  if (result.tokens_per_iteration != 0) {
    result.tokens_per_second =
        (static_cast<double>(result.tokens_per_iteration) * 1'000'000.0) /
        result.timing.median_us;
  }
  result.effective_gbps =
      static_cast<double>(result.estimated_bytes_per_iteration) /
      (result.timing.median_us * 1000.0);
  result.raw_microseconds = std::move(samples_us);
  result.correctness_verified = true;
  return result;
}

std::uint64_t EstimateDecodeAttentionBytes(std::size_t context) {
  const std::uint64_t cache_reads =
      static_cast<std::uint64_t>(kAttentionHeads) * context *
      kAttentionHeadDim * 2U * sizeof(float);
  const std::uint64_t query_gate_and_output =
      static_cast<std::uint64_t>(kAttentionHeads) * kAttentionHeadDim * 3U *
      sizeof(float);
  const std::uint64_t cache_writes =
      static_cast<std::uint64_t>(kAttentionKvHeads) * kAttentionHeadDim *
      ((2U * sizeof(float)) + (2U * sizeof(std::uint16_t)));
  return cache_reads + query_gate_and_output + cache_writes;
}

strix::bench::KernelBenchResult BenchmarkDecodeAttention(
    std::size_t context, const CommandLineOptions& options,
    hipStream_t stream) {
  const std::size_t query_elements =
      static_cast<std::size_t>(kAttentionHeads) * kAttentionHeadDim;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kAttentionKvHeads) * kAttentionHeadDim;
  const std::size_t cache_elements = kv_elements * context;

  HipBuffer<float> query(query_elements);
  HipBuffer<float> gate(query_elements);
  HipBuffer<float> key(kv_elements);
  HipBuffer<float> value(kv_elements);
  HipBuffer<float> key_cache(cache_elements);
  HipBuffer<float> value_cache(cache_elements);
  HipBuffer<std::uint16_t> key_cache_f16(cache_elements);
  HipBuffer<std::uint16_t> value_cache_f16(cache_elements);
  HipBuffer<float> output(query_elements);
  HipBuffer<float> split_k_scratch(
      strix::hip::detail::DecodeAttentionScratchElements(kAttentionHeads,
                                                         kAttentionHeadDim));

  FillBytes(query, 0x3c, stream);
  FillBytes(gate, 0x3c, stream);
  FillBytes(key, 0x3c, stream);
  FillBytes(value, 0x3d, stream);
  FillBytes(key_cache, 0, stream);
  FillBytes(value_cache, 0, stream);
  FillBytes(key_cache_f16, 0, stream);
  FillBytes(value_cache_f16, 0, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  const std::uint32_t split_count =
      strix::hip::detail::SelectDecodeAttentionSplitCount(context);
  const bool use_split_k = strix::hip::detail::IsSplitKDecodeAttentionSupported(
      context, kAttentionHeads, kAttentionKvHeads, kAttentionHeadDim);
  const std::string backend = use_split_k ? "split_k_fp32" : "online_fp32";
  const std::string marker =
      "decode_attention/context=" + std::to_string(context) +
      "/heads=24/kv_heads=4/head_dim=256/backend=" + backend +
      "/splits=" + std::to_string(split_count);
  const auto launch = [&] {
    strix::hip::LaunchAttention(
        query.Get(), key.Get(), value.Get(), gate.Get(), key_cache.Get(),
        value_cache.Get(), key_cache_f16.Get(), value_cache_f16.Get(),
        output.Get(), 0, static_cast<std::uint32_t>(context - 1),
        static_cast<std::uint32_t>(context), kAttentionHeads, kAttentionKvHeads,
        kAttentionHeadDim, stream, split_k_scratch.Get());
  };
  auto samples_us = MeasureKernel(options, marker, stream, launch);
  VerifyFiniteNonzero(output);

  strix::bench::KernelBenchResult result;
  result.kernel = "decode_attention";
  result.backend = backend;
  result.marker = marker;
  result.data_type = "f32";
  result.layout = "head_major";
  result.batch_size = 1;
  result.context_tokens = context;
  result.elements = query_elements;
  result.tokens_per_iteration = 1;
  result.num_heads = kAttentionHeads;
  result.num_kv_heads = kAttentionKvHeads;
  result.head_dim = kAttentionHeadDim;
  result.estimated_bytes_per_iteration = EstimateDecodeAttentionBytes(context);
  if (use_split_k) {
    result.estimated_bytes_per_iteration +=
        2U *
        strix::hip::detail::DecodeAttentionScratchElements(kAttentionHeads,
                                                           kAttentionHeadDim) *
        sizeof(float);
    result.dispatch.selected_attention_backend = "decode_split_k_fp32";
  } else {
    result.dispatch.selected_attention_backend = "decode_online_fp32";
    result.dispatch.rejected_fast_paths.emplace_back(
        "decode_split_k_fp32: below_threshold");
  }
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkGemv(const CommandLineOptions& options,
                                              hipStream_t stream) {
  const bool is_bf16 = options.data_type == "bf16";
  HipBuffer<float> input(options.k);
  HipBuffer<float> output(options.m);
  FillBytes(input, 0x3c, stream);

  const auto strategy =
      strix::hip::detail::SelectGemvStrategy(options.m, options.k, is_bf16);
  const std::string marker =
      "gemv/m=" + std::to_string(options.m) +
      "/k=" + std::to_string(options.k) + "/type=" + options.data_type +
      "/layout=row_major/strategy=" +
      std::string(strix::hip::detail::GemvStrategyName(strategy));
  std::vector<double> samples_us;
  float weight_value = 0.0F;
  if (is_bf16) {
    HipBuffer<std::uint16_t> weights(options.m * options.k);
    FillBytes(weights, 0x3f, stream);
    weight_value = Bf16FromBits(kBf16PatternBits);
    samples_us = MeasureKernel(options, marker, stream, [&] {
      strix::hip::LaunchGEMV(weights.Get(), strix::core::GgmlType::kBF16, input.Get(), output.Get(),
                             options.m, options.k, stream);
    });
  } else {
    HipBuffer<float> weights(options.m * options.k);
    FillBytes(weights, 0x3c, stream);
    weight_value = FloatFromBits(kFloatPatternBits);
    samples_us = MeasureKernel(options, marker, stream, [&] {
      strix::hip::LaunchGEMV(weights.Get(), strix::core::GgmlType::kF32, input.Get(), output.Get(),
                             options.m, options.k, stream);
    });
  }

  const float input_value = FloatFromBits(kFloatPatternBits);
  const float expected =
      static_cast<float>(options.k) * weight_value * input_value;
  VerifyUniformOutput(output, expected, 2e-3F, 2e-2F);
  const std::uint64_t element_bytes =
      is_bf16 ? sizeof(std::uint16_t) : sizeof(float);
  const std::uint64_t estimated_bytes =
      (static_cast<std::uint64_t>(options.m) * options.k * element_bytes) +
      (static_cast<std::uint64_t>(options.k) * sizeof(float)) +
      (static_cast<std::uint64_t>(options.m) * sizeof(float));

  strix::bench::KernelBenchResult result;
  result.kernel = "gemv";
  result.backend = "custom_hip";
  result.marker = marker;
  result.data_type = options.data_type;
  result.layout = "row_major";
  result.batch_size = 1;
  result.m = options.m;
  result.n = 1;
  result.k = options.k;
  result.elements = static_cast<std::uint64_t>(options.m) * options.k;
  result.tokens_per_iteration = 1;
  result.estimated_bytes_per_iteration = estimated_bytes;
  result.dispatch.gemv_strategy =
      std::string(strix::hip::detail::GemvStrategyName(strategy));
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkGemm(std::size_t batch_size,
                                              const CommandLineOptions& options,
                                              hipStream_t stream) {
  const bool is_bf16 = options.data_type == "bf16";
  HipBuffer<float> output(batch_size * options.m);
  HipblasHandle hipblas;
  strix::hip::HipblasLtDispatchInfo dispatch_info;
  bool used_hipblaslt = false;
  bool hipblaslt_rejected = false;
  bool dispatch_captured = false;
  std::vector<double> samples_us;

  const std::string marker = "gemm/batch=" + std::to_string(batch_size) +
                             "/m=" + std::to_string(options.m) +
                             "/k=" + std::to_string(options.k) +
                             "/type=" + options.data_type + "/layout=row_major";
  float operand_value = 0.0F;
  if (is_bf16) {
    HipBuffer<std::uint16_t> weights(options.m * options.k);
    HipBuffer<std::uint16_t> input(batch_size * options.k);
    FillBytes(weights, 0x3f, stream);
    FillBytes(input, 0x3f, stream);
    operand_value = Bf16FromBits(kBf16PatternBits);
    strix::hip::HipblasLtGemm hipblaslt;
    samples_us = MeasureKernel(options, marker, stream, [&] {
      strix::hip::HipblasLtDispatchInfo current_info;
      auto* current_info_ptr = dispatch_captured ? nullptr : &current_info;
      if (hipblaslt.RunBf16(weights.Get(), input.Get(), output.Get(),
                            batch_size, options.m, options.k, stream,
                            current_info_ptr)) {
        used_hipblaslt = true;
        if (current_info_ptr != nullptr) {
          dispatch_info = std::move(current_info);
          dispatch_captured = true;
        }
      } else {
        hipblaslt_rejected = true;
        strix::hip::LaunchHipblasGEMMBF16(hipblas.Get(), weights.Get(),
                                          input.Get(), output.Get(), batch_size,
                                          options.m, options.k, stream);
      }
    });
  } else {
    HipBuffer<float> weights(options.m * options.k);
    HipBuffer<float> input(batch_size * options.k);
    FillBytes(weights, 0x3c, stream);
    FillBytes(input, 0x3c, stream);
    operand_value = FloatFromBits(kFloatPatternBits);
    samples_us = MeasureKernel(options, marker, stream, [&] {
      strix::hip::LaunchHipblasGEMM(hipblas.Get(), weights.Get(), false,
                                    input.Get(), output.Get(), batch_size,
                                    options.m, options.k, nullptr, stream);
    });
  }

  const float expected =
      static_cast<float>(options.k) * operand_value * operand_value;
  VerifyUniformOutput(output, expected, 2e-3F, 5e-2F);
  const std::uint64_t element_bytes =
      is_bf16 ? sizeof(std::uint16_t) : sizeof(float);
  const std::uint64_t estimated_bytes =
      (static_cast<std::uint64_t>(options.m) * options.k * element_bytes) +
      (static_cast<std::uint64_t>(batch_size) * options.k * element_bytes) +
      (static_cast<std::uint64_t>(batch_size) * options.m * sizeof(float));

  strix::bench::KernelDispatchTelemetry dispatch;
  if (used_hipblaslt) {
    dispatch.hipblaslt_algorithm_id = dispatch_info.algorithm_id;
    dispatch.hipblaslt_solution_name = dispatch_info.solution_name;
    dispatch.hipblaslt_kernel_name = dispatch_info.kernel_name;
    dispatch.hipblaslt_plan_source = dispatch_info.plan_source;
    dispatch.hipblaslt_workspace_bytes = dispatch_info.workspace_bytes;
    dispatch.hipblaslt_plan_resolution_us = dispatch_info.plan_resolution_us;
    dispatch.plan_cache_status = "hit";
    dispatch.persistent_plan_cache_status =
        dispatch_info.persistent_cache_status;
  } else if (hipblaslt_rejected) {
    dispatch.rejected_fast_paths.push_back("hipblaslt: no supported plan");
    dispatch.plan_cache_status = "miss";
    dispatch.persistent_plan_cache_status = "miss";
  }

  strix::bench::KernelBenchResult result;
  result.kernel = "gemm";
  result.backend = used_hipblaslt ? "hipblaslt" : "hipblas";
  result.marker = marker;
  result.data_type = options.data_type;
  result.layout = "row_major";
  result.batch_size = batch_size;
  result.context_tokens = batch_size;
  result.m = options.m;
  result.n = batch_size;
  result.k = options.k;
  result.elements =
      static_cast<std::uint64_t>(batch_size) * options.m * options.k;
  result.tokens_per_iteration = batch_size;
  result.estimated_bytes_per_iteration = estimated_bytes;
  result.dispatch = std::move(dispatch);
  return FinalizeResult(std::move(result), std::move(samples_us));
}

std::uint64_t EstimateBatchedAttentionBytes(std::size_t context,
                                            bool uses_f16_cache) {
  const std::uint64_t attention_width =
      static_cast<std::uint64_t>(kAttentionHeads) * kAttentionHeadDim;
  const std::uint64_t kv_width =
      static_cast<std::uint64_t>(kAttentionKvHeads) * kAttentionHeadDim;
  const std::uint64_t sequence_pairs =
      static_cast<std::uint64_t>(context) * (context + 1U) / 2U;
  const std::uint64_t cache_element_bytes =
      uses_f16_cache ? sizeof(std::uint16_t) : sizeof(float);
  return (static_cast<std::uint64_t>(context) * attention_width * 3U *
          sizeof(float)) +
         (static_cast<std::uint64_t>(context) * kv_width * 2U * sizeof(float)) +
         (sequence_pairs * attention_width * 2U * cache_element_bytes);
}

strix::bench::KernelBenchResult BenchmarkBatchedAttention(
    std::size_t context, const CommandLineOptions& options,
    hipStream_t stream) {
  const std::size_t attention_width =
      static_cast<std::size_t>(kAttentionHeads) * kAttentionHeadDim;
  const std::size_t kv_width =
      static_cast<std::size_t>(kAttentionKvHeads) * kAttentionHeadDim;
  const std::size_t query_elements = context * attention_width;
  const std::size_t kv_elements = context * kv_width;
  const std::size_t cache_elements = context * kv_width;

  HipBuffer<float> query(query_elements);
  HipBuffer<float> key(kv_elements);
  HipBuffer<float> value(kv_elements);
  HipBuffer<float> gate(query_elements);
  HipBuffer<float> key_cache(cache_elements);
  HipBuffer<float> value_cache(cache_elements);
  HipBuffer<std::uint16_t> key_cache_f16(cache_elements);
  HipBuffer<std::uint16_t> value_cache_f16(cache_elements);
  HipBuffer<std::uint16_t> scratch_f16(query_elements);
  HipBuffer<float> output(query_elements);
  FillBytes(query, 0x3c, stream);
  FillBytes(key, 0x3c, stream);
  FillBytes(value, 0x3d, stream);
  FillBytes(gate, 0x3c, stream);
  FillBytes(key_cache, 0, stream);
  FillBytes(value_cache, 0, stream);
  FillBytes(key_cache_f16, 0, stream);
  FillBytes(value_cache_f16, 0, stream);

  std::string selected_backend;
  std::vector<std::string> rejected_fast_paths;
  const std::string marker =
      "batched_attention/context=" + std::to_string(context) +
      "/heads=24/kv_heads=4/head_dim=256/backend=auto";
  const auto launch = [&] {
    if (!strix::hip::detail::ShouldAttemptOptimizedAttention(context)) {
      selected_backend = "baseline";
      strix::hip::LaunchBatchedAttention(
          query.Get(), key.Get(), value.Get(), gate.Get(), key_cache.Get(),
          value_cache.Get(), key_cache_f16.Get(), value_cache_f16.Get(),
          output.Get(), 0, 0, context, static_cast<std::uint32_t>(context),
          kAttentionHeads, kAttentionKvHeads, kAttentionHeadDim, stream);
      return;
    }
    if (strix::hip::LaunchBatchedAttentionTile(
            query.Get(), key.Get(), value.Get(), gate.Get(), key_cache.Get(),
            value_cache.Get(), key_cache_f16.Get(), value_cache_f16.Get(),
            output.Get(), 0, 0, context, static_cast<std::uint32_t>(context),
            kAttentionHeads, kAttentionKvHeads, kAttentionHeadDim, stream)) {
      selected_backend = "tiled";
      return;
    }
    if (std::ranges::find(rejected_fast_paths, "tiled: rejected") ==
        rejected_fast_paths.end()) {
      rejected_fast_paths.emplace_back("tiled: rejected");
    }
    if (strix::hip::LaunchBatchedAttentionCk(
            query.Get(), key.Get(), value.Get(), gate.Get(), key_cache.Get(),
            value_cache.Get(), key_cache_f16.Get(), value_cache_f16.Get(),
            scratch_f16.Get(), output.Get(), 0, 0, context,
            static_cast<std::uint32_t>(context), kAttentionHeads,
            kAttentionKvHeads, kAttentionHeadDim, stream)) {
      selected_backend = "composable_kernel";
      return;
    }
    if (std::ranges::find(rejected_fast_paths, "composable_kernel: rejected") ==
        rejected_fast_paths.end()) {
      rejected_fast_paths.emplace_back("composable_kernel: rejected");
    }
    selected_backend = "baseline";
    strix::hip::LaunchBatchedAttention(
        query.Get(), key.Get(), value.Get(), gate.Get(), key_cache.Get(),
        value_cache.Get(), key_cache_f16.Get(), value_cache_f16.Get(),
        output.Get(), 0, 0, context, static_cast<std::uint32_t>(context),
        kAttentionHeads, kAttentionKvHeads, kAttentionHeadDim, stream);
  };
  auto samples_us = MeasureKernel(options, marker, stream, launch);
  VerifyFiniteNonzero(output);

  if (!strix::hip::detail::ShouldAttemptOptimizedAttention(context)) {
    rejected_fast_paths = {
        "tiled: below optimized-attention threshold",
        "composable_kernel: below optimized-attention threshold",
    };
  }
  const bool uses_f16_cache =
      selected_backend == "tiled" || selected_backend == "composable_kernel";
  strix::bench::KernelBenchResult result;
  result.kernel = "batched_attention";
  result.backend = selected_backend;
  result.marker = marker;
  result.data_type = uses_f16_cache ? "f16/f32" : "f32";
  result.layout = "token_head_major";
  result.batch_size = context;
  result.context_tokens = context;
  result.elements = query_elements;
  result.tokens_per_iteration = context;
  result.num_heads = kAttentionHeads;
  result.num_kv_heads = kAttentionKvHeads;
  result.head_dim = kAttentionHeadDim;
  result.estimated_bytes_per_iteration =
      EstimateBatchedAttentionBytes(context, uses_f16_cache);
  result.dispatch.selected_attention_backend = selected_backend;
  result.dispatch.rejected_fast_paths = std::move(rejected_fast_paths);
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkDeltaNet(
    std::size_t batch_size, const CommandLineOptions& options,
    hipStream_t stream) {
  const strix::core::ModelConfig config;
  const std::uint32_t num_key_heads = config.ssm_group_count;
  const std::uint32_t num_heads = config.ssm_time_step_rank;
  const std::uint32_t key_dim = config.ssm_state_size;
  const std::uint32_t val_dim = config.SsmValueSize();
  const std::size_t qkv_size = config.SsmQkvSize();
  const std::size_t inner_size = config.ssm_inner_size;
  const std::size_t delta_size =
      static_cast<std::size_t>(num_heads) * key_dim * val_dim;

  HipBuffer<float> qkv(batch_size * qkv_size);
  HipBuffer<float> conv_weights(qkv_size * config.ssm_conv_kernel);
  HipBuffer<float> conv_state(qkv_size * config.ssm_conv_kernel);
  HipBuffer<float> conv_output(batch_size * qkv_size);
  HipBuffer<float> deltanet_state(delta_size);
  HipBuffer<float> alpha(batch_size * num_heads);
  HipBuffer<float> beta(batch_size * num_heads);
  HipBuffer<float> ssm_a(num_heads);
  HipBuffer<float> ssm_dt(num_heads);
  HipBuffer<float> ssm_norm(val_dim);
  HipBuffer<float> gate(batch_size * inner_size);
  HipBuffer<float> output(batch_size * inner_size);

  FillBytes(qkv, 0x3c, stream);
  FillBytes(conv_weights, 0x3c, stream);
  FillBytes(conv_state, 0, stream);
  FillBytes(conv_output, 0, stream);
  FillBytes(deltanet_state, 0, stream);
  FillBytes(alpha, 0, stream);
  FillBytes(beta, 0, stream);
  FillBytes(gate, 0x3c, stream);
  const std::vector<float> host_a(num_heads, -0.05F);
  const std::vector<float> host_dt(num_heads, 0.01F);
  const std::vector<float> host_norm(val_dim, 1.0F);
  CopyToDevice(ssm_a, std::span<const float>{host_a}, stream);
  CopyToDevice(ssm_dt, std::span<const float>{host_dt}, stream);
  CopyToDevice(ssm_norm, std::span<const float>{host_norm}, stream);

  const std::string marker = "deltanet/batch=" + std::to_string(batch_size) +
                             "/qkv=" + std::to_string(qkv_size) +
                             "/heads=" + std::to_string(num_heads) +
                             "/key_dim=" + std::to_string(key_dim) +
                             "/val_dim=" + std::to_string(val_dim);
  auto samples_us = MeasureKernel(options, marker, stream, [&] {
    strix::hip::LaunchBatchedSSMConvRecurrence(
        qkv.Get(), conv_weights.Get(), conv_state.Get(), conv_output.Get(),
        deltanet_state.Get(), alpha.Get(), beta.Get(), ssm_a.Get(),
        ssm_dt.Get(), ssm_norm.Get(), gate.Get(), output.Get(), 0, batch_size,
        qkv_size, num_key_heads, num_heads, key_dim, val_dim, stream);
  });
  VerifyFiniteNonzero(output);

  const std::uint64_t streaming_elements =
      (static_cast<std::uint64_t>(batch_size) * qkv_size * 2U) +
      (static_cast<std::uint64_t>(batch_size) * inner_size * 2U) +
      (static_cast<std::uint64_t>(batch_size) * num_heads * 2U);
  const std::uint64_t persistent_elements =
      (static_cast<std::uint64_t>(qkv_size) * config.ssm_conv_kernel * 2U) +
      (static_cast<std::uint64_t>(delta_size) * 2U);
  strix::bench::KernelBenchResult result;
  result.kernel = "deltanet_recurrence";
  result.backend = "batched_gpu";
  result.marker = marker;
  result.data_type = "f32";
  result.layout = "token_head_major";
  result.batch_size = batch_size;
  result.context_tokens = batch_size;
  result.m = inner_size;
  result.k = qkv_size;
  result.elements = static_cast<std::uint64_t>(batch_size) * inner_size;
  result.tokens_per_iteration = batch_size;
  result.num_heads = num_heads;
  result.num_kv_heads = num_key_heads;
  result.head_dim = key_dim;
  result.estimated_bytes_per_iteration =
      (streaming_elements + persistent_elements) * sizeof(float);
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkRmsNorm(
    std::size_t batch_size, const CommandLineOptions& options,
    hipStream_t stream) {
  const std::size_t elements = batch_size * options.k;
  HipBuffer<float> input(elements);
  HipBuffer<float> weight(options.k);
  HipBuffer<float> output(elements);
  FillBytes(input, 0x3c, stream);
  const std::vector<float> host_weight(options.k, 1.25F);
  CopyToDevice(weight, std::span<const float>{host_weight}, stream);

  const std::string marker = "rmsnorm/batch=" + std::to_string(batch_size) +
                             "/dim=" + std::to_string(options.k);
  auto samples_us = MeasureKernel(options, marker, stream, [&] {
    strix::hip::LaunchBatchedRMSNorm(input.Get(), weight.Get(), output.Get(),
                                     nullptr, batch_size, options.k, 1e-6F,
                                     stream);
  });
  const float input_value = FloatFromBits(kFloatPatternBits);
  const float expected =
      (input_value / std::sqrt((input_value * input_value) + 1e-6F)) * 1.25F;
  VerifyUniformOutput(output, expected, 2e-3F, 2e-3F);

  strix::bench::KernelBenchResult result;
  result.kernel = "rmsnorm";
  result.backend = "batched_gpu";
  result.marker = marker;
  result.data_type = "f32";
  result.layout = "row_major";
  result.batch_size = batch_size;
  result.context_tokens = batch_size;
  result.m = batch_size;
  result.k = options.k;
  result.elements = elements;
  result.tokens_per_iteration = batch_size;
  result.estimated_bytes_per_iteration =
      (static_cast<std::uint64_t>(elements) * 2U * sizeof(float)) +
      (static_cast<std::uint64_t>(options.k) * sizeof(float));
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkResidualAdd(
    std::size_t batch_size, const CommandLineOptions& options,
    hipStream_t stream) {
  const std::size_t elements = batch_size * options.k;
  HipBuffer<float> a(elements);
  HipBuffer<float> b(elements);
  HipBuffer<float> output(elements);
  FillBytes(a, 0x3c, stream);
  FillBytes(b, 0x3d, stream);

  const std::string marker =
      "residual_add/batch=" + std::to_string(batch_size) +
      "/dim=" + std::to_string(options.k);
  auto samples_us = MeasureKernel(options, marker, stream, [&] {
    strix::hip::LaunchBatchedResidualAdd(a.Get(), b.Get(), output.Get(),
                                         batch_size, options.k, stream);
  });
  const float expected =
      FloatFromBits(kFloatPatternBits) + FloatFromBits(0x3d3d3d3dU);
  VerifyUniformOutput(output, expected);

  strix::bench::KernelBenchResult result;
  result.kernel = "residual_add";
  result.backend = "batched_gpu";
  result.marker = marker;
  result.data_type = "f32";
  result.layout = "row_major";
  result.batch_size = batch_size;
  result.context_tokens = batch_size;
  result.m = batch_size;
  result.k = options.k;
  result.elements = elements;
  result.tokens_per_iteration = batch_size;
  result.estimated_bytes_per_iteration =
      static_cast<std::uint64_t>(elements) * 3U * sizeof(float);
  return FinalizeResult(std::move(result), std::move(samples_us));
}

strix::bench::KernelBenchResult BenchmarkSwiGlu(
    std::size_t batch_size, const CommandLineOptions& options,
    hipStream_t stream) {
  const std::size_t elements = batch_size * options.k;
  HipBuffer<float> gate(elements);
  HipBuffer<float> up(elements);
  HipBuffer<float> output(elements);
  FillBytes(gate, 0x3c, stream);
  FillBytes(up, 0x3d, stream);

  const std::string marker = "swiglu/batch=" + std::to_string(batch_size) +
                             "/dim=" + std::to_string(options.k);
  auto samples_us = MeasureKernel(options, marker, stream, [&] {
    strix::hip::LaunchBatchedSwiGLUActivation(
        gate.Get(), up.Get(), output.Get(), nullptr, elements, stream);
  });
  const float gate_value = FloatFromBits(kFloatPatternBits);
  const float up_value = FloatFromBits(0x3d3d3d3dU);
  const float expected =
      (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
  VerifyUniformOutput(output, expected);

  strix::bench::KernelBenchResult result;
  result.kernel = "swiglu";
  result.backend = "batched_gpu";
  result.marker = marker;
  result.data_type = "f32";
  result.layout = "row_major";
  result.batch_size = batch_size;
  result.context_tokens = batch_size;
  result.m = batch_size;
  result.k = options.k;
  result.elements = elements;
  result.tokens_per_iteration = batch_size;
  result.estimated_bytes_per_iteration =
      static_cast<std::uint64_t>(elements) * 3U * sizeof(float);
  return FinalizeResult(std::move(result), std::move(samples_us));
}

void AppendKernelResults(strix::bench::KernelBenchReport& report,
                         const CommandLineOptions& options,
                         hipStream_t stream) {
  for (const auto& kernel : options.kernels) {
    if (kernel == "decode-attention") {
      for (const std::size_t context : options.contexts) {
        report.results.push_back(
            BenchmarkDecodeAttention(context, options, stream));
      }
    } else if (kernel == "gemv") {
      report.results.push_back(BenchmarkGemv(options, stream));
    } else if (kernel == "gemm") {
      for (const std::size_t batch_size : options.contexts) {
        report.results.push_back(BenchmarkGemm(batch_size, options, stream));
      }
    } else if (kernel == "batched-attention") {
      for (const std::size_t context : options.contexts) {
        report.results.push_back(
            BenchmarkBatchedAttention(context, options, stream));
      }
    } else if (kernel == "deltanet") {
      for (const std::size_t batch_size : options.contexts) {
        report.results.push_back(
            BenchmarkDeltaNet(batch_size, options, stream));
      }
    } else if (kernel == "elementwise") {
      for (const std::size_t batch_size : options.contexts) {
        report.results.push_back(BenchmarkRmsNorm(batch_size, options, stream));
        report.results.push_back(
            BenchmarkResidualAdd(batch_size, options, stream));
        report.results.push_back(BenchmarkSwiGlu(batch_size, options, stream));
      }
    }
  }
}

int Run(std::span<const char* const> args) {
  const auto options = ParseOptions(args);
  for (const std::size_t context : options.contexts) {
    if (context > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("context exceeds uint32 range");
    }
  }

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (std::string_view(properties.gcnArchName).find("gfx1151") ==
      std::string_view::npos) {
    throw std::runtime_error("strix-kernel-bench requires gfx1151");
  }

  const auto inventory = strix::diagnostics::CollectSystemInventory();
  const auto fingerprint =
      strix::diagnostics::GenerateMachineFingerprint(inventory);

  strix::bench::KernelBenchReport report;
  report.fingerprint_id = fingerprint.fingerprint_id;
  report.engine_revision = STRIX_VERSION;
  report.device_name = properties.name;
  report.gpu_architecture = properties.gcnArchName;
  report.compute_units = properties.multiProcessorCount;
  report.total_memory_bytes = properties.totalGlobalMem;
  report.options = {
      .warmup = options.warmup,
      .repetitions = options.repetitions,
  };

  HipStream stream;
  AppendKernelResults(report, options, stream.Get());

  const std::string json = report.ToJson();
  if (!options.output_path.empty()) {
    std::ofstream output(options.output_path);
    if (!output) {
      throw std::runtime_error("failed to open benchmark output path");
    }
    output << json;
  }
  std::cout << (options.json ? json : report.ToHuman());
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return Run(
        std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& error) {
    std::cerr << "strix-kernel-bench: " << error.what() << "\n";
    return 1;
  }
}
