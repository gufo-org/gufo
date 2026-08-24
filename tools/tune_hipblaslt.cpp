#include <hip/hip_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"

namespace {

struct GemmShape {
  std::size_t m;
  std::size_t k;
  std::string name;
};

struct Options {
  std::string output_path;
  std::vector<std::size_t> batches{32, 64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<GemmShape> shapes = {
      {12288, 5120, "attention_q_gate"},
      {1024, 5120, "attention_kv"},
      {5120, 6144, "attention_ssm_output"},
      {10240, 5120, "ssm_qkv"},
      {6144, 5120, "ssm_gate"},
      {17408, 5120, "ffn_gate_up"},
      {5120, 17408, "ffn_down"},
  };
  std::uint32_t warmup{2};
  std::uint32_t repetitions{5};
  std::size_t max_algorithms{64};
  std::size_t workspace_bytes{32ULL * 1024 * 1024};
};

class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) {
      throw std::invalid_argument("device buffer must not be empty");
    }
    HIP_CHECK(hipMalloc(&data_, bytes_));
    HIP_CHECK(hipMemset(data_, 0, bytes_));
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* Get() noexcept { return data_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
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

void PrintHelp(std::string_view program) {
  std::cout
      << "Usage: " << program << " --out PATH [options]\n\n"
      << "Offline hipBLASLt autotuner for Qwen3.8 BF16 projection shapes.\n\n"
      << "Options:\n"
      << "  --out PATH             Binary plan database output path\n"
      << "  --batch N,N,...        Prompt batch sizes\n"
      << "  --shape MxK            Replace defaults; may be repeated\n"
      << "  --warmup N             Warmups per algorithm (default: 2)\n"
      << "  --repetitions N        Samples per algorithm (default: 5)\n"
      << "  --max-algorithms N     Candidate limit per shape (default: 64)\n"
      << "  --workspace-mib N      Workspace budget (default: 32)\n"
      << "  --quick                 Tune batch 128, FFN gate/up only\n"
      << "  -h, --help              Print help\n";
}

template<typename Integer>
Integer ParsePositive(std::string_view value, std::string_view option) {
  Integer parsed = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() ||
      parsed == 0) {
    throw std::invalid_argument("invalid value for " + std::string(option));
  }
  return parsed;
}

std::vector<std::size_t> ParseBatches(std::string_view value) {
  std::vector<std::size_t> batches;
  std::size_t start = 0;
  while (start < value.size()) {
    const auto separator = value.find(',', start);
    const auto end =
        separator == std::string_view::npos ? value.size() : separator;
    batches.push_back(ParsePositive<std::size_t>(
        value.substr(start, end - start), "--batch"));
    start = end + 1;
  }
  if (batches.empty()) {
    throw std::invalid_argument("--batch requires at least one size");
  }
  return batches;
}

GemmShape ParseShape(std::string_view value) {
  const auto separator = value.find_first_of("xX");
  if (separator == std::string_view::npos) {
    throw std::invalid_argument("--shape must use MxK");
  }
  const auto m =
      ParsePositive<std::size_t>(value.substr(0, separator), "--shape");
  const auto k =
      ParsePositive<std::size_t>(value.substr(separator + 1), "--shape");
  return {.m = m, .k = k, .name = std::to_string(m) + "x" + std::to_string(k)};
}

Options ParseOptions(std::span<const char* const> arguments) {
  Options options;
  bool custom_shapes = false;
  for (std::size_t index = 1; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    const auto require_value = [&]() -> std::string_view {
      if (++index >= arguments.size()) {
        throw std::invalid_argument("missing value for " +
                                    std::string(argument));
      }
      return arguments[index];
    };
    if (argument == "--out") {
      options.output_path = require_value();
    } else if (argument == "--batch") {
      options.batches = ParseBatches(require_value());
    } else if (argument == "--shape") {
      if (!custom_shapes) {
        options.shapes.clear();
        custom_shapes = true;
      }
      options.shapes.push_back(ParseShape(require_value()));
    } else if (argument == "--warmup") {
      options.warmup =
          ParsePositive<std::uint32_t>(require_value(), "--warmup");
    } else if (argument == "--repetitions") {
      options.repetitions =
          ParsePositive<std::uint32_t>(require_value(), "--repetitions");
    } else if (argument == "--max-algorithms") {
      options.max_algorithms =
          ParsePositive<std::size_t>(require_value(), "--max-algorithms");
    } else if (argument == "--workspace-mib") {
      const auto workspace_mib =
          ParsePositive<std::size_t>(require_value(), "--workspace-mib");
      if (workspace_mib >
          std::numeric_limits<std::size_t>::max() / (1024 * 1024)) {
        throw std::invalid_argument("--workspace-mib is too large");
      }
      options.workspace_bytes = workspace_mib * 1024 * 1024;
    } else if (argument == "--quick") {
      options.batches = {128};
      options.shapes = {{17408, 5120, "ffn_gate_up"}};
      options.max_algorithms =
          std::min<std::size_t>(options.max_algorithms, 16);
    } else if (argument == "--help" || argument == "-h") {
      PrintHelp(arguments.front());
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("--out is required");
  }
  return options;
}

std::size_t CheckedBytes(std::size_t left, std::size_t right,
                         std::size_t element_bytes) {
  if (left > std::numeric_limits<std::size_t>::max() / right ||
      left * right > std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::overflow_error("GEMM buffer size overflow");
  }
  return left * right * element_bytes;
}

int Run(std::span<const char* const> arguments) {
  const Options options = ParseOptions(arguments);
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  if (std::string_view(properties.gcnArchName).find("gfx1151") ==
      std::string_view::npos) {
    throw std::runtime_error("tune_hipblaslt requires gfx1151");
  }

  HipStream stream;
  strix::hip::HipblasLtGemm gemm({
      .plan_database_path = {},
      .tuning_workspace_bytes = options.workspace_bytes,
      .ignore_environment = true,
  });
  const strix::hip::HipblasLtTuningOptions tuning_options{
      .warmup = options.warmup,
      .repetitions = options.repetitions,
      .max_algorithms = options.max_algorithms,
  };

  std::size_t tuned_shapes = 0;
  for (const auto batch_size : options.batches) {
    for (const auto& shape : options.shapes) {
      DeviceBuffer weights(
          CheckedBytes(shape.m, shape.k, sizeof(std::uint16_t)));
      DeviceBuffer input(
          CheckedBytes(batch_size, shape.k, sizeof(std::uint16_t)));
      DeviceBuffer output(CheckedBytes(batch_size, shape.m, sizeof(float)));

      strix::hip::HipblasLtTuningResult result;
      if (!gemm.TuneBf16(weights.Get(), input.Get(),
                         static_cast<float*>(output.Get()), batch_size, shape.m,
                         shape.k, tuning_options, &result, stream.Get())) {
        throw std::runtime_error(
            "no tunable algorithm for batch=" + std::to_string(batch_size) +
            " shape=" + std::to_string(shape.m) + "x" +
            std::to_string(shape.k));
      }
      ++tuned_shapes;
      std::cout << shape.name << " batch=" << batch_size
                << " algorithm=" << result.algorithm_id
                << " heuristic=" << result.heuristic_algorithm_id
                << " workspace=" << result.workspace_bytes
                << " median_us=" << result.median_us
                << " verified_speedup=" << result.verified_speedup
                << " tuned=" << (result.retained_tuned_algorithm ? "yes" : "no")
                << " measured=" << result.measured_algorithms << "/"
                << result.supported_algorithms << '\n';
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream.Get()));

  std::string error;
  if (!gemm.SavePlans(options.output_path, &error)) {
    throw std::runtime_error("failed to save plan database: " + error);
  }
  std::cout << "saved " << tuned_shapes << " plans to " << options.output_path
            << '\n';
  std::cout << "runtime: STRIX_HIPBLASLT_PLAN_CACHE=" << options.output_path
            << '\n';
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return Run(
        std::span<const char* const>(argv, static_cast<std::size_t>(argc)));
  } catch (const std::exception& error) {
    std::cerr << "tune_hipblaslt: " << error.what() << '\n';
    return 1;
  }
}
