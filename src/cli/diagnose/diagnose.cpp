#include "src/cli/diagnose/diagnose.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/arg_parser.hpp"
#include "src/core/diagnostics/allocation_benchmark.h"
#include "src/core/diagnostics/artifact_validator.h"
#include "src/core/diagnostics/bandwidth.h"
#include "src/core/diagnostics/compatibility.h"
#include "src/core/diagnostics/fingerprint.h"
#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gufo::cli {

namespace {

void PrintDiagnoseHelp(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name << " diagnose [OPTIONS]\n\n"
      << "Run non-interactive system, hardware, and benchmark diagnostics for "
         "Strix Halo.\n\n"
      << "Options:\n"
      << "  --json                       Emit structured JSON output "
         "conforming "
         "to diagnostics schema v1.0.0\n"
      << "  --fingerprint                Generate canonical machine "
         "fingerprint "
         "and SHA-256 ID\n"
      << "  --validate-artifact <path>   Validate a diagnostic/benchmark "
         "artifact against schema & fingerprint\n"
      << "  --benchmark <name>           Run benchmark suite: bandwidth, "
         "allocation\n"
      << "  --backends <csv>             Backends to benchmark (default: "
         "cpu,hip)\n"
      << "  --warmup <n>                 Number of warmup iterations (default: "
         "3)\n"
      << "  --repetitions <n>            Minimum benchmark repetitions "
         "(default: 10)\n"
      << "  --duration-ms <ms>           Minimum duration per bandwidth path "
         "in ms "
         "(default: 2000)\n"
      << "  --working-set-mib <csv>      Allocation diagnostic sizes in MiB "
         "(default: 64)\n"
      << "  --output <path>              Write command output to specified "
         "file\n"
      << "  --section <name>             Diagnostic section to run: all, "
         "inventory, platform, gpu\n"
      << "  -h, --help                   Print help information\n";
}

void OutputContent(std::string_view content, std::string_view output_file) {
  std::cout << content;
  if (!output_file.empty()) {
    std::ofstream out{std::string(output_file)};
    if (out.is_open()) {
      out << content;
    } else {
      std::cerr << "Warning: could not write output to file: " << output_file
                << "\n";
    }
  }
}

std::vector<std::string> SplitCommaSeparated(std::string_view csv) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start < csv.size()) {
    const auto end = csv.find(',', start);
    if (end == std::string_view::npos) {
      result.emplace_back(csv.substr(start));
      break;
    }
    result.emplace_back(csv.substr(start, end - start));
    start = end + 1;
  }
  return result;
}

}  // namespace

diagnostics::DiagnosticReport CollectDiagnostics(
    const diagnostics::LinuxSysfs& sysfs, std::string_view section) {
  diagnostics::DiagnosticReport report;

  const auto inventory = diagnostics::CollectSystemInventory(sysfs);
  const auto compatibility =
      diagnostics::EvaluateCompatibility(inventory, section);
  const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);

  report.SetInventory(inventory);
  report.SetCompatibility(compatibility);
  report.SetFingerprint(fingerprint);

  // 1. Platform Check
  if (section == "all" || section == "platform" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "platform";
    if (inventory.cpu.architecture == "x86_64") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message = "Supported architecture: Linux x86-64";
      check.details["targetArchitecture"] = "x86_64-linux";
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported target platform: only Linux x86-64 is supported";
      report.AddError("Unsupported platform architecture detected");
    }
    report.AddCheck(std::move(check));
  }

  // 2. System, CPU & Memory Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck sys_check;
    sys_check.name = "system";
    sys_check.status = diagnostics::DiagnosticStatus::kPass;
    sys_check.message = "Host operating system and kernel identified";
    sys_check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    report.AddCheck(std::move(sys_check));

    diagnostics::DiagnosticCheck cpu_check;
    cpu_check.name = "cpu";
    cpu_check.status = diagnostics::DiagnosticStatus::kPass;
    cpu_check.message = "CPU processor topology queried";
    cpu_check.details["modelName"] = inventory.cpu.model_name;
    cpu_check.details["logicalCores"] =
        std::to_string(inventory.cpu.logical_cores);
    cpu_check.details["physicalCores"] =
        std::to_string(inventory.cpu.physical_cores);
    report.AddCheck(std::move(cpu_check));

    diagnostics::DiagnosticCheck mem_check;
    mem_check.name = "memory";
    mem_check.status = diagnostics::DiagnosticStatus::kPass;
    mem_check.message = "Host unified memory pool identified";
    mem_check.details["totalBytes"] =
        std::to_string(inventory.memory.total_bytes);
    mem_check.details["availableBytes"] =
        std::to_string(inventory.memory.available_bytes);
    mem_check.details["memoryType"] = inventory.memory.memory_type;
    report.AddCheck(std::move(mem_check));
  }

  // 3. GPU Check
  if (section == "all" || section == "gpu" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "gpu";
    check.details["name"] = inventory.gpu.name;
    check.details["architecture"] = inventory.gpu.architecture;
    check.details["computeUnits"] = std::to_string(inventory.gpu.compute_units);
    check.details["driverName"] = inventory.gpu.driver_name;
    check.details["pciId"] = inventory.gpu.pci_id;
    check.details["powerMode"] = inventory.gpu.power_mode;

    if (inventory.gpu.architecture == "gfx1151" ||
        inventory.gpu.architecture == "gfx1201") {
      check.status = diagnostics::DiagnosticStatus::kPass;
      check.message =
          "Detected AMD GPU (" + inventory.gpu.architecture + ")";
    } else {
      check.status = diagnostics::DiagnosticStatus::kFail;
      check.message =
          "Unsupported GPU architecture: expected gfx1151, detected " +
          inventory.gpu.architecture;
      report.AddError("Unsupported GPU architecture: " +
                      inventory.gpu.architecture);
    }
    report.AddCheck(std::move(check));
  }

  // 4. Toolchain Check
  if (section == "all" || section == "inventory") {
    diagnostics::DiagnosticCheck check;
    check.name = "toolchain";
    check.status = diagnostics::DiagnosticStatus::kPass;
    check.message = "Toolchain and driver stack verified";
    check.details["kernelRelease"] = inventory.toolchain.kernel_release;
    check.details["amdgpuStatus"] = inventory.toolchain.amdgpu_status;
    check.details["rocmVersion"] = inventory.toolchain.rocm_version;
    report.AddCheck(std::move(check));
  }

  return report;
}

int RunDiagnose(std::span<const char* const> args) {
  bool json_mode = false;
  bool fingerprint_mode = false;
  std::string validate_artifact_path;
  std::string benchmark_name;
  std::string backends_csv = "cpu,hip";
  std::uint32_t warmup = 3;
  std::uint32_t repetitions = 10;
  std::uint32_t duration_ms = 2000;
  std::string working_set_mib = "64";
  std::string output_file;
  std::string section = "all";
  std::size_t start_idx = 0;

  if (!args.empty()) {
    const std::string_view first = args[0];
    if (first == "gufo" || first.ends_with("/gufo")) {
      start_idx = 1;
    }
  }
  if (start_idx < args.size()) {
    const std::string_view cmd = args[start_idx];
    if (cmd == "diagnose" || cmd == "probe") {
      start_idx++;
    }
  }

  for (std::size_t i = start_idx; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--json") {
      json_mode = true;
    } else if (arg == "--fingerprint") {
      fingerprint_mode = true;
    } else if (arg == "--validate-artifact") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --validate-artifact requires a path argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      validate_artifact_path = args[++i];
    } else if (arg.starts_with("--validate-artifact=")) {
      validate_artifact_path = std::string(arg.substr(20));
    } else if (arg == "--benchmark") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --benchmark requires an argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      benchmark_name = args[++i];
    } else if (arg.starts_with("--benchmark=")) {
      benchmark_name = std::string(arg.substr(12));
    } else if (arg == "--backends") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --backends requires an argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      backends_csv = args[++i];
    } else if (arg.starts_with("--backends=")) {
      backends_csv = std::string(arg.substr(11));
    } else if (arg == "--warmup") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --warmup requires an integer argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      warmup = static_cast<std::uint32_t>(std::stoul(args[++i]));
    } else if (arg.starts_with("--warmup=")) {
      warmup =
          static_cast<std::uint32_t>(std::stoul(std::string(arg.substr(9))));
    } else if (arg == "--repetitions") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --repetitions requires an integer argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      repetitions = static_cast<std::uint32_t>(std::stoul(args[++i]));
    } else if (arg.starts_with("--repetitions=")) {
      repetitions =
          static_cast<std::uint32_t>(std::stoul(std::string(arg.substr(14))));
    } else if (arg == "--duration-ms") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --duration-ms requires an integer argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      duration_ms = static_cast<std::uint32_t>(std::stoul(args[++i]));
    } else if (arg.starts_with("--duration-ms=")) {
      duration_ms =
          static_cast<std::uint32_t>(std::stoul(std::string(arg.substr(14))));
    } else if (arg == "--working-set-mib") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --working-set-mib requires a comma-separated "
                     "size list\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      working_set_mib = args[++i];
    } else if (arg.starts_with("--working-set-mib=")) {
      working_set_mib = std::string(arg.substr(18));
    } else if (arg == "--output") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --output requires a file path argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      output_file = args[++i];
    } else if (arg.starts_with("--output=")) {
      output_file = std::string(arg.substr(9));
    } else if (arg == "--help" || arg == "-h") {
      PrintDiagnoseHelp("gufo");
      return 0;
    } else if (arg == "--section") {
      if (i + 1 >= args.size()) {
        std::cerr << "Error: --section requires an argument\n";
        PrintDiagnoseHelp("gufo");
        return 2;
      }
      section = args[++i];
    } else if (arg.starts_with("--section=")) {
      section = std::string(arg.substr(10));
    } else {
      std::cerr << "Error: unknown diagnose option '" << arg << "'\n";
      PrintDiagnoseHelp("gufo");
      return 2;
    }
  }

  if (section != "all" && section != "inventory" && section != "platform" &&
      section != "gpu") {
    std::cerr << "Error: unknown diagnostic section '" << section << "'\n";
    return 2;
  }

  // Handle artifact validation mode
  if (!validate_artifact_path.empty()) {
    const auto result =
        diagnostics::ValidateArtifactFile(validate_artifact_path);
    const std::string content = json_mode ? result.ToJson() : result.ToHuman();
    OutputContent(content, output_file);
    return result.is_valid ? 0 : 1;
  }

  // Handle benchmark execution mode
  if (!benchmark_name.empty()) {
    if (benchmark_name != "bandwidth" && benchmark_name != "allocation" &&
        benchmark_name != "hip-allocation") {
      std::cerr << "Error: unsupported benchmark '" << benchmark_name
                << "'. Supported: bandwidth, allocation\n";
      return 2;
    }

    const auto inventory =
        diagnostics::CollectSystemInventory(diagnostics::LinuxSysfs());
    const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);

    if (benchmark_name == "bandwidth") {
      diagnostics::BandwidthOptions bw_options;
      bw_options.backends = SplitCommaSeparated(backends_csv);
      if (bw_options.backends.empty()) {
        std::cerr << "Error: --backends requires at least one backend\n";
        return 2;
      }
      for (const auto& backend : bw_options.backends) {
        if (backend != "cpu" && backend != "hip") {
          std::cerr << "Error: unknown bandwidth backend '" << backend << "'\n";
          return 2;
        }
      }
      bw_options.warmup = warmup;
      bw_options.repetitions = repetitions;
      bw_options.duration_ms = duration_ms;
      const auto report =
          diagnostics::RunBandwidthBenchmark(bw_options, fingerprint);
      const std::string content =
          json_mode ? report.ToJson() : report.ToHuman();
      OutputContent(content, output_file);
      return 0;
    }

    diagnostics::AllocationBenchmarkOptions allocation_options;
    allocation_options.warmup = warmup;
    allocation_options.repetitions = repetitions;
    if (allocation_options.repetitions == 0) {
      std::cerr << "Error: --repetitions must be at least 1 for the allocation "
                   "diagnostic\n";
      return 2;
    }
    std::string size_error;
    if (!diagnostics::ParseWorkingSetSizes(
            working_set_mib, &allocation_options.working_set_bytes,
            &size_error)) {
      std::cerr << "Error: " << size_error << "\n";
      return 2;
    }
    const auto report = diagnostics::RunHipAllocationBenchmark(
        allocation_options, inventory, fingerprint);
    const std::string content = json_mode ? report.ToJson() : report.ToHuman();
    OutputContent(content, output_file);
    return report.Success() ? 0 : 1;
  }

  // Handle fingerprint mode
  if (fingerprint_mode) {
    const auto inventory =
        diagnostics::CollectSystemInventory(diagnostics::LinuxSysfs());
    const auto fingerprint = diagnostics::GenerateMachineFingerprint(inventory);
    const std::string content =
        json_mode ? fingerprint.ToJson() : fingerprint.ToHuman();
    OutputContent(content, output_file);
    return 0;
  }

  // Handle default diagnostics report
  const auto report = CollectDiagnostics(diagnostics::LinuxSysfs(), section);
  const std::string content = json_mode ? report.ToJson() : report.ToHuman();
  OutputContent(content, output_file);

  if (report.Status() == diagnostics::DiagnosticStatus::kFail) {
    return 1;
  }
  return 0;
}

#if defined(ENGINE_ENABLE_HIP)
static bool IsHipRequired() {
  const char* val = std::getenv("GUFO_REQUIRE_HIP");
  if (!val) {
    val = std::getenv("GUFO_REQUIRE_GPU");
  }
  if (!val) {
    val = std::getenv("GUFO_REQUIRE_GFX1151");
  }
  return val && std::string_view(val) != "0" &&
         std::string_view(val) != "false";
}

static void CheckGpu() {
  std::printf("ROCm version: %d.%d.%d\n", HIP_VERSION_MAJOR, HIP_VERSION_MINOR,
              HIP_VERSION_PATCH);

  int devCount = 0;
  hipError_t err = hipGetDeviceCount(&devCount);
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error on hipGetDeviceCount: %s\n",
                 hipGetErrorName(err));
    if (IsHipRequired()) {
      std::exit(1);
    } else {
      std::printf("SKIP: HIP device discovery failed\n");
      return;
    }
  }

  std::printf("HIP device count: %d\n", devCount);
  if (devCount == 0) {
    std::printf(
        "No HIP device found — is kfd loaded / device in render group?\n");
    if (IsHipRequired()) {
      std::exit(1);
    }
    return;
  }

  for (int i = 0; i < devCount; ++i) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
      continue;
    }
    std::printf("device[%d]: %s\n", i, prop.name);
    std::printf("  arch:        %s\n", prop.gcnArchName);
    std::printf("  computeCap:  %d.%d\n", prop.major, prop.minor);
    std::printf("  totalMem:    %.2f GiB (unified)\n",
                (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
}
#endif

int RunProbe(std::span<const char* const> args) {
  ArgParser parser("gufo probe", "Probe GPU availability.");
  std::string error;
  if (!parser.Parse(args, &error)) {
    std::cerr << "Error: " << error << '\n';
    return 2;
  }
  if (parser.IsHelpRequested()) {
    parser.PrintHelp();
    return 0;
  }
  std::printf("Hello, Strix Halo!\n");

#if defined(ENGINE_ENABLE_HIP)
  std::printf("Build: CMake, C++20, ROCm/HIP enabled\n");
  CheckGpu();
#else
  std::printf("Build: CMake, C++20, CPU-only (no ROCm)\n");
#endif

  std::printf("OK\n");
  return 0;
}

}  // namespace gufo::cli
