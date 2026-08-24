#include "src/models/qwen/xdna2/mtp_rmsnorm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/diagnostics/linux_sysfs.h"
#include "src/core/diagnostics/system_inventory.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/xdna2/device.h"
#include "src/models/qwen/oracles.hpp"

namespace {

constexpr float kEpsilon = 1.0e-6F;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

std::vector<float> MakeInput(std::size_t iteration) {
  std::vector<float> input(strix::xdna2::kQwenMtpRmsNormElements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const float phase =
        static_cast<float>((iteration * input.size()) + index) * 0.013F;
    input[index] = (0.75F * std::sin(phase)) + (0.2F * std::cos(phase * 0.37F));
  }
  return input;
}

std::vector<float> MakeSyntheticWeight() {
  std::vector<float> weight(strix::xdna2::kQwenMtpRmsNormElements);
  for (std::size_t index = 0; index < weight.size(); ++index) {
    weight[index] = 0.25F + (static_cast<float>(index % 257) / 512.0F);
  }
  return weight;
}

std::vector<float> LoadModelWeight(const std::filesystem::path& model_path) {
  std::string error;
  const auto reader = strix::core::GgufReader::OpenFile(model_path, &error);
  Expect(reader != nullptr, "MTP GGUF opens: " + error);
  const auto* tensor = reader->FindTensor("blk.64.nextn.enorm.weight");
  Expect(tensor != nullptr, "nextn.enorm weight exists");
  Expect(tensor->type == strix::core::GgmlType::kF32,
         "nextn.enorm weight is F32");
  Expect(tensor->ElementCount() == strix::xdna2::kQwenMtpRmsNormElements,
         "nextn.enorm weight has 5120 elements");
  const auto* data = static_cast<const float*>(tensor->data);
  return std::vector<float>(data, data + strix::xdna2::kQwenMtpRmsNormElements);
}

std::vector<std::uint16_t> ToBf16(std::span<const float> values) {
  std::vector<std::uint16_t> result(values.size());
  std::ranges::transform(values, result.begin(),
                         strix::models::qwen::FloatToBf16);
  return result;
}

std::vector<float> FromBf16(std::span<const std::uint16_t> values) {
  std::vector<float> result(values.size());
  std::ranges::transform(values, result.begin(),
                         strix::models::qwen::Bf16ToFloat);
  return result;
}

struct ErrorMetrics {
  double max_absolute{0.0};
  double rmse{0.0};
  double cosine{0.0};
};

ErrorMetrics Compare(std::span<const float> actual,
                     std::span<const float> expected) {
  double squared_error = 0.0;
  double dot = 0.0;
  double actual_norm = 0.0;
  double expected_norm = 0.0;
  double max_absolute = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double actual_value = actual[index];
    const double expected_value = expected[index];
    const double difference = actual_value - expected_value;
    squared_error += difference * difference;
    dot += actual_value * expected_value;
    actual_norm += actual_value * actual_value;
    expected_norm += expected_value * expected_value;
    max_absolute = std::max(max_absolute, std::abs(difference));
  }

  ErrorMetrics metrics;
  metrics.max_absolute = max_absolute;
  metrics.rmse = std::sqrt(squared_error / static_cast<double>(actual.size()));
  metrics.cosine =
      dot / std::sqrt(std::max(actual_norm * expected_norm, 1.0e-30));
  return metrics;
}

void RunCase(const strix::xdna2::XrtDeviceInfo& device_info,
             std::span<const float> weight, std::string_view label) {
  const auto weight_bf16 = ToBf16(weight);
  const auto rounded_weight = FromBf16(weight_bf16);

  strix::xdna2::QwenMtpRmsNormOptions options;
  options.timeout_ms = 30000;
#ifdef STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
  options.program_dir = STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR;
#endif

  strix::xdna2::QwenMtpRmsNormFailure failure;
  auto session = strix::xdna2::QwenMtpRmsNormSession::Create(
      options, device_info, weight_bf16, &failure);
  Expect(session != nullptr, "RMSNorm session opens: " + failure.message);
  Expect(session->ProgramInfo().bo_allocations == 3,
         "exactly three BOs are allocated");
  Expect(session->ProgramInfo().buffers_reused, "BOs are reused");
  Expect(session->ProgramInfo().requested_partition_columns == 1,
         "program requests one partition column");
  Expect(session->ProgramInfo().assigned_partition_columns == 1,
         "XRT context accepts one assigned partition column");
  Expect(session->ProgramInfo().context_mode == "shared",
         "session uses a shared XRT context");
  Expect(session->ProgramInfo().device_name == device_info.name,
         "device identity is recorded");
  Expect(session->ProgramInfo().device_architecture == device_info.architecture,
         "device architecture is recorded");
  Expect(session->ProgramInfo().driver == device_info.driver,
         "driver identity is recorded");
  Expect(session->ProgramInfo().firmware == device_info.firmware,
         "firmware identity is recorded");
  Expect(!session->ProgramInfo().xrt_version.empty(),
         "XRT version is recorded");
  Expect(!session->ProgramInfo().mlir_aie_version.empty(),
         "MLIR-AIE version is recorded");
  Expect(!session->ProgramInfo().llvm_aie_version.empty(),
         "LLVM-AIE version is recorded");
  Expect(!session->ProgramInfo().aiebu_revision.empty(),
         "AIEBU revision is recorded");
  Expect(!session->ProgramInfo().resource_evidence.empty(),
         "partition evidence is recorded");
  Expect(
      strix::xdna2::QwenMtpRmsNormSession::ActiveSessionCountForDiagnostics() ==
          1,
      "one session is live");
  Expect(
      strix::xdna2::QwenMtpRmsNormSession::ActiveBoCountForDiagnostics() == 3,
      "three BO wrappers are live");

  std::vector<double> command_us;
  std::vector<double> submission_us;
  std::vector<double> completion_us;
  ErrorMetrics worst;
  for (std::size_t iteration = 0; iteration < 8; ++iteration) {
    const auto input = MakeInput(iteration);
    const auto input_bf16 = ToBf16(input);
    const auto rounded_input = FromBf16(input_bf16);
    std::vector<std::uint16_t> output_bf16(input.size());
    std::vector<float> expected(input.size());
    strix::models::qwen::ReferenceRMSNorm(rounded_input, rounded_weight,
                                          kEpsilon, expected);

    strix::xdna2::QwenMtpRmsNormRunMetrics run_metrics;
    Expect(session->Run(input_bf16, output_bf16, &run_metrics, &failure),
           "RMSNorm command completes: " + failure.message);
    Expect(!run_metrics.quarantined, "session remains usable");
    command_us.push_back(run_metrics.command_us);
    submission_us.push_back(run_metrics.submission_us);
    completion_us.push_back(run_metrics.completion_us);

    const auto actual = FromBf16(output_bf16);
    const auto metrics = Compare(actual, expected);
    worst.max_absolute = std::max(worst.max_absolute, metrics.max_absolute);
    worst.rmse = std::max(worst.rmse, metrics.rmse);
    worst.cosine = worst.cosine == 0.0 ? metrics.cosine
                                       : std::min(worst.cosine, metrics.cosine);
  }

  Expect(worst.max_absolute <= 0.04, "maximum absolute error is bounded");
  Expect(worst.rmse <= 0.012, "RMSE is bounded");
  Expect(worst.cosine >= 0.9998, "cosine similarity is preserved");

  std::ranges::sort(command_us);
  std::ranges::sort(submission_us);
  std::ranges::sort(completion_us);
  const double median_us = (command_us[3] + command_us[4]) / 2.0;
  const double median_submission_us =
      (submission_us[3] + submission_us[4]) / 2.0;
  const double median_completion_us =
      (completion_us[3] + completion_us[4]) / 2.0;
  const auto info = session->ProgramInfo();
  const auto teardown_start = std::chrono::steady_clock::now();
  session.reset();
  Expect(
      strix::xdna2::QwenMtpRmsNormSession::ActiveSessionCountForDiagnostics() ==
          0,
      "session count returns to baseline");
  Expect(
      strix::xdna2::QwenMtpRmsNormSession::ActiveBoCountForDiagnostics() == 0,
      "BO count returns to baseline");
  const double teardown_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - teardown_start)
          .count();

  std::cout << label << ": setup=" << info.setup_ms
            << " ms device=" << info.device_open_ms
            << " ms xclbin=" << info.xclbin_load_ms
            << " ms context=" << info.context_create_ms
            << " ms program=" << info.program_load_ms
            << " ms buffers=" << info.buffer_setup_ms
            << " ms weight=" << info.weight_upload_ms
            << " ms median=" << median_us
            << " us submit=" << median_submission_us
            << " us complete=" << median_completion_us
            << " us teardown=" << teardown_ms << " ms xrt=" << info.xrt_version
            << " mlir-aie=" << info.mlir_aie_version
            << " max_abs=" << worst.max_absolute << " rmse=" << worst.rmse
            << " cosine=" << worst.cosine << "\n";
}

void TestFailureCategories(const strix::xdna2::XrtDeviceInfo& device_info,
                           std::span<const std::uint16_t> weight_bf16) {
  strix::xdna2::QwenMtpRmsNormOptions invalid_options;
  invalid_options.timeout_ms = 0;
  strix::xdna2::QwenMtpRmsNormFailure failure;
  auto session = strix::xdna2::QwenMtpRmsNormSession::Create(
      invalid_options, device_info, weight_bf16, &failure);
  Expect(session == nullptr && failure.category == "invalid_options",
         "zero timeout is categorized");

  strix::xdna2::QwenMtpRmsNormOptions missing_program;
  missing_program.program_dir =
      std::filesystem::temp_directory_path() /
      ("strix-qwen-mtp-rmsnorm-missing-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  session = strix::xdna2::QwenMtpRmsNormSession::Create(
      missing_program, device_info, weight_bf16, &failure);
  Expect(session == nullptr && failure.category == "program_missing",
         "missing program is categorized");

  auto unavailable_device = device_info;
  unavailable_device.available = false;
  unavailable_device.error_category = "firmware_unsupported";
  strix::xdna2::QwenMtpRmsNormOptions valid_options;
  session = strix::xdna2::QwenMtpRmsNormSession::Create(
      valid_options, unavailable_device, weight_bf16, &failure);
  Expect(session == nullptr && failure.category == "firmware_unsupported",
         "device compatibility failure is preserved");

#ifdef STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
  const auto substituted_dir =
      std::filesystem::temp_directory_path() /
      ("strix-qwen-mtp-rmsnorm-substituted-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(substituted_dir);
  const auto source_dir =
      std::filesystem::path(STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR);
  std::filesystem::copy_file(source_dir / "qwen_mtp_rmsnorm.xclbin",
                             substituted_dir / "qwen_mtp_rmsnorm.xclbin");
  std::filesystem::copy_file(source_dir / "qwen_mtp_rmsnorm.insts.elf",
                             substituted_dir / "qwen_mtp_rmsnorm.insts.elf");
  std::filesystem::permissions(substituted_dir / "qwen_mtp_rmsnorm.insts.elf",
                               std::filesystem::perms::owner_write,
                               std::filesystem::perm_options::add);
  {
    std::ofstream substituted(substituted_dir / "qwen_mtp_rmsnorm.insts.elf",
                              std::ios::binary | std::ios::app);
    Expect(substituted.is_open(), "substituted ELF opens for modification");
    substituted.put('\0');
  }
  Expect(std::filesystem::file_size(substituted_dir /
                                    "qwen_mtp_rmsnorm.insts.elf") ==
             std::filesystem::file_size(source_dir /
                                        "qwen_mtp_rmsnorm.insts.elf") +
                 1,
         "substituted ELF size differs from reviewed artifact");
  strix::xdna2::QwenMtpRmsNormOptions substituted_program;
  substituted_program.program_dir = substituted_dir;
  session = strix::xdna2::QwenMtpRmsNormSession::Create(
      substituted_program, device_info, weight_bf16, &failure);
  Expect(session == nullptr && failure.category == "program_incompatible",
         "substituted program is rejected by content hash: category=" +
             failure.category + " message=" + failure.message);
  std::filesystem::remove_all(substituted_dir);
#endif
}

void TestRepeatedLifecycle(const strix::xdna2::XrtDeviceInfo& device_info,
                           std::span<const std::uint16_t> weight_bf16) {
  strix::xdna2::QwenMtpRmsNormOptions options;
#ifdef STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
  options.program_dir = STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR;
#endif
  const auto input_bf16 = ToBf16(MakeInput(0));
  std::vector<std::uint16_t> output_bf16(input_bf16.size());
  for (std::size_t cycle = 0; cycle < 3; ++cycle) {
    Expect(strix::xdna2::QwenMtpRmsNormSession::
                   ActiveSessionCountForDiagnostics() == 0,
           "lifecycle starts at the session baseline");
    Expect(
        strix::xdna2::QwenMtpRmsNormSession::ActiveBoCountForDiagnostics() == 0,
        "lifecycle starts at the BO baseline");
    strix::xdna2::QwenMtpRmsNormFailure failure;
    auto session = strix::xdna2::QwenMtpRmsNormSession::Create(
        options, device_info, weight_bf16, &failure);
    Expect(session != nullptr,
           "repeated RMSNorm session opens: " + failure.message);
    Expect(session->Run(input_bf16, output_bf16, nullptr, &failure),
           "repeated RMSNorm command completes: " + failure.message);
    session.reset();
    Expect(strix::xdna2::QwenMtpRmsNormSession::
                   ActiveSessionCountForDiagnostics() == 0,
           "lifecycle returns to the session baseline");
    Expect(
        strix::xdna2::QwenMtpRmsNormSession::ActiveBoCountForDiagnostics() == 0,
        "lifecycle returns to the BO baseline");
  }
}

}  // namespace

int main() {
  const auto inventory = strix::diagnostics::CollectSystemInventory(
      strix::diagnostics::LinuxSysfs());
  const auto device_info = strix::xdna2::DiscoverXrtDevice(0, inventory);
  Expect(device_info.available, "compatible XDNA2 device is available");

  const auto synthetic_weight = MakeSyntheticWeight();
  const auto synthetic_weight_bf16 = ToBf16(synthetic_weight);
  TestFailureCategories(device_info, synthetic_weight_bf16);
  TestRepeatedLifecycle(device_info, synthetic_weight_bf16);
  RunCase(device_info, synthetic_weight, "synthetic");

  if (const char* model = std::getenv("STRIX_MTP_MODEL");
      model != nullptr && std::string_view(model).size() > 0) {
    const auto model_weight = LoadModelWeight(model);
    RunCase(device_info, model_weight, "qwen3.8-mtp");
  } else {
    std::cout << "qwen3.8-mtp XDNA2 RMSNorm: skipped "
                 "(STRIX_MTP_MODEL not set)\n";
  }
  return 0;
}
