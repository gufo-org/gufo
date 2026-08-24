#include "src/models/qwen/xdna2/mtp_rmsnorm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/diagnostics/fingerprint.h"

#ifdef ENGINE_ENABLE_XRT
#include <strix/aie_qwen_mtp_rmsnorm_manifest.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#endif

namespace strix::xdna2 {

namespace {

constexpr std::size_t kBufferBytes =
    kQwenMtpRmsNormElements * sizeof(std::uint16_t);
std::atomic<std::size_t> g_active_sessions{0};
std::atomic<std::size_t> g_active_bos{0};

void SetFailure(QwenMtpRmsNormFailure* failure, std::string category,
                std::string message) {
  if (failure != nullptr) {
    failure->category = std::move(category);
    failure->message = std::move(message);
  }
}

void ClearFailure(QwenMtpRmsNormFailure* failure) {
  if (failure != nullptr) {
    failure->category.clear();
    failure->message.clear();
  }
}

#ifdef ENGINE_ENABLE_XRT

std::filesystem::path DefaultProgramDir() {
#ifdef STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR
  return STRIX_AIE_QWEN_MTP_RMSNORM_PROGRAM_DIR;
#else
  return {};
#endif
}

bool IsNonemptyFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) &&
         std::filesystem::file_size(path, error) > 0 && !error;
}

std::string ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::string SelectKernelName(const xrt::xclbin& xclbin) {
  const auto kernels = xclbin.get_kernels();
  const auto match =
      std::ranges::find_if(kernels, [](const xrt::xclbin::kernel& kernel) {
        return kernel.get_name().starts_with("MLIR_AIE");
      });
  if (match != kernels.end()) {
    return match->get_name();
  }
  if (kernels.size() == 1) {
    return kernels.front().get_name();
  }
  return {};
}

bool ValidateProgram(const std::filesystem::path& program_dir,
                     QwenMtpRmsNormFailure* failure) {
  const auto xclbin_path = program_dir / "qwen_mtp_rmsnorm.xclbin";
  const auto elf_path = program_dir / "qwen_mtp_rmsnorm.insts.elf";
  if (!IsNonemptyFile(xclbin_path) || !IsNonemptyFile(elf_path)) {
    SetFailure(failure, "program_missing",
               "required qwen_mtp_rmsnorm XCLBIN or instruction ELF is absent");
    return false;
  }

  const bool compatible =
      generated::kQwenMtpRmsNormTarget == "npu2" &&
      generated::kQwenMtpRmsNormAbi == "xrt-elf-v1" &&
      generated::kQwenMtpRmsNormModelKind == "qwen3.8-27b-mtp" &&
      generated::kQwenMtpRmsNormTensorContract ==
          "bf16-rmsnorm-fp32acc-n5120-eps1e-6" &&
      generated::kQwenMtpRmsNormPartitionColumns > 0;
  if (!compatible) {
    SetFailure(
        failure, "program_incompatible",
        "Qwen MTP RMSNorm artifact manifest is incompatible with runtime");
    return false;
  }

  const std::string xclbin = ReadBinaryFile(xclbin_path);
  const std::string elf = ReadBinaryFile(elf_path);
  const std::string xclbin_sha = diagnostics::ComputeSha256Hex(xclbin);
  const std::string elf_sha = diagnostics::ComputeSha256Hex(elf);
  const std::string program_sha =
      diagnostics::ComputeSha256Hex(std::string(xclbin).append(elf));
  const bool hashes_match =
      xclbin_sha == generated::kQwenMtpRmsNormXclbinSha256 &&
      elf_sha == generated::kQwenMtpRmsNormElfSha256 &&
      program_sha == generated::kQwenMtpRmsNormProgramSha256;
  if (!hashes_match) {
    SetFailure(failure, "program_incompatible",
               "Qwen MTP RMSNorm artifact content hash does not match the "
               "reviewed manifest");
    return false;
  }
  return true;
}

#endif

}  // namespace

struct QwenMtpRmsNormSession::Impl {
  Impl() { g_active_sessions.fetch_add(1, std::memory_order_relaxed); }
  ~Impl() {
    if (buffers_allocated) {
      g_active_bos.fetch_sub(3, std::memory_order_relaxed);
    }
    g_active_sessions.fetch_sub(1, std::memory_order_relaxed);
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  bool buffers_allocated{false};
#ifdef ENGINE_ENABLE_XRT
  std::unique_ptr<xrt::device> device;
  std::unique_ptr<xrt::xclbin> xclbin;
  std::unique_ptr<xrt::hw_context> context;
  std::unique_ptr<xrt::elf> elf;
  std::unique_ptr<xrt::module> module;
  std::unique_ptr<xrt::ext::kernel> kernel;
  std::unique_ptr<xrt::bo> input_bo;
  std::unique_ptr<xrt::bo> weight_bo;
  std::unique_ptr<xrt::bo> output_bo;
  std::uint16_t* input{nullptr};
  std::uint16_t* output{nullptr};
#endif
};

QwenMtpRmsNormSession::QwenMtpRmsNormSession(
    std::unique_ptr<Impl> impl, QwenMtpRmsNormProgramInfo program_info,
    std::uint32_t timeout_ms)
    : impl_(std::move(impl)),
      program_info_(std::move(program_info)),
      timeout_ms_(timeout_ms) {}

QwenMtpRmsNormSession::~QwenMtpRmsNormSession() = default;
QwenMtpRmsNormSession::QwenMtpRmsNormSession(QwenMtpRmsNormSession&&) noexcept =
    default;
QwenMtpRmsNormSession& QwenMtpRmsNormSession::operator=(
    QwenMtpRmsNormSession&&) noexcept = default;

std::size_t QwenMtpRmsNormSession::ActiveSessionCountForDiagnostics() noexcept {
  return g_active_sessions.load(std::memory_order_relaxed);
}

std::size_t QwenMtpRmsNormSession::ActiveBoCountForDiagnostics() noexcept {
  return g_active_bos.load(std::memory_order_relaxed);
}

std::unique_ptr<QwenMtpRmsNormSession> QwenMtpRmsNormSession::Create(
    const QwenMtpRmsNormOptions& options, const XrtDeviceInfo& device_info,
    std::span<const std::uint16_t> weight_bf16,
    QwenMtpRmsNormFailure* failure) {
  ClearFailure(failure);
  if (options.timeout_ms == 0) {
    SetFailure(failure, "invalid_options",
               "Qwen MTP RMSNorm timeout must be positive");
    return nullptr;
  }
  if (!device_info.available ||
      device_info.device_index != options.device_index) {
    SetFailure(failure,
               device_info.error_category.empty() ? "device_unavailable"
                                                  : device_info.error_category,
               "Qwen MTP RMSNorm requires a compatible discovered device");
    return nullptr;
  }
  if (weight_bf16.size() != kQwenMtpRmsNormElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen MTP RMSNorm weight must contain 5120 BF16 elements");
    return nullptr;
  }

#ifdef ENGINE_ENABLE_XRT
  const auto program_dir =
      options.program_dir.empty() ? DefaultProgramDir() : options.program_dir;
  if (!ValidateProgram(program_dir, failure)) {
    return nullptr;
  }

  const auto setup_start = std::chrono::steady_clock::now();
  try {
    QwenMtpRmsNormProgramInfo info;
    const auto device_start = std::chrono::steady_clock::now();
    auto impl = std::make_unique<Impl>();
    impl->device = std::make_unique<xrt::device>(options.device_index);
    info.device_open_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - device_start)
                              .count();

    const auto xclbin_start = std::chrono::steady_clock::now();
    impl->xclbin = std::make_unique<xrt::xclbin>(
        (program_dir / "qwen_mtp_rmsnorm.xclbin").string());
    info.xclbin_load_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - xclbin_start)
                              .count();
    const std::string kernel_name = SelectKernelName(*impl->xclbin);
    if (kernel_name.empty()) {
      SetFailure(failure, "program_incompatible",
                 "Qwen MTP RMSNorm XCLBIN has no unique MLIR_AIE kernel");
      return nullptr;
    }

    const auto context_start = std::chrono::steady_clock::now();
    impl->device->register_xclbin(*impl->xclbin);
    impl->context = std::make_unique<xrt::hw_context>(*impl->device,
                                                      impl->xclbin->get_uuid());
    info.context_create_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - context_start)
            .count();

    const auto program_start = std::chrono::steady_clock::now();
    impl->elf = std::make_unique<xrt::elf>(
        (program_dir / "qwen_mtp_rmsnorm.insts.elf").string());
    impl->module = std::make_unique<xrt::module>(*impl->elf);
    impl->kernel = std::make_unique<xrt::ext::kernel>(
        *impl->context, *impl->module, kernel_name);
    info.program_load_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - program_start)
                               .count();

    const auto buffer_start = std::chrono::steady_clock::now();
    impl->input_bo =
        std::make_unique<xrt::bo>(xrt::ext::bo(*impl->device, kBufferBytes));
    impl->weight_bo =
        std::make_unique<xrt::bo>(xrt::ext::bo(*impl->device, kBufferBytes));
    impl->output_bo =
        std::make_unique<xrt::bo>(xrt::ext::bo(*impl->device, kBufferBytes));
    impl->buffers_allocated = true;
    g_active_bos.fetch_add(3, std::memory_order_relaxed);

    impl->input = impl->input_bo->map<std::uint16_t*>();
    auto* weight = impl->weight_bo->map<std::uint16_t*>();
    impl->output = impl->output_bo->map<std::uint16_t*>();
    if (impl->input == nullptr || weight == nullptr ||
        impl->output == nullptr) {
      SetFailure(failure, "buffer_failure",
                 "XRT returned a null Qwen MTP RMSNorm BO mapping");
      return nullptr;
    }
    info.buffer_setup_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - buffer_start)
                               .count();

    const auto weight_start = std::chrono::steady_clock::now();
    std::ranges::copy(weight_bf16, weight);
    impl->weight_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    info.weight_upload_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - weight_start)
                                .count();

    info.target = generated::kQwenMtpRmsNormTarget;
    info.abi = generated::kQwenMtpRmsNormAbi;
    info.model_kind = generated::kQwenMtpRmsNormModelKind;
    info.tensor_contract = generated::kQwenMtpRmsNormTensorContract;
    info.program_sha256 = generated::kQwenMtpRmsNormProgramSha256;
    info.xclbin_sha256 = generated::kQwenMtpRmsNormXclbinSha256;
    info.elf_sha256 = generated::kQwenMtpRmsNormElfSha256;
    info.device_name = device_info.name;
    info.device_architecture = device_info.architecture;
    info.driver = device_info.driver;
    info.firmware = device_info.firmware;
    info.xrt_version = generated::kQwenMtpRmsNormXrtVersion;
    info.mlir_aie_version = generated::kQwenMtpRmsNormMlirAieVersion;
    info.llvm_aie_version = generated::kQwenMtpRmsNormLlvmAieVersion;
    info.aiebu_revision = generated::kQwenMtpRmsNormAiebuRevision;
    info.xclbin_uuid = impl->xclbin->get_uuid().to_string();
    info.kernel_name = kernel_name;
    info.context_mode =
        impl->context->get_mode() == xrt::hw_context::access_mode::shared
            ? "shared"
            : "exclusive";
    info.resource_evidence =
        "XCLBIN AIE_PARTITION.column_width accepted by XRT context";
    info.requested_partition_columns =
        generated::kQwenMtpRmsNormPartitionColumns;
    info.assigned_partition_columns =
        generated::kQwenMtpRmsNormPartitionColumns;
    info.bo_allocations = 3;
    info.buffers_reused = true;
    info.setup_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - setup_start)
                        .count();

    return std::unique_ptr<QwenMtpRmsNormSession>(new QwenMtpRmsNormSession(
        std::move(impl), std::move(info), options.timeout_ms));
  } catch (const std::exception& exception) {
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen MTP RMSNorm XRT setup failed: ") + exception.what());
    return nullptr;
  }
#else
  (void)options;
  (void)device_info;
  (void)weight_bf16;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return nullptr;
#endif
}

bool QwenMtpRmsNormSession::Run(std::span<const std::uint16_t> input_bf16,
                                std::span<std::uint16_t> output_bf16,
                                QwenMtpRmsNormRunMetrics* metrics,
                                QwenMtpRmsNormFailure* failure) {
  ClearFailure(failure);
  if (input_bf16.size() != kQwenMtpRmsNormElements ||
      output_bf16.size() != kQwenMtpRmsNormElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen MTP RMSNorm input and output must contain 5120 elements");
    return false;
  }
  if (quarantined_) {
    SetFailure(failure, "session_quarantined",
               "Qwen MTP RMSNorm session is quarantined");
    if (metrics != nullptr) {
      metrics->quarantined = true;
    }
    return false;
  }

#ifdef ENGINE_ENABLE_XRT
  try {
    std::ranges::copy(input_bf16, impl_->input);
    std::fill_n(impl_->output, kQwenMtpRmsNormElements,
                std::numeric_limits<std::uint16_t>::max());
    impl_->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->output_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const auto command_start = std::chrono::steady_clock::now();
    const auto submission_start = command_start;
    auto run = (*impl_->kernel)(3U, 0U, 0U, *impl_->input_bo, *impl_->weight_bo,
                                *impl_->output_bo);
    const auto submission_end = std::chrono::steady_clock::now();
    const auto wait_status = run.wait2(std::chrono::milliseconds(timeout_ms_));
    const auto completion_end = std::chrono::steady_clock::now();
    if (wait_status == std::cv_status::timeout) {
      run.abort();
      quarantined_ = true;
      SetFailure(failure, "timeout", "Qwen MTP RMSNorm command timed out");
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }

    if (metrics != nullptr) {
      metrics->submission_us = std::chrono::duration<double, std::micro>(
                                   submission_end - submission_start)
                                   .count();
      metrics->completion_us = std::chrono::duration<double, std::micro>(
                                   completion_end - submission_end)
                                   .count();
      metrics->command_us = std::chrono::duration<double, std::micro>(
                                completion_end - command_start)
                                .count();
    }
    impl_->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::ranges::copy_n(impl_->output, kQwenMtpRmsNormElements,
                        output_bf16.begin());
    return true;
  } catch (const xrt::run::command_error& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "command_error",
        std::string("Qwen MTP RMSNorm command failed: ") + exception.what());
  } catch (const std::exception& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen MTP RMSNorm XRT failure: ") + exception.what());
  }
  if (metrics != nullptr) {
    metrics->quarantined = true;
  }
  return false;
#else
  (void)input_bf16;
  (void)output_bf16;
  (void)metrics;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return false;
#endif
}

}  // namespace strix::xdna2
