#include "src/models/qwen/xdna2/mtp_eh_proj.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "src/core/quant/ggml_dequant.hpp"

#ifdef ENGINE_ENABLE_XRT
#include <strix/aie_qwen_mtp_eh_proj_manifest.h>
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

constexpr std::size_t kBlockElements = 256;
constexpr std::size_t kGroupElements = 32;
constexpr std::size_t kGroupsPerBlock = kBlockElements / kGroupElements;
constexpr std::size_t kMmulK = 16;
constexpr std::size_t kMmulRows = 4;
constexpr std::size_t kOutputTileElements = 16;
constexpr std::size_t kMmulActivationElements = kMmulRows * kMmulK;
constexpr std::size_t kMmulWeightElements = kMmulK * kOutputTileElements;
constexpr std::size_t kMmulWeightBytes = kMmulWeightElements / 2;
constexpr std::size_t kWeightRecordBytes = 4096;
constexpr std::size_t kInputRecordBytes = 2048;
constexpr std::size_t kWeightCodeBytes = kGroupsPerBlock * 2 * kMmulWeightBytes;
constexpr std::size_t kWeightScaleOffset = kWeightCodeBytes;
constexpr std::size_t kWeightMinOffset =
    kWeightScaleOffset +
    (kGroupsPerBlock * kOutputTileElements * sizeof(float));
constexpr std::size_t kInputCodeBytes =
    kGroupsPerBlock * 2 * kMmulActivationElements;
constexpr std::size_t kInputScaleOffset = kInputCodeBytes;
constexpr std::size_t kInputSumOffset =
    kInputScaleOffset + (kGroupsPerBlock * sizeof(float));
constexpr std::size_t kInputBlocks =
    kQwenMtpEhProjInputElements / kBlockElements;
constexpr std::size_t kOutputTiles =
    kQwenMtpEhProjOutputElements / kOutputTileElements;
constexpr std::size_t kWeightRowChunkBytes = 256;
constexpr std::size_t kWeightRowBytes = kInputBlocks * kWeightRowChunkBytes;
constexpr std::size_t kPackedWeightBytes =
    kQwenMtpEhProjOutputElements * kWeightRowBytes;
constexpr std::size_t kPackedInputBytes = kInputBlocks * kInputRecordBytes;
constexpr std::size_t kOutputBytes =
    kQwenMtpEhProjOutputElements * sizeof(float);

#pragma pack(push, 1)
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(kWeightCodeBytes == 2048);
static_assert(kInputCodeBytes == 1024);
static_assert(kWeightMinOffset +
                  (kGroupsPerBlock * kOutputTileElements * sizeof(float)) <=
              kWeightRecordBytes);
static_assert(kInputSumOffset + (kGroupsPerBlock * sizeof(std::int32_t)) <=
              kInputRecordBytes);

std::atomic<std::size_t> g_active_sessions{0};
std::atomic<std::size_t> g_active_bos{0};

void SetFailure(QwenMtpEhProjFailure* failure, std::string category,
                std::string message) {
  if (failure != nullptr) {
    failure->category = std::move(category);
    failure->message = std::move(message);
  }
}

void ClearFailure(QwenMtpEhProjFailure* failure) {
  if (failure != nullptr) {
    failure->category.clear();
    failure->message.clear();
  }
}

std::uint8_t* WeightRecordByte(std::uint8_t* weights, std::size_t output_tile,
                               std::size_t input_block,
                               std::size_t byte_offset) {
  const std::size_t local_row = byte_offset / kWeightRowChunkBytes;
  const std::size_t local_column = byte_offset % kWeightRowChunkBytes;
  const std::size_t output_row =
      (output_tile * kOutputTileElements) + local_row;
  return weights + (output_row * kWeightRowBytes) +
         (input_block * kWeightRowChunkBytes) + local_column;
}

template<typename T>
void StoreWeightValue(std::uint8_t* weights, std::size_t output_tile,
                      std::size_t input_block, std::size_t byte_offset,
                      const T& value) {
  std::memcpy(WeightRecordByte(weights, output_tile, input_block, byte_offset),
              &value, sizeof(T));
}

void SetWeightNibble(std::uint8_t* weights, std::size_t output_tile,
                     std::size_t input_block, std::size_t group,
                     std::size_t half, std::size_t k_index,
                     std::size_t output_index, std::uint8_t value) {
  const std::size_t tile_offset = ((group * 2) + half) * kMmulWeightBytes;
  const std::size_t element_index =
      (k_index * kOutputTileElements) + output_index;
  const std::size_t byte_offset = tile_offset + (element_index / 2);
  auto* packed =
      WeightRecordByte(weights, output_tile, input_block, byte_offset);
  if ((element_index & 1U) == 0U) {
    *packed = static_cast<std::uint8_t>((*packed & 0xF0U) | value);
  } else {
    *packed = static_cast<std::uint8_t>((*packed & 0x0FU) | (value << 4U));
  }
}

void GetQ4ScaleMin(std::size_t group, const std::uint8_t* packed,
                   std::uint8_t& scale, std::uint8_t& minimum) noexcept {
  if (group < 4) {
    scale = packed[group] & 0x3FU;
    minimum = packed[group + 4] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[group + 4] & 0x0FU) |
                                    ((packed[group - 4] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[group + 4] >> 4U) |
                                      ((packed[group] >> 6U) << 4U));
}

bool PackQ4KWeights(const models::QwenTensorRef& tensor, std::uint8_t* packed,
                    QwenMtpEhProjFailure* failure) {
  if (tensor.type != core::GgmlType::kQ4_K ||
      tensor.num_elements !=
          kQwenMtpEhProjOutputElements * kQwenMtpEhProjInputElements ||
      tensor.data == nullptr) {
    SetFailure(failure, "invalid_tensor",
               "Qwen MTP eh_proj must be a 5120x10240 Q4_K tensor");
    return false;
  }

  std::fill_n(packed, kPackedWeightBytes, std::uint8_t{0});
  const auto* rows = static_cast<const std::uint8_t*>(tensor.data);
  const std::size_t source_row_bytes =
      quant::QuantizedRowBytes(tensor.type, kQwenMtpEhProjInputElements);
  for (std::size_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    for (std::size_t input_block = 0; input_block < kInputBlocks;
         ++input_block) {
      for (std::size_t output_index = 0; output_index < kOutputTileElements;
           ++output_index) {
        const std::size_t output_row =
            (output_tile * kOutputTileElements) + output_index;
        const auto* block = reinterpret_cast<const BlockQ4K*>(
            rows + (output_row * source_row_bytes) +
            (input_block * sizeof(BlockQ4K)));
        const float d = quant::Fp16ToFloat(block->d);
        const float dmin = quant::Fp16ToFloat(block->dmin);
        for (std::size_t group = 0; group < kGroupsPerBlock; ++group) {
          const std::size_t pair = group / 2;
          const bool high_nibble = (group & 1U) != 0U;
          for (std::size_t half = 0; half < 2; ++half) {
            for (std::size_t k_index = 0; k_index < kMmulK; ++k_index) {
              const std::size_t lane = (half * kMmulK) + k_index;
              const std::uint8_t source = block->qs[(pair * 32) + lane];
              const std::uint8_t quantized =
                  high_nibble ? source >> 4U : source & 0x0FU;
              SetWeightNibble(packed, output_tile, input_block, group, half,
                              k_index, output_index, quantized);
            }
          }
          std::uint8_t scale = 0;
          std::uint8_t minimum = 0;
          GetQ4ScaleMin(group, block->scales, scale, minimum);
          const std::size_t parameter_index =
              (group * kOutputTileElements) + output_index;
          StoreWeightValue(
              packed, output_tile, input_block,
              kWeightScaleOffset + (parameter_index * sizeof(float)),
              d * static_cast<float>(scale));
          StoreWeightValue(packed, output_tile, input_block,
                           kWeightMinOffset + (parameter_index * sizeof(float)),
                           dmin * static_cast<float>(minimum));
        }
      }
    }
  }
  return true;
}

double PackActivations(std::span<const float> input, std::uint8_t* packed) {
  const auto start = std::chrono::steady_clock::now();
  std::fill_n(packed, kPackedInputBytes, std::uint8_t{0});
  for (std::size_t input_block = 0; input_block < kInputBlocks; ++input_block) {
    auto* record = packed + (input_block * kInputRecordBytes);
    for (std::size_t group = 0; group < kGroupsPerBlock; ++group) {
      const std::size_t group_start =
          (input_block * kBlockElements) + (group * kGroupElements);
      float maximum = 0.0F;
      for (std::size_t lane = 0; lane < kGroupElements; ++lane) {
        maximum = std::max(maximum, std::abs(input[group_start + lane]));
      }
      const float scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
      std::int32_t quantized_sum = 0;
      for (std::size_t half = 0; half < 2; ++half) {
        auto* tile = reinterpret_cast<std::int8_t*>(
            record + (((group * 2) + half) * kMmulActivationElements));
        for (std::size_t k_index = 0; k_index < kMmulK; ++k_index) {
          const std::size_t lane = (half * kMmulK) + k_index;
          const auto rounded = static_cast<int>(
              std::nearbyint(input[group_start + lane] / scale));
          const auto quantized =
              static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
          quantized_sum += quantized;
          for (std::size_t row = 0; row < kMmulRows; ++row) {
            tile[(row * kMmulK) + k_index] = quantized;
          }
        }
      }
      std::memcpy(record + kInputScaleOffset + (group * sizeof(float)), &scale,
                  sizeof(scale));
      std::memcpy(record + kInputSumOffset + (group * sizeof(std::int32_t)),
                  &quantized_sum, sizeof(quantized_sum));
    }
  }
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

#ifdef ENGINE_ENABLE_XRT

std::filesystem::path DefaultProgramDir() {
#ifdef STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR
  return STRIX_AIE_QWEN_MTP_EH_PROJ_PROGRAM_DIR;
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
  return kernels.size() == 1 ? kernels.front().get_name() : std::string{};
}

bool ValidateProgram(const std::filesystem::path& program_dir,
                     QwenMtpEhProjFailure* failure) {
  const auto xclbin_path = program_dir / "qwen_mtp_eh_proj.xclbin";
  const auto elf_path = program_dir / "qwen_mtp_eh_proj.insts.elf";
  if (!IsNonemptyFile(xclbin_path) || !IsNonemptyFile(elf_path)) {
    SetFailure(failure, "program_missing",
               "required Qwen MTP eh_proj XCLBIN or instruction ELF is absent");
    return false;
  }
  const bool compatible =
      generated::kQwenMtpEhProjTarget == "npu2" &&
      generated::kQwenMtpEhProjAbi == "xrt-elf-v1" &&
      generated::kQwenMtpEhProjModelKind == "qwen3.8-27b-mtp" &&
      generated::kQwenMtpEhProjTensorContract ==
          "q4_k-u4-dyn-i8-g32-int32-fp32-m5120-k10240" &&
      generated::kQwenMtpEhProjPartitionColumns == 8;
  if (!compatible) {
    SetFailure(failure, "program_incompatible",
               "Qwen MTP eh_proj artifact manifest is incompatible");
    return false;
  }
  const std::string xclbin = ReadBinaryFile(xclbin_path);
  const std::string elf = ReadBinaryFile(elf_path);
  const bool hashes_match =
      diagnostics::ComputeSha256Hex(xclbin) ==
          generated::kQwenMtpEhProjXclbinSha256 &&
      diagnostics::ComputeSha256Hex(elf) ==
          generated::kQwenMtpEhProjElfSha256 &&
      diagnostics::ComputeSha256Hex(std::string(xclbin).append(elf)) ==
          generated::kQwenMtpEhProjProgramSha256;
  if (!hashes_match) {
    SetFailure(failure, "program_incompatible",
               "Qwen MTP eh_proj artifact hash does not match its manifest");
    return false;
  }
  return true;
}

#endif

}  // namespace

struct QwenMtpEhProjSession::Impl {
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
  std::unique_ptr<xrt::bo> weight_bo;
  std::unique_ptr<xrt::bo> input_bo;
  std::unique_ptr<xrt::bo> output_bo;
  std::uint8_t* weight{nullptr};
  std::uint8_t* input{nullptr};
  float* output{nullptr};
#endif
};

QwenMtpEhProjSession::QwenMtpEhProjSession(
    std::unique_ptr<Impl> impl, QwenMtpEhProjProgramInfo program_info,
    std::uint32_t timeout_ms)
    : impl_(std::move(impl)),
      program_info_(std::move(program_info)),
      timeout_ms_(timeout_ms) {}

QwenMtpEhProjSession::~QwenMtpEhProjSession() = default;
QwenMtpEhProjSession::QwenMtpEhProjSession(QwenMtpEhProjSession&&) noexcept =
    default;
QwenMtpEhProjSession& QwenMtpEhProjSession::operator=(
    QwenMtpEhProjSession&&) noexcept = default;

std::size_t QwenMtpEhProjSession::ActiveSessionCountForDiagnostics() noexcept {
  return g_active_sessions.load(std::memory_order_relaxed);
}

std::size_t QwenMtpEhProjSession::ActiveBoCountForDiagnostics() noexcept {
  return g_active_bos.load(std::memory_order_relaxed);
}

std::unique_ptr<QwenMtpEhProjSession> QwenMtpEhProjSession::Create(
    const QwenMtpEhProjOptions& options, const XrtDeviceInfo& device_info,
    const models::QwenTensorRef& q4k_weights, QwenMtpEhProjFailure* failure) {
  ClearFailure(failure);
  if (options.timeout_ms == 0) {
    SetFailure(failure, "invalid_options",
               "Qwen MTP eh_proj timeout must be positive");
    return nullptr;
  }
  if (!device_info.available ||
      device_info.device_index != options.device_index) {
    SetFailure(failure,
               device_info.error_category.empty() ? "device_unavailable"
                                                  : device_info.error_category,
               "Qwen MTP eh_proj requires a compatible discovered device");
    return nullptr;
  }
  if (q4k_weights.type != core::GgmlType::kQ4_K ||
      q4k_weights.num_elements !=
          kQwenMtpEhProjOutputElements * kQwenMtpEhProjInputElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen MTP eh_proj requires the original Q4_K matrix");
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
    auto impl = std::make_unique<Impl>();
    QwenMtpEhProjProgramInfo info;
    impl->device = std::make_unique<xrt::device>(options.device_index);
    impl->xclbin = std::make_unique<xrt::xclbin>(
        (program_dir / "qwen_mtp_eh_proj.xclbin").string());
    const std::string kernel_name = SelectKernelName(*impl->xclbin);
    if (kernel_name.empty()) {
      SetFailure(failure, "program_incompatible",
                 "Qwen MTP eh_proj XCLBIN has no unique MLIR_AIE kernel");
      return nullptr;
    }
    impl->device->register_xclbin(*impl->xclbin);
    impl->context = std::make_unique<xrt::hw_context>(*impl->device,
                                                      impl->xclbin->get_uuid());
    impl->elf = std::make_unique<xrt::elf>(
        (program_dir / "qwen_mtp_eh_proj.insts.elf").string());
    impl->module = std::make_unique<xrt::module>(*impl->elf);
    impl->kernel = std::make_unique<xrt::ext::kernel>(
        *impl->context, *impl->module, kernel_name);
    impl->weight_bo = std::make_unique<xrt::bo>(
        xrt::ext::bo(*impl->device, kPackedWeightBytes));
    impl->input_bo = std::make_unique<xrt::bo>(
        xrt::ext::bo(*impl->device, kPackedInputBytes));
    impl->output_bo =
        std::make_unique<xrt::bo>(xrt::ext::bo(*impl->device, kOutputBytes));
    impl->buffers_allocated = true;
    g_active_bos.fetch_add(3, std::memory_order_relaxed);
    impl->weight = impl->weight_bo->map<std::uint8_t*>();
    impl->input = impl->input_bo->map<std::uint8_t*>();
    impl->output = impl->output_bo->map<float*>();
    if (impl->weight == nullptr || impl->input == nullptr ||
        impl->output == nullptr) {
      SetFailure(failure, "buffer_failure",
                 "XRT returned a null Qwen MTP eh_proj BO mapping");
      return nullptr;
    }

    const auto pack_start = std::chrono::steady_clock::now();
    if (!PackQ4KWeights(q4k_weights, impl->weight, failure)) {
      return nullptr;
    }
    info.weight_pack_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - pack_start)
                              .count();
    const auto upload_start = std::chrono::steady_clock::now();
    impl->weight_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    info.weight_upload_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - upload_start)
                                .count();

    info.target = generated::kQwenMtpEhProjTarget;
    info.abi = generated::kQwenMtpEhProjAbi;
    info.model_kind = generated::kQwenMtpEhProjModelKind;
    info.tensor_contract = generated::kQwenMtpEhProjTensorContract;
    info.program_sha256 = generated::kQwenMtpEhProjProgramSha256;
    info.xclbin_sha256 = generated::kQwenMtpEhProjXclbinSha256;
    info.elf_sha256 = generated::kQwenMtpEhProjElfSha256;
    info.device_name = device_info.name;
    info.device_architecture = device_info.architecture;
    info.driver = device_info.driver;
    info.firmware = device_info.firmware;
    info.xrt_version = generated::kQwenMtpEhProjXrtVersion;
    info.mlir_aie_version = generated::kQwenMtpEhProjMlirAieVersion;
    info.llvm_aie_version = generated::kQwenMtpEhProjLlvmAieVersion;
    info.aiebu_revision = generated::kQwenMtpEhProjAiebuRevision;
    info.xclbin_uuid = impl->xclbin->get_uuid().to_string();
    info.kernel_name = kernel_name;
    info.context_mode =
        impl->context->get_mode() == xrt::hw_context::access_mode::shared
            ? "shared"
            : "exclusive";
    info.partition_columns = generated::kQwenMtpEhProjPartitionColumns;
    info.bo_allocations = 3;
    info.packed_weight_bytes = kPackedWeightBytes;
    info.packed_input_bytes = kPackedInputBytes;
    info.setup_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - setup_start)
                        .count();
    return std::unique_ptr<QwenMtpEhProjSession>(new QwenMtpEhProjSession(
        std::move(impl), std::move(info), options.timeout_ms));
  } catch (const std::exception& exception) {
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen MTP eh_proj XRT setup failed: ") + exception.what());
    return nullptr;
  }
#else
  (void)options;
  (void)device_info;
  (void)q4k_weights;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return nullptr;
#endif
}

bool QwenMtpEhProjSession::Run(std::span<const float> input,
                               std::span<float> output,
                               QwenMtpEhProjRunMetrics* metrics,
                               QwenMtpEhProjFailure* failure) {
  ClearFailure(failure);
  if (input.size() != kQwenMtpEhProjInputElements ||
      output.size() != kQwenMtpEhProjOutputElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen MTP eh_proj input/output dimensions are invalid");
    return false;
  }
  if (quarantined_) {
    SetFailure(failure, "session_quarantined",
               "Qwen MTP eh_proj session is quarantined");
    if (metrics != nullptr) {
      metrics->quarantined = true;
    }
    return false;
  }

#ifdef ENGINE_ENABLE_XRT
  try {
    const auto end_to_end_start = std::chrono::steady_clock::now();
    const double activation_pack_us = PackActivations(input, impl_->input);
    std::fill_n(impl_->output, kQwenMtpEhProjOutputElements, 0.0F);

    const auto upload_start = std::chrono::steady_clock::now();
    impl_->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    impl_->output_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    const auto upload_end = std::chrono::steady_clock::now();
    const auto command_start = upload_end;
    auto run = (*impl_->kernel)(3U, 0U, 0U, *impl_->weight_bo, *impl_->input_bo,
                                *impl_->output_bo);
    const auto submission_end = std::chrono::steady_clock::now();
    const auto status = run.wait2(std::chrono::milliseconds(timeout_ms_));
    const auto completion_end = std::chrono::steady_clock::now();
    if (status == std::cv_status::timeout) {
      run.abort();
      quarantined_ = true;
      SetFailure(failure, "timeout", "Qwen MTP eh_proj command timed out");
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto command_state = run.state();
    if (command_state != ERT_CMD_STATE_COMPLETED) {
      quarantined_ = true;
      SetFailure(failure, "command_error",
                 "Qwen MTP eh_proj command ended in ERT state " +
                     std::to_string(static_cast<int>(command_state)));
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto download_start = completion_end;
    impl_->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto download_end = std::chrono::steady_clock::now();
    std::ranges::copy_n(impl_->output, kQwenMtpEhProjOutputElements,
                        output.begin());
    if (metrics != nullptr) {
      metrics->activation_pack_us = activation_pack_us;
      metrics->input_upload_us =
          std::chrono::duration<double, std::micro>(upload_end - upload_start)
              .count();
      metrics->submission_us = std::chrono::duration<double, std::micro>(
                                   submission_end - command_start)
                                   .count();
      metrics->completion_us = std::chrono::duration<double, std::micro>(
                                   completion_end - submission_end)
                                   .count();
      metrics->command_us = std::chrono::duration<double, std::micro>(
                                completion_end - command_start)
                                .count();
      metrics->output_download_us = std::chrono::duration<double, std::micro>(
                                        download_end - download_start)
                                        .count();
      metrics->end_to_end_us = std::chrono::duration<double, std::micro>(
                                   download_end - end_to_end_start)
                                   .count();
    }
    return true;
  } catch (const xrt::run::command_error& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "command_error",
        std::string("Qwen MTP eh_proj command failed: ") + exception.what());
  } catch (const std::exception& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen MTP eh_proj XRT failure: ") + exception.what());
  }
  if (metrics != nullptr) {
    metrics->quarantined = true;
  }
  return false;
#else
  (void)input;
  (void)output;
  (void)metrics;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return false;
#endif
}

}  // namespace strix::xdna2
