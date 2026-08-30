#include "src/models/qwen/xdna2/dflash_head.h"

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
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/quant/ggml_dequant.hpp"

#ifdef ENGINE_ENABLE_XRT
#include <gufo/aie_qwen_dflash_head_manifest.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#endif

namespace gufo::xdna2 {
namespace {

constexpr std::size_t kBlockElements = 32;
constexpr std::size_t kInputTileElements = 256;
constexpr std::size_t kGroupsPerTile = kInputTileElements / kBlockElements;
constexpr std::size_t kMmulK = 8;
constexpr std::size_t kMmulRows = 4;
constexpr std::size_t kOutputTileElements = 8;
constexpr std::size_t kTilesPerGroup = kBlockElements / kMmulK;
constexpr std::size_t kBatchGroups = kQwenDFlashHeadBatchRows / kMmulRows;
constexpr std::size_t kMmulActivationElements = kMmulRows * kMmulK;
constexpr std::size_t kMmulWeightElements = kMmulK * kOutputTileElements;
constexpr std::size_t kWeightCodeBytes =
    kGroupsPerTile * kTilesPerGroup * kMmulWeightElements;
constexpr std::size_t kWeightScaleOffset = kWeightCodeBytes;
constexpr std::size_t kInputCodeBytes =
    kBatchGroups * kGroupsPerTile * kTilesPerGroup * kMmulActivationElements;
constexpr std::size_t kInputScaleOffset = kInputCodeBytes;
constexpr std::size_t kRecordBytes = 2304;
constexpr std::size_t kWeightRowChunkBytes = kRecordBytes / kOutputTileElements;
constexpr std::size_t kInputTiles =
    kQwenDFlashHeadInputElements / kInputTileElements;
constexpr std::size_t kOutputTiles =
    kQwenDFlashHeadOutputElements / kOutputTileElements;
constexpr std::size_t kWeightRowBytes = kInputTiles * kWeightRowChunkBytes;
constexpr std::size_t kPackedWeightBytes =
    kQwenDFlashHeadOutputElements * kWeightRowBytes;
constexpr std::size_t kPackedInputBytes = kInputTiles * kRecordBytes;
constexpr std::size_t kOutputTileValues =
    kQwenDFlashHeadBatchRows * kOutputTileElements;
constexpr std::size_t kOutputBytes =
    kQwenDFlashHeadBatchRows * kQwenDFlashHeadOutputElements * sizeof(float);

using BlockQ8 = quant::block_q8_0;

static_assert(kWeightCodeBytes == 2048);
static_assert(kInputCodeBytes == 2048);
static_assert(kWeightRowChunkBytes == 288);
static_assert(sizeof(BlockQ8) == 34);

std::atomic<std::size_t> g_active_sessions{0};
std::atomic<std::size_t> g_active_bos{0};

void SetFailure(QwenDFlashHeadFailure* failure, std::string category,
                std::string message) {
  if (failure != nullptr) {
    failure->category = std::move(category);
    failure->message = std::move(message);
  }
}

void ClearFailure(QwenDFlashHeadFailure* failure) {
  if (failure != nullptr) {
    failure->category.clear();
    failure->message.clear();
  }
}

std::uint8_t* WeightRecordByte(std::uint8_t* weights, std::size_t output_tile,
                               std::size_t input_tile,
                               std::size_t byte_offset) {
  const std::size_t local_row = byte_offset / kWeightRowChunkBytes;
  const std::size_t local_column = byte_offset % kWeightRowChunkBytes;
  const std::size_t output_row =
      (output_tile * kOutputTileElements) + local_row;
  return weights + (output_row * kWeightRowBytes) +
         (input_tile * kWeightRowChunkBytes) + local_column;
}

template<typename T>
void StoreWeightValue(std::uint8_t* weights, std::size_t output_tile,
                      std::size_t input_tile, std::size_t byte_offset,
                      const T& value) {
  std::memcpy(WeightRecordByte(weights, output_tile, input_tile, byte_offset),
              &value, sizeof(T));
}

bool PackQ8Weights(const models::QwenTensorRef& tensor, std::uint8_t* packed,
                   QwenDFlashHeadFailure* failure) {
  if (tensor.type != core::GgmlType::kQ8_0 ||
      tensor.num_elements !=
          kQwenDFlashHeadOutputElements * kQwenDFlashHeadInputElements ||
      tensor.data == nullptr) {
    SetFailure(failure, "invalid_tensor",
               "Qwen DFlash head must be an 8192x5120 Q8_0 tensor");
    return false;
  }

  std::fill_n(packed, kPackedWeightBytes, std::uint8_t{0});
  const auto* rows = static_cast<const std::uint8_t*>(tensor.data);
  const std::size_t source_row_bytes =
      quant::QuantizedRowBytes(tensor.type, kQwenDFlashHeadInputElements);
  for (std::size_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    for (std::size_t input_tile = 0; input_tile < kInputTiles; ++input_tile) {
      for (std::size_t output_index = 0; output_index < kOutputTileElements;
           ++output_index) {
        const std::size_t output_row =
            (output_tile * kOutputTileElements) + output_index;
        for (std::size_t group = 0; group < kGroupsPerTile; ++group) {
          const auto* block = reinterpret_cast<const BlockQ8*>(
              rows + (output_row * source_row_bytes) +
              (((input_tile * kGroupsPerTile) + group) * sizeof(BlockQ8)));
          for (std::size_t tile = 0; tile < 4; ++tile) {
            for (std::size_t k_index = 0; k_index < kMmulK; ++k_index) {
              const std::size_t byte_offset =
                  (((group * 4) + tile) * kMmulWeightElements) +
                  (k_index * kOutputTileElements) + output_index;
              const std::int8_t value = block->qs[(tile * kMmulK) + k_index];
              StoreWeightValue(packed, output_tile, input_tile, byte_offset,
                               value);
            }
          }
          const float scale = quant::Fp16ToFloat(block->d);
          const std::size_t scale_index =
              (group * kOutputTileElements) + output_index;
          StoreWeightValue(packed, output_tile, input_tile,
                           kWeightScaleOffset + (scale_index * sizeof(float)),
                           scale);
        }
      }
    }
  }
  return true;
}

double PackActivations(std::span<const float> input, std::size_t row_count,
                       std::uint8_t* packed) {
  const auto start = std::chrono::steady_clock::now();
  std::fill_n(packed, kPackedInputBytes, std::uint8_t{0});
  for (std::size_t input_tile = 0; input_tile < kInputTiles; ++input_tile) {
    auto* record = packed + (input_tile * kRecordBytes);
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::size_t batch_group = row / kMmulRows;
      const std::size_t local_row = row % kMmulRows;
      for (std::size_t group = 0; group < kGroupsPerTile; ++group) {
        const std::size_t group_start = (row * kQwenDFlashHeadInputElements) +
                                        (input_tile * kInputTileElements) +
                                        (group * kBlockElements);
        float maximum = 0.0F;
        for (std::size_t lane = 0; lane < kBlockElements; ++lane) {
          maximum = std::max(maximum, std::abs(input[group_start + lane]));
        }
        const float scale = maximum == 0.0F ? 1.0F : maximum / 127.0F;
        for (std::size_t tile_index = 0; tile_index < 4; ++tile_index) {
          auto* tile = reinterpret_cast<std::int8_t*>(
              record +
              ((((batch_group * kGroupsPerTile) + group) * 4 + tile_index) *
               kMmulActivationElements));
          for (std::size_t k_index = 0; k_index < kMmulK; ++k_index) {
            const std::size_t lane = (tile_index * kMmulK) + k_index;
            const auto rounded = static_cast<int>(
                std::nearbyint(input[group_start + lane] / scale));
            tile[(local_row * kMmulK) + k_index] =
                static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
          }
        }
        std::memcpy(record + kInputScaleOffset +
                        (((row * kGroupsPerTile) + group) * sizeof(float)),
                    &scale, sizeof(scale));
      }
    }
  }
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

double UnpackOutput(const float* packed, std::size_t row_count,
                    std::span<float> output) {
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t output_tile = 0; output_tile < kOutputTiles; ++output_tile) {
    const float* tile = packed + (output_tile * kOutputTileValues);
    for (std::size_t row = 0; row < row_count; ++row) {
      std::copy_n(tile + (row * kOutputTileElements), kOutputTileElements,
                  output.begin() + (row * kQwenDFlashHeadOutputElements) +
                      (output_tile * kOutputTileElements));
    }
  }
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

#ifdef ENGINE_ENABLE_XRT

std::filesystem::path DefaultProgramDir() {
#ifdef GUFO_AIE_QWEN_DFLASH_HEAD_PROGRAM_DIR
  return GUFO_AIE_QWEN_DFLASH_HEAD_PROGRAM_DIR;
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
                     QwenDFlashHeadFailure* failure) {
  const auto xclbin_path = program_dir / "qwen_dflash_head.xclbin";
  const auto elf_path = program_dir / "qwen_dflash_head.insts.elf";
  if (!IsNonemptyFile(xclbin_path) || !IsNonemptyFile(elf_path)) {
    SetFailure(failure, "program_missing",
               "required Qwen DFlash head XCLBIN or instruction ELF is absent");
    return false;
  }
  const bool compatible =
      generated::kQwenDFlashHeadTarget == "npu2" &&
      generated::kQwenDFlashHeadAbi == "xrt-elf-v1" &&
      generated::kQwenDFlashHeadModelKind == "qwen3.8-27b-dflash2" &&
      generated::kQwenDFlashHeadTensorContract ==
          "q8_0-i8-dyn-i8-g32-int32-fp32-b8-m8192-s1x8192-k5120" &&
      generated::kQwenDFlashHeadPartitionColumns == 8;
  if (!compatible) {
    SetFailure(failure, "program_incompatible",
               "Qwen DFlash head artifact manifest is incompatible");
    return false;
  }
  const std::string xclbin = ReadBinaryFile(xclbin_path);
  const std::string elf = ReadBinaryFile(elf_path);
  const bool hashes_match =
      diagnostics::ComputeSha256Hex(xclbin) ==
          generated::kQwenDFlashHeadXclbinSha256 &&
      diagnostics::ComputeSha256Hex(elf) ==
          generated::kQwenDFlashHeadElfSha256 &&
      diagnostics::ComputeSha256Hex(std::string(xclbin).append(elf)) ==
          generated::kQwenDFlashHeadProgramSha256;
  if (!hashes_match) {
    SetFailure(failure, "program_incompatible",
               "Qwen DFlash head artifact hash does not match its manifest");
    return false;
  }
  return true;
}

#endif

}  // namespace

struct QwenDFlashHeadSession::Impl {
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
  std::uint8_t* input{nullptr};
  float* output{nullptr};
#endif
};

QwenDFlashHeadSession::QwenDFlashHeadSession(
    std::unique_ptr<Impl> impl, QwenDFlashHeadProgramInfo program_info,
    std::uint32_t timeout_ms)
    : impl_(std::move(impl)),
      program_info_(std::move(program_info)),
      timeout_ms_(timeout_ms) {}

QwenDFlashHeadSession::~QwenDFlashHeadSession() = default;
QwenDFlashHeadSession::QwenDFlashHeadSession(QwenDFlashHeadSession&&) noexcept =
    default;
QwenDFlashHeadSession& QwenDFlashHeadSession::operator=(
    QwenDFlashHeadSession&&) noexcept = default;

std::size_t QwenDFlashHeadSession::ActiveSessionCountForDiagnostics() noexcept {
  return g_active_sessions.load(std::memory_order_relaxed);
}

std::size_t QwenDFlashHeadSession::ActiveBoCountForDiagnostics() noexcept {
  return g_active_bos.load(std::memory_order_relaxed);
}

std::unique_ptr<QwenDFlashHeadSession> QwenDFlashHeadSession::Create(
    const QwenDFlashHeadOptions& options, const XrtDeviceInfo& device_info,
    const models::QwenTensorRef& q8_weights, QwenDFlashHeadFailure* failure) {
  ClearFailure(failure);
  if (options.timeout_ms == 0) {
    SetFailure(failure, "invalid_options",
               "Qwen DFlash head timeout must be positive");
    return nullptr;
  }
  if (!device_info.available ||
      device_info.device_index != options.device_index) {
    SetFailure(failure,
               device_info.error_category.empty() ? "device_unavailable"
                                                  : device_info.error_category,
               "Qwen DFlash head requires a compatible discovered device");
    return nullptr;
  }
  if (q8_weights.type != core::GgmlType::kQ8_0 ||
      q8_weights.num_elements !=
          kQwenDFlashHeadOutputElements * kQwenDFlashHeadInputElements ||
      q8_weights.data == nullptr) {
    SetFailure(failure, "invalid_tensor",
               "Qwen DFlash head requires an 8192x5120 Q8_0 weight slice");
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
    QwenDFlashHeadProgramInfo info;
    impl->device = std::make_unique<xrt::device>(options.device_index);
    impl->xclbin = std::make_unique<xrt::xclbin>(
        (program_dir / "qwen_dflash_head.xclbin").string());
    const std::string kernel_name = SelectKernelName(*impl->xclbin);
    if (kernel_name.empty()) {
      SetFailure(failure, "program_incompatible",
                 "Qwen DFlash head XCLBIN has no unique MLIR_AIE kernel");
      return nullptr;
    }
    impl->device->register_xclbin(*impl->xclbin);
    impl->context = std::make_unique<xrt::hw_context>(*impl->device,
                                                      impl->xclbin->get_uuid());
    impl->elf = std::make_unique<xrt::elf>(
        (program_dir / "qwen_dflash_head.insts.elf").string());
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
    auto* packed_weights = impl->weight_bo->map<std::uint8_t*>();
    impl->input = impl->input_bo->map<std::uint8_t*>();
    impl->output = impl->output_bo->map<float*>();
    if (packed_weights == nullptr || impl->input == nullptr ||
        impl->output == nullptr) {
      SetFailure(failure, "buffer_failure",
                 "XRT returned a null Qwen DFlash head BO mapping");
      return nullptr;
    }

    const auto pack_start = std::chrono::steady_clock::now();
    if (!PackQ8Weights(q8_weights, packed_weights, failure)) {
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

    info.target = generated::kQwenDFlashHeadTarget;
    info.abi = generated::kQwenDFlashHeadAbi;
    info.model_kind = generated::kQwenDFlashHeadModelKind;
    info.tensor_contract = generated::kQwenDFlashHeadTensorContract;
    info.program_sha256 = generated::kQwenDFlashHeadProgramSha256;
    info.xclbin_sha256 = generated::kQwenDFlashHeadXclbinSha256;
    info.elf_sha256 = generated::kQwenDFlashHeadElfSha256;
    info.device_name = device_info.name;
    info.device_architecture = device_info.architecture;
    info.driver = device_info.driver;
    info.firmware = device_info.firmware;
    info.xrt_version = generated::kQwenDFlashHeadXrtVersion;
    info.mlir_aie_version = generated::kQwenDFlashHeadMlirAieVersion;
    info.llvm_aie_version = generated::kQwenDFlashHeadLlvmAieVersion;
    info.aiebu_revision = generated::kQwenDFlashHeadAiebuRevision;
    info.xclbin_uuid = impl->xclbin->get_uuid().to_string();
    info.kernel_name = kernel_name;
    info.context_mode =
        impl->context->get_mode() == xrt::hw_context::access_mode::shared
            ? "shared"
            : "exclusive";
    info.partition_columns = generated::kQwenDFlashHeadPartitionColumns;
    info.bo_allocations = 3;
    info.packed_weight_bytes = kPackedWeightBytes;
    info.packed_input_bytes = kPackedInputBytes;
    info.setup_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - setup_start)
                        .count();
    return std::unique_ptr<QwenDFlashHeadSession>(new QwenDFlashHeadSession(
        std::move(impl), std::move(info), options.timeout_ms));
  } catch (const std::exception& exception) {
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen DFlash head XRT setup failed: ") + exception.what());
    return nullptr;
  }
#else
  (void)options;
  (void)device_info;
  (void)q8_weights;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return nullptr;
#endif
}

bool QwenDFlashHeadSession::Run(std::span<const float> input,
                                std::size_t row_count, std::span<float> output,
                                QwenDFlashHeadRunMetrics* metrics,
                                QwenDFlashHeadFailure* failure) {
  ClearFailure(failure);
  if (metrics != nullptr) {
    *metrics = {};
  }
  if (row_count == 0 || row_count > kQwenDFlashHeadBatchRows ||
      input.size() != row_count * kQwenDFlashHeadInputElements ||
      output.size() != row_count * kQwenDFlashHeadOutputElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen DFlash head input/output dimensions are invalid");
    return false;
  }
  if (quarantined_) {
    SetFailure(failure, "session_quarantined",
               "Qwen DFlash head session is quarantined");
    if (metrics != nullptr) {
      metrics->quarantined = true;
    }
    return false;
  }

#ifdef ENGINE_ENABLE_XRT
  try {
    const auto end_to_end_start = std::chrono::steady_clock::now();
    const double activation_pack_us =
        PackActivations(input, row_count, impl_->input);

    const auto upload_start = std::chrono::steady_clock::now();
    impl_->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
      SetFailure(failure, "timeout", "Qwen DFlash head command timed out");
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto command_state = run.state();
    if (command_state != ERT_CMD_STATE_COMPLETED) {
      quarantined_ = true;
      SetFailure(failure, "command_error",
                 "Qwen DFlash head command ended in ERT state " +
                     std::to_string(static_cast<int>(command_state)));
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto download_start = completion_end;
    impl_->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto download_end = std::chrono::steady_clock::now();
    const double output_unpack_us =
        UnpackOutput(impl_->output, row_count, output);
    const auto unpack_end = std::chrono::steady_clock::now();
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
      metrics->output_unpack_us = output_unpack_us;
      metrics->end_to_end_us = std::chrono::duration<double, std::micro>(
                                   unpack_end - end_to_end_start)
                                   .count();
    }
    return true;
  } catch (const xrt::run::command_error& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "command_error",
        std::string("Qwen DFlash head command failed: ") + exception.what());
  } catch (const std::exception& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen DFlash head XRT failure: ") + exception.what());
  }
  if (metrics != nullptr) {
    metrics->quarantined = true;
  }
  return false;
#else
  (void)input;
  (void)row_count;
  (void)output;
  (void)metrics;
  SetFailure(failure, "xrt_uncompiled",
             "binary was built without ENGINE_ENABLE_XRT");
  return false;
#endif
}

}  // namespace gufo::xdna2
