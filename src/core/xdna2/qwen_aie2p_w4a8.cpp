#include "src/core/xdna2/qwen_aie2p_w4a8.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/diagnostics/fingerprint.h"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/core/xdna2/qwen_aie2p_w4a8_pack.hpp"

#ifdef ENGINE_ENABLE_XRT
#include <strix/aie_qwen_aie2p_w4a8_manifest.h>
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

using w4a8::kArrayColumns;
using w4a8::kArrayRows;
using w4a8::kBlockElements;
using w4a8::kInputRecordBytes;
using w4a8::kMmulM;
using w4a8::kMmulN;
using w4a8::kWeightRecordBytes;

// Per-column tile: 4 core rows x 16 lanes = 64 lanes; each lane holds 192 B of
// one 3072 B record (see qwen_aie2p_w4a8.py WEIGHT_RECORD_LANE_BYTES).
constexpr std::size_t kLanesPerColumnTile = kArrayRows * kMmulN;
constexpr std::size_t kWeightGroupBytes =
    kLanesPerColumnTile * (kWeightRecordBytes / kMmulN);

#pragma pack(push, 1)
struct BlockQ4K {
  std::uint16_t d;
  std::uint16_t dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4K) == 144);
static_assert(kWeightRecordBytes == 3072);
static_assert(kInputRecordBytes == 1280);
static_assert(kWeightGroupBytes ==
              kArrayRows * kWeightRecordBytes);  // 4 x 3072 = 12288

std::atomic<std::size_t> g_active_sessions{0};
std::atomic<std::size_t> g_active_bos{0};

void SetFailure(QwenAie2pW4a8Failure* failure, std::string category,
                std::string message) {
  if (failure != nullptr) {
    failure->category = std::move(category);
    failure->message = std::move(message);
  }
}

void ClearFailure(QwenAie2pW4a8Failure* failure) {
  if (failure != nullptr) {
    failure->category.clear();
    failure->message.clear();
  }
}

#ifdef ENGINE_ENABLE_XRT

std::size_t PackedWeightBytes() {
  return static_cast<std::size_t>(kArrayColumns) *
         generated::kQwenAie2pW4a8TilesPerColumn *
         generated::kQwenAie2pW4a8Blocks * kWeightGroupBytes;
}

std::size_t PackedInputBytes() {
  return static_cast<std::size_t>(generated::kQwenAie2pW4a8Blocks) *
         generated::kQwenAie2pW4a8Rounds * kInputRecordBytes;
}

std::size_t OutputBytes() {
  return static_cast<std::size_t>(generated::kQwenAie2pW4a8Rounds) *
         kArrayColumns * generated::kQwenAie2pW4a8TilesPerColumn * 256 *
         sizeof(float);
}

// Pack the Q4_K eh_proj matrix (5120 rows x 10240 columns) into the AIE2P
// weight tensor. Layout (qwen_aie2p_w4a8.py, TensorTiler2D.group_tiler):
//   column p region at p * (TPC*BLOCKS*12288); group = t*BLOCKS + b (tile
//   major, K-block minor); inside a group: core row rc at rc*3072 holds the
//   16 N lanes p*TPC*64 + t*64 + rc*16 + l.
bool PackQ4KWeights(const models::QwenTensorRef& tensor, std::uint8_t* packed,
                    QwenAie2pW4a8Failure* failure) {
  if (tensor.type != core::GgmlType::kQ4_K ||
      tensor.num_elements != kQwenAie2pW4a8OutputElements *
                                 kQwenAie2pW4a8InputElements ||
      tensor.data == nullptr) {
    SetFailure(failure, "invalid_tensor",
               "Qwen AIE2P W4A8 requires the original Q4_K eh_proj matrix");
    return false;
  }

  std::fill_n(packed, PackedWeightBytes(), std::uint8_t{0});
  const auto* rows = static_cast<const std::uint8_t*>(tensor.data);
  const std::size_t source_row_bytes =
      quant::QuantizedRowBytes(tensor.type, kQwenAie2pW4a8InputElements);
  for (std::uint32_t column = 0; column < kArrayColumns; ++column) {
    for (std::uint32_t tile = 0;
         tile < generated::kQwenAie2pW4a8TilesPerColumn; ++tile) {
      for (std::uint32_t block = 0; block < generated::kQwenAie2pW4a8Blocks;
           ++block) {
        std::uint8_t* group_base =
            packed + (static_cast<std::size_t>(
                          (column * generated::kQwenAie2pW4a8TilesPerColumn +
                           tile) *
                              generated::kQwenAie2pW4a8Blocks +
                          block) *
                      kWeightGroupBytes);
        for (std::uint32_t core_row = 0; core_row < kArrayRows; ++core_row) {
          w4a8::WeightBlockSpec spec;
          for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
            const std::size_t output_row =
                static_cast<std::size_t>(column) *
                    generated::kQwenAie2pW4a8TilesPerColumn * 64 +
                static_cast<std::size_t>(tile) * 64 +
                static_cast<std::size_t>(core_row) * kMmulN + lane;
            const auto* src =
                rows + (output_row * source_row_bytes) +
                (static_cast<std::size_t>(block) * sizeof(BlockQ4K));
            w4a8::DecodeQ4KRowIntoSpec(src, lane, spec);
          }
          w4a8::PackWeightBlock(
              spec, group_base + (core_row * kWeightRecordBytes));
        }
      }
    }
  }
  return true;
}

// Quantize `m` activation rows into the input tensor: BLOCKS*ROUNDS records of
// 1280 B, record (block, round) at (b*ROUNDS + r)*1280; all columns share the
// same stream (input_tap pattern_repeat=TPC in qwen_aie2p_w4a8.py).
double PackActivations(std::span<const float> input, std::uint8_t* packed,
                       QwenAie2pW4a8Failure* failure) {
  const std::size_t m =
      input.size() / kQwenAie2pW4a8InputElements;
  if (input.size() % kQwenAie2pW4a8InputElements != 0 || m == 0 ||
      m > kQwenAie2pW4a8MaxM) {
    SetFailure(failure, "invalid_tensor",
               "Qwen AIE2P W4A8 input must be 1..4 rows of 10240 floats");
    return 0.0;
  }
  const auto start = std::chrono::steady_clock::now();
  std::fill_n(packed, PackedInputBytes(), std::uint8_t{0});
  for (std::uint32_t round = 0; round < generated::kQwenAie2pW4a8Rounds;
       ++round) {
    const std::size_t m_base = static_cast<std::size_t>(round) * kMmulM;
    for (std::uint32_t block = 0; block < generated::kQwenAie2pW4a8Blocks;
         ++block) {
      std::span<const float> rows[kMmulM];
      for (std::uint32_t row = 0; row < kMmulM; ++row) {
        const std::size_t logical_row = m_base + row;
        if (logical_row < m) {
          rows[row] = input.subspan(
              (logical_row * kQwenAie2pW4a8InputElements) +
                  (static_cast<std::size_t>(block) * kBlockElements),
              kBlockElements);
        }
      }
      std::uint8_t* record =
          packed + ((static_cast<std::size_t>(block) *
                         generated::kQwenAie2pW4a8Rounds +
                     round) *
                    kInputRecordBytes);
      w4a8::QuantizeActivationBlock(rows, kBlockElements, record);
    }
  }
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - start)
      .count();
}

// Scatter the linear output tensor (ROUNDS*8*TPC*256 floats: column p at
// p*ROUNDS*TPC*256, tile t at t*ROUNDS*256, round r at r*256, core row rc at
// rc*64, then 4 M rows x 16 lanes) into row-major M x N output. Padded rows
// and lanes stay zero.
void UnpackOutput(const float* buffer, std::span<float> output) {
  const std::size_t m = output.size() / kQwenAie2pW4a8OutputElements;
  std::fill_n(output.begin(), output.size(), 0.0F);
  for (std::uint32_t column = 0; column < kArrayColumns; ++column) {
    for (std::uint32_t tile = 0;
         tile < generated::kQwenAie2pW4a8TilesPerColumn; ++tile) {
      for (std::uint32_t core_row = 0; core_row < kArrayRows; ++core_row) {
        for (std::uint32_t lane = 0; lane < kMmulN; ++lane) {
          const std::size_t n =
              static_cast<std::size_t>(column) *
                  generated::kQwenAie2pW4a8TilesPerColumn * 64 +
              static_cast<std::size_t>(tile) * 64 +
              static_cast<std::size_t>(core_row) * kMmulN + lane;
          if (n >= kQwenAie2pW4a8OutputElements) {
            continue;  // padded N lanes stay zero
          }
          for (std::size_t round = 0;
               round < generated::kQwenAie2pW4a8Rounds; ++round) {
            const std::size_t tile_region =
                (static_cast<std::size_t>(column) *
                     generated::kQwenAie2pW4a8Rounds *
                     generated::kQwenAie2pW4a8TilesPerColumn +
                 tile * generated::kQwenAie2pW4a8Rounds + round) *
                256;
            for (std::size_t m_local = 0; m_local < kMmulM; ++m_local) {
              const std::size_t m_row = round * kMmulM + m_local;
              if (m_row >= m) {
                continue;  // padded tail M rows stay zero
              }
              output[m_row * kQwenAie2pW4a8OutputElements + n] =
                  buffer[tile_region + core_row * 64 + m_local * kMmulN + lane];
            }
          }
        }
      }
    }
  }
}

std::filesystem::path DefaultProgramDir() {
#ifdef STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR
  return STRIX_AIE_QWEN_AIE2P_W4A8_PROGRAM_DIR;
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
                     QwenAie2pW4a8Failure* failure) {
  const auto xclbin_path = program_dir / "qwen_aie2p_w4a8.xclbin";
  const auto elf_path = program_dir / "qwen_aie2p_w4a8.insts.elf";
  if (!IsNonemptyFile(xclbin_path) || !IsNonemptyFile(elf_path)) {
    SetFailure(failure, "program_missing",
               "required Qwen AIE2P W4A8 XCLBIN or instruction ELF is absent");
    return false;
  }

  const bool compatible =
      generated::kQwenAie2pW4a8Target == "npu2" &&
      generated::kQwenAie2pW4a8Abi == "xrt-elf-v1" &&
      generated::kQwenAie2pW4a8ModelKind == "qwen3.8-27b-mtp" &&
      generated::kQwenAie2pW4a8TensorContract ==
          "q4_k-u4-dyn-i8-g32-int32-fp32-m1-k10240-n5120-b40-t10-r1" &&
      generated::kQwenAie2pW4a8PartitionColumns == 8 &&
      generated::kQwenAie2pW4a8Blocks * kBlockElements ==
          kQwenAie2pW4a8InputElements &&
      generated::kQwenAie2pW4a8TilesPerColumn * kArrayColumns * kArrayRows *
              kMmulN ==
          kQwenAie2pW4a8OutputElements &&
      generated::kQwenAie2pW4a8Rounds == 1;
  if (!compatible) {
    SetFailure(failure, "program_incompatible",
               "Qwen AIE2P W4A8 artifact manifest is incompatible");
    return false;
  }
  const std::string xclbin = ReadBinaryFile(xclbin_path);
  const std::string elf = ReadBinaryFile(elf_path);
  const bool hashes_match =
      diagnostics::ComputeSha256Hex(xclbin) ==
          generated::kQwenAie2pW4a8XclbinSha256 &&
      diagnostics::ComputeSha256Hex(elf) ==
          generated::kQwenAie2pW4a8ElfSha256 &&
      diagnostics::ComputeSha256Hex(std::string(xclbin).append(elf)) ==
          generated::kQwenAie2pW4a8ProgramSha256;
  if (!hashes_match) {
    SetFailure(failure, "program_incompatible",
               "Qwen AIE2P W4A8 artifact hash does not match its manifest");
    return false;
  }
  return true;
}

#endif

}  // namespace

struct QwenAie2pW4a8Session::Impl {
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

QwenAie2pW4a8Session::QwenAie2pW4a8Session(
    std::unique_ptr<Impl> impl, QwenAie2pW4a8ProgramInfo program_info,
    std::uint32_t timeout_ms)
    : impl_(std::move(impl)),
      program_info_(std::move(program_info)),
      timeout_ms_(timeout_ms) {}

QwenAie2pW4a8Session::~QwenAie2pW4a8Session() = default;
QwenAie2pW4a8Session::QwenAie2pW4a8Session(QwenAie2pW4a8Session&&) noexcept =
    default;
QwenAie2pW4a8Session& QwenAie2pW4a8Session::operator=(
    QwenAie2pW4a8Session&&) noexcept = default;

std::size_t QwenAie2pW4a8Session::ActiveSessionCountForDiagnostics() noexcept {
  return g_active_sessions.load(std::memory_order_relaxed);
}

std::size_t QwenAie2pW4a8Session::ActiveBoCountForDiagnostics() noexcept {
  return g_active_bos.load(std::memory_order_relaxed);
}

std::unique_ptr<QwenAie2pW4a8Session> QwenAie2pW4a8Session::Create(
    const QwenAie2pW4a8Options& options, const XrtDeviceInfo& device_info,
    const models::QwenTensorRef& q4k_weights, QwenAie2pW4a8Failure* failure) {
  ClearFailure(failure);
  if (options.timeout_ms == 0) {
    SetFailure(failure, "invalid_options",
               "Qwen AIE2P W4A8 timeout must be positive");
    return nullptr;
  }
  if (!device_info.available ||
      device_info.device_index != options.device_index) {
    SetFailure(failure,
               device_info.error_category.empty() ? "device_unavailable"
                                                  : device_info.error_category,
               "Qwen AIE2P W4A8 requires a compatible discovered device");
    return nullptr;
  }
  if (q4k_weights.type != core::GgmlType::kQ4_K ||
      q4k_weights.num_elements !=
          kQwenAie2pW4a8OutputElements * kQwenAie2pW4a8InputElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen AIE2P W4A8 requires the original Q4_K eh_proj matrix");
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
    QwenAie2pW4a8ProgramInfo info;
    impl->device = std::make_unique<xrt::device>(options.device_index);
    impl->xclbin = std::make_unique<xrt::xclbin>(
        (program_dir / "qwen_aie2p_w4a8.xclbin").string());
    const std::string kernel_name = SelectKernelName(*impl->xclbin);
    if (kernel_name.empty()) {
      SetFailure(failure, "program_incompatible",
                 "Qwen AIE2P W4A8 XCLBIN has no unique MLIR_AIE kernel");
      return nullptr;
    }
    impl->device->register_xclbin(*impl->xclbin);
    impl->context = std::make_unique<xrt::hw_context>(*impl->device,
                                                      impl->xclbin->get_uuid());
    impl->elf = std::make_unique<xrt::elf>(
        (program_dir / "qwen_aie2p_w4a8.insts.elf").string());
    impl->module = std::make_unique<xrt::module>(*impl->elf);
    impl->kernel = std::make_unique<xrt::ext::kernel>(
        *impl->context, *impl->module, kernel_name);
    impl->weight_bo = std::make_unique<xrt::bo>(
        xrt::ext::bo(*impl->device, PackedWeightBytes()));
    impl->input_bo = std::make_unique<xrt::bo>(
        xrt::ext::bo(*impl->device, PackedInputBytes()));
    impl->output_bo =
        std::make_unique<xrt::bo>(xrt::ext::bo(*impl->device, OutputBytes()));
    impl->buffers_allocated = true;
    g_active_bos.fetch_add(3, std::memory_order_relaxed);
    impl->weight = impl->weight_bo->map<std::uint8_t*>();
    impl->input = impl->input_bo->map<std::uint8_t*>();
    impl->output = impl->output_bo->map<float*>();
    if (impl->weight == nullptr || impl->input == nullptr ||
        impl->output == nullptr) {
      SetFailure(failure, "buffer_failure",
                 "XRT returned a null Qwen AIE2P W4A8 BO mapping");
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

    info.target = generated::kQwenAie2pW4a8Target;
    info.abi = generated::kQwenAie2pW4a8Abi;
    info.model_kind = generated::kQwenAie2pW4a8ModelKind;
    info.tensor_contract = generated::kQwenAie2pW4a8TensorContract;
    info.program_sha256 = generated::kQwenAie2pW4a8ProgramSha256;
    info.xclbin_sha256 = generated::kQwenAie2pW4a8XclbinSha256;
    info.elf_sha256 = generated::kQwenAie2pW4a8ElfSha256;
    info.device_name = device_info.name;
    info.device_architecture = device_info.architecture;
    info.driver = device_info.driver;
    info.firmware = device_info.firmware;
    info.xrt_version = generated::kQwenAie2pW4a8XrtVersion;
    info.mlir_aie_version = generated::kQwenAie2pW4a8MlirAieVersion;
    info.llvm_aie_version = generated::kQwenAie2pW4a8LlvmAieVersion;
    info.aiebu_revision = generated::kQwenAie2pW4a8AiebuRevision;
    info.xclbin_uuid = impl->xclbin->get_uuid().to_string();
    info.kernel_name = kernel_name;
    info.context_mode =
        impl->context->get_mode() == xrt::hw_context::access_mode::shared
            ? "shared"
            : "exclusive";
    info.partition_columns = generated::kQwenAie2pW4a8PartitionColumns;
    info.blocks = generated::kQwenAie2pW4a8Blocks;
    info.tiles_per_column = generated::kQwenAie2pW4a8TilesPerColumn;
    info.rounds = generated::kQwenAie2pW4a8Rounds;
    info.bo_allocations = 3;
    info.packed_weight_bytes = PackedWeightBytes();
    info.packed_input_bytes = PackedInputBytes();
    info.setup_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - setup_start)
                        .count();
    return std::unique_ptr<QwenAie2pW4a8Session>(new QwenAie2pW4a8Session(
        std::move(impl), std::move(info), options.timeout_ms));
  } catch (const std::exception& exception) {
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen AIE2P W4A8 XRT setup failed: ") + exception.what());
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

bool QwenAie2pW4a8Session::Run(std::span<const float> input,
                               std::span<float> output,
                               QwenAie2pW4a8RunMetrics* metrics,
                               QwenAie2pW4a8Failure* failure) {
  ClearFailure(failure);
  const std::size_t m = input.size() / kQwenAie2pW4a8InputElements;
  if (input.size() % kQwenAie2pW4a8InputElements != 0 || m == 0 ||
      m > kQwenAie2pW4a8MaxM ||
      output.size() != m * kQwenAie2pW4a8OutputElements) {
    SetFailure(failure, "invalid_tensor",
               "Qwen AIE2P W4A8 input/output dimensions are invalid");
    return false;
  }
  if (quarantined_) {
    SetFailure(failure, "session_quarantined",
               "Qwen AIE2P W4A8 session is quarantined");
    if (metrics != nullptr) {
      metrics->quarantined = true;
    }
    return false;
  }

#ifdef ENGINE_ENABLE_XRT
  try {
    const auto end_to_end_start = std::chrono::steady_clock::now();
    const double activation_pack_us =
        PackActivations(input, impl_->input, failure);
    std::fill_n(impl_->output, OutputBytes() / sizeof(float), 0.0F);

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
      SetFailure(failure, "timeout",
                 "Qwen AIE2P W4A8 command timed out");
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto command_state = run.state();
    if (command_state != ERT_CMD_STATE_COMPLETED) {
      quarantined_ = true;
      SetFailure(failure, "command_error",
                 "Qwen AIE2P W4A8 command ended in ERT state " +
                     std::to_string(static_cast<int>(command_state)));
      if (metrics != nullptr) {
        metrics->quarantined = true;
      }
      return false;
    }
    const auto download_start = completion_end;
    impl_->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    const auto download_end = std::chrono::steady_clock::now();
    UnpackOutput(impl_->output, output);
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
        std::string("Qwen AIE2P W4A8 command failed: ") + exception.what());
  } catch (const std::exception& exception) {
    quarantined_ = true;
    SetFailure(
        failure, "xrt_error",
        std::string("Qwen AIE2P W4A8 XRT failure: ") + exception.what());
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