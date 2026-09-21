#if defined(ENGINE_ENABLE_HIP)

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/denoiser.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/dit_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::size_t kTimeInputWidth = 256;
constexpr std::size_t kTimeHiddenWidth = 5376;
constexpr std::size_t kTimeOutputWidth = 2688;
constexpr std::size_t kDitModalities = 3;
constexpr std::size_t kFinalModulationSlots = 2;
constexpr std::size_t kBlockModulationWidth =
    kDitModalities * kDitModulationSlots * kDitHiddenSize;
constexpr std::size_t kFinalModulationWidth =
    kFinalModulationSlots * kDitHiddenSize;
constexpr std::size_t kRopeFrequencies = 16;
constexpr std::size_t kMaximumRetainedGateRankingBytes = 512ULL << 20U;
constexpr rocblas_int kGfx1151RocblasBf16Solution = -1162085109;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

bool ValidateTarget(int* device, std::string* error) {
  hipError_t status = hipGetDevice(device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDevice", status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, *device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDeviceProperties", status));
    return false;
  }
  if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 denoising supports only integrated gfx1151 Gufo "
             "Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

struct Faults {
  std::uint64_t minor{0};
  std::uint64_t major{0};
};

Faults ReadFaults() {
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return {};
  }
  return {static_cast<std::uint64_t>(usage.ru_minflt),
          static_cast<std::uint64_t>(usage.ru_majflt)};
}

std::uint64_t ReadSwapBytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  std::uint64_t kib = 0;
  std::string unit;
  while (status >> key) {
    if (key == "VmSwap:") {
      status >> kib >> unit;
      return kib * 1024ULL;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0;
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      SetError(error, HipError("hipMalloc MiniMax H3 denoiser", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
    bytes_ = 0;
  }

  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

bool HasShape(const TensorDescriptor& descriptor, DType dtype,
              std::initializer_list<std::uint64_t> shape) {
  return descriptor.dtype == dtype &&
         descriptor.shape ==
             std::vector<std::uint64_t>(shape.begin(), shape.end());
}

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name, DType dtype,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error,
             "MiniMax H3 denoiser tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (!HasShape(*descriptor, dtype, shape)) {
    SetError(error, "MiniMax H3 denoiser tensor contract differs: " +
                        std::string(name));
    return nullptr;
  }
  return descriptor;
}

bool LoadTensor(const ModelInventory& inventory,
                const TensorDescriptor& descriptor, int device,
                hipStream_t stream, void* staging, std::size_t staging_bytes,
                const CancellationToken* cancellation, DeviceBuffer* output,
                std::string* error) {
  if (descriptor.payload_bytes > std::numeric_limits<std::size_t>::max() ||
      !output->Allocate(static_cast<std::size_t>(descriptor.payload_bytes),
                        error)) {
    return false;
  }
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0) {
    SetError(error, "cannot open MiniMax H3 shard " + path.string());
    output->Reset();
    return false;
  }
  const std::size_t bytes = output->bytes();
  std::size_t copied = 0;
  bool success = true;
  while (copied < bytes) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 tensor load cancelled");
      success = false;
      break;
    }
    const std::size_t chunk = std::min(staging_bytes, bytes - copied);
    std::size_t read_bytes = 0;
    while (read_bytes < chunk) {
      const ssize_t result = pread(
          file, static_cast<std::byte*>(staging) + read_bytes,
          chunk - read_bytes,
          static_cast<off_t>(descriptor.file_offset + copied + read_bytes));
      if (result <= 0) {
        SetError(error,
                 "short read loading MiniMax H3 tensor " + descriptor.name);
        success = false;
        break;
      }
      read_bytes += static_cast<std::size_t>(result);
    }
    if (!success) {
      break;
    }
    hipError_t status = hipSetDevice(device);
    if (status == hipSuccess) {
      status = hipMemcpyAsync(static_cast<std::byte*>(output->data()) + copied,
                              staging, chunk, hipMemcpyHostToDevice, stream);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("copy MiniMax H3 tensor", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  (void)close(file);
  if (!success) {
    output->Reset();
  }
  return success;
}

bool ReadTensorHost(const ModelInventory& inventory,
                    const TensorDescriptor& descriptor, void* output,
                    std::size_t bytes, std::string* error) {
  if (bytes != descriptor.payload_bytes) {
    SetError(error, "invalid MiniMax H3 host tensor read size");
    return false;
  }
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0) {
    SetError(error, "cannot open MiniMax H3 shard " + path.string());
    return false;
  }
  std::size_t completed = 0;
  while (completed < bytes) {
    const ssize_t result = pread(
        file, static_cast<std::byte*>(output) + completed, bytes - completed,
        static_cast<off_t>(descriptor.file_offset + completed));
    if (result <= 0) {
      (void)close(file);
      SetError(error,
               "short host read loading MiniMax H3 tensor " + descriptor.name);
      return false;
    }
    completed += static_cast<std::size_t>(result);
  }
  (void)close(file);
  return true;
}

bool CheckLaunch(std::string_view operation, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(operation, status));
  return false;
}

class Bf16Gemm {
private:
  struct Plan {
    std::size_t batch{0};
    std::size_t rows{0};
    std::size_t columns{0};
    bool selected{false};
  };

public:
  Bf16Gemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    plans_.reserve(8);
  }
  ~Bf16Gemm() { Reset(); }
  void Reset() noexcept {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
      handle_ = nullptr;
    }
    plans_.clear();
  }
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  bool Select(const std::uint16_t* weight, const std::uint16_t* input,
              std::uint16_t* output, DeviceBuffer* reference,
              DeviceBuffer* first_mismatch, std::size_t batch, std::size_t rows,
              std::size_t columns, hipStream_t stream, std::string* error) {
    if (!Prepare(batch, rows, columns, error) ||
        !reference->Allocate(batch * rows * sizeof(std::uint16_t), error) ||
        !first_mismatch->Allocate(sizeof(unsigned long long), error) ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "cannot prepare MiniMax H3 deterministic BF16 GEMM");
      return false;
    }
    if (Execute(weight, input, output, batch, rows, columns) !=
        rocblas_status_success) {
      SetError(error,
               "pinned gfx1151 MiniMax H3 BF16 GEMM solution is unavailable");
      return false;
    }
    hipError_t status =
        hipMemcpyAsync(reference->data(), output, reference->bytes(),
                       hipMemcpyDeviceToDevice, stream);
    if (status == hipSuccess && Execute(weight, input, output, batch, rows,
                                        columns) != rocblas_status_success) {
      status = hipErrorUnknown;
    }
    if (status == hipSuccess) {
      status = hipMemsetAsync(first_mismatch->data(), 0xFF,
                              first_mismatch->bytes(), stream);
    }
    if (status == hipSuccess) {
      dit_ops::LaunchFirstMismatch(
          reference->As<std::uint16_t>(), output, batch * rows,
          first_mismatch->As<unsigned long long>(), stream);
      status = hipPeekAtLastError();
    }
    unsigned long long mismatch = 0;
    if (status == hipSuccess) {
      status = hipMemcpyAsync(&mismatch, first_mismatch->data(),
                              sizeof(mismatch), hipMemcpyDeviceToHost, stream);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    reference->Reset();
    first_mismatch->Reset();
    if (status != hipSuccess) {
      SetError(error, HipError("validate MiniMax H3 BF16 GEMM", status));
      return false;
    }
    if (mismatch != std::numeric_limits<unsigned long long>::max()) {
      SetError(error, "MiniMax H3 BF16 GEMM is not byte deterministic");
      return false;
    }
    Find(batch, rows, columns)->selected = true;
    return true;
  }

  bool Run(const std::uint16_t* weight, const std::uint16_t* input,
           std::uint16_t* output, std::size_t batch, std::size_t rows,
           std::size_t columns, hipStream_t stream, std::string* error) {
    Plan* plan = Find(batch, rows, columns);
    if (plan == nullptr || !plan->selected ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "MiniMax H3 BF16 GEMM plan is not selected");
      return false;
    }
    const rocblas_status status =
        Execute(weight, input, output, batch, rows, columns);
    if (status != rocblas_status_success) {
      SetError(error, "MiniMax H3 BF16 GEMM dispatch failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  bool Prepare(std::size_t batch, std::size_t rows, std::size_t columns,
               std::string* error) {
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        batch >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        rows >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        columns >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max())) {
      SetError(error, "invalid MiniMax H3 BF16 GEMM dimensions");
      return false;
    }
    if (Find(batch, rows, columns) == nullptr) {
      plans_.push_back({batch, rows, columns, false});
    }
    return true;
  }

  Plan* Find(std::size_t batch, std::size_t rows, std::size_t columns) {
    for (Plan& plan : plans_) {
      if (plan.batch == batch && plan.rows == rows && plan.columns == columns) {
        return &plan;
      }
    }
    return nullptr;
  }

  rocblas_status Execute(const std::uint16_t* weight,
                         const std::uint16_t* input, std::uint16_t* output,
                         std::size_t batch, std::size_t rows,
                         std::size_t columns) const {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    return rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(columns), &alpha, weight,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(columns), input,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(columns), &beta,
        output, rocblas_datatype_bf16_r, static_cast<rocblas_int>(rows), output,
        rocblas_datatype_bf16_r, static_cast<rocblas_int>(rows),
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index,
        kGfx1151RocblasBf16Solution, 0);
  }

  rocblas_handle handle_{nullptr};
  std::vector<Plan> plans_;
};

class F32Gemm {
public:
  F32Gemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
  }
  ~F32Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  F32Gemm(const F32Gemm&) = delete;
  F32Gemm& operator=(const F32Gemm&) = delete;

  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    std::size_t bytes = 0;
    if (handle_ != nullptr)
      (void)rocblas_get_device_memory_size(handle_, &bytes);
    return bytes;
  }

  bool Run(const float* weight, const float* input, float* output,
           std::size_t batch, std::size_t rows, std::size_t columns,
           hipStream_t stream, std::string* error) {
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        batch >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        rows >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        columns >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "invalid MiniMax H3 F32 projection GEMM");
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const rocblas_status status = rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(columns), &alpha, weight,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(columns), input,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(columns), &beta,
        output, rocblas_datatype_f32_r, static_cast<rocblas_int>(rows), output,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(rows),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      SetError(error, "MiniMax H3 F32 projection GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  rocblas_handle handle_{nullptr};
};

struct RefinerWeights {
  DeviceBuffer norm1;
  DeviceBuffer norm2;
  DeviceBuffer qkv;
  DeviceBuffer query_norm;
  DeviceBuffer key_norm;
  DeviceBuffer output;
  DeviceBuffer fc1;
  DeviceBuffer fc2;
};

bool LoadRefinerBlock(const ModelInventory& inventory, std::size_t block,
                      int device, hipStream_t stream, void* staging,
                      const CancellationToken* cancellation,
                      RefinerWeights* weights, std::uint64_t* loaded_bytes,
                      std::string* error) {
  const std::string prefix =
      "token_refiner.blocks." + std::to_string(block) + ".";
  const std::array<const TensorDescriptor*, 8> tensors = {
      RequireTensor(inventory, prefix + "norm1.weight", DType::kBFloat16,
                    {kDitHiddenSize}, error),
      RequireTensor(inventory, prefix + "norm2.weight", DType::kBFloat16,
                    {kDitHiddenSize}, error),
      RequireTensor(inventory, prefix + "attn.qkv_proj.weight",
                    DType::kBFloat16, {kDitAttentionWidth * 3U, kDitHiddenSize},
                    error),
      RequireTensor(inventory, prefix + "attn.q_norm.weight", DType::kBFloat16,
                    {kDitHeadDimension}, error),
      RequireTensor(inventory, prefix + "attn.k_norm.weight", DType::kBFloat16,
                    {kDitHeadDimension}, error),
      RequireTensor(inventory, prefix + "attn.out_proj.weight",
                    DType::kBFloat16, {kDitHiddenSize, kDitAttentionWidth},
                    error),
      RequireTensor(inventory, prefix + "mlp.fc1.weight", DType::kBFloat16,
                    {kDitFeedForwardSize * 2U, kDitHiddenSize}, error),
      RequireTensor(inventory, prefix + "mlp.fc2.weight", DType::kBFloat16,
                    {kDitHiddenSize, kDitFeedForwardSize}, error),
  };
  if (std::ranges::any_of(tensors, [](const TensorDescriptor* value) {
        return value == nullptr;
      })) {
    return false;
  }
  const std::array<DeviceBuffer*, 8> buffers = {
      &weights->norm1,    &weights->norm2,  &weights->qkv, &weights->query_norm,
      &weights->key_norm, &weights->output, &weights->fc1, &weights->fc2,
  };
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    if (!LoadTensor(inventory, *tensors[index], device, stream, staging,
                    kWeightStagingBytes, cancellation, buffers[index], error)) {
      return false;
    }
    *loaded_bytes += tensors[index]->payload_bytes;
  }
  return true;
}

bool IsFinite(std::span<const float> values, std::string_view name,
              std::string* first_non_finite) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      *first_non_finite = std::string(name) + "[" + std::to_string(index) + "]";
      return false;
    }
  }
  return true;
}

}  // namespace

struct DenoiserSession::Impl {
  const ModelInventory* inventory{nullptr};
  GenerationGeometry geometry;
  PackedLayout layout;
  SigmaSchedule schedule;
  TimeRowSchedule time_rows;
  std::vector<EulerStepPlan> euler_plan;
  std::vector<std::uint8_t> text_tags;
  std::vector<std::vector<std::uint32_t>> row_maps;
  std::vector<std::uint16_t> refined_text_host;
  std::vector<std::uint16_t> block0_modulation_host;
  std::vector<std::uint16_t> final_modulation_host;
  std::vector<float> video_rows_host;
  std::vector<float> audio_rows_host;
  std::vector<std::unique_ptr<DitBlockSession>> blocks;
  std::vector<double> forward_history;

  int device{0};
  hipStream_t stream{nullptr};
  void* staging{nullptr};
  Bf16Gemm gemm;
  F32Gemm projection_gemm;

  DeviceBuffer video_patch_weight;
  DeviceBuffer video_patch_bias;
  DeviceBuffer audio_patch_weight;
  DeviceBuffer audio_patch_bias;
  DeviceBuffer final_norm_weight;
  DeviceBuffer final_video_weight;
  DeviceBuffer final_video_bias;
  DeviceBuffer final_audio_weight;
  DeviceBuffer final_audio_bias;
  DeviceBuffer final_modulation;

  DeviceBuffer video_sample;
  DeviceBuffer audio_sample;
  DeviceBuffer projection_scratch_f32;
  DeviceBuffer hidden_device;
  DeviceBuffer block_row_map;
  DeviceBuffer video_velocity_f32;
  DeviceBuffer audio_velocity_f32;
  DeviceBuffer previous_video_velocity_f32;
  DeviceBuffer previous_audio_velocity_f32;

  DenoiserTelemetry base_telemetry;
  int active_blocks{static_cast<int>(kDitBlocks)};
  int reuse_interval{1};
  bool row_parallel_attention{true};

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  bool CopyAsync(void* target, const void* source, std::size_t bytes,
                 hipMemcpyKind kind, std::string_view label,
                 std::string* error) {
    const hipError_t status =
        hipMemcpyAsync(target, source, bytes, kind, stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  }

  bool Load(const TensorDescriptor& descriptor, DeviceBuffer* output,
            const CancellationToken* cancellation, std::string* error) {
    if (!LoadTensor(*inventory, descriptor, device, stream, staging,
                    kWeightStagingBytes, cancellation, output, error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes += descriptor.payload_bytes;
    return true;
  }

  bool RefineText(std::span<const std::uint16_t> conditioning,
                  const CancellationToken* cancellation, std::string* error) {
    const auto begin = Clock::now();
    const std::size_t rows = layout.text_rows;
    const std::size_t hidden_elements = rows * kDitHiddenSize;
    const std::size_t attention_elements = rows * kDitAttentionWidth;
    const std::size_t qkv_elements = attention_elements * 3U;
    const std::size_t fc1_elements = rows * kDitFeedForwardSize * 2U;
    const std::size_t activated_elements = rows * kDitFeedForwardSize;

    const TensorDescriptor* condition_weight =
        RequireTensor(*inventory, "condition_proj.weight", DType::kBFloat16,
                      {kDitHiddenSize, kH3TextConditionWidth}, error);
    const TensorDescriptor* condition_bias =
        RequireTensor(*inventory, "condition_proj.bias", DType::kBFloat16,
                      {kDitHiddenSize}, error);
    const TensorDescriptor* final_norm =
        RequireTensor(*inventory, "token_refiner.final_norm.weight",
                      DType::kBFloat16, {kDitHiddenSize}, error);
    if (condition_weight == nullptr || condition_bias == nullptr ||
        final_norm == nullptr) {
      return false;
    }

    DeviceBuffer source;
    DeviceBuffer hidden;
    DeviceBuffer norm;
    DeviceBuffer qkv;
    DeviceBuffer query;
    DeviceBuffer key;
    DeviceBuffer value;
    DeviceBuffer heads;
    DeviceBuffer branch;
    DeviceBuffer fc1;
    DeviceBuffer activated;
    DeviceBuffer condition_weight_device;
    DeviceBuffer condition_bias_device;
    DeviceBuffer final_norm_device;
    DeviceBuffer reference;
    DeviceBuffer mismatch;
    const auto bf16 = [](std::size_t elements) {
      return elements * sizeof(std::uint16_t);
    };
    if (!source.Allocate(conditioning.size_bytes(), error) ||
        !hidden.Allocate(bf16(hidden_elements), error) ||
        !norm.Allocate(bf16(hidden_elements), error) ||
        !qkv.Allocate(bf16(qkv_elements), error) ||
        !query.Allocate(bf16(attention_elements), error) ||
        !key.Allocate(bf16(attention_elements), error) ||
        !value.Allocate(bf16(attention_elements), error) ||
        !heads.Allocate(bf16(attention_elements), error) ||
        !branch.Allocate(bf16(hidden_elements), error) ||
        !fc1.Allocate(bf16(fc1_elements), error) ||
        !activated.Allocate(bf16(activated_elements), error) ||
        !Load(*condition_weight, &condition_weight_device, cancellation,
              error) ||
        !Load(*condition_bias, &condition_bias_device, cancellation, error) ||
        !Load(*final_norm, &final_norm_device, cancellation, error) ||
        !CopyAsync(source.data(), conditioning.data(),
                   conditioning.size_bytes(), hipMemcpyHostToDevice,
                   "copy H3 text conditioning", error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes +=
        source.bytes() + hidden.bytes() + norm.bytes() + qkv.bytes() +
        query.bytes() + key.bytes() + value.bytes() + heads.bytes() +
        branch.bytes() + fc1.bytes() + activated.bytes();

    dit_ops::LaunchFillBf16Pattern(hidden.As<std::uint16_t>(), hidden_elements,
                                   0x170001U, stream);
    if (!CheckLaunch("fill H3 text refiner", error) ||
        !gemm.Select(condition_weight_device.As<std::uint16_t>(),
                     source.As<std::uint16_t>(), hidden.As<std::uint16_t>(),
                     &reference, &mismatch, rows, kDitHiddenSize,
                     kH3TextConditionWidth, stream, error)) {
      return false;
    }
    dit_ops::LaunchAddBiasBf16(
        hidden.As<std::uint16_t>(), condition_bias_device.As<std::uint16_t>(),
        static_cast<std::uint32_t>(rows), kDitHiddenSize, stream);
    if (!CheckLaunch("H3 condition projection bias", error)) {
      return false;
    }

    for (std::size_t block = 0; block < 2; ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 text refinement cancelled");
        return false;
      }
      RefinerWeights weights;
      std::uint64_t loaded_bytes = 0;
      if (!LoadRefinerBlock(*inventory, block, device, stream, staging,
                            cancellation, &weights, &loaded_bytes, error)) {
        return false;
      }
      base_telemetry.cumulative_allocation_bytes += loaded_bytes;
      dit_ops::LaunchRmsNorm(
          hidden.As<std::uint16_t>(), weights.norm1.As<std::uint16_t>(),
          norm.As<std::uint16_t>(), static_cast<std::uint32_t>(rows),
          kDitHiddenSize, kDitNormEpsilon, stream);
      if (!CheckLaunch("H3 refiner attention norm", error)) {
        return false;
      }
      if (block == 0) {
        if (!gemm.Select(weights.qkv.As<std::uint16_t>(),
                         norm.As<std::uint16_t>(), qkv.As<std::uint16_t>(),
                         &reference, &mismatch, rows, kDitAttentionWidth * 3U,
                         kDitHiddenSize, stream, error)) {
          return false;
        }
      } else if (!gemm.Run(weights.qkv.As<std::uint16_t>(),
                           norm.As<std::uint16_t>(), qkv.As<std::uint16_t>(),
                           rows, kDitAttentionWidth * 3U, kDitHiddenSize,
                           stream, error)) {
        return false;
      }
      dit_ops::LaunchGroupedQkvNormRope(
          qkv.As<std::uint16_t>(), weights.query_norm.As<std::uint16_t>(),
          weights.key_norm.As<std::uint16_t>(), nullptr, nullptr,
          query.As<std::uint16_t>(), key.As<std::uint16_t>(),
          value.As<std::uint16_t>(), static_cast<std::uint32_t>(rows),
          kDitAttentionHeads, kDitHeadDimension, 0, kDitNormEpsilon, stream);
      dit_ops::LaunchFullAttention(
          query.As<std::uint16_t>(), key.As<std::uint16_t>(),
          value.As<std::uint16_t>(), heads.As<std::uint16_t>(),
          static_cast<std::uint32_t>(rows), kDitAttentionHeads,
          kDitHeadDimension,
          1.0F / std::sqrt(static_cast<float>(kDitHeadDimension)), stream,
          row_parallel_attention);
      if (!CheckLaunch("H3 text refiner attention", error)) {
        return false;
      }
      if (block == 0) {
        if (!gemm.Select(weights.output.As<std::uint16_t>(),
                         heads.As<std::uint16_t>(), branch.As<std::uint16_t>(),
                         &reference, &mismatch, rows, kDitHiddenSize,
                         kDitAttentionWidth, stream, error)) {
          return false;
        }
      } else if (!gemm.Run(weights.output.As<std::uint16_t>(),
                           heads.As<std::uint16_t>(),
                           branch.As<std::uint16_t>(), rows, kDitHiddenSize,
                           kDitAttentionWidth, stream, error)) {
        return false;
      }
      dit_ops::LaunchAddSubtract(
          hidden.As<std::uint16_t>(), branch.As<std::uint16_t>(),
          hidden.As<std::uint16_t>(), hidden_elements, false, stream);
      dit_ops::LaunchRmsNorm(
          hidden.As<std::uint16_t>(), weights.norm2.As<std::uint16_t>(),
          norm.As<std::uint16_t>(), static_cast<std::uint32_t>(rows),
          kDitHiddenSize, kDitNormEpsilon, stream);
      if (!CheckLaunch("H3 refiner residual/norm", error)) {
        return false;
      }
      if (block == 0) {
        if (!gemm.Select(weights.fc1.As<std::uint16_t>(),
                         norm.As<std::uint16_t>(), fc1.As<std::uint16_t>(),
                         &reference, &mismatch, rows, kDitFeedForwardSize * 2U,
                         kDitHiddenSize, stream, error)) {
          return false;
        }
      } else if (!gemm.Run(weights.fc1.As<std::uint16_t>(),
                           norm.As<std::uint16_t>(), fc1.As<std::uint16_t>(),
                           rows, kDitFeedForwardSize * 2U, kDitHiddenSize,
                           stream, error)) {
        return false;
      }
      dit_ops::LaunchSwiGlu(
          fc1.As<std::uint16_t>(), activated.As<std::uint16_t>(),
          static_cast<std::uint32_t>(rows), kDitFeedForwardSize, stream);
      if (!CheckLaunch("H3 refiner SwiGLU", error)) {
        return false;
      }
      if (block == 0) {
        if (!gemm.Select(
                weights.fc2.As<std::uint16_t>(), activated.As<std::uint16_t>(),
                branch.As<std::uint16_t>(), &reference, &mismatch, rows,
                kDitHiddenSize, kDitFeedForwardSize, stream, error)) {
          return false;
        }
      } else if (!gemm.Run(weights.fc2.As<std::uint16_t>(),
                           activated.As<std::uint16_t>(),
                           branch.As<std::uint16_t>(), rows, kDitHiddenSize,
                           kDitFeedForwardSize, stream, error)) {
        return false;
      }
      dit_ops::LaunchAddSubtract(
          hidden.As<std::uint16_t>(), branch.As<std::uint16_t>(),
          hidden.As<std::uint16_t>(), hidden_elements, false, stream);
      if (!CheckLaunch("H3 refiner MLP residual", error)) {
        return false;
      }
    }
    dit_ops::LaunchRmsNorm(
        hidden.As<std::uint16_t>(), final_norm_device.As<std::uint16_t>(),
        hidden.As<std::uint16_t>(), static_cast<std::uint32_t>(rows),
        kDitHiddenSize, kDitNormEpsilon, stream);
    if (!CheckLaunch("H3 refiner final norm", error)) {
      return false;
    }
    refined_text_host.resize(hidden_elements);
    if (!CopyAsync(refined_text_host.data(), hidden.data(),
                   refined_text_host.size() * sizeof(std::uint16_t),
                   hipMemcpyDeviceToHost, "read H3 refined text", error) ||
        hipStreamSynchronize(stream) != hipSuccess) {
      if (error != nullptr && error->empty()) {
        *error = "cannot synchronize MiniMax H3 text refinement";
      }
      return false;
    }
    base_telemetry.text_refiner_ms = Milliseconds(begin, Clock::now());
    return true;
  }

  bool BuildRope(std::vector<std::uint16_t>* cosine,
                 std::vector<std::uint16_t>* sine, std::string* error) {
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, "rope.inv_freq", DType::kFloat32,
                      {kRopeFrequencies}, error);
    if (descriptor == nullptr) {
      return false;
    }
    std::array<float, kRopeFrequencies> inverse{};
    if (!ReadTensorHost(*inventory, *descriptor, inverse.data(),
                        sizeof(inverse), error)) {
      return false;
    }
    cosine->resize(layout.rows() * kDitRopeHalf);
    sine->resize(layout.rows() * kDitRopeHalf);
    for (std::size_t row = 0; row < layout.rows(); ++row) {
      const std::array<double, 3> axes = {layout.positions[row].temporal,
                                          layout.positions[row].height,
                                          layout.positions[row].width};
      for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        for (std::size_t frequency = 0; frequency < kRopeFrequencies;
             ++frequency) {
          const std::size_t index =
              row * kDitRopeHalf + axis * kRopeFrequencies + frequency;
          const float angle =
              static_cast<float>(axes[axis]) * inverse[frequency];
          (*cosine)[index] = ToBf16(std::cos(angle));
          (*sine)[index] = ToBf16(std::sin(angle));
        }
      }
    }
    return true;
  }

  bool BuildTimeEmbedding(DeviceBuffer* output,
                          const CancellationToken* cancellation,
                          std::string* error) {
    const TensorDescriptor* input_weight = RequireTensor(
        *inventory, "time_embedder.proj_in.weight", DType::kFloat32,
        {kTimeHiddenWidth, kTimeInputWidth}, error);
    const TensorDescriptor* input_bias =
        RequireTensor(*inventory, "time_embedder.proj_in.bias", DType::kFloat32,
                      {kTimeHiddenWidth}, error);
    const TensorDescriptor* output_weight = RequireTensor(
        *inventory, "time_embedder.proj_out.weight", DType::kFloat32,
        {kTimeOutputWidth, kTimeHiddenWidth}, error);
    const TensorDescriptor* output_bias =
        RequireTensor(*inventory, "time_embedder.proj_out.bias",
                      DType::kFloat32, {kTimeOutputWidth}, error);
    if (input_weight == nullptr || input_bias == nullptr ||
        output_weight == nullptr || output_bias == nullptr) {
      return false;
    }
    std::vector<float> times(time_rows.time_rows, 0.0F);
    for (int step = 0; step < schedule.evaluations(); ++step) {
      const std::size_t index = static_cast<std::size_t>(step);
      times[time_rows.video_rows[index]] = 1.0F - schedule.video[index];
      times[time_rows.audio_rows[index]] = 1.0F - schedule.audio[index];
    }
    std::vector<float> features(time_rows.time_rows * kTimeInputWidth);
    for (std::size_t row = 0; row < time_rows.time_rows; ++row) {
      for (std::size_t index = 0; index < kTimeInputWidth / 2U; ++index) {
        const float frequency =
            std::exp(-std::log(10000.0F) * static_cast<float>(index) /
                     static_cast<float>(kTimeInputWidth / 2U));
        const float angle = times[row] * frequency;
        features[row * kTimeInputWidth + index] = std::cos(angle);
        features[row * kTimeInputWidth + kTimeInputWidth / 2U + index] =
            std::sin(angle);
      }
    }
    DeviceBuffer input;
    DeviceBuffer hidden;
    DeviceBuffer activated;
    DeviceBuffer projected;
    DeviceBuffer input_weight_device;
    DeviceBuffer input_bias_device;
    DeviceBuffer output_weight_device;
    DeviceBuffer output_bias_device;
    if (!input.Allocate(features.size() * sizeof(float), error) ||
        !hidden.Allocate(time_rows.time_rows * kTimeHiddenWidth * sizeof(float),
                         error) ||
        !activated.Allocate(hidden.bytes(), error) ||
        !projected.Allocate(
            time_rows.time_rows * kTimeOutputWidth * sizeof(float), error) ||
        !output->Allocate(
            time_rows.time_rows * kTimeOutputWidth * sizeof(std::uint16_t),
            error) ||
        !Load(*input_weight, &input_weight_device, cancellation, error) ||
        !Load(*input_bias, &input_bias_device, cancellation, error) ||
        !Load(*output_weight, &output_weight_device, cancellation, error) ||
        !Load(*output_bias, &output_bias_device, cancellation, error) ||
        !CopyAsync(input.data(), features.data(),
                   features.size() * sizeof(float), hipMemcpyHostToDevice,
                   "copy H3 timestep features", error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes +=
        input.bytes() + hidden.bytes() + activated.bytes() + projected.bytes() +
        output->bytes();
    dit_ops::LaunchF32Projection(
        input.As<float>(), input_weight_device.As<float>(),
        input_bias_device.As<float>(), hidden.As<float>(), time_rows.time_rows,
        kTimeInputWidth, kTimeHiddenWidth, stream);
    dit_ops::LaunchSiluF32(hidden.As<float>(), activated.As<float>(),
                           time_rows.time_rows * kTimeHiddenWidth, stream);
    dit_ops::LaunchF32Projection(
        activated.As<float>(), output_weight_device.As<float>(),
        output_bias_device.As<float>(), projected.As<float>(),
        time_rows.time_rows, kTimeHiddenWidth, kTimeOutputWidth, stream);
    // The released timestep MLP is F32. Each BF16 AdaLN projection applies
    // SiLU at the timestep embedding's own precision and casts only the
    // activated result. Rounding before SiLU introduces the same coherent
    // modulation bias in every transformer block.
    dit_ops::LaunchSiluF32(projected.As<float>(), hidden.As<float>(),
                           time_rows.time_rows * kTimeOutputWidth, stream);
    dit_ops::LaunchF32ToBf16(hidden.As<float>(), output->As<std::uint16_t>(),
                             time_rows.time_rows * kTimeOutputWidth, stream);
    if (!CheckLaunch("H3 timestep embedding", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 timestep embedding", status));
      return false;
    }
    return true;
  }

  bool ComputeModulation(std::string_view prefix, std::size_t output_width,
                         const DeviceBuffer& time_embedding, bool select_plan,
                         const CancellationToken* cancellation,
                         std::vector<std::uint16_t>* host_output,
                         std::string* error) {
    const TensorDescriptor* weight = RequireTensor(
        *inventory, std::string(prefix) + ".linear.weight", DType::kBFloat16,
        {output_width, kTimeOutputWidth}, error);
    const TensorDescriptor* bias =
        RequireTensor(*inventory, std::string(prefix) + ".linear.bias",
                      DType::kBFloat16, {output_width}, error);
    if (weight == nullptr || bias == nullptr) {
      return false;
    }
    DeviceBuffer weight_device;
    DeviceBuffer bias_device;
    DeviceBuffer output;
    DeviceBuffer reference;
    DeviceBuffer mismatch;
    if (!Load(*weight, &weight_device, cancellation, error) ||
        !Load(*bias, &bias_device, cancellation, error) ||
        !output.Allocate(
            time_rows.time_rows * output_width * sizeof(std::uint16_t),
            error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes += output.bytes();
    const bool gemm_ok =
        select_plan ? gemm.Select(weight_device.As<std::uint16_t>(),
                                  time_embedding.As<std::uint16_t>(),
                                  output.As<std::uint16_t>(), &reference,
                                  &mismatch, time_rows.time_rows, output_width,
                                  kTimeOutputWidth, stream, error)
                    : gemm.Run(weight_device.As<std::uint16_t>(),
                               time_embedding.As<std::uint16_t>(),
                               output.As<std::uint16_t>(), time_rows.time_rows,
                               output_width, kTimeOutputWidth, stream, error);
    if (!gemm_ok) {
      return false;
    }
    dit_ops::LaunchAddBiasBf16(output.As<std::uint16_t>(),
                               bias_device.As<std::uint16_t>(),
                               time_rows.time_rows, output_width, stream);
    if (!CheckLaunch("H3 AdaLN projection bias", error)) {
      return false;
    }
    host_output->resize(time_rows.time_rows * output_width);
    if (!CopyAsync(host_output->data(), output.data(),
                   host_output->size() * sizeof(std::uint16_t),
                   hipMemcpyDeviceToHost, "read H3 AdaLN projection", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 AdaLN projection", status));
      return false;
    }
    return true;
  }

  bool BuildBlocks(const CancellationToken* cancellation, std::string* error) {
    DeviceBuffer time_embedding;
    if (!BuildTimeEmbedding(&time_embedding, cancellation, error)) {
      return false;
    }
    std::vector<std::uint16_t> rope_cos;
    std::vector<std::uint16_t> rope_sin;
    if (!BuildRope(&rope_cos, &rope_sin, error)) {
      return false;
    }
    std::vector<std::size_t> selected_blocks;
    std::vector<std::vector<std::uint16_t>> ranked_modulations;
    std::vector<std::uint16_t> modulation;
    auto block_workspace = DitBlockWorkspace::Create(
        {.rows = layout.rows(),
         .row_parallel_attention = row_parallel_attention},
        error);
    if (block_workspace == nullptr) {
      return false;
    }

    if (active_blocks == static_cast<int>(kDitBlocks)) {
      selected_blocks.resize(kDitBlocks);
      std::iota(selected_blocks.begin(), selected_blocks.end(), 0);
    } else {
      std::vector<double> gate_scores(kDitBlocks, 0.0);
      const std::size_t modulation_elements =
          static_cast<std::size_t>(time_rows.time_rows) * kBlockModulationWidth;
      const std::size_t ranking_candidates = kDitBlocks - 3;
      const bool retain_modulations =
          modulation_elements <=
              std::numeric_limits<std::size_t>::max() /
                  (ranking_candidates * sizeof(std::uint16_t)) &&
          modulation_elements * ranking_candidates * sizeof(std::uint16_t) <=
              kMaximumRetainedGateRankingBytes;
      if (retain_modulations) {
        ranked_modulations.resize(kDitBlocks);
      }
      for (std::size_t block = 2; block + 1 < kDitBlocks; ++block) {
        if (IsCancelled(cancellation)) {
          SetError(error, "MiniMax H3 gate ranking cancelled");
          return false;
        }
        const std::string prefix =
            "blocks." + std::to_string(block) + ".adaln_proj";
        const auto modulation_begin = Clock::now();
        if (!ComputeModulation(prefix, kBlockModulationWidth, time_embedding,
                               block == 2, cancellation, &modulation, error)) {
          return false;
        }
        base_telemetry.adaln_precompute_ms +=
            Milliseconds(modulation_begin, Clock::now());
        const auto score =
            DenoiserGateScore(modulation, time_rows.time_rows, error);
        if (!score.has_value()) {
          return false;
        }
        gate_scores[block] = *score;
        if (retain_modulations) {
          ranked_modulations[block] = std::move(modulation);
        }
      }
      auto selected =
          SelectGateRankedDenoiserBlocks(gate_scores, active_blocks, error);
      if (!selected.has_value()) {
        return false;
      }
      selected_blocks = std::move(*selected);
      if (retain_modulations) {
        std::vector<bool> selected_mask(kDitBlocks, false);
        for (const std::size_t block : selected_blocks) {
          selected_mask[block] = true;
        }
        for (std::size_t block = 2; block + 1 < kDitBlocks; ++block) {
          if (!selected_mask[block]) {
            std::vector<std::uint16_t>().swap(ranked_modulations[block]);
          }
        }
      }
    }

    if (active_blocks == static_cast<int>(kDitBlocks)) {
      struct CoreLoadResult {
        std::vector<std::unique_ptr<DitBlockSession>> sessions;
        double load_ms{0.0};
        std::uint64_t weight_bytes{0};
        std::uint64_t scratch_bytes{0};
        std::uint64_t cumulative_allocation_bytes{0};
        std::uint64_t gemm_plan_validations{0};
        std::string error;
      };
      const std::size_t rows = layout.rows();
      const std::size_t modulation_rows =
          static_cast<std::size_t>(time_rows.time_rows) * kDitModalities;
      auto core_future = std::async(std::launch::async, [this, rows,
                                                         modulation_rows,
                                                         block_workspace,
                                                         cancellation]() {
        CoreLoadResult result;
        const auto load_begin = Clock::now();
        struct WorkerResult {
          std::vector<std::pair<std::size_t, std::unique_ptr<DitBlockSession>>>
              sessions;
          std::uint64_t weight_bytes{0};
          std::uint64_t scratch_bytes{0};
          std::uint64_t cumulative_allocation_bytes{0};
          std::uint64_t gemm_plan_validations{0};
          std::string error;
        };
        std::atomic<std::size_t> next_block{0};
        const auto worker = [&]() {
          WorkerResult worker_result;
          worker_result.sessions.reserve(kDitBlocks / 2U + 1U);
          while (true) {
            const std::size_t block =
                next_block.fetch_add(1, std::memory_order_relaxed);
            if (block >= kDitBlocks) {
              break;
            }
            if (IsCancelled(cancellation)) {
              worker_result.error = "MiniMax H3 core load cancelled";
              break;
            }
            auto session = DitBlockSession::Create(
                *inventory,
                {.block_index = block,
                 .rows = rows,
                 .modulation_rows = modulation_rows,
                 .row_parallel_attention = row_parallel_attention,
                 .validate_gemm_plans = block == 0},
                block_workspace, cancellation, &worker_result.error);
            if (session == nullptr) {
              break;
            }
            worker_result.weight_bytes += session->weight_bytes();
            worker_result.scratch_bytes += session->activation_bytes();
            worker_result.gemm_plan_validations +=
                session->gemm_plan_validations();
            worker_result.cumulative_allocation_bytes +=
                session->weight_bytes() + session->activation_bytes();
            worker_result.sessions.emplace_back(block, std::move(session));
          }
          return worker_result;
        };
        auto second_worker = std::async(std::launch::async, worker);
        WorkerResult first = worker();
        WorkerResult second = second_worker.get();
        result.sessions.resize(kDitBlocks);
        const auto merge = [&](WorkerResult* worker_result) {
          if (result.error.empty() && !worker_result->error.empty()) {
            result.error = std::move(worker_result->error);
          }
          result.weight_bytes += worker_result->weight_bytes;
          result.scratch_bytes += worker_result->scratch_bytes;
          result.cumulative_allocation_bytes +=
              worker_result->cumulative_allocation_bytes;
          result.gemm_plan_validations += worker_result->gemm_plan_validations;
          for (auto& [block, session] : worker_result->sessions) {
            result.sessions[block] = std::move(session);
          }
        };
        merge(&first);
        merge(&second);
        result.load_ms = Milliseconds(load_begin, Clock::now());
        return result;
      });

      ranked_modulations.resize(kDitBlocks);
      for (const std::size_t block : selected_blocks) {
        if (IsCancelled(cancellation)) {
          SetError(error, "MiniMax H3 AdaLN precompute cancelled");
          return false;
        }
        const std::string prefix =
            "blocks." + std::to_string(block) + ".adaln_proj";
        const auto modulation_begin = Clock::now();
        if (!ComputeModulation(prefix, kBlockModulationWidth, time_embedding,
                               block == 0, cancellation,
                               &ranked_modulations[block], error)) {
          return false;
        }
        base_telemetry.adaln_precompute_ms +=
            Milliseconds(modulation_begin, Clock::now());
      }
      block0_modulation_host = ranked_modulations[0];
      const auto final_begin = Clock::now();
      if (!ComputeModulation("final_layer.adaln_proj", kFinalModulationWidth,
                             time_embedding, true, cancellation,
                             &final_modulation_host, error)) {
        return false;
      }
      base_telemetry.adaln_precompute_ms +=
          Milliseconds(final_begin, Clock::now());

      CoreLoadResult loaded = core_future.get();
      const bool missing_block = std::ranges::any_of(
          loaded.sessions,
          [](const auto& session) { return session == nullptr; });
      if (!loaded.error.empty() || missing_block) {
        SetError(error, loaded.error.empty()
                            ? "MiniMax H3 core load did not return every block"
                            : std::move(loaded.error));
        return false;
      }
      base_telemetry.core_load_ms += loaded.load_ms;
      base_telemetry.core_weight_bytes += loaded.weight_bytes;
      base_telemetry.scratch_bytes += loaded.scratch_bytes;
      base_telemetry.core_gemm_plan_validations += loaded.gemm_plan_validations;
      base_telemetry.cumulative_allocation_bytes +=
          loaded.cumulative_allocation_bytes;
      blocks = std::move(loaded.sessions);
      for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto prepare_begin = Clock::now();
        if (!blocks[index]->PrepareConstants(ranked_modulations[index],
                                             rope_cos, rope_sin, cancellation,
                                             error)) {
          return false;
        }
        base_telemetry.core_load_ms +=
            Milliseconds(prepare_begin, Clock::now());
      }
      return true;
    }

    blocks.reserve(selected_blocks.size());
    for (const std::size_t block : selected_blocks) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 core load cancelled");
        return false;
      }
      if (block < ranked_modulations.size() &&
          !ranked_modulations[block].empty()) {
        modulation = std::move(ranked_modulations[block]);
      } else {
        const std::string prefix =
            "blocks." + std::to_string(block) + ".adaln_proj";
        const auto modulation_begin = Clock::now();
        const bool select_plan =
            active_blocks == static_cast<int>(kDitBlocks) && block == 0;
        if (!ComputeModulation(prefix, kBlockModulationWidth, time_embedding,
                               select_plan, cancellation, &modulation, error)) {
          return false;
        }
        base_telemetry.adaln_precompute_ms +=
            Milliseconds(modulation_begin, Clock::now());
      }
      if (block == 0) {
        block0_modulation_host = modulation;
      }
      const auto load_begin = Clock::now();
      auto session = DitBlockSession::Create(
          *inventory,
          {.block_index = block,
           .rows = layout.rows(),
           .modulation_rows =
               static_cast<std::size_t>(time_rows.time_rows) * kDitModalities,
           .row_parallel_attention = row_parallel_attention,
           .validate_gemm_plans = blocks.empty()},
          block_workspace, cancellation, error);
      if (session == nullptr ||
          !session->PrepareConstants(modulation, rope_cos, rope_sin,
                                     cancellation, error)) {
        return false;
      }
      base_telemetry.core_load_ms += Milliseconds(load_begin, Clock::now());
      base_telemetry.core_weight_bytes += session->weight_bytes();
      base_telemetry.scratch_bytes += session->activation_bytes();
      base_telemetry.core_gemm_plan_validations +=
          session->gemm_plan_validations();
      base_telemetry.cumulative_allocation_bytes +=
          session->weight_bytes() + session->activation_bytes();
      blocks.push_back(std::move(session));
    }
    const auto final_begin = Clock::now();
    if (!ComputeModulation("final_layer.adaln_proj", kFinalModulationWidth,
                           time_embedding, true, cancellation,
                           &final_modulation_host, error)) {
      return false;
    }
    base_telemetry.adaln_precompute_ms +=
        Milliseconds(final_begin, Clock::now());
    return true;
  }

  bool LoadHeads(const CancellationToken* cancellation, std::string* error) {
    const TensorDescriptor* video_weight =
        RequireTensor(*inventory, "video_patch_proj.weight", DType::kFloat32,
                      {kDitHiddenSize, kH3VideoPatchWidth}, error);
    const TensorDescriptor* video_bias =
        RequireTensor(*inventory, "video_patch_proj.bias", DType::kFloat32,
                      {kDitHiddenSize}, error);
    const TensorDescriptor* audio_weight =
        RequireTensor(*inventory, "audio_patch_proj.weight", DType::kFloat32,
                      {kDitHiddenSize, kH3AudioLatentChannels}, error);
    const TensorDescriptor* audio_bias =
        RequireTensor(*inventory, "audio_patch_proj.bias", DType::kFloat32,
                      {kDitHiddenSize}, error);
    const TensorDescriptor* final_norm =
        RequireTensor(*inventory, "final_layer.norm.weight", DType::kBFloat16,
                      {kDitHiddenSize}, error);
    const TensorDescriptor* final_video_weight_descriptor = RequireTensor(
        *inventory, "final_layer.video_out.weight", DType::kFloat32,
        {kH3VideoPatchWidth, kDitHiddenSize}, error);
    const TensorDescriptor* final_video_bias_descriptor =
        RequireTensor(*inventory, "final_layer.video_out.bias", DType::kFloat32,
                      {kH3VideoPatchWidth}, error);
    const TensorDescriptor* final_audio_weight_descriptor = RequireTensor(
        *inventory, "final_layer.audio_out.weight", DType::kFloat32,
        {kH3AudioLatentChannels, kDitHiddenSize}, error);
    const TensorDescriptor* final_audio_bias_descriptor =
        RequireTensor(*inventory, "final_layer.audio_out.bias", DType::kFloat32,
                      {kH3AudioLatentChannels}, error);
    if (video_weight == nullptr || video_bias == nullptr ||
        audio_weight == nullptr || audio_bias == nullptr ||
        final_norm == nullptr || final_video_weight_descriptor == nullptr ||
        final_video_bias_descriptor == nullptr ||
        final_audio_weight_descriptor == nullptr ||
        final_audio_bias_descriptor == nullptr) {
      return false;
    }
    if (!Load(*video_weight, &video_patch_weight, cancellation, error) ||
        !Load(*video_bias, &video_patch_bias, cancellation, error) ||
        !Load(*audio_weight, &audio_patch_weight, cancellation, error) ||
        !Load(*audio_bias, &audio_patch_bias, cancellation, error) ||
        !Load(*final_norm, &final_norm_weight, cancellation, error) ||
        !Load(*final_video_weight_descriptor, &final_video_weight, cancellation,
              error) ||
        !Load(*final_video_bias_descriptor, &final_video_bias, cancellation,
              error) ||
        !Load(*final_audio_weight_descriptor, &final_audio_weight, cancellation,
              error) ||
        !Load(*final_audio_bias_descriptor, &final_audio_bias, cancellation,
              error) ||
        !final_modulation.Allocate(
            final_modulation_host.size() * sizeof(std::uint16_t), error) ||
        !CopyAsync(final_modulation.data(), final_modulation_host.data(),
                   final_modulation.bytes(), hipMemcpyHostToDevice,
                   "copy H3 final modulation", error)) {
      return false;
    }
    base_telemetry.cumulative_allocation_bytes += final_modulation.bytes();
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 persistent heads", status));
      return false;
    }
    return true;
  }

  bool AllocateIo(std::string* error) {
    const std::size_t video_elements =
        kH3VideoLatentChannels *
        static_cast<std::size_t>(geometry.temporal.video_latent_frames) *
        static_cast<std::size_t>(geometry.latent_height) *
        static_cast<std::size_t>(geometry.latent_width);
    const std::size_t audio_elements =
        kH3AudioLatentChannels * kH3AudioTracks *
        static_cast<std::size_t>(geometry.temporal.audio_latent_frames);
    const std::size_t video_rows = layout.video_target_rows;
    const std::size_t audio_rows = layout.audio_target_rows;
    const std::size_t hidden_elements = layout.rows() * kDitHiddenSize;
    const std::size_t maximum_projection_rows =
        std::max(video_rows, audio_rows);
    const auto bf16 = [](std::size_t elements) {
      return elements * sizeof(std::uint16_t);
    };
    if (!video_sample.Allocate(video_elements * sizeof(float), error) ||
        !audio_sample.Allocate(audio_elements * sizeof(float), error) ||
        !projection_scratch_f32.Allocate(
            maximum_projection_rows * kDitHiddenSize * sizeof(float), error) ||
        !hidden_device.Allocate(bf16(hidden_elements), error) ||
        !block_row_map.Allocate(layout.rows() * sizeof(std::uint32_t), error) ||
        !video_velocity_f32.Allocate(video_elements * sizeof(float), error) ||
        !audio_velocity_f32.Allocate(audio_elements * sizeof(float), error) ||
        (reuse_interval > 1 && (!previous_video_velocity_f32.Allocate(
                                    video_elements * sizeof(float), error) ||
                                !previous_audio_velocity_f32.Allocate(
                                    audio_elements * sizeof(float), error)))) {
      return false;
    }
    video_rows_host.resize(video_elements);
    audio_rows_host.resize(audio_elements);
    forward_history.reserve(static_cast<std::size_t>(schedule.evaluations()) +
                            16U);
    const std::array<const DeviceBuffer*, 9> buffers = {
        &video_sample,
        &audio_sample,
        &projection_scratch_f32,
        &hidden_device,
        &block_row_map,
        &video_velocity_f32,
        &audio_velocity_f32,
        &previous_video_velocity_f32,
        &previous_audio_velocity_f32,
    };
    std::uint64_t bytes = projection_gemm.workspace_bytes();
    for (const DeviceBuffer* buffer : buffers) {
      bytes += buffer->bytes();
    }
    base_telemetry.persistent_bytes =
        base_telemetry.core_weight_bytes + video_patch_weight.bytes() +
        video_patch_bias.bytes() + audio_patch_weight.bytes() +
        audio_patch_bias.bytes() + final_norm_weight.bytes() +
        final_video_weight.bytes() + final_video_bias.bytes() +
        final_audio_weight.bytes() + final_audio_bias.bytes() +
        final_modulation.bytes() + bytes;
    base_telemetry.cumulative_allocation_bytes += bytes;
    base_telemetry.peak_live_bytes =
        base_telemetry.persistent_bytes + base_telemetry.scratch_bytes;
    base_telemetry.scratch_address =
        blocks.empty() ? 0 : blocks.front()->scratch_address();
    return true;
  }

  bool UploadLatents(std::span<const float> video_latent,
                     std::span<const float> audio_latent, std::string* error) {
    if (!PatchifyVideo(video_latent, kH3VideoLatentChannels,
                       geometry.temporal.video_latent_frames,
                       geometry.latent_height, geometry.latent_width,
                       video_rows_host, error) ||
        !PackAudio(audio_latent, kH3AudioLatentChannels,
                   geometry.temporal.audio_latent_frames, audio_rows_host,
                   error) ||
        !CopyAsync(video_sample.data(), video_rows_host.data(),
                   video_rows_host.size() * sizeof(float),
                   hipMemcpyHostToDevice, "copy H3 video latent", error) ||
        !CopyAsync(audio_sample.data(), audio_rows_host.data(),
                   audio_rows_host.size() * sizeof(float),
                   hipMemcpyHostToDevice, "copy H3 audio latent", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 latent upload", status));
      return false;
    }
    return true;
  }

  bool ForwardDevice(int step, const CancellationToken* cancellation,
                     bool read_velocity, DenoiserVelocity* velocity,
                     std::string* error) {
    if (step < 0 || step >= schedule.evaluations()) {
      SetError(error, "MiniMax H3 denoiser step is out of range");
      return false;
    }
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 denoiser cancelled before forward");
      return false;
    }
    const auto begin = Clock::now();
    const std::size_t audio_offset = layout.text_rows * kDitHiddenSize;
    const std::size_t video_offset =
        (layout.text_rows + layout.audio_target_rows) * kDitHiddenSize;
    // Compose the token sequence directly in its first trunk input buffer.
    // Separate projected-token allocations and D2D copies serve no purpose.
    if (!projection_gemm.Run(
            video_patch_weight.As<float>(), video_sample.As<float>(),
            projection_scratch_f32.As<float>(), layout.video_target_rows,
            kDitHiddenSize, kH3VideoPatchWidth, stream, error)) {
      return false;
    }
    dit_ops::LaunchAddBiasF32ToBf16(
        projection_scratch_f32.As<float>(), video_patch_bias.As<float>(),
        hidden_device.As<std::uint16_t>() + video_offset,
        static_cast<std::uint32_t>(layout.video_target_rows), kDitHiddenSize,
        stream);
    if (!projection_gemm.Run(
            audio_patch_weight.As<float>(), audio_sample.As<float>(),
            projection_scratch_f32.As<float>(), layout.audio_target_rows,
            kDitHiddenSize, kH3AudioLatentChannels, stream, error)) {
      return false;
    }
    dit_ops::LaunchAddBiasF32ToBf16(
        projection_scratch_f32.As<float>(), audio_patch_bias.As<float>(),
        hidden_device.As<std::uint16_t>() + audio_offset,
        static_cast<std::uint32_t>(layout.audio_target_rows), kDitHiddenSize,
        stream);
    const std::size_t text_bytes =
        refined_text_host.size() * sizeof(std::uint16_t);
    if (!CheckLaunch("H3 patch projections", error) ||
        !CopyAsync(hidden_device.data(), refined_text_host.data(), text_bytes,
                   hipMemcpyHostToDevice, "copy H3 refined text", error) ||
        !CopyAsync(block_row_map.data(),
                   row_maps[static_cast<std::size_t>(step)].data(),
                   block_row_map.bytes(), hipMemcpyHostToDevice,
                   "copy H3 block row map", error)) {
      return false;
    }

    std::uint64_t block_dispatches = 0;
    // A block consumes its input before producing the final gated residual;
    // carry the trunk in place instead of reserving a second full hidden state.
    std::uint16_t* hidden_input = hidden_device.As<std::uint16_t>();
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 denoiser cancelled after block " +
                            std::to_string(block));
        return false;
      }
      DitBlockTelemetry block_telemetry;
      if (!blocks[block]->RunPreparedDevice(
              hidden_input, block_row_map.As<std::uint32_t>(), hidden_input,
              static_cast<void*>(stream), block != 0,
              block + 1U < blocks.size(), cancellation, &block_telemetry,
              error)) {
        return false;
      }
      block_dispatches += block_telemetry.dispatches;
    }

    dit_ops::LaunchFinalAdaLn(
        hidden_input + audio_offset, final_norm_weight.As<std::uint16_t>(),
        final_modulation.As<std::uint16_t>(), time_rows.audio_rows[step],
        projection_scratch_f32.As<float>(),
        static_cast<std::uint32_t>(layout.audio_target_rows), kDitHiddenSize,
        kFinalModulationSlots, 0, 1, kDitNormEpsilon, stream);
    if (!CheckLaunch("H3 final audio normalization", error) ||
        !projection_gemm.Run(
            final_audio_weight.As<float>(), projection_scratch_f32.As<float>(),
            audio_velocity_f32.As<float>(), layout.audio_target_rows,
            kH3AudioLatentChannels, kDitHiddenSize, stream, error)) {
      return false;
    }
    dit_ops::LaunchAddBiasF32(
        audio_velocity_f32.As<float>(), final_audio_bias.As<float>(),
        static_cast<std::uint32_t>(layout.audio_target_rows),
        kH3AudioLatentChannels, stream);
    dit_ops::LaunchFinalAdaLn(
        hidden_input + video_offset, final_norm_weight.As<std::uint16_t>(),
        final_modulation.As<std::uint16_t>(), time_rows.video_rows[step],
        projection_scratch_f32.As<float>(),
        static_cast<std::uint32_t>(layout.video_target_rows), kDitHiddenSize,
        kFinalModulationSlots, 0, 1, kDitNormEpsilon, stream);
    if (!CheckLaunch("H3 final video normalization", error) ||
        !projection_gemm.Run(
            final_video_weight.As<float>(), projection_scratch_f32.As<float>(),
            video_velocity_f32.As<float>(), layout.video_target_rows,
            kH3VideoPatchWidth, kDitHiddenSize, stream, error)) {
      return false;
    }
    dit_ops::LaunchAddBiasF32(
        video_velocity_f32.As<float>(), final_video_bias.As<float>(),
        static_cast<std::uint32_t>(layout.video_target_rows),
        kH3VideoPatchWidth, stream);
    if (!CheckLaunch("H3 final velocity heads", error)) {
      return false;
    }
    if (read_velocity) {
      if (velocity == nullptr ||
          velocity->video.size() !=
              video_velocity_f32.bytes() / sizeof(float) ||
          velocity->audio.size() !=
              audio_velocity_f32.bytes() / sizeof(float)) {
        SetError(error, "MiniMax H3 velocity output shape differs");
        return false;
      }
      if (!CopyAsync(velocity->video.data(), video_velocity_f32.data(),
                     video_velocity_f32.bytes(), hipMemcpyDeviceToHost,
                     "read H3 video velocity", error) ||
          !CopyAsync(velocity->audio.data(), audio_velocity_f32.data(),
                     audio_velocity_f32.bytes(), hipMemcpyDeviceToHost,
                     "read H3 audio velocity", error)) {
        return false;
      }
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 full forward", status));
      return false;
    }
    if (read_velocity && (!IsFinite(velocity->video, "video_velocity",
                                    &base_telemetry.first_non_finite) ||
                          !IsFinite(velocity->audio, "audio_velocity",
                                    &base_telemetry.first_non_finite))) {
      SetError(error, "MiniMax H3 forward produced non-finite velocity");
      return false;
    }
    forward_history.push_back(Milliseconds(begin, Clock::now()));
    // Two patch projections and their casts; two final norms, projections
    // and biases. Memory copies are not model-kernel dispatches.
    base_telemetry.dispatches += 4 + block_dispatches + 6;
    return true;
  }

  void PublishTelemetry(DenoiserTelemetry* telemetry) const {
    if (telemetry == nullptr) {
      return;
    }
    *telemetry = base_telemetry;
    telemetry->forward_ms = forward_history;
  }
};

DenoiserSession::DenoiserSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DenoiserSession::~DenoiserSession() = default;
DenoiserSession::DenoiserSession(DenoiserSession&&) noexcept = default;
DenoiserSession& DenoiserSession::operator=(DenoiserSession&&) noexcept =
    default;

std::unique_ptr<DenoiserSession> DenoiserSession::Create(
    const ModelInventory& inventory, const DenoiserOptions& options,
    const DenoiserConditioning& conditioning,
    const CancellationToken* cancellation, DenoiserTelemetry* telemetry,
    std::string* error) {
  const auto setup_begin = Clock::now();
  const Faults faults_before = ReadFaults();
  if (options.evaluations < 2 || options.evaluations > kH3MaximumEvaluations ||
      options.active_blocks < 3 ||
      options.active_blocks > static_cast<int>(kDitBlocks) ||
      options.reuse_interval < 1 || options.reuse_interval > 32 ||
      options.text_rows == 0 ||
      conditioning.layer50.size() !=
          options.text_rows * kH3TextConditionWidth ||
      (!conditioning.tags.empty() &&
       conditioning.tags.size() != options.text_rows) ||
      IsCancelled(cancellation)) {
    SetError(error, "invalid MiniMax H3 denoiser creation arguments");
    return nullptr;
  }
  auto layout = BuildTextOnlyLayout(options.text_rows, options.geometry, error);
  auto schedule = BuildServingSchedule(options.evaluations, error);
  if (!layout || !schedule) {
    return nullptr;
  }
  auto time_rows = BuildTimeRowSchedule(*schedule, error);
  if (!time_rows) {
    return nullptr;
  }
  auto euler_plan = BuildEulerPlan(*schedule, options.reuse_interval, error);
  if (!euler_plan) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  impl->geometry = options.geometry;
  impl->layout = std::move(*layout);
  impl->schedule = std::move(*schedule);
  impl->time_rows = std::move(*time_rows);
  impl->euler_plan = std::move(*euler_plan);
  impl->active_blocks = options.active_blocks;
  impl->reuse_interval = options.reuse_interval;
  impl->row_parallel_attention = options.row_parallel_attention;
  impl->base_telemetry.row_parallel_attention = options.row_parallel_attention;
  impl->text_tags.assign(conditioning.tags.begin(), conditioning.tags.end());
  if (impl->text_tags.empty()) {
    impl->text_tags.assign(options.text_rows, 1);
  }
  impl->row_maps.reserve(static_cast<std::size_t>(options.evaluations));
  for (int step = 0; step < options.evaluations; ++step) {
    auto rows = BuildModulationRowMap(impl->time_rows, step, impl->layout,
                                      impl->text_tags, error);
    if (!rows) {
      return nullptr;
    }
    impl->row_maps.push_back(std::move(*rows));
  }
  if (!ValidateTarget(&impl->device, error)) {
    return nullptr;
  }
  hipError_t status =
      hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("create MiniMax H3 denoiser resources", status));
    return nullptr;
  }
  if (!impl->RefineText(conditioning.layer50, cancellation, error) ||
      !impl->BuildBlocks(cancellation, error) ||
      !impl->LoadHeads(cancellation, error) || !impl->AllocateIo(error)) {
    return nullptr;
  }
  // Text refinement and modulation precomputation are complete. Inference
  // uses the blocks' shared handle and the FP32 projection handle.
  impl->gemm.Reset();
  (void)hipHostFree(impl->staging);
  impl->staging = nullptr;
  impl->base_telemetry.setup_ms = Milliseconds(setup_begin, Clock::now());
  const Faults faults_after = ReadFaults();
  impl->base_telemetry.minor_page_faults =
      faults_after.minor - faults_before.minor;
  impl->base_telemetry.major_page_faults =
      faults_after.major - faults_before.major;
  impl->base_telemetry.swap_bytes = ReadSwapBytes();
  impl->PublishTelemetry(telemetry);
  return std::unique_ptr<DenoiserSession>(new DenoiserSession(std::move(impl)));
}

bool DenoiserSession::Forward(int step, std::span<const float> video_latent,
                              std::span<const float> audio_latent,
                              const CancellationToken* cancellation,
                              DenoiserVelocity* velocity,
                              DenoiserTelemetry* telemetry,
                              std::string* error) {
  const Faults faults_before = ReadFaults();
  if (impl_ == nullptr || velocity == nullptr) {
    SetError(error, "invalid MiniMax H3 forward output");
    return false;
  }
  velocity->video.resize(impl_->video_velocity_f32.bytes() / sizeof(float));
  velocity->audio.resize(impl_->audio_velocity_f32.bytes() / sizeof(float));
  if (!impl_->UploadLatents(video_latent, audio_latent, error) ||
      !impl_->ForwardDevice(step, cancellation, true, velocity, error)) {
    return false;
  }
  std::swap(velocity->video, impl_->video_rows_host);
  std::swap(velocity->audio, impl_->audio_rows_host);
  if (!UnpatchifyVideo(impl_->video_rows_host, kH3VideoLatentChannels,
                       impl_->geometry.temporal.video_latent_frames,
                       impl_->geometry.latent_height,
                       impl_->geometry.latent_width, velocity->video, error) ||
      !UnpackAudio(impl_->audio_rows_host, kH3AudioLatentChannels,
                   impl_->geometry.temporal.audio_latent_frames,
                   velocity->audio, error)) {
    return false;
  }
  const Faults faults_after = ReadFaults();
  impl_->base_telemetry.minor_page_faults +=
      faults_after.minor - faults_before.minor;
  impl_->base_telemetry.major_page_faults +=
      faults_after.major - faults_before.major;
  impl_->base_telemetry.swap_bytes =
      std::max(impl_->base_telemetry.swap_bytes, ReadSwapBytes());
  impl_->PublishTelemetry(telemetry);
  return true;
}

bool DenoiserSession::Denoise(std::span<float> video_latent,
                              std::span<float> audio_latent,
                              const CancellationToken* cancellation,
                              DenoiserTelemetry* telemetry,
                              std::string* error) {
  return DenoiseWithProgress(video_latent, audio_latent, cancellation, nullptr,
                             nullptr, telemetry, error);
}

bool DenoiserSession::DenoiseWithProgress(
    std::span<float> video_latent, std::span<float> audio_latent,
    const CancellationToken* cancellation, DenoiserProgress progress,
    void* progress_opaque, DenoiserTelemetry* telemetry, std::string* error) {
  const Faults faults_before = ReadFaults();
  if (impl_ == nullptr ||
      !impl_->UploadLatents(video_latent, audio_latent, error)) {
    return false;
  }
  double sampler_ms = 0.0;
  bool have_velocity = false;
  for (int step = 0; step < impl_->schedule.evaluations(); ++step) {
    const EulerStepPlan& plan =
        impl_->euler_plan[static_cast<std::size_t>(step)];
    if (plan.evaluate) {
      if (have_velocity && impl_->reuse_interval > 1 &&
          (!impl_->CopyAsync(impl_->previous_video_velocity_f32.data(),
                             impl_->video_velocity_f32.data(),
                             impl_->video_velocity_f32.bytes(),
                             hipMemcpyDeviceToDevice,
                             "retain H3 previous video velocity", error) ||
           !impl_->CopyAsync(impl_->previous_audio_velocity_f32.data(),
                             impl_->audio_velocity_f32.data(),
                             impl_->audio_velocity_f32.bytes(),
                             hipMemcpyDeviceToDevice,
                             "retain H3 previous audio velocity", error))) {
        return false;
      }
      if (!impl_->ForwardDevice(step, cancellation, false, nullptr, error)) {
        return false;
      }
      if (!have_velocity && impl_->reuse_interval > 1 &&
          (!impl_->CopyAsync(impl_->previous_video_velocity_f32.data(),
                             impl_->video_velocity_f32.data(),
                             impl_->video_velocity_f32.bytes(),
                             hipMemcpyDeviceToDevice,
                             "initialize H3 previous video velocity", error) ||
           !impl_->CopyAsync(impl_->previous_audio_velocity_f32.data(),
                             impl_->audio_velocity_f32.data(),
                             impl_->audio_velocity_f32.bytes(),
                             hipMemcpyDeviceToDevice,
                             "initialize H3 previous audio velocity", error))) {
        return false;
      }
      have_velocity = true;
    }
    const void* previous_video = impl_->reuse_interval > 1
                                     ? impl_->previous_video_velocity_f32.data()
                                     : impl_->video_velocity_f32.data();
    const void* previous_audio = impl_->reuse_interval > 1
                                     ? impl_->previous_audio_velocity_f32.data()
                                     : impl_->audio_velocity_f32.data();
    const auto sampler_begin = Clock::now();
    if (!HipEulerUpdate(impl_->video_sample.data(),
                        impl_->video_sample.bytes() / sizeof(float), 0,
                        impl_->video_velocity_f32.data(), previous_video,
                        impl_->video_velocity_f32.bytes() / sizeof(float),
                        plan.video_sigma_from_timestep, plan.video_ratio,
                        plan.video_extrapolation, impl_->stream, error) ||
        !HipEulerUpdate(impl_->audio_sample.data(),
                        impl_->audio_sample.bytes() / sizeof(float), 0,
                        impl_->audio_velocity_f32.data(), previous_audio,
                        impl_->audio_velocity_f32.bytes() / sizeof(float),
                        plan.audio_sigma_from_timestep, plan.audio_ratio,
                        plan.audio_extrapolation, impl_->stream, error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(impl_->stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 Euler step", status));
      return false;
    }
    sampler_ms += Milliseconds(sampler_begin, Clock::now());
    impl_->base_telemetry.dispatches += 2;
    if (progress != nullptr) {
      progress(step + 1, impl_->schedule.evaluations(), progress_opaque);
    }
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 denoiser cancelled after step " +
                          std::to_string(step + 1));
      return false;
    }
  }
  if (!impl_->CopyAsync(impl_->video_rows_host.data(),
                        impl_->video_sample.data(), impl_->video_sample.bytes(),
                        hipMemcpyDeviceToHost, "read H3 final video latent",
                        error) ||
      !impl_->CopyAsync(impl_->audio_rows_host.data(),
                        impl_->audio_sample.data(), impl_->audio_sample.bytes(),
                        hipMemcpyDeviceToHost, "read H3 final audio latent",
                        error)) {
    return false;
  }
  const hipError_t status = hipStreamSynchronize(impl_->stream);
  if (status != hipSuccess) {
    SetError(error, HipError("synchronize H3 final latent read", status));
    return false;
  }
  if (!UnpatchifyVideo(impl_->video_rows_host, kH3VideoLatentChannels,
                       impl_->geometry.temporal.video_latent_frames,
                       impl_->geometry.latent_height,
                       impl_->geometry.latent_width, video_latent, error) ||
      !UnpackAudio(impl_->audio_rows_host, kH3AudioLatentChannels,
                   impl_->geometry.temporal.audio_latent_frames, audio_latent,
                   error)) {
    return false;
  }
  if (!IsFinite(video_latent, "video_latent",
                &impl_->base_telemetry.first_non_finite) ||
      !IsFinite(audio_latent, "audio_latent",
                &impl_->base_telemetry.first_non_finite)) {
    SetError(error, "MiniMax H3 denoiser produced non-finite latent");
    return false;
  }
  impl_->base_telemetry.sampler_ms = sampler_ms;
  const Faults faults_after = ReadFaults();
  impl_->base_telemetry.minor_page_faults +=
      faults_after.minor - faults_before.minor;
  impl_->base_telemetry.major_page_faults +=
      faults_after.major - faults_before.major;
  impl_->base_telemetry.swap_bytes =
      std::max(impl_->base_telemetry.swap_bytes, ReadSwapBytes());
  impl_->PublishTelemetry(telemetry);
  return true;
}

const GenerationGeometry& DenoiserSession::geometry() const noexcept {
  static const GenerationGeometry empty;
  return impl_ == nullptr ? empty : impl_->geometry;
}

const PackedLayout& DenoiserSession::layout() const noexcept {
  static const PackedLayout empty;
  return impl_ == nullptr ? empty : impl_->layout;
}

const SigmaSchedule& DenoiserSession::schedule() const noexcept {
  static const SigmaSchedule empty;
  return impl_ == nullptr ? empty : impl_->schedule;
}

std::span<const std::uint16_t> DenoiserSession::block0_modulation()
    const noexcept {
  return impl_ == nullptr
             ? std::span<const std::uint16_t>{}
             : std::span<const std::uint16_t>(impl_->block0_modulation_host);
}

std::span<const std::uint16_t> DenoiserSession::refined_text() const noexcept {
  return impl_ == nullptr
             ? std::span<const std::uint16_t>{}
             : std::span<const std::uint16_t>(impl_->refined_text_host);
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
