#if defined(ENGINE_ENABLE_HIP)

// Qwen3-VL layer sequencing and tensor boundaries adapted from antirez/h3.c
// h3_text_encoder.c at 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT).
// Dense projections and residency/prefetch are independent ROCm/HIP code.

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/prompt_encoder_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 5'000'000.0F;
constexpr std::size_t kMaximumPromptTokens = 4096;
constexpr std::size_t kGemmWorkspaceBytes = 64ULL << 20U;
constexpr std::size_t kMaximumGemmAlgorithms = 16;

std::atomic<std::uint64_t> g_registered_host_bytes{0};
std::atomic<bool> g_prompt_encoder_active{false};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

class PromptPhaseLease {
public:
  static std::optional<PromptPhaseLease> Acquire(std::string* error) {
    bool expected = false;
    if (!g_prompt_encoder_active.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      SetError(error, "MiniMax H3 prompt encoder already has an active phase");
      return std::nullopt;
    }
    return PromptPhaseLease();
  }

  ~PromptPhaseLease() {
    if (active_) {
      g_prompt_encoder_active.store(false, std::memory_order_release);
    }
  }
  PromptPhaseLease(const PromptPhaseLease&) = delete;
  PromptPhaseLease& operator=(const PromptPhaseLease&) = delete;
  PromptPhaseLease(PromptPhaseLease&& other) noexcept
      : active_(std::exchange(other.active_, false)) {}
  PromptPhaseLease& operator=(PromptPhaseLease&&) = delete;

private:
  PromptPhaseLease() = default;
  bool active_{true};
};

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
             "MiniMax H3 prompt encoder supports only integrated gfx1151 "
             "Strix Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
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
      SetError(error, HipError("hipMalloc", status));
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

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

struct LoadedTensor {
  const TensorDescriptor* descriptor{nullptr};
  const std::uint16_t* data{nullptr};
};

struct HostMapping;

struct LoadedBatch {
  std::vector<LoadedTensor> tensors;
  std::vector<HostMapping> mappings;
  std::uint64_t bytes{0};
  std::uint64_t peak_registered_bytes{0};
  double load_ms{0.0};

  [[nodiscard]] const std::uint16_t* Find(std::string_view name) const {
    for (const auto& tensor : tensors) {
      if (tensor.descriptor != nullptr && tensor.descriptor->name == name) {
        return tensor.data;
      }
    }
    return nullptr;
  }
};

struct FileInterval {
  std::filesystem::path shard;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct HostMapping {
  std::filesystem::path shard;
  void* host{nullptr};
  std::size_t bytes{0};
  std::uint64_t file_offset{0};
  void* device{nullptr};
  bool registered{false};

  HostMapping() = default;
  HostMapping(const HostMapping&) = delete;
  HostMapping& operator=(const HostMapping&) = delete;
  HostMapping(HostMapping&& other) noexcept
      : shard(std::move(other.shard)),
        host(std::exchange(other.host, nullptr)),
        bytes(std::exchange(other.bytes, 0)),
        file_offset(other.file_offset),
        device(std::exchange(other.device, nullptr)),
        registered(std::exchange(other.registered, false)) {}
  HostMapping& operator=(HostMapping&& other) noexcept {
    if (this != &other) {
      Release();
      shard = std::move(other.shard);
      host = std::exchange(other.host, nullptr);
      bytes = std::exchange(other.bytes, 0);
      file_offset = other.file_offset;
      device = std::exchange(other.device, nullptr);
      registered = std::exchange(other.registered, false);
    }
    return *this;
  }
  ~HostMapping() { Release(); }

  void Release() noexcept {
    if (registered && host != nullptr) {
      (void)hipHostUnregister(host);
      g_registered_host_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
      registered = false;
    }
    if (host != nullptr && bytes != 0) {
      (void)munmap(host, bytes);
      host = nullptr;
    }
    device = nullptr;
    bytes = 0;
  }
};

std::vector<FileInterval> BuildIntervals(
    std::span<const TensorDescriptor* const> descriptors) {
  const long page_size_value = sysconf(_SC_PAGESIZE);
  if (page_size_value <= 0) {
    throw std::runtime_error("cannot determine host page size");
  }
  const auto page_size = static_cast<std::uint64_t>(page_size_value);
  std::vector<FileInterval> intervals;
  intervals.reserve(descriptors.size());
  for (const TensorDescriptor* descriptor : descriptors) {
    if (descriptor->payload_bytes >
        std::numeric_limits<std::uint64_t>::max() - descriptor->file_offset) {
      throw std::overflow_error("H3 tensor mapping interval overflows");
    }
    intervals.push_back({
        .shard = descriptor->shard,
        .begin = (descriptor->file_offset / page_size) * page_size,
        .end = descriptor->file_offset + descriptor->payload_bytes,
    });
  }
  std::ranges::sort(intervals, [](const auto& left, const auto& right) {
    return std::tie(left.shard, left.begin, left.end) <
           std::tie(right.shard, right.begin, right.end);
  });
  std::vector<FileInterval> merged;
  for (const auto& interval : intervals) {
    if (!merged.empty() && merged.back().shard == interval.shard &&
        interval.begin <= merged.back().end) {
      merged.back().end = std::max(merged.back().end, interval.end);
    } else {
      merged.push_back(interval);
    }
  }
  return merged;
}

bool HasShape(const TensorDescriptor& descriptor,
              std::initializer_list<std::uint64_t> expected) {
  return descriptor.dtype == DType::kBFloat16 &&
         descriptor.shape ==
             std::vector<std::uint64_t>(expected.begin(), expected.end());
}

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error, "required H3 text tensor is absent: " + std::string(name));
    return nullptr;
  }
  if (descriptor->phase != Phase::kPromptEncoder ||
      !HasShape(*descriptor, shape)) {
    SetError(error, "H3 text tensor contract mismatch: " + std::string(name));
    return nullptr;
  }
  return descriptor;
}

struct BatchLoadResult {
  std::optional<LoadedBatch> batch;
  std::string error;
};

BatchLoadResult LoadBatch(const ModelInventory& inventory,
                          std::vector<const TensorDescriptor*> descriptors,
                          int device, const CancellationToken* cancellation) {
  const auto begin = Clock::now();
  BatchLoadResult result;
  if (IsCancelled(cancellation)) {
    result.error = "MiniMax H3 prompt encoder cancelled during weight load";
    return result;
  }
  hipError_t status = hipSetDevice(device);
  if (status != hipSuccess) {
    result.error = HipError("hipSetDevice", status);
    return result;
  }

  try {
    LoadedBatch batch;
    std::vector<HostMapping> mappings;
    for (const auto& interval : BuildIntervals(descriptors)) {
      if (IsCancelled(cancellation)) {
        result.error =
            "MiniMax H3 prompt encoder cancelled during file mapping";
        return result;
      }
      const auto path = inventory.model_root() / interval.shard;
      const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (file < 0) {
        result.error = "cannot open H3 text shard " + path.string() + ": " +
                       std::strerror(errno);
        return result;
      }
      const std::uint64_t length64 = interval.end - interval.begin;
      if (length64 > std::numeric_limits<std::size_t>::max()) {
        close(file);
        result.error = "H3 text mapping is too large";
        return result;
      }
      const auto length = static_cast<std::size_t>(length64);
      void* host = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, file,
                        static_cast<off_t>(interval.begin));
      close(file);
      if (host == MAP_FAILED) {
        result.error = "cannot map H3 text shard " + path.string() + ": " +
                       std::strerror(errno);
        return result;
      }
      HostMapping mapping;
      mapping.shard = interval.shard;
      mapping.host = host;
      mapping.bytes = length;
      mapping.file_offset = interval.begin;
      core::PrefaultMappedRange(host, length);
      status = hipHostRegister(host, length,
                               hipHostRegisterMapped | hipHostRegisterReadOnly);
      if (status != hipSuccess) {
        result.error = HipError("hipHostRegister(mapped|readonly)", status);
        return result;
      }
      mapping.registered = true;
      status = hipHostGetDevicePointer(&mapping.device, host, 0);
      if (status != hipSuccess) {
        result.error =
            HipError("hipHostGetDevicePointer(H3 text weight)", status);
        return result;
      }
      const std::uint64_t registered =
          g_registered_host_bytes.fetch_add(length, std::memory_order_acq_rel) +
          length;
      batch.peak_registered_bytes =
          std::max(batch.peak_registered_bytes, registered);
      mappings.push_back(std::move(mapping));
    }

    batch.tensors.reserve(descriptors.size());
    for (const TensorDescriptor* descriptor : descriptors) {
      if (IsCancelled(cancellation)) {
        result.error = "MiniMax H3 prompt encoder cancelled during weight map";
        return result;
      }
      if (descriptor->payload_bytes > std::numeric_limits<std::size_t>::max()) {
        result.error = "H3 text tensor is too large";
        return result;
      }
      const auto bytes = static_cast<std::size_t>(descriptor->payload_bytes);
      const HostMapping* mapping = nullptr;
      for (const auto& candidate : mappings) {
        if (candidate.shard == descriptor->shard &&
            descriptor->file_offset >= candidate.file_offset &&
            descriptor->file_offset + descriptor->payload_bytes <=
                candidate.file_offset + candidate.bytes) {
          mapping = &candidate;
          break;
        }
      }
      if (mapping == nullptr) {
        result.error = "H3 text tensor mapping was not found";
        return result;
      }
      const auto delta = static_cast<std::size_t>(descriptor->file_offset -
                                                  mapping->file_offset);
      LoadedTensor tensor{
          .descriptor = descriptor,
          .data = reinterpret_cast<const std::uint16_t*>(
              static_cast<const std::byte*>(mapping->device) + delta),
      };
      batch.bytes += bytes;
      batch.tensors.push_back(std::move(tensor));
    }
    batch.mappings = std::move(mappings);
    batch.load_ms = Milliseconds(begin, Clock::now());
    result.batch = std::move(batch);
    return result;
  } catch (const std::exception& exception) {
    result.error = "MiniMax H3 prompt weight load failed: " +
                   std::string(exception.what());
    return result;
  }
}

std::vector<const TensorDescriptor*> LayerDescriptors(
    const ModelInventory& inventory, std::size_t layer, std::string* error) {
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  struct Expected {
    std::string_view suffix;
    std::initializer_list<std::uint64_t> shape;
  };
  static const std::array<Expected, 11> kExpected = {{
      {"input_layernorm.weight", {kPromptEncoderHiddenSize}},
      {"self_attn.q_proj.weight",
       {kPromptEncoderQueryHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.k_proj.weight",
       {kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.v_proj.weight",
       {kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.q_norm.weight", {kPromptEncoderHeadDimension}},
      {"self_attn.k_norm.weight", {kPromptEncoderHeadDimension}},
      {"self_attn.o_proj.weight",
       {kPromptEncoderHiddenSize,
        kPromptEncoderQueryHeads * kPromptEncoderHeadDimension}},
      {"post_attention_layernorm.weight", {kPromptEncoderHiddenSize}},
      {"mlp.gate_proj.weight",
       {kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize}},
      {"mlp.up_proj.weight",
       {kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize}},
      {"mlp.down_proj.weight",
       {kPromptEncoderHiddenSize, kPromptEncoderIntermediateSize}},
  }};
  std::vector<const TensorDescriptor*> result;
  result.reserve(kExpected.size());
  for (const auto& expected : kExpected) {
    const TensorDescriptor* descriptor =
        RequireTensor(inventory, prefix + std::string(expected.suffix),
                      expected.shape, error);
    if (descriptor == nullptr) {
      return {};
    }
    result.push_back(descriptor);
  }
  return result;
}

class Bf16Gemm {
public:
  Bf16Gemm() {
    if (hipblasLtCreate(&handle_) != HIPBLAS_STATUS_SUCCESS) {
      handle_ = nullptr;
      return;
    }
    if (hipMalloc(&workspace_, kGemmWorkspaceBytes) == hipSuccess) {
      workspace_capacity_ = kGemmWorkspaceBytes;
    }
  }
  ~Bf16Gemm() {
    plans_.clear();
    if (workspace_ != nullptr) {
      (void)hipFree(workspace_);
    }
    if (handle_ != nullptr) {
      (void)hipblasLtDestroy(handle_);
    }
  }
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return workspace_capacity_;
  }

  bool Run(const std::uint16_t* weight, const std::uint16_t* input,
           std::uint16_t* output, std::size_t batch, std::size_t rows,
           std::size_t columns, hipStream_t stream, std::string* error) {
    Plan* plan = FindPlan(batch, rows, columns, error);
    if (plan == nullptr) {
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const hipblasStatus_t status = hipblasLtMatmul(
        handle_, plan->operation, &alpha, weight, plan->weight_layout, input,
        plan->input_layout, &beta, output, plan->output_layout, output,
        plan->output_layout, &plan->algorithm, workspace_,
        plan->workspace_bytes, stream);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      SetError(error, "hipBLASLt BF16 matmul dispatch failed");
      return false;
    }
    return true;
  }

private:
  struct Plan {
    std::size_t batch{0};
    std::size_t rows{0};
    std::size_t columns{0};
    hipblasLtMatmulDesc_t operation{nullptr};
    hipblasLtMatrixLayout_t weight_layout{nullptr};
    hipblasLtMatrixLayout_t input_layout{nullptr};
    hipblasLtMatrixLayout_t output_layout{nullptr};
    hipblasLtMatmulAlgo_t algorithm{};
    std::size_t workspace_bytes{0};

    ~Plan() {
      if (output_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(output_layout);
      }
      if (input_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(input_layout);
      }
      if (weight_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(weight_layout);
      }
      if (operation != nullptr) {
        (void)hipblasLtMatmulDescDestroy(operation);
      }
    }
  };

  Plan* FindPlan(std::size_t batch, std::size_t rows, std::size_t columns,
                 std::string* error) {
    for (const auto& plan : plans_) {
      if (plan->batch == batch && plan->rows == rows &&
          plan->columns == columns) {
        return plan.get();
      }
    }
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        batch > std::numeric_limits<std::uint32_t>::max() ||
        rows > std::numeric_limits<std::uint32_t>::max() ||
        columns > std::numeric_limits<std::uint32_t>::max()) {
      SetError(error, "invalid hipBLASLt BF16 matmul dimensions");
      return nullptr;
    }
    auto plan = std::make_unique<Plan>();
    plan->batch = batch;
    plan->rows = rows;
    plan->columns = columns;
    if (hipblasLtMatmulDescCreate(&plan->operation, HIPBLAS_COMPUTE_32F,
                                  HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) {
      SetError(error, "cannot create hipBLASLt operation");
      return nullptr;
    }
    const hipblasOperation_t transpose = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescSetAttribute(
            plan->operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
            sizeof(transpose)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->weight_layout, HIP_R_16BF, columns,
                                    rows, columns) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->input_layout, HIP_R_16BF, columns,
                                    batch, columns) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&plan->output_layout, HIP_R_16BF, rows,
                                    batch, rows) != HIPBLAS_STATUS_SUCCESS) {
      SetError(error, "cannot create hipBLASLt BF16 layouts");
      return nullptr;
    }
    hipblasLtMatmulPreference_t preference = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&preference) !=
        HIPBLAS_STATUS_SUCCESS) {
      SetError(error, "cannot create hipBLASLt preference");
      return nullptr;
    }
    const hipblasStatus_t preference_status =
        hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_capacity_, sizeof(workspace_capacity_));
    std::array<hipblasLtMatmulHeuristicResult_t, kMaximumGemmAlgorithms>
        results{};
    int count = 0;
    const hipblasStatus_t heuristic_status =
        preference_status == HIPBLAS_STATUS_SUCCESS
            ? hipblasLtMatmulAlgoGetHeuristic(
                  handle_, plan->operation, plan->weight_layout,
                  plan->input_layout, plan->output_layout, plan->output_layout,
                  preference, static_cast<int>(results.size()), results.data(),
                  &count)
            : preference_status;
    (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (heuristic_status != HIPBLAS_STATUS_SUCCESS || count <= 0) {
      SetError(error, "hipBLASLt has no BF16 algorithm for H3 projection");
      return nullptr;
    }
    bool selected = false;
    for (int index = 0; index < count; ++index) {
      if (results[index].state == HIPBLAS_STATUS_SUCCESS &&
          results[index].workspaceSize <= workspace_capacity_) {
        plan->algorithm = results[index].algo;
        plan->workspace_bytes = results[index].workspaceSize;
        selected = true;
        break;
      }
    }
    if (!selected) {
      SetError(error, "hipBLASLt BF16 algorithms exceed workspace");
      return nullptr;
    }
    Plan* result = plan.get();
    plans_.push_back(std::move(plan));
    return result;
  }

  hipblasLtHandle_t handle_{nullptr};
  void* workspace_{nullptr};
  std::size_t workspace_capacity_{0};
  std::vector<std::unique_ptr<Plan>> plans_;
};

struct Activations {
  DeviceBuffer ids;
  DeviceBuffer hidden;
  DeviceBuffer norm;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer rope_cosine;
  DeviceBuffer rope_sine;
  DeviceBuffer value;
  DeviceBuffer attention;
  DeviceBuffer projection;
  DeviceBuffer gate;
  DeviceBuffer up;
  DeviceBuffer mlp;

  [[nodiscard]] std::uint64_t Bytes() const noexcept {
    return ids.bytes() + hidden.bytes() + norm.bytes() + query.bytes() +
           key.bytes() + rope_cosine.bytes() + rope_sine.bytes() +
           value.bytes() + attention.bytes() + projection.bytes() +
           gate.bytes() + up.bytes() + mlp.bytes();
  }
};

bool AllocateActivations(std::size_t tokens, Activations* activations,
                         std::string* error) {
  const auto allocate_bf16 = [&](DeviceBuffer* buffer, std::size_t elements) {
    if (elements >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
      SetError(error, "H3 prompt activation size overflows");
      return false;
    }
    return buffer->Allocate(elements * sizeof(std::uint16_t), error);
  };
  const auto allocate_f32 = [&](DeviceBuffer* buffer, std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      SetError(error, "H3 prompt activation size overflows");
      return false;
    }
    return buffer->Allocate(elements * sizeof(float), error);
  };
  return activations->ids.Allocate(tokens * sizeof(std::uint32_t), error) &&
         allocate_bf16(&activations->hidden,
                       tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->norm, tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->query, tokens * kPromptEncoderQueryHeads *
                                                kPromptEncoderHeadDimension) &&
         allocate_bf16(&activations->key, tokens * kPromptEncoderKeyValueHeads *
                                              kPromptEncoderHeadDimension) &&
         allocate_f32(&activations->rope_cosine,
                      tokens * (kPromptEncoderHeadDimension / 2U)) &&
         allocate_f32(&activations->rope_sine,
                      tokens * (kPromptEncoderHeadDimension / 2U)) &&
         allocate_bf16(&activations->value, tokens *
                                                kPromptEncoderKeyValueHeads *
                                                kPromptEncoderHeadDimension) &&
         allocate_bf16(
             &activations->attention,
             tokens * kPromptEncoderQueryHeads * kPromptEncoderHeadDimension) &&
         allocate_bf16(&activations->projection,
                       tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->gate,
                       tokens * kPromptEncoderIntermediateSize) &&
         allocate_bf16(&activations->up,
                       tokens * kPromptEncoderIntermediateSize) &&
         allocate_bf16(&activations->mlp, tokens * kPromptEncoderHiddenSize);
}

bool RunLayer(const LoadedBatch& weights, std::size_t layer, std::size_t tokens,
              Bf16Gemm* gemm, Activations* activations, hipStream_t stream,
              PromptEncoderLayerTelemetry* telemetry, std::string* error) {
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  const auto tensor = [&](std::string_view suffix) {
    return weights.Find(prefix + std::string(suffix));
  };
  const std::uint16_t* input_norm = tensor("input_layernorm.weight");
  const std::uint16_t* query_weight = tensor("self_attn.q_proj.weight");
  const std::uint16_t* key_weight = tensor("self_attn.k_proj.weight");
  const std::uint16_t* value_weight = tensor("self_attn.v_proj.weight");
  const std::uint16_t* query_norm = tensor("self_attn.q_norm.weight");
  const std::uint16_t* key_norm = tensor("self_attn.k_norm.weight");
  const std::uint16_t* output_weight = tensor("self_attn.o_proj.weight");
  const std::uint16_t* post_norm = tensor("post_attention_layernorm.weight");
  const std::uint16_t* gate_weight = tensor("mlp.gate_proj.weight");
  const std::uint16_t* up_weight = tensor("mlp.up_proj.weight");
  const std::uint16_t* down_weight = tensor("mlp.down_proj.weight");
  if (input_norm == nullptr || query_weight == nullptr ||
      key_weight == nullptr || value_weight == nullptr ||
      query_norm == nullptr || key_norm == nullptr ||
      output_weight == nullptr || post_norm == nullptr ||
      gate_weight == nullptr || up_weight == nullptr ||
      down_weight == nullptr) {
    SetError(error, "loaded H3 text layer is incomplete");
    return false;
  }
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  if (hipEventCreate(&start_event) != hipSuccess ||
      hipEventCreate(&stop_event) != hipSuccess) {
    if (start_event != nullptr) {
      (void)hipEventDestroy(start_event);
    }
    SetError(error, "cannot create H3 prompt encoder timing events");
    return false;
  }
  const auto submit_begin = Clock::now();
  (void)hipEventRecord(start_event, stream);
  const auto token_count = static_cast<std::uint32_t>(tokens);
  ops::LaunchRmsNorm(activations->hidden.As<std::uint16_t>(), input_norm,
                     activations->norm.As<std::uint16_t>(), token_count,
                     kPromptEncoderHiddenSize, kRmsEpsilon, stream);
  if (!gemm->Run(query_weight, activations->norm.As<std::uint16_t>(),
                 activations->query.As<std::uint16_t>(), tokens,
                 kPromptEncoderQueryHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error) ||
      !gemm->Run(key_weight, activations->norm.As<std::uint16_t>(),
                 activations->key.As<std::uint16_t>(), tokens,
                 kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error) ||
      !gemm->Run(value_weight, activations->norm.As<std::uint16_t>(),
                 activations->value.As<std::uint16_t>(), tokens,
                 kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchHeadRmsNorm(activations->query.As<std::uint16_t>(), query_norm,
                         token_count, kPromptEncoderQueryHeads,
                         kPromptEncoderHeadDimension, kRmsEpsilon, stream);
  ops::LaunchHeadRmsNorm(activations->key.As<std::uint16_t>(), key_norm,
                         token_count, kPromptEncoderKeyValueHeads,
                         kPromptEncoderHeadDimension, kRmsEpsilon, stream);
  ops::LaunchRope(activations->query.As<std::uint16_t>(),
                  activations->key.As<std::uint16_t>(),
                  activations->rope_cosine.As<float>(),
                  activations->rope_sine.As<float>(), token_count,
                  kPromptEncoderQueryHeads, kPromptEncoderKeyValueHeads,
                  kPromptEncoderHeadDimension, stream);
  ops::LaunchCausalGqa(
      activations->query.As<std::uint16_t>(),
      activations->key.As<std::uint16_t>(),
      activations->value.As<std::uint16_t>(),
      activations->attention.As<std::uint16_t>(), token_count,
      kPromptEncoderQueryHeads, kPromptEncoderKeyValueHeads,
      kPromptEncoderHeadDimension,
      1.0F / std::sqrt(static_cast<float>(kPromptEncoderHeadDimension)),
      stream);
  if (!gemm->Run(output_weight, activations->attention.As<std::uint16_t>(),
                 activations->projection.As<std::uint16_t>(), tokens,
                 kPromptEncoderHiddenSize,
                 kPromptEncoderQueryHeads * kPromptEncoderHeadDimension, stream,
                 error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchAdd(activations->hidden.As<std::uint16_t>(),
                 activations->projection.As<std::uint16_t>(),
                 activations->hidden.As<std::uint16_t>(),
                 tokens * kPromptEncoderHiddenSize, stream);
  ops::LaunchRmsNorm(activations->hidden.As<std::uint16_t>(), post_norm,
                     activations->norm.As<std::uint16_t>(), token_count,
                     kPromptEncoderHiddenSize, kRmsEpsilon, stream);
  if (!gemm->Run(gate_weight, activations->norm.As<std::uint16_t>(),
                 activations->gate.As<std::uint16_t>(), tokens,
                 kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize,
                 stream, error) ||
      !gemm->Run(up_weight, activations->norm.As<std::uint16_t>(),
                 activations->up.As<std::uint16_t>(), tokens,
                 kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize,
                 stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchSwiGlu(activations->gate.As<std::uint16_t>(),
                    activations->up.As<std::uint16_t>(),
                    activations->gate.As<std::uint16_t>(),
                    tokens * kPromptEncoderIntermediateSize, stream);
  if (!gemm->Run(down_weight, activations->gate.As<std::uint16_t>(),
                 activations->mlp.As<std::uint16_t>(), tokens,
                 kPromptEncoderHiddenSize, kPromptEncoderIntermediateSize,
                 stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchAdd(activations->hidden.As<std::uint16_t>(),
                 activations->mlp.As<std::uint16_t>(),
                 activations->hidden.As<std::uint16_t>(),
                 tokens * kPromptEncoderHiddenSize, stream);
  const hipError_t launch_status = hipGetLastError();
  if (launch_status != hipSuccess) {
    SetError(error, HipError("H3 prompt encoder kernel launch", launch_status));
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  (void)hipEventRecord(stop_event, stream);
  telemetry->submit_ms = Milliseconds(submit_begin, Clock::now());
  const hipError_t wait_status = hipEventSynchronize(stop_event);
  if (wait_status != hipSuccess) {
    SetError(error, HipError("H3 prompt encoder layer wait", wait_status));
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  float elapsed_ms = 0.0F;
  if (hipEventElapsedTime(&elapsed_ms, start_event, stop_event) != hipSuccess) {
    SetError(error, "cannot read H3 prompt encoder GPU timing");
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  telemetry->gpu_ms = elapsed_ms;
  telemetry->dispatches = 17;
  (void)hipEventDestroy(stop_event);
  (void)hipEventDestroy(start_event);
  return true;
}

}  // namespace

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return g_registered_host_bytes.load(std::memory_order_acquire);
}

bool EncodePromptLayer50(const ModelInventory& inventory,
                         std::span<const std::uint32_t> token_ids,
                         const PromptEncoderOptions& options,
                         const CancellationToken* cancellation,
                         PromptEmbedding* output,
                         PromptEncoderTelemetry* telemetry,
                         std::string* error) {
  if (output != nullptr) {
    *output = {};
  }
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (token_ids.size() > kMaximumPromptTokens) {
    SetError(error, "MiniMax H3 prompt exceeds 4096 tokens");
    return false;
  }
  if (output == nullptr || telemetry == nullptr || token_ids.empty() ||
      options.layer_count == 0 || options.layer_count > kPromptEncoderLayers) {
    SetError(error, "invalid MiniMax H3 prompt encoder arguments");
    return false;
  }
  if (options.vision_inputs || options.deepstack_inputs) {
    SetError(error,
             "MiniMax H3 initial prompt encoder is text-only; vision and "
             "deepstack inputs are rejected");
    return false;
  }
  for (const std::uint32_t token : token_ids) {
    if (token >= kPromptEncoderVocabularySize) {
      SetError(error, "MiniMax H3 prompt token is outside the vocabulary");
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 prompt encoder cancelled");
    return false;
  }
  auto phase_lease = PromptPhaseLease::Acquire(error);
  if (!phase_lease.has_value()) {
    return false;
  }

  int device = 0;
  if (!ValidateTarget(&device, error)) {
    return false;
  }
  hipStream_t compute_stream = nullptr;
  hipError_t status =
      hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    SetError(error, HipError("hipStreamCreateWithFlags", status));
    return false;
  }
  const auto destroy_stream = [&]() {
    if (compute_stream != nullptr) {
      (void)hipStreamDestroy(compute_stream);
      compute_stream = nullptr;
    }
  };

  Activations activations;
  if (!AllocateActivations(token_ids.size(), &activations, error)) {
    destroy_stream();
    return false;
  }
  telemetry->activation_bytes = activations.Bytes();
  status = hipMemcpyAsync(activations.ids.data(), token_ids.data(),
                          token_ids.size_bytes(), hipMemcpyHostToDevice,
                          compute_stream);
  if (status != hipSuccess) {
    SetError(error, HipError("hipMemcpyAsync(prompt token IDs)", status));
    destroy_stream();
    return false;
  }
  ops::LaunchPrepareRope(
      activations.rope_cosine.As<float>(), activations.rope_sine.As<float>(),
      static_cast<std::uint32_t>(token_ids.size()), kPromptEncoderHeadDimension,
      kRopeTheta, compute_stream);

  const TensorDescriptor* embedding_descriptor = RequireTensor(
      inventory, "model.language_model.embed_tokens.weight",
      {kPromptEncoderVocabularySize, kPromptEncoderHiddenSize}, error);
  if (embedding_descriptor == nullptr) {
    destroy_stream();
    return false;
  }
  BatchLoadResult embedding_result =
      LoadBatch(inventory, {embedding_descriptor}, device, cancellation);
  if (!embedding_result.batch.has_value()) {
    SetError(error, std::move(embedding_result.error));
    destroy_stream();
    return false;
  }
  LoadedBatch embedding = std::move(*embedding_result.batch);
  telemetry->embedding_weight_bytes = embedding.bytes;
  telemetry->peak_registered_host_bytes = std::max(
      telemetry->peak_registered_host_bytes, embedding.peak_registered_bytes);
  telemetry->peak_device_bytes = telemetry->activation_bytes + embedding.bytes;
  const std::uint16_t* embedding_weight =
      embedding.Find("model.language_model.embed_tokens.weight");
  if (embedding_weight == nullptr) {
    SetError(error, "loaded H3 embedding tensor is absent");
    destroy_stream();
    return false;
  }
  ops::LaunchEmbedding(embedding_weight, activations.ids.As<std::uint32_t>(),
                       activations.hidden.As<std::uint16_t>(),
                       static_cast<std::uint32_t>(token_ids.size()),
                       kPromptEncoderVocabularySize, kPromptEncoderHiddenSize,
                       compute_stream);
  status = hipStreamSynchronize(compute_stream);
  if (status != hipSuccess) {
    SetError(error, HipError("H3 embedding lookup", status));
    destroy_stream();
    return false;
  }
  embedding = {};

  Bf16Gemm gemm;
  telemetry->peak_device_bytes = std::max<std::uint64_t>(
      telemetry->peak_device_bytes,
      telemetry->activation_bytes + gemm.workspace_bytes());

  std::string descriptor_error;
  auto current_descriptors = LayerDescriptors(inventory, 0, &descriptor_error);
  if (current_descriptors.empty()) {
    SetError(error, std::move(descriptor_error));
    destroy_stream();
    return false;
  }
  auto load_start = Clock::now();
  BatchLoadResult current_result = LoadBatch(
      inventory, std::move(current_descriptors), device, cancellation);
  double current_wait_ms = Milliseconds(load_start, Clock::now());
  if (!current_result.batch.has_value()) {
    SetError(error, std::move(current_result.error));
    destroy_stream();
    return false;
  }
  std::optional<LoadedBatch> current = std::move(*current_result.batch);

  for (std::size_t layer = 0; layer < options.layer_count; ++layer) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 prompt encoder cancelled between layers");
      destroy_stream();
      return false;
    }
    std::future<BatchLoadResult> next_future;
    bool future_valid = false;
    if (layer + 1 < options.layer_count && options.prefetch_layers) {
      auto next_descriptors =
          LayerDescriptors(inventory, layer + 1, &descriptor_error);
      if (next_descriptors.empty()) {
        SetError(error, std::move(descriptor_error));
        destroy_stream();
        return false;
      }
      next_future =
          std::async(std::launch::async,
                     [&inventory, descriptors = std::move(next_descriptors),
                      device, cancellation]() mutable {
                       return LoadBatch(inventory, std::move(descriptors),
                                        device, cancellation);
                     });
      future_valid = true;
    }

    PromptEncoderLayerTelemetry layer_telemetry;
    layer_telemetry.layer = layer;
    layer_telemetry.weight_bytes = current->bytes;
    layer_telemetry.load_ms = current->load_ms;
    layer_telemetry.prefetch_wait_ms = current_wait_ms;
    telemetry->peak_registered_host_bytes = std::max(
        telemetry->peak_registered_host_bytes, current->peak_registered_bytes);
    telemetry->peak_device_bytes = std::max<std::uint64_t>(
        telemetry->peak_device_bytes,
        telemetry->activation_bytes + gemm.workspace_bytes() + current->bytes);
    if (!RunLayer(*current, layer, token_ids.size(), &gemm, &activations,
                  compute_stream, &layer_telemetry, error)) {
      if (future_valid) {
        (void)next_future.get();
      }
      destroy_stream();
      return false;
    }
    telemetry->total_dispatches += layer_telemetry.dispatches;
    telemetry->layers.push_back(layer_telemetry);

    if (layer + 1 >= options.layer_count) {
      break;
    }
    const auto wait_begin = Clock::now();
    BatchLoadResult next_result;
    if (future_valid) {
      next_result = next_future.get();
    } else {
      current.reset();
      auto next_descriptors =
          LayerDescriptors(inventory, layer + 1, &descriptor_error);
      if (next_descriptors.empty()) {
        SetError(error, std::move(descriptor_error));
        destroy_stream();
        return false;
      }
      next_result = LoadBatch(inventory, std::move(next_descriptors), device,
                              cancellation);
    }
    current_wait_ms = Milliseconds(wait_begin, Clock::now());
    if (!next_result.batch.has_value()) {
      SetError(error, std::move(next_result.error));
      destroy_stream();
      return false;
    }
    const std::uint64_t old_bytes = current.has_value() ? current->bytes : 0;
    const std::uint64_t next_bytes = next_result.batch->bytes;
    telemetry->peak_device_bytes = std::max<std::uint64_t>(
        telemetry->peak_device_bytes, telemetry->activation_bytes +
                                          gemm.workspace_bytes() + old_bytes +
                                          next_bytes);
    current = std::move(*next_result.batch);
  }

  output->tokens = token_ids.size();
  output->width = kPromptEncoderHiddenSize;
  output->values.resize(token_ids.size() * kPromptEncoderHiddenSize);
  status = hipMemcpy(output->values.data(), activations.hidden.data(),
                     output->values.size() * sizeof(std::uint16_t),
                     hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    *output = {};
    SetError(error, HipError("hipMemcpy(H3 prompt output)", status));
    destroy_stream();
    return false;
  }
  current.reset();
  destroy_stream();
  telemetry->live_registered_host_bytes_after =
      PromptEncoderLiveRegisteredHostBytes();
  if (telemetry->live_registered_host_bytes_after != 0) {
    *output = {};
    SetError(error, "MiniMax H3 prompt encoder leaked registered host pages");
    return false;
  }
  return true;
}

}  // namespace gufo::minimax_h3

#endif
