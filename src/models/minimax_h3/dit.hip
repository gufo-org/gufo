#if defined(ENGINE_ENABLE_HIP)

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "src/models/minimax_h3/attention.hip.hpp"
#include "src/models/minimax_h3/dit.hpp"
#include "src/models/minimax_h3/dit_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::size_t kMaximumAttentionRows = 131'072;
constexpr std::size_t kMaximumScalarAttentionRows = 16'384;
constexpr std::size_t kMaximumMaterializedAttentionRows = 4'096;
// Pinned by byte-repeat validation on gfx1151 with the repository's ROCm
// 7.2.3 toolchain. A changed rocBLAS database fails session construction.
constexpr rocblas_int kGfx1151RocblasBf16Solution = -1162085109;

rocblas_int PinnedBf16Solution(std::size_t batch, std::size_t rows,
                               std::size_t columns) {
  if (batch == 1872) {
    if (rows == 21504 && columns == 5376) {
      return -1162085131;
    }
    if (rows == 5376 && columns == 7168) {
      return -1162085139;
    }
    if (rows == 28672 && columns == 5376) {
      return -1162085137;
    }
    if (rows == 5376 && columns == 14336) {
      return -1162085127;
    }
  }
  // The exact 1344x768 / 124-frame layout has 37,710 non-text rows and at
  // most 4,096 prompt rows. These solutions were selected by a one-pass
  // exhaustive search and retained only when their output was byte-identical
  // to the generic deterministic solution on gfx1151 / ROCm 7.2.3.
  if (batch >= 37'711 && batch <= 41'806) {
    if (rows == 21504 && columns == 5376) {
      return -1162085131;
    }
    if (rows == 5376 && columns == 7168) {
      return -1162085134;
    }
    if (rows == 28672 && columns == 5376) {
      return -1162085118;
    }
    if (rows == 5376 && columns == 14336) {
      return -1162085122;
    }
  }
  return kGfx1151RocblasBf16Solution;
}

template<ck::index_t... Values>
using CkSequence = ck::Sequence<Values...>;

using CkPassThrough = ck::tensor_operation::element_wise::PassThrough;
using CkScale = ck::tensor_operation::element_wise::Scale;

using H3FusedAttentionShort = ck::tensor_operation::device::
    DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffle<
        2, 1, 1, 1, 1, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t,
        ck::Tuple<>, float, ck::Tuple<>, float, float, CkPassThrough,
        CkPassThrough, CkScale, CkPassThrough, CkPassThrough,
        ck::tensor_operation::device::GemmSpecialization::MNKOPadding,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default, 1, 128, 64,
        128, 80, 8, 8, 80, 64, 8, 16, 16, 16, 1, 8, 5, CkSequence<2, 64, 1>,
        CkSequence<1, 0, 2>, CkSequence<1, 0, 2>, 2, 8, 8, true,
        CkSequence<2, 64, 1>, CkSequence<1, 0, 2>, CkSequence<1, 0, 2>, 2, 8, 8,
        true, CkSequence<2, 8, 8>, CkSequence<0, 2, 1>, CkSequence<0, 2, 1>, 1,
        2, 1, false, 1, 1, CkSequence<1, 64, 1, 2>, 8,
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled>;

template<typename Operation>
auto MakeH3FusedAttentionArgument(const std::uint16_t* query,
                                  const std::uint16_t* key,
                                  const std::uint16_t* value,
                                  std::uint16_t* output, std::size_t sequence) {
  Operation operation;
  return operation.MakeArgument(
      reinterpret_cast<const ck::bhalf_t*>(query),
      reinterpret_cast<const ck::bhalf_t*>(key),
      reinterpret_cast<const ck::bhalf_t*>(value),
      reinterpret_cast<ck::bhalf_t*>(output),
      static_cast<ck::index_t>(sequence), static_cast<ck::index_t>(sequence),
      static_cast<ck::index_t>(kDitHeadDimension),
      static_cast<ck::index_t>(kDitHeadDimension), 1,
      static_cast<ck::index_t>(kDitAttentionHeads),
      1.0F / std::sqrt(static_cast<float>(kDitHeadDimension)), true, true);
}

bool SupportsH3FusedAttention(std::size_t sequence) {
  if (sequence == 0 || sequence > kMaximumAttentionRows) {
    return false;
  }
  if (sequence <= kMaximumMaterializedAttentionRows) {
    H3FusedAttentionShort operation;
    const auto argument = MakeH3FusedAttentionArgument<H3FusedAttentionShort>(
        nullptr, nullptr, nullptr, nullptr, sequence);
    return operation.IsSupportedArgument(argument);
  }
  return true;
}

template<typename Operation>
bool LaunchH3FusedAttention(const std::uint16_t* query,
                            const std::uint16_t* key,
                            const std::uint16_t* value, std::uint16_t* output,
                            std::size_t sequence, hipStream_t stream) {
  Operation operation;
  const auto argument = MakeH3FusedAttentionArgument<Operation>(
      query, key, value, output, sequence);
  if (!operation.IsSupportedArgument(argument)) {
    return false;
  }
  StreamConfig config;
  config.stream_id_ = stream;
  config.time_kernel_ = false;
  operation.MakeInvoker().Run(argument, config);
  return true;
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
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
             "MiniMax H3 DiT supports only integrated gfx1151 Strix Halo; "
             "detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  if (properties.sharedMemPerBlock <
      kMaximumScalarAttentionRows * sizeof(float)) {
    SetError(error, "gfx1151 reports insufficient shared memory for H3 SDPA");
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
        bytes_(std::exchange(other.bytes_, 0)),
        owned_(std::exchange(other.owned_, false)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
      owned_ = std::exchange(other.owned_, false);
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
    owned_ = true;
    return true;
  }

  void Alias(void* data, std::size_t bytes) noexcept {
    Reset();
    data_ = data;
    bytes_ = bytes;
    owned_ = false;
  }

  void Reset() noexcept {
    if (data_ != nullptr && owned_) {
      (void)hipFree(data_);
    }
    data_ = nullptr;
    bytes_ = 0;
    owned_ = false;
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
  bool owned_{false};
};

struct WorkspaceRegion {
  std::size_t offset{0};
  std::size_t bytes{0};
};

struct SharedWorkspaceLayout {
  WorkspaceRegion state;
  WorkspaceRegion row_map;
  WorkspaceRegion qkv_fc1;
  WorkspaceRegion activated;
  WorkspaceRegion query_modulation_mlp;
  WorkspaceRegion key_modulation_attention;
  WorkspaceRegion value_outputs;
  WorkspaceRegion normalization;
  WorkspaceRegion attention_scores;
  WorkspaceRegion attention_probabilities;
  std::size_t bytes{0};
};

SharedWorkspaceLayout MakeSharedWorkspaceLayout(std::size_t rows,
                                                bool fused_attention,
                                                bool row_parallel_attention,
                                                std::string* error) {
  constexpr unsigned __int128 kAlignment = 256;
  unsigned __int128 total = 0;
  bool overflow = false;
  const auto add = [&](unsigned __int128 bytes) {
    total = (total + kAlignment - 1U) & ~(kAlignment - 1U);
    const unsigned __int128 offset = total;
    total += bytes;
    if (offset > std::numeric_limits<std::size_t>::max() ||
        bytes > std::numeric_limits<std::size_t>::max() ||
        total > std::numeric_limits<std::size_t>::max()) {
      overflow = true;
      return WorkspaceRegion{};
    }
    return WorkspaceRegion{.offset = static_cast<std::size_t>(offset),
                           .bytes = static_cast<std::size_t>(bytes)};
  };
  const unsigned __int128 row_count = rows;
  const unsigned __int128 bf16 = sizeof(std::uint16_t);
  const unsigned __int128 f32 = sizeof(float);
  const unsigned __int128 hidden = row_count * kDitHiddenSize;
  const unsigned __int128 attention = row_count * kDitAttentionWidth;
  const unsigned __int128 activated = row_count * kDitFeedForwardSize;

  SharedWorkspaceLayout layout;
  layout.state = add(hidden * bf16);
  layout.row_map = add(row_count * sizeof(std::uint32_t));
  layout.qkv_fc1 = add(std::max(attention * 3U, activated * 2U) * bf16);
  layout.activated = add(activated * bf16);
  layout.query_modulation_mlp = add(attention * bf16);
  layout.key_modulation_attention = add(attention * bf16);
  layout.value_outputs = add(attention * bf16);
  layout.normalization = add(row_count * kDitAttentionHeads * 2U * f32);
  if (row_parallel_attention && !fused_attention) {
    const unsigned __int128 matrix = row_count * row_count * kDitAttentionHeads;
    layout.attention_scores = add(matrix * f32);
    layout.attention_probabilities = add(matrix * bf16);
  }
  if (overflow) {
    SetError(error, "MiniMax H3 shared DiT workspace size overflows");
    return {};
  }
  layout.bytes = static_cast<std::size_t>(total);
  return layout;
}

bool HasShape(const TensorDescriptor& descriptor,
              std::initializer_list<std::uint64_t> shape) {
  return descriptor.dtype == DType::kBFloat16 &&
         descriptor.shape ==
             std::vector<std::uint64_t>(shape.begin(), shape.end());
}

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error, "MiniMax H3 DiT tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (descriptor->phase != Phase::kDitCore || !HasShape(*descriptor, shape)) {
    SetError(error,
             "MiniMax H3 DiT tensor contract differs: " + std::string(name));
    return nullptr;
  }
  return descriptor;
}

struct LoadedWeight {
  const TensorDescriptor* descriptor{nullptr};
  DeviceBuffer storage;
};

bool LoadWeight(const ModelInventory& inventory,
                const TensorDescriptor& descriptor, int device,
                hipStream_t stream, void* staging, std::size_t staging_bytes,
                const CancellationToken* cancellation, LoadedWeight* output,
                std::string* error) {
  if (descriptor.payload_bytes > std::numeric_limits<std::size_t>::max()) {
    SetError(error, "MiniMax H3 DiT weight exceeds addressable memory");
    return false;
  }
  const auto bytes = static_cast<std::size_t>(descriptor.payload_bytes);
  if (!output->storage.Allocate(bytes, error)) {
    return false;
  }
  const std::filesystem::path path = inventory.model_root() / descriptor.shard;
  const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file < 0) {
    SetError(error, "cannot open MiniMax H3 DiT shard " + path.string());
    return false;
  }
  std::size_t copied = 0;
  bool success = true;
  while (copied < bytes) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 DiT weight load cancelled");
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
                 "short read loading MiniMax H3 DiT tensor " + descriptor.name);
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
      status = hipMemcpyAsync(
          static_cast<std::byte*>(output->storage.data()) + copied, staging,
          chunk, hipMemcpyHostToDevice, stream);
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(stream);
    }
    if (status != hipSuccess) {
      SetError(error, HipError("copy MiniMax H3 DiT weight", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  (void)close(file);
  if (!success) {
    output->storage.Reset();
    return false;
  }
  output->descriptor = &descriptor;
  return true;
}

class Bf16Gemm {
private:
  struct Plan {
    std::size_t batch;
    std::size_t rows;
    std::size_t columns;
    rocblas_int solution_index;
    bool selected;
  };

public:
  Bf16Gemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
    plans_.reserve(4);
  }
  ~Bf16Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  [[nodiscard]] std::size_t workspace_bytes() const noexcept { return 0; }

  bool Prepare(std::size_t batch, std::size_t rows, std::size_t columns,
               std::string* error) {
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        batch >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        rows >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        columns >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max())) {
      SetError(error, "invalid rocBLAS BF16 DiT matmul dimensions");
      return false;
    }
    if (FindPlan(batch, rows, columns) == nullptr) {
      plans_.push_back({batch, rows, columns,
                        PinnedBf16Solution(batch, rows, columns), false});
    }
    return true;
  }

  bool TrustPinnedSolution(std::size_t batch, std::size_t rows,
                           std::size_t columns, std::string* error) {
    if (!Prepare(batch, rows, columns, error)) {
      return false;
    }
    Plan* plan = FindPlan(batch, rows, columns);
    if (plan == nullptr) {
      SetError(error, "rocBLAS BF16 DiT plan lookup failed");
      return false;
    }
    plan->selected = true;
    return true;
  }

  bool SelectDeterministic(const std::uint16_t* weight,
                           const std::uint16_t* input, std::uint16_t* output,
                           std::uint16_t* reference,
                           unsigned long long* first_mismatch,
                           std::size_t batch, std::size_t rows,
                           std::size_t columns, hipStream_t stream,
                           std::string* error) {
    if (!Prepare(batch, rows, columns, error) ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "cannot prepare deterministic rocBLAS DiT solution");
      return false;
    }
    Plan* plan = FindPlan(batch, rows, columns);
    if (plan == nullptr) {
      SetError(error, "rocBLAS BF16 DiT plan lookup failed");
      return false;
    }
    const std::size_t output_elements = batch * rows;
    if (Execute(weight, input, output, batch, rows, columns,
                plan->solution_index) != rocblas_status_success) {
      SetError(error,
               "pinned gfx1151 rocBLAS BF16 DiT solution is unavailable");
      return false;
    }
    hipError_t hip_status = hipMemcpyAsync(
        reference, output, output_elements * sizeof(std::uint16_t),
        hipMemcpyDeviceToDevice, stream);
    if (hip_status == hipSuccess &&
        Execute(weight, input, output, batch, rows, columns,
                plan->solution_index) != rocblas_status_success) {
      hip_status = hipErrorUnknown;
    }
    if (hip_status == hipSuccess) {
      hip_status = hipMemsetAsync(first_mismatch, 0xFF,
                                  sizeof(unsigned long long), stream);
    }
    if (hip_status == hipSuccess) {
      dit_ops::LaunchFirstMismatch(reference, output, output_elements,
                                   first_mismatch, stream);
      hip_status = hipPeekAtLastError();
    }
    unsigned long long host_mismatch = 0;
    if (hip_status == hipSuccess) {
      hip_status =
          hipMemcpyAsync(&host_mismatch, first_mismatch, sizeof(host_mismatch),
                         hipMemcpyDeviceToHost, stream);
    }
    if (hip_status == hipSuccess) {
      hip_status = hipStreamSynchronize(stream);
    }
    if (hip_status != hipSuccess) {
      SetError(error,
               HipError("validate pinned rocBLAS DiT solution", hip_status));
      return false;
    }
    if (host_mismatch != std::numeric_limits<unsigned long long>::max()) {
      SetError(error,
               "pinned gfx1151 rocBLAS BF16 DiT solution is not deterministic");
      return false;
    }
    plan->selected = true;
    return true;
  }

  bool Run(const std::uint16_t* weight, const std::uint16_t* input,
           std::uint16_t* output, std::size_t batch, std::size_t rows,
           std::size_t columns, hipStream_t stream, std::string* error) {
    Plan* plan = FindPlan(batch, rows, columns);
    if (plan == nullptr || !plan->selected) {
      SetError(error, "rocBLAS BF16 DiT solution was not pinned");
      return false;
    }
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "cannot set rocBLAS DiT stream");
      return false;
    }
    const rocblas_status status = Execute(weight, input, output, batch, rows,
                                          columns, plan->solution_index);
    if (status != rocblas_status_success) {
      SetError(error, "rocBLAS BF16 DiT matmul dispatch failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  Plan* FindPlan(std::size_t batch, std::size_t rows, std::size_t columns) {
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
                         std::size_t columns,
                         rocblas_int solution_index) const {
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
        solution_index, 0);
  }

  rocblas_handle handle_{nullptr};
  std::vector<Plan> plans_;
};

class AttentionGemm {
public:
  AttentionGemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
  }
  ~AttentionGemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
    }
  }
  AttentionGemm(const AttentionGemm&) = delete;
  AttentionGemm& operator=(const AttentionGemm&) = delete;

  bool Scores(const std::uint16_t* query, const std::uint16_t* key,
              float* scores, std::size_t sequence, float scale,
              hipStream_t stream, std::string* error) {
    if (!SetStream(sequence, stream, error)) {
      return false;
    }
    constexpr float beta = 0.0F;
    const auto s = static_cast<rocblas_int>(sequence);
    const auto row_stride =
        static_cast<rocblas_int>(kDitAttentionHeads * kDitHeadDimension);
    const auto head_stride = static_cast<rocblas_stride>(kDitHeadDimension);
    const auto score_stride = static_cast<rocblas_stride>(sequence * sequence);
    return Check(
        rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_transpose, rocblas_operation_none, s, s,
            kDitHeadDimension, &scale, key, rocblas_datatype_bf16_r, row_stride,
            head_stride, query, rocblas_datatype_bf16_r, row_stride,
            head_stride, &beta, scores, rocblas_datatype_f32_r, s, score_stride,
            scores, rocblas_datatype_f32_r, s, score_stride, kDitAttentionHeads,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
        "score", error);
  }

  bool Values(const std::uint16_t* value, const std::uint16_t* probabilities,
              std::uint16_t* output, std::size_t sequence, hipStream_t stream,
              std::string* error) {
    if (!SetStream(sequence, stream, error)) {
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto s = static_cast<rocblas_int>(sequence);
    const auto row_stride =
        static_cast<rocblas_int>(kDitAttentionHeads * kDitHeadDimension);
    const auto head_stride = static_cast<rocblas_stride>(kDitHeadDimension);
    const auto probability_stride =
        static_cast<rocblas_stride>(sequence * sequence);
    return Check(
        rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_none, rocblas_operation_none,
            kDitHeadDimension, s, s, &alpha, value, rocblas_datatype_bf16_r,
            row_stride, head_stride, probabilities, rocblas_datatype_bf16_r, s,
            probability_stride, &beta, output, rocblas_datatype_bf16_r,
            row_stride, head_stride, output, rocblas_datatype_bf16_r,
            row_stride, head_stride, kDitAttentionHeads, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "value", error);
  }

private:
  bool SetStream(std::size_t sequence, hipStream_t stream, std::string* error) {
    if (handle_ == nullptr || sequence == 0 ||
        sequence > kMaximumMaterializedAttentionRows ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "invalid rocBLAS H3 attention GEMM");
      return false;
    }
    return true;
  }
  bool Check(rocblas_status status, std::string_view phase,
             std::string* error) {
    if (status == rocblas_status_success) {
      return true;
    }
    SetError(error,
             "rocBLAS H3 attention " + std::string(phase) +
                 " GEMM failed: " + std::to_string(static_cast<int>(status)));
    return false;
  }
  rocblas_handle handle_{nullptr};
};

bool CheckLaunch(std::string_view label, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(label, status));
  return false;
}

std::string FindFirstNonFinite(std::string_view name,
                               std::span<const std::uint16_t> values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::uint32_t bits = static_cast<std::uint32_t>(values[index]) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      return std::string(name) + "[" + std::to_string(index) +
             "]=" + std::to_string(value);
    }
  }
  return {};
}

}  // namespace

struct DitBlockWorkspace::Impl {
  std::size_t rows{0};
  bool row_parallel_attention{true};
  bool fused_attention{false};
  SharedWorkspaceLayout layout;
  DeviceBuffer storage;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer silu_lookup;
  std::mutex rope_mutex;
  std::vector<std::uint16_t> rope_cos_host;
  std::vector<std::uint16_t> rope_sin_host;
  bool rope_prepared{false};
  std::atomic_bool accounting_claimed{false};
};

DitBlockWorkspace::DitBlockWorkspace(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DitBlockWorkspace::~DitBlockWorkspace() = default;

std::shared_ptr<DitBlockWorkspace> DitBlockWorkspace::Create(
    const DitBlockWorkspaceOptions& options, std::string* error) {
  if (options.rows == 0 || options.rows > kMaximumAttentionRows ||
      options.rows > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "invalid MiniMax H3 shared DiT workspace options");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->rows = options.rows;
  impl->row_parallel_attention = options.row_parallel_attention;
  int device = 0;
  if (!ValidateTarget(&device, error)) {
    return nullptr;
  }
  impl->fused_attention =
      options.row_parallel_attention && SupportsH3FusedAttention(options.rows);
  if (!options.row_parallel_attention &&
      options.rows > kMaximumScalarAttentionRows) {
    SetError(error, "MiniMax H3 scalar attention row limit exceeded");
    return nullptr;
  }
  impl->layout =
      MakeSharedWorkspaceLayout(options.rows, impl->fused_attention,
                                options.row_parallel_attention, error);
  const std::size_t rope_bytes =
      options.rows * kDitRopeHalf * sizeof(std::uint16_t);
  constexpr std::size_t kBf16SiluLookupBytes = (1ULL << 16U) * sizeof(float);
  if (impl->layout.bytes == 0 ||
      !impl->storage.Allocate(impl->layout.bytes, error) ||
      !impl->rope_cos.Allocate(rope_bytes, error) ||
      !impl->rope_sin.Allocate(rope_bytes, error) ||
      (options.rows >= 32'768 &&
       !impl->silu_lookup.Allocate(kBf16SiluLookupBytes, error))) {
    return nullptr;
  }
  if (impl->silu_lookup.data() != nullptr) {
    hipStream_t lookup_stream = nullptr;
    hipError_t status =
        hipStreamCreateWithFlags(&lookup_stream, hipStreamNonBlocking);
    if (status == hipSuccess) {
      dit_ops::LaunchBf16SiluLookup(impl->silu_lookup.As<float>(),
                                    lookup_stream);
      status = hipPeekAtLastError();
    }
    if (status == hipSuccess) {
      status = hipStreamSynchronize(lookup_stream);
    }
    if (lookup_stream != nullptr) {
      const hipError_t destroy_status = hipStreamDestroy(lookup_stream);
      if (status == hipSuccess) {
        status = destroy_status;
      }
    }
    if (status != hipSuccess) {
      SetError(error, HipError("initialize H3 BF16 SiLU lookup", status));
      return nullptr;
    }
  }
  return std::shared_ptr<DitBlockWorkspace>(
      new DitBlockWorkspace(std::move(impl)));
}

std::size_t DitBlockWorkspace::rows() const noexcept {
  return impl_ == nullptr ? 0 : impl_->rows;
}

std::uint64_t DitBlockWorkspace::bytes() const noexcept {
  return impl_ == nullptr
             ? 0
             : impl_->storage.bytes() + impl_->rope_cos.bytes() +
                   impl_->rope_sin.bytes() + impl_->silu_lookup.bytes();
}

struct DitBlockSession::Impl {
  std::size_t block_index{0};
  std::size_t rows{0};
  std::size_t modulation_rows{0};
  bool row_parallel_attention{true};
  bool fused_attention{false};
  int device{0};
  hipStream_t stream{nullptr};
  hipEvent_t start_event{nullptr};
  hipEvent_t stop_event{nullptr};
  void* staging{nullptr};
  Bf16Gemm gemm;
  AttentionGemm attention_gemm;
  std::shared_ptr<DitBlockWorkspace> workspace;

  LoadedWeight norm1;
  LoadedWeight norm2;
  LoadedWeight qkv_weight;
  LoadedWeight query_norm;
  LoadedWeight key_norm;
  LoadedWeight output_weight;
  LoadedWeight fc1_weight;
  LoadedWeight fc2_weight;

  DeviceBuffer hidden;
  DeviceBuffer modulation;
  DeviceBuffer row_map;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer modulation_attention;
  DeviceBuffer qkv;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer normalization_factors;
  DeviceBuffer attention_heads;
  DeviceBuffer owned_attention_scores;
  DeviceBuffer owned_attention_probabilities;
  float* attention_scores{nullptr};
  std::uint16_t* attention_probabilities{nullptr};
  std::size_t attention_workspace_bytes{0};
  DeviceBuffer attention_output;
  DeviceBuffer gated_attention;
  DeviceBuffer modulation_mlp;
  DeviceBuffer fc1;
  DeviceBuffer activated;
  DeviceBuffer mlp_output;
  DeviceBuffer block_output;
  DeviceBuffer gemm_reference;
  DeviceBuffer gemm_first_mismatch;
  const float* silu_lookup{nullptr};

  std::uint64_t weight_bytes{0};
  std::uint64_t activation_bytes{0};
  std::uint64_t gemm_plan_validations{0};
  double load_ms{0.0};
  bool constants_prepared{false};

  ~Impl() {
    if (staging != nullptr) {
      (void)hipHostFree(staging);
    }
    if (start_event != nullptr) {
      (void)hipEventDestroy(start_event);
    }
    if (stop_event != nullptr) {
      (void)hipEventDestroy(stop_event);
    }
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }
};

DitBlockSession::DitBlockSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DitBlockSession::~DitBlockSession() = default;
DitBlockSession::DitBlockSession(DitBlockSession&&) noexcept = default;
DitBlockSession& DitBlockSession::operator=(DitBlockSession&&) noexcept =
    default;

std::unique_ptr<DitBlockSession> DitBlockSession::Create(
    const ModelInventory& inventory, const DitBlockOptions& options,
    const CancellationToken* cancellation, std::string* error) {
  return Create(inventory, options, nullptr, cancellation, error);
}

std::unique_ptr<DitBlockSession> DitBlockSession::Create(
    const ModelInventory& inventory, const DitBlockOptions& options,
    std::shared_ptr<DitBlockWorkspace> workspace,
    const CancellationToken* cancellation, std::string* error) {
  if (options.block_index >= kDitBlocks || options.rows == 0 ||
      options.rows > kMaximumAttentionRows ||
      options.rows > std::numeric_limits<std::uint32_t>::max() ||
      options.modulation_rows == 0 ||
      options.modulation_rows > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "invalid MiniMax H3 DiT block options");
    return nullptr;
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT session creation cancelled");
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->block_index = options.block_index;
  impl->rows = options.rows;
  impl->modulation_rows = options.modulation_rows;
  impl->row_parallel_attention = options.row_parallel_attention;
  if (!ValidateTarget(&impl->device, error)) {
    return nullptr;
  }
  impl->fused_attention =
      options.row_parallel_attention && SupportsH3FusedAttention(options.rows);
  if (!options.row_parallel_attention &&
      options.rows > kMaximumScalarAttentionRows) {
    SetError(error, "MiniMax H3 scalar attention row limit exceeded");
    return nullptr;
  }
  if (workspace != nullptr) {
    const DitBlockWorkspace::Impl* workspace_impl = workspace->impl_.get();
    if (workspace_impl == nullptr || workspace_impl->rows != options.rows ||
        workspace_impl->row_parallel_attention !=
            options.row_parallel_attention ||
        workspace_impl->fused_attention != impl->fused_attention) {
      SetError(error, "MiniMax H3 shared DiT workspace contract differs");
      return nullptr;
    }
    impl->workspace = std::move(workspace);
  } else {
    impl->workspace = DitBlockWorkspace::Create(
        {.rows = options.rows,
         .row_parallel_attention = options.row_parallel_attention},
        error);
    if (impl->workspace == nullptr) {
      return nullptr;
    }
  }
  hipError_t status =
      hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking);
  if (status == hipSuccess) {
    status = hipEventCreate(&impl->start_event);
  }
  if (status == hipSuccess) {
    status = hipEventCreate(&impl->stop_event);
  }
  if (status == hipSuccess) {
    status = hipHostMalloc(&impl->staging, kWeightStagingBytes,
                           hipHostMallocDefault);
  }
  if (status != hipSuccess) {
    SetError(error, HipError("create MiniMax H3 DiT resources", status));
    return nullptr;
  }

  const std::string prefix =
      "blocks." + std::to_string(options.block_index) + ".";
  const TensorDescriptor* norm1 = RequireTensor(
      inventory, prefix + "norm1.weight", {kDitHiddenSize}, error);
  const TensorDescriptor* norm2 = RequireTensor(
      inventory, prefix + "norm2.weight", {kDitHiddenSize}, error);
  const TensorDescriptor* qkv_weight =
      RequireTensor(inventory, prefix + "attn.qkv_proj.weight",
                    {kDitAttentionWidth * 3U, kDitHiddenSize}, error);
  const TensorDescriptor* query_norm = RequireTensor(
      inventory, prefix + "attn.q_norm.weight", {kDitHeadDimension}, error);
  const TensorDescriptor* key_norm = RequireTensor(
      inventory, prefix + "attn.k_norm.weight", {kDitHeadDimension}, error);
  const TensorDescriptor* output_weight =
      RequireTensor(inventory, prefix + "attn.out_proj.weight",
                    {kDitHiddenSize, kDitAttentionWidth}, error);
  const TensorDescriptor* fc1_weight =
      RequireTensor(inventory, prefix + "mlp.fc1.weight",
                    {kDitFeedForwardSize * 2U, kDitHiddenSize}, error);
  const TensorDescriptor* fc2_weight =
      RequireTensor(inventory, prefix + "mlp.fc2.weight",
                    {kDitHiddenSize, kDitFeedForwardSize}, error);
  if (norm1 == nullptr || norm2 == nullptr || qkv_weight == nullptr ||
      query_norm == nullptr || key_norm == nullptr ||
      output_weight == nullptr || fc1_weight == nullptr ||
      fc2_weight == nullptr) {
    return nullptr;
  }

  const auto load_begin = Clock::now();
  const std::array<std::pair<const TensorDescriptor*, LoadedWeight*>, 8>
      weights = {{
          {norm1, &impl->norm1},
          {norm2, &impl->norm2},
          {qkv_weight, &impl->qkv_weight},
          {query_norm, &impl->query_norm},
          {key_norm, &impl->key_norm},
          {output_weight, &impl->output_weight},
          {fc1_weight, &impl->fc1_weight},
          {fc2_weight, &impl->fc2_weight},
      }};
  for (const auto& [descriptor, weight] : weights) {
    if (!LoadWeight(inventory, *descriptor, impl->device, impl->stream,
                    impl->staging, kWeightStagingBytes, cancellation, weight,
                    error)) {
      return nullptr;
    }
    impl->weight_bytes += descriptor->payload_bytes;
  }
  (void)hipHostFree(impl->staging);
  impl->staging = nullptr;
  impl->load_ms = Milliseconds(load_begin, Clock::now());

  const auto bf16_bytes = [](std::size_t elements) {
    return elements * sizeof(std::uint16_t);
  };
  const std::size_t hidden_elements = options.rows * kDitHiddenSize;
  const std::size_t attention_elements = options.rows * kDitAttentionWidth;
  const std::size_t qkv_elements = attention_elements * 3U;
  const std::size_t normalization_factor_elements =
      options.rows * kDitAttentionHeads * 2U;
  const std::size_t fc1_elements = options.rows * kDitFeedForwardSize * 2U;
  const std::size_t activated_elements = options.rows * kDitFeedForwardSize;
  const std::size_t modulation_elements =
      options.modulation_rows * kDitModulationSlots * kDitHiddenSize;
  const std::size_t rope_elements = options.rows * kDitRopeHalf;
  const std::size_t attention_matrix_elements =
      options.rows * options.rows * kDitAttentionHeads;
  const auto allocate_private = [&](DeviceBuffer* buffer, std::size_t bytes) {
    if (!buffer->Allocate(bytes, error)) {
      return false;
    }
    impl->activation_bytes += bytes;
    return true;
  };
  DitBlockWorkspace::Impl* workspace_impl = impl->workspace->impl_.get();
  impl->silu_lookup = workspace_impl->silu_lookup.As<float>();
  const auto alias_region = [&](DeviceBuffer* buffer,
                                const WorkspaceRegion& region,
                                std::size_t bytes) {
    if (workspace_impl == nullptr ||
        region.offset > workspace_impl->storage.bytes() ||
        region.bytes > workspace_impl->storage.bytes() - region.offset ||
        bytes > region.bytes) {
      SetError(error, "MiniMax H3 shared DiT workspace is undersized");
      return false;
    }
    buffer->Alias(
        static_cast<std::byte*>(workspace_impl->storage.data()) + region.offset,
        bytes);
    return true;
  };
  const SharedWorkspaceLayout& layout = workspace_impl->layout;
  if (!alias_region(&impl->hidden, layout.state, bf16_bytes(hidden_elements)) ||
      !allocate_private(&impl->modulation, bf16_bytes(modulation_elements)) ||
      !alias_region(&impl->row_map, layout.row_map,
                    options.rows * sizeof(std::uint32_t)) ||
      !alias_region(&impl->modulation_attention,
                    layout.key_modulation_attention,
                    bf16_bytes(hidden_elements)) ||
      !alias_region(&impl->qkv, layout.qkv_fc1, bf16_bytes(qkv_elements)) ||
      !alias_region(&impl->query, layout.query_modulation_mlp,
                    bf16_bytes(attention_elements)) ||
      !alias_region(&impl->key, layout.key_modulation_attention,
                    bf16_bytes(attention_elements)) ||
      !alias_region(&impl->value, layout.value_outputs,
                    bf16_bytes(attention_elements)) ||
      !alias_region(&impl->normalization_factors, layout.normalization,
                    normalization_factor_elements * sizeof(float)) ||
      !alias_region(&impl->attention_heads, layout.qkv_fc1,
                    bf16_bytes(attention_elements)) ||
      !alias_region(&impl->attention_output, layout.value_outputs,
                    bf16_bytes(hidden_elements)) ||
      !alias_region(&impl->gated_attention, layout.state,
                    bf16_bytes(hidden_elements)) ||
      !alias_region(&impl->modulation_mlp, layout.query_modulation_mlp,
                    bf16_bytes(hidden_elements)) ||
      !alias_region(&impl->fc1, layout.qkv_fc1, bf16_bytes(fc1_elements)) ||
      !alias_region(&impl->activated, layout.activated,
                    bf16_bytes(activated_elements)) ||
      !alias_region(&impl->mlp_output, layout.value_outputs,
                    bf16_bytes(hidden_elements)) ||
      !alias_region(&impl->block_output, layout.state,
                    bf16_bytes(hidden_elements))) {
    return nullptr;
  }
  impl->rope_cos.Alias(workspace_impl->rope_cos.data(),
                       bf16_bytes(rope_elements));
  impl->rope_sin.Alias(workspace_impl->rope_sin.data(),
                       bf16_bytes(rope_elements));
  if (options.validate_gemm_plans &&
      (!allocate_private(&impl->gemm_reference, bf16_bytes(fc1_elements)) ||
       !allocate_private(&impl->gemm_first_mismatch,
                         sizeof(unsigned long long)))) {
    return nullptr;
  }
  if (options.row_parallel_attention && !impl->fused_attention) {
    if (options.rows > kMaximumMaterializedAttentionRows) {
      SetError(error, "MiniMax H3 batched attention row limit exceeded");
      return nullptr;
    }
    const std::size_t score_bytes = attention_matrix_elements * sizeof(float);
    const std::size_t probability_bytes =
        attention_matrix_elements * sizeof(std::uint16_t);
    if (!alias_region(&impl->owned_attention_scores, layout.attention_scores,
                      score_bytes) ||
        !alias_region(&impl->owned_attention_probabilities,
                      layout.attention_probabilities, probability_bytes)) {
      return nullptr;
    }
    impl->attention_scores = impl->owned_attention_scores.As<float>();
    impl->attention_probabilities =
        impl->owned_attention_probabilities.As<std::uint16_t>();
    impl->attention_workspace_bytes = score_bytes + probability_bytes;
  }
  if (!workspace_impl->accounting_claimed.exchange(true,
                                                   std::memory_order_acq_rel)) {
    impl->activation_bytes += impl->workspace->bytes();
  }
  if (!impl->gemm.Prepare(options.rows, kDitAttentionWidth * 3U, kDitHiddenSize,
                          error) ||
      !impl->gemm.Prepare(options.rows, kDitHiddenSize, kDitAttentionWidth,
                          error) ||
      !impl->gemm.Prepare(options.rows, kDitFeedForwardSize * 2U,
                          kDitHiddenSize, error) ||
      !impl->gemm.Prepare(options.rows, kDitHiddenSize, kDitFeedForwardSize,
                          error)) {
    return nullptr;
  }
  if (options.validate_gemm_plans) {
    // Validate each fixed orientation once. Denoiser construction performs
    // this on its first selected block and reuses the pinned gfx1151 contract
    // for every remaining block with the same shapes.
    const auto fill = [&](DeviceBuffer* buffer, std::uint32_t seed) {
      dit_ops::LaunchFillBf16Pattern(buffer->As<std::uint16_t>(),
                                     buffer->bytes() / sizeof(std::uint16_t),
                                     seed, impl->stream);
      return hipPeekAtLastError() == hipSuccess;
    };
    if (!fill(&impl->hidden, 0x169001U) ||
        !fill(&impl->attention_heads, 0x169002U) ||
        !fill(&impl->modulation_mlp, 0x169003U) ||
        !fill(&impl->activated, 0x169004U) ||
        !impl->gemm.SelectDeterministic(
            impl->qkv_weight.storage.As<std::uint16_t>(),
            impl->hidden.As<std::uint16_t>(), impl->qkv.As<std::uint16_t>(),
            impl->gemm_reference.As<std::uint16_t>(),
            impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
            kDitAttentionWidth * 3U, kDitHiddenSize, impl->stream, error) ||
        !impl->gemm.SelectDeterministic(
            impl->output_weight.storage.As<std::uint16_t>(),
            impl->attention_heads.As<std::uint16_t>(),
            impl->attention_output.As<std::uint16_t>(),
            impl->gemm_reference.As<std::uint16_t>(),
            impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
            kDitHiddenSize, kDitAttentionWidth, impl->stream, error) ||
        !impl->gemm.SelectDeterministic(
            impl->fc1_weight.storage.As<std::uint16_t>(),
            impl->modulation_mlp.As<std::uint16_t>(),
            impl->fc1.As<std::uint16_t>(),
            impl->gemm_reference.As<std::uint16_t>(),
            impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
            kDitFeedForwardSize * 2U, kDitHiddenSize, impl->stream, error) ||
        !impl->gemm.SelectDeterministic(
            impl->fc2_weight.storage.As<std::uint16_t>(),
            impl->activated.As<std::uint16_t>(),
            impl->mlp_output.As<std::uint16_t>(),
            impl->gemm_reference.As<std::uint16_t>(),
            impl->gemm_first_mismatch.As<unsigned long long>(), options.rows,
            kDitHiddenSize, kDitFeedForwardSize, impl->stream, error)) {
      return nullptr;
    }
    status = hipStreamSynchronize(impl->stream);
    if (status != hipSuccess) {
      SetError(error, HipError("validate MiniMax H3 DiT GEMMs", status));
      return nullptr;
    }
    impl->gemm_plan_validations = 4;
    impl->activation_bytes -=
        impl->gemm_reference.bytes() + impl->gemm_first_mismatch.bytes();
    impl->gemm_reference.Reset();
    impl->gemm_first_mismatch.Reset();
  } else if (!impl->gemm.TrustPinnedSolution(options.rows,
                                             kDitAttentionWidth * 3U,
                                             kDitHiddenSize, error) ||
             !impl->gemm.TrustPinnedSolution(options.rows, kDitHiddenSize,
                                             kDitAttentionWidth, error) ||
             !impl->gemm.TrustPinnedSolution(options.rows,
                                             kDitFeedForwardSize * 2U,
                                             kDitHiddenSize, error) ||
             !impl->gemm.TrustPinnedSolution(options.rows, kDitHiddenSize,
                                             kDitFeedForwardSize, error)) {
    return nullptr;
  }
  return std::unique_ptr<DitBlockSession>(new DitBlockSession(std::move(impl)));
}

bool DitBlockSession::PrepareConstants(
    std::span<const std::uint16_t> modulation,
    std::span<const std::uint16_t> rope_cos,
    std::span<const std::uint16_t> rope_sin,
    const CancellationToken* cancellation, std::string* error) {
  if (impl_ == nullptr) {
    SetError(error, "invalid MiniMax H3 DiT constant preparation");
    return false;
  }
  const std::size_t modulation_elements =
      impl_->modulation_rows * kDitModulationSlots * kDitHiddenSize;
  const std::size_t rope_elements = impl_->rows * kDitRopeHalf;
  if (modulation.size() != modulation_elements ||
      rope_cos.size() != rope_elements || rope_sin.size() != rope_elements) {
    SetError(error, "MiniMax H3 DiT constant tensor size differs from session");
    return false;
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT constant preparation cancelled");
    return false;
  }
  const auto copy = [&](void* target, const void* source, std::size_t bytes,
                        std::string_view label) {
    const hipError_t status = hipMemcpyAsync(
        target, source, bytes, hipMemcpyHostToDevice, impl_->stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  };
  if (!copy(impl_->modulation.data(), modulation.data(),
            modulation.size_bytes(), "copy H3 DiT modulation constant")) {
    return false;
  }
  DitBlockWorkspace::Impl* workspace = impl_->workspace->impl_.get();
  if (workspace == nullptr) {
    SetError(error, "invalid MiniMax H3 shared RoPE workspace");
    return false;
  }
  std::scoped_lock rope_lock(workspace->rope_mutex);
  if (workspace->rope_prepared) {
    if (!std::ranges::equal(workspace->rope_cos_host, rope_cos) ||
        !std::ranges::equal(workspace->rope_sin_host, rope_sin)) {
      SetError(error, "MiniMax H3 shared RoPE constants differ");
      return false;
    }
  } else {
    workspace->rope_cos_host.assign(rope_cos.begin(), rope_cos.end());
    workspace->rope_sin_host.assign(rope_sin.begin(), rope_sin.end());
    if (!copy(workspace->rope_cos.data(), rope_cos.data(),
              rope_cos.size_bytes(), "copy H3 DiT RoPE cosine constant") ||
        !copy(workspace->rope_sin.data(), rope_sin.data(),
              rope_sin.size_bytes(), "copy H3 DiT RoPE sine constant")) {
      return false;
    }
  }
  const hipError_t status = hipStreamSynchronize(impl_->stream);
  if (status != hipSuccess) {
    SetError(error, HipError("synchronize H3 DiT constants", status));
    return false;
  }
  workspace->rope_prepared = true;
  impl_->constants_prepared = true;
  return true;
}

bool DitBlockSession::RunPrepared(const DitPreparedInput& input,
                                  const CancellationToken* cancellation,
                                  std::span<std::uint16_t> output,
                                  DitBlockTelemetry* telemetry,
                                  std::string* error) {
  return RunInternal(input, cancellation, {}, {}, {}, output, telemetry, error);
}

bool DitBlockSession::RunPreparedDevice(
    const std::uint16_t* hidden_device, const std::uint32_t* row_map_device,
    std::uint16_t* output_device, void* stream, bool input_rms_prepared,
    bool output_rms_required, const CancellationToken* cancellation,
    DitBlockTelemetry* telemetry, std::string* error) {
  if (hidden_device == nullptr || row_map_device == nullptr ||
      output_device == nullptr || stream == nullptr) {
    SetError(error, "invalid MiniMax H3 DiT device run arguments");
    return false;
  }
  return RunInternal({}, cancellation, {}, {}, {}, {}, telemetry, error,
                     hidden_device, row_map_device, output_device, stream,
                     input_rms_prepared, output_rms_required);
}

bool DitBlockSession::Run(const DitBlockInput& input,
                          const CancellationToken* cancellation,
                          DitBlockRetained* retained,
                          DitBlockTelemetry* telemetry, std::string* error) {
  if (impl_ == nullptr || retained == nullptr || telemetry == nullptr) {
    SetError(error, "invalid MiniMax H3 DiT run arguments");
    return false;
  }
  const std::size_t hidden_elements = impl_->rows * kDitHiddenSize;
  if (input.hidden.size() != hidden_elements ||
      input.row_map.size() != impl_->rows ||
      !PrepareConstants(input.modulation, input.rope_cos, input.rope_sin,
                        cancellation, error)) {
    if (error != nullptr && error->empty()) {
      *error = "MiniMax H3 DiT input tensor size differs from session";
    }
    return false;
  }
  retained->rows = impl_->rows;
  retained->width = kDitHiddenSize;
  retained->modulation_attention.resize(hidden_elements);
  retained->attention_output.resize(hidden_elements);
  retained->modulation_mlp.resize(hidden_elements);
  retained->block_output.resize(hidden_elements);
  return RunInternal({input.hidden, input.row_map}, cancellation,
                     retained->modulation_attention, retained->attention_output,
                     retained->modulation_mlp, retained->block_output,
                     telemetry, error);
}

bool DitBlockSession::RunInternal(
    const DitPreparedInput& input, const CancellationToken* cancellation,
    std::span<std::uint16_t> modulation_attention,
    std::span<std::uint16_t> attention_output,
    std::span<std::uint16_t> modulation_mlp,
    std::span<std::uint16_t> block_output, DitBlockTelemetry* telemetry,
    std::string* error, const std::uint16_t* hidden_device,
    const std::uint32_t* row_map_device, std::uint16_t* output_device,
    void* external_stream, bool input_rms_prepared, bool output_rms_required) {
  if (impl_ == nullptr || telemetry == nullptr || !impl_->constants_prepared) {
    SetError(error, "MiniMax H3 DiT constants are not prepared");
    return false;
  }
  const bool device_run = external_stream != nullptr;
  const std::size_t hidden_elements = impl_->rows * kDitHiddenSize;
  const auto valid_optional = [&](std::span<std::uint16_t> values) {
    return values.empty() || values.size() == hidden_elements;
  };
  if ((!device_run && (input.hidden.size() != hidden_elements ||
                       input.row_map.size() != impl_->rows ||
                       block_output.size() != hidden_elements)) ||
      (device_run && (hidden_device == nullptr || row_map_device == nullptr ||
                      output_device == nullptr)) ||
      !valid_optional(modulation_attention) ||
      !valid_optional(attention_output) || !valid_optional(modulation_mlp) ||
      (device_run &&
       (!modulation_attention.empty() || !attention_output.empty() ||
        !modulation_mlp.empty() || !block_output.empty()))) {
    SetError(error, "MiniMax H3 DiT prepared input size differs from session");
    return false;
  }
  if (!device_run) {
    for (const std::uint32_t row : input.row_map) {
      if (row >= impl_->modulation_rows) {
        SetError(error, "MiniMax H3 DiT row map exceeds modulation rows");
        return false;
      }
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT block cancelled before submission");
    return false;
  }

  *telemetry = {};
  telemetry->weight_bytes = impl_->weight_bytes;
  telemetry->activation_bytes = impl_->activation_bytes;
  telemetry->gemm_workspace_bytes =
      impl_->gemm.workspace_bytes() + impl_->attention_workspace_bytes;
  telemetry->load_ms = impl_->load_ms;
  telemetry->scratch_address =
      reinterpret_cast<std::uintptr_t>(impl_->qkv.data());

  const auto copy = [&](void* target, const void* source, std::size_t bytes,
                        hipMemcpyKind kind, std::string_view label) {
    const hipError_t status =
        hipMemcpyAsync(target, source, bytes, kind, impl_->stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  };
  if (!device_run) {
    if (!copy(impl_->hidden.data(), input.hidden.data(),
              input.hidden.size_bytes(), hipMemcpyHostToDevice,
              "copy H3 DiT hidden") ||
        !copy(impl_->row_map.data(), input.row_map.data(),
              input.row_map.size_bytes(), hipMemcpyHostToDevice,
              "copy H3 DiT row map")) {
      return false;
    }
  }

  const auto rows = static_cast<std::uint32_t>(impl_->rows);
  const hipStream_t run_stream =
      device_run ? static_cast<hipStream_t>(external_stream) : impl_->stream;
  const auto snapshot = [&](std::span<std::uint16_t> target,
                            const DeviceBuffer& source,
                            std::string_view label) {
    if (target.empty()) {
      return true;
    }
    const hipError_t status =
        hipMemcpyAsync(target.data(), source.data(), target.size_bytes(),
                       hipMemcpyDeviceToHost, run_stream);
    if (status != hipSuccess) {
      SetError(error, HipError(label, status));
      return false;
    }
    return true;
  };
  const std::uint16_t* hidden =
      device_run ? hidden_device : impl_->hidden.As<std::uint16_t>();
  const std::uint32_t* row_map =
      device_run ? row_map_device : impl_->row_map.As<std::uint32_t>();
  std::uint16_t* block_target =
      device_run ? output_device : impl_->block_output.As<std::uint16_t>();
  if (!device_run &&
      hipEventRecord(impl_->start_event, run_stream) != hipSuccess) {
    SetError(error, "cannot record MiniMax H3 DiT start event");
    return false;
  }
  if (input_rms_prepared) {
    dit_ops::LaunchAdaLnApplyPrepared(
        hidden, impl_->norm1.storage.As<std::uint16_t>(),
        impl_->modulation.As<std::uint16_t>(), row_map,
        impl_->normalization_factors.As<float>(),
        impl_->modulation_attention.As<std::uint16_t>(), rows, kDitHiddenSize,
        kDitModulationSlots, 0, 1, run_stream);
  } else {
    dit_ops::LaunchAdaLnSplit(hidden, impl_->norm1.storage.As<std::uint16_t>(),
                              impl_->modulation.As<std::uint16_t>(), row_map,
                              impl_->normalization_factors.As<float>(),
                              impl_->modulation_attention.As<std::uint16_t>(),
                              rows, kDitHiddenSize, kDitModulationSlots, 0, 1,
                              kDitNormEpsilon, run_stream);
  }
  if (!CheckLaunch("H3 DiT attention AdaLN", error) ||
      !snapshot(modulation_attention, impl_->modulation_attention,
                "snapshot H3 DiT attention modulation") ||
      !impl_->gemm.Run(impl_->qkv_weight.storage.As<std::uint16_t>(),
                       impl_->modulation_attention.As<std::uint16_t>(),
                       impl_->qkv.As<std::uint16_t>(), impl_->rows,
                       kDitAttentionWidth * 3U, kDitHiddenSize, run_stream,
                       error)) {
    return false;
  }
  dit_ops::LaunchGroupedQkvNormRopeSplit(
      impl_->qkv.As<std::uint16_t>(),
      impl_->query_norm.storage.As<std::uint16_t>(),
      impl_->key_norm.storage.As<std::uint16_t>(),
      impl_->rope_cos.As<std::uint16_t>(), impl_->rope_sin.As<std::uint16_t>(),
      impl_->normalization_factors.As<float>(),
      impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
      impl_->value.As<std::uint16_t>(), rows, kDitAttentionHeads,
      kDitNormEpsilon, impl_->rows > kMaximumMaterializedAttentionRows,
      run_stream);
  if (!CheckLaunch("H3 DiT grouped QKV norm/RoPE", error)) {
    return false;
  }
  if (impl_->fused_attention) {
    bool launched = false;
    if (impl_->rows <= kMaximumMaterializedAttentionRows) {
      launched = LaunchH3FusedAttention<H3FusedAttentionShort>(
          impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
          impl_->value.As<std::uint16_t>(),
          impl_->attention_heads.As<std::uint16_t>(), impl_->rows, run_stream);
    } else {
      dit_ops::LaunchWmmaAttention(impl_->query.As<std::uint16_t>(),
                                   impl_->key.As<std::uint16_t>(),
                                   impl_->value.As<std::uint16_t>(),
                                   impl_->attention_heads.As<std::uint16_t>(),
                                   rows, kDitAttentionHeads, run_stream);
      launched = true;
    }
    if (!launched) {
      SetError(error, "MiniMax H3 fused attention rejected the prepared shape");
      return false;
    }
  } else if (impl_->row_parallel_attention) {
    if (!impl_->attention_gemm.Scores(
            impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
            impl_->attention_scores, impl_->rows,
            1.0F / std::sqrt(static_cast<float>(kDitHeadDimension)), run_stream,
            error)) {
      return false;
    }
    dit_ops::LaunchSoftmaxScoresToBf16(impl_->attention_scores,
                                       impl_->attention_probabilities, rows,
                                       kDitAttentionHeads, run_stream);
    if (!CheckLaunch("H3 DiT attention softmax", error) ||
        !impl_->attention_gemm.Values(
            impl_->value.As<std::uint16_t>(), impl_->attention_probabilities,
            impl_->attention_heads.As<std::uint16_t>(), impl_->rows, run_stream,
            error)) {
      return false;
    }
  } else {
    dit_ops::LaunchFullAttention(
        impl_->query.As<std::uint16_t>(), impl_->key.As<std::uint16_t>(),
        impl_->value.As<std::uint16_t>(),
        impl_->attention_heads.As<std::uint16_t>(), rows, kDitAttentionHeads,
        kDitHeadDimension,
        1.0F / std::sqrt(static_cast<float>(kDitHeadDimension)), run_stream,
        false);
  }
  if (!CheckLaunch("H3 DiT full SDPA", error) ||
      !impl_->gemm.Run(impl_->output_weight.storage.As<std::uint16_t>(),
                       impl_->attention_heads.As<std::uint16_t>(),
                       impl_->attention_output.As<std::uint16_t>(), impl_->rows,
                       kDitHiddenSize, kDitAttentionWidth, run_stream, error)) {
    return false;
  }
  if (!snapshot(attention_output, impl_->attention_output,
                "snapshot H3 DiT attention output")) {
    return false;
  }
  dit_ops::LaunchGateRmsInverse(
      hidden, impl_->attention_output.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(), row_map,
      impl_->gated_attention.As<std::uint16_t>(),
      impl_->normalization_factors.As<float>(), rows, kDitHiddenSize,
      kDitModulationSlots, 2, kDitNormEpsilon, run_stream);
  if (!CheckLaunch("H3 DiT attention gated residual", error)) {
    return false;
  }
  dit_ops::LaunchAdaLnApplyPrepared(
      impl_->gated_attention.As<std::uint16_t>(),
      impl_->norm2.storage.As<std::uint16_t>(),
      impl_->modulation.As<std::uint16_t>(), row_map,
      impl_->normalization_factors.As<float>(),
      impl_->modulation_mlp.As<std::uint16_t>(), rows, kDitHiddenSize,
      kDitModulationSlots, 3, 4, run_stream);
  if (!CheckLaunch("H3 DiT MLP AdaLN", error) ||
      !impl_->gemm.Run(impl_->fc1_weight.storage.As<std::uint16_t>(),
                       impl_->modulation_mlp.As<std::uint16_t>(),
                       impl_->fc1.As<std::uint16_t>(), impl_->rows,
                       kDitFeedForwardSize * 2U, kDitHiddenSize, run_stream,
                       error)) {
    return false;
  }
  if (impl_->silu_lookup != nullptr) {
    dit_ops::LaunchSwiGluLookup(
        impl_->fc1.As<std::uint16_t>(), impl_->activated.As<std::uint16_t>(),
        impl_->silu_lookup, rows, kDitFeedForwardSize, run_stream);
  } else {
    dit_ops::LaunchSwiGlu(impl_->fc1.As<std::uint16_t>(),
                          impl_->activated.As<std::uint16_t>(), rows,
                          kDitFeedForwardSize, run_stream);
  }
  if (!CheckLaunch("H3 DiT SwiGLU", error) ||
      !impl_->gemm.Run(impl_->fc2_weight.storage.As<std::uint16_t>(),
                       impl_->activated.As<std::uint16_t>(),
                       impl_->mlp_output.As<std::uint16_t>(), impl_->rows,
                       kDitHiddenSize, kDitFeedForwardSize, run_stream,
                       error)) {
    return false;
  }
  if (output_rms_required) {
    dit_ops::LaunchGateRmsInverse(
        impl_->gated_attention.As<std::uint16_t>(),
        impl_->mlp_output.As<std::uint16_t>(),
        impl_->modulation.As<std::uint16_t>(), row_map, block_target,
        impl_->normalization_factors.As<float>(), rows, kDitHiddenSize,
        kDitModulationSlots, 5, kDitNormEpsilon, run_stream);
  } else {
    dit_ops::LaunchGate(impl_->gated_attention.As<std::uint16_t>(),
                        impl_->mlp_output.As<std::uint16_t>(),
                        impl_->modulation.As<std::uint16_t>(), row_map,
                        block_target, rows, kDitHiddenSize, kDitModulationSlots,
                        5, run_stream);
  }
  if (!CheckLaunch("H3 DiT MLP gated residual", error)) {
    return false;
  }
  telemetry->dispatches = impl_->fused_attention
                              ? (input_rms_prepared ? 10 : 11)
                              : (input_rms_prepared ? 12 : 13);
  telemetry->attention_backend =
      impl_->fused_attention
          ? (impl_->rows > kMaximumMaterializedAttentionRows
                 ? "hip_bf16_wmma_flash"
                 : "ck_bf16_wmma_fused")
          : (impl_->row_parallel_attention ? "rocblas_materialized" : "scalar");
  if (device_run) {
    return true;
  }
  if (hipEventRecord(impl_->stop_event, run_stream) != hipSuccess) {
    if (error != nullptr && error->empty()) {
      *error = "cannot record MiniMax H3 DiT stop event";
    }
    return false;
  }

  const auto read_optional = [&](std::span<std::uint16_t> target,
                                 const DeviceBuffer& source,
                                 std::string_view label) {
    return target.empty() ||
           copy(target.data(), source.data(), target.size_bytes(),
                hipMemcpyDeviceToHost, label);
  };
  if (!read_optional(modulation_mlp, impl_->modulation_mlp,
                     "read H3 DiT MLP modulation") ||
      !copy(block_output.data(), impl_->block_output.data(),
            block_output.size_bytes(), hipMemcpyDeviceToHost,
            "read H3 DiT block output")) {
    return false;
  }
  const hipError_t synchronize = hipStreamSynchronize(impl_->stream);
  if (synchronize != hipSuccess) {
    SetError(error, HipError("synchronize MiniMax H3 DiT block", synchronize));
    return false;
  }
  float elapsed = 0.0F;
  if (hipEventElapsedTime(&elapsed, impl_->start_event, impl_->stop_event) !=
      hipSuccess) {
    SetError(error, "cannot read MiniMax H3 DiT elapsed time");
    return false;
  }
  telemetry->gpu_ms = elapsed;
  const std::array<std::pair<std::string_view, std::span<std::uint16_t>>, 4>
      retained_tensors = {{
          {"modulation_attention", modulation_attention},
          {"attention_output", attention_output},
          {"modulation_mlp", modulation_mlp},
          {"block_output", block_output},
      }};
  for (const auto& [name, values] : retained_tensors) {
    if (values.empty()) {
      continue;
    }
    telemetry->first_non_finite = FindFirstNonFinite(name, values);
    if (!telemetry->first_non_finite.empty()) {
      SetError(error, "MiniMax H3 DiT produced non-finite value at " +
                          telemetry->first_non_finite);
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 DiT block cancelled after completion");
    return false;
  }
  return true;
}

std::size_t DitBlockSession::rows() const noexcept {
  return impl_ == nullptr ? 0 : impl_->rows;
}

std::size_t DitBlockSession::block_index() const noexcept {
  return impl_ == nullptr ? 0 : impl_->block_index;
}

std::uintptr_t DitBlockSession::scratch_address() const noexcept {
  return impl_ == nullptr ? 0
                          : reinterpret_cast<std::uintptr_t>(impl_->qkv.data());
}

std::uint64_t DitBlockSession::weight_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->weight_bytes;
}

std::uint64_t DitBlockSession::activation_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->activation_bytes;
}

std::uint64_t DitBlockSession::gemm_plan_validations() const noexcept {
  return impl_ == nullptr ? 0 : impl_->gemm_plan_validations;
}

double DitBlockSession::load_ms() const noexcept {
  return impl_ == nullptr ? 0.0 : impl_->load_ms;
}

}  // namespace gufo::minimax_h3

#endif
