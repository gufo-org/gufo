#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/os_info.hpp>
#include "src/models/minimax_h3/json.hpp"
#include "src/models/minimax_h3/video_vae.hpp"
#include "src/models/minimax_h3/video_vae_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kWeightStagingBytes = 16ULL << 20U;
constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kBlocks = 36;
constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kHeadDimension = 64;
constexpr std::uint32_t kAttentionWidth = kHeads * kHeadDimension;
constexpr std::uint32_t kFeedForward = 8192;
constexpr std::uint32_t kRegisters = 4;
constexpr std::uint32_t kRopeHalf = 24;
constexpr std::uint32_t kOutputPatch = 3 * 4 * 16 * 16;
constexpr float kNormEpsilon = 1.0e-5F;
constexpr rocblas_int kGfx1151RocblasF32CompactSolution = -386;
constexpr rocblas_int kGfx1151RocblasF32WideSolution = -387;

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

bool CheckLaunch(std::string_view operation, std::string* error) {
  const hipError_t status = hipPeekAtLastError();
  if (status == hipSuccess) {
    return true;
  }
  SetError(error, HipError(operation, status));
  return false;
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
             "MiniMax H3 VisualVAE supports only integrated gfx1151 Gufo "
             "Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

using Faults = platform::PageFaults;
Faults ReadFaults() {
  return platform::ReadPageFaults();
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
      SetError(error, HipError("hipMalloc MiniMax H3 VisualVAE", status));
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
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    SetError(error,
             "MiniMax H3 VisualVAE tensor is missing: " + std::string(name));
    return nullptr;
  }
  if (!HasShape(*descriptor, DType::kFloat32, shape)) {
    SetError(error, "MiniMax H3 VisualVAE tensor contract differs: " +
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
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    SetError(error, "cannot open MiniMax H3 VisualVAE shard " + path.string() +
                        ": " + open_error);
    output->Reset();
    return false;
  }
  std::size_t copied = 0;
  bool success = true;
  while (copied < output->bytes()) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 VisualVAE weight load cancelled");
      success = false;
      break;
    }
    const std::size_t chunk = std::min(staging_bytes, output->bytes() - copied);
    std::size_t read_bytes = 0;
    while (read_bytes < chunk) {
      const std::size_t result = platform::ReadFileAt(
          file, static_cast<std::byte*>(staging) + read_bytes,
          chunk - read_bytes, descriptor.file_offset + copied + read_bytes);
      if (result == 0) {
        SetError(error, "short read loading MiniMax H3 VisualVAE tensor " +
                            descriptor.name);
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
      SetError(error, HipError("copy MiniMax H3 VisualVAE tensor", status));
      success = false;
      break;
    }
    copied += chunk;
  }
  platform::CloseFile(file);
  if (!success) {
    output->Reset();
  }
  return success;
}

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
      SetError(error, "invalid MiniMax H3 VisualVAE F32 GEMM");
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const rocblas_int pinned =
        (rows == kFeedForward * 2U && columns == kHidden) ||
                (rows == kHidden && columns == kFeedForward)
            ? kGfx1151RocblasF32WideSolution
            : kGfx1151RocblasF32CompactSolution;
    const auto gemm = [&](rocblas_int solution) {
      return rocblas_gemm_ex(
          handle_, rocblas_operation_transpose, rocblas_operation_none,
          static_cast<rocblas_int>(rows), static_cast<rocblas_int>(batch),
          static_cast<rocblas_int>(columns), &alpha, weight,
          rocblas_datatype_f32_r, static_cast<rocblas_int>(columns), input,
          rocblas_datatype_f32_r, static_cast<rocblas_int>(columns), &beta,
          output, rocblas_datatype_f32_r, static_cast<rocblas_int>(rows),
          output, rocblas_datatype_f32_r, static_cast<rocblas_int>(rows),
          rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, solution,
          0);
    };
    rocblas_status status =
        gemm(pinned_unavailable_ ? 0 : pinned);
    if (status != rocblas_status_success && !pinned_unavailable_) {
      // The pinned indices come from one rocBLAS database (ROCm 10.1); a
      // library without them takes its default solution instead.
      pinned_unavailable_ = true;
      status = gemm(0);
    }
    if (status != rocblas_status_success) {
      SetError(error, "MiniMax H3 VisualVAE F32 GEMM failed: " +
                          std::to_string(static_cast<int>(status)));
      return false;
    }
    return true;
  }

private:
  rocblas_handle handle_{nullptr};
  bool pinned_unavailable_{false};
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

  bool Scores(const float* query, const float* key, float* scores,
              std::size_t sequence, float scale, hipStream_t stream,
              std::string* error) {
    if (!SetStream(sequence, stream, error)) {
      return false;
    }
    constexpr float beta = 0.0F;
    const auto rows = static_cast<rocblas_int>(sequence);
    const auto token_stride = static_cast<rocblas_int>(kHeads * kHeadDimension);
    const auto head_stride = static_cast<rocblas_stride>(kHeadDimension);
    const auto score_stride = static_cast<rocblas_stride>(sequence * sequence);
    return Check(rocblas_gemm_strided_batched_ex(
                     handle_, rocblas_operation_transpose,
                     rocblas_operation_none, rows, rows, kHeadDimension, &scale,
                     key, rocblas_datatype_f32_r, token_stride, head_stride,
                     query, rocblas_datatype_f32_r, token_stride, head_stride,
                     &beta, scores, rocblas_datatype_f32_r, rows, score_stride,
                     scores, rocblas_datatype_f32_r, rows, score_stride, kHeads,
                     rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0),
                 "score", error);
  }

  bool Values(const float* value, const float* probabilities, float* output,
              std::size_t sequence, hipStream_t stream, std::string* error) {
    if (!SetStream(sequence, stream, error)) {
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto rows = static_cast<rocblas_int>(sequence);
    const auto token_stride = static_cast<rocblas_int>(kHeads * kHeadDimension);
    const auto head_stride = static_cast<rocblas_stride>(kHeadDimension);
    const auto probability_stride =
        static_cast<rocblas_stride>(sequence * sequence);
    return Check(
        rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_none, rocblas_operation_none,
            kHeadDimension, rows, rows, &alpha, value, rocblas_datatype_f32_r,
            token_stride, head_stride, probabilities, rocblas_datatype_f32_r,
            rows, probability_stride, &beta, output, rocblas_datatype_f32_r,
            token_stride, head_stride, output, rocblas_datatype_f32_r,
            token_stride, head_stride, kHeads, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0),
        "value", error);
  }

private:
  bool SetStream(std::size_t sequence, hipStream_t stream, std::string* error) {
    if (handle_ == nullptr || sequence == 0 ||
        sequence >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "invalid MiniMax H3 VisualVAE attention GEMM");
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
             "MiniMax H3 VisualVAE attention " + std::string(phase) +
                 " GEMM failed: " + std::to_string(static_cast<int>(status)));
    return false;
  }

  rocblas_handle handle_{nullptr};
};

bool ReadLatentNormalization(const ModelInventory& inventory,
                             std::array<float, kH3VideoVaeLatentChannels>* mean,
                             std::array<float, kH3VideoVaeLatentChannels>* std,
                             std::string* error) {
  try {
    const json::Value config = json::ParseFile(
        inventory.model_root() / "FL2VA/video_vae/config.json", 1U << 20U);
    const json::Value* mean_value = config.Find("latents_mean");
    const json::Value* std_value = config.Find("latents_std");
    if (mean_value == nullptr || std_value == nullptr ||
        !mean_value->IsArray() || !std_value->IsArray() ||
        mean_value->AsArray().size() != mean->size() ||
        std_value->AsArray().size() != std->size()) {
      throw json::Error("latent normalization shape differs");
    }
    for (std::size_t index = 0; index < mean->size(); ++index) {
      (*mean)[index] =
          static_cast<float>(mean_value->AsArray()[index].AsDouble());
      (*std)[index] =
          static_cast<float>(std_value->AsArray()[index].AsDouble());
      if (!std::isfinite((*mean)[index]) || !std::isfinite((*std)[index]) ||
          (*std)[index] <= 0.0F) {
        throw json::Error("latent normalization contains invalid values");
      }
    }
    return true;
  } catch (const std::exception& exception) {
    SetError(error, "MiniMax H3 VisualVAE config failed: " +
                        std::string(exception.what()));
    return false;
  }
}

struct BlockWeights {
  DeviceBuffer norm1;
  DeviceBuffer qkv_weight;
  DeviceBuffer qkv_bias;
  DeviceBuffer output_weight;
  DeviceBuffer output_bias;
  DeviceBuffer scale1;
  DeviceBuffer norm2;
  DeviceBuffer fc1_weight;
  DeviceBuffer fc1_bias;
  DeviceBuffer fc2_weight;
  DeviceBuffer fc2_bias;
  DeviceBuffer scale2;
};

struct ChunkPixels {
  std::vector<int> local_frames;
  std::vector<float> rgb;
};

bool IsFinite(std::span<const float> values, std::string* first_non_finite) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      *first_non_finite = "video_rgb[" + std::to_string(index) + "]";
      return false;
    }
  }
  return true;
}

}  // namespace

struct VideoVaeDecoder::Impl {
  const ModelInventory* inventory{nullptr};
  GenerationGeometry geometry;
  int output_frame_count{0};
  int chunks{0};
  int tile_size{0};
  VideoVaeTileAxis y_axis;
  VideoVaeTileAxis x_axis;
  std::uint32_t tile_latent_height{0};
  std::uint32_t tile_latent_width{0};
  std::uint32_t patches{0};
  std::uint32_t sequence{0};

  int device{0};
  hipStream_t stream{nullptr};
  void* staging{nullptr};
  F32Gemm gemm;
  AttentionGemm attention_gemm;

  DeviceBuffer latent_mean;
  DeviceBuffer latent_std;
  DeviceBuffer post_weight;
  DeviceBuffer post_bias;
  DeviceBuffer embed_weight;
  DeviceBuffer embed_bias;
  DeviceBuffer registers;
  std::vector<BlockWeights> blocks;
  DeviceBuffer final_norm_weight;
  DeviceBuffer final_norm_bias;
  DeviceBuffer output_weight;
  DeviceBuffer output_bias;

  DeviceBuffer normalized_latent;
  DeviceBuffer latent_rows;
  DeviceBuffer post;
  DeviceBuffer patch_hidden;
  DeviceBuffer hidden;
  DeviceBuffer norm;
  DeviceBuffer qkv;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer attention_scores;
  DeviceBuffer attention_heads;
  DeviceBuffer branch;
  DeviceBuffer fc1;
  DeviceBuffer activated;
  DeviceBuffer rope_cos;
  DeviceBuffer rope_sin;
  DeviceBuffer projected;
  DeviceBuffer local_frames_device;
  DeviceBuffer tile_rgb;

  std::vector<float> tile_rgb_host;
  VideoVaeTelemetry base_telemetry;

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
    base_telemetry.weight_bytes += descriptor.payload_bytes;
    base_telemetry.cumulative_allocation_bytes += descriptor.payload_bytes;
    return true;
  }

  bool LoadNamed(std::string_view name,
                 std::initializer_list<std::uint64_t> shape,
                 DeviceBuffer* output, const CancellationToken* cancellation,
                 std::string* error) {
    const TensorDescriptor* descriptor =
        RequireTensor(*inventory, name, shape, error);
    return descriptor != nullptr &&
           Load(*descriptor, output, cancellation, error);
  }

  bool LoadWeights(const CancellationToken* cancellation, std::string* error) {
    const auto begin = Clock::now();
    if (!LoadNamed(
            "post_quant_conv.weight",
            {kH3VideoVaeLatentChannels, kH3VideoVaeLatentChannels, 1, 1, 1},
            &post_weight, cancellation, error) ||
        !LoadNamed("post_quant_conv.bias", {kH3VideoVaeLatentChannels},
                   &post_bias, cancellation, error) ||
        !LoadNamed("decoder.x_embedder.weight",
                   {kHidden, kH3VideoVaeLatentChannels}, &embed_weight,
                   cancellation, error) ||
        !LoadNamed("decoder.x_embedder.bias", {kHidden}, &embed_bias,
                   cancellation, error) ||
        !LoadNamed("decoder.register_tokens", {1, kRegisters, kHidden},
                   &registers, cancellation, error)) {
      return false;
    }
    blocks.resize(kBlocks);
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 VisualVAE load cancelled");
        return false;
      }
      const std::string prefix =
          "decoder.transformer_blocks." + std::to_string(block) + ".";
      BlockWeights& weight = blocks[block];
      if (!LoadNamed(prefix + "norm1.weight", {kHidden}, &weight.norm1,
                     cancellation, error) ||
          !LoadNamed(prefix + "attn.to_qkv.weight",
                     {kAttentionWidth * 3U, kHidden}, &weight.qkv_weight,
                     cancellation, error) ||
          !LoadNamed(prefix + "attn.to_qkv.bias", {kAttentionWidth * 3U},
                     &weight.qkv_bias, cancellation, error) ||
          !LoadNamed(prefix + "attn.to_out.weight", {kHidden, kAttentionWidth},
                     &weight.output_weight, cancellation, error) ||
          !LoadNamed(prefix + "attn.to_out.bias", {kHidden},
                     &weight.output_bias, cancellation, error) ||
          !LoadNamed(prefix + "scale1", {kHidden}, &weight.scale1, cancellation,
                     error) ||
          !LoadNamed(prefix + "norm2.weight", {kHidden}, &weight.norm2,
                     cancellation, error) ||
          !LoadNamed(prefix + "ff.w1.weight", {kFeedForward * 2U, kHidden},
                     &weight.fc1_weight, cancellation, error) ||
          !LoadNamed(prefix + "ff.w1.bias", {kFeedForward * 2U},
                     &weight.fc1_bias, cancellation, error) ||
          !LoadNamed(prefix + "ff.w2.weight", {kHidden, kFeedForward},
                     &weight.fc2_weight, cancellation, error) ||
          !LoadNamed(prefix + "ff.w2.bias", {kHidden}, &weight.fc2_bias,
                     cancellation, error) ||
          !LoadNamed(prefix + "scale2", {kHidden}, &weight.scale2, cancellation,
                     error)) {
        return false;
      }
    }
    if (!LoadNamed("decoder.norm_out.weight", {kHidden}, &final_norm_weight,
                   cancellation, error) ||
        !LoadNamed("decoder.norm_out.bias", {kHidden}, &final_norm_bias,
                   cancellation, error) ||
        !LoadNamed("decoder.proj_out.weight", {kOutputPatch, kHidden},
                   &output_weight, cancellation, error) ||
        !LoadNamed("decoder.proj_out.bias", {kOutputPatch}, &output_bias,
                   cancellation, error)) {
      return false;
    }
    base_telemetry.load_ms = Milliseconds(begin, Clock::now());
    return true;
  }

  bool AllocateRuntime(
      const std::array<float, kH3VideoVaeLatentChannels>& mean,
      const std::array<float, kH3VideoVaeLatentChannels>& deviation,
      std::string* error) {
    const std::size_t latent_elements =
        static_cast<std::size_t>(kH3VideoVaeLatentChannels) *
        geometry.temporal.video_latent_frames * geometry.latent_height *
        geometry.latent_width;
    const auto f32 = [](std::size_t elements) {
      return elements * sizeof(float);
    };
    if (!latent_mean.Allocate(f32(mean.size()), error) ||
        !latent_std.Allocate(f32(deviation.size()), error) ||
        !normalized_latent.Allocate(f32(latent_elements), error) ||
        !latent_rows.Allocate(
            f32(static_cast<std::size_t>(patches) * kH3VideoVaeLatentChannels),
            error) ||
        !post.Allocate(latent_rows.bytes(), error) ||
        !patch_hidden.Allocate(f32(static_cast<std::size_t>(patches) * kHidden),
                               error) ||
        !hidden.Allocate(f32(static_cast<std::size_t>(sequence) * kHidden),
                         error) ||
        !norm.Allocate(hidden.bytes(), error) ||
        !qkv.Allocate(
            f32(static_cast<std::size_t>(sequence) * kAttentionWidth * 3U),
            error) ||
        !query.Allocate(
            f32(static_cast<std::size_t>(sequence) * kAttentionWidth), error) ||
        !key.Allocate(query.bytes(), error) ||
        !value.Allocate(query.bytes(), error) ||
        !attention_scores.Allocate(
            f32(static_cast<std::size_t>(kHeads) * sequence * sequence),
            error) ||
        !attention_heads.Allocate(query.bytes(), error) ||
        !branch.Allocate(hidden.bytes(), error) ||
        !fc1.Allocate(
            f32(static_cast<std::size_t>(sequence) * kFeedForward * 2U),
            error) ||
        !activated.Allocate(
            f32(static_cast<std::size_t>(sequence) * kFeedForward), error) ||
        !rope_cos.Allocate(f32(static_cast<std::size_t>(sequence) * kRopeHalf),
                           error) ||
        !rope_sin.Allocate(rope_cos.bytes(), error) ||
        !projected.Allocate(
            f32(static_cast<std::size_t>(sequence) * kOutputPatch), error) ||
        !local_frames_device.Allocate(
            kH3VideoVaeChunkFrames * sizeof(std::int32_t), error) ||
        !tile_rgb.Allocate(
            f32(static_cast<std::size_t>(kH3VideoVaeChunkFrames) *
                y_axis.length * x_axis.length * 3U),
            error) ||
        !CopyAsync(latent_mean.data(), mean.data(), f32(mean.size()),
                   hipMemcpyHostToDevice, "copy H3 VisualVAE latent mean",
                   error) ||
        !CopyAsync(latent_std.data(), deviation.data(), f32(deviation.size()),
                   hipMemcpyHostToDevice, "copy H3 VisualVAE latent deviation",
                   error)) {
      return false;
    }
    video_vae_ops::LaunchRope(rope_cos.As<float>(), rope_sin.As<float>(),
                              patches, sequence, kH3VideoVaeChunkLatents,
                              tile_latent_height, tile_latent_width, stream);
    if (!CheckLaunch("generate H3 VisualVAE RoPE", error)) {
      return false;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 VisualVAE setup", status));
      return false;
    }
    tile_rgb_host.resize(tile_rgb.bytes() / sizeof(float));
    const std::array<const DeviceBuffer*, 20> runtime = {
        &latent_mean,
        &latent_std,
        &normalized_latent,
        &latent_rows,
        &post,
        &patch_hidden,
        &hidden,
        &norm,
        &qkv,
        &query,
        &key,
        &value,
        &attention_scores,
        &attention_heads,
        &branch,
        &fc1,
        &activated,
        &rope_cos,
        &rope_sin,
        &projected,
    };
    std::uint64_t runtime_bytes = 0;
    for (const DeviceBuffer* buffer : runtime) {
      runtime_bytes += buffer->bytes();
    }
    runtime_bytes += local_frames_device.bytes() + tile_rgb.bytes();
    base_telemetry.persistent_bytes = base_telemetry.weight_bytes +
                                      latent_mean.bytes() + latent_std.bytes() +
                                      normalized_latent.bytes();
    base_telemetry.scratch_bytes = runtime_bytes - latent_mean.bytes() -
                                   latent_std.bytes() -
                                   normalized_latent.bytes();
    base_telemetry.peak_live_bytes =
        base_telemetry.persistent_bytes + base_telemetry.scratch_bytes;
    base_telemetry.cumulative_allocation_bytes += runtime_bytes;
    base_telemetry.scratch_address =
        reinterpret_cast<std::uintptr_t>(qkv.data());
    return true;
  }

  bool Linear(const DeviceBuffer& input, const DeviceBuffer& weight,
              const DeviceBuffer& bias, DeviceBuffer* output, std::size_t rows,
              std::size_t input_width, std::size_t output_width,
              std::string* error) {
    if (!Matmul(input, weight, output, rows, input_width, output_width,
                error)) {
      return false;
    }
    video_vae_ops::LaunchAddBias(
        output->As<float>(), bias.As<float>(), static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(output_width), stream);
    return CheckLaunch("H3 VisualVAE linear bias", error);
  }

  bool Matmul(const DeviceBuffer& input, const DeviceBuffer& weight,
              DeviceBuffer* output, std::size_t rows, std::size_t input_width,
              std::size_t output_width, std::string* error) {
    return gemm.Run(weight.As<float>(), input.As<float>(), output->As<float>(),
                    rows, output_width, input_width, stream, error);
  }

  bool RunTile(int chunk, int tile_y, int tile_x,
               std::span<const int> local_frames,
               const CancellationToken* cancellation,
               std::vector<float>* output, std::string* error) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 VisualVAE cancelled before tile");
      return false;
    }
    const auto begin = Clock::now();
    video_vae_ops::LaunchPrepareTile(
        normalized_latent.As<float>(), latent_mean.As<float>(),
        latent_std.As<float>(), latent_rows.As<float>(),
        geometry.temporal.video_latent_frames, geometry.latent_height,
        geometry.latent_width, chunk * kH3VideoVaeChunkStride,
        y_axis.starts[static_cast<std::size_t>(tile_y)] /
            kH3VideoVaeSpatialRatio,
        x_axis.starts[static_cast<std::size_t>(tile_x)] /
            kH3VideoVaeSpatialRatio,
        kH3VideoVaeChunkLatents, tile_latent_height, tile_latent_width,
        kH3VideoVaeLatentChannels, stream);
    if (!CheckLaunch("prepare H3 VisualVAE tile", error) ||
        !Linear(latent_rows, post_weight, post_bias, &post, patches,
                kH3VideoVaeLatentChannels, kH3VideoVaeLatentChannels, error) ||
        !Linear(post, embed_weight, embed_bias, &patch_hidden, patches,
                kH3VideoVaeLatentChannels, kHidden, error) ||
        !CopyAsync(hidden.data(), patch_hidden.data(), patch_hidden.bytes(),
                   hipMemcpyDeviceToDevice, "pack H3 VisualVAE patch tokens",
                   error) ||
        !CopyAsync(static_cast<float*>(hidden.data()) +
                       static_cast<std::size_t>(patches) * kHidden,
                   registers.data(), registers.bytes(), hipMemcpyDeviceToDevice,
                   "pack H3 VisualVAE register tokens", error)) {
      return false;
    }
    hipError_t status = hipMemsetAsync(
        static_cast<float*>(hidden.data()) +
            static_cast<std::size_t>(patches + kRegisters) * kHidden,
        0, static_cast<std::size_t>(kHidden) * sizeof(float), stream);
    if (status != hipSuccess) {
      SetError(error, HipError("pack H3 VisualVAE suffix token", status));
      return false;
    }
    for (std::size_t block = 0; block < blocks.size(); ++block) {
      if (IsCancelled(cancellation)) {
        SetError(error, "MiniMax H3 VisualVAE cancelled before block " +
                            std::to_string(block));
        return false;
      }
      BlockWeights& weight = blocks[block];
      video_vae_ops::LaunchRmsNorm(hidden.As<float>(), weight.norm1.As<float>(),
                                   norm.As<float>(), sequence, kHidden,
                                   kNormEpsilon, stream);
      if (!Matmul(norm, weight.qkv_weight, &qkv, sequence, kHidden,
                  kAttentionWidth * 3U, error)) {
        return false;
      }
      video_vae_ops::LaunchVideoQkvRope(
          qkv.As<float>(), weight.qkv_bias.As<float>(), rope_cos.As<float>(),
          rope_sin.As<float>(), query.As<float>(), key.As<float>(),
          value.As<float>(), sequence, kHeads, kHeadDimension, kRopeHalf,
          kNormEpsilon, stream);
      if (!attention_gemm.Scores(
              query.As<float>(), key.As<float>(), attention_scores.As<float>(),
              sequence, 1.0F / std::sqrt(static_cast<float>(kHeadDimension)),
              stream, error)) {
        return false;
      }
      video_vae_ops::LaunchSoftmaxScoresInPlace(attention_scores.As<float>(),
                                                sequence, kHeads, stream);
      if (!CheckLaunch("H3 VisualVAE attention softmax", error) ||
          !attention_gemm.Values(
              value.As<float>(), attention_scores.As<float>(),
              attention_heads.As<float>(), sequence, stream, error) ||
          !Matmul(attention_heads, weight.output_weight, &branch, sequence,
                  kAttentionWidth, kHidden, error)) {
        return false;
      }
      video_vae_ops::LaunchScaleAdd(hidden.As<float>(), branch.As<float>(),
                                    weight.output_bias.As<float>(),
                                    weight.scale1.As<float>(), sequence,
                                    kHidden, stream);
      video_vae_ops::LaunchRmsNorm(hidden.As<float>(), weight.norm2.As<float>(),
                                   norm.As<float>(), sequence, kHidden,
                                   kNormEpsilon, stream);
      if (!CheckLaunch("H3 VisualVAE attention residual/norm", error) ||
          !Matmul(norm, weight.fc1_weight, &fc1, sequence, kHidden,
                  kFeedForward * 2U, error)) {
        return false;
      }
      video_vae_ops::LaunchSwiGlu(fc1.As<float>(), weight.fc1_bias.As<float>(),
                                  activated.As<float>(), sequence, kFeedForward,
                                  stream);
      if (!CheckLaunch("H3 VisualVAE SwiGLU", error) ||
          !Matmul(activated, weight.fc2_weight, &branch, sequence, kFeedForward,
                  kHidden, error)) {
        return false;
      }
      video_vae_ops::LaunchScaleAdd(
          hidden.As<float>(), branch.As<float>(), weight.fc2_bias.As<float>(),
          weight.scale2.As<float>(), sequence, kHidden, stream);
      if (!CheckLaunch("H3 VisualVAE MLP residual", error)) {
        return false;
      }
      status = hipStreamSynchronize(stream);
      if (status != hipSuccess) {
        SetError(error, HipError("synchronize H3 VisualVAE block", status));
        return false;
      }
      base_telemetry.dispatches += 9;
    }
    video_vae_ops::LaunchLayerNorm(
        hidden.As<float>(), final_norm_weight.As<float>(),
        final_norm_bias.As<float>(), norm.As<float>(), sequence, kHidden,
        kNormEpsilon, stream);
    if (!CheckLaunch("H3 VisualVAE output norm", error) ||
        !Linear(norm, output_weight, output_bias, &projected, sequence, kHidden,
                kOutputPatch, error)) {
      return false;
    }
    if (!CopyAsync(local_frames_device.data(), local_frames.data(),
                   local_frames.size_bytes(), hipMemcpyHostToDevice,
                   "copy H3 VisualVAE selected frames", error)) {
      return false;
    }
    video_vae_ops::LaunchUnpackRgb(
        projected.As<float>(), local_frames_device.As<std::int32_t>(),
        tile_rgb.As<float>(), static_cast<std::uint32_t>(local_frames.size()),
        tile_latent_height, tile_latent_width, kH3VideoVaeChunkFrames, stream);
    const std::size_t tile_elements = local_frames.size() *
                                      static_cast<std::size_t>(y_axis.length) *
                                      x_axis.length * 3U;
    if (!CheckLaunch("unpack H3 VisualVAE RGB", error) ||
        !CopyAsync(tile_rgb_host.data(), tile_rgb.data(),
                   tile_elements * sizeof(float), hipMemcpyDeviceToHost,
                   "read H3 VisualVAE RGB tile", error)) {
      return false;
    }
    status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
      SetError(error, HipError("synchronize H3 VisualVAE tile", status));
      return false;
    }
    output->assign(
        tile_rgb_host.begin(),
        tile_rgb_host.begin() + static_cast<std::ptrdiff_t>(tile_elements));
    base_telemetry.dispatches += 10;
    base_telemetry.tile_ms.push_back(Milliseconds(begin, Clock::now()));
    return true;
  }

  bool StitchTiles(const std::vector<std::vector<float>>& tiles,
                   std::size_t frame_count, std::vector<float>* output,
                   std::string* error) const {
    return StitchVideoVaeTiles(y_axis, x_axis, frame_count, tiles, output,
                               error);
  }

  bool RunChunk(int chunk, std::span<const int> local_frames,
                const CancellationToken* cancellation,
                VideoVaeProgress progress, void* progress_opaque,
                int* completed_tiles, int total_tiles, ChunkPixels* output,
                std::string* error) {
    if (local_frames.empty() ||
        std::ranges::any_of(local_frames, [](int frame) {
          return frame < 0 || frame >= kH3VideoVaeChunkFrames;
        })) {
      SetError(error, "invalid MiniMax H3 VisualVAE local frame selection");
      return false;
    }
    const int tile_count =
        static_cast<int>(y_axis.starts.size() * x_axis.starts.size());
    std::vector<std::vector<float>> tiles(static_cast<std::size_t>(tile_count));
    const int x_count = static_cast<int>(x_axis.starts.size());
    for (int tile_y = 0; tile_y < static_cast<int>(y_axis.starts.size());
         ++tile_y) {
      for (int tile_x = 0; tile_x < x_count; ++tile_x) {
        if (!RunTile(
                chunk, tile_y, tile_x, local_frames, cancellation,
                &tiles[static_cast<std::size_t>(tile_y * x_count + tile_x)],
                error)) {
          return false;
        }
        ++*completed_tiles;
        if (progress != nullptr) {
          progress(*completed_tiles, total_tiles, progress_opaque);
        }
      }
    }
    output->local_frames.assign(local_frames.begin(), local_frames.end());
    return StitchTiles(tiles, local_frames.size(), &output->rgb, error);
  }

  void PublishTelemetry(VideoVaeTelemetry* telemetry) const {
    if (telemetry != nullptr) {
      *telemetry = base_telemetry;
    }
  }
};

VideoVaeDecoder::VideoVaeDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VideoVaeDecoder::~VideoVaeDecoder() = default;
VideoVaeDecoder::VideoVaeDecoder(VideoVaeDecoder&&) noexcept = default;
VideoVaeDecoder& VideoVaeDecoder::operator=(VideoVaeDecoder&&) noexcept =
    default;

std::unique_ptr<VideoVaeDecoder> VideoVaeDecoder::Create(
    const ModelInventory& inventory, const VideoVaeOptions& options,
    const CancellationToken* cancellation, VideoVaeTelemetry* telemetry,
    std::string* error) {
  const Faults faults_before = ReadFaults();
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 VisualVAE creation cancelled");
    return nullptr;
  }
  auto plan = ResolveVideoVaePlan(options.geometry, error);
  if (!plan) {
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  impl->geometry = options.geometry;
  impl->chunks = plan->chunks;
  impl->output_frame_count = plan->output_frames;
  impl->tile_size = plan->tile_pixels;
  impl->y_axis = std::move(plan->y_axis);
  impl->x_axis = std::move(plan->x_axis);
  impl->tile_latent_height = plan->tile_latent_height;
  impl->tile_latent_width = plan->tile_latent_width;
  impl->patches = plan->patches;
  impl->sequence = plan->sequence;
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
    SetError(error, HipError("create MiniMax H3 VisualVAE resources", status));
    return nullptr;
  }
  std::array<float, kH3VideoVaeLatentChannels> mean{};
  std::array<float, kH3VideoVaeLatentChannels> deviation{};
  if (!ReadLatentNormalization(inventory, &mean, &deviation, error) ||
      !impl->LoadWeights(cancellation, error) ||
      !impl->AllocateRuntime(mean, deviation, error)) {
    return nullptr;
  }
  (void)hipHostFree(impl->staging);
  impl->staging = nullptr;
  const Faults faults_after = ReadFaults();
  impl->base_telemetry.minor_page_faults =
      faults_after.minor - faults_before.minor;
  impl->base_telemetry.major_page_faults =
      faults_after.major - faults_before.major;
  impl->base_telemetry.swap_bytes = ReadSwapBytes();
  impl->PublishTelemetry(telemetry);
  return std::unique_ptr<VideoVaeDecoder>(new VideoVaeDecoder(std::move(impl)));
}

bool VideoVaeDecoder::Decode(std::span<const float> normalized_latent,
                             const CancellationToken* cancellation,
                             VideoVaeProgress progress, void* progress_opaque,
                             VideoFrames* output, VideoVaeTelemetry* telemetry,
                             std::string* error) {
  if (impl_ == nullptr) {
    SetError(error, "invalid MiniMax H3 VisualVAE decoder");
    return false;
  }
  std::vector<int> frames(static_cast<std::size_t>(impl_->output_frame_count));
  std::iota(frames.begin(), frames.end(), 0);
  return DecodeSelected(normalized_latent, frames, cancellation, progress,
                        progress_opaque, output, telemetry, error);
}

bool VideoVaeDecoder::DecodeSelected(std::span<const float> normalized_latent,
                                     std::span<const int> frame_indices,
                                     const CancellationToken* cancellation,
                                     VideoVaeProgress progress,
                                     void* progress_opaque, VideoFrames* output,
                                     VideoVaeTelemetry* telemetry,
                                     std::string* error) {
  const Faults faults_before = ReadFaults();
  if (output != nullptr) {
    *output = {};
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 VisualVAE cancelled before decode");
    return false;
  }
  if (impl_ == nullptr || output == nullptr || frame_indices.empty() ||
      normalized_latent.size_bytes() != impl_->normalized_latent.bytes() ||
      frame_indices.front() < 0 ||
      frame_indices.back() >= impl_->output_frame_count ||
      !std::ranges::is_sorted(frame_indices) ||
      std::adjacent_find(frame_indices.begin(), frame_indices.end()) !=
          frame_indices.end()) {
    SetError(error, "invalid MiniMax H3 VisualVAE decode arguments");
    return false;
  }
  if (!impl_->CopyAsync(impl_->normalized_latent.data(),
                        normalized_latent.data(),
                        normalized_latent.size_bytes(), hipMemcpyHostToDevice,
                        "copy H3 VisualVAE normalized latent", error)) {
    return false;
  }
  hipError_t status = hipStreamSynchronize(impl_->stream);
  if (status != hipSuccess) {
    SetError(error, HipError("synchronize H3 VisualVAE input", status));
    return false;
  }
  const auto begin = Clock::now();
  VideoVaePlan frame_plan;
  frame_plan.chunks = impl_->chunks;
  frame_plan.output_frames = impl_->output_frame_count;
  std::vector<VideoVaeFrameRecipe> recipes(frame_indices.size());
  std::map<int, std::array<bool, kH3VideoVaeChunkFrames>> requested;
  for (std::size_t index = 0; index < frame_indices.size(); ++index) {
    auto recipe =
        ResolveVideoVaeFrameRecipe(frame_plan, frame_indices[index], error);
    if (!recipe) {
      return false;
    }
    recipes[index] = *recipe;
    for (int source = 0; source < recipe->source_count; ++source) {
      requested[recipe->sources[static_cast<std::size_t>(source)]
                    .chunk][static_cast<std::size_t>(
          recipe->sources[static_cast<std::size_t>(source)].local_frame)] =
          true;
    }
  }
  std::map<int, ChunkPixels> decoded;
  const int spatial_tiles = static_cast<int>(impl_->y_axis.starts.size() *
                                             impl_->x_axis.starts.size());
  const int total_tiles = static_cast<int>(requested.size()) * spatial_tiles;
  int completed_tiles = 0;
  for (const auto& [chunk, mask] : requested) {
    std::vector<int> local_frames;
    for (int frame = 0; frame < kH3VideoVaeChunkFrames; ++frame) {
      if (mask[static_cast<std::size_t>(frame)]) {
        local_frames.push_back(frame);
      }
    }
    if (!impl_->RunChunk(chunk, local_frames, cancellation, progress,
                         progress_opaque, &completed_tiles, total_tiles,
                         &decoded[chunk], error)) {
      return false;
    }
  }
  const std::size_t frame_elements =
      static_cast<std::size_t>(impl_->geometry.height) * impl_->geometry.width *
      3U;
  output->height = impl_->geometry.height;
  output->width = impl_->geometry.width;
  output->frame_indices.assign(frame_indices.begin(), frame_indices.end());
  output->rgb.assign(frame_indices.size() * frame_elements, 0.0F);
  for (std::size_t index = 0; index < recipes.size(); ++index) {
    const VideoVaeFrameRecipe& recipe = recipes[index];
    float* destination = output->rgb.data() + index * frame_elements;
    for (int source_index = 0; source_index < recipe.source_count;
         ++source_index) {
      const VideoVaeFrameSource& source =
          recipe.sources[static_cast<std::size_t>(source_index)];
      const ChunkPixels& chunk = decoded.at(source.chunk);
      const auto iterator =
          std::ranges::find(chunk.local_frames, source.local_frame);
      if (iterator == chunk.local_frames.end()) {
        SetError(error, "MiniMax H3 VisualVAE selected frame is missing");
        return false;
      }
      const std::size_t local_index =
          static_cast<std::size_t>(iterator - chunk.local_frames.begin());
      const float* pixels = chunk.rgb.data() + local_index * frame_elements;
      if (source_index == 0 && source.weight == 1.0F) {
        std::copy_n(pixels, frame_elements, destination);
      } else {
        for (std::size_t pixel = 0; pixel < frame_elements; ++pixel) {
          destination[pixel] += pixels[pixel] * source.weight;
        }
      }
    }
  }
  if (!IsFinite(output->rgb, &impl_->base_telemetry.first_non_finite)) {
    SetError(error, "MiniMax H3 VisualVAE produced non-finite RGB");
    return false;
  }
  impl_->base_telemetry.decode_ms = Milliseconds(begin, Clock::now());
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

const GenerationGeometry& VideoVaeDecoder::geometry() const noexcept {
  static const GenerationGeometry empty;
  return impl_ == nullptr ? empty : impl_->geometry;
}

int VideoVaeDecoder::output_frames() const noexcept {
  return impl_ == nullptr ? 0 : impl_->output_frame_count;
}

int VideoVaeDecoder::tile_pixels() const noexcept {
  return impl_ == nullptr ? 0 : impl_->tile_size;
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
