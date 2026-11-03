// Model-private VisualVAE encoder device graph (issue #177 keyframe
// conditioning). Single-frame (T=1) semantics: the causal 3x3x3 convolutions
// of EncoderFCN3D collapse onto their last temporal tap (conv.py
// _apply_temporal_padding prepends kernel_t-1 zero frames, so only tap index
// 2 sees data) with reflect-padded 2-D im2col + F32 GEMM. GroupNorm
// (32 groups, eps 1e-6, affine) stays per-plane exactly as
// norm.py:get_group_norm_3d with use_t_isolated_gn=true; Downsample3D
// pre-pads right/bottom only (F.pad (0,1,0,1) reflect) before a stride-2 k3
// conv with no conv-side spatial padding. Activations are position-major
// (NHWC) between kernels; only the public boundaries are planar CHW. Nothing
// here is shared with any other model.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/models/minimax_h3/runtime.hpp"
#include "src/models/minimax_h3/video_vae_encoder.hpp"
#include "src/models/minimax_h3/video_vae_encoder_ops.hip.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::minimax_h3 {
namespace {

namespace enc_ops = video_vae_encoder_ops;

void EncSetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

bool EncCheckLaunch(std::string_view phase, std::string* error) {
  const hipError_t status = hipGetLastError();
  if (status == hipSuccess) {
    return true;
  }
  EncSetError(error, std::string("MiniMax H3 VisualVAE encoder ") +
                         std::string(phase) + " launch: " +
                         hipGetErrorString(status));
  return false;
}

bool EncSync(hipStream_t stream, std::string_view phase, std::string* error) {
  const hipError_t status = hipStreamSynchronize(stream);
  if (status == hipSuccess) {
    return true;
  }
  EncSetError(error, std::string("MiniMax H3 VisualVAE encoder ") +
                         std::string(phase) + " synchronize: " +
                         hipGetErrorString(status));
  return false;
}

class EncDeviceBuffer {
public:
  EncDeviceBuffer() = default;
  ~EncDeviceBuffer() { Reset(); }
  EncDeviceBuffer(const EncDeviceBuffer&) = delete;
  EncDeviceBuffer& operator=(const EncDeviceBuffer&) = delete;
  EncDeviceBuffer(EncDeviceBuffer&& other) noexcept
      : data_(other.data_), bytes_(other.bytes_) {
    other.data_ = nullptr;
    other.bytes_ = 0;
  }
  EncDeviceBuffer& operator=(EncDeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = other.data_;
      bytes_ = other.bytes_;
      other.data_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  [[nodiscard]] bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      EncSetError(error, std::string("allocate MiniMax H3 VisualVAE encoder "
                                     "buffer: ") +
                             hipGetErrorString(status));
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

  template <typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template <typename T>
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

// Bias broadcast-add over a position-major output: out[p][c] += bias[c].
__global__ void AddBiasKernel(float* output, const float* bias,
                              std::uint32_t batch, std::uint32_t channels) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(batch) * channels) {
    return;
  }
  output[index] += bias[index % channels];
}

void LaunchAddBias(float* output, const float* bias, std::uint32_t batch,
                   std::uint32_t channels, hipStream_t stream) {
  const std::size_t elements = static_cast<std::size_t>(batch) * channels;
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(AddBiasKernel,
                     dim3(static_cast<std::uint32_t>(
                         (elements + kThreads - 1U) / kThreads)),
                     dim3(kThreads), 0, stream, output, bias, batch, channels);
}

// One flattened convolution: row-major weight [out][columns] with
// columns = in*9 for 3x3 (temporal tap 2 preselected) or in for 1x1, plus
// bias.
struct FlatConv {
  EncDeviceBuffer weight;
  EncDeviceBuffer bias;
  std::uint32_t out_channels{0};
  std::uint32_t columns{0};
};

struct LevelWeights {
  std::array<FlatConv, 4> convs{};  // b0c0 b0c1 b1c0 b1c1
  std::array<EncDeviceBuffer, 4> norm_weights{};
  std::array<EncDeviceBuffer, 4> norm_biases{};
  std::array<FlatConv, 2> shortcut{};  // 1x1, block 0/1
  std::array<bool, 2> has_shortcut{};
  FlatConv downsample;
  bool has_downsample{false};
  // Channel bookkeeping for the tile loop.
  std::array<std::uint32_t, 2> block_in{};
  std::array<std::uint32_t, 2> block_out{};
};

bool LoadTensorHost(const ModelInventory& inventory, std::string_view name,
                    std::vector<float>& out, std::string* error) {
  const TensorDescriptor* descriptor = inventory.FindTensor(name);
  if (descriptor == nullptr) {
    EncSetError(error, "MiniMax H3 VisualVAE encoder tensor is missing: " +
                           std::string(name));
    return false;
  }
  if (descriptor->dtype != DType::kFloat32) {
    EncSetError(error, "MiniMax H3 VisualVAE encoder tensor dtype differs: " +
                           std::string(name));
    return false;
  }
  if (descriptor->payload_bytes >
      std::numeric_limits<std::size_t>::max() - sizeof(float)) {
    EncSetError(error, "MiniMax H3 VisualVAE encoder tensor too large: " +
                           std::string(name));
    return false;
  }
  out.resize(static_cast<std::size_t>(descriptor->payload_bytes) /
             sizeof(float));
  const std::filesystem::path path = inventory.model_root() / descriptor->shard;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    EncSetError(error, "cannot open MiniMax H3 VisualVAE encoder shard " +
                           path.string() + ": " + open_error);
    return false;
  }
  std::size_t read_bytes = 0;
  while (read_bytes < out.size() * sizeof(float)) {
    const std::size_t result = platform::ReadFileAt(
        file, reinterpret_cast<std::byte*>(out.data()) + read_bytes,
        out.size() * sizeof(float) - read_bytes,
        descriptor->file_offset + read_bytes);
    if (result == 0) {
      platform::CloseFile(file);
      EncSetError(error, "short read loading MiniMax H3 VisualVAE encoder "
                         "tensor " +
                             std::string(name));
      return false;
    }
    read_bytes += result;
  }
  platform::CloseFile(file);
  return true;
}

bool UploadBuffer(hipStream_t stream, const EncDeviceBuffer& target,
                  const std::vector<float>& source, std::string* error) {
  if (source.empty()) {
    return true;
  }
  const hipError_t status = hipMemcpyAsync(
      const_cast<void*>(target.data()), source.data(),
      source.size() * sizeof(float), hipMemcpyHostToDevice, stream);
  if (status != hipSuccess) {
    EncSetError(error,
                std::string("upload MiniMax H3 VisualVAE encoder weight: ") +
                    hipGetErrorString(status));
    return false;
  }
  return EncSync(stream, "weight upload", error);
}

// [cout][cin][3][3][3] row-major -> [cout][cin*9] taking temporal tap 2.
void FlattenConvWeight(const std::vector<float>& source, int out_channels,
                       int in_channels, std::vector<float>& destination) {
  destination.assign(static_cast<std::size_t>(out_channels) * in_channels * 9U,
                     0.0F);
  for (int oc = 0; oc < out_channels; ++oc) {
    for (int ic = 0; ic < in_channels; ++ic) {
      const std::size_t origin =
          (static_cast<std::size_t>(oc) * in_channels + ic) * 27U + 18U;
      float* row = destination.data() +
                   (static_cast<std::size_t>(oc) * in_channels + ic) * 9U;
      for (std::size_t tap = 0; tap < 9U; ++tap) {
        row[tap] = source[origin + tap];
      }
    }
  }
}

// Position-major -> planar transpose of a small host block.
void TransposeToPlanar(const std::vector<float>& source, int channels,
                       int height, int width, std::vector<float>& destination) {
  destination.assign(source.size(), 0.0F);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t position = static_cast<std::size_t>(y) * width + x;
      for (int c = 0; c < channels; ++c) {
        destination[(static_cast<std::size_t>(c) * height + y) * width + x] =
            source[position * channels + c];
      }
    }
  }
}

}  // namespace

struct VideoVaeEncoder::Impl {
  static constexpr std::uint32_t kIm2ColChunk = 4096U;

  const ModelInventory* inventory{nullptr};
  int device{0};
  hipStream_t stream{nullptr};

  FlatConv conv_in;                                   // 128 x 27
  std::array<LevelWeights, kH3VaeEncoderLevels> levels{};
  EncDeviceBuffer norm_out_weight;
  EncDeviceBuffer norm_out_bias;
  FlatConv conv_out;  // 48 x 9216
  FlatConv quant;     // 48 x 48
  std::uint64_t weight_bytes{0};

  EncDeviceBuffer buffer_a;
  EncDeviceBuffer buffer_b;
  EncDeviceBuffer buffer_s;  // residual source
  EncDeviceBuffer columns;
  EncDeviceBuffer input_tile;
  EncDeviceBuffer gn_sums;
  EncDeviceBuffer gn_squares;
  EncDeviceBuffer gn_means;
  EncDeviceBuffer gn_inverses;

  ~Impl() {
    if (stream != nullptr) {
      (void)hipStreamDestroy(stream);
    }
  }

  bool Load(std::string* error) {
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder device query failed");
      return false;
    }
    status = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (status != hipSuccess) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder stream creation failed");
      return false;
    }
    if (!LoadConv3x3("encoder.conv_in", 128, 3, &conv_in, error)) {
      return false;
    }
    for (int level = 0; level < kH3VaeEncoderLevels; ++level) {
      const std::size_t l = static_cast<std::size_t>(level);
      const std::uint32_t out =
          static_cast<std::uint32_t>(kH3VaeEncoderLevelChannels[l]);
      const std::uint32_t block_in =
          level == 0 ? static_cast<std::uint32_t>(
                           kH3VaeEncoderLevelChannels[0])
                     : static_cast<std::uint32_t>(
                           kH3VaeEncoderLevelChannels[l - 1U]);
      LevelWeights& weights = levels[l];
      weights.block_in = {block_in, out};
      weights.block_out = {out, out};
      const std::string base = "encoder.down." + std::to_string(level);
      for (int block = 0; block < kH3VaeEncoderResidualBlocks; ++block) {
        const std::size_t b = static_cast<std::size_t>(block);
        const std::uint32_t cin = weights.block_in[b];
        const std::uint32_t cout = weights.block_out[b];
        const std::string block_base = base + ".block." + std::to_string(block);
        if (!LoadNorm(block_base + ".norm1", cin,
                      &weights.norm_weights[b * 2U],
                      &weights.norm_biases[b * 2U], error) ||
            !LoadConv3x3(block_base + ".conv1", static_cast<int>(cout),
                         static_cast<int>(cin), &weights.convs[b * 2U],
                         error) ||
            !LoadNorm(block_base + ".norm2", cout,
                      &weights.norm_weights[b * 2U + 1U],
                      &weights.norm_biases[b * 2U + 1U], error) ||
            !LoadConv3x3(block_base + ".conv2", static_cast<int>(cout),
                         static_cast<int>(cout), &weights.convs[b * 2U + 1U],
                         error)) {
          return false;
        }
        weights.has_shortcut[b] = cin != cout;
        if (weights.has_shortcut[b] &&
            !LoadConv1x1(block_base + ".nin_shortcut", static_cast<int>(cout),
                         static_cast<int>(cin), &weights.shortcut[b], error)) {
          return false;
        }
      }
      weights.has_downsample =
          kH3VaeEncoderSpatialDownsample[l] > 1;
      if (weights.has_downsample &&
          !LoadConv3x3(base + ".downsample.conv", static_cast<int>(out),
                       static_cast<int>(out), &weights.downsample, error)) {
        return false;
      }
    }
    if (!LoadNorm("encoder.norm_out", 1024, &norm_out_weight, &norm_out_bias,
                  error)) {
      return false;
    }
    if (!LoadConv3x3("encoder.conv_out", kH3VaeEncoderMomentChannels, 1024,
                     &conv_out, error)) {
      return false;
    }
    if (!LoadConv1x1("quant_conv", kH3VaeEncoderMomentChannels,
                     kH3VaeEncoderMomentChannels, &quant, error)) {
      return false;
    }

    const std::size_t max_positions =
        static_cast<std::size_t>(kH3VaeEncoderTilePixels) *
        kH3VaeEncoderTilePixels;
    const std::size_t max_activation =
        max_positions * 128U * sizeof(float);
    const std::size_t max_columns =
        static_cast<std::size_t>(kIm2ColChunk) * 1024U * 9U * sizeof(float);
    const std::uint32_t gn_chunks =
        static_cast<std::uint32_t>((max_positions + 31U) / 32U);
    if (!buffer_a.Allocate(max_activation, error) ||
        !buffer_b.Allocate(max_activation, error) ||
        !buffer_s.Allocate(max_activation, error) ||
        !columns.Allocate(max_columns, error) ||
        !input_tile.Allocate(max_positions * 3U * sizeof(float), error) ||
        !gn_sums.Allocate(static_cast<std::size_t>(gn_chunks) *
                              kH3VaeEncoderGroupNormGroups * sizeof(float),
                          error) ||
        !gn_squares.Allocate(gn_sums.bytes(), error) ||
        !gn_means.Allocate(kH3VaeEncoderGroupNormGroups * sizeof(float),
                           error) ||
        !gn_inverses.Allocate(gn_means.bytes(), error)) {
      return false;
    }
    return true;
  }

  bool LoadConv3x3(const std::string& base, int out_channels, int in_channels,
                   FlatConv* conv, std::string* error) {
    std::vector<float> source;
    std::vector<float> bias;
    if (!LoadTensorHost(*inventory, base + ".weight", source, error) ||
        !LoadTensorHost(*inventory, base + ".bias", bias, error)) {
      return false;
    }
    if (source.size() !=
        static_cast<std::size_t>(out_channels) * in_channels * 27U) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder conv contract: " + base);
      return false;
    }
    std::vector<float> flat;
    FlattenConvWeight(source, out_channels, in_channels, flat);
    conv->out_channels = static_cast<std::uint32_t>(out_channels);
    conv->columns = static_cast<std::uint32_t>(in_channels) * 9U;
    if (!conv->weight.Allocate(flat.size() * sizeof(float), error) ||
        !conv->bias.Allocate(bias.size() * sizeof(float), error) ||
        !UploadBuffer(stream, conv->weight, flat, error) ||
        !UploadBuffer(stream, conv->bias, bias, error)) {
      return false;
    }
    weight_bytes += conv->weight.bytes() + conv->bias.bytes();
    return true;
  }

  bool LoadConv1x1(const std::string& base, int out_channels, int in_channels,
                   FlatConv* conv, std::string* error) {
    std::vector<float> source;
    std::vector<float> bias;
    if (!LoadTensorHost(*inventory, base + ".weight", source, error) ||
        !LoadTensorHost(*inventory, base + ".bias", bias, error)) {
      return false;
    }
    if (source.size() !=
        static_cast<std::size_t>(out_channels) * in_channels) {
      EncSetError(error,
                  "MiniMax H3 VisualVAE encoder 1x1 contract: " + base);
      return false;
    }
    conv->out_channels = static_cast<std::uint32_t>(out_channels);
    conv->columns = static_cast<std::uint32_t>(in_channels);
    if (!conv->weight.Allocate(source.size() * sizeof(float), error) ||
        !conv->bias.Allocate(bias.size() * sizeof(float), error) ||
        !UploadBuffer(stream, conv->weight, source, error) ||
        !UploadBuffer(stream, conv->bias, bias, error)) {
      return false;
    }
    weight_bytes += conv->weight.bytes() + conv->bias.bytes();
    return true;
  }

  bool LoadNorm(const std::string& base, std::uint32_t channels,
                EncDeviceBuffer* weight, EncDeviceBuffer* bias,
                std::string* error) {
    std::vector<float> w;
    std::vector<float> b;
    if (!LoadTensorHost(*inventory, base + ".weight", w, error) ||
        !LoadTensorHost(*inventory, base + ".bias", b, error) ||
        w.size() != channels || b.size() != channels) {
      if (error != nullptr && error->empty()) {
        EncSetError(error,
                    "MiniMax H3 VisualVAE encoder norm contract differs: " +
                        base);
      }
      return false;
    }
    if (!weight->Allocate(w.size() * sizeof(float), error) ||
        !bias->Allocate(b.size() * sizeof(float), error) ||
        !UploadBuffer(stream, *weight, w, error) ||
        !UploadBuffer(stream, *bias, b, error)) {
      return false;
    }
    weight_bytes += weight->bytes() + bias->bytes();
    return true;
  }

  bool CopyDevice(const void* source, void* destination, std::size_t bytes,
                  std::string_view phase, std::string* error) {
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMemcpyAsync(destination, source, bytes,
                                             hipMemcpyDeviceToDevice, stream);
    if (status != hipSuccess) {
      EncSetError(error, std::string("MiniMax H3 VisualVAE encoder copy ") +
                             std::string(phase) + ": " +
                             hipGetErrorString(status));
      return false;
    }
    return true;
  }

  bool Gemm(const FlatConv& conv, const float* input, float* output,
            std::size_t batch, std::string* error) {
    if (batch == 0) {
      return true;
    }
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder set GEMM stream failed");
      return false;
    }
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const rocblas_status status = rocblas_gemm_ex(
        handle_, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(conv.out_channels),
        static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(conv.columns), &alpha, conv.weight.As<float>(),
        rocblas_datatype_f32_r, static_cast<rocblas_int>(conv.columns), input,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(conv.columns), &beta,
        output, rocblas_datatype_f32_r,
        static_cast<rocblas_int>(conv.out_channels), output,
        rocblas_datatype_f32_r, static_cast<rocblas_int>(conv.out_channels),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder F32 GEMM failed: " +
                             std::to_string(static_cast<int>(status)));
      return false;
    }
    LaunchAddBias(output, conv.bias.As<float>(),
                  static_cast<std::uint32_t>(batch), conv.out_channels, stream);
    return EncCheckLaunch("gemm bias", error);
  }

  bool GroupNormSilu(float* plane, const EncDeviceBuffer& weight,
                     const EncDeviceBuffer& bias, std::uint32_t positions,
                     std::uint32_t channels, std::string* error) {
    enc_ops::LaunchGroupNormSilu(
        plane, weight.As<float>(), bias.As<float>(), gn_sums.As<float>(),
        gn_squares.As<float>(), gn_means.As<float>(), gn_inverses.As<float>(),
        plane, positions, channels, kH3VaeEncoderGroupNormGroups,
        kH3VaeEncoderGroupNormEpsilon, stream);
    return EncCheckLaunch("group norm", error);
  }

  // 3x3 conv (stride 1 or 2) with reflect handling. pad=1 is symmetric
  // reflect on all sides; pad=0 models the downsample's pre-padded right and
  // bottom only (im2col reflect mirrors exactly that for out-of-range taps).
  bool Conv3x3(const float* input, std::uint32_t in_height,
               std::uint32_t in_width, std::uint32_t channels,
               const FlatConv& conv, std::uint32_t stride, std::int32_t pad,
               float* output, std::string* error) {
    // pad=0 with stride 2 models Downsample3D: the plane is pre-padded
    // (0,1,0,1) reflect, so torch's output is (in + 1 - 3) / 2 + 1 = in / 2
    // for the even canvases this encoder accepts; im2col's reflect index
    // reproduces the pre-padding for out-of-range taps exactly.
    std::uint32_t out_height;
    std::uint32_t out_width;
    if (pad == 0 && stride == 2U) {
      out_height = in_height / 2U;
      out_width = in_width / 2U;
    } else {
      out_height = (in_height + 2U * static_cast<std::uint32_t>(pad) - 3U) /
                       stride +
                   1U;
      out_width = (in_width + 2U * static_cast<std::uint32_t>(pad) - 3U) /
                      stride +
                  1U;
    }
    const std::size_t positions =
        static_cast<std::size_t>(out_height) * out_width;
    const std::size_t columns_per_position =
        static_cast<std::size_t>(channels) * 9U;
    for (std::size_t base = 0; base < positions; base += kIm2ColChunk) {
      const std::uint32_t count = static_cast<std::uint32_t>(
          positions - base < kIm2ColChunk ? positions - base : kIm2ColChunk);
      enc_ops::LaunchIm2ColReflect(input, columns.As<float>(), in_height,
                                   in_width, out_height, out_width, channels,
                                   stride, pad, pad,
                                   static_cast<std::uint32_t>(base), count,
                                   stream);
      if (!EncCheckLaunch("im2col", error)) {
        return false;
      }
      if (!Gemm(conv, columns.As<float>(),
                output + base * conv.out_channels, count, error)) {
        return false;
      }
    }
    return true;
  }

  bool Conv1x1(const float* input, std::size_t positions,
               const FlatConv& conv, float* output, std::string* error) {
    for (std::size_t base = 0; base < positions; base += kIm2ColChunk) {
      const std::size_t count =
          positions - base < kIm2ColChunk ? positions - base : kIm2ColChunk;
      if (!Gemm(conv, input + base * conv.columns,
                output + base * conv.out_channels, count, error)) {
        return false;
      }
    }
    return true;
  }

  bool ReadPlanar(const float* device_position_major, int channels, int height,
                  int width, std::vector<float>& planar, std::string* error) {
    std::vector<float> host(static_cast<std::size_t>(height) * width *
                            channels);
    const hipError_t status = hipMemcpyAsync(
        host.data(), const_cast<float*>(device_position_major),
        host.size() * sizeof(float), hipMemcpyDeviceToHost, stream);
    if (status != hipSuccess) {
      EncSetError(error, "MiniMax H3 VisualVAE encoder readback failed");
      return false;
    }
    if (!EncSync(stream, "readback", error)) {
      return false;
    }
    TransposeToPlanar(host, channels, height, width, planar);
    return true;
  }

  bool Capture(VideoVaeEncoderTrace* trace, const char* name,
               const float* device_position_major, std::uint32_t channels,
               std::uint32_t height, std::uint32_t width, std::string* error) {
    VideoVaeEncoderStage stage;
    stage.name = name;
    stage.channels = static_cast<int>(channels);
    stage.height = static_cast<int>(height);
    stage.width = static_cast<int>(width);
    if (!ReadPlanar(device_position_major, static_cast<int>(channels),
                    static_cast<int>(height), static_cast<int>(width),
                    stage.values, error)) {
      return false;
    }
    trace->tile_height = static_cast<int>(height);
    trace->tile_width = static_cast<int>(width);
    trace->stages.push_back(std::move(stage));
    return true;
  }

  // Runs the full encoder on the uploaded position-major tile in
  // input_tile; writes planar (48, lh, lw) moments.
  bool EncodeTile(std::uint32_t height, std::uint32_t width,
                  const CancellationToken* cancellation,
                  VideoVaeEncoderTrace* trace, std::vector<float>* planar,
                  std::string* error) {
    float* a = buffer_a.As<float>();
    float* b = buffer_b.As<float>();
    float* s = buffer_s.As<float>();

    // conv_in (3 -> 128).
    if (!Conv3x3(input_tile.As<float>(), height, width, 3, conv_in, 1, 1, b,
                 error)) {
      return false;
    }
    std::swap(a, b);
    if (trace != nullptr && !Capture(trace, "conv_in", a, 128, height, width,
                                     error)) {
      return false;
    }
    std::uint32_t channels = 128;
    for (int level = 0; level < kH3VaeEncoderLevels; ++level) {
      const LevelWeights& w = levels[static_cast<std::size_t>(level)];
      for (int block = 0; block < kH3VaeEncoderResidualBlocks; ++block) {
        if (IsCancelled(cancellation)) {
          EncSetError(error, "MiniMax H3 VisualVAE encoder cancelled");
          return false;
        }
        const std::size_t bi = static_cast<std::size_t>(block);
        const std::uint32_t cin = w.block_in[bi];
        const std::uint32_t cout = w.block_out[bi];
        const std::size_t elements =
            static_cast<std::size_t>(height) * width * cin;
        if (!CopyDevice(a, s, elements * sizeof(float), "residual", error)) {
          return false;
        }
        if (!GroupNormSilu(a, w.norm_weights[bi * 2U], w.norm_biases[bi * 2U],
                           height * width, cin, error)) {
          return false;
        }
        if (!Conv3x3(a, height, width, cin, w.convs[bi * 2U], 1, 1, b, error)) {
          return false;
        }
        std::swap(a, b);
        if (!GroupNormSilu(a, w.norm_weights[bi * 2U + 1U],
                           w.norm_biases[bi * 2U + 1U], height * width, cout,
                           error)) {
          return false;
        }
        if (!Conv3x3(a, height, width, cout, w.convs[bi * 2U + 1U], 1, 1, b,
                     error)) {
          return false;
        }
        std::swap(a, b);
        if (w.has_shortcut[bi]) {
          if (!Conv1x1(s, static_cast<std::size_t>(height) * width,
                       w.shortcut[bi], b, error)) {
            return false;
          }
          enc_ops::LaunchAddInPlace(a, b,
                                    static_cast<std::size_t>(height) * width *
                                        cout,
                                    stream);
        } else {
          enc_ops::LaunchAddInPlace(a, s, elements, stream);
        }
        if (!EncCheckLaunch("residual add", error)) {
          return false;
        }
        channels = cout;
      }
      if (w.has_downsample) {
        if (!Conv3x3(a, height, width, channels, w.downsample, 2, 0, b,
                     error)) {
          return false;
        }
        std::swap(a, b);
        height = height / 2U;
        width = width / 2U;
      }
      if (trace != nullptr &&
          !Capture(trace, ("down." + std::to_string(level)).c_str(), a,
                   channels, height, width, error)) {
        return false;
      }
    }
    if (!GroupNormSilu(a, norm_out_weight, norm_out_bias, height * width, 1024,
                       error)) {
      return false;
    }
    if (trace != nullptr &&
        !Capture(trace, "norm_out", a, 1024, height, width, error)) {
      return false;
    }
    if (!Conv3x3(a, height, width, 1024, conv_out, 1, 1, b, error)) {
      return false;
    }
    std::swap(a, b);
    if (trace != nullptr && !Capture(trace, "pre_quant", a, 48, height, width,
                                     error)) {
      return false;
    }
    if (!Conv1x1(a, static_cast<std::size_t>(height) * width, quant, b,
                 error)) {
      return false;
    }
    std::swap(a, b);
    if (trace != nullptr &&
        !Capture(trace, "moments", a, 48, height, width, error)) {
      return false;
    }
    return ReadPlanar(a, 48, static_cast<int>(height), static_cast<int>(width),
                      *planar, error);
  }

  rocblas_handle handle_{nullptr};
};

VideoVaeEncoder::VideoVaeEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VideoVaeEncoder::~VideoVaeEncoder() = default;
VideoVaeEncoder::VideoVaeEncoder(VideoVaeEncoder&&) noexcept = default;
VideoVaeEncoder& VideoVaeEncoder::operator=(VideoVaeEncoder&&) noexcept =
    default;

std::unique_ptr<VideoVaeEncoder> VideoVaeEncoder::Create(
    const ModelInventory& inventory, const CancellationToken* cancellation,
    VideoVaeTelemetry* telemetry, std::string* error) {
  const auto begin = std::chrono::steady_clock::now();
  if (IsCancelled(cancellation)) {
    EncSetError(error, "MiniMax H3 VisualVAE encoder creation cancelled");
    return nullptr;
  }
  auto impl = std::make_unique<Impl>();
  impl->inventory = &inventory;
  if (rocblas_create_handle(&impl->handle_) != rocblas_status_success) {
    EncSetError(error, "MiniMax H3 VisualVAE encoder rocBLAS init failed");
    return nullptr;
  }
  (void)rocblas_set_atomics_mode(impl->handle_, rocblas_atomics_not_allowed);
  if (!impl->Load(error)) {
    return nullptr;
  }
  if (telemetry != nullptr) {
    telemetry->load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    telemetry->weight_bytes = impl->weight_bytes;
    telemetry->persistent_bytes =
        impl->buffer_a.bytes() + impl->buffer_b.bytes() +
        impl->buffer_s.bytes() + impl->columns.bytes() +
        impl->input_tile.bytes() + impl->gn_sums.bytes() +
        impl->gn_squares.bytes() + impl->gn_means.bytes() +
        impl->gn_inverses.bytes();
  }
  return std::unique_ptr<VideoVaeEncoder>(
      new VideoVaeEncoder(std::move(impl)));
}

bool VideoVaeEncoder::EncodeKeyframeMoments(
    const float* rgb_imagenet_f32, int height, int width,
    const CancellationToken* cancellation, std::vector<float>* moments,
    std::string* error) {
  return EncodeKeyframeMoments(rgb_imagenet_f32, height, width, cancellation,
                               moments, nullptr, nullptr, nullptr, nullptr,
                               error);
}

bool VideoVaeEncoder::EncodeKeyframeMoments(
    const float* rgb_imagenet_f32, int height, int width,
    const CancellationToken* cancellation, std::vector<float>* moments,
    VideoVaeEncoderTrace* trace, VideoVaeProgress progress,
    void* progress_opaque, VideoVaeTelemetry* telemetry, std::string* error) {
  if (moments == nullptr || impl_ == nullptr || rgb_imagenet_f32 == nullptr) {
    EncSetError(error, "invalid MiniMax H3 VisualVAE encoder request");
    return false;
  }
  auto plan = ResolveVideoVaeEncoderTilePlan(height, width, error);
  if (!plan) {
    return false;
  }
  if (trace != nullptr) {
    trace->stages.clear();
  }
  const int rows = static_cast<int>(plan->vertical.starts.size());
  const int columns_count = static_cast<int>(plan->horizontal.starts.size());
  std::vector<std::vector<float>> tile_moments(
      static_cast<std::size_t>(plan->tiles));
  int completed = 0;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns_count; ++column) {
      const auto tile_begin = std::chrono::steady_clock::now();
      if (IsCancelled(cancellation)) {
        EncSetError(error, "MiniMax H3 VisualVAE encoder cancelled");
        return false;
      }
      const int top = plan->vertical.starts[static_cast<std::size_t>(row)];
      const int left = plan->horizontal.starts[static_cast<std::size_t>(column)];
      const int tile_height = row + 1 < rows ? plan->vertical.length
                                             : height - top;
      const int tile_width = column + 1 < columns_count
                                 ? plan->horizontal.length
                                 : width - left;
      // Gather the planar canvas tile into position-major and upload.
      std::vector<float> gathered(
          static_cast<std::size_t>(tile_height) * tile_width * 3U);
      for (int y = 0; y < tile_height; ++y) {
        for (int x = 0; x < tile_width; ++x) {
          const std::size_t source =
              static_cast<std::size_t>(top + y) * width + (left + x);
          const std::size_t destination =
              (static_cast<std::size_t>(y) * tile_width + x) * 3U;
          gathered[destination] = rgb_imagenet_f32[source];
          gathered[destination + 1U] =
              rgb_imagenet_f32[static_cast<std::size_t>(height) * width +
                               source];
          gathered[destination + 2U] =
              rgb_imagenet_f32[2U * static_cast<std::size_t>(height) * width +
                               source];
        }
      }
      const hipError_t status = hipMemcpyAsync(
          impl_->input_tile.data(), gathered.data(),
          gathered.size() * sizeof(float), hipMemcpyHostToDevice,
          impl_->stream);
      if (status != hipSuccess ||
          !EncSync(impl_->stream, "tile upload", error)) {
        if (status != hipSuccess && error != nullptr && error->empty()) {
          EncSetError(error, "MiniMax H3 VisualVAE encoder tile upload "
                             "failed");
        }
        return false;
      }
      if (!impl_->EncodeTile(static_cast<std::uint32_t>(tile_height),
                             static_cast<std::uint32_t>(tile_width),
                             cancellation,
                             trace != nullptr && completed == 0 ? trace
                                                                : nullptr,
                             &tile_moments[static_cast<std::size_t>(
                                 row * columns_count + column)],
                             error)) {
        return false;
      }
      ++completed;
      if (telemetry != nullptr) {
        telemetry->tile_ms.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tile_begin)
                .count());
      }
      if (progress != nullptr) {
        progress(completed, plan->tiles, progress_opaque);
      }
    }
  }
  if (!StitchVideoVaeEncoderTiles(*plan, tile_moments, moments, error)) {
    return false;
  }
  for (const float value : *moments) {
    if (!std::isfinite(value)) {
      if (telemetry != nullptr && telemetry->first_non_finite.empty()) {
        telemetry->first_non_finite = "moments";
      }
      EncSetError(error, "MiniMax H3 VisualVAE encoder produced non-finite "
                         "moments");
      return false;
    }
  }
  return true;
}

std::uint64_t VideoVaeEncoder::weight_bytes() const noexcept {
  return impl_ == nullptr ? 0 : impl_->weight_bytes;
}

}  // namespace gufo::minimax_h3

#endif  // ENGINE_ENABLE_HIP
