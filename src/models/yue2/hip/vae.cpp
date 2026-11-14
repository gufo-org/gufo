#include "src/models/yue2/hip/vae.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/models/yue2/hip/kernels.hpp"
#include "src/models/yue2/safetensors.hpp"

namespace gufo::models::yue2::hip {
namespace {

// Widest convolution margin: kernel 7 at dilation 9.
constexpr std::size_t kPad = 27;

void RequireHip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess)
    throw std::runtime_error("YuE2 VAE " + std::string(operation) + ": " +
                             hipGetErrorString(status));
}

void RequireBlas(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("YuE2 VAE hipBLAS failed: status " +
                             std::to_string(status));
}

class DeviceFloats {
public:
  DeviceFloats() = default;
  explicit DeviceFloats(std::span<const float> host) { Upload(host); }
  ~DeviceFloats() {
    if (data_ != nullptr)
      (void)hipFree(data_);
  }
  DeviceFloats(DeviceFloats&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  DeviceFloats& operator=(DeviceFloats&& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
    return *this;
  }
  void Reserve(std::size_t count) {
    if (count <= count_)
      return;
    if (data_ != nullptr)
      (void)hipFree(data_);
    data_ = nullptr;
    RequireHip(
        hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(float)),
        "hipMalloc");
    count_ = count;
  }
  void Upload(std::span<const float> host) {
    Reserve(host.size());
    RequireHip(hipMemcpy(data_, host.data(), host.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "hipMemcpy");
  }
  [[nodiscard]] float* get() const { return data_; }

private:
  float* data_{nullptr};
  std::size_t count_{0};
};

struct Snake {
  DeviceFloats alpha;
  DeviceFloats inverse_beta;
};

// Conv1d with weight norm folded: `taps` holds one `[out][in]` matrix per tap.
struct Conv {
  std::size_t inputs{0}, outputs{0}, kernel{0}, dilation{1};
  DeviceFloats taps;
  DeviceFloats bias;
};

// ConvTranspose1d as `stride` phases: `phases` holds per phase r the
// `[out][in]` matrices of taps r and r + stride.
struct Transpose {
  std::size_t inputs{0}, outputs{0}, stride{0}, padding{0};
  DeviceFloats phases;
  DeviceFloats bias;
};

struct Residual {
  Snake first;
  Conv dilated;
  Snake second;
  Conv pointwise;
};

struct Block {
  Snake snake;
  Transpose upsample;
  std::array<Residual, 3> residuals;
};

}  // namespace

struct VaeDecoder::Impl {
  VaeConfig config;
  std::unique_ptr<SafetensorsFile> file;
  Conv input;
  std::array<Block, 6> blocks;
  Snake final_snake;
  Conv output;
  hipStream_t stream{nullptr};
  hipblasHandle_t blas{nullptr};
  DeviceFloats stream_a, stream_b, work, temp, tile_audio;

  ~Impl() {
    if (blas != nullptr)
      (void)hipblasDestroy(blas);
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
  }

  std::vector<float> Read(const std::string& name,
                          std::initializer_list<std::uint64_t> shape) {
    const std::vector<std::uint64_t> dims(shape);
    return file->ReadFloat(file->Require(name, "F32", dims));
  }

  Snake LoadSnake(const std::string& name, std::size_t channels) {
    std::vector<float> alpha = Read(name + ".alpha", {channels});
    std::vector<float> beta = Read(name + ".beta", {channels});
    for (std::size_t c = 0; c < channels; ++c) {
      alpha[c] = std::exp(alpha[c]);
      beta[c] = 1.0F / (std::exp(beta[c]) + 0.000000001F);
    }
    return {DeviceFloats(alpha), DeviceFloats(beta)};
  }

  Conv LoadConv(const std::string& name, std::size_t inputs,
                std::size_t outputs, std::size_t kernel, std::size_t dilation,
                bool has_bias) {
    const std::vector<float> g = Read(name + ".weight_g", {outputs, 1, 1});
    const std::vector<float> v =
        Read(name + ".weight_v", {outputs, inputs, kernel});
    std::vector<float> taps(kernel * outputs * inputs);
    for (std::size_t o = 0; o < outputs; ++o) {
      double norm = 0.0;
      for (std::size_t i = 0; i < inputs * kernel; ++i)
        norm += static_cast<double>(v[o * inputs * kernel + i]) *
                v[o * inputs * kernel + i];
      const float scale = g[o] / static_cast<float>(std::sqrt(norm));
      for (std::size_t i = 0; i < inputs; ++i)
        for (std::size_t k = 0; k < kernel; ++k)
          taps[(k * outputs + o) * inputs + i] =
              v[(o * inputs + i) * kernel + k] * scale;
    }
    Conv conv;
    conv.inputs = inputs;
    conv.outputs = outputs;
    conv.kernel = kernel;
    conv.dilation = dilation;
    conv.taps.Upload(taps);
    if (has_bias)
      conv.bias.Upload(Read(name + ".bias", {outputs}));
    return conv;
  }

  Transpose LoadTranspose(const std::string& name, std::size_t inputs,
                          std::size_t outputs, std::size_t stride) {
    const std::size_t kernel = 2 * stride;
    const std::vector<float> g = Read(name + ".weight_g", {inputs, 1, 1});
    const std::vector<float> v =
        Read(name + ".weight_v", {inputs, outputs, kernel});
    // Transposed weight norm normalizes per input channel.
    std::vector<float> folded(v.size());
    for (std::size_t i = 0; i < inputs; ++i) {
      double norm = 0.0;
      for (std::size_t j = 0; j < outputs * kernel; ++j)
        norm += static_cast<double>(v[i * outputs * kernel + j]) *
                v[i * outputs * kernel + j];
      const float scale = g[i] / static_cast<float>(std::sqrt(norm));
      for (std::size_t j = 0; j < outputs * kernel; ++j)
        folded[i * outputs * kernel + j] = v[i * outputs * kernel + j] * scale;
    }
    // phases[r][m][o][i] = W[i][o][r + m * stride].
    std::vector<float> phases(stride * 2 * outputs * inputs);
    for (std::size_t r = 0; r < stride; ++r)
      for (std::size_t m = 0; m < 2; ++m)
        for (std::size_t o = 0; o < outputs; ++o)
          for (std::size_t i = 0; i < inputs; ++i)
            phases[((r * 2 + m) * outputs + o) * inputs + i] =
                folded[(i * outputs + o) * kernel + r + m * stride];
    Transpose t;
    t.inputs = inputs;
    t.outputs = outputs;
    t.stride = stride;
    t.padding = (stride + 1) / 2;
    t.phases.Upload(phases);
    t.bias.Upload(Read(name + ".bias", {outputs}));
    return t;
  }

  void Load() {
    const int base = config.base_channels;
    const auto channels = [&](int index) {
      return static_cast<std::size_t>(config.multipliers[index] * base);
    };
    input = LoadConv("decoder.layers.0", config.latent_dim, channels(6), 7, 1,
                     true);
    for (int b = 0; b < 6; ++b) {
      const int level = 6 - b;
      const std::size_t in = channels(level);
      const std::size_t out = channels(level - 1);
      const std::string name = "decoder.layers." + std::to_string(b + 1);
      Block& block = blocks[b];
      block.snake = LoadSnake(name + ".layers.0", in);
      block.upsample =
          LoadTranspose(name + ".layers.1", in, out,
                        static_cast<std::size_t>(config.strides[level - 1]));
      constexpr std::array<std::size_t, 3> kDilations{1, 3, 9};
      for (int r = 0; r < 3; ++r) {
        const std::string unit =
            name + ".layers." + std::to_string(r + 2) + ".layers.";
        Residual& residual = block.residuals[r];
        residual.first = LoadSnake(unit + "0", out);
        residual.dilated =
            LoadConv(unit + "1", out, out, 7, kDilations[r], true);
        residual.second = LoadSnake(unit + "2", out);
        residual.pointwise = LoadConv(unit + "3", out, out, 1, 1, true);
      }
    }
    final_snake = LoadSnake("decoder.layers.7", channels(0));
    output = LoadConv("decoder.layers.8", channels(0),
                      static_cast<std::size_t>(config.channels), 7, 1, false);
  }

  // out[o][t] (+)= sum_k W_k x[t + (k - kernel/2) * dilation].
  void Convolve(const Conv& conv, const float* x, std::size_t length,
                float* out, bool accumulate) {
    const float one = 1.0F, zero = 0.0F;
    const std::size_t pitch = length + 2 * kPad;
    const std::ptrdiff_t half = static_cast<std::ptrdiff_t>(conv.kernel / 2);
    for (std::size_t k = 0; k < conv.kernel; ++k) {
      const std::ptrdiff_t shift = (static_cast<std::ptrdiff_t>(k) - half) *
                                   static_cast<std::ptrdiff_t>(conv.dilation);
      RequireBlas(hipblasSgemm(
          blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(length),
          static_cast<int>(conv.outputs), static_cast<int>(conv.inputs), &one,
          x + kPad + shift, static_cast<int>(pitch),
          conv.taps.get() + k * conv.outputs * conv.inputs,
          static_cast<int>(conv.inputs), accumulate || k > 0 ? &one : &zero,
          out + kPad, static_cast<int>(pitch)));
    }
  }

  // Polyphase transposed convolution into `out` (length `out_length`).
  void Upsample(const Transpose& t, const float* x, std::size_t length,
                float* out, std::size_t out_length) {
    const float one = 1.0F, zero = 0.0F;
    const std::size_t pitch = length + 2 * kPad;
    const std::size_t phase_length = length + 1;
    for (std::size_t r = 0; r < t.stride; ++r) {
      float* phase = temp.get() + r * t.outputs * phase_length;
      for (std::size_t m = 0; m < 2; ++m) {
        RequireBlas(hipblasSgemm(
            blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(phase_length),
            static_cast<int>(t.outputs), static_cast<int>(t.inputs), &one,
            x + kPad - static_cast<std::ptrdiff_t>(m), static_cast<int>(pitch),
            t.phases.get() + (r * 2 + m) * t.outputs * t.inputs,
            static_cast<int>(t.inputs), m > 0 ? &one : &zero, phase,
            static_cast<int>(phase_length)));
      }
    }
    LaunchInterleavePhases(temp.get(), phase_length, t.stride, t.padding,
                           t.outputs, out_length, kPad, t.bias.get(), out,
                           stream);
  }

  /// Decodes one latent tile already in `stream_a` (`[64][frames]`); the
  /// stereo result is left in `stream_a`. Returns the output length.
  std::size_t DecodeTile(std::size_t frames) {
    float* x = stream_a.get();
    float* y = stream_b.get();
    std::size_t length = frames;
    LaunchFillBias(y, input.outputs, length, kPad, input.bias.get(), stream);
    Convolve(input, x, length, y, true);
    std::swap(x, y);
    for (const Block& block : blocks) {
      const Transpose& up = block.upsample;
      LaunchSnake(x, work.get(), up.inputs, length, kPad,
                  block.snake.alpha.get(), block.snake.inverse_beta.get(),
                  stream);
      const std::size_t out_length =
          (length - 1) * up.stride - 2 * up.padding + 2 * up.stride;
      Upsample(up, work.get(), length, y, out_length);
      std::swap(x, y);
      length = out_length;
      const std::size_t channels = up.outputs;
      for (const Residual& residual : block.residuals) {
        LaunchSnake(x, work.get(), channels, length, kPad,
                    residual.first.alpha.get(),
                    residual.first.inverse_beta.get(), stream);
        LaunchFillBias(y, channels, length, kPad, residual.dilated.bias.get(),
                       stream);
        Convolve(residual.dilated, work.get(), length, y, true);
        LaunchSnake(y, work.get(), channels, length, kPad,
                    residual.second.alpha.get(),
                    residual.second.inverse_beta.get(), stream);
        LaunchAddBias(x, channels, length, kPad, residual.pointwise.bias.get(),
                      stream);
        Convolve(residual.pointwise, work.get(), length, x, true);
      }
    }
    LaunchSnake(x, work.get(), output.inputs, length, kPad,
                final_snake.alpha.get(), final_snake.inverse_beta.get(),
                stream);
    LaunchFillBias(y, output.outputs, length, kPad, nullptr, stream);
    Convolve(output, work.get(), length, y, true);
    if (y != stream_a.get())
      std::swap(stream_a, stream_b);
    return length;
  }

  void Reserve(std::size_t frames) {
    // Largest stage: `channels * (length + 2 * pad)` for each activation and
    // `stride * channels * (input + 1)` for the phases.
    std::size_t length = frames;
    std::size_t activations = input.outputs * (length + 2 * kPad);
    std::size_t phases = 0;
    for (const Block& block : blocks) {
      const Transpose& up = block.upsample;
      phases = std::max(phases, up.stride * up.outputs * (length + 1));
      activations = std::max(activations, up.inputs * (length + 2 * kPad));
      length = (length - 1) * up.stride - 2 * up.padding + 2 * up.stride;
      activations = std::max(activations, up.outputs * (length + 2 * kPad));
    }
    stream_a.Reserve(activations);
    stream_b.Reserve(activations);
    work.Reserve(activations);
    temp.Reserve(phases);
  }
};

VaeDecoder::VaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
VaeDecoder::~VaeDecoder() = default;

const VaeConfig& VaeDecoder::config() const {
  return impl_->config;
}

std::unique_ptr<VaeDecoder> VaeDecoder::Create(
    const std::filesystem::path& vae_root, std::string* error) {
  try {
    auto impl = std::make_unique<Impl>();
    if (!LoadVaeConfig(vae_root, &impl->config, error))
      return nullptr;
    impl->file = SafetensorsFile::Open(vae_root / "model.safetensors", error);
    if (!impl->file)
      return nullptr;
    RequireHip(hipStreamCreateWithFlags(&impl->stream, hipStreamNonBlocking),
               "hipStreamCreateWithFlags");
    RequireBlas(hipblasCreate(&impl->blas));
    RequireBlas(hipblasSetStream(impl->blas, impl->stream));
    impl->Load();
    impl->file.reset();
    return std::unique_ptr<VaeDecoder>(new VaeDecoder(std::move(impl)));
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = exception.what();
    return nullptr;
  }
}

bool VaeDecoder::Decode(std::span<const float> latents,
                        const TileProgress& progress, std::vector<float>* audio,
                        std::string* error) {
  auto& impl = *impl_;
  const auto& c = impl.config;
  const std::size_t dim = static_cast<std::size_t>(c.latent_dim);
  const std::size_t frames = latents.size() / dim;
  if (frames == 0 || latents.size() % dim != 0) {
    if (error != nullptr)
      *error = "YuE2 VAE needs [frames][64] latents";
    return false;
  }
  try {
    const std::size_t ratio = 1920;
    const std::size_t core = static_cast<std::size_t>(c.core_frames);
    const std::size_t halo = static_cast<std::size_t>(c.halo_frames);
    const std::size_t total = ratio * frames - 64;
    const std::size_t tiles = (frames + core - 1) / core;
    audio->assign(2 * total, 0.0F);
    impl.Reserve(std::min(frames, core + 2 * halo));
    impl.tile_audio.Reserve(2 * core * ratio);
    std::vector<float> channel_major;
    for (std::size_t tile = 0; tile < tiles; ++tile) {
      const std::size_t start = tile * core;
      const std::size_t end = std::min(frames, start + core);
      const std::size_t left = start > halo ? start - halo : 0;
      const std::size_t right = std::min(frames, end + halo);
      const std::size_t length = right - left;
      // [frames][64] -> padded [64][length].
      const std::size_t pitch = length + 2 * kPad;
      channel_major.assign(dim * pitch, 0.0F);
      for (std::size_t t = 0; t < length; ++t)
        for (std::size_t d = 0; d < dim; ++d)
          channel_major[d * pitch + kPad + t] = latents[(left + t) * dim + d];
      RequireHip(hipMemcpyAsync(impl.stream_a.get(), channel_major.data(),
                                channel_major.size() * sizeof(float),
                                hipMemcpyHostToDevice, impl.stream),
                 "hipMemcpyAsync latents");
      const std::size_t out_length = impl.DecodeTile(length);
      const std::size_t out_start = start * ratio;
      const std::size_t out_end = std::min(end * ratio, total);
      const std::size_t crop = (start - left) * ratio;
      if (crop + (out_end - out_start) > out_length)
        throw std::runtime_error("YuE2 VAE tile did not cover its core");
      LaunchInterleaveStereo(impl.stream_a.get(), out_length, kPad, crop,
                             out_end - out_start, impl.tile_audio.get(),
                             impl.stream);
      RequireHip(
          hipMemcpyAsync(audio->data() + 2 * out_start, impl.tile_audio.get(),
                         2 * (out_end - out_start) * sizeof(float),
                         hipMemcpyDeviceToHost, impl.stream),
          "hipMemcpyAsync audio");
      RequireHip(hipStreamSynchronize(impl.stream), "tile");
      RequireHip(hipGetLastError(), "tile launch");
      if (progress &&
          !progress(static_cast<int>(tile + 1), static_cast<int>(tiles))) {
        if (error != nullptr)
          *error = "cancelled";
        return false;
      }
    }
    if (!std::ranges::all_of(
            *audio, [](float value) { return std::isfinite(value); })) {
      if (error != nullptr)
        *error = "YuE2 VAE produced non-finite audio";
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    if (error != nullptr)
      *error = exception.what();
    return false;
  }
}

}  // namespace gufo::models::yue2::hip
