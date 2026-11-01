#include <hipblas/hipblas.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gufo/ext/experimental/hip_utils.hpp>
#include "src/models/embedding_gemma2/hip/audio_runtime.hpp"
#include "src/models/embedding_gemma2/loader.hpp"

namespace gufo::models::embedding_gemma2::hip {
namespace {
constexpr std::size_t kMaxFrames = 2999;
constexpr std::size_t kMaxTokens = (kMaxFrames + 3) / 4;
struct Buffer {
  float* data{nullptr};
  Buffer() = default;
  explicit Buffer(std::size_t count) {
    HIP_CHECK(hipMalloc(&data, count * sizeof(float)));
  }
  ~Buffer() {
    if (data)
      gufo::hip::LogCleanupError(hipFree(data));
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : data(std::exchange(other.data, nullptr)) {}
};
__device__ float Sum(float v) {
  for (int d = 16; d; d >>= 1)
    v += __shfl_xor(v, d, 32);
  return v;
}
// One wave per row. LayerNorm is used only by the two subsampling blocks;
// all other norms use RMS and the checkpoint scale directly (no +1).
__global__ void Norm(const float* x, const float* weight, float* y, int width,
                     bool centered, bool clipped = false) {
  const int row = blockIdx.x, lane = threadIdx.x;
  float mean = 0, variance = 0;
  if (centered) {
    for (int c = lane; c < width; c += 32)
      mean += x[row * width + c];
    mean = Sum(mean) / width;
  }
  for (int c = lane; c < width; c += 32) {
    const float raw = x[row * width + c];
    const float v = (clipped ? fminf(fmaxf(raw, -1e10F), 1e10F) : raw) - mean;
    variance += v * v;
  }
  const float scale = rsqrtf(Sum(variance) / width + 1e-6F);
  for (int c = lane; c < width; c += 32) {
    const float raw = x[row * width + c];
    float v =
        ((clipped ? fminf(fmaxf(raw, -1e10F), 1e10F) : raw) - mean) * scale;
    if (weight)
      v *= weight[c];
    y[row * width + c] = centered ? fmaxf(v, 0.0F) : v;
  }
}
__global__ void Activate(float* x, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = x[i] / (1 + expf(-x[i]));
}
__global__ void Residual(float* state, const float* update, int count,
                         float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    state[i] += update[i] * scale;
}
__global__ void Clip(const float* x, float* y, int count, float low,
                     float high) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    y[i] = fminf(fmaxf(x[i], low), high);
}
// Subsample convolution in channels-last layout. Mask gates each input time
// row; layer1 uses the already subsampled [::2] input mask.
__global__ void Conv2d(const float* x, const float* w, float* y, int frames,
                       int bins, int in_channels, int out_channels,
                       int valid_frames) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_bins = (bins + 1) / 2, out_frames = (frames + 1) / 2;
  if (i >= out_frames * out_bins * out_channels)
    return;
  const int oc = i % out_channels;
  const int f = (i / out_channels) % out_bins, t = i / out_channels / out_bins;
  float sum = 0;
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int dt = 0; dt < 3; ++dt) {
      const int time = t * 2 + dt - 1;
      if (time < 0 || time >= frames || time >= valid_frames)
        continue;
      for (int df = 0; df < 3; ++df) {
        const int bin = f * 2 + df - 1;
        if (bin >= 0 && bin < bins)
          sum += x[(time * bins + bin) * in_channels + ic] *
                 w[((oc * in_channels + ic) * 3 + dt) * 3 + df];
      }
    }
  }
  y[i] = sum;
}
__global__ void Glu(const float* x, float* y, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * 1024) {
    const int base = (i / 1024) * 2048 + i % 1024;
    y[i] = x[base] / (1 + expf(-x[base + 1024]));
  }
}
__global__ void Depthwise(const float* x, const float* w, float* y, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 1024)
    return;
  const int t = i / 1024, c = i % 1024;
  float sum = 0;
  for (int tap = 0; tap < 5; ++tap)
    if (t + tap >= 4)
      sum += x[(t + tap - 4) * 1024 + c] * w[c * 5 + tap];
  y[i] = sum;
}
__global__ void Attention(const float* q, const float* k, const float* v,
                          const float* relative, const float* dim_scale,
                          float* output, int rows, int valid) {
  const int t = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int chunk_start = (t / 12) * 12;
  float scores[24];
  float maximum = -1e9F;
  const float qscale = (1 / sqrtf(128.0F)) / logf(2.0F);
  const float kscale = logf(1 + expf(1.0F)) / logf(2.0F);
  float query[4];
  for (int j = 0; j < 4; ++j) {
    const int d = lane + j * 32;
    query[j] =
        q[t * 1024 + head * 128 + d] * qscale * log1pf(expf(dim_scale[d]));
  }
  for (int context = 0; context < 24; ++context) {
    const int key_t = chunk_start - 12 + context;
    float dot = 0;
    // _rel_shift flattens [12,25] then reshapes [12,24]. For
    // query position r the relative index is context-r. Only 0..12
    // survives padding; the causal 12-left mask ensures exactly that range.
    const int rel = context - t % 12;
    if (key_t >= 0 && key_t < rows) {
      for (int j = 0; j < 4; ++j) {
        const int d = head * 128 + lane + j * 32;
        dot += query[j] * k[key_t * 1024 + d] * kscale;
      }
    }
    if (rel >= 0 && rel <= 12) {
      for (int j = 0; j < 4; ++j)
        dot += query[j] * relative[rel * 1024 + head * 128 + lane + j * 32];
    }
    dot = Sum(dot);
    const bool allowed =
        key_t >= 0 && key_t < valid && key_t <= t && key_t > t - 12;
    scores[context] = allowed ? 50 * tanhf(dot / 50) : -1e9F;
    maximum = fmaxf(maximum, scores[context]);
  }
  float denom = 0;
  for (int j = 0; j < 24; ++j) {
    scores[j] = expf(scores[j] - maximum);
    denom += scores[j];
  }
  for (int j = 0; j < 4; ++j) {
    float sum = 0;
    for (int context = 0; context < 24; ++context) {
      const int key_t = chunk_start - 12 + context;
      if (key_t >= 0 && key_t < rows)
        sum += scores[context] / denom *
               v[key_t * 1024 + head * 128 + lane + j * 32];
    }
    output[t * 1024 + head * 128 + lane + j * 32] = sum;
  }
}
__global__ void Bias(float* x, const float* bias, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * 1536)
    x[i] += bias[i % 1536];
}
void BlasCheck(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("embedding_gemma2: audio hipBLAS failed: " +
                             std::to_string(status));
}
}  // namespace

struct AudioRuntime::Impl {
  hipStream_t stream;
  hipblasHandle_t blas{nullptr};
  std::map<std::string, Buffer> weights;
  std::map<std::string, float> scalars;
  Buffer input{kMaxFrames * 128}, conv0{1500 * 64 * 128},
      conv1{kMaxTokens * 32 * 32};
  Buffer state{kMaxTokens * 1024}, normed{kMaxTokens * 4096},
      temp{kMaxTokens * 4096};
  Buffer update{kMaxTokens * 2048}, q{kMaxTokens * 1024}, k{kMaxTokens * 1024},
      v{kMaxTokens * 1024};
  Buffer context{kMaxTokens * 1024}, relative{13 * 1024}, positions{13 * 1024};
  Buffer projected{kMaxTokens * 1536}, features{kMaxTokens * 512};
  ~Impl() {
    if (blas)
      gufo::hip::LogCleanupError(hipblasDestroy(blas));
  }
  explicit Impl(hipStream_t supplied) : stream(supplied) {}
  const float* W(const std::string& name) const {
    return weights.at(name).data;
  }
  void Gemm(const std::string& name, const float* x, float* y, int n, int m,
            int width) {
    constexpr float alpha = 1, beta = 0;
    BlasCheck(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, width,
                           &alpha, W(name), width, x, width, &beta, y, m));
  }
  void Linear(const std::string& name, const float* x, float* y, int n, int m,
              int width) {
    Clip<<<(n * width + 255) / 256, 256, 0, stream>>>(
        x, normed.data, n * width, scalars.at(name + ".input_min"),
        scalars.at(name + ".input_max"));
    Gemm(name + ".linear.weight", normed.data, y, n, m, width);
    Clip<<<(n * m + 255) / 256, 256, 0, stream>>>(
        y, y, n * m, scalars.at(name + ".output_min"),
        scalars.at(name + ".output_max"));
  }
  void Normalize(const float* x, const std::string& weight, float* y, int n,
                 int width = 1024, bool clipped = false) {
    Norm<<<n, 32, 0, stream>>>(x, weight.empty() ? nullptr : W(weight), y,
                               width, false, clipped);
  }
  void Add(const float* x, int n, float scale = 1) {
    Residual<<<(n * 1024 + 255) / 256, 256, 0, stream>>>(state.data, x,
                                                         n * 1024, scale);
  }
  void FeedForward(const std::string& p, int n) {
    Normalize(state.data, p + "pre_layer_norm.weight", update.data, n, 1024,
              true);
    Linear(p + "ffw_layer_1", update.data, temp.data, n, 4096, 1024);
    Activate<<<(n * 4096 + 255) / 256, 256, 0, stream>>>(temp.data, n * 4096);
    Linear(p + "ffw_layer_2", temp.data, update.data, n, 1024, 4096);
    Normalize(update.data, p + "post_layer_norm.weight", update.data, n, 1024,
              true);
    Add(update.data, n, 0.5F);
  }
};

AudioRuntime::AudioRuntime(const std::string& model_dir, hipStream_t stream)
    : impl_(std::make_unique<Impl>(stream)) {
  auto loaded = LoadModelDirectory(model_dir, false, true);
  if (!loaded.ok)
    throw std::runtime_error(loaded.error);
  auto& a = *impl_;
  BlasCheck(hipblasCreate(&a.blas));
  BlasCheck(hipblasSetStream(a.blas, stream));
  for (const auto& expected : ExpectedAudioTensors()) {
    const auto* tensor = loaded.store->Find(expected.name);
    const auto count = *tensor->NumElements();
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t bits;
      std::memcpy(&bits, tensor->data + i * 2, 2);
      values[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
    }
    if (expected.shape.empty()) {
      a.scalars.emplace(expected.name, values[0]);
    } else {
      auto [it, inserted] = a.weights.try_emplace(expected.name, count);
      (void)inserted;
      HIP_CHECK(hipMemcpy(it->second.data, values.data(), count * sizeof(float),
                          hipMemcpyHostToDevice));
    }
  }
  std::vector<float> positions(13 * 1024);
  for (int t = 0; t < 13; ++t) {
    for (int d = 0; d < 512; ++d) {
      // Official inv_timescales and multiplication are float32.
      const float inv = std::exp(-static_cast<float>(d) *
                                 static_cast<float>(std::log(10000.0) / 511));
      const float angle = (12 - t) * inv;
      positions[t * 1024 + d] = std::sin(angle);
      positions[t * 1024 + d + 512] = std::cos(angle);
    }
  }
  HIP_CHECK(hipMemcpy(a.positions.data, positions.data(),
                      positions.size() * sizeof(float), hipMemcpyHostToDevice));
}
AudioRuntime::~AudioRuntime() = default;

const float* AudioRuntime::Run(const AudioFeatures& features,
                               const std::function<bool()>& is_cancelled) {
  auto& a = *impl_;
  const int frames = static_cast<int>(features.mask.size());
  const int valid_frames =
      static_cast<int>(std::ranges::count(features.mask, 1));
  const int n0 = (frames + 1) / 2, n = (n0 + 1) / 2;
  const int valid = static_cast<int>(features.soft_tokens());
  if (frames <= 0 || frames > static_cast<int>(kMaxFrames) || valid <= 0 ||
      features.values.size() != static_cast<std::size_t>(frames) * 128)
    throw std::invalid_argument(
        "embedding_gemma2: invalid audio feature shape");
  HIP_CHECK(hipMemcpyAsync(a.input.data, features.values.data(),
                           features.values.size() * sizeof(float),
                           hipMemcpyHostToDevice, a.stream));
  const auto* s = "audio_tower.subsample_conv_projection.";
  Conv2d<<<(n0 * 64 * 128 + 255) / 256, 256, 0, a.stream>>>(
      a.input.data, a.W(std::string(s) + "layer0.conv.weight"), a.conv0.data,
      frames, 128, 1, 128, valid_frames);
  Norm<<<n0 * 64, 32, 0, a.stream>>>(a.conv0.data,
                                     a.W(std::string(s) + "layer0.norm.weight"),
                                     a.conv0.data, 128, true);
  Conv2d<<<(n * 32 * 32 + 255) / 256, 256, 0, a.stream>>>(
      a.conv0.data, a.W(std::string(s) + "layer1.conv.weight"), a.conv1.data,
      n0, 64, 128, 32, (valid_frames + 1) / 2);
  Norm<<<n * 32, 32, 0, a.stream>>>(a.conv1.data,
                                    a.W(std::string(s) + "layer1.norm.weight"),
                                    a.conv1.data, 32, true);
  a.Gemm(std::string(s) + "input_proj_linear.weight", a.conv1.data,
         a.state.data, n, 1024, 1024);
  for (int layer = 0; layer < 12; ++layer) {
    const std::string p = "audio_tower.layers." + std::to_string(layer) + ".";
    a.FeedForward(p + "feed_forward1.", n);
    a.Normalize(a.state.data, p + "norm_pre_attn.weight", a.update.data, n,
                1024, true);
    a.Linear(p + "self_attn.q_proj", a.update.data, a.q.data, n, 1024, 1024);
    a.Linear(p + "self_attn.k_proj", a.update.data, a.k.data, n, 1024, 1024);
    a.Linear(p + "self_attn.v_proj", a.update.data, a.v.data, n, 1024, 1024);
    a.Gemm(p + "self_attn.relative_k_proj.weight", a.positions.data,
           a.relative.data, 13, 1024, 1024);
    Attention<<<dim3(n, 8), 32, 0, a.stream>>>(
        a.q.data, a.k.data, a.v.data, a.relative.data,
        a.W(p + "self_attn.per_dim_scale"), a.context.data, n, valid);
    a.Linear(p + "self_attn.post", a.context.data, a.update.data, n, 1024,
             1024);
    a.Normalize(a.update.data, p + "norm_post_attn.weight", a.update.data, n,
                1024, true);
    a.Add(a.update.data, n);
    a.Normalize(a.state.data, p + "lconv1d.pre_layer_norm.weight",
                a.update.data, n);
    a.Linear(p + "lconv1d.linear_start", a.update.data, a.temp.data, n, 2048,
             1024);
    Glu<<<(n * 1024 + 255) / 256, 256, 0, a.stream>>>(a.temp.data,
                                                      a.context.data, n);
    Depthwise<<<(n * 1024 + 255) / 256, 256, 0, a.stream>>>(
        a.context.data, a.W(p + "lconv1d.depthwise_conv1d.weight"),
        a.update.data, n);
    a.Normalize(a.update.data, p + "lconv1d.conv_norm.weight", a.update.data, n,
                1024, true);
    Activate<<<(n * 1024 + 255) / 256, 256, 0, a.stream>>>(a.update.data,
                                                           n * 1024);
    a.Linear(p + "lconv1d.linear_end", a.update.data, a.temp.data, n, 1024,
             1024);
    a.Add(a.temp.data, n);
    a.FeedForward(p + "feed_forward2.", n);
    a.Normalize(a.state.data, p + "norm_out.weight", a.state.data, n, 1024,
                true);
    if (is_cancelled) {
      HIP_CHECK(hipStreamSynchronize(a.stream));
      if (is_cancelled())
        throw std::runtime_error("audio encoding cancelled");
    }
  }
  a.Gemm("audio_tower.output_proj.weight", a.state.data, a.projected.data, n,
         1536, 1024);
  Bias<<<(n * 1536 + 255) / 256, 256, 0, a.stream>>>(
      a.projected.data, a.W("audio_tower.output_proj.bias"), n);
  a.Normalize(a.projected.data, "", a.normed.data, valid, 1536);
  a.Gemm("embed_audio.embedding_projection.weight", a.normed.data,
         a.features.data, valid, 512, 1536);
  HIP_CHECK(hipGetLastError());
  return a.features.data;
}
const float* AudioRuntime::encoded_states() const {
  return impl_->projected.data;
}
}  // namespace gufo::models::embedding_gemma2::hip
