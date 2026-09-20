#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "src/models/qwen_image_21/hip/runtime.hpp"

namespace gufo::models::qwen_image_21::hip {
namespace {

using B = hip_bfloat16;

__device__ float Round(float x) {
  return static_cast<float>(B(x));
}

template<bool Maximum>
__device__ float Reduce(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  if (tid < 128)
    shared[tid] = Maximum ? fmaxf(shared[tid], shared[tid + 128])
                          : shared[tid] + shared[tid + 128];
  __syncthreads();
  if (tid < 64)
    shared[tid] = Maximum ? fmaxf(shared[tid], shared[tid + 64])
                          : shared[tid] + shared[tid + 64];
  __syncthreads();
  value = tid < 32 ? (Maximum ? fmaxf(shared[tid], shared[tid + 32])
                              : shared[tid] + shared[tid + 32])
                   : 0;
  // Finish exactly the same descending tree within the first 32 lanes.
#pragma unroll
  for (int stride = 16; stride; stride /= 2) {
    const float peer = __shfl_down(value, stride, 32);
    if (tid < stride)
      value = Maximum ? fmaxf(value, peer) : value + peer;
  }
  if (tid == 0)
    shared[0] = value;
  __syncthreads();
  return shared[0];
}

__device__ float Sum(float value, float* shared) {
  return Reduce<false>(value, shared);
}

__global__ void FloatToBf16(const float* x, B* y, std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = B(x[i]);
}

__global__ void Bf16ToFloat(const B* x, float* y, std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = float(x[i]);
}

template<bool Convolution>
__global__ void Bias(const float* x, const B* bias, B* y, int cols,
                     std::size_t n) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    // PyTorch/MIOpen adds convolution bias after rounding the BF16 product.
    // Linear/addmm instead includes bias before the output conversion.
    const float product = Convolution ? Round(x[i]) : x[i];
    y[i] = B(product + float(bias[i % cols]));
  }
}

__global__ void NormKernel(const B* x, B* y, const B* w, const B* bias,
                           int cols, int kind, float eps) {
  __shared__ float scratch[256];
  const auto offset = static_cast<std::size_t>(blockIdx.x) * cols;
  float mean = 0;
  if (kind == static_cast<int>(Norm::kLayer)) {
    float value = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
      value += float(x[offset + c]);
    mean = Sum(value, scratch) / cols;
    __syncthreads();
  }
  float square = 0;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = float(x[offset + c]) - mean;
    square += v * v;
  }
  float sum = Sum(square, scratch);
  const float inv = kind == static_cast<int>(Norm::kVae)
                        ? 1.0F / fmaxf(sqrtf(sum), 1e-12F)
                        : rsqrtf(sum / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (float(x[offset + c]) - mean) * inv;
    if (kind == static_cast<int>(Norm::kRms)) {
      v = Round(v);
      if (w)
        v *= float(w[c]);
    } else if (kind == static_cast<int>(Norm::kZeroRms)) {
      v *= 1.0F + float(w[c]);
    } else if (kind == static_cast<int>(Norm::kVae)) {
      v = Round(Round(v) * sqrtf(static_cast<float>(cols)));
      if (w)
        v = Round(v * float(w[c]));
    } else {
      if (w)
        v *= float(w[c]);
      if (bias)
        v += float(bias[c]);
    }
    y[offset + c] = B(v);
  }
}

// One wave owns each short row. Fold the eight original 32-lane
// partials in the same 256-thread reduction order before the wave reduction.
__device__ float WaveTree(float* p) {
#pragma unroll
  for (int j = 0; j < 4; j++)
    p[j] += p[j + 4];
#pragma unroll
  for (int j = 0; j < 2; j++)
    p[j] += p[j + 2];
  float sum = p[0] + p[1];
#pragma unroll
  for (int d = 16; d; d /= 2) {
    float v = __shfl_down(sum, d, 32);
    if ((threadIdx.x % 32) < d)
      sum += v;
  }
  return __shfl(sum, 0, 32);
}
__global__ void WaveNorm(const B* x, B* y, const B* w, const B* bias, int rows,
                         int cols, int kind, float eps) {
  int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows)
    return;
  size_t offset = size_t(row) * cols;
  float mean = 0;
  if (kind == static_cast<int>(Norm::kLayer)) {
    float partial[8]{};
#pragma unroll
    for (int j = 0; j < 8; j++)
      for (int c = lane + j * 32; c < cols; c += 256)
        partial[j] += float(x[offset + c]);
    mean = WaveTree(partial) / cols;
  }
  float partial[8]{};
#pragma unroll
  for (int j = 0; j < 8; j++)
    for (int c = lane + j * 32; c < cols; c += 256) {
      float v = float(x[offset + c]) - mean;
      partial[j] += v * v;
    }
  float sum = WaveTree(partial);
  float inv = kind == static_cast<int>(Norm::kVae)
                  ? 1.0F / fmaxf(sqrtf(sum), 1e-12F)
                  : rsqrtf(sum / cols + eps);
  for (int c = lane; c < cols; c += 32) {
    float v = (float(x[offset + c]) - mean) * inv;
    if (kind == static_cast<int>(Norm::kRms)) {
      v = Round(v);
      if (w)
        v *= float(w[c]);
    } else if (kind == static_cast<int>(Norm::kZeroRms))
      v *= 1.0F + float(w[c]);
    else if (kind == static_cast<int>(Norm::kVae)) {
      v = Round(Round(v) * sqrtf(float(cols)));
      if (w)
        v = Round(v * float(w[c]));
    } else {
      if (w)
        v *= float(w[c]);
      if (bias)
        v += float(bias[c]);
    }
    y[offset + c] = B(v);
  }
}

__global__ void Elementwise(const B* x, const B* other, B* y, std::size_t n,
                            int op, float scalar) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float a = float(x[i]);
  float value = a;
  if (op == 0)
    value = a / (1.0F + expf(-a));
  if (op == 1)
    value = 0.5F * a *
            (1.0F + tanhf(0.7978845608028654F * (a + 0.044715F * a * a * a)));
  if (op == 2)
    value = 0.5F * a * (1.0F + erff(a * 0.7071067811865475F));
  if (op == 4)
    value = Round(a / (1.0F + expf(-a))) * float(other[i]);
  if (op == 5)
    value = a + float(other[i]);
  // FlowMatchEuler keeps dt * model_output in BF16 (a zero-dimensional
  // FP32 dt does not promote the BF16 tensor), then adds to the FP32 sample.
  if (op == 6)
    value = a + Round(scalar * float(other[i]));
  y[i] = B(value);
}

__global__ void Modulation(const B* x, const B* mod, const B* residual, B* y,
                           int rows, int cols, int mod_cols, int part,
                           int prefix, bool gate) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * cols)
    return;
  const int row = i / cols, c = i % cols;
  const float m = float(mod[(row < prefix ? mod_cols : 0) + part * cols + c]);
  float value = float(x[i]) * (gate ? Round(tanhf(m)) : Round(1.0F + m));
  if (residual)
    value = float(residual[i]) + Round(value);
  y[i] = B(value);
}

__global__ void Rotary(const B* x, B* y, const float* frequencies, int rows,
                       int heads, int dim, int mode) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * heads * dim / 2)
    return;
  const int half = dim / 2;
  const int pair = i % half;
  const int row = i / (half * heads);
  const auto base = (i / half) * dim;
  const auto left = base + (mode == 0 ? pair * 2 : pair);
  const auto right = mode == 0 ? left + 1 : left + half;
  float c = frequencies[(static_cast<std::size_t>(row) * half + pair) * 2];
  float s = frequencies[(static_cast<std::size_t>(row) * half + pair) * 2 + 1];
  const float a = float(x[left]), b = float(x[right]);
  if (mode == 2) {
    c = Round(c);
    s = Round(s);
    y[left] = B(Round(a * c) - Round(b * s));
    y[right] = B(Round(b * c) + Round(a * s));
  } else {
    y[left] = B(a * c - b * s);
    y[right] = B(b * c + a * s);
  }
}

__global__ void HeadLayout(const B* x, B* y, int rows, int heads, int kv_heads,
                           int dim, bool inverse) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * heads * dim)
    return;
  const int d = i % dim, row = (i / dim) % rows, head = i / (dim * rows);
  if (inverse)
    y[(static_cast<std::size_t>(row) * heads + head) * dim + d] = x[i];
  else
    y[i] = x[(static_cast<std::size_t>(row) * kv_heads +
              head / (heads / kv_heads)) *
                 dim +
             d];
}

template<bool Inverse>
__global__ void HeadLayout128(const B* x, B* y, int rows, int heads,
                              int kv_heads) {
  __shared__ __align__(16) B tile[8 * 8 * 128];
  const int row_start = blockIdx.y * 8, head_start = blockIdx.x * 8;
#pragma unroll
  for (int i = threadIdx.x * 8; i < 8 * 8 * 128; i += 256 * 8) {
    const int major = i / (8 * 128), minor = i / 128 % 8, d = i % 128;
    const int row = row_start + (Inverse ? minor : major);
    const int head = head_start + (Inverse ? major : minor);
    uint4 value{};
    if (row < rows && head < heads) {
      const auto index =
          Inverse ? (static_cast<std::size_t>(head) * rows + row) * 128 + d
                  : (static_cast<std::size_t>(row) * kv_heads +
                     head / (heads / kv_heads)) *
                            128 +
                        d;
      value = *reinterpret_cast<const uint4*>(x + index);
    }
    *reinterpret_cast<uint4*>(tile + i) = value;
  }
  __syncthreads();
#pragma unroll
  for (int i = threadIdx.x * 8; i < 8 * 8 * 128; i += 256 * 8) {
    const int major = i / (8 * 128), minor = i / 128 % 8, d = i % 128;
    const int row = row_start + (Inverse ? major : minor);
    const int head = head_start + (Inverse ? minor : major);
    if (row < rows && head < heads) {
      const auto value =
          *reinterpret_cast<const uint4*>(tile + (minor * 8 + major) * 128 + d);
      const auto index =
          Inverse ? (static_cast<std::size_t>(row) * heads + head) * 128 + d
                  : (static_cast<std::size_t>(head) * rows + row) * 128 + d;
      *reinterpret_cast<uint4*>(y + index) = value;
    }
  }
}

__global__ void Softmax(const float* scores, float* probabilities, int kv_rows,
                        int query_count, int query_start, const int* limits,
                        float scale) {
  __shared__ float scratch[256];
  const auto row = blockIdx.x;
  const auto offset = static_cast<std::size_t>(row) * kv_rows;
  const int limit = limits ? limits[query_start + row % query_count] : kv_rows;
  float maximum = -INFINITY;
  for (int i = threadIdx.x; i < limit; i += blockDim.x)
    maximum = fmaxf(maximum, scores[offset + i] * scale);
  maximum = Reduce<true>(maximum, scratch);
  __syncthreads();
  float sum = 0;
  for (int i = threadIdx.x; i < limit; i += blockDim.x)
    sum += expf(scores[offset + i] * scale - maximum);
  sum = Sum(sum, scratch);
  for (int i = threadIdx.x; i < kv_rows; i += blockDim.x)
    probabilities[offset + i] =
        i < limit ? expf(scores[offset + i] * scale - maximum) / sum : 0.0F;
}

// Reuse each scaled score and exponential for ordinary image contexts. Keep
// the same per-thread additions and 256-thread reduction as the long-row path.
template<int Capacity>
__global__ void SoftmaxCached(const float* scores, float* probabilities,
                              int kv_rows, int query_count, int query_start,
                              const int* limits, float scale) {
  __shared__ float scratch[256];
  float values[Capacity];
  const auto offset = static_cast<std::size_t>(blockIdx.x) * kv_rows;
  const int limit =
      limits ? limits[query_start + blockIdx.x % query_count] : kv_rows;
  float maximum = -INFINITY;
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    values[j] = index < limit ? scores[offset + index] * scale : -INFINITY;
    maximum = fmaxf(maximum, values[j]);
  }
  maximum = Reduce<true>(maximum, scratch);
  __syncthreads();
  float sum = 0;
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    values[j] = index < limit ? expf(values[j] - maximum) : 0.0F;
    if (index < limit)
      sum += values[j];
  }
  sum = Sum(sum, scratch);
#pragma unroll
  for (int j = 0; j < Capacity; ++j) {
    const int index = threadIdx.x + j * 256;
    if (index < kv_rows)
      probabilities[offset + index] = index < limit ? values[j] / sum : 0.0F;
  }
}

__global__ void CopyColumns(const B* x, B* y, int rows, int in_cols, int start,
                            int out_cols) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<std::size_t>(rows) * out_cols)
    y[i] = x[(i / out_cols) * in_cols + start + i % out_cols];
}

__global__ void Embedding(const B* w, B* y, const std::uint32_t* ids, int rows,
                          int cols) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<std::size_t>(rows) * cols)
    y[i] = w[static_cast<std::size_t>(ids[i / cols]) * cols + i % cols];
}

__global__ void Scatter(const B* values, B* y, const int* positions, int count,
                        int cols, bool add) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(count) * cols)
    return;
  const auto dst =
      static_cast<std::size_t>(positions[i / cols]) * cols + i % cols;
  y[dst] = add ? B(float(y[dst]) + float(values[i])) : values[i];
}

__global__ void Im2col(const B* x, B* y, int height, int width, int channels,
                       int kernel, int stride, int padding, int upscale,
                       int out_width, int start, int count) {
  const int k = channels * kernel * kernel;
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(count) * k)
    return;
  const int pixel = start + i / k;
  const int kc = i % k;
  const int c = kc / (kernel * kernel);
  const int sy =
      (pixel / out_width) * stride - padding + (kc / kernel) % kernel;
  const int sx = (pixel % out_width) * stride - padding + kc % kernel;
  y[i] =
      (sy >= 0 && sx >= 0 && sy < height * upscale && sx < width * upscale)
          ? x[(static_cast<std::size_t>(sy / upscale) * width + sx / upscale) *
                  channels +
              c]
          : B(0.0F);
}

__global__ void Shortcut(const B* x, B* y, int height, int width,
                         int in_channels, int out_channels, int temporal,
                         int spatial, bool up) {
  const int out_height = up ? height * spatial : height / spatial;
  const int out_width = up ? width * spatial : width / spatial;
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(out_height) * out_width * out_channels)
    return;
  const int c = i % out_channels;
  const int py = (i / out_channels) / out_width;
  const int px = (i / out_channels) % out_width;
  if (up) {
    const int repeats =
        out_channels * temporal * spatial * spatial / in_channels;
    const int expanded =
        ((c * temporal + temporal - 1) * spatial + py % spatial) * spatial +
        px % spatial;
    const int ic = expanded / repeats;
    y[i] = x[(static_cast<std::size_t>(py / spatial) * width + px / spatial) *
                 in_channels +
             ic];
  } else {
    const int group = in_channels * temporal * spatial * spatial / out_channels;
    float sum = 0;
    for (int j = 0; j < group; ++j) {
      const int index = c * group + j;
      const int ic = index / (temporal * spatial * spatial);
      const int time = (index / (spatial * spatial)) % temporal;
      if (time == temporal - 1) {
        const int iy = py * spatial + (index / spatial) % spatial;
        const int ix = px * spatial + index % spatial;
        sum += float(
            x[(static_cast<std::size_t>(iy) * width + ix) * in_channels + ic]);
      }
    }
    y[i] = B(sum / group);
  }
}

__global__ void ScaleLatents(const B* x, B* y, const B* mean, const B* stddev,
                             std::size_t n, bool encode) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int c = i % 64;
  const float v = float(x[i]), m = float(mean[c]), s = float(stddev[c]);
  y[i] = B(encode ? Round(v - m) / s : Round(v * s) + m);
}

__global__ void Position(const B* x, const B* table, B* y, int height,
                         int width) {
  const auto i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(height) * width * 1152)
    return;
  const int token = i / 1152, c = i % 1152;
  const int py = (token / 4 / (width / 2)) * 2 + (token / 2) % 2;
  const int px = ((token / 4) % (width / 2)) * 2 + token % 2;
  const float fy = static_cast<float>(py) * 47.0F / (height - 1);
  const float fx = static_cast<float>(px) * 47.0F / (width - 1);
  const int y0 = static_cast<int>(fy), x0 = static_cast<int>(fx);
  const int y1 = min(y0 + 1, 47), x1 = min(x0 + 1, 47);
  const float dy = fy - y0, dx = fx - x0;
  // Transformers 5.17 interpolates the BF16 position table with FP32
  // coefficients and accumulates in FP32 before one final BF16 conversion.
  const float p0 =
      float(table[(y0 * 48 + x0) * 1152 + c]) * ((1 - dy) * (1 - dx));
  const float p1 = float(table[(y0 * 48 + x1) * 1152 + c]) * ((1 - dy) * dx);
  const float p2 = float(table[(y1 * 48 + x0) * 1152 + c]) * (dy * (1 - dx));
  const float p3 = float(table[(y1 * 48 + x1) * 1152 + c]) * (dy * dx);
  y[i] = B(float(x[i]) + Round(p0 + p1 + p2 + p3));
}

void BlasCheck(hipblasStatus_t status) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Qwen-Image hipBLAS failed: " +
                             std::to_string(status));
}

unsigned Blocks(std::size_t n) {
  return static_cast<unsigned>((n + 255) / 256);
}
B* Ptr(const Matrix& m) {
  return reinterpret_cast<B*>(m.data);
}

}  // namespace

void NormalizeRows(const std::uint16_t* input, std::uint16_t* output,
                   const std::uint16_t* weight, const std::uint16_t* bias,
                   int rows, int cols, Norm kind, float epsilon,
                   hipStream_t stream) {
  const auto* x = reinterpret_cast<const B*>(input);
  auto* y = reinterpret_cast<B*>(output);
  const auto* w = reinterpret_cast<const B*>(weight);
  const auto* b = reinterpret_cast<const B*>(bias);
  if (cols <= 1152)
    WaveNorm<<<(rows + 7) / 8, 256, 0, stream>>>(
        x, y, w, b, rows, cols, static_cast<int>(kind), epsilon);
  else
    NormKernel<<<rows, 256, 0, stream>>>(x, y, w, b, cols,
                                         static_cast<int>(kind), epsilon);
}

void ReorderHeads(const std::uint16_t* input, std::uint16_t* output, int rows,
                  int heads, int kv_heads, int dim, bool inverse,
                  hipStream_t stream) {
  const auto* x = reinterpret_cast<const B*>(input);
  auto* y = reinterpret_cast<B*>(output);
  if (dim == 128 && rows >= 128) {
    const dim3 grid((heads + 7) / 8, (rows + 7) / 8);
    if (inverse)
      HeadLayout128<true>
          <<<grid, 256, 0, stream>>>(x, y, rows, heads, kv_heads);
    else
      HeadLayout128<false>
          <<<grid, 256, 0, stream>>>(x, y, rows, heads, kv_heads);
  } else {
    HeadLayout<<<Blocks(static_cast<std::size_t>(rows) * heads * dim), 256, 0,
                 stream>>>(x, y, rows, heads, kv_heads, dim, inverse);
  }
}

void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string("Qwen-Image HIP: ") +
                             hipGetErrorString(status));
}

Matrix Matrix::Slice(int start, int count) const {
  if (packed || start < 0 || count < 0 || start > rows - count)
    throw std::logic_error("invalid Qwen-Image tensor slice");
  return {owner, data + static_cast<std::size_t>(start) * cols, count, cols};
}

Matrix Matrix::Reshape(int r, int c) const {
  if (packed || r <= 0 || c <= 0 || static_cast<std::size_t>(r) * c != size())
    throw std::logic_error("invalid Qwen-Image tensor reshape");
  return {owner, data, r, c};
}

struct Runtime::Pool {
  std::multimap<std::size_t, void*> free;
  ~Pool() {
    for (auto [size, p] : free) {
      (void)size;
      (void)hipFree(p);
    }
  }
  std::shared_ptr<void> Allocate(std::size_t bytes,
                                 const std::shared_ptr<Pool>& self) {
    void* p = nullptr;
    auto it = free.lower_bound(bytes);
    if (it != free.end() && it->first <= bytes * 2) {
      bytes = it->first;
      p = it->second;
      free.erase(it);
    } else {
      Check(hipMalloc(&p, bytes));
    }
    return std::shared_ptr<void>(
        p, [self, bytes](void* value) { self->free.emplace(bytes, value); });
  }
};

Runtime::Runtime(const Weights& weights)
    : pool_(std::make_shared<Pool>()), weights_(weights) {
  Check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
  try {
    BlasCheck(hipblasCreate(&blas_));
    BlasCheck(hipblasSetStream(blas_, stream_));
  } catch (...) {
    if (blas_)
      (void)hipblasDestroy(blas_);
    (void)hipStreamDestroy(stream_);
    throw;
  }
}

Runtime::~Runtime() {
  (void)hipStreamSynchronize(stream_);
  device_weights_.clear();
  packed_source_ = {};
  packed_input_ = {};
  pool_.reset();
  if (blas_)
    (void)hipblasDestroy(blas_);
  if (stream_)
    (void)hipStreamDestroy(stream_);
}

Matrix Runtime::Raw(std::size_t bytes) {
  if (!bytes || bytes > (32ULL << 30))
    throw std::length_error("Qwen-Image tensor exceeds memory limit");
  auto owner = pool_->Allocate(bytes, pool_);
  return {owner, static_cast<std::uint16_t*>(owner.get()), 1, 1};
}

Matrix Runtime::New(int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    throw std::logic_error("empty Qwen-Image tensor");
  auto m = Raw(static_cast<std::size_t>(rows) * cols * 2);
  m.rows = rows;
  m.cols = cols;
  return m;
}

Matrix Runtime::Upload(std::span<const float> values, int rows, int cols) {
  if (values.size() != static_cast<std::size_t>(rows) * cols)
    throw std::logic_error("Qwen-Image upload size mismatch");
  Matrix result = New(rows, cols);
  std::vector<std::uint16_t> packed(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto bits = std::bit_cast<std::uint32_t>(values[i]);
    packed[i] = static_cast<std::uint16_t>(
        (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
  }
  Check(hipMemcpyAsync(result.data, packed.data(), packed.size() * 2,
                       hipMemcpyHostToDevice, stream_));
  Synchronize();
  return result;
}

Matrix Runtime::UploadFloat(std::span<const float> values, int rows, int cols) {
  if (values.size() != static_cast<std::size_t>(rows) * cols)
    throw std::logic_error("FP32 upload size mismatch");
  auto out = New(rows, cols * 2);
  Check(hipMemcpyAsync(out.data, values.data(), values.size() * 4,
                       hipMemcpyHostToDevice, stream_));
  Synchronize();
  return out;
}

Matrix Runtime::Weight(std::string_view name) {
  if (auto it = device_weights_.find(name); it != device_weights_.end())
    return it->second;
  const auto& w = weights_.Get(name);
  const int rows = w.shape.empty() ? 1 : w.shape[0];
  const auto count = w.bytes / (w.bf16 ? 2 : 4);
  if (count / rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("Qwen-Image weight exceeds matrix dimensions");
  const int cols = static_cast<int>(count / rows);
  Matrix m = New(rows, cols);
  if (w.bf16)
    Check(hipMemcpyAsync(m.data, w.data, w.bytes, hipMemcpyHostToDevice,
                         stream_));
  else {
    auto raw = Raw(w.bytes);
    Check(hipMemcpyAsync(raw.data, w.data, w.bytes, hipMemcpyHostToDevice,
                         stream_));
    FloatToBf16<<<Blocks(count), 256, 0, stream_>>>(
        reinterpret_cast<const float*>(raw.data), Ptr(m), count);
  }
  if (w.shape.size() == 2 && DenseShape(rows, cols) &&
      name.ends_with(".weight") &&
      !weights_.Contains(std::string(name.substr(0, name.size() - 7)) +
                         ".bias")) {
    auto packed = New(rows, cols);
    PackDense16(m.data, packed.data, rows, cols, stream_);
    packed.packed = true;
    m = std::move(packed);
  }
  Synchronize();
  weight_bytes_ += count * 2;
  device_weights_.emplace(std::string(name), m);
  return m;
}

void Runtime::Matmul(const Matrix& x, const Matrix& w, Matrix& output,
                     const Matrix* bias, bool convolution) {
  if (x.cols != w.cols || output.cols != w.rows || output.rows != x.rows)
    throw std::logic_error("Qwen-Image GEMM shape mismatch");
  if (bias && bias->size() != static_cast<std::size_t>(output.cols))
    throw std::logic_error("Qwen-Image bias shape mismatch");
  if (w.packed) {
    if (bias || x.packed)
      throw std::logic_error("Qwen-Image packed projection contract mismatch");
    // Inputs are immutable. Retaining the source owner prevents pool reuse
    // from mistaking a new tensor for a cached QKV/gate-up input.
    if (packed_source_.data != x.data || packed_source_.rows != x.rows ||
        packed_source_.cols != x.cols) {
      auto packed = New(x.rows, x.cols);
      PackDense16(x.data, packed.data, x.rows, x.cols, stream_);
      packed_source_ = x;
      packed_input_ = std::move(packed);
    }
    if (!DenseBf16(w.data, packed_input_.data, output.data, x.rows, w.rows,
                   x.cols, stream_))
      throw std::logic_error("unsupported Qwen-Image packed projection");
    Check(hipGetLastError());
    return;
  }
  const float alpha = 1, beta = 0;
  Matrix accumulation;
  void* dst = output.data;
  if (bias) {
    accumulation = Raw(output.size() * 4);
    dst = accumulation.data;
  }
  if (!gemm_.Run(w.data, x.data, dst, x.rows, w.rows, x.cols, bias != nullptr,
                 stream_))
    BlasCheck(hipblasGemmEx(blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, w.rows, x.rows,
                            x.cols, &alpha, w.data, HIP_R_16BF, w.cols, x.data,
                            HIP_R_16BF, x.cols, &beta, dst,
                            bias ? HIP_R_32F : HIP_R_16BF, output.cols,
                            HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  if (bias) {
    if (convolution)
      Bias<true><<<Blocks(output.size()), 256, 0, stream_>>>(
          reinterpret_cast<float*>(accumulation.data), Ptr(*bias), Ptr(output),
          output.cols, output.size());
    else
      Bias<false><<<Blocks(output.size()), 256, 0, stream_>>>(
          reinterpret_cast<float*>(accumulation.data), Ptr(*bias), Ptr(output),
          output.cols, output.size());
  }
  Check(hipGetLastError());
}

Matrix Runtime::Linear(const Matrix& x, const std::string& name) {
  const auto w = Weight(name + ".weight");
  auto out = New(x.rows, w.rows);
  Matrix bias;
  if (weights_.Contains(name + ".bias"))
    bias = Weight(name + ".bias");
  Matmul(x, w, out, bias.data ? &bias : nullptr,
         weights_.Get(name + ".weight").shape.size() > 2);
  return out;
}

Matrix Runtime::Normalize(const Matrix& x, Norm kind, const std::string& weight,
                          float epsilon) {
  Matrix w, bias;
  if (!weight.empty()) {
    w = Weight(weight + (kind == Norm::kVae ? ".gamma" : ".weight"));
    if (weights_.Contains(weight + ".bias"))
      bias = Weight(weight + ".bias");
    if (w.size() != static_cast<std::size_t>(x.cols))
      throw std::logic_error("Qwen-Image norm shape mismatch");
    if (bias.data && bias.size() != w.size())
      throw std::logic_error("Qwen-Image norm bias shape mismatch");
  }
  auto out = New(x.rows, x.cols);
  NormalizeRows(x.data, out.data, w.data, bias.data, x.rows, x.cols, kind,
                epsilon, stream_);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Activate(const Matrix& x, Activation activation) {
  auto out = New(x.rows, x.cols);
  Elementwise<<<Blocks(x.size()), 256, 0, stream_>>>(
      Ptr(x), nullptr, Ptr(out), x.size(), static_cast<int>(activation), 0);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::SwiGlu(const Matrix& gate, const Matrix& up) {
  if (gate.rows != up.rows || gate.cols != up.cols)
    throw std::logic_error("SwiGLU shape");
  auto out = New(gate.rows, gate.cols);
  Elementwise<<<Blocks(gate.size()), 256, 0, stream_>>>(
      Ptr(gate), Ptr(up), Ptr(out), gate.size(), 4, 0);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Add(const Matrix& a, const Matrix& b) {
  if (a.rows != b.rows || a.cols != b.cols)
    throw std::logic_error("residual shape");
  auto out = New(a.rows, a.cols);
  Elementwise<<<Blocks(a.size()), 256, 0, stream_>>>(Ptr(a), Ptr(b), Ptr(out),
                                                     a.size(), 5, 0);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Modulate(const Matrix& x, const Matrix& mod, int part,
                         int prefix, bool gate, const Matrix* residual) {
  if (mod.rows != 2 || (mod.cols != x.cols && mod.cols != x.cols * 4) ||
      part < 0 || part >= mod.cols / x.cols || prefix < 0 || prefix > x.rows ||
      (residual && (residual->rows != x.rows || residual->cols != x.cols)))
    throw std::logic_error("Qwen-Image modulation shape mismatch");
  auto out = New(x.rows, x.cols);
  Modulation<<<Blocks(x.size()), 256, 0, stream_>>>(
      Ptr(x), Ptr(mod), residual ? Ptr(*residual) : nullptr, Ptr(out), x.rows,
      x.cols, mod.cols, part, prefix, gate);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Rope(const Matrix& x, int heads, int dim,
                     const Matrix& frequencies, int mode) {
  if (x.cols != heads * dim || frequencies.rows != x.rows ||
      frequencies.cols != dim * 2)
    throw std::logic_error("rotary shape mismatch");
  auto out = New(x.rows, x.cols);
  Rotary<<<Blocks(x.size() / 2), 256, 0, stream_>>>(
      Ptr(x), Ptr(out), reinterpret_cast<const float*>(frequencies.data),
      x.rows, heads, dim, mode);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Attention(const Matrix& q, const Matrix& k, const Matrix& v,
                          int heads, int kv_heads, int dim,
                          const std::vector<int>& key_limits) {
  if (q.cols != heads * dim || k.cols != kv_heads * dim || v.cols != k.cols ||
      v.rows != k.rows || heads % kv_heads ||
      (!key_limits.empty() &&
       key_limits.size() != static_cast<std::size_t>(q.rows)))
    throw std::logic_error("attention shape mismatch");
  auto qh = New(q.rows, heads * dim), kh = New(k.rows, heads * dim),
       vh = New(v.rows, heads * dim);
  auto out = New(q.rows, q.cols);
  ReorderHeads(q.data, qh.data, q.rows, heads, heads, dim, false, stream_);
  ReorderHeads(k.data, kh.data, k.rows, heads, kv_heads, dim, false, stream_);
  ReorderHeads(v.data, vh.data, v.rows, heads, kv_heads, dim, false, stream_);
  Matrix limits;
  if (!key_limits.empty()) {
    for (int n : key_limits)
      if (n < 1 || n > k.rows)
        throw std::logic_error("invalid attention key limit");
    limits = Raw(key_limits.size() * 4);
    Check(hipMemcpyAsync(limits.data, key_limits.data(), key_limits.size() * 4,
                         hipMemcpyHostToDevice, stream_));
    Synchronize();
  }
  if (dim == 128 && k.rows >= 512 && q.rows >= 128) {
    int masked_rows = q.rows;
    while (!key_limits.empty() && masked_rows > 0 &&
           key_limits[masked_rows - 1] == k.rows)
      --masked_rows;
    // First-step target image rows see every key. Separating this suffix
    // avoids per-query mask work and keeps short causal prefixes bounded.
    if (!key_limits.empty() && q.rows - masked_rows >= 128) {
      if (masked_rows)
        FusedAttention128(qh.data, kh.data, vh.data, out.data, heads, q.rows,
                          k.rows, 0, stream_, masked_rows,
                          reinterpret_cast<const int*>(limits.data));
      FusedAttention128(qh.data, kh.data, vh.data, out.data, heads, q.rows,
                        k.rows, masked_rows, stream_, q.rows - masked_rows);
    } else {
      FusedAttention128(qh.data, kh.data, vh.data, out.data, heads, q.rows,
                        k.rows, 0, stream_, q.rows,
                        reinterpret_cast<const int*>(limits.data));
    }
    Check(hipGetLastError());
    return out;
  }
  auto oh = New(q.rows, heads * dim);
  auto vf = Raw(vh.size() * 4), of = Raw(oh.size() * 4);
  Bf16ToFloat<<<Blocks(vh.size()), 256, 0, stream_>>>(
      Ptr(vh), reinterpret_cast<float*>(vf.data), vh.size());
  const int tile = std::min(128, q.rows);
  auto scores = Raw(static_cast<std::size_t>(heads) * tile * k.rows * 4);
  auto probabilities = Raw(static_cast<std::size_t>(heads) * tile * k.rows * 4);
  const float alpha = 1, beta = 0;
  for (int start = 0; start < q.rows; start += tile) {
    const int count = std::min(tile, q.rows - start);
    if (!gemm_.RunBatched(kh.data,
                          qh.data + static_cast<std::size_t>(start) * dim,
                          scores.data, count, k.rows, dim, heads,
                          static_cast<std::int64_t>(k.rows) * dim,
                          static_cast<std::int64_t>(q.rows) * dim,
                          static_cast<std::int64_t>(k.rows) * count, stream_))
      BlasCheck(hipblasGemmStridedBatchedEx(
          blas_, HIPBLAS_OP_T, HIPBLAS_OP_N, k.rows, count, dim, &alpha,
          kh.data, HIP_R_16BF, dim, static_cast<hipblasStride>(k.rows) * dim,
          qh.data + static_cast<std::size_t>(start) * dim, HIP_R_16BF, dim,
          static_cast<hipblasStride>(q.rows) * dim, &beta, scores.data,
          HIP_R_32F, k.rows, static_cast<hipblasStride>(k.rows) * count, heads,
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    if (k.rows >= 4096 && k.rows <= 4352)
      SoftmaxCached<17><<<heads * count, 256, 0, stream_>>>(
          reinterpret_cast<float*>(scores.data),
          reinterpret_cast<float*>(probabilities.data), k.rows, count, start,
          reinterpret_cast<int*>(limits.data), 1.0F / std::sqrt(float(dim)));
    else if (k.rows > 4352 && k.rows <= 8448)
      SoftmaxCached<33><<<heads * count, 256, 0, stream_>>>(
          reinterpret_cast<float*>(scores.data),
          reinterpret_cast<float*>(probabilities.data), k.rows, count, start,
          reinterpret_cast<int*>(limits.data), 1.0F / std::sqrt(float(dim)));
    else
      Softmax<<<heads * count, 256, 0, stream_>>>(
          reinterpret_cast<float*>(scores.data),
          reinterpret_cast<float*>(probabilities.data), k.rows, count, start,
          reinterpret_cast<int*>(limits.data), 1.0F / std::sqrt(float(dim)));
    BlasCheck(hipblasGemmStridedBatchedEx(
        blas_, HIPBLAS_OP_N, HIPBLAS_OP_N, dim, count, k.rows, &alpha, vf.data,
        HIP_R_32F, dim, static_cast<hipblasStride>(k.rows) * dim,
        probabilities.data, HIP_R_32F, k.rows,
        static_cast<hipblasStride>(k.rows) * count, &beta,
        reinterpret_cast<float*>(of.data) +
            static_cast<std::size_t>(start) * dim,
        HIP_R_32F, dim, static_cast<hipblasStride>(q.rows) * dim, heads,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
  FloatToBf16<<<Blocks(oh.size()), 256, 0, stream_>>>(
      reinterpret_cast<const float*>(of.data), Ptr(oh), oh.size());
  ReorderHeads(oh.data, out.data, q.rows, heads, heads, dim, true, stream_);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Concat(std::span<const Matrix> matrices) {
  if (matrices.empty())
    throw std::logic_error("empty tensor concatenation");
  int rows = 0;
  for (const auto& m : matrices) {
    if (m.cols != matrices[0].cols)
      throw std::logic_error("concat shape");
    rows += m.rows;
  }
  auto out = New(rows, matrices[0].cols);
  std::size_t offset = 0;
  for (const auto& m : matrices) {
    Check(hipMemcpyAsync(out.data + offset, m.data, m.size() * 2,
                         hipMemcpyDeviceToDevice, stream_));
    offset += m.size();
  }
  return out;
}

Matrix Runtime::Columns(const Matrix& x, int start, int count) {
  if (start < 0 || count <= 0 || start > x.cols - count)
    throw std::logic_error("column slice");
  auto out = New(x.rows, count);
  CopyColumns<<<Blocks(out.size()), 256, 0, stream_>>>(Ptr(x), Ptr(out), x.rows,
                                                       x.cols, start, count);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Embed(const Matrix& weights,
                      std::span<const std::uint32_t> ids) {
  for (auto id : ids)
    if (id >= static_cast<unsigned>(weights.rows))
      throw std::logic_error("token ID range");
  auto indices = Raw(ids.size() * 4);
  Check(hipMemcpyAsync(indices.data, ids.data(), ids.size() * 4,
                       hipMemcpyHostToDevice, stream_));
  Synchronize();
  auto out = New(static_cast<int>(ids.size()), weights.cols);
  Embedding<<<Blocks(out.size()), 256, 0, stream_>>>(
      Ptr(weights), Ptr(out), reinterpret_cast<std::uint32_t*>(indices.data),
      out.rows, out.cols);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::ReplaceRows(const Matrix& x, const Matrix& values,
                            std::span<const int> positions, bool add) {
  if (values.rows != static_cast<int>(positions.size()) ||
      values.cols != x.cols)
    throw std::logic_error("image scatter shape");
  for (int i : positions)
    if (i < 0 || i >= x.rows)
      throw std::logic_error("image scatter index");
  auto indices = Raw(positions.size() * 4);
  Check(hipMemcpyAsync(indices.data, positions.data(), positions.size() * 4,
                       hipMemcpyHostToDevice, stream_));
  Synchronize();
  auto out = New(x.rows, x.cols);
  Check(hipMemcpyAsync(out.data, x.data, x.size() * 2, hipMemcpyDeviceToDevice,
                       stream_));
  Scatter<<<Blocks(values.size()), 256, 0, stream_>>>(
      Ptr(values), Ptr(out), reinterpret_cast<int*>(indices.data), values.rows,
      values.cols, add);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Conv(const Matrix& x, int height, int width,
                     const std::string& name, int stride, int padding,
                     int upscale) {
  const auto& spec = weights_.Get(name + ".weight");
  if (spec.shape.size() != 4 || spec.shape[1] != x.cols ||
      spec.shape[2] != spec.shape[3] || x.rows != height * width)
    throw std::logic_error("convolution shape: " + name);
  const int kernel = spec.shape[2];
  const int total_padding = padding == -1 ? 1 : 2 * padding;
  const int out_height =
      (height * upscale + total_padding - kernel) / stride + 1;
  const int out_width = (width * upscale + total_padding - kernel) / stride + 1;
  auto out = New(out_height * out_width, spec.shape[0]);
  const auto w = Weight(name + ".weight");
  Matrix bias;
  if (weights_.Contains(name + ".bias"))
    bias = Weight(name + ".bias");
  constexpr int tile = 1024;
  auto col = New(std::min(tile, out.rows), w.cols);
  for (int start = 0; start < out.rows; start += tile) {
    const int count = std::min(tile, out.rows - start);
    auto input = col.Slice(0, count);
    auto output = out.Slice(start, count);
    if (kernel != 3 ||
        !ExpandConvolution3(x.data, input.data, height, width, x.cols, stride,
                            std::max(0, padding), upscale, out_width, start,
                            count, stream_))
      Im2col<<<Blocks(input.size()), 256, 0, stream_>>>(
          Ptr(x), Ptr(input), height, width, x.cols, kernel, stride,
          std::max(0, padding), upscale, out_width, start, count);
    Matmul(input, w, output, bias.data ? &bias : nullptr, true);
  }
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::DownShortcut(const Matrix& x, int height, int width,
                             int out_channels, int temporal, int spatial) {
  auto out = New((height / spatial) * (width / spatial), out_channels);
  Shortcut<<<Blocks(out.size()), 256, 0, stream_>>>(Ptr(x), Ptr(out), height,
                                                    width, x.cols, out_channels,
                                                    temporal, spatial, false);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::UpShortcut(const Matrix& x, int height, int width,
                           int out_channels, int temporal) {
  auto out = New(height * width * 4, out_channels);
  Shortcut<<<Blocks(out.size()), 256, 0, stream_>>>(
      Ptr(x), Ptr(out), height, width, x.cols, out_channels, temporal, 2, true);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::Euler(const Matrix& x, const Matrix& velocity, float dt) {
  auto out = New(x.rows, x.cols);
  Elementwise<<<Blocks(out.size()), 256, 0, stream_>>>(
      Ptr(x), Ptr(velocity), Ptr(out), out.size(), 6, dt);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::LatentScale(const Matrix& x, std::span<const float> mean,
                            std::span<const float> stddev, bool encode) {
  auto m = Upload(mean, 1, 64), s = Upload(stddev, 1, 64);
  auto out = New(x.rows, x.cols);
  ScaleLatents<<<Blocks(out.size()), 256, 0, stream_>>>(
      Ptr(x), Ptr(out), Ptr(m), Ptr(s), out.size(), encode);
  Check(hipGetLastError());
  return out;
}

Matrix Runtime::VisionPosition(const Matrix& x, int height, int width) {
  const auto w = Weight("text_encoder.model.visual.pos_embed.weight");
  if (x.cols != 1152 || x.rows != height * width || w.rows != 2304 ||
      w.cols != 1152)
    throw std::logic_error("Qwen-Image vision position shape mismatch");
  auto out = New(x.rows, x.cols);
  Position<<<Blocks(out.size()), 256, 0, stream_>>>(Ptr(x), Ptr(w), Ptr(out),
                                                    height, width);
  Check(hipGetLastError());
  return out;
}

std::vector<float> Runtime::Download(const Matrix& matrix) {
  auto tmp = Raw(matrix.size() * 4);
  Bf16ToFloat<<<Blocks(matrix.size()), 256, 0, stream_>>>(
      Ptr(matrix), reinterpret_cast<float*>(tmp.data), matrix.size());
  std::vector<float> result(matrix.size());
  Check(hipMemcpyAsync(result.data(), tmp.data, result.size() * 4,
                       hipMemcpyDeviceToHost, stream_));
  Synchronize();
  return result;
}

void Runtime::Observe(const Observer& observer, std::string_view name,
                      const Matrix& matrix) {
  if (observer.Wants(name))
    observer.write(name, Download(matrix), matrix.rows, matrix.cols);
}

void Runtime::Synchronize() {
  Check(hipStreamSynchronize(stream_));
}

}  // namespace gufo::models::qwen_image_21::hip
