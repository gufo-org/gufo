#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/lfm3_audio/hip/ops.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
namespace gufo::models::lfm3_audio::gpu {
namespace {
__device__ float Value(const Tensor& w, std::size_t i) {
  if (w.type == core::GgmlType::kF32)
    return static_cast<const float*>(w.data)[i];
  if (w.type == core::GgmlType::kF16)
    return __half2float(static_cast<const __half*>(w.data)[i]);
  if (w.type == core::GgmlType::kBF16)
    return float(static_cast<const hip_bfloat16*>(w.data)[i]);
  if (w.type == core::GgmlType::kQ8_0) {
    const auto& b = static_cast<const gufo::hip::Q8_0Block*>(w.data)[i / 32];
    return __half2float(b.d) * b.qs[i % 32];
  }
  return NAN;  // The host binder rejects other formats.
}
__device__ inline float Bf16(std::uint32_t v) {
  return __uint_as_float(v << 16);
}
__device__ float Activate(float x, Activation act) {
  if (act == Activation::kSilu)
    return x / (1 + expf(-x));
  if (act == Activation::kGelu)
    return 0.5F * x * (1 + erff(x * 0.70710678118654752F));
  return x;
}
constexpr int kTile = 64, kDepth = 16;
__device__ inline float Weight(const float* w, std::size_t i) {
  return w[i];
}
__device__ inline float Weight(const std::uint16_t* w, std::size_t i) {
  return __uint_as_float(std::uint32_t(w[i]) << 16);
}
// 64x64 output tile per block, 256 threads, 4x4 outputs each; weights are
// FP32 or BF16 (widened exactly), accumulation FP32.
template<class W>
__global__ void SgemmKernel(const float* __restrict__ x,
                            const W* __restrict__ w,
                            const float* __restrict__ bias, float* y, int n,
                            int M, int K, Activation act) {
  __shared__ float xs[kDepth][kTile + 1], ws[kDepth][kTile + 1];
  const int tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
  const int row0 = blockIdx.x * kTile, col0 = blockIdx.y * kTile;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += kDepth) {
    for (int i = tid; i < kTile * kDepth; i += 256) {
      const int r = i / kDepth, kk = i % kDepth;
      const int gr = row0 + r, gc = col0 + r, gk = k0 + kk;
      xs[kk][r] = gr < n && gk < K ? x[std::size_t(gr) * K + gk] : 0.0F;
      ws[kk][r] = gc < M && gk < K ? Weight(w, std::size_t(gc) * K + gk) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = xs[kk][tr * 4 + i];
        b[i] = ws[kk][tc * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = row0 + tr * 4 + i;
    if (r >= n)
      continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = col0 + tc * 4 + j;
      if (c < M)
        y[std::size_t(r) * M + c] =
            Activate(acc[i][j] + (bias ? bias[c] : 0.0F), act);
    }
  }
}
__global__ void SwiGluKernel(const float* gate, const float* up, float* out,
                             std::size_t count) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = gate[i];
    out[i] = g / (1.0F + expf(-g)) * up[i];
  }
}
__global__ void BiasActKernel(float* y, const float* bias, std::size_t count,
                              int M, Activation act) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    y[i] = Activate(y[i] + (bias ? bias[i % M] : 0.0F), act);
}
__global__ void FramesKernel(const float* x, const float* window, float* y,
                             int n, int length, int hop) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < std::size_t(n) * length)
    y[i] = x[(i / length) * hop + i % length] * window[i % length];
}
__global__ void PowerKernel(const float* s, float* p, int n, int bins) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(n) * bins)
    return;
  const float* row = s + (i / bins) * 2 * bins;
  const float re = row[i % bins], im = row[bins + i % bins];
  p[i] = re * re + im * im;
}
__global__ void LogMelKernel(float* x, std::size_t count, float floor,
                             float offset, float divisor) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] = (fmaxf(log10f(x[i]), floor) + offset) / divisor;
}
// in: mel [frames][bins]; out [W][H][C] with H = ceil(bins / 2),
// W = ceil(frames / 2). Weight [C][3][3] (kh, kw).
__global__ void ConvFirstKernel(const float* mel, const float* w,
                                const float* bias, float* out, int bins,
                                int frames, int C) {
  const int H = (bins + 1) / 2, W = (frames + 1) / 2;
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(W) * H * C)
    return;
  const int c = i % C, h = (i / C) % H, t = i / (std::size_t(C) * H);
  float sum = bias[c];
  for (int kh = 0; kh < 3; ++kh) {
    const int ih = 2 * h - 1 + kh;
    if (ih < 0 || ih >= bins)
      continue;
    for (int kw = 0; kw < 3; ++kw) {
      const int it = 2 * t - 1 + kw;
      if (it >= 0 && it < frames)
        sum = fmaf(w[c * 9 + kh * 3 + kw], mel[std::size_t(it) * bins + ih],
                   sum);
    }
  }
  out[i] = sum / (1 + expf(-sum));
}
__global__ void ConvDepthwiseKernel(const float* in, const float* w,
                                    const float* bias, float* out, int Wi,
                                    int Hi, int C) {
  const int H = (Hi + 1) / 2, W = (Wi + 1) / 2;
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(W) * H * C)
    return;
  const int c = i % C, h = (i / C) % H, t = i / (std::size_t(C) * H);
  float sum = bias[c];
  for (int kh = 0; kh < 3; ++kh) {
    const int ih = 2 * h - 1 + kh;
    if (ih < 0 || ih >= Hi)
      continue;
    for (int kw = 0; kw < 3; ++kw) {
      const int it = 2 * t - 1 + kw;
      if (it >= 0 && it < Wi)
        sum = fmaf(w[c * 9 + kh * 3 + kw],
                   in[(std::size_t(it) * Hi + ih) * C + c], sum);
    }
  }
  out[i] = sum;
}
__global__ void FlattenKernel(const float* in, float* out, int W, int H,
                              int C) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(W) * H * C)
    return;
  const int c = i % C, h = (i / C) % H, t = i / (std::size_t(C) * H);
  out[std::size_t(t) * C * H + c * H + h] = in[i];
}
__device__ float BlockSum(float value, float* scratch) {
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_down(value, offset, 32);
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  if (!lane)
    scratch[wave] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : 0.0F;
    for (int offset = 16; offset; offset /= 2)
      value += __shfl_down(value, offset, 32);
    if (!threadIdx.x)
      scratch[0] = value;
  }
  __syncthreads();
  value = scratch[0];
  __syncthreads();
  return value;
}
__global__ void LayerNormKernel(const float* x, const float* w, const float* b,
                                float* y, int dim, float eps) {
  __shared__ float scratch[32];
  const float* row = x + std::size_t(blockIdx.x) * dim;
  float sum = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    sum += row[i];
  const float mean = BlockSum(sum, scratch) / dim;
  float squares = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = row[i] - mean;
    squares = fmaf(d, d, squares);
  }
  const float inv = rsqrtf(BlockSum(squares, scratch) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    y[std::size_t(blockIdx.x) * dim + i] = (row[i] - mean) * inv * w[i] + b[i];
}
// One wave per (head, query). Lanes own dim / 32 elements each.
template<int P>
__global__ void AttentionKernel(const float* q, const float* k, const float* v,
                                const float* gate, const float* ck,
                                const float* cv, float* y, int n, int heads,
                                int kh, int position, int capacity, int window,
                                bool causal) {
  constexpr int D = P * 32;
  const int h = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  const int query = position + row;
  int first = 0, last = position + n - 1;
  if (causal) {
    last = query;
    if (window > 0)
      first = max(0, query + 1 - window);
  }
  const int kvh = h / (heads / kh), stride = kh * D;
  float qv[P], acc[P] = {};
  const float scale = rsqrtf(float(D));
#pragma unroll
  for (int p = 0; p < P; ++p)
    qv[p] = q[(std::size_t(row) * heads + h) * D + lane + p * 32];
  float maximum = -INFINITY, denominator = 0;
  for (int t = first; t <= last; ++t) {
    const float* key = t >= position ? k + std::size_t(t - position) * stride
                                     : ck + std::size_t(t % capacity) * stride;
    const float* value = t >= position
                             ? v + std::size_t(t - position) * stride
                             : cv + std::size_t(t % capacity) * stride;
    float dot = 0;
#pragma unroll
    for (int p = 0; p < P; ++p)
      dot = fmaf(qv[p], key[kvh * D + lane + p * 32], dot);
    for (int offset = 16; offset; offset /= 2)
      dot += __shfl_xor(dot, offset, 32);
    const float score = dot * scale;
    const float next = fmaxf(maximum, score);
    const float alpha = expf(maximum - next), beta = expf(score - next);
    denominator = denominator * alpha + beta;
    maximum = next;
#pragma unroll
    for (int p = 0; p < P; ++p)
      acc[p] = acc[p] * alpha + beta * value[kvh * D + lane + p * 32];
  }
  const float g =
      gate ? 1.0F / (1.0F + expf(-gate[std::size_t(row) * heads + h])) : 1.0F;
#pragma unroll
  for (int p = 0; p < P; ++p)
    y[(std::size_t(row) * heads + h) * D + lane + p * 32] =
        acc[p] / denominator * g;
}
__device__ void DepthInputsBody(const float* base, const float* mask,
                                Tensor table, const std::int32_t* codes,
                                const std::int32_t* masked, float* out,
                                int size, int c) {
  const int dim = table.cols;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float add =
        masked[c] ? mask[i]
                  : Value(table, (std::size_t(c) * size + codes[c]) * dim + i);
    out[std::size_t(c) * dim + i] = base[std::size_t(c) * dim + i] + add;
  }
}
__global__ void DepthInputsKernel(const float* base, const float* mask,
                                  Tensor table, const std::int32_t* codes,
                                  const std::int32_t* masked, float* out,
                                  int codebooks, int size) {
  DepthInputsBody(base, mask, table, codes, masked, out, size, blockIdx.x);
}
__global__ void FrameEmbeddingsKernel(Tensor table, const std::int32_t* codes,
                                      float* out, int codebooks, int size) {
  const int r = blockIdx.y, dim = table.cols;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim)
    return;
  float sum = 0;
  for (int c = 0; c < codebooks; ++c)
    sum += Value(table,
                 (std::size_t(c) * size + codes[r * codebooks + c]) * dim + i);
  out[std::size_t(r) * dim + i] = sum;
}
__global__ void FsqKernel(const float* fsq, const std::int32_t* codes,
                          float* latents, int rows, int codebooks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * codebooks * 4)
    return;
  const int j = i % 4, slot = i / 4;
  latents[i] = fsq[codes[slot] * 4 + j];
}
__global__ void RepeatKernel(const float* in, float* out, int rows, int dim,
                             int repeat) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(rows) * repeat * dim)
    return;
  out[i] = in[(i / dim) / repeat * dim + i % dim];
}
__global__ void PolarKernel(float* s, int n, int bins) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(n) * bins)
    return;
  float* row = s + (i / bins) * 2 * bins;
  const int b = i % bins;
  const float magnitude = expf(row[b]);
  float sn, cs;
  sincosf(row[bins + b], &sn, &cs);
  row[b] = magnitude * cs;
  row[bins + b] = magnitude * sn;
}
__device__ void CodebookNormBody(const float* x, const float* norms,
                                 const std::int32_t* which, float* out,
                                 int dim, float eps, int slot) {
  __shared__ float scratch[32];
  const int c = which[slot];
  const float* row = x + std::size_t(c) * dim;
  float squares = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    squares = fmaf(row[i], row[i], squares);
  const float inv = rsqrtf(BlockSum(squares, scratch) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[std::size_t(slot) * dim + i] =
        row[i] * inv * norms[std::size_t(c) * dim + i];
}
__global__ void CodebookNormKernel(const float* x, const float* norms,
                                   const std::int32_t* which, float* out,
                                   int dim, float eps) {
  CodebookNormBody(x, norms, which, out, dim, eps, blockIdx.x);
}

// One wave per logits row; Q8_0 rows read whole blocks per lane.
template<core::GgmlType kType>
__device__ void CodebookLogitsBody(Tensor table, const float* x,
                                   const std::int32_t* which, float* logits,
                                   int size, int i) {
  const int lane = threadIdx.x % 32;
  const int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (r >= size)
    return;
  const int K = table.cols;
  const std::size_t row = std::size_t(which[i]) * size + r;
  const float* v = x + std::size_t(i) * K;
  float sum = 0;
  if constexpr (kType == core::GgmlType::kQ8_0) {
    const auto* blocks =
        static_cast<const gufo::hip::Q8_0Block*>(table.data) + row * (K / 32);
    for (int b = lane; b < K / 32; b += 32) {
      const auto& q = blocks[b];
      float dot = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j)
        dot = fmaf(float(q.qs[j]), v[b * 32 + j], dot);
      sum = fmaf(__half2float(q.d), dot, sum);
    }
  } else if constexpr (kType == core::GgmlType::kBF16) {
    const auto* w = static_cast<const std::uint16_t*>(table.data) + row * K;
    for (int k = lane * 8; k < K; k += 256) {
      const uint4 packed = *reinterpret_cast<const uint4*>(w + k);
      const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum = fmaf(Bf16(words[j] & 0xFFFF), v[k + 2 * j], sum);
        sum = fmaf(Bf16(words[j] >> 16), v[k + 2 * j + 1], sum);
      }
    }
  } else {
    for (int k = lane; k < K; k += 32)
      sum = fmaf(Value(table, row * K + k), v[k], sum);
  }
  for (int offset = 16; offset; offset /= 2)
    sum += __shfl_down(sum, offset, 32);
  if (!lane)
    logits[std::size_t(i) * size + r] = sum;
}
template<core::GgmlType kType>
__global__ void CodebookLogitsKernel(Tensor table, const float* x,
                                     const std::int32_t* which, float* logits,
                                     int size) {
  CodebookLogitsBody<kType>(table, x, which, logits, size, blockIdx.y);
}

// One-row projection of a wide BF16 or Q8_0 matrix: a block owns four rows
// and its eight waves split the columns in tiles of 128, so each wave
// issues all its loads at once; partial sums meet in LDS. FP32 accumulation.
constexpr int kGemvTile = 128, kGemvWaves = 8, kGemvRows = 4;
template<bool kQ8>
__device__ inline void LoadWeights(const void* w, int row, int K, int kt,
                                   float* wv) {
  const int lane = threadIdx.x % 32;
  if constexpr (kQ8) {
    const auto& blk = static_cast<const gufo::hip::Q8_0Block*>(
        w)[std::size_t(row) * (K / 32) + kt / 32 + lane / 8];
    const float d = __half2float(blk.d);
    const int part = (lane % 8) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wv[j] = d * float(blk.qs[part + j]);
  } else {
    const uint2 packed = *reinterpret_cast<const uint2*>(
        static_cast<const std::uint16_t*>(w) + std::size_t(row) * K + kt +
        lane * 4);
    wv[0] = Bf16(packed.x & 0xFFFF);
    wv[1] = Bf16(packed.x >> 16);
    wv[2] = Bf16(packed.y & 0xFFFF);
    wv[3] = Bf16(packed.y >> 16);
  }
}
template<bool kQ8>
__global__ void __launch_bounds__(256)
    WideGemvKernel(const void* w, const float* x, float* y, int M, int K) {
  __shared__ float partial[kGemvWaves][kGemvRows];
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int first = blockIdx.x * kGemvRows;
  float acc[kGemvRows] = {};
  for (int kt = wave * kGemvTile; kt < K; kt += kGemvWaves * kGemvTile) {
    float wv[kGemvRows][4];
#pragma unroll
    for (int j = 0; j < kGemvRows; ++j)
      LoadWeights<kQ8>(w, min(first + j, M - 1), K, kt, wv[j]);
    const float4 v = *reinterpret_cast<const float4*>(x + kt + lane * 4);
#pragma unroll
    for (int j = 0; j < kGemvRows; ++j) {
      acc[j] = fmaf(wv[j][0], v.x, acc[j]);
      acc[j] = fmaf(wv[j][1], v.y, acc[j]);
      acc[j] = fmaf(wv[j][2], v.z, acc[j]);
      acc[j] = fmaf(wv[j][3], v.w, acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kGemvRows; ++j) {
    float v = acc[j];
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
      v += __shfl_xor(v, offset, 32);
    if (!lane)
      partial[wave][j] = v;
  }
  __syncthreads();
  if (threadIdx.x < kGemvRows && first + int(threadIdx.x) < M) {
    float sum = 0;
    for (int i = 0; i < kGemvWaves; ++i)
      sum += partial[i][threadIdx.x];
    y[first + threadIdx.x] = sum;
  }
}
constexpr int kSortSize = 2048;
// One block per sampled codebook. Ties keep the lowest index, like argmax
// and the top-k threshold of the reference sampler.
__device__ void SampleCodesBody(const float* logits, const std::int32_t* which,
                                const float* uniforms, FrameState state,
                                FrameSampling sampling, int size, int i) {
  __shared__ float values[kSortSize];
  __shared__ int indices[kSortSize];
  const int c = which[i], tid = threadIdx.x;
  const float* row = logits + std::size_t(i) * size;
  const std::int32_t* counts = state.counts + std::size_t(c) * size;
  for (int r = tid; r < kSortSize; r += blockDim.x) {
    float v = -INFINITY;
    if (r < size) {
      v = row[r];
      if (sampling.penalty != 1.0F && counts[r] > 0)
        v = v < 0 ? v * sampling.penalty : v / sampling.penalty;
      if (sampling.temperature > 0)
        v /= sampling.temperature;
    }
    values[r] = v;
    indices[r] = r;
  }
  __syncthreads();
  // Bitonic sort, descending by value, ascending by index on ties.
  for (int k = 2; k <= kSortSize; k *= 2)
    for (int j = k / 2; j > 0; j /= 2) {
      for (int a = tid; a < kSortSize; a += blockDim.x) {
        const int b = a ^ j;
        if (b <= a)
          continue;
        const bool before = values[a] > values[b] ||
                            (values[a] == values[b] && indices[a] < indices[b]);
        const bool descending = (a & k) == 0;
        if (descending != before) {
          const float tv = values[a];
          values[a] = values[b];
          values[b] = tv;
          const int ti = indices[a];
          indices[a] = indices[b];
          indices[b] = ti;
        }
      }
      __syncthreads();
    }
  if (tid)
    return;
  int pick = indices[0];
  if (sampling.temperature > 0) {
    int kept = size;
    if (sampling.top_k > 0 && sampling.top_k < size) {
      const float kth = values[sampling.top_k - 1];
      kept = sampling.top_k;
      while (kept < size && values[kept] >= kth)
        ++kept;
    }
    float total = 0;
    for (int r = 0; r < kept; ++r)
      total += expf(values[r] - values[0]);
    float target = uniforms[c] * total;
    for (int r = 0; r < kept; ++r) {
      pick = indices[r];
      target -= expf(values[r] - values[0]);
      if (target < 0)
        break;
    }
  }
  state.codes[c] = pick;
  state.masked[c] = 0;
}
__global__ void SampleCodesKernel(const float* logits,
                                  const std::int32_t* which,
                                  const float* uniforms, FrameState state,
                                  FrameSampling sampling, int size) {
  SampleCodesBody(logits, which, uniforms, state, sampling, size, blockIdx.x);
}
__device__ void StartFrameBody(FrameState state, int codebooks) {
  const int c = threadIdx.x;
  if (c < codebooks) {
    state.codes[c] = 0;
    state.masked[c] = 1;
  }
}
__global__ void StartFrameKernel(FrameState state, int codebooks) {
  StartFrameBody(state, codebooks);
}
__device__ void FinishFrameBody(Tensor table, FrameState state,
                                FrameSampling sampling, float* embedding,
                                int codebooks, int size, int end_code,
                                int block) {
  __shared__ int codes[32];
  __shared__ bool end;
  const int tid = threadIdx.x;
  if (tid < codebooks)
    codes[tid] = state.codes[tid];
  __syncthreads();
  if (!tid) {
    end = false;
    for (int c = 0; c < codebooks; ++c)
      end = end || codes[c] == end_code;
  }
  __syncthreads();
  if (end && tid < codebooks)
    codes[tid] = end_code;
  __syncthreads();
  const int dim = table.cols;
  const int i = block * blockDim.x + tid;
  if (i < dim) {
    float sum = 0;
    for (int c = 0; c < codebooks; ++c)
      sum += Value(table, (std::size_t(c) * size + codes[c]) * dim + i);
    embedding[i] = sum;
  }
  if (block)
    return;
  if (end && tid < codebooks)
    state.codes[tid] = end_code;
  if (end || sampling.window <= 0)
    return;
  const int head = state.position[0], filled = state.position[1];
  if (tid < codebooks) {
    std::int32_t* counts = state.counts + std::size_t(tid) * size;
    std::int32_t& slot = state.ring[std::size_t(head) * codebooks + tid];
    if (filled == sampling.window)
      --counts[slot];
    slot = codes[tid];
    ++counts[codes[tid]];
  }
  __syncthreads();
  if (!tid) {
    state.position[0] = (head + 1) % sampling.window;
    state.position[1] = min(filled + 1, sampling.window);
  }
}
__global__ void FinishFrameKernel(Tensor table, FrameState state,
                                  FrameSampling sampling, float* embedding,
                                  int codebooks, int size, int end_code) {
  FinishFrameBody(table, state, sampling, embedding, codebooks, size, end_code,
                  blockIdx.x);
}
// Frames of several voices: blockIdx.y (z for logits) selects the voice.
__global__ void StartFramesKernel(const VoiceFrame* frames, int codebooks) {
  StartFrameBody(frames[blockIdx.y].state, codebooks);
}
__global__ void DepthInputsFramesKernel(const VoiceFrame* frames,
                                        const float* mask, Tensor table,
                                        int size) {
  const auto& f = frames[blockIdx.y];
  DepthInputsBody(f.base, mask, table, f.state.codes, f.state.masked, f.x,
                  size, blockIdx.x);
}
__global__ void CodebookNormFramesKernel(const VoiceFrame* frames,
                                         const float* norms, int pass,
                                         int dim, float eps) {
  const auto& f = frames[blockIdx.y];
  CodebookNormBody(f.z, norms, f.order + pass * kFramePass, f.head_inputs,
                   dim, eps, blockIdx.x);
}
template<core::GgmlType kType>
__global__ void CodebookLogitsFramesKernel(const VoiceFrame* frames,
                                           Tensor table, int pass, int size) {
  const auto& f = frames[blockIdx.z];
  CodebookLogitsBody<kType>(table, f.head_inputs, f.order + pass * kFramePass,
                            f.logits, size, blockIdx.y);
}
__global__ void SampleCodesFramesKernel(const VoiceFrame* frames, int pass,
                                        int size) {
  const auto& f = frames[blockIdx.y];
  SampleCodesBody(f.logits, f.order + pass * kFramePass, f.uniforms, f.state,
                  f.sampling, size, blockIdx.x);
}
__global__ void FinishFramesKernel(const VoiceFrame* frames, Tensor table,
                                   int codebooks, int size, int end_code) {
  const auto& f = frames[blockIdx.y];
  FinishFrameBody(table, f.state, f.sampling, f.embedding, codebooks, size,
                  end_code, blockIdx.x);
}
unsigned Blocks(std::size_t count, int threads = 256) {
  return unsigned((count + threads - 1) / threads);
}
}  // namespace
void Sgemm(const float* x, const float* w, const float* bias, float* y, int n,
           int M, int K, Activation act, hipStream_t s) {
  SgemmKernel<<<dim3((n + kTile - 1) / kTile, (M + kTile - 1) / kTile), 256,
                0, s>>>(x, w, bias, y, n, M, K, act);
}
void SgemmBf16(const float* x, const void* w, const float* bias, float* y,
               int n, int M, int K, Activation act, hipStream_t s) {
  SgemmKernel<<<dim3((n + kTile - 1) / kTile, (M + kTile - 1) / kTile), 256,
                0, s>>>(x, static_cast<const std::uint16_t*>(w), bias, y, n, M,
                        K, act);
}
void SwiGlu(const float* gate, const float* up, float* out, std::size_t count,
            hipStream_t s) {
  SwiGluKernel<<<Blocks(count), 256, 0, s>>>(gate, up, out, count);
}
void BiasAct(float* y, const float* bias, int n, int M, Activation act,
             hipStream_t s) {
  const auto count = std::size_t(n) * M;
  BiasActKernel<<<Blocks(count), 256, 0, s>>>(y, bias, count, M, act);
}
void Frames(const float* x, const float* window, float* y, int n,
            int length, int hop, hipStream_t s) {
  FramesKernel<<<Blocks(std::size_t(n) * length), 256, 0, s>>>(x, window, y, n,
                                                               length, hop);
}
void Power(const float* spectrum, float* power, int n, int bins,
           hipStream_t s) {
  PowerKernel<<<Blocks(std::size_t(n) * bins), 256, 0, s>>>(spectrum, power, n,
                                                            bins);
}
void LogMelNormalize(float* x, std::size_t count, float floor, float offset,
                     float divisor, hipStream_t s) {
  LogMelKernel<<<Blocks(count), 256, 0, s>>>(x, count, floor, offset, divisor);
}
void ConvFirst(const float* mel, const float* w, const float* bias, float* out,
               int bins, int frames, int C, hipStream_t s) {
  const auto count = std::size_t((frames + 1) / 2) * ((bins + 1) / 2) * C;
  ConvFirstKernel<<<Blocks(count), 256, 0, s>>>(mel, w, bias, out, bins,
                                                frames, C);
}
void ConvDepthwise(const float* in, const float* w, const float* bias,
                   float* out, int W, int H, int C, hipStream_t s) {
  const auto count = std::size_t((W + 1) / 2) * ((H + 1) / 2) * C;
  ConvDepthwiseKernel<<<Blocks(count), 256, 0, s>>>(in, w, bias, out, W, H, C);
}
void FlattenChannels(const float* in, float* out, int W, int H, int C,
                     hipStream_t s) {
  FlattenKernel<<<Blocks(std::size_t(W) * H * C), 256, 0, s>>>(in, out, W, H,
                                                               C);
}
void LayerNorm(const float* x, const float* w, const float* b, float* y,
               int n, int dim, float eps, hipStream_t s) {
  LayerNormKernel<<<n, 256, 0, s>>>(x, w, b, y, dim, eps);
}
void Attention(const float* q, const float* k, const float* v,
               const float* gate, const float* ck, const float* cv, float* out,
               int n, int heads, int kh, int dim, int position, int capacity,
               int window, bool causal, hipStream_t s) {
  const dim3 grid(heads, n);
  switch (dim) {
    case 32:
      AttentionKernel<1><<<grid, 32, 0, s>>>(q, k, v, gate, ck, cv, out, n,
                                             heads, kh, position, capacity,
                                             window, causal);
      break;
    case 64:
      AttentionKernel<2><<<grid, 32, 0, s>>>(q, k, v, gate, ck, cv, out, n,
                                             heads, kh, position, capacity,
                                             window, causal);
      break;
    case 96:
      AttentionKernel<3><<<grid, 32, 0, s>>>(q, k, v, gate, ck, cv, out, n,
                                             heads, kh, position, capacity,
                                             window, causal);
      break;
    default:
      AttentionKernel<4><<<grid, 32, 0, s>>>(q, k, v, gate, ck, cv, out, n,
                                             heads, kh, position, capacity,
                                             window, causal);
  }
}
void DepthInputs(const float* base, const float* mask, const Tensor& table,
                 const std::int32_t* codes, const std::int32_t* masked,
                 float* out, int codebooks, int size, hipStream_t s) {
  DepthInputsKernel<<<codebooks, 256, 0, s>>>(base, mask, table, codes, masked,
                                              out, codebooks, size);
}
void FrameEmbeddings(const Tensor& table, const std::int32_t* codes,
                     float* out, int rows, int codebooks, int size,
                     hipStream_t s) {
  FrameEmbeddingsKernel<<<dim3(Blocks(table.cols), rows), 256, 0, s>>>(
      table, codes, out, codebooks, size);
}
void FsqLatents(const float* fsq, const std::int32_t* codes, float* latents,
                int rows, int codebooks, hipStream_t s) {
  FsqKernel<<<Blocks(std::size_t(rows) * codebooks * 4), 256, 0, s>>>(
      fsq, codes, latents, rows, codebooks);
}
void Repeat(const float* in, float* out, int rows, int dim, int repeat,
            hipStream_t s) {
  RepeatKernel<<<Blocks(std::size_t(rows) * repeat * dim), 256, 0, s>>>(
      in, out, rows, dim, repeat);
}
void Polar(float* spectrum, int n, int bins, hipStream_t s) {
  PolarKernel<<<Blocks(std::size_t(n) * bins), 256, 0, s>>>(spectrum, n, bins);
}
void CodebookNorm(const float* x, const float* norms, const std::int32_t* which,
                  float* out, int count, int dim, float eps, hipStream_t s) {
  CodebookNormKernel<<<count, 256, 0, s>>>(x, norms, which, out, dim, eps);
}
void CodebookLogits(const Tensor& table, const float* x,
                    const std::int32_t* which, float* logits, int count,
                    int size, hipStream_t s) {
  const dim3 grid((size + 7) / 8, count);
  if (table.type == core::GgmlType::kQ8_0)
    CodebookLogitsKernel<core::GgmlType::kQ8_0>
        <<<grid, 256, 0, s>>>(table, x, which, logits, size);
  else if (table.type == core::GgmlType::kBF16)
    CodebookLogitsKernel<core::GgmlType::kBF16>
        <<<grid, 256, 0, s>>>(table, x, which, logits, size);
  else
    CodebookLogitsKernel<core::GgmlType::kF32>
        <<<grid, 256, 0, s>>>(table, x, which, logits, size);
}
void StartFrame(FrameState state, int codebooks, hipStream_t s) {
  StartFrameKernel<<<1, 32, 0, s>>>(state, codebooks);
}
void SampleCodes(const float* logits, const std::int32_t* which,
                 const float* uniforms, FrameState state, FrameSampling sampling,
                 int count, int size, hipStream_t s) {
  SampleCodesKernel<<<count, 256, 0, s>>>(logits, which, uniforms, state,
                                          sampling, size);
}
void FinishFrame(const Tensor& audio_out, FrameState state,
                 FrameSampling sampling, float* embedding, int codebooks,
                 int size, int end_code, hipStream_t s) {
  FinishFrameKernel<<<Blocks(audio_out.cols), 256, 0, s>>>(
      audio_out, state, sampling, embedding, codebooks, size, end_code);
}
void StartFrames(const VoiceFrame* frames, int voices, int codebooks,
                 hipStream_t s) {
  StartFramesKernel<<<dim3(1, voices), 32, 0, s>>>(frames, codebooks);
}
void DepthInputsFrames(const VoiceFrame* frames, int voices, const float* mask,
                       const Tensor& table, int codebooks, int size,
                       hipStream_t s) {
  DepthInputsFramesKernel<<<dim3(codebooks, voices), 256, 0, s>>>(
      frames, mask, table, size);
}
void CodebookNormFrames(const VoiceFrame* frames, int voices,
                        const float* norms, int pass, int dim, float eps,
                        hipStream_t s) {
  CodebookNormFramesKernel<<<dim3(kFramePass, voices), 256, 0, s>>>(
      frames, norms, pass, dim, eps);
}
void CodebookLogitsFrames(const VoiceFrame* frames, int voices,
                          const Tensor& table, int pass, int size,
                          hipStream_t s) {
  const dim3 grid((size + 7) / 8, kFramePass, voices);
  if (table.type == core::GgmlType::kQ8_0)
    CodebookLogitsFramesKernel<core::GgmlType::kQ8_0>
        <<<grid, 256, 0, s>>>(frames, table, pass, size);
  else if (table.type == core::GgmlType::kBF16)
    CodebookLogitsFramesKernel<core::GgmlType::kBF16>
        <<<grid, 256, 0, s>>>(frames, table, pass, size);
  else
    CodebookLogitsFramesKernel<core::GgmlType::kF32>
        <<<grid, 256, 0, s>>>(frames, table, pass, size);
}
void SampleCodesFrames(const VoiceFrame* frames, int voices, int pass,
                       int size, hipStream_t s) {
  SampleCodesFramesKernel<<<dim3(kFramePass, voices), 256, 0, s>>>(
      frames, pass, size);
}
void FinishFrames(const VoiceFrame* frames, int voices, const Tensor& audio_out,
                  int codebooks, int size, int end_code, hipStream_t s) {
  FinishFramesKernel<<<dim3(Blocks(audio_out.cols), voices), 256, 0, s>>>(
      frames, audio_out, codebooks, size, end_code);
}
bool WideGemv(const Tensor& w, const float* x, float* y, hipStream_t s) {
  const bool q8 = w.type == core::GgmlType::kQ8_0;
  if ((!q8 && w.type != core::GgmlType::kBF16) || w.cols % kGemvTile)
    return false;
  const dim3 grid((w.rows + kGemvRows - 1) / kGemvRows);
  if (q8)
    WideGemvKernel<true><<<grid, 256, 0, s>>>(w.data, x, y, w.rows, w.cols);
  else
    WideGemvKernel<false><<<grid, 256, 0, s>>>(w.data, x, y, w.rows, w.cols);
  return true;
}
}  // namespace gufo::models::lfm3_audio::gpu
