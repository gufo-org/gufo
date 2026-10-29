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
__device__ float Activate(float x, Activation act) {
  if (act == Activation::kSilu)
    return x / (1 + expf(-x));
  if (act == Activation::kGelu)
    return 0.5F * x * (1 + erff(x * 0.70710678118654752F));
  return x;
}
constexpr int kTile = 64, kDepth = 16;
// 64x64 output tile per block, 256 threads, 4x4 outputs each.
__global__ void SgemmKernel(const float* __restrict__ x,
                            const float* __restrict__ w,
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
      ws[kk][r] = gc < M && gk < K ? w[std::size_t(gc) * K + gk] : 0.0F;
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
__global__ void DepthInputsKernel(const float* base, const float* mask,
                                  Tensor table, const std::int32_t* codes,
                                  const std::int32_t* masked, float* out,
                                  int codebooks, int size) {
  const int c = blockIdx.x, dim = table.cols;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float add =
        masked[c] ? mask[i]
                  : Value(table, (std::size_t(c) * size + codes[c]) * dim + i);
    out[std::size_t(c) * dim + i] = base[std::size_t(c) * dim + i] + add;
  }
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
__global__ void CodebookNormKernel(const float* x, const float* norms,
                                   const std::int32_t* which, float* out,
                                   int dim, float eps) {
  __shared__ float scratch[32];
  const int c = which[blockIdx.x];
  const float* row = x + std::size_t(c) * dim;
  float squares = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    squares = fmaf(row[i], row[i], squares);
  const float inv = rsqrtf(BlockSum(squares, scratch) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[std::size_t(blockIdx.x) * dim + i] =
        row[i] * inv * norms[std::size_t(c) * dim + i];
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
}  // namespace gufo::models::lfm3_audio::gpu
