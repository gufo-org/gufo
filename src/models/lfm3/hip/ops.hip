#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <type_traits>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/lfm3/hip/ops.hpp"
#include "src/models/lfm3/hip/q8_prefill.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/norm_residual.hpp"
#include "src/models/qwen/hip/quant_ops.hpp"
namespace gufo::models::lfm3::gpu {
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
  return NAN;  // The host binder rejects unsupported formats.
}
__global__ void Embedding(Tensor w, const std::uint32_t* ids, float* y, int n) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < std::size_t(n) * w.cols)
    y[i] = Value(w, std::size_t(ids[i / w.cols]) * w.cols + i % w.cols);
}
__global__ void F16Dense(Tensor w, const float* x, float* y, int n) {
  int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32,
      token = blockIdx.y;
  if (row >= int(w.rows) || token >= n)
    return;
  float sum = 0;
  for (int k = lane; k < int(w.cols); k += 32)
    sum = fmaf(Value(w, std::size_t(row) * w.cols + k),
               x[std::size_t(token) * w.cols + k], sum);
  for (int d = 16; d; d /= 2)
    sum += __shfl_down(sum, d, 32);
  if (!lane)
    y[std::size_t(token) * w.rows + row] = sum;
}
// One thread per (row, channel): earlier rows' B*x products are recomputed
// from the projection, or read from the history before the chunk. The FMA
// order over the window matches a sequential scan.
__global__ void Convolution(const float* p, const float* w,
                            const float* history, float* y, int n, int H,
                            int L) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (ch >= H)
    return;
  float sum = 0;
  for (int t = 0; t < L; ++t) {
    int source = row - (L - 1) + t;
    float bx = source >= 0 ? p[std::size_t(source) * 3 * H + ch] *
                                 p[std::size_t(source) * 3 * H + 2 * H + ch]
                           : history[ch * (L - 1) + (L - 1) + source];
    sum = fmaf(w[ch * L + t], bx, sum);
  }
  y[std::size_t(row) * H + ch] = sum * p[std::size_t(row) * 3 * H + H + ch];
}
// Runs after Convolution: keeps the last L - 1 B*x products.
__global__ void ConvolutionHistory(const float* p, float* history, int n,
                                   int H, int L) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= H)
    return;
  float past[31];
  for (int j = 0; j < L - 1; ++j) {
    int source = n - (L - 1) + j;
    past[j] = source >= 0 ? p[std::size_t(source) * 3 * H + ch] *
                                p[std::size_t(source) * 3 * H + 2 * H + ch]
                          : history[ch * (L - 1) + (L - 1) + source];
  }
  for (int j = 0; j < L - 1; ++j)
    history[ch * (L - 1) + j] = past[j];
}
// Each thread owns one frequency pair and rotates kRows (token, head) rows,
// so the double-precision frequency power is evaluated once per pair.
constexpr int kRopeRows = 8;
__global__ void Rotary(float* x, const float* factors, int n, int heads, int D,
                       int position, float theta) {
  int pair = threadIdx.x % (D / 2), slots = blockDim.x / (D / 2);
  int first = blockIdx.x * slots * kRopeRows + threadIdx.x / (D / 2);
  float frequency = float(pow(double(theta), double(-2.0F * pair / D)));
  for (int i = 0; i < kRopeRows; ++i) {
    int head = first + i * slots, token = head / heads;
    if (head >= n * heads)
      return;
    float angle = (position + token) * frequency;
    if (factors)
      angle /= factors[pair];
    float sn, cs;
    sincosf(angle, &sn, &cs);
    float a = x[head * D + pair], b = x[head * D + pair + D / 2];
    x[head * D + pair] = a * cs - b * sn;
    x[head * D + pair + D / 2] = b * cs + a * sn;
  }
}
// One wave owns a query/head. Four independent dot chains preserve the
// original four 32-lane sums, without a block barrier for every cached token.
__global__ void Attn(const float* q, const float* k, const float* v,
                     const float* gate, const float* ck, const float* cv,
                     float* y, int heads, int kh, int D, int position,
                     int capacity, int window) {
  int h = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  int query = position + row;
  int first = window > 0 ? max(0, query + 1 - window) : 0;
  int kvh = h / (heads / kh), stride = kh * D;
  float query_values[4], acc[4] = {};
  for (int part = 0; part < 4; ++part)
    query_values[part] =
        q[(std::size_t(row) * heads + h) * D + lane + part * 32];
  float maximum = -INFINITY, denominator = 0;
  constexpr int kLookahead = 8;
  for (int base = first; base <= query; base += kLookahead) {
    float keys[kLookahead][4], values[kLookahead][4];
#pragma unroll
    for (int j = 0; j < kLookahead; ++j) {
      int t = base + j;
      if (t <= query) {
        const float* key = t >= position
                               ? k + std::size_t(t - position) * stride
                               : ck + std::size_t(t % capacity) * stride;
        const float* value = t >= position
                                 ? v + std::size_t(t - position) * stride
                                 : cv + std::size_t(t % capacity) * stride;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          keys[j][part] = key[kvh * D + lane + part * 32];
          values[j][part] = value[kvh * D + lane + part * 32];
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kLookahead; ++j) {
      if (base + j <= query) {
        float dot[4];
#pragma unroll
        for (int part = 0; part < 4; ++part)
          dot[part] = query_values[part] * keys[j][part];
        for (int offset = 16; offset; offset /= 2)
#pragma unroll
          for (int part = 0; part < 4; ++part)
            dot[part] += __shfl_down(dot[part], offset, 32);
        float score = __shfl(
            (dot[0] + dot[1] + dot[2] + dot[3]) * rsqrtf(float(D)), 0, 32);
        float next = fmaxf(maximum, score);
        float alpha = expf(maximum - next), beta = expf(score - next);
        denominator = denominator * alpha + beta;
        maximum = next;
#pragma unroll
        for (int part = 0; part < 4; ++part)
          acc[part] = acc[part] * alpha + beta * values[j][part];
      }
    }
  }
  for (int part = 0; part < 4; ++part)
    y[(std::size_t(row) * heads + h) * D + lane + part * 32] =
        (acc[part] / denominator) / (1 + expf(-gate[row * heads + h]));
}
// acc += (lane N of this 16-lane row's x) * y, as one DPP row_share FMA.
template<int N>
__device__ __forceinline__ float FmaRowShare(float x, float y, float acc) {
  asm("v_fmac_f32_dpp %0, %1, %2 row_share:%3 row_mask:0xf bank_mask:0xf"
      : "+v"(acc)
      : "v"(x), "v"(y), "n"(N));
  return acc;
}
template<class F, int... I>
__device__ __forceinline__ void Unroll(F f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
// Prompt attention: a block owns 64 (head, query) rows of one KV group and
// streams 32-key K/V tiles through LDS. Each wave owns eight rows and each
// lane one key of the tile. Query values stay in registers and reach every
// lane through DPP broadcasts, so K/V is read once per wave. Scores, the
// online softmax and P*V accumulate in FP32.
template<int D>
__launch_bounds__(256) __global__
    void TiledAttn(const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ gate,
                   const float* __restrict__ ck, const float* __restrict__ cv,
                   float* __restrict__ y, int n, int heads, int kh,
                   int position, int capacity, int window) {
  constexpr int kRows = 8, kKeys = 32, kSlices = D / 16;
  static_assert(D == 128);
  __shared__ float4 sk[kKeys][D / 4 + 1], sv[kKeys][D / 4];
  const int group = heads / kh, queries = 64 / group;
  const int g = blockIdx.y, q0 = blockIdx.x * queries, tid = threadIdx.x;
  const int lane = tid % 32, wave = tid / 32;
  const int h = g * group + wave * kRows / queries;
  const int wq0 = q0 + wave * kRows % queries;
  const int stride = kh * D;
  float qreg[kRows][kSlices];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < kSlices; ++m)
      qreg[r][m] = wq0 + r < n
                       ? q[(std::size_t(wq0 + r) * heads + h) * D + 16 * m +
                           lane % 16]
                       : 0;
  float4 acc[kRows];
  float maximum[kRows], denominator[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    acc[r] = make_float4(0, 0, 0, 0);
    maximum[r] = -INFINITY;
    denominator[r] = 0;
  }
  const float scale = rsqrtf(float(D));
  const int last_query = position + min(q0 + queries, n) - 1;
  const int begin = window > 0 ? max(0, position + q0 + 1 - window) : 0;
  const int wave_last = position + min(wq0 + kRows, n) - 1;
  const int wave_first = window > 0 ? max(0, position + wq0 + 1 - window) : 0;
  for (int base = begin; base <= last_query; base += kKeys) {
    __syncthreads();
    for (int i = tid; i < kKeys * D / 4; i += 256) {
      int t = base + i / (D / 4), d4 = i % (D / 4);
      float4 kk = make_float4(0, 0, 0, 0), vv = kk;
      if (t <= last_query) {
        const float* key = t >= position
                               ? k + std::size_t(t - position) * stride
                               : ck + std::size_t(t % capacity) * stride;
        const float* value = t >= position
                                 ? v + std::size_t(t - position) * stride
                                 : cv + std::size_t(t % capacity) * stride;
        kk = reinterpret_cast<const float4*>(key + g * D)[d4];
        vv = reinterpret_cast<const float4*>(value + g * D)[d4];
      }
      sk[i / (D / 4)][d4] = kk;
      sv[i / (D / 4)][d4] = vv;
    }
    __syncthreads();
    if (wq0 >= n || base > wave_last || base + kKeys <= wave_first)
      continue;
    float score[kRows] = {};
#pragma unroll
    for (int m = 0; m < kSlices; ++m) {
      float4 k4[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        k4[i] = sk[lane][m * 4 + i];
      const float* kf = reinterpret_cast<const float*>(k4);
      Unroll(
          [&](auto N) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
              score[r] = FmaRowShare<N>(qreg[r][m], kf[N], score[r]);
          },
          std::make_integer_sequence<int, 16>{});
    }
    const int t = base + lane;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int query = position + wq0 + r;
      const int first = window > 0 ? max(0, query + 1 - window) : 0;
      const bool valid = wq0 + r < n && t >= first && t <= query;
      float s = valid ? score[r] * scale : -INFINITY, tile_max = s;
#pragma unroll
      for (int offset = 16; offset; offset /= 2)
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 32));
      if (tile_max == -INFINITY) {
        score[r] = 0;
        continue;
      }
      const float next = fmaxf(maximum[r], tile_max);
      const float alpha = expf(maximum[r] - next), p = expf(s - next);
      denominator[r] = denominator[r] * alpha + p;
      acc[r].x *= alpha;
      acc[r].y *= alpha;
      acc[r].z *= alpha;
      acc[r].w *= alpha;
      maximum[r] = next;
      score[r] = p;
    }
#pragma unroll
    for (int j = 0; j < kKeys; ++j) {
      const float4 vv = sv[j][lane];
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const float p = __builtin_bit_cast(
            float,
            __builtin_amdgcn_readlane(__builtin_bit_cast(int, score[r]), j));
        acc[r].x = fmaf(p, vv.x, acc[r].x);
        acc[r].y = fmaf(p, vv.y, acc[r].y);
        acc[r].z = fmaf(p, vv.z, acc[r].z);
        acc[r].w = fmaf(p, vv.w, acc[r].w);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const int row = wq0 + r;
    float total = denominator[r];
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
      total += __shfl_xor(total, offset, 32);
    if (row >= n)
      continue;
    const float g_scale = 1 + expf(-gate[row * heads + h]);
    reinterpret_cast<float4*>(y + (std::size_t(row) * heads + h) * D)[lane] =
        make_float4((acc[r].x / total) / g_scale, (acc[r].y / total) / g_scale,
                    (acc[r].z / total) / g_scale,
                    (acc[r].w / total) / g_scale);
  }
}
__global__ void Cache(const float* k, const float* v, float* ck, float* cv,
                      int n, int stride, int position, int capacity) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // Only the last capacity rows have distinct destinations.
  int kept = min(n, capacity), first = n - kept;
  if (i >= std::size_t(kept) * stride)
    return;
  auto src = std::size_t(first) * stride + i;
  auto dst = std::size_t((position + first + i / stride) % capacity) * stride +
             i % stride;
  ck[dst] = k[src];
  cv[dst] = v[src];
}
}  // namespace
void Embed(const Tensor& w, const std::uint32_t* tokens, float* out, int n,
           hipStream_t s) {
  Embedding<<<(std::size_t(n) * w.cols + 255) / 256, 256, 0, s>>>(w, tokens,
                                                                  out, n);
}
std::size_t DenseWorkspaceBytes(std::size_t rows, std::size_t K) {
  return SplitActivationBytes(rows, K);
}
void Dense(const Tensor& w, const float* x, float* out, int n, hipStream_t s,
           DenseWorkspace workspace) {
  if (n >= 32 && w.type == core::GgmlType::kQ8_0 && workspace.quantized) {
    const auto split = SplitActivations::In(workspace.quantized, n, w.cols);
    SplitRows<<<n, 256, 0, s>>>(x, split, w.cols);
    if (w.rows >= 1024)
      SplitQ8Gemm<256, 64, 8, 1>
          <<<dim3((n + 63) / 64, (w.rows + 255) / 256), 256, 0, s>>>(
              w.data, split, out, n, w.rows, w.cols);
    else
      SplitQ8Gemm<64, 64, 4, 2>
          <<<dim3((n + 63) / 64, (w.rows + 63) / 64), 256, 0, s>>>(
              w.data, split, out, n, w.rows, w.cols);
    return;
  }
  if (w.type == core::GgmlType::kF16)
    F16Dense<<<dim3((w.rows + 3) / 4, n), 128, 0, s>>>(w, x, out, n);
  else if (n == 1)
    gufo::hip::LaunchGEMV(w.data, w.type, x, out, w.rows, w.cols, s);
  else if (quant::QuantizedBlockElements(w.type))
    gufo::hip::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, out, n, w.rows,
                                          w.cols, s);
  else if (w.type == core::GgmlType::kBF16)
    gufo::hip::LaunchExactBf16GEMMFp32SmallBatch(w.data, x, out, n, w.rows,
                                                 w.cols, s);
  else
    for (int i = 0; i < n; ++i)
      gufo::hip::LaunchGEMV(w.data, w.type, x + std::size_t(i) * w.cols,
                            out + std::size_t(i) * w.rows, w.rows, w.cols, s);
}
void ShortConv(const float* p, const float* w, float* h, float* out, int n,
               int H, int L, hipStream_t s) {
  Convolution<<<dim3((H + 127) / 128, n), 128, 0, s>>>(p, w, h, out, n, H, L);
  ConvolutionHistory<<<(H + 127) / 128, 128, 0, s>>>(p, h, n, H, L);
}
void Rope(float* x, const float* f, int n, int h, int d, int p, float theta,
          hipStream_t s) {
  const int threads = d / 2 * std::max(1, 256 / (d / 2)), rows = threads / (d / 2);
  Rotary<<<(n * h + rows * kRopeRows - 1) / (rows * kRopeRows), threads, 0, s>>>(
      x, f, n, h, d, p, theta);
}
void Attention(const float* q, const float* k, const float* v, const float* g,
               const float* ck, const float* cv, float* out, int n, int h,
               int kh, int d, int p, int cap, int win, hipStream_t s) {
  const int group = h / kh;
  if (d == 128 && group <= 8 && 64 % group == 0 && n >= 64 / group) {
    TiledAttn<128><<<dim3((n + 64 / group - 1) / (64 / group), kh), 256, 0, s>>>(
        q, k, v, g, ck, cv, out, n, h, kh, p, cap, win);
    return;
  }
  Attn<<<dim3(h, n), 32, 0, s>>>(q, k, v, g, ck, cv, out, h, kh, d, p, cap,
                                 win);
}
void StoreKV(const float* k, const float* v, float* ck, float* cv, int n,
             int stride, int p, int cap, hipStream_t s) {
  Cache<<<(std::size_t(std::min(n, cap)) * stride + 255) / 256, 256, 0, s>>>(
      k, v, ck, cv, n, stride, p, cap);
}
}  // namespace gufo::models::lfm3::gpu
