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
#include "src/models/qwen/hip/ops/swiglu.hpp"
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
// Block-wide sum in the shared RMSNorm's order: a per-thread FMA chain over
// strided elements, then a halving tree in LDS.
__device__ float RowSquares(float sum, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = sum;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (tid < stride)
      scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  sum = scratch[0];
  __syncthreads();
  return sum;
}
constexpr int kNormItems = 16;
// One block per row: x += post_norm(branch), then z = norm(x) as FP32 rows
// or as split prompt activations. A null branch only normalizes x; a null
// norm weight only updates x. Each step keeps the separate kernels' rounding.
__global__ void ResidualNormKernel(const float* __restrict__ branch,
                                   const float* __restrict__ post_weight,
                                   float* __restrict__ x,
                                   const float* __restrict__ weight, float* z,
                                   SplitActivations split, int H, float eps) {
  __shared__ float scratch[256];
  const int row = blockIdx.x, tid = threadIdx.x, items = H / 256;
  float* xr = x + std::size_t(row) * H;
  // Every load is issued before the first reduction.
  float v[kNormItems], pw[kNormItems], nw[kNormItems];
  for (int i = 0; i < items; ++i) {
    v[i] = xr[tid + 256 * i];
    pw[i] = branch && post_weight ? post_weight[tid + 256 * i] : 0;
    nw[i] = weight ? weight[tid + 256 * i] : 0;
  }
  if (branch) {
    float b[kNormItems], squares = 0;
    for (int i = 0; i < items; ++i) {
      b[i] = branch[std::size_t(row) * H + tid + 256 * i];
      squares = __fmaf_rn(b[i], b[i], squares);
    }
    if (post_weight) {
      const float inv = rsqrtf(RowSquares(squares, scratch) / float(H) + eps);
      for (int i = 0; i < items; ++i)
        b[i] = (b[i] * inv) * pw[i];
    }
    for (int i = 0; i < items; ++i) {
      v[i] = v[i] + b[i];
      xr[tid + 256 * i] = v[i];
    }
  }
  if (!weight)
    return;
  float squares = 0;
  for (int i = 0; i < items; ++i)
    squares = __fmaf_rn(v[i], v[i], squares);
  const float inv = rsqrtf(RowSquares(squares, scratch) / float(H) + eps);
  float maximum = 0;
  for (int i = 0; i < items; ++i) {
    v[i] = (v[i] * inv) * nw[i];
    maximum = fmaxf(maximum, fabsf(v[i]));
  }
  if (z) {
    for (int i = 0; i < items; ++i)
      z[std::size_t(row) * H + tid + 256 * i] = v[i];
    return;
  }
  for (int d = 16; d; d /= 2)
    maximum = fmaxf(maximum, __shfl_xor(maximum, d, 32));
  if (!(tid % 32))
    scratch[tid / 32] = maximum;
  __syncthreads();
  maximum = 0;
  for (int w = 0; w < 8; ++w)
    maximum = fmaxf(maximum, scratch[w]);
  const int e = maximum > 0 ? 14 - ilogbf(maximum) : 0;
  if (!tid)
    split.scale[row] = ldexpf(1.0F, -e);
  for (int i = 0; i < items; ++i) {
    const float scaled = ldexpf(v[i], e);
    const __half h = __float2half_rn(scaled);
    const auto at = std::size_t(row) * H + tid + 256 * i;
    split.high[at] = h;
    split.low[at] = __float2half_rn(scaled - __half2float(h));
  }
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
__global__ void ConvolutionHistory(const float* p, float* history, int n, int H,
                                   int L) {
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
// Frequencies use a double-precision power rounded to FP32, computed once.
__global__ void Frequencies(float* out, int D, float theta) {
  int pair = threadIdx.x;
  if (pair < D / 2)
    out[pair] = float(pow(double(theta), double(-2.0F * pair / D)));
}
// Each thread owns one frequency pair and rotates kRopeRows (token, head) rows.
constexpr int kRopeRows = 8;
__global__ void Rotary(float* x, const float* frequencies, const float* factors,
                       int n, int heads, int D, int position) {
  int pair = threadIdx.x % (D / 2), slots = blockDim.x / (D / 2);
  int first = blockIdx.x * slots * kRopeRows + threadIdx.x / (D / 2);
  float frequency = frequencies[pair];
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
template<int D, bool bidirectional = false>
__launch_bounds__(256) __global__
    void TiledAttn(const float* __restrict__ q, const float* __restrict__ k,
                   const float* __restrict__ v, const float* __restrict__ gate,
                   const float* __restrict__ ck, const float* __restrict__ cv,
                   float* __restrict__ y, float* __restrict__ partial, int n,
                   int heads, int kh, int position, int capacity, int window,
                   int span) {
  constexpr int kRows = 8, kKeys = 32, kSlices = D / 16;
  static_assert(D == 128);
  __shared__ float4 sk[kKeys][D / 4 + 1], sv[kKeys][D / 4];
  const int group = heads / kh, queries = 64 / group;
  const int g = blockIdx.z, split = blockIdx.y, q0 = blockIdx.x * queries,
            tid = threadIdx.x;
  const int lane = tid % 32, wave = tid / 32;
  const int h = g * group + wave * kRows / queries;
  const int wq0 = q0 + wave * kRows % queries;
  const int stride = kh * D;
  float qreg[kRows][kSlices];
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int m = 0; m < kSlices; ++m)
      qreg[r][m] =
          wq0 + r < n
              ? q[(std::size_t(wq0 + r) * heads + h) * D + 16 * m + lane % 16]
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
  // Long histories split their keys into spans of span keys, one per
  // blockIdx.y; blocks of one span run together and share its K/V in cache.
  const int last_query =
      min(bidirectional ? n - 1 : position + min(q0 + queries, n) - 1,
          (split + 1) * span - 1);
  const int begin =
      max(window > 0 ? max(0, position + q0 + 1 - window) : 0, split * span);
  const int wave_last =
      bidirectional ? n - 1 : position + min(wq0 + kRows, n) - 1;
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
      const bool valid = wq0 + r < n && t >= first &&
                         (bidirectional || t <= query) && t <= last_query;
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
    if (gridDim.y > 1) {
      float* out =
          partial + ((std::size_t(split) * n + row) * heads + h) * (D + 2);
      if (!lane) {
        out[0] = maximum[r];
        out[1] = total;
      }
      reinterpret_cast<float4*>(out + 2)[lane] = acc[r];
      continue;
    }
    const float g_scale = 1 + expf(-gate[row * heads + h]);
    reinterpret_cast<float4*>(y + (std::size_t(row) * heads + h) * D)[lane] =
        make_float4((acc[r].x / total) / g_scale, (acc[r].y / total) / g_scale,
                    (acc[r].z / total) / g_scale, (acc[r].w / total) / g_scale);
  }
}
// One decode query per head. A block owns one KV group and a span of keys,
// processed in kDecodeKeys tiles with an online softmax; four threads split
// each key's dot products and all heads of the group share every loaded key
// and value row. With several spans, each block stores its maximum,
// denominator and unnormalized output for DecodeCombine.
constexpr int kDecodeKeys = 64, kMaxGroup = 8, kMaxDecodeSplits = 64;
template<int D>
__launch_bounds__(256) __global__
    void DecodeAttn(const float* __restrict__ q, const float* __restrict__ k,
                    const float* __restrict__ v, const float* __restrict__ gate,
                    const float* __restrict__ ck, const float* __restrict__ cv,
                    float* __restrict__ y, float* __restrict__ partial,
                    int heads, int kh, int position, int capacity, int window,
                    int span) {
  constexpr int kChunks = D / 16;  // float4 chunks per thread and row
  __shared__ float4 sq[kMaxGroup][D / 4];
  __shared__ float4 sv[kDecodeKeys][D / 4];
  __shared__ float sp[kMaxGroup][kDecodeKeys];
  __shared__ float reduce[8][kMaxGroup];
  const int g = blockIdx.x, split = blockIdx.y, splits = gridDim.y;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int j = tid / 4, part = tid % 4;
  const int group = heads / kh, stride = kh * D;
  const int first = window > 0 ? max(0, position + 1 - window) : 0;
  const int begin = first + split * span;
  const int end = min(begin + span, position + 1);
  for (int i = tid; i < group * D / 4; i += 256)
    sq[i / (D / 4)][i % (D / 4)] = reinterpret_cast<const float4*>(
        q + (std::size_t(g) * group + i / (D / 4)) * D)[i % (D / 4)];
  // Reduces one value per head across the block (maximum or sum).
  const auto block_reduce = [&](float (&value)[kMaxGroup], bool maximum) {
    for (int hh = 0; hh < group; ++hh) {
      for (int offset = 16; offset; offset /= 2) {
        const float other = __shfl_xor(value[hh], offset, 32);
        value[hh] = maximum ? fmaxf(value[hh], other) : value[hh] + other;
      }
      if (!lane)
        reduce[wave][hh] = value[hh];
    }
    __syncthreads();
    for (int hh = 0; hh < group; ++hh) {
      value[hh] = reduce[0][hh];
      for (int w = 1; w < 8; ++w)
        value[hh] = maximum ? fmaxf(value[hh], reduce[w][hh])
                            : value[hh] + reduce[w][hh];
    }
    __syncthreads();
  };
  // Threads split the group's heads; each owns D * group / 256 dimensions.
  const int per_head = 256 / group, dims = D / per_head;
  const int oh = tid / per_head, d0 = (tid % per_head) * dims;
  const auto* sv_floats = reinterpret_cast<const float*>(sv);
  const float scale = rsqrtf(float(D));
  float maxima[kMaxGroup], sums[kMaxGroup], acc[kMaxGroup / 2] = {};
  for (int hh = 0; hh < kMaxGroup; ++hh) {
    maxima[hh] = -INFINITY;
    sums[hh] = 0;
  }
  for (int base = begin; base < end; base += kDecodeKeys) {
    const int keys = min(kDecodeKeys, end - base), t = base + j;
    const bool valid = j < keys;
    float4 kk[kChunks];
    __syncthreads();
    if (valid) {
      const float* key = t >= position
                             ? k + std::size_t(t - position) * stride
                             : ck + std::size_t(t % capacity) * stride;
      const float* value = t >= position
                               ? v + std::size_t(t - position) * stride
                               : cv + std::size_t(t % capacity) * stride;
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        kk[c] = reinterpret_cast<const float4*>(key + g * D)[c * 4 + part];
        sv[j][c * 4 + part] =
            reinterpret_cast<const float4*>(value + g * D)[c * 4 + part];
      }
    }
    __syncthreads();
    float score[kMaxGroup], tile[kMaxGroup];
    for (int hh = 0; hh < group; ++hh) {
      float dot = 0;
      if (valid)
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          const float4 qq = sq[hh][c * 4 + part];
          dot = fmaf(qq.x, kk[c].x, dot);
          dot = fmaf(qq.y, kk[c].y, dot);
          dot = fmaf(qq.z, kk[c].z, dot);
          dot = fmaf(qq.w, kk[c].w, dot);
        }
      dot += __shfl_xor(dot, 1, 4);
      dot += __shfl_xor(dot, 2, 4);
      score[hh] = valid ? dot * scale : -INFINITY;
      tile[hh] = score[hh];
    }
    block_reduce(tile, true);
    float alpha[kMaxGroup];
    for (int hh = 0; hh < group; ++hh) {
      const float next = fmaxf(maxima[hh], tile[hh]);
      alpha[hh] = expf(maxima[hh] - next);
      maxima[hh] = next;
      const float p = valid ? expf(score[hh] - next) : 0;
      if (!part)
        sp[hh][j] = p;
      // Each key appears in four lanes; count it once.
      tile[hh] = part ? 0 : p;
    }
    block_reduce(tile, false);
    for (int hh = 0; hh < group; ++hh)
      sums[hh] = sums[hh] * alpha[hh] + tile[hh];
    float own_alpha = alpha[0];
    for (int hh = 1; hh < group; ++hh)
      own_alpha = oh == hh ? alpha[hh] : own_alpha;
    for (int d = 0; d < dims; ++d)
      acc[d] *= own_alpha;
    for (int key = 0; key < keys; ++key) {
      const float p = sp[oh][key];
      for (int d = 0; d < dims; ++d)
        acc[d] = fmaf(p, sv_floats[key * D + d0 + d], acc[d]);
    }
  }
  float own_max = maxima[0], own_sum = sums[0];
  for (int hh = 1; hh < group; ++hh)
    if (oh == hh) {
      own_max = maxima[hh];
      own_sum = sums[hh];
    }
  const int h = g * group + oh;
  if (splits == 1) {
    const float g_scale = 1 + expf(-gate[h]);
    for (int d = 0; d < dims; ++d)
      y[std::size_t(h) * D + d0 + d] = (acc[d] / own_sum) / g_scale;
    return;
  }
  float* out = partial + (std::size_t(split) * heads + h) * (D + 2);
  if (!(tid % per_head)) {
    out[0] = own_max;
    out[1] = own_sum;
  }
  for (int d = 0; d < dims; ++d)
    out[2 + d0 + d] = acc[d];
}
// Merges TiledAttn key spans: one block per (row, head).
template<int D>
__global__ void PromptCombine(const float* __restrict__ partial,
                              const float* __restrict__ gate,
                              float* __restrict__ y, int rows, int splits) {
  const int rh = blockIdx.x, d = threadIdx.x;
  float maximum = -INFINITY;
  for (int s = 0; s < splits; ++s)
    maximum = fmaxf(maximum, partial[(std::size_t(s) * rows + rh) * (D + 2)]);
  float sum = 0, acc = 0;
  for (int s = 0; s < splits; ++s) {
    const float* part = partial + (std::size_t(s) * rows + rh) * (D + 2);
    const float w = expf(part[0] - maximum);
    sum = fmaf(part[1], w, sum);
    acc = fmaf(part[2 + d], w, acc);
  }
  y[std::size_t(rh) * D + d] = (acc / sum) / (1 + expf(-gate[rh]));
}
template<int D>
__global__ void DecodeCombine(const float* __restrict__ partial,
                              const float* __restrict__ gate,
                              float* __restrict__ y, int heads, int splits) {
  const int h = blockIdx.x, d = threadIdx.x;
  float maximum = -INFINITY;
  for (int s = 0; s < splits; ++s)
    maximum = fmaxf(maximum, partial[(std::size_t(s) * heads + h) * (D + 2)]);
  float sum = 0, acc = 0;
  for (int s = 0; s < splits; ++s) {
    const float* part = partial + (std::size_t(s) * heads + h) * (D + 2);
    const float w = expf(part[0] - maximum);
    sum = fmaf(part[1], w, sum);
    acc = fmaf(part[2 + d], w, acc);
  }
  y[std::size_t(h) * D + d] = (acc / sum) / (1 + expf(-gate[h]));
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
namespace {
bool SplitPrompt(const Tensor& w, int n, DenseWorkspace workspace) {
  return n >= 32 && w.type == core::GgmlType::kQ8_0 && workspace.quantized;
}
template<bool kSwiGLU = false, bool kRaw = false>
void LaunchSplit(std::span<const Output> outputs, SplitActivations split, int n,
                 int K, hipStream_t s) {
  constexpr int BN = 64;
  Projections jobs{};
  jobs.count = int(outputs.size());
  int row_blocks = 0;
  const bool wide = outputs[0].weight->rows >= 1024;
  const int tile_rows = (wide ? 256 : 64) / (kSwiGLU ? 2 : 1);
  for (int i = 0; i < jobs.count; ++i) {
    const auto& w = *outputs[i].weight;
    jobs.job[i] = {w.data, outputs[i].out, int(w.rows)};
    jobs.first_block[i] = row_blocks;
    if (!kSwiGLU || i == 0)
      row_blocks += (int(w.rows) + tile_rows - 1) / tile_rows;
  }
  const dim3 grid((n + BN - 1) / BN, row_blocks);
  if (kSwiGLU)
    jobs.count = 1;  // Both weights belong to every row block.
  if (wide)
    SplitQ8Gemm<256, BN, 8, 1, kSwiGLU, kRaw>
        <<<grid, 256, 0, s>>>(jobs, split, n, K);
  else
    SplitQ8Gemm<64, BN, 4, 2, kSwiGLU, kRaw>
        <<<grid, 256, 0, s>>>(jobs, split, n, K);
}
}  // namespace
bool SharesSplit(std::span<const Output> outputs, int n,
                 DenseWorkspace workspace) {
  bool shared = outputs.size() <= Projections::kMax;
  for (const auto& o : outputs)
    shared = shared && o.weight->cols == outputs[0].weight->cols &&
             SplitPrompt(*o.weight, n, workspace);
  return shared;
}
bool ResidualNorm(const float* branch, const Tensor* post_norm, float* x,
                  const Tensor* norm, float* z, int n, int H, float eps,
                  bool split, hipStream_t s, DenseWorkspace workspace) {
  if (H % 256 || H / 256 > kNormItems)
    return false;
  ResidualNormKernel<<<n, 256, 0, s>>>(
      branch, post_norm ? static_cast<const float*>(post_norm->data) : nullptr,
      x, norm ? static_cast<const float*>(norm->data) : nullptr,
      split ? nullptr : z, SplitActivations::In(workspace.quantized, n, H), H,
      eps);
  return true;
}
void Project(std::span<const Output> outputs, const float* x, int n,
             hipStream_t s, DenseWorkspace workspace, bool split_ready) {
  const auto K = outputs[0].weight->cols;
  if (!SharesSplit(outputs, n, workspace)) {
    for (const auto& o : outputs)
      Dense(*o.weight, x, o.out, n, s, workspace);
    return;
  }
  const auto split = SplitActivations::In(workspace.quantized, n, K);
  if (!split_ready)
    SplitRows<<<n, 256, 0, s>>>(x, split, K);
  // Wide and narrow projections use different tiles.
  Output wide[Projections::kMax], narrow[Projections::kMax];
  std::size_t wide_count = 0, narrow_count = 0;
  for (const auto& o : outputs)
    (o.weight->rows >= 1024 ? wide[wide_count++] : narrow[narrow_count++]) = o;
  if (wide_count)
    LaunchSplit({wide, wide_count}, split, n, K, s);
  if (narrow_count)
    LaunchSplit({narrow, narrow_count}, split, n, K, s);
}
void ProjectSwiGLU(const Tensor& gate, const Tensor& up, const float* x,
                   float* out, float* up_scratch, int n, hipStream_t s,
                   DenseWorkspace workspace, bool split_ready) {
  const Output outputs[] = {{&gate, out}, {&up, nullptr}};
  if (SharesSplit(outputs, n, workspace) && gate.rows == up.rows) {
    const auto split = SplitActivations::In(workspace.quantized, n, gate.cols);
    if (!split_ready)
      SplitRows<<<n, 256, 0, s>>>(x, split, gate.cols);
    LaunchSplit<true>(outputs, split, n, gate.cols, s);
    return;
  }
  Dense(gate, x, out, n, s, workspace);
  Dense(up, x, up_scratch, n, s, workspace);
  gufo::hip::LaunchBatchedSwiGLUActivation(out, up_scratch, out, nullptr,
                                           std::size_t(n) * gate.rows, s);
}
void ProjectFfn(const Tensor& gate, const Tensor& up, const Tensor& down,
                const float* x, float* hidden, float* up_scratch, float* out,
                int n, hipStream_t s, DenseWorkspace workspace,
                bool split_ready) {
  const Output first[] = {{&gate, hidden}, {&up, nullptr}};
  if (workspace.row_max && SharesSplit(first, n, workspace) &&
      gate.rows == up.rows && SplitPrompt(down, n, workspace) &&
      down.cols == gate.rows) {
    auto split = SplitActivations::In(workspace.quantized, n, gate.cols);
    if (!split_ready)
      SplitRows<<<n, 256, 0, s>>>(x, split, gate.cols);
    split.row_max = workspace.row_max;
    (void)hipMemsetAsync(workspace.row_max, 0, std::size_t(n) * 4, s);
    LaunchSplit<true>(first, split, n, gate.cols, s);
    SplitActivations raw{};
    raw.raw = hidden;
    raw.row_max = workspace.row_max;
    const Output second[] = {{&down, out}};
    LaunchSplit<false, true>(second, raw, n, down.cols, s);
    return;
  }
  ProjectSwiGLU(gate, up, x, hidden, up_scratch, n, s, workspace, split_ready);
  Dense(down, hidden, out, n, s, workspace);
}
void Dense(const Tensor& w, const float* x, float* out, int n, hipStream_t s,
           DenseWorkspace workspace) {
  if (SplitPrompt(w, n, workspace)) {
    const Output output{&w, out};
    Project({&output, 1}, x, n, s, workspace);
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
void RopeFrequencies(float* out, int d, float theta, hipStream_t s) {
  Frequencies<<<1, d / 2, 0, s>>>(out, d, theta);
}
void Rope(float* x, const float* frequencies, const float* f, int n, int h,
          int d, int p, hipStream_t s) {
  const int threads = d / 2 * std::max(1, 256 / (d / 2)),
            rows = threads / (d / 2);
  Rotary<<<(n * h + rows * kRopeRows - 1) / (rows * kRopeRows), threads, 0,
           s>>>(x, frequencies, f, n, h, d, p);
}
constexpr int kMinPromptSpan = 8192, kMaxPromptSplits = 16;
int PromptSplits(int keys, int& span) {
  span = std::max(kMinPromptSpan,
                  (keys + kMaxPromptSplits - 1) / kMaxPromptSplits);
  return (keys + span - 1) / span;
}
std::size_t AttentionScratchBytes(int heads, int dim, int context, int rows) {
  int span;
  const int prompt = PromptSplits(context, span);
  const std::size_t decode =
      std::min((context + kDecodeKeys - 1) / kDecodeKeys, kMaxDecodeSplits);
  return std::max(decode, std::size_t(prompt > 1 ? prompt * rows : 0)) * heads *
         (dim + 2) * sizeof(float);
}
void Attention(const float* q, const float* k, const float* v, const float* g,
               const float* ck, const float* cv, float* out, int n, int h,
               int kh, int d, int p, int cap, int win, hipStream_t s,
               float* scratch) {
  const int group = h / kh;
  if (n == 1 && scratch && d == 128 && group >= 2 && group <= kMaxGroup &&
      256 % group == 0) {
    const int first = win > 0 ? std::max(0, p + 1 - win) : 0;
    const int tiles = (p + 1 - first + kDecodeKeys - 1) / kDecodeKeys;
    const int span =
        (tiles + kMaxDecodeSplits - 1) / kMaxDecodeSplits * kDecodeKeys;
    const int splits = (p + 1 - first + span - 1) / span;
    DecodeAttn<128><<<dim3(kh, splits), 256, 0, s>>>(
        q, k, v, g, ck, cv, out, scratch, h, kh, p, cap, win, span);
    if (splits > 1)
      DecodeCombine<128><<<h, 128, 0, s>>>(scratch, g, out, h, splits);
    return;
  }
  if (d == 128 && group <= 8 && 64 % group == 0 && n >= 64 / group) {
    int span;
    const int splits = scratch && !win ? PromptSplits(p + n, span) : 1;
    if (splits == 1)
      span = p + n;
    TiledAttn<128>
        <<<dim3((n + 64 / group - 1) / (64 / group), splits, kh), 256, 0, s>>>(
            q, k, v, g, ck, cv, out, scratch, n, h, kh, p, cap, win, span);
    if (splits > 1)
      PromptCombine<128><<<n * h, 128, 0, s>>>(scratch, g, out, n * h, splits);
    return;
  }
  Attn<<<dim3(h, n), 32, 0, s>>>(q, k, v, g, ck, cv, out, h, kh, d, p, cap,
                                 win);
}
void BidirectionalAttention(const float* q, const float* k, const float* v,
                            const float* g, float* out, int n, int h, int kh,
                            hipStream_t s) {
  const int queries = 64 / (h / kh);
  TiledAttn<128, true><<<dim3((n + queries - 1) / queries, 1, kh), 256, 0, s>>>(
      q, k, v, g, nullptr, nullptr, out, nullptr, n, h, kh, 0, n, 0, n);
}
void StoreKV(const float* k, const float* v, float* ck, float* cv, int n,
             int stride, int p, int cap, hipStream_t s) {
  Cache<<<(std::size_t(std::min(n, cap)) * stride + 255) / 256, 256, 0, s>>>(
      k, v, ck, cv, n, stride, p, cap);
}
}  // namespace gufo::models::lfm3::gpu
