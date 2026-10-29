#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <cfloat>
#include <cmath>

#include "src/models/nemotron35_bolt/kernels/rocm/drafter_kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/small_gemv.hip.hpp"

namespace gufo::models::nemotron35_bolt::rocm {
namespace {

constexpr int kBlock = 256;
constexpr int kLanes = 32;
constexpr int kHeadDim = 128;
constexpr int kGroup = 16;    ///< Query heads per KV head.
constexpr int kKeyTile = 32;  ///< Keys per attention tile.
constexpr int kTopK = kLiliCorrTopK;
constexpr int kFeatures = 5;

__device__ __forceinline__ float Silu(float x) {
  return x / (1.0F + expf(-x));
}

__device__ __forceinline__ float LaneSum(float v) {
  for (int offset = kLanes / 2; offset > 0; offset /= 2)
    v += __shfl_xor(v, offset, kLanes);
  return v;
}

/// Sum over the block; every thread receives the result.
__device__ float BlockSum(float v, float* scratch) {
  v = LaneSum(v);
  const int lane = threadIdx.x % kLanes;
  const int group = threadIdx.x / kLanes;
  __syncthreads();
  if (lane == 0)
    scratch[group] = v;
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(blockDim.x) / kLanes; ++i)
    total += scratch[i];
  return total;
}

/// Maximum over the block; every thread receives the result.
__device__ float BlockMax(float v, float* scratch) {
  for (int offset = kLanes / 2; offset > 0; offset /= 2)
    v = fmaxf(v, __shfl_xor(v, offset, kLanes));
  const int lane = threadIdx.x % kLanes;
  const int group = threadIdx.x / kLanes;
  __syncthreads();
  if (lane == 0)
    scratch[group] = v;
  __syncthreads();
  float best = -INFINITY;
  for (int i = 0; i < static_cast<int>(blockDim.x) / kLanes; ++i)
    best = fmaxf(best, scratch[i]);
  return best;
}

unsigned Blocks(std::size_t n, int block = kBlock) {
  return static_cast<unsigned>((n + block - 1) / block);
}

__global__ void ToBf16Kernel(const float* x, __hip_bfloat16* out,
                             std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = __float2bfloat16(x[i]);
}

__global__ void ScaleBiasKernel(float* x, float scale, const float* bias,
                                int rows, int cols, Activation act) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * cols)
    return;
  float v = x[i] * scale;
  if (bias != nullptr)
    v += bias[i % static_cast<std::size_t>(cols)];
  x[i] = act == Activation::kSilu ? Silu(v) : v;
}

__global__ void SiluGateKernel(const float* gate, float gate_scale,
                               const float* up, float up_scale, float* out,
                               std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = Silu(gate[i] * gate_scale) * (up[i] * up_scale);
}

__global__ void DflashConvKernel(const float* x, const float* dynamic,
                                 const float* base, float* out, int n_tokens,
                                 int block, int hidden, int group, int kernel,
                                 int side) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * hidden)
    return;
  const int t = static_cast<int>(i / hidden);
  const int c = static_cast<int>(i % hidden);
  const int groups = hidden / group;
  const float* coeff =
      dynamic + static_cast<std::size_t>(t) * 2 * kernel * groups;
  const int taps = min(kernel, t % block + 1);
  float acc = 0.0F;
  for (int tap = 0; tap < taps; ++tap) {
    const int k = side * kernel + tap;
    const float w = coeff[k * groups + c / group] +
                    base[static_cast<std::size_t>(k) * hidden + c];
    acc += w * x[static_cast<std::size_t>(t - tap) * hidden + c];
  }
  out[i] = acc;
}

__global__ void RopeNeoxKernel(float* x, const int* first, int n_tokens,
                               int heads, int head_dim, float theta_scale) {
  const int half = head_dim / 2;
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * heads * half)
    return;
  const int pair = static_cast<int>(i % half);
  const std::size_t row = i / half;
  const int t = static_cast<int>(row / heads);
  const float angle =
      static_cast<float>(*first + t) * powf(theta_scale, static_cast<float>(pair));
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* r = x + row * head_dim;
  const float a = r[pair];
  const float b = r[pair + half];
  r[pair] = a * c - b * s;
  r[pair + half] = a * s + b * c;
}

__global__ void RingAppendKernel(const float* k, const float* v,
                                 __half* k_ring, __half* v_ring,
                                 const int* first, int n_tokens, int kv_dim,
                                 int ring) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * kv_dim)
    return;
  const int t = static_cast<int>(i / kv_dim);
  const int j = static_cast<int>(i % kv_dim);
  const std::size_t slot =
      static_cast<std::size_t>((*first + t) % ring) * kv_dim + j;
  k_ring[slot] = __float2half(k[i]);
  v_ring[slot] = __float2half(v[i]);
}

__global__ void RowsToRingKernel(const float* x, float* ring,
                                 const int* first, int n_tokens, int width,
                                 int ring_rows) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * width)
    return;
  const int t = static_cast<int>(i / width);
  const int c = static_cast<int>(i % width);
  ring[static_cast<std::size_t>((*first + t) % ring_rows) * width + c] = x[i];
}

/// One block per (query token, KV head): 16 query heads of 16 lanes each.
/// Keys stream through LDS in tiles of 32; the sink starts each head's online
/// softmax as a logit with a zero value.
__global__ __launch_bounds__(kBlock) void DrafterAttentionKernel(
    const float* q, const __half* k_ring, const __half* v_ring,
    const float* sinks, float* out, const int* first, int n_tokens, int heads,
    int kv_heads, int ring, int window, float scale) {
  __shared__ float q_s[kGroup][kHeadDim];
  __shared__ float k_s[kKeyTile][kHeadDim + 1];
  __shared__ float v_s[kKeyTile][kHeadDim];
  __shared__ float p_s[kGroup][kKeyTile];
  const int t = static_cast<int>(blockIdx.x);
  const int kvh = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int h = tid / kGroup;  // head within the group
  const int lane = tid % kGroup;
  const int head = kvh * kGroup + h;
  const int kv_dim = kv_heads * kHeadDim;
  const int pos = *first + t;
  const int lo = max(0, pos - window + 1);
  const int hi = *first + n_tokens - 1;

  for (int i = tid; i < kGroup * kHeadDim; i += kBlock)
    q_s[i / kHeadDim][i % kHeadDim] =
        q[(static_cast<std::size_t>(t) * heads + kvh * kGroup) * kHeadDim + i];
  float m = sinks[head];
  float l = 1.0F;
  float acc[8] = {};
  for (int j0 = lo; j0 <= hi; j0 += kKeyTile) {
    __syncthreads();
    for (int i = tid; i < kKeyTile * kHeadDim; i += kBlock) {
      const int j = i / kHeadDim;
      const int d = i % kHeadDim;
      const int p = j0 + j;
      float kv = 0.0F;
      float vv = 0.0F;
      if (p <= hi) {
        const std::size_t slot =
            static_cast<std::size_t>(p % ring) * kv_dim + kvh * kHeadDim + d;
        kv = __half2float(k_ring[slot]);
        vv = __half2float(v_ring[slot]);
      }
      k_s[j][d] = kv;
      v_s[j][d] = vv;
    }
    __syncthreads();
    float s[2];
    float tile_max = -INFINITY;
    for (int r = 0; r < 2; ++r) {
      const int j = lane + r * kGroup;
      float dot = 0.0F;
      for (int d = 0; d < kHeadDim; ++d)
        dot += q_s[h][d] * k_s[j][d];
      s[r] = j0 + j <= hi ? dot * scale : -INFINITY;
      tile_max = fmaxf(tile_max, s[r]);
    }
    for (int offset = kGroup / 2; offset > 0; offset /= 2)
      tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, kGroup));
    const float m_new = fmaxf(m, tile_max);
    const float correction = expf(m - m_new);
    float sum = 0.0F;
    for (int r = 0; r < 2; ++r) {
      const float p = expf(s[r] - m_new);
      p_s[h][lane + r * kGroup] = p;
      sum += p;
    }
    for (int offset = kGroup / 2; offset > 0; offset /= 2)
      sum += __shfl_xor(sum, offset, kGroup);
    l = l * correction + sum;
    m = m_new;
    __syncthreads();
    for (int d = 0; d < 8; ++d) {
      float a = acc[d] * correction;
      for (int j = 0; j < kKeyTile; ++j)
        a += p_s[h][j] * v_s[j][lane * 8 + d];
      acc[d] = a;
    }
  }
  float* dst = out + (static_cast<std::size_t>(t) * heads + head) * kHeadDim +
               lane * 8;
  for (int d = 0; d < 8; ++d)
    dst[d] = acc[d] / l;
}

/// One block per row: per-thread top-8 lists and online log-sum-exp, then
/// eight block-wide selections.
__global__ __launch_bounds__(kBlock) void TopK8Kernel(const float* logits,
                                                      int vocab,
                                                      std::int32_t* ids,
                                                      float* values,
                                                      float* lse) {
  __shared__ float scratch[kBlock / kLanes];
  __shared__ float cand_v[kBlock];
  __shared__ int cand_i[kBlock];
  __shared__ int winner;
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  float top_v[kTopK];
  int top_i[kTopK];
  for (int k = 0; k < kTopK; ++k) {
    top_v[k] = -INFINITY;
    top_i[k] = INT32_MAX;
  }
  float m = -INFINITY;
  float s = 0.0F;
  for (int i = tid; i < vocab; i += kBlock) {
    const float v = x[i];
    if (v > m) {
      s = s * expf(m - v) + 1.0F;
      m = v;
    } else {
      s += expf(v - m);
    }
    if (v > top_v[kTopK - 1]) {
      int k = kTopK - 1;
      while (k > 0 && v > top_v[k - 1]) {
        top_v[k] = top_v[k - 1];
        top_i[k] = top_i[k - 1];
        --k;
      }
      top_v[k] = v;
      top_i[k] = i;
    }
  }
  const float row_max = BlockMax(m, scratch);
  const float total =
      BlockSum(m == -INFINITY ? 0.0F : s * expf(m - row_max), scratch);
  int head = 0;
  for (int k = 0; k < kTopK; ++k) {
    __syncthreads();
    cand_v[tid] = head < kTopK ? top_v[head] : -INFINITY;
    cand_i[tid] = head < kTopK ? top_i[head] : INT32_MAX;
    __syncthreads();
    if (tid == 0) {
      int best = 0;
      for (int j = 1; j < kBlock; ++j) {
        if (cand_v[j] > cand_v[best] ||
            (cand_v[j] == cand_v[best] && cand_i[j] < cand_i[best]))
          best = j;
      }
      winner = best;
      ids[row * kTopK + k] = cand_i[best];
      values[row * kTopK + k] = cand_v[best];
    }
    __syncthreads();
    if (tid == winner)
      ++head;
  }
  if (tid == 0)
    lse[row] = row_max + logf(total);
}

__global__ void LiliCorrFeaturesKernel(const float* values, const float* lse,
                                       const float* w, const float* b,
                                       float* out, int nodes) {
  const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= nodes)
    return;
  const int slot = node / kTopK;
  const int rank = node % kTopK;
  const float lp = values[node] - lse[slot];
  const float lp0 = values[slot * kTopK] - lse[slot];
  const float f[kFeatures] = {lp, expf(lp), lp - lp0,
                              static_cast<float>(rank) / (kTopK - 1),
                              rank == 0 ? 1.0F : 0.0F};
  float mean = 0.0F;
  for (float v : f)
    mean += v;
  mean /= kFeatures;
  float var = 0.0F;
  for (float v : f)
    var += (v - mean) * (v - mean);
  var /= kFeatures;
  const float inv = 1.0F / sqrtf(var + 1e-5F);
  for (int i = 0; i < kFeatures; ++i)
    out[node * kFeatures + i] = (f[i] - mean) * inv * w[i] + b[i];
}

__global__ void LiliCorrCombineKernel(const float* token, const float* pass,
                                      const float* feature,
                                      const float* slot_embd,
                                      const float* rank_embd, float* x,
                                      int nodes, int width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(nodes) * width)
    return;
  const int node = static_cast<int>(i / width);
  const int c = static_cast<int>(i % width);
  const int slot = node / kTopK;
  const int rank = node % kTopK;
  const float position = slot_embd[static_cast<std::size_t>(slot) * width + c] +
                         rank_embd[static_cast<std::size_t>(rank) * width + c];
  x[i] = token[i] + pass[static_cast<std::size_t>(slot) * width + c] +
         feature[i] + position;
}

/// One block per (query node, head) with one thread per head dimension.
__global__ __launch_bounds__(kHeadDim) void NodeAttentionKernel(
    const float* qkv, const float* bias, float* out, int nodes, int heads,
    float scale) {
  __shared__ float q_s[kHeadDim];
  __shared__ float p_s[kHeadDim];
  __shared__ float norm;
  const int qi = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int d = static_cast<int>(threadIdx.x);
  const int width = heads * kHeadDim;
  const std::size_t stride = 3 * static_cast<std::size_t>(width);
  q_s[d] = qkv[qi * stride + h * kHeadDim + d];
  __syncthreads();
  if (d < nodes) {
    const float* k = qkv + d * stride + width + h * kHeadDim;
    float dot = 0.0F;
    for (int i = 0; i < kHeadDim; ++i)
      dot += q_s[i] * k[i];
    p_s[d] = dot * scale +
             bias[(static_cast<std::size_t>(h) * nodes + qi) * nodes + d];
  }
  __syncthreads();
  if (d == 0) {
    float m = -INFINITY;
    for (int j = 0; j < nodes; ++j)
      m = fmaxf(m, p_s[j]);
    float sum = 0.0F;
    for (int j = 0; j < nodes; ++j) {
      p_s[j] = expf(p_s[j] - m);
      sum += p_s[j];
    }
    norm = 1.0F / sum;
  }
  __syncthreads();
  float acc = 0.0F;
  for (int j = 0; j < nodes; ++j)
    acc += p_s[j] * qkv[j * stride + 2 * width + h * kHeadDim + d];
  out[static_cast<std::size_t>(qi) * width + h * kHeadDim + d] = acc * norm;
}

__global__ void FactorInputKernel(const float* x, const float* anchor,
                                  float* out, int nodes, int width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(nodes) * width)
    return;
  const std::size_t node = i / width;
  const int c = static_cast<int>(i % width);
  float* dst = out + node * 3 * width;
  dst[c] = x[i];
  dst[width + c] = anchor[c];
  dst[2 * width + c] = x[i] * anchor[c];
}

__global__ void BiasL2NormKernel(float* x, const float* bias, int width,
                                 float eps) {
  __shared__ float scratch[kBlock / kLanes];
  float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = row[i] + bias[i];
    row[i] = v;
    ss += v * v;
  }
  ss = BlockSum(ss, scratch);
  const float scale = 1.0F / fmaxf(sqrtf(ss), eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    row[i] *= scale;
}

/// Block 0 scores the first slot against the anchor; block 1 + (s - 1) * 8 +
/// prev scores slot s's candidates after candidate prev of slot s - 1. One
/// 32-lane group per dot product.
__global__ __launch_bounds__(kBlock) void LatticeScoresKernel(
    const float* in, const float* out, const float* anchor, float* scores,
    int width, float scale) {
  const int b = static_cast<int>(blockIdx.x);
  const int next = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const float* a;
  const float* c;
  float* dst;
  if (b == 0) {
    a = in + static_cast<std::size_t>(next) * width;
    c = anchor;
    dst = scores + next;
  } else {
    const int s = (b - 1) / kTopK + 1;
    const int prev = (b - 1) % kTopK;
    a = in + static_cast<std::size_t>(s * kTopK + next) * width;
    c = out + static_cast<std::size_t>((s - 1) * kTopK + prev) * width;
    dst = scores + kTopK + (s - 1) * kTopK * kTopK + prev * kTopK + next;
  }
  float dot = 0.0F;
  for (int i = lane; i < width; i += kLanes)
    dot += a[i] * c[i];
  dot = LaneSum(dot);
  if (lane == 0)
    *dst = dot * scale;
}

}  // namespace

hipError_t ToBf16(const float* x, __hip_bfloat16* out, std::size_t n,
                  hipStream_t stream) {
  ToBf16Kernel<<<Blocks(n), kBlock, 0, stream>>>(x, out, n);
  return hipGetLastError();
}

hipError_t GemvBf16(const __hip_bfloat16* w, const float* x, float* out,
                    int rows, int cols, int n_tokens, hipStream_t stream) {
  return small_gemv::Launch<small_gemv::Bf16Weights>(w, x, out, rows, cols,
                                                     n_tokens, stream);
}

hipError_t ScaleBias(float* x, float scale, const float* bias, int rows,
                     int cols, Activation act, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  ScaleBiasKernel<<<Blocks(n), kBlock, 0, stream>>>(x, scale, bias, rows, cols,
                                                    act);
  return hipGetLastError();
}

hipError_t SiluGate(const float* gate, float gate_scale, const float* up,
                    float up_scale, float* out, std::size_t n,
                    hipStream_t stream) {
  SiluGateKernel<<<Blocks(n), kBlock, 0, stream>>>(gate, gate_scale, up,
                                                   up_scale, out, n);
  return hipGetLastError();
}

hipError_t DflashConv(const float* x, const float* dynamic, const float* base,
                      float* out, int n_tokens, int block, int hidden,
                      int group, int kernel, int side, hipStream_t stream) {
  if (x == out || block < 1 || hidden % group != 0 || side < 0 || side > 1)
    return hipErrorInvalidValue;
  const std::size_t n = static_cast<std::size_t>(n_tokens) * hidden;
  DflashConvKernel<<<Blocks(n), kBlock, 0, stream>>>(
      x, dynamic, base, out, n_tokens, block, hidden, group, kernel, side);
  return hipGetLastError();
}

hipError_t RopeNeox(float* x, const int* first, int n_tokens, int heads,
                    int head_dim, float theta, hipStream_t stream) {
  if (head_dim % 2 != 0)
    return hipErrorInvalidValue;
  const std::size_t n = static_cast<std::size_t>(n_tokens) * heads * head_dim / 2;
  const float theta_scale = powf(theta, -2.0F / static_cast<float>(head_dim));
  RopeNeoxKernel<<<Blocks(n), kBlock, 0, stream>>>(x, first, n_tokens, heads,
                                                   head_dim, theta_scale);
  return hipGetLastError();
}

hipError_t RingAppend(const float* k, const float* v, __half* k_ring,
                      __half* v_ring, const int* first, int n_tokens,
                      int kv_dim, int ring, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * kv_dim;
  RingAppendKernel<<<Blocks(n), kBlock, 0, stream>>>(k, v, k_ring, v_ring,
                                                     first, n_tokens, kv_dim,
                                                     ring);
  return hipGetLastError();
}

hipError_t RowsToRing(const float* x, float* ring, const int* first,
                      int n_tokens, int width, int ring_rows,
                      hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * width;
  RowsToRingKernel<<<Blocks(n), kBlock, 0, stream>>>(x, ring, first, n_tokens,
                                                     width, ring_rows);
  return hipGetLastError();
}

hipError_t DrafterAttention(const float* q, const __half* k_ring,
                            const __half* v_ring, const float* sinks,
                            float* out, const int* first, int n_tokens,
                            int heads, int kv_heads, int ring, int window,
                            float scale, hipStream_t stream) {
  if (heads != kv_heads * kGroup || ring < window + n_tokens)
    return hipErrorInvalidValue;
  DrafterAttentionKernel<<<dim3(n_tokens, kv_heads), kBlock, 0, stream>>>(
      q, k_ring, v_ring, sinks, out, first, n_tokens, heads, kv_heads, ring,
      window, scale);
  return hipGetLastError();
}

hipError_t TopK8(const float* logits, int rows, int vocab, std::int32_t* ids,
                 float* values, float* lse, hipStream_t stream) {
  TopK8Kernel<<<rows, kBlock, 0, stream>>>(logits, vocab, ids, values, lse);
  return hipGetLastError();
}

hipError_t LiliCorrFeatures(const float* values, const float* lse,
                            const float* w, const float* b, float* out,
                            int slots, hipStream_t stream) {
  const int nodes = slots * kTopK;
  LiliCorrFeaturesKernel<<<Blocks(nodes, 64), 64, 0, stream>>>(values, lse, w,
                                                               b, out, nodes);
  return hipGetLastError();
}

hipError_t LiliCorrCombine(const float* token, const float* pass,
                           const float* feature, const float* slot_embd,
                           const float* rank_embd, float* x, int slots,
                           int width, hipStream_t stream) {
  const int nodes = slots * kTopK;
  const std::size_t n = static_cast<std::size_t>(nodes) * width;
  LiliCorrCombineKernel<<<Blocks(n), kBlock, 0, stream>>>(
      token, pass, feature, slot_embd, rank_embd, x, nodes, width);
  return hipGetLastError();
}

hipError_t NodeAttention(const float* qkv, const float* bias, float* out,
                         int nodes, int heads, float scale,
                         hipStream_t stream) {
  if (nodes > kHeadDim)
    return hipErrorInvalidValue;
  NodeAttentionKernel<<<dim3(nodes, heads), kHeadDim, 0, stream>>>(
      qkv, bias, out, nodes, heads, scale);
  return hipGetLastError();
}

hipError_t FactorInput(const float* x, const float* anchor, float* out,
                       int nodes, int width, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(nodes) * width;
  FactorInputKernel<<<Blocks(n), kBlock, 0, stream>>>(x, anchor, out, nodes,
                                                      width);
  return hipGetLastError();
}

hipError_t BiasL2Norm(float* x, const float* bias, int rows, int width,
                      float eps, hipStream_t stream) {
  BiasL2NormKernel<<<rows, kBlock, 0, stream>>>(x, bias, width, eps);
  return hipGetLastError();
}

hipError_t LatticeScores(const float* in, const float* out,
                         const float* anchor, float* scores, int slots,
                         int width, float scale, hipStream_t stream) {
  if (slots < 1)
    return hipErrorInvalidValue;
  LatticeScoresKernel<<<1 + (slots - 1) * kTopK, kTopK * kLanes, 0, stream>>>(
      in, out, anchor, scores, width, scale);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm
