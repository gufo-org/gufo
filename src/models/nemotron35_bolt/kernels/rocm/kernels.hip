#include <hip/hip_fp16.h>

#include <cfloat>
#include <cmath>

#include "src/models/nemotron35_bolt/kernels/rocm/kernels.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/lanes.hip.hpp"
#include "src/models/nemotron35_bolt/kernels/rocm/small_gemv.hip.hpp"

namespace gufo::models::nemotron35_bolt::rocm {
namespace {

constexpr int kBlock = 256;
constexpr int kHeadDim = 128;
constexpr int kStateSize = 128;
constexpr int kLanes = 32;  ///< Lanes of a wave.
constexpr float kMinRoutingWeightSum = 6.103515625e-5F;

__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}
__device__ __forceinline__ float Silu(float x) {
  return x * Sigmoid(x);
}
// ggml's softplus: identity above 20.
__device__ __forceinline__ float Softplus(float x) {
  return x > 20.0F ? x : log1pf(expf(x));
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

__global__ void EmbedQ8_0Kernel(const std::uint8_t* table,
                                const std::int32_t* tokens, float* x,
                                int n_tokens, int hidden) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * hidden)
    return;
  const int t = static_cast<int>(i / hidden);
  const int c = static_cast<int>(i % hidden);
  const std::size_t row_bytes = static_cast<std::size_t>(hidden / 32) * 34;
  const std::uint8_t* block = table +
                              static_cast<std::size_t>(tokens[t]) * row_bytes +
                              static_cast<std::size_t>(c / 32) * 34;
  const __half d = *reinterpret_cast<const __half*>(block);
  const auto q = static_cast<std::int8_t>(block[2 + c % 32]);
  x[i] = __half2float(d) * static_cast<float>(q);
}

/// F32 as computed, or that F32 value rounded to FP16 (nearest even). The
/// empty asm keeps the F32 rounding: fused into v_fma_mixlo_f16, the last
/// product would round once, straight to FP16, and break F32 ties apart.
__device__ __forceinline__ void StoreAs(float* p, float v) { *p = v; }
__device__ __forceinline__ void StoreAs(__half* p, float v) {
  asm volatile("" : "+v"(v));
  *p = __float2half(v);
}

template<class Out>
__global__ void RmsNormKernel(const float* x, const float* w, Out* out,
                              int width, float eps) {
  __shared__ float scratch[kBlock / kLanes];
  const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
  Out* dst = out + static_cast<std::size_t>(blockIdx.x) * width;
  float ss = 0.0F;
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    ss += row[i] * row[i];
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / static_cast<float>(width) + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x)
    StoreAs(dst + i, row[i] * scale * w[i]);
}

__global__ void AddScaledKernel(float* x, const float* a, float scale,
                                const float* b, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = x[i] + scale * a[i];
  if (b != nullptr)
    v += b[i];
  x[i] = v;
}

__global__ void Relu2Kernel(float* x, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const float r = fmaxf(x[i], 0.0F);
  x[i] = r * r;
}

__global__ void MambaConvKernel(const float* proj, int stride, int xbc_offset,
                                float in_scale, const float* weight,
                                const float* bias, const float* state,
                                float* out, int n_tokens, int channels,
                                int kernel) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * channels)
    return;
  const int t = static_cast<int>(i / channels);
  const int c = static_cast<int>(i % channels);
  float acc = 0.0F;
  for (int k = 0; k < kernel; ++k) {
    // Tap 0 multiplies the oldest input; negative times come from the state.
    const int src = t - (kernel - 1) + k;
    const float v =
        src >= 0
            ? proj[static_cast<std::size_t>(src) * stride + xbc_offset + c] *
                  in_scale
            : state[static_cast<std::size_t>(kernel - 1 + src) * channels + c];
    acc += v * weight[static_cast<std::size_t>(c) * kernel + k];
  }
  out[i] = Silu(acc + bias[c]);
}

__global__ void MambaConvStateKernel(const float* proj, int stride,
                                     int xbc_offset, float in_scale,
                                     float* state, int n_tokens, int channels,
                                     int kernel) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= channels)
    return;
  constexpr int kMaxHistory = 8;
  float next[kMaxHistory];
  const int history = kernel - 1;
  for (int r = 0; r < history; ++r) {
    // New row r holds the input at chunk time n_tokens - history + r.
    const int src = n_tokens - history + r;
    next[r] =
        src >= 0
            ? proj[static_cast<std::size_t>(src) * stride + xbc_offset + c] *
                  in_scale
            : state[static_cast<std::size_t>(history + src) * channels + c];
  }
  for (int r = 0; r < history; ++r)
    state[static_cast<std::size_t>(r) * channels + c] = next[r];
}

/// MambaConv then MambaConvState of a one-token chunk, for one token of each
/// of `rows` sequences: row r reads and shifts the history at states[r].
__global__ void MambaConvRowsKernel(const float* proj, int stride,
                                    int xbc_offset, float in_scale,
                                    const float* weight, const float* bias,
                                    float* const* states, float* out, int rows,
                                    int channels, int kernel) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * channels)
    return;
  const int r = static_cast<int>(i / channels);
  const int c = static_cast<int>(i % channels);
  float* state = states[r];
  const float input =
      proj[static_cast<std::size_t>(r) * stride + xbc_offset + c] * in_scale;
  float acc = 0.0F;
  for (int k = 0; k < kernel; ++k) {
    // The last tap multiplies the token; earlier ones the history.
    const float v = k == kernel - 1
                        ? input
                        : state[static_cast<std::size_t>(k) * channels + c];
    acc += v * weight[static_cast<std::size_t>(c) * kernel + k];
  }
  out[i] = Silu(acc + bias[c]);
  for (int h = 0; h + 2 < kernel; ++h)
    state[static_cast<std::size_t>(h) * channels + c] =
        state[static_cast<std::size_t>(h + 1) * channels + c];
  state[static_cast<std::size_t>(kernel - 2) * channels + c] = input;
}

/// Rows of a scan head; the launcher rejects other head widths.
constexpr int kScanRows = 64;
/// Threads of a scan block, one block per head.
constexpr int kScanThreads = 128;
/// Tokens whose B, C, x and dt are staged in LDS at a time. The next tile is
/// loaded into registers while the current one is scanned.
constexpr int kScanTile = 8;
/// Chunks up to this many tokens (decode, verification and its replay) take
/// the layout for short chunks.
constexpr int kScanShortTokens = 8;

/// One block per head. Each row's state is split over kScanSplit lanes, and
/// each thread keeps kScanRowsPerThread rows, which share every B and C value
/// read from LDS. Long chunks are bound by those reads and take four rows
/// per thread; short ones by the state traffic, where two lanes per row keep
/// a wave's loads on two contiguous segments. The state is stored
/// state-major per head, [head][n][row], so consecutive lanes load and store
/// consecutive rows.
template<int kScanRowsPerThread, int kScanSplit, bool kRows = false>
__global__ __launch_bounds__(kScanThreads) void MambaScanKernel(
    MambaScanArgs a) {
  if constexpr (kRows) {
    // Row blockIdx.y is a one-token sequence with its own state.
    const int r = static_cast<int>(blockIdx.y);
    a.state = a.row_states[r];
    a.xbc += static_cast<std::size_t>(r) * a.channels;
    a.proj += static_cast<std::size_t>(r) * a.stride;
    a.y += static_cast<std::size_t>(r) * a.heads * kScanRows;
    a.n_tokens = 1;
  }
  static_assert(kScanRows * kScanSplit == kScanThreads * kScanRowsPerThread);
  constexpr int kPer = kStateSize / kScanSplit;
  constexpr int kQuads = kStateSize / 4;
  constexpr int kRowStep = kScanRows / kScanRowsPerThread;
  constexpr int kRowLanes = kLanes / kScanSplit;
  constexpr int kQuadLoads = kScanTile * kQuads / kScanThreads;
  constexpr int kXLoads = kScanTile * kScanRows / kScanThreads;
  static_assert(kQuadLoads * kScanThreads == kScanTile * kQuads &&
                kXLoads * kScanThreads == kScanTile * kScanRows &&
                kScanTile <= kScanThreads);
  __shared__ float4 s_b[kScanTile][kQuads];
  __shared__ float4 s_c[kScanTile][kQuads];
  __shared__ float s_x[kScanTile][kScanRows];
  __shared__ float s_y[kScanTile][kScanRows];
  __shared__ float s_dt[kScanTile];
  __shared__ float s_da[kScanTile];
  const int h = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kLanes;
  const int part = lane / kRowLanes;
  const int row = tid / kLanes * kRowLanes + lane % kRowLanes;
  const int inner = a.heads * kScanRows;
  const int g = h / (a.heads / a.groups);
  float* st = a.state +
              (static_cast<std::size_t>(h) * kStateSize + part * kPer) *
                  kScanRows +
              row;
  float s[kScanRowsPerThread][kPer];
#pragma unroll
  for (int r = 0; r < kScanRowsPerThread; ++r)
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      s[r][j] = st[j * kScanRows + r * kRowStep];
  const float D = a.d[h];
  const float A = a.a[h];
  const float dt_bias = a.dt_bias[h];

  // Next tile in registers; rows past the chunk repeat its last token.
  float4 nb[kQuadLoads], nc[kQuadLoads];
  float nx[kXLoads];
  float ndt = 0.0F;
  const auto load = [&](int t0) {
    const int last = a.n_tokens - 1;
#pragma unroll
    for (int i = 0; i < kQuadLoads; ++i) {
      const int idx = tid + i * kScanThreads;
      const int t = min(t0 + idx / kQuads, last);
      const auto* bc = reinterpret_cast<const float4*>(
          a.xbc + static_cast<std::size_t>(t) * a.channels + inner);
      nb[i] = bc[g * kQuads + idx % kQuads];
      nc[i] = bc[(a.groups + g) * kQuads + idx % kQuads];
    }
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + i * kScanThreads;
      const int t = min(t0 + idx / kScanRows, last);
      nx[i] = a.xbc[static_cast<std::size_t>(t) * a.channels + h * kScanRows +
                    idx % kScanRows];
    }
    if (tid < kScanTile)
      ndt = a.proj[static_cast<std::size_t>(min(t0 + tid, last)) * a.stride +
                   a.dt_offset + h];
  };
  load(0);
  for (int t0 = 0; t0 < a.n_tokens; t0 += kScanTile) {
    const int count = min(kScanTile, a.n_tokens - t0);
#pragma unroll
    for (int i = 0; i < kQuadLoads; ++i) {
      const int idx = tid + i * kScanThreads;
      s_b[idx / kQuads][idx % kQuads] = nb[i];
      s_c[idx / kQuads][idx % kQuads] = nc[i];
    }
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + i * kScanThreads;
      s_x[idx / kScanRows][idx % kScanRows] = nx[i];
    }
    if (tid < kScanTile) {
      const float dt_sp = Softplus(ndt * a.in_scale + dt_bias);
      s_dt[tid] = dt_sp;
      s_da[tid] = expf(dt_sp * A);
    }
    __syncthreads();
    if (t0 + kScanTile < a.n_tokens)
      load(t0 + kScanTile);
    for (int tt = 0; tt < count; ++tt) {
      const float dA = s_da[tt];
      float x[kScanRowsPerThread];
      float x_dt[kScanRowsPerThread];
      float p[kScanRowsPerThread][4];
#pragma unroll
      for (int r = 0; r < kScanRowsPerThread; ++r) {
        x[r] = s_x[tt][row + r * kRowStep];
        x_dt[r] = x[r] * s_dt[tt];
        p[r][0] = p[r][1] = p[r][2] = p[r][3] = 0.0F;
      }
#pragma unroll
      for (int q = 0; q < kPer / 4; ++q) {
        const float4 b4 = s_b[tt][part * (kPer / 4) + q];
        const float4 c4 = s_c[tt][part * (kPer / 4) + q];
        const float b[4] = {b4.x, b4.y, b4.z, b4.w};
        const float c[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
        for (int r = 0; r < kScanRowsPerThread; ++r)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float& v = s[r][4 * q + e];
            v = v * dA + b[e] * x_dt[r];
            p[r][e] += v * c[e];
          }
      }
      float sum[kScanRowsPerThread];
#pragma unroll
      for (int r = 0; r < kScanRowsPerThread; ++r)
        sum[r] = (p[r][0] + p[r][1]) + (p[r][2] + p[r][3]);
      if constexpr (kRowLanes <= 4) {
#pragma unroll
        for (int r = 0; r < kScanRowsPerThread; ++r)
          sum[r] = AddXorLane<4>(sum[r]);
      }
      if constexpr (kRowLanes <= 8) {
#pragma unroll
        for (int r = 0; r < kScanRowsPerThread; ++r)
          sum[r] = AddXorLane<8>(sum[r]);
      }
#pragma unroll
      for (int r = 0; r < kScanRowsPerThread; ++r)
        sum[r] = AddXorLane<16>(sum[r]);
      if (part == 0) {
#pragma unroll
        for (int r = 0; r < kScanRowsPerThread; ++r)
          s_y[tt][row + r * kRowStep] = sum[r] + x[r] * D;
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kXLoads; ++i) {
      const int idx = tid + i * kScanThreads;
      if (idx / kScanRows < count)
        a.y[static_cast<std::size_t>(t0 + idx / kScanRows) * inner +
            h * kScanRows + idx % kScanRows] =
            s_y[idx / kScanRows][idx % kScanRows];
    }
  }
  if (a.write_state) {
#pragma unroll
    for (int r = 0; r < kScanRowsPerThread; ++r)
#pragma unroll
      for (int j = 0; j < kPer; ++j)
        st[j * kScanRows + r * kRowStep] = s[r][j];
  }
}

template<class Out>
__global__ void MambaGateNormKernel(float* y, const float* proj, int stride,
                                    float in_scale, const float* weight,
                                    Out* out, int inner, int group_size,
                                    float eps) {
  __shared__ float scratch[kBlock / kLanes];
  const int t = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  float* row = y + static_cast<std::size_t>(t) * inner + g * group_size;
  Out* dst = out + static_cast<std::size_t>(t) * inner + g * group_size;
  const float* z = proj + static_cast<std::size_t>(t) * stride + g * group_size;
  const float* w = weight + g * group_size;
  float ss = 0.0F;
  for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
    const float v = row[j] * Silu(z[j] * in_scale);
    row[j] = v;
    ss += v * v;
  }
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / static_cast<float>(group_size) + eps);
  for (int j = threadIdx.x; j < group_size; j += blockDim.x)
    StoreAs(dst + j, row[j] * scale * w[j]);
}

__global__ void KvAppendKernel(const float* k, const float* v, __half* k_cache,
                               __half* v_cache, const int* pos, int n_tokens,
                               int kv_dim) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * kv_dim)
    return;
  const std::size_t dst = static_cast<std::size_t>(*pos) * kv_dim + i;
  k_cache[dst] = __float2half(k[i]);
  v_cache[dst] = __float2half(v[i]);
}

__global__ void KvAppendRowsKernel(const float* k, const float* v,
                                   __half* const* k_caches,
                                   __half* const* v_caches, const int* pos,
                                   int rows, int kv_dim) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(rows) * kv_dim)
    return;
  const int r = static_cast<int>(i / kv_dim);
  const std::size_t dst =
      static_cast<std::size_t>(pos[r]) * kv_dim + i % kv_dim;
  k_caches[r][dst] = __float2half(k[i]);
  v_caches[r][dst] = __float2half(v[i]);
}

using V16h = _Float16 __attribute__((ext_vector_type(16)));
using V8f = float __attribute__((ext_vector_type(8)));

/// Query heads per KV head that the WMMA attention requires: one 16-row tile.
constexpr int kGroupHeads = 16;
/// Keys per staged tile of the WMMA attention.
constexpr int kAttnKeys = 64;
/// Query tokens per block; each wave takes one.
constexpr int kAttnTokens = 8;
/// Halves per staged row: payload plus bank padding.
constexpr int kKeyStride = kHeadDim + 8;
constexpr int kVtStride = kAttnKeys + 8;

__device__ __forceinline__ V16h LoadHalves16(const _Float16* p) {
  union {
    V16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// Max (kMax) or sum over the 16 lanes holding one row of a WMMA C tile.
template<bool kMax>
__device__ __forceinline__ float RowReduce16(float v) {
  if constexpr (kMax) {
    v = fmaxf(v, XorLane<1>(v));
    v = fmaxf(v, XorLane<2>(v));
    v = fmaxf(v, XorLane<4>(v));
    return fmaxf(v, XorLane<8>(v));
  } else {
    v = AddXorLane<1>(v);
    v = AddXorLane<2>(v);
    v = AddXorLane<4>(v);
    return AddXorLane<8>(v);
  }
}

/// Causal GQA flash attention on FP16 WMMA. A block serves kAttnTokens
/// consecutive query tokens of one KV head; wave w takes token t0 + w, whose
/// 16 query heads form one WMMA row tile, so every row of a wave shares the
/// causal limit. K and V tiles of kAttnKeys keys are staged once per block
/// (V transposed); S = Q K^T and O += P V run on the matrix units with F32
/// accumulation, the online softmax in F32 in the exp2 domain. Q (scaled)
/// and P round to FP16 as in llama.cpp's flash attention. In a C tile,
/// element l of a lane is row 2l + lane / 16 and column lane % 16.
///
/// With kSplit, block z of the grid's third dimension takes key tiles z,
/// z + splits, ... and writes, per query row, the unnormalized output, the
/// maximum and the normalizer to `out` as `partial`
/// ([token][head][split][head_dim + 2]); AttentionMergeKernel combines them.
/// A tile's split does not depend on the number of keys, so each row of a
/// verified block sums its keys as a one-token pass at its position does.
///
/// With kRows, block x of the grid is a one-token sequence: its query is
/// row x of q, its cache k_rows[x] / v_rows[x] and its position pos[x].
template<bool kSplit, bool kRows = false>
__global__ __launch_bounds__(kAttnTokens * 32) void AttentionKernel(
    const float* q, const __half* k_cache, const __half* v_cache, float* out,
    const int* pos, int n_tokens, int heads, int kv_heads, float scale,
    const __half* const* k_rows = nullptr,
    const __half* const* v_rows = nullptr) {
  const int seq = kRows ? static_cast<int>(blockIdx.x) : 0;
  if constexpr (kRows) {
    k_cache = k_rows[seq];
    v_cache = v_rows[seq];
    q += static_cast<std::size_t>(seq) * heads * kHeadDim;
    pos += seq;
    n_tokens = 1;
  }
  __shared__ _Float16 s_k[kAttnKeys][kKeyStride];
  __shared__ _Float16 s_vt[kHeadDim][kVtStride];
  __shared__ _Float16 s_p[kAttnTokens][16][kVtStride];
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / 32;
  const int lane = tid % 32;
  const int col = lane % 16;
  const int hi = lane / 16;
  const int g = static_cast<int>(blockIdx.y);
  const int t0 = kRows ? 0 : static_cast<int>(blockIdx.x) * kAttnTokens;
  const int t = t0 + wave;
  // Output row: the token, or the sequence of a row grid (whose token is 0).
  const int out_t = seq + t;
  const int kv_dim = kv_heads * kHeadDim;
  const int base = *pos;
  // Tokens past the chunk compute on the last one and are not stored.
  const int tq = min(t, n_tokens - 1);
  const int query_pos = base + tq;
  const int n_keys = base + min(t0 + kAttnTokens, n_tokens);
  const int key_end = n_keys;
  int key_begin = 0;
  int key_step = kAttnKeys;
  if constexpr (kSplit) {
    key_begin = static_cast<int>(blockIdx.z) * kAttnKeys;
    key_step = static_cast<int>(gridDim.z) * kAttnKeys;
  }
  // Waves without a token only help stage the tiles.
  const bool active = t < n_tokens;

  // Q fragments: lane row col is head g * 16 + col, eight 16-dim steps.
  constexpr float kLog2e = 1.4426950408889634F;
  const float qs = scale * kLog2e;
  V16h qf[kHeadDim / 16];
  {
    const float* qrow =
        q + (static_cast<std::size_t>(tq) * heads + g * kGroupHeads + col) *
                kHeadDim;
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ++ks) {
#pragma unroll
      for (int e = 0; e < 16; ++e)
        qf[ks][e] = static_cast<_Float16>(qrow[16 * ks + e] * qs);
    }
  }
  V8f o[kHeadDim / 16] = {};
  float m[8];
  float l[8];  // Per-lane partial row sums, reduced at the end.
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.0F;
  }

  for (int k0 = key_begin; k0 < key_end; k0 += key_step) {
    __syncthreads();
    {
      // Each thread stages a quarter key row of K and V (32 dims).
      const int key = tid / 4;
      const int d0 = (tid % 4) * 32;
      const bool live = k0 + key < key_end;
      const std::size_t src =
          static_cast<std::size_t>(k0 + key) * kv_dim + g * kHeadDim + d0;
      uint4 kk[4];
      uint4 vv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kk[i] = live ? reinterpret_cast<const uint4*>(k_cache + src)[i]
                     : make_uint4(0U, 0U, 0U, 0U);
        vv[i] = live ? reinterpret_cast<const uint4*>(v_cache + src)[i]
                     : make_uint4(0U, 0U, 0U, 0U);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
        reinterpret_cast<uint4*>(&s_k[key][d0])[i] = kk[i];
      const auto* vh = reinterpret_cast<const _Float16*>(vv);
#pragma unroll
      for (int e = 0; e < 32; ++e)
        s_vt[d0 + e][key] = vh[e];
    }
    __syncthreads();
    if (!active)
      continue;

    V8f s[kAttnKeys / 16];
#pragma unroll
    for (int j = 0; j < kAttnKeys / 16; ++j) {
      s[j] = V8f{};
#pragma unroll
      for (int ks = 0; ks < kHeadDim / 16; ++ks)
        s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            qf[ks], LoadHalves16(&s_k[16 * j + col][16 * ks]), s[j]);
    }
    // Columns past the query's position, or past the cache, are masked.
#pragma unroll
    for (int j = 0; j < kAttnKeys / 16; ++j) {
      if (k0 + 16 * j + col > query_pos) {
#pragma unroll
        for (int r = 0; r < 8; ++r)
          s[j][r] = -INFINITY;
      }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      float tile_max = s[0][r];
#pragma unroll
      for (int j = 1; j < kAttnKeys / 16; ++j)
        tile_max = fmaxf(tile_max, s[j][r]);
      tile_max = RowReduce16<true>(tile_max);
      // A row that has seen no visible key yet keeps its empty state.
      const float new_m = fmaxf(m[r], tile_max);
      const bool empty = new_m == -INFINITY;
      const float corr = empty ? 1.0F : exp2f(m[r] - new_m);
      m[r] = new_m;
      float sum = 0.0F;
#pragma unroll
      for (int j = 0; j < kAttnKeys / 16; ++j) {
        const float p = empty ? 0.0F : exp2f(s[j][r] - new_m);
        sum += p;
        s_p[wave][2 * r + hi][16 * j + col] = static_cast<_Float16>(p);
      }
      l[r] = l[r] * corr + sum;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 16; ++dt)
        o[dt][r] *= corr;
    }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int kk = 0; kk < kAttnKeys / 16; ++kk) {
      const V16h pf = LoadHalves16(&s_p[wave][col][16 * kk]);
      // A short chunk stages its later tokens' values, which a one-token
      // pass leaves at zero. Their probabilities are zero, but the WMMA
      // sum still rounds differently beside a nonzero value, so they are
      // zeroed here: a verified row equals that token decoded alone.
      const int masked_from = kSplit ? query_pos + 1 - (k0 + 16 * kk) : 16;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 16; ++dt) {
        V16h vf = LoadHalves16(&s_vt[16 * dt + col][16 * kk]);
        if (masked_from < 16) {
#pragma unroll
          for (int e = 0; e < 16; ++e) {
            if (e >= masked_from)
              vf[e] = static_cast<_Float16>(0.0F);
          }
        }
        o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pf, vf, o[dt]);
      }
    }
  }
  if (!active)
    return;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const float sum = RowReduce16<false>(l[r]);
    if constexpr (kSplit) {
      float* dst =
          out + ((static_cast<std::size_t>(out_t) * heads + g * kGroupHeads +
                  2 * r + hi) *
                     gridDim.z +
                 blockIdx.z) *
                    (kHeadDim + 2);
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 16; ++dt)
        dst[16 * dt + col] = o[dt][r];
      if (col == 0) {
        dst[kHeadDim] = m[r];
        dst[kHeadDim + 1] = sum;
      }
      continue;
    }
    const float inv = 1.0F / sum;
    float* dst = out +
                 (static_cast<std::size_t>(out_t) * heads + g * kGroupHeads +
                  2 * r + hi) *
                     kHeadDim +
                 col;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 16; ++dt)
      dst[16 * dt] = o[dt][r] * inv;
  }
}

/// Merges the splits of one (token, head) row: thread d owns dimension d.
/// The partial maxima are in the exp2 domain.
__global__ void AttentionMergeKernel(const float* partial, float* out,
                                     int splits) {
  const std::size_t row = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const float* src = partial + row * splits * (kHeadDim + 2);
  float m = -INFINITY;
  for (int s = 0; s < splits; ++s)
    m = fmaxf(m, src[s * (kHeadDim + 2) + kHeadDim]);
  float l = 0.0F;
  float acc = 0.0F;
  for (int s = 0; s < splits; ++s) {
    const float* part = src + s * (kHeadDim + 2);
    const float ms = part[kHeadDim];
    if (ms == -INFINITY)
      continue;
    const float w = exp2f(ms - m);
    l += part[kHeadDim + 1] * w;
    acc += part[tid] * w;
  }
  out[row * kHeadDim + tid] = acc / l;
}

__global__ void RouterTopKKernel(const float* logits, const float* bias,
                                 int n_experts, int k, bool normalize,
                                 float scale, std::int32_t* ids,
                                 float* weights) {
  constexpr int kPerThread = 2;  // Up to 512 experts with 256 threads.
  __shared__ float best_v[kBlock / kLanes];
  __shared__ int best_i[kBlock / kLanes];
  __shared__ float chosen[32];
  const int t = static_cast<int>(blockIdx.x);
  const float* row = logits + static_cast<std::size_t>(t) * n_experts;
  float prob[kPerThread];
  float sel[kPerThread];
  for (int r = 0; r < kPerThread; ++r) {
    const int e = static_cast<int>(threadIdx.x) + r * kBlock;
    prob[r] = e < n_experts ? Sigmoid(row[e]) : 0.0F;
    sel[r] = e < n_experts ? prob[r] + bias[e] : -INFINITY;
  }
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int group = static_cast<int>(threadIdx.x) / kLanes;
  for (int round = 0; round < k; ++round) {
    // Arg-max with ties on the lower expert index.
    float v = -INFINITY;
    int idx = 0x7FFFFFFF;
    for (int r = 0; r < kPerThread; ++r) {
      const int e = static_cast<int>(threadIdx.x) + r * kBlock;
      if (sel[r] > v || (sel[r] == v && e < idx)) {
        v = sel[r];
        idx = e;
      }
    }
    for (int offset = kLanes / 2; offset > 0; offset /= 2) {
      const float ov = __shfl_xor(v, offset, kLanes);
      const int oi = __shfl_xor(idx, offset, kLanes);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      best_v[group] = v;
      best_i[group] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = best_v[0];
      int bi = best_i[0];
      for (int i = 1; i < kBlock / kLanes; ++i) {
        if (best_v[i] > bv || (best_v[i] == bv && best_i[i] < bi)) {
          bv = best_v[i];
          bi = best_i[i];
        }
      }
      ids[static_cast<std::size_t>(t) * k + round] = bi;
      best_i[0] = bi;
    }
    __syncthreads();
    const int winner = best_i[0];
    for (int r = 0; r < kPerThread; ++r) {
      if (static_cast<int>(threadIdx.x) + r * kBlock == winner) {
        chosen[round] = prob[r];
        sel[r] = -INFINITY;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (int r = 0; r < k; ++r)
      sum += chosen[r];
    const float denom = normalize ? fmaxf(sum, kMinRoutingWeightSum) : 1.0F;
    for (int r = 0; r < k; ++r)
      weights[static_cast<std::size_t>(t) * k + r] = chosen[r] / denom * scale;
  }
}

__global__ void WeightedSlotSumKernel(const float* slots, const float* weights,
                                      float* out, int n_tokens, int k,
                                      int width) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(n_tokens) * width)
    return;
  const int t = static_cast<int>(i / width);
  const int d = static_cast<int>(i % width);
  float acc = 0.0F;
  for (int s = 0; s < k; ++s) {
    const std::size_t r = static_cast<std::size_t>(t) * k + s;
    acc += weights[r] * slots[r * width + d];
  }
  out[i] = acc;
}

/// Quantizes four consecutive values per thread into Q8_1 as the vendored
/// quantize_q8_1 does (d = amax / 127, q = round(x / d)). Eight consecutive
/// lanes, aligned to eight, hold one 32-value block: quad `quad` of a row.
__device__ void StoreQ8_1(float4 v, std::uint8_t* row, int quad) {
  float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  float sum = (v.x + v.y) + (v.z + v.w);
  amax = fmaxf(amax, XorLane<1>(amax));
  amax = fmaxf(amax, XorLane<2>(amax));
  amax = fmaxf(amax, XorLane<4>(amax));
  sum = AddXorLane<4>(AddXorLane<2>(AddXorLane<1>(sum)));
  const float d = amax / 127.0F;
  const auto q = [&](float x) -> std::uint32_t {
    return amax == 0.0F ? 0U
                        : static_cast<std::uint8_t>(
                              static_cast<std::int8_t>(roundf(x / d)));
  };
  std::uint8_t* block = row + (quad / 8) * kQ8_1BlockBytes;
  *reinterpret_cast<std::uint32_t*>(block + 4 + 4 * (quad % 8)) =
      q(v.x) | (q(v.y) << 8) | (q(v.z) << 16) | (q(v.w) << 24);
  if (quad % 8 == 0)
    *reinterpret_cast<__half2*>(block) = __halves2half2(
        __float2half(d), __float2half(sum));
}

/// Sum over a block of 32-lane waves, added in wave order; every thread
/// receives it.
__device__ float BlockSumWaves(float v, float* scratch) {
  v = WaveSum(v);
  const int lane = threadIdx.x % kLanes;
  const int wave = threadIdx.x / kLanes;
  if (lane == 0)
    scratch[wave] = v;
  __syncthreads();
  float total = 0.0F;
  for (int i = 0; i < static_cast<int>(blockDim.x) / kLanes; ++i)
    total += scratch[i];
  return total;
}

/// One block per row, one float4 per thread.
__global__ void RmsNormQ8Kernel(const float* x, const float* w, float* out,
                                std::uint8_t* q8, int width, float eps) {
  __shared__ float scratch[1024 / kLanes];
  const int t = static_cast<int>(threadIdx.x);
  const std::size_t row = blockIdx.x;
  const float4 v = reinterpret_cast<const float4*>(x + row * width)[t];
  const float ss =
      BlockSumWaves(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, scratch);
  const float scale = rsqrtf(ss / static_cast<float>(width) + eps);
  const float4 g = reinterpret_cast<const float4*>(w)[t];
  const float4 r = make_float4(v.x * scale * g.x, v.y * scale * g.y,
                               v.z * scale * g.z, v.w * scale * g.w);
  if (out != nullptr)
    reinterpret_cast<float4*>(out + row * width)[t] = r;
  if (q8 != nullptr)
    StoreQ8_1(r, q8 + row * (width / 32) * kQ8_1BlockBytes, t);
}

__global__ void QuantizeQ8_1Kernel(const float* x, std::uint8_t* q8,
                                   int width) {
  const int t = static_cast<int>(threadIdx.x);
  const std::size_t row = blockIdx.x;
  StoreQ8_1(reinterpret_cast<const float4*>(x + row * width)[t],
            q8 + row * (width / 32) * kQ8_1BlockBytes, t);
}

/// One block of 256 threads per (token, group of 1024).
__global__ void MambaGateNormQ8Kernel(const float* y, const float* proj,
                                      int stride, float in_scale,
                                      const float* weight, std::uint8_t* q8,
                                      int inner, float eps) {
  __shared__ float scratch[kBlock / kLanes];
  constexpr int kGroup = 4 * kBlock;
  const int t = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  const int j = static_cast<int>(threadIdx.x);
  const float4 a = reinterpret_cast<const float4*>(
      y + static_cast<std::size_t>(t) * inner + g * kGroup)[j];
  const float4 z = reinterpret_cast<const float4*>(
      proj + static_cast<std::size_t>(t) * stride + g * kGroup)[j];
  const float4 v = make_float4(
      a.x * Silu(z.x * in_scale), a.y * Silu(z.y * in_scale),
      a.z * Silu(z.z * in_scale), a.w * Silu(z.w * in_scale));
  const float ss =
      BlockSumWaves(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w, scratch);
  const float scale = rsqrtf(ss / static_cast<float>(kGroup) + eps);
  const float4 w = reinterpret_cast<const float4*>(weight + g * kGroup)[j];
  const float4 r = make_float4(v.x * scale * w.x, v.y * scale * w.y,
                               v.z * scale * w.z, v.w * scale * w.w);
  StoreQ8_1(r, q8 + static_cast<std::size_t>(t) * (inner / 32) * kQ8_1BlockBytes,
            g * kBlock + j);
}

/// One block per token, one float4 per thread.
__global__ void WeightedSlotSumQ8Kernel(const float* slots,
                                        const float* weights, std::uint8_t* q8,
                                        int k, int width) {
  const int t = static_cast<int>(blockIdx.x);
  const int j = static_cast<int>(threadIdx.x);
  float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  for (int s = 0; s < k; ++s) {
    const std::size_t r = static_cast<std::size_t>(t) * k + s;
    const float wr = weights[r];
    const float4 v = reinterpret_cast<const float4*>(slots + r * width)[j];
    acc.x += wr * v.x;
    acc.y += wr * v.y;
    acc.z += wr * v.z;
    acc.w += wr * v.w;
  }
  StoreQ8_1(acc, q8 + static_cast<std::size_t>(t) * (width / 32) * kQ8_1BlockBytes,
            j);
}

}  // namespace

hipError_t EmbedQ8_0(const void* table, const std::int32_t* tokens, float* x,
                     int n_tokens, int hidden, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * hidden;
  EmbedQ8_0Kernel<<<Blocks(n), kBlock, 0, stream>>>(
      static_cast<const std::uint8_t*>(table), tokens, x, n_tokens, hidden);
  return hipGetLastError();
}

hipError_t RmsNorm(const float* x, const float* w, float* out, int rows,
                   int width, float eps, hipStream_t stream) {
  RmsNormKernel<<<rows, kBlock, 0, stream>>>(x, w, out, width, eps);
  return hipGetLastError();
}

hipError_t RmsNorm(const float* x, const float* w, __half* out, int rows,
                   int width, float eps, hipStream_t stream) {
  RmsNormKernel<<<rows, kBlock, 0, stream>>>(x, w, out, width, eps);
  return hipGetLastError();
}

hipError_t AddScaled(float* x, const float* a, float scale, const float* b,
                     std::size_t n, hipStream_t stream) {
  AddScaledKernel<<<Blocks(n), kBlock, 0, stream>>>(x, a, scale, b, n);
  return hipGetLastError();
}

hipError_t Relu2(float* x, std::size_t n, hipStream_t stream) {
  Relu2Kernel<<<Blocks(n), kBlock, 0, stream>>>(x, n);
  return hipGetLastError();
}

hipError_t MambaConv(const float* proj, int stride, int xbc_offset,
                     float in_scale, const float* weight, const float* bias,
                     const float* state, float* out, int n_tokens, int channels,
                     int kernel, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * channels;
  MambaConvKernel<<<Blocks(n), kBlock, 0, stream>>>(
      proj, stride, xbc_offset, in_scale, weight, bias, state, out, n_tokens,
      channels, kernel);
  return hipGetLastError();
}

hipError_t MambaConvState(const float* proj, int stride, int xbc_offset,
                          float in_scale, float* state, int n_tokens,
                          int channels, int kernel, hipStream_t stream) {
  if (kernel < 2 || kernel > 9)
    return hipErrorInvalidValue;
  MambaConvStateKernel<<<Blocks(channels), kBlock, 0, stream>>>(
      proj, stride, xbc_offset, in_scale, state, n_tokens, channels, kernel);
  return hipGetLastError();
}

hipError_t MambaScan(const MambaScanArgs& args, hipStream_t stream) {
  if (args.heads % args.groups != 0 || args.head_dim != kScanRows ||
      args.channels % 4 != 0)
    return hipErrorInvalidValue;
  if (args.row_states != nullptr) {
    if (args.n_tokens < 1 || !args.write_state)
      return hipErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>(args.heads),
                    static_cast<unsigned>(args.n_tokens));
    MambaScanKernel<1, 2, true><<<grid, kScanThreads, 0, stream>>>(args);
    return hipGetLastError();
  }
  if (args.n_tokens <= kScanShortTokens)
    MambaScanKernel<1, 2><<<args.heads, kScanThreads, 0, stream>>>(args);
  else
    MambaScanKernel<4, 8><<<args.heads, kScanThreads, 0, stream>>>(args);
  return hipGetLastError();
}

hipError_t MambaGateNorm(float* y, const float* proj, int stride,
                         float in_scale, const float* weight, int n_tokens,
                         int inner, int groups, float eps, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(n_tokens),
                  static_cast<unsigned>(groups));
  MambaGateNormKernel<<<grid, kBlock, 0, stream>>>(
      y, proj, stride, in_scale, weight, y, inner, inner / groups, eps);
  return hipGetLastError();
}

hipError_t MambaGateNorm(float* y, const float* proj, int stride,
                         float in_scale, const float* weight, __half* out,
                         int n_tokens, int inner, int groups, float eps,
                         hipStream_t stream) {
  const dim3 grid(static_cast<unsigned>(n_tokens),
                  static_cast<unsigned>(groups));
  MambaGateNormKernel<<<grid, kBlock, 0, stream>>>(
      y, proj, stride, in_scale, weight, out, inner, inner / groups, eps);
  return hipGetLastError();
}

hipError_t MambaConvRows(const float* proj, int stride, int xbc_offset,
                         float in_scale, const float* weight,
                         const float* bias, float* const* states, float* out,
                         int rows, int channels, int kernel,
                         hipStream_t stream) {
  if (kernel < 2 || kernel > 9)
    return hipErrorInvalidValue;
  const std::size_t n = static_cast<std::size_t>(rows) * channels;
  MambaConvRowsKernel<<<Blocks(n), kBlock, 0, stream>>>(
      proj, stride, xbc_offset, in_scale, weight, bias, states, out, rows,
      channels, kernel);
  return hipGetLastError();
}

hipError_t KvAppendRows(const float* k, const float* v,
                        __half* const* k_caches, __half* const* v_caches,
                        const int* pos, int rows, int kv_dim,
                        hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(rows) * kv_dim;
  KvAppendRowsKernel<<<Blocks(n), kBlock, 0, stream>>>(
      k, v, k_caches, v_caches, pos, rows, kv_dim);
  return hipGetLastError();
}

hipError_t KvAppend(const float* k, const float* v, __half* k_cache,
                    __half* v_cache, const int* pos, int n_tokens, int kv_dim,
                    hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * kv_dim;
  KvAppendKernel<<<Blocks(n), kBlock, 0, stream>>>(k, v, k_cache, v_cache, pos,
                                                   n_tokens, kv_dim);
  return hipGetLastError();
}

hipError_t Attention(const float* q, const __half* k_cache,
                     const __half* v_cache, float* out, const int* pos,
                     int n_tokens, int heads, int kv_heads, float scale,
                     hipStream_t stream) {
  if (heads != kv_heads * kGroupHeads)
    return hipErrorInvalidValue;
  const dim3 grid(
      static_cast<unsigned>((n_tokens + kAttnTokens - 1) / kAttnTokens),
      static_cast<unsigned>(kv_heads));
  AttentionKernel<false><<<grid, kAttnTokens * 32, 0, stream>>>(
      q, k_cache, v_cache, out, pos, n_tokens, heads, kv_heads, scale);
  return hipGetLastError();
}

hipError_t AttentionShort(const float* q, const __half* k_cache,
                          const __half* v_cache, float* out, const int* pos,
                          int n_tokens, int heads, int kv_heads, float scale,
                          float* partial, hipStream_t stream) {
  if (heads != kv_heads * kGroupHeads || n_tokens < 1 ||
      n_tokens > kAttentionShortTokens)
    return hipErrorInvalidValue;
  const dim3 grid(1, static_cast<unsigned>(kv_heads), kAttentionSplits);
  AttentionKernel<true><<<grid, kAttnTokens * 32, 0, stream>>>(
      q, k_cache, v_cache, partial, pos, n_tokens, heads, kv_heads, scale);
  AttentionMergeKernel<<<n_tokens * heads, kHeadDim, 0, stream>>>(
      partial, out, kAttentionSplits);
  return hipGetLastError();
}

hipError_t AttentionShortRows(const float* q, const __half* const* k_caches,
                              const __half* const* v_caches, float* out,
                              const int* pos, int rows, int heads,
                              int kv_heads, float scale, float* partial,
                              hipStream_t stream) {
  if (heads != kv_heads * kGroupHeads || rows < 1 ||
      rows > kAttentionShortTokens)
    return hipErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads),
                  kAttentionSplits);
  AttentionKernel<true, true><<<grid, kAttnTokens * 32, 0, stream>>>(
      q, nullptr, nullptr, partial, pos, rows, heads, kv_heads, scale,
      k_caches, v_caches);
  AttentionMergeKernel<<<rows * heads, kHeadDim, 0, stream>>>(
      partial, out, kAttentionSplits);
  return hipGetLastError();
}

hipError_t RouterLogits(const float* w, const float* x, float* out, int rows,
                        int cols, int n_tokens, hipStream_t stream) {
  if (n_tokens > kRouterMaxTokens)
    return hipErrorInvalidValue;
  return small_gemv::Launch<small_gemv::F32Weights>(w, x, out, rows, cols,
                                                    n_tokens, stream);
}

hipError_t RouterTopK(const float* logits, const float* bias, int n_experts,
                      int n_tokens, int k, bool normalize, float scale,
                      std::int32_t* ids, float* weights, hipStream_t stream) {
  if (n_experts > 2 * kBlock || k > 32 || k > n_experts)
    return hipErrorInvalidValue;
  RouterTopKKernel<<<n_tokens, kBlock, 0, stream>>>(
      logits, bias, n_experts, k, normalize, scale, ids, weights);
  return hipGetLastError();
}

hipError_t WeightedSlotSum(const float* slots, const float* weights, float* out,
                           int n_tokens, int k, int width, hipStream_t stream) {
  const std::size_t n = static_cast<std::size_t>(n_tokens) * width;
  WeightedSlotSumKernel<<<Blocks(n), kBlock, 0, stream>>>(slots, weights, out,
                                                          n_tokens, k, width);
  return hipGetLastError();
}

namespace {

bool ValidQ8Width(int width) {
  return width > 0 && width % 128 == 0 && width <= 4096;
}

}  // namespace

hipError_t RmsNormQ8(const float* x, const float* w, float* out, void* q8,
                     int rows, int width, float eps, hipStream_t stream) {
  if (!ValidQ8Width(width) || rows < 1)
    return hipErrorInvalidValue;
  RmsNormQ8Kernel<<<rows, width / 4, 0, stream>>>(
      x, w, out, static_cast<std::uint8_t*>(q8), width, eps);
  return hipGetLastError();
}

hipError_t QuantizeQ8_1(const float* x, void* q8, int rows, int width,
                        hipStream_t stream) {
  if (!ValidQ8Width(width) || rows < 1)
    return hipErrorInvalidValue;
  QuantizeQ8_1Kernel<<<rows, width / 4, 0, stream>>>(
      x, static_cast<std::uint8_t*>(q8), width);
  return hipGetLastError();
}

hipError_t MambaGateNormQ8(const float* y, const float* proj, int stride,
                           float in_scale, const float* weight, void* q8,
                           int n_tokens, int inner, int groups, float eps,
                           hipStream_t stream) {
  if (groups < 1 || inner != groups * 4 * kBlock || stride % 4 != 0)
    return hipErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(n_tokens),
                  static_cast<unsigned>(groups));
  MambaGateNormQ8Kernel<<<grid, kBlock, 0, stream>>>(
      y, proj, stride, in_scale, weight, static_cast<std::uint8_t*>(q8), inner,
      eps);
  return hipGetLastError();
}

hipError_t WeightedSlotSumQ8(const float* slots, const float* weights, void* q8,
                             int n_tokens, int k, int width,
                             hipStream_t stream) {
  if (!ValidQ8Width(width) || n_tokens < 1)
    return hipErrorInvalidValue;
  WeightedSlotSumQ8Kernel<<<n_tokens, width / 4, 0, stream>>>(
      slots, weights, static_cast<std::uint8_t*>(q8), k, width);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm
