#pragma once

#include <hip/hip_runtime.h>

inline constexpr uint32_t DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP = 1024u;

// DSpark support attention. The launcher guarantees r < n_raw <= raw_cap and
// raw_start < raw_cap, so ring indexing needs at most one subtraction. Keep the
// floating-point accumulation order unchanged; integer division in the value
// loop was particularly expensive at long contexts.
__global__ static void attention_noncausal_raw_batch_heads_kernel(
    float* heads, const float* sinks, const float* q, const float* raw_kv,
    uint32_t n_tokens, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
    uint32_t n_head, uint32_t head_dim) {
  const uint32_t tok = blockIdx.x;
  const uint32_t h = blockIdx.y;
  if (tok >= n_tokens || h >= n_head)
    return;

  extern __shared__ float sh_scores[];
  const float* qh = q + ((uint64_t)tok * n_head + h) * head_dim;
  const float scale = rsqrtf((float)head_dim);
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    const uint32_t row =
        (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
    const float* kv = raw_kv + (uint64_t)row * head_dim;
    float dot = 0.0f;
    for (uint32_t d = 0; d < head_dim; d++)
      dot += qh[d] * kv[d];
    sh_scores[r] = dot * scale;
  }
  __syncthreads();

  __shared__ float partial[256];
  __shared__ float max_s;
  __shared__ float denom;
  float local_max = sinks[h];
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    local_max = fmaxf(local_max, sh_scores[r]);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    max_s = partial[0];
  __syncthreads();

  float den_local = 0.0f;
  for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
    sh_scores[r] = expf(sh_scores[r] - max_s);
    den_local += sh_scores[r];
  }
  partial[threadIdx.x] = den_local;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    denom = partial[0] + expf(sinks[h] - max_s);
  __syncthreads();

  float* oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
#pragma clang fp reassociate(off)
    // Preserve the qualified row-by-row FMA order under newer LLVM.
    float acc = 0.0f;
    for (uint32_t r = 0; r < n_raw; r++) {
      const uint32_t row =
          (raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap);
      acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
    }
    oh[d] = acc / denom;
  }
}

// Warp reductions over N independent values, one shuffle step at a time
// across the values so their exchanges overlap. Each value follows the
// original 32-lane shuffle-down tree.
template<uint32_t N>
__device__ __forceinline__ static void attention_warp_sum_n(float (&v)[N]) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float t[N];
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      t[i] = __shfl_down(v[i], offset, 32);
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      v[i] += t[i];
  }
}

template<uint32_t N>
__device__ __forceinline__ static void attention_warp_max_n(float (&v)[N]) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float t[N];
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      t[i] = __shfl_down(v[i], offset, 32);
#pragma unroll
    for (uint32_t i = 0; i < N; ++i)
      v[i] = fmaxf(v[i], t[i]);
  }
}

inline constexpr uint32_t DS4_DECODE_ATTENTION_DIM = 512u;

// Single-token decode attention in two launches. The query heads share one KV
// head: the score kernel splits rows across workgroups and loads each row
// once for a group of heads, and the softmax/value kernel splits heads and
// output dimensions. Each head keeps the qualified scalar arithmetic: one
// thread forms each dot product from four FMA chains summed
// (s0 + s2) + (s1 + s3), thread t of 256 sums the weights of rows t,
// t + 256, ... before the shuffle trees, and one thread accumulates each
// output element row by row. With topk, compressed rows are the first CAP
// visible top-k entries in order; otherwise every compressed row is used,
// optionally with comp_mask.

// Writes the selected compressed rows in top-k order and returns their count.
template<uint32_t kThreads>
__device__ static uint32_t attention_decode_select_rows(uint32_t* comp_rows,
                                                        uint32_t* wave_counts,
                                                        const int32_t* topk,
                                                        uint32_t top_k,
                                                        uint32_t visible_comp) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wave = tid >> 5u;
  uint32_t n_sel = 0;
  for (uint32_t base = 0; base < top_k; base += kThreads) {
    const uint32_t i = base + tid;
    const int32_t ci = i < top_k ? topk[i] : -1;
    const bool keep = ci >= 0 && (uint32_t)ci < visible_comp;
    const uint64_t ballot = __ballot(keep);
    if (lane == 0u)
      wave_counts[wave] = __popcll(ballot);
    __syncthreads();
    uint32_t rank = n_sel + __popcll(ballot & ((1ull << lane) - 1ull));
    for (uint32_t w = 0; w < kThreads / 32u; ++w) {
      rank += w < wave ? wave_counts[w] : 0u;
      n_sel += wave_counts[w];
    }
    if (keep && rank < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP)
      comp_rows[rank] = (uint32_t)ci;
    __syncthreads();
  }
  return n_sel < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP
             ? n_sel
             : DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
}

__device__ __forceinline__ static const float* attention_decode_row(
    const float* raw_kv, const float* comp_kv, const uint32_t* comp_rows,
    bool indexed, uint32_t r, uint32_t n_raw, uint32_t raw_cap,
    uint32_t raw_start) {
  if (r < n_raw) {
    const uint32_t row =
        raw_start + r < raw_cap ? raw_start + r : raw_start + r - raw_cap;
    return raw_kv + (uint64_t)row * DS4_DECODE_ATTENTION_DIM;
  }
  const uint32_t c = r - n_raw;
  return comp_kv +
         (uint64_t)(indexed ? comp_rows[c] : c) * DS4_DECODE_ATTENTION_DIM;
}

inline constexpr uint32_t DS4_DECODE_SCORE_ROWS = 32u;
inline constexpr uint32_t DS4_DECODE_SCORE_LANE_HEADS = 2u;

// Block (x, y) scores rows 32y..32y+31 for 16 heads: lane l of wave w forms
// row 32y + l for heads 16x + 2w and 16x + 2w + 1. The block loads its rows
// with coalesced reads and transposes them through LDS, 128 values per row at
// a time.
template<bool kIndexed>
__global__ __launch_bounds__(256) static void attention_decode_scores_kernel(
    float* scores, uint32_t score_stride, const float* q, const float* raw_kv,
    const float* comp_kv, const int32_t* topk, const float* comp_mask,
    uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
    uint32_t top_k, uint32_t visible_comp, uint32_t n_head, uint32_t head_dim) {
  constexpr uint32_t D = DS4_DECODE_ATTENTION_DIM;
  constexpr uint32_t kRows = DS4_DECODE_SCORE_ROWS;
  constexpr uint32_t kH = DS4_DECODE_SCORE_LANE_HEADS;
  constexpr uint32_t kChunk = 32u;  // float4 per row per chunk
  constexpr uint32_t kChunks = D / 4u / kChunk;
  constexpr uint32_t kRowsPerThread = kRows / 8u;
  __shared__ uint32_t
      comp_rows[kIndexed ? DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP : 1u];
  __shared__ uint32_t wave_counts[8];
  __shared__ float4 tile[kRows][kChunk + 1u];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5u);
  const uint32_t n_sel =
      kIndexed ? attention_decode_select_rows<256>(comp_rows, wave_counts, topk,
                                                   top_k, visible_comp)
               : n_comp;
  const uint32_t n_rows = n_raw + n_sel;
  const uint32_t r0 = blockIdx.y * kRows;
  if (r0 >= n_rows)
    return;

  // Thread t loads float4 t % 32 of every chunk of rows r0 + t / 32 + 8k.
  float4 staged[kChunks][kRowsPerThread];
#pragma unroll
  for (uint32_t k = 0; k < kRowsPerThread; ++k) {
    const uint32_t r = r0 + wave + 8u * k;
    bool load = r < n_rows;
    if (!kIndexed && comp_mask && load && r >= n_raw)
      load = comp_mask[r - n_raw] > -5.0e29f;
    const float4* kv4 = (const float4*)attention_decode_row(
        raw_kv, comp_kv, comp_rows, kIndexed, load ? r : 0u, n_raw, raw_cap,
        raw_start);
#pragma unroll
    for (uint32_t c = 0; c < kChunks; ++c)
      staged[c][k] =
          load ? kv4[c * kChunk + lane] : make_float4(0.f, 0.f, 0.f, 0.f);
  }

  const uint32_t r = r0 + lane;
  const uint32_t h0 = blockIdx.x * 8u * kH + wave * kH;
  float bias = 0.0f;
  bool masked = false;
  if (!kIndexed && comp_mask && r < n_rows && r >= n_raw) {
    bias = comp_mask[r - n_raw];
    masked = bias <= -5.0e29f;
  }
  const float4* qh[kH];
#pragma unroll
  for (uint32_t g = 0; g < kH; ++g)
    qh[g] = (const float4*)(q + (uint64_t)(h0 + g < n_head ? h0 + g : 0u) * D);
  float s[kH][4] = {};
#pragma unroll
  for (uint32_t c = 0; c < kChunks; ++c) {
    if (c)
      __syncthreads();
#pragma unroll
    for (uint32_t k = 0; k < kRowsPerThread; ++k)
      tile[wave + 8u * k][lane] = staged[c][k];
    __syncthreads();
    // Explicit FMAs keep each chain's order.
#pragma unroll 8
    for (uint32_t i = 0; i < kChunk; ++i) {
      const float4 b = tile[lane][i];
#pragma unroll
      for (uint32_t g = 0; g < kH; ++g) {
        const float4 a = qh[g][c * kChunk + i];
        s[g][0] = __builtin_fmaf(a.x, b.x, s[g][0]);
        s[g][1] = __builtin_fmaf(a.y, b.y, s[g][1]);
        s[g][2] = __builtin_fmaf(a.z, b.z, s[g][2]);
        s[g][3] = __builtin_fmaf(a.w, b.w, s[g][3]);
      }
    }
  }
  if (r >= n_rows || h0 >= n_head)
    return;
  // The runtime head_dim keeps v_rsq_f32's scale rather than a folded
  // constant. The pragma keeps the original's (s0 + s2) + (s1 + s3) and its
  // separate mask add.
  const float scale = rsqrtf((float)head_dim);
#pragma unroll
  for (uint32_t g = 0; g < kH; ++g) {
#pragma clang fp reassociate(off) contract(off)
    float score = -3.4e38f;
    if (!masked) {
      score = ((s[g][0] + s[g][2]) + (s[g][1] + s[g][3])) * scale;
      if (!kIndexed && comp_mask && r >= n_raw)
        score += bias;
    }
    if (h0 + g < n_head)
      scores[(uint64_t)(h0 + g) * score_stride + r] = score;
  }
}

inline constexpr uint32_t DS4_DECODE_PV_HEADS = 16u;
inline constexpr uint32_t DS4_DECODE_PV_DIMS = 128u;
inline constexpr uint32_t DS4_DECODE_PV_THREADS =
    DS4_DECODE_PV_DIMS / 4u * (DS4_DECODE_PV_HEADS / 4u);

// Block (x, y) produces dimensions 128y.. of heads 16x..; each thread
// accumulates four consecutive dimensions for four heads. Each head's softmax
// replays the original 256-thread layout: physical thread p holds virtual
// threads p and p + 128.
template<bool kIndexed>
__global__
__launch_bounds__(DS4_DECODE_PV_THREADS) static void attention_decode_softmax_pv_kernel(
    float* heads, const float* scores, uint32_t score_stride,
    const float* raw_kv, const float* comp_kv, const int32_t* topk,
    const float* sinks, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
    uint32_t n_comp, uint32_t top_k, uint32_t visible_comp, uint32_t n_head) {
  constexpr uint32_t D = DS4_DECODE_ATTENTION_DIM;
  constexpr uint32_t kG = DS4_DECODE_PV_HEADS;
  constexpr uint32_t kDims = DS4_DECODE_PV_DIMS;
  constexpr uint32_t kD4 = kDims / 4u;
  constexpr uint32_t kT = DS4_DECODE_PV_THREADS;
  constexpr uint32_t kV = 256u / kT;
  constexpr uint32_t kW = kT / 32u;
  constexpr uint32_t kChunk = 4096u / kG;
  __shared__ uint32_t
      comp_rows[kIndexed ? DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP : 1u];
  __shared__ uint32_t wave_counts[kW];
  __shared__ float wave_red[kG][8];
  __shared__ float head_max[kG], head_inv[kG];
  __shared__ float4 weights[kChunk][kG / 4u];
  __shared__ uint64_t row_addr[kChunk];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t wave = tid >> 5u;
  const uint32_t h0 = blockIdx.x * kG;
  const uint32_t n_sel =
      kIndexed ? attention_decode_select_rows<kT>(comp_rows, wave_counts, topk,
                                                  top_k, visible_comp)
               : n_comp;
  const uint32_t n_rows = n_raw + n_sel;

  float sink[kG];
  const float* sc[kG];
#pragma unroll
  for (uint32_t g = 0; g < kG; ++g) {
    const uint32_t h = h0 + g < n_head ? h0 + g : 0u;
    sink[g] = sinks[h];
    sc[g] = scores + (uint64_t)h * score_stride;
  }
#pragma unroll
  for (uint32_t v = 0; v < kV; ++v) {
    float m[kG];
#pragma unroll
    for (uint32_t g = 0; g < kG; ++g)
      m[g] = sink[g];
    for (uint32_t r = tid + v * kT; r < n_rows; r += 256u)
#pragma unroll
      for (uint32_t g = 0; g < kG; ++g)
        m[g] = fmaxf(m[g], sc[g][r]);
    attention_warp_max_n(m);
    if (lane == 0u)
#pragma unroll
      for (uint32_t g = 0; g < kG; ++g)
        wave_red[g][wave + v * kW] = m[g];
  }
  __syncthreads();
  {
    // The g < kG guards always hold, but dropping them changes the code.
    constexpr uint32_t kPer = kG / kW;
    float x[kPer];
#pragma unroll
    for (uint32_t i = 0; i < kPer; ++i) {
      const uint32_t g = wave + i * kW;
      x[i] = lane < 8u && g < kG ? wave_red[g][lane] : -3.4e38f;
    }
    attention_warp_max_n(x);
    if (lane == 0u)
#pragma unroll
      for (uint32_t i = 0; i < kPer; ++i)
        if (wave + i * kW < kG)
          head_max[wave + i * kW] = x[i];
  }
  __syncthreads();
  float max_score[kG];
#pragma unroll
  for (uint32_t g = 0; g < kG; ++g)
    max_score[g] = head_max[g];
#pragma unroll
  for (uint32_t v = 0; v < kV; ++v) {
    float sum[kG] = {};
    for (uint32_t r = tid + v * kT; r < n_rows; r += 256u)
#pragma unroll
      for (uint32_t g = 0; g < kG; ++g)
        sum[g] += expf(sc[g][r] - max_score[g]);
    if (tid + v * kT == 0u)
#pragma unroll
      for (uint32_t g = 0; g < kG; ++g)
        sum[g] += expf(sink[g] - max_score[g]);
    attention_warp_sum_n(sum);
    if (lane == 0u)
#pragma unroll
      for (uint32_t g = 0; g < kG; ++g)
        wave_red[g][wave + v * kW] = sum[g];
  }
  __syncthreads();
  {
    constexpr uint32_t kPer = kG / kW;
    float x[kPer];
#pragma unroll
    for (uint32_t i = 0; i < kPer; ++i) {
      const uint32_t g = wave + i * kW;
      x[i] = lane < 8u && g < kG ? wave_red[g][lane] : 0.0f;
    }
    attention_warp_sum_n(x);
    if (lane == 0u)
#pragma unroll
      for (uint32_t i = 0; i < kPer; ++i)
        if (wave + i * kW < kG)
          head_inv[wave + i * kW] = 1.0f / x[i];
  }
  // The loop below starts with a barrier only when n_rows > 0.
  __syncthreads();

  const uint32_t d = blockIdx.y * kDims + (tid % kD4) * 4u;
  const uint32_t quad = tid / kD4;
  float acc[4][4] = {};
  for (uint32_t c0 = 0; c0 < n_rows; c0 += kChunk) {
    const uint32_t n = n_rows - c0 < kChunk ? n_rows - c0 : kChunk;
    __syncthreads();
    for (uint32_t j = n + tid; j < kChunk; j += kT)
      row_addr[j] = (uint64_t)attention_decode_row(
          raw_kv, comp_kv, comp_rows, kIndexed, c0, n_raw, raw_cap, raw_start);
    for (uint32_t j = tid; j < n; j += kT) {
      const uint32_t r = c0 + j;
#pragma unroll
      for (uint32_t k = 0; k < kG / 4u; ++k)
        weights[j][k] =
            make_float4(expf(sc[4u * k][r] - max_score[4u * k]),
                        expf(sc[4u * k + 1u][r] - max_score[4u * k + 1u]),
                        expf(sc[4u * k + 2u][r] - max_score[4u * k + 2u]),
                        expf(sc[4u * k + 3u][r] - max_score[4u * k + 3u]));
      row_addr[j] = (uint64_t)attention_decode_row(
          raw_kv, comp_kv, comp_rows, kIndexed, r, n_raw, raw_cap, raw_start);
    }
    __syncthreads();
    // A ring of 16 value loads stays in flight: slot k holds row j + k and
    // is refilled with row j + k + 16 once used. Lane l holds the addresses
    // of rows j + l and j + 32 + l, so each load takes a scalar base from
    // v_readlane. Rows past the chunk read a valid dummy row. FMAs stay in
    // row order.
    auto fma_row = [&](const float4& w, const float4& v) {
      const float wv[4] = {w.x, w.y, w.z, w.w};
      const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (uint32_t a = 0; a < 4u; ++a)
#pragma unroll
        for (uint32_t e = 0; e < 4u; ++e)
          acc[a][e] = __builtin_fmaf(wv[a], vv[e], acc[a][e]);
    };
    typedef float v4f __attribute__((ext_vector_type(4)));
    typedef const __attribute__((address_space(1))) v4f global_v4f;
    auto value = [&](uint64_t lane_addr, uint32_t l) {
      const uint64_t addr = (uint64_t)(uint32_t)__builtin_amdgcn_readlane(
                                (uint32_t)lane_addr, l) |
                            (uint64_t)(uint32_t)__builtin_amdgcn_readlane(
                                (uint32_t)(lane_addr >> 32), l)
                                << 32;
      const v4f v = ((global_v4f*)addr)[d / 4u];
      return make_float4(v.x, v.y, v.z, v.w);
    };
    constexpr uint32_t P = 16;
    static_assert(kChunk % 32u == 0u, "whole 32-row blocks");
    uint64_t off_cur = row_addr[lane];
    float4 ring[P];
#pragma unroll
    for (uint32_t k = 0; k < P; ++k)
      ring[k] = value(off_cur, k);
    auto block = [&](uint32_t j, bool partial) {
      const uint64_t off_nxt =
          j + 32u < kChunk ? row_addr[j + 32u + lane] : off_cur;
#pragma unroll
      for (uint32_t k0 = 0; k0 < 32u; k0 += 8u) {
        float4 wg[8];
#pragma unroll
        for (uint32_t k = 0; k < 8u; ++k)
          wg[k] = weights[j + k0 + k][quad];
#pragma unroll
        for (uint32_t k = 0; k < 8u; ++k) {
          const uint32_t kk = k0 + k;
          if (!partial || j + kk < n)
            fma_row(wg[k], ring[kk % P]);
          ring[kk % P] = kk + P < 32u ? value(off_cur, kk + P)
                                      : value(off_nxt, kk + P - 32u);
        }
      }
      off_cur = off_nxt;
    };
    uint32_t j = 0;
    for (; j + 32u <= n; j += 32u)
      block(j, false);
    if (j < n)
      block(j, true);
  }
#pragma unroll
  for (uint32_t a = 0; a < 4u; ++a) {
    const uint32_t g = 4u * quad + a;
    if (h0 + g < n_head)
      *(float4*)(heads + (uint64_t)(h0 + g) * D + d) =
          make_float4(acc[a][0] * head_inv[g], acc[a][1] * head_inv[g],
                      acc[a][2] * head_inv[g], acc[a][3] * head_inv[g]);
  }
}

static inline uint32_t attention_decode_score_stride(uint32_t n_raw,
                                                     uint32_t max_comp) {
  return n_raw + max_comp;
}

// scores holds n_head * attention_decode_score_stride(n_raw, max_comp)
// floats, where max_comp is n_comp, or min(top_k, n_comp) with topk. Returns
// false for unsupported shapes.
static inline bool attention_decode_one_launch(
    float* heads, float* scores, const float* q, const float* raw_kv,
    const float* comp_kv, const int32_t* topk, const float* comp_mask,
    const float* sinks, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
    uint32_t n_comp, uint32_t top_k, uint32_t visible_comp, uint32_t n_head,
    uint32_t head_dim) {
  if (head_dim != DS4_DECODE_ATTENTION_DIM || n_head == 0u)
    return false;
  const uint32_t max_comp = topk && top_k < n_comp ? top_k : n_comp;
  const uint32_t stride = attention_decode_score_stride(n_raw, max_comp);
  const uint32_t max_rows = n_raw + max_comp;
  if (max_rows != 0u) {
    constexpr uint32_t kScoreBlockHeads = 8u * DS4_DECODE_SCORE_LANE_HEADS;
    const dim3 grid(
        (n_head + kScoreBlockHeads - 1u) / kScoreBlockHeads,
        (max_rows + DS4_DECODE_SCORE_ROWS - 1u) / DS4_DECODE_SCORE_ROWS);
    if (topk)
      attention_decode_scores_kernel<true><<<grid, 256>>>(
          scores, stride, q, raw_kv, comp_kv, topk, comp_mask, n_raw, raw_cap,
          raw_start, n_comp, top_k, visible_comp, n_head, head_dim);
    else
      attention_decode_scores_kernel<false><<<grid, 256>>>(
          scores, stride, q, raw_kv, comp_kv, topk, comp_mask, n_raw, raw_cap,
          raw_start, n_comp, top_k, visible_comp, n_head, head_dim);
  }
  const dim3 grid((n_head + DS4_DECODE_PV_HEADS - 1u) / DS4_DECODE_PV_HEADS,
                  DS4_DECODE_ATTENTION_DIM / DS4_DECODE_PV_DIMS);
  if (topk)
    attention_decode_softmax_pv_kernel<true><<<grid, DS4_DECODE_PV_THREADS>>>(
        heads, scores, stride, raw_kv, comp_kv, topk, sinks, n_raw, raw_cap,
        raw_start, n_comp, top_k, visible_comp, n_head);
  else
    attention_decode_softmax_pv_kernel<false><<<grid, DS4_DECODE_PV_THREADS>>>(
        heads, scores, stride, raw_kv, comp_kv, topk, sinks, n_raw, raw_cap,
        raw_start, n_comp, top_k, visible_comp, n_head);
  return true;
}

static inline bool attention_decode_mixed_one_launch(
    float* heads, float* scores, const float* q, const float* raw_kv,
    const float* comp_kv, const float* comp_mask, const float* sinks,
    uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
    uint32_t use_mask, uint32_t n_head, uint32_t head_dim) {
  return attention_decode_one_launch(heads, scores, q, raw_kv, comp_kv, nullptr,
                                     use_mask ? comp_mask : nullptr, sinks,
                                     n_raw, raw_cap, raw_start, n_comp, 0,
                                     n_comp, n_head, head_dim);
}

static inline bool attention_decode_indexed_mixed_one_launch(
    float* heads, float* scores, const float* q, const float* raw_kv,
    const float* comp_kv, const int32_t* topk, const float* sinks,
    uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
    uint32_t top_k, uint32_t pos0, uint32_t ratio, uint32_t n_head,
    uint32_t head_dim) {
  uint32_t visible = n_comp;
  if (ratio != 0u && (pos0 + 1u) / ratio < visible)
    visible = (pos0 + 1u) / ratio;
  return attention_decode_one_launch(heads, scores, q, raw_kv, comp_kv, topk,
                                     nullptr, sinks, n_raw, raw_cap, raw_start,
                                     n_comp, top_k, visible, n_head, head_dim);
}
