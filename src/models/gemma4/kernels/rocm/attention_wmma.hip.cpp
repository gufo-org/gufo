// Prefill attention for Gemma 4 on the gfx1151 WMMA matrix cores.
//
// Tiling. A block of eight wave32 waves covers kRowBlocks row blocks of 16
// query rows each; kHeads heads sharing one KV head times kQueryBlocks
// query blocks form those rows, so every staged K/V tile serves all of them.
// Each row block is split across kWavesPerRow waves by head dimension: a
// wave keeps a 128-dim slice of its rows' Q fragments and O accumulators in
// registers (64 + 64 VGPRs for both hd256 and hd512).
//
// Per 16-key tile: stage K ([key][dim]) and V transposed ([dim][key]) in
// LDS, reading the ring slot of each key for sliding layers; every wave adds
// its slice's partial scores to LDS; one wave per row block runs the online
// softmax over the summed scores (window, causal and key-limit masks per
// element) and writes the binary16 probabilities; every wave rescales its O
// slice and accumulates P V.
//
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16
// contiguous k, B holds column L%16 and 16 contiguous k, and C element i is
// row 2i + L/16, column L%16; both half-waves carry the same operands.
#include "src/models/gemma4/kernels/rocm/attention_wmma.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "src/models/gemma4/kernels/rocm/half_store.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kSlice = 128;  // head dims per wave
constexpr std::uint32_t kSliceSteps = kSlice / 16;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

struct Bounds {
  std::uint32_t lo;
  std::uint32_t hi;
};

__device__ inline Bounds KeyBounds(const AttentionArgs& a, std::uint32_t row,
                                   std::uint32_t position) {
  const std::uint32_t end =
      a.key_ends != nullptr ? max(position + 1, a.key_ends[row]) : position + 1;
  const std::uint32_t hi = min(end, a.key_limit);
  std::uint32_t lo = 0;
  if (a.window != 0 && position + 1 > a.window) {
    lo = position + 1 - a.window;
  }
  return {min(lo, hi), hi};
}

template<std::uint32_t D, std::uint32_t kHeads, std::uint32_t kQueryBlocks>
constexpr std::uint32_t kPrefillThreads =
    32 * kHeads * kQueryBlocks * D / kSlice;

/// Rotated pairs of the production global layers' derived keys.
constexpr std::uint32_t kGlobalPairs = 64;

/// kGlobal: a production global layer (hd512 derived keys with
/// kGlobalPairs pairs, no window, ring or image ranges). Its non-rotated key
/// dims are staged from the value registers and only the rotated dims come
/// from the K cache, which frees the registers two blocks per WGP need.
template<std::uint32_t D, std::uint32_t kHeads, std::uint32_t kQueryBlocks,
         bool kGlobal>
__global__ void __launch_bounds__((kPrefillThreads<D, kHeads, kQueryBlocks>))
    __attribute__((amdgpu_waves_per_eu(kGlobal || D == 256 ? 8 : 1)))
    WmmaPrefillAttentionKernel(AttentionArgs a) {
  if constexpr (kGlobal) {
    a.window = 0;
    a.ring = 0;
    a.key_ends = nullptr;
    a.rope_pairs = kGlobalPairs;
  }
  constexpr std::uint32_t kWavesPerRow = D / kSlice;
  constexpr std::uint32_t kRowBlocks = kHeads * kQueryBlocks;
  constexpr std::uint32_t kThreads = kPrefillThreads<D, kHeads, kQueryBlocks>;
  static_assert(kThreads <= 1024, "at most 32 waves per block");
  constexpr std::uint32_t kKStride = D + 8;       // K rows, halves
  constexpr std::uint32_t kVtStride = kKeys + 8;  // V^T rows, halves
  constexpr std::uint32_t kQueries = kQueryBlocks * 16;

  __shared__ __align__(16) __half k_lds[kKeys * kKStride];
  __shared__ __align__(16) __half vt_lds[D * kVtStride];
  __shared__ float s_lds[kRowBlocks][kWavesPerRow][16][17];
  __shared__ __align__(16) __half p_lds[kRowBlocks][16][kKeys + 8];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  // Wave-uniform: the row block, head and slice offsets stay scalar.
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t rb = wave / kWavesPerRow;
  const std::uint32_t part = wave % kWavesPerRow;
  const std::uint32_t dim0 = part * kSlice;

  const std::uint32_t query_start = blockIdx.x * kQueries;
  const std::uint32_t head_base = blockIdx.y * kHeads;
  const std::uint32_t gqa = a.heads / a.kv_heads;
  const std::uint32_t kv_head = head_base / gqa;
  const std::uint32_t head = head_base + rb % kHeads;
  const std::uint32_t row0 = query_start + (rb / kHeads) * 16;  // this rb

  const auto position_of = [&](std::uint32_t row) {
    return a.shared_position ? a.first_position : a.first_position + row;
  };
  // Keys the whole block may need.
  const std::uint32_t last_row = min(query_start + kQueries, a.rows) - 1;
  const std::uint32_t block_lo =
      KeyBounds(a, query_start, position_of(query_start)).lo & ~(kKeys - 1);
  const std::uint32_t block_hi =
      KeyBounds(a, last_row, position_of(last_row)).hi;

  // Q slice fragments: row `sub` of this row block, dims [dim0, dim0 + 128).
  // kGlobal keeps only its half-wave's half of each fragment (the halves are
  // duplicates) and rebuilds the fragment with a cross-half swap before its
  // product: 32 registers fewer.
  v16h q_frag[kGlobal ? 1 : kSliceSteps];
  uint4 q_half[kGlobal ? kSliceSteps : 1];
  {
    const std::uint32_t row = row0 + sub;
    const bool live = row < a.rows;
    const float* q =
        a.q + (static_cast<std::size_t>(row) * a.heads + head) * D + dim0;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
      if constexpr (kGlobal) {
        const float* src = q + ks * 16 + half_id * 8;
        const float4 f0 = live ? *reinterpret_cast<const float4*>(src)
                               : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        const float4 f1 = live ? *reinterpret_cast<const float4*>(src + 4)
                               : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        const __half2 h[4] = {
            __floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w),
            __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
        __builtin_memcpy(&q_half[ks], h, 16);
      } else {
#pragma unroll
        for (std::uint32_t v = 0; v < 16; v += 4) {
          const float4 f =
              live ? *reinterpret_cast<const float4*>(q + ks * 16 + v)
                   : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
          q_frag[ks][v + 0] = static_cast<_Float16>(f.x);
          q_frag[ks][v + 1] = static_cast<_Float16>(f.y);
          q_frag[ks][v + 2] = static_cast<_Float16>(f.z);
          q_frag[ks][v + 3] = static_cast<_Float16>(f.w);
        }
      }
    }
  }
  const auto q_fragment = [&](std::uint32_t ks) -> v16h {
    if constexpr (kGlobal) {
      const uint4 own = q_half[ks];
      const auto swap = [](std::uint32_t v) {
        return static_cast<std::uint32_t>(__builtin_amdgcn_permlanex16(
            v, v, 0x76543210U, 0xFEDCBA98U, false, false));
      };
      const uint4 other =
          make_uint4(swap(own.x), swap(own.y), swap(own.z), swap(own.w));
      union {
        v16h f;
        uint4 u[2];
      } cvt;
      cvt.u[0] = half_id == 0 ? own : other;
      cvt.u[1] = half_id == 0 ? other : own;
      return cvt.f;
    } else {
      return q_frag[ks];
    }
  };
  v8f o_acc[kSliceSteps] = {};

  // Online softmax state, spread over the row block's waves: wave `part`
  // owns rows [part R, part R + R) with R = 16 / kWavesPerRow, and each row's
  // kSegLanes lanes cover kSegKeys keys of every tile.
  constexpr std::uint32_t kRowsPerPart = 16 / kWavesPerRow;
  constexpr std::uint32_t kSegLanes = 32 / kRowsPerPart;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t sm_row = part * kRowsPerPart + lane / kSegLanes;
  const std::uint32_t sm_seg = lane % kSegLanes;
  const std::uint32_t sm_query = row0 + sm_row;
  const std::uint32_t sm_clamped = min(sm_query, a.rows - 1);
  const Bounds sm_keys = KeyBounds(a, sm_clamped, position_of(sm_clamped));

  const std::size_t kv_stride = static_cast<std::size_t>(a.kv_heads) * D;
  const auto* k_cache = reinterpret_cast<const __half*>(a.k_cache);
  const auto* v_cache = reinterpret_cast<const __half*>(a.v_cache);
  const auto slot_of = [&](std::uint32_t key) {
    return a.ring != 0 ? key % a.ring : key;
  };

  // Each thread stages kStage 8-half chunks of K and V per tile; the next
  // tile's chunks are loaded into registers while the current one computes.
  constexpr std::uint32_t kChunks = kKeys * (D / 8);
  static_assert(kChunks % kThreads == 0, "whole chunks per thread");
  constexpr std::uint32_t kStage = kChunks / kThreads;
  uint4 k_next[kGlobal ? 1 : kStage];
  uint4 v_next[kStage];
  // Derived keys (rope_pairs > 0): a thread's two chunks are dims d and
  // d + D / 2 of one key; rotated dims come from the rotated-dims K cache,
  // every other key dim is its value (the query carries the key weight).
  // Each chunk's key source is fixed per thread; only the key slot moves.
  static_assert(D != 512 || (kStage == 2 && kThreads == D),
                "a thread's chunks must pair dims d and d + D / 2");
  const __half* key_base[kStage];
  std::uint32_t key_stride[kStage];
#pragma unroll
  for (std::uint32_t j = 0; j < (kGlobal ? 0 : kStage); ++j) {
    const std::uint32_t d8 = ((tid + j * kThreads) / kKeys) * 8;
    const std::uint32_t pair0 = d8 % (D / 2);
    const std::uint32_t pairs = a.rope_pairs;
    if (pairs == 0) {
      key_base[j] = k_cache + kv_head * D + d8;
      key_stride[j] = static_cast<std::uint32_t>(kv_stride);
    } else if (pair0 < pairs) {
      key_base[j] =
          k_cache + kv_head * 2 * pairs + (d8 < D / 2 ? 0 : pairs) + pair0;
      key_stride[j] = a.kv_heads * 2 * pairs;
    } else {
      key_base[j] = v_cache + kv_head * D + d8;
      key_stride[j] = static_cast<std::uint32_t>(kv_stride);
    }
  }
  // Value chunk j of this thread: generic, key idx % 16 and dims 8 (idx / 16)
  // (idx = tid + j * threads); kGlobal, the key pair 2 (tid % 8) + j over
  // dims 8 (tid / 8), so the transposed pairs form in one thread.
  const auto chunk_key = [&](std::uint32_t j) {
    return kGlobal ? 2 * (tid % 8) + j : (tid + j * kThreads) % kKeys;
  };
  const auto chunk_d8 = [&](std::uint32_t j) {
    return kGlobal ? (tid / 8) * 8 : ((tid + j * kThreads) / kKeys) * 8;
  };
  // kGlobal: rotated chunk r = tid < 256 is dims 8 (r / 16) of the rotated
  // K row of key r % 16 (dims [0, 64) then [D / 2, D / 2 + 64)).
  constexpr std::uint32_t kRotChunks = kKeys * 2 * kGlobalPairs / 8;
  uint4 k_rot;
  const auto fetch = [&](std::uint32_t key0) {
#pragma unroll
    for (std::uint32_t j = 0; j < kStage; ++j) {
      const std::uint32_t position = key0 + chunk_key(j);
      const std::uint32_t d8 = chunk_d8(j);
      if constexpr (!kGlobal) {
        k_next[j] = make_uint4(0U, 0U, 0U, 0U);
      }
      v_next[j] = make_uint4(0U, 0U, 0U, 0U);
      if (position < block_hi) {
        const std::size_t slot = slot_of(position);
        if constexpr (!kGlobal) {
          k_next[j] = *reinterpret_cast<const uint4*>(key_base[j] +
                                                      slot * key_stride[j]);
        }
        v_next[j] = *reinterpret_cast<const uint4*>(
            v_cache + slot * kv_stride + static_cast<std::size_t>(kv_head) * D +
            d8);
      }
    }
    if constexpr (kGlobal) {
      const std::uint32_t position = key0 + tid % kKeys;
      k_rot = make_uint4(0U, 0U, 0U, 0U);
      if (tid < kRotChunks && position < block_hi) {
        k_rot = *reinterpret_cast<const uint4*>(
            k_cache +
            (static_cast<std::size_t>(position) * a.kv_heads + kv_head) * 2 *
                kGlobalPairs +
            (tid / kKeys) * 8);
      }
    }
  };
  // Staging. K goes row-major; V is transposed with adjacent keys paired
  // into 32-bit words: lanes 2m and 2m + 1 hold keys 2m and 2m + 1 of the
  // same 8 dims, swap half of them, and each writes 4 dims' key pairs.
  const auto store_keys = [&] {
#pragma unroll
    for (std::uint32_t j = 0; j < kStage; ++j) {
      const std::uint32_t d8 = chunk_d8(j);
      if constexpr (kGlobal) {
        // Non-rotated key dims are the values.
        if (d8 % (D / 2) >= kGlobalPairs) {
          *reinterpret_cast<uint4*>(&k_lds[chunk_key(j) * kKStride + d8]) =
              v_next[j];
        }
      } else {
        *reinterpret_cast<uint4*>(&k_lds[chunk_key(j) * kKStride + d8]) =
            k_next[j];
      }
    }
    if constexpr (kGlobal) {
      if (tid < kRotChunks) {
        const std::uint32_t c = tid / kKeys;  // 8-dim group of the rotated row
        const std::uint32_t d8 =
            c < kGlobalPairs / 8 ? c * 8 : D / 2 + (c - kGlobalPairs / 8) * 8;
        *reinterpret_cast<uint4*>(&k_lds[(tid % kKeys) * kKStride + d8]) =
            k_rot;
      }
    }
  };
  const auto store_values = [&] {
    if constexpr (kGlobal) {
      // Keys 2 m and 2 m + 1 of dims d8 .. d8 + 7: one 32-bit key pair per
      // dim, built by byte permutes.
      const std::uint32_t m = tid % 8;
      const std::uint32_t d8 = chunk_d8(0);
      const uint4 a = v_next[0];
      const uint4 b = v_next[1];
      const std::uint32_t aw[4] = {a.x, a.y, a.z, a.w};
      const std::uint32_t bw[4] = {b.x, b.y, b.z, b.w};
      auto* rows = reinterpret_cast<std::uint32_t*>(vt_lds);
#pragma unroll
      for (std::uint32_t w = 0; w < 4; ++w) {
        rows[(d8 + 2 * w) * (kVtStride / 2) + m] =
            __builtin_amdgcn_perm(bw[w], aw[w], 0x05040100U);
        rows[(d8 + 2 * w + 1) * (kVtStride / 2) + m] =
            __builtin_amdgcn_perm(bw[w], aw[w], 0x07060302U);
      }
      return;
    }
    const bool even = (tid & 1U) == 0;
#pragma unroll
    for (std::uint32_t j = 0; j < kStage; ++j) {
      const std::uint32_t idx = tid + j * kThreads;
      const std::uint32_t key = idx % kKeys;
      const std::uint32_t d8 = (idx / kKeys) * 8;
      const uint4 v = v_next[j];
      // Even lanes keep dims 0..3 and receive the odd key's, odd lanes keep
      // dims 4..7 and receive the even key's.
      const std::uint32_t send0 = even ? v.z : v.x;
      const std::uint32_t send1 = even ? v.w : v.y;
      const std::uint32_t recv0 =
          static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
              static_cast<int>(send0), 0xB1, 0xF, 0xF, true));
      const std::uint32_t recv1 =
          static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
              static_cast<int>(send1), 0xB1, 0xF, 0xF, true));
      const std::uint32_t own[2] = {even ? v.x : v.z, even ? v.y : v.w};
      const std::uint32_t other[2] = {recv0, recv1};
      auto* rows = reinterpret_cast<std::uint32_t*>(vt_lds);
      const std::uint32_t dim = d8 + (even ? 0U : 4U);
#pragma unroll
      for (std::uint32_t i = 0; i < 4; ++i) {
        const std::uint32_t mine = (own[i / 2] >> (16 * (i % 2))) & 0xFFFFU;
        const std::uint32_t theirs = (other[i / 2] >> (16 * (i % 2))) & 0xFFFFU;
        const std::uint32_t pair =
            even ? (mine | (theirs << 16)) : (theirs | (mine << 16));
        rows[(dim + i) * (kVtStride / 2) + key / 2] = pair;
      }
    }
  };
  if (block_lo < block_hi) {
    fetch(block_lo);
    store_keys();
    store_values();
  }
  __syncthreads();

  // Three barriers per tile: after the scores (the next keys may replace
  // this tile's), after the softmax (probabilities and next keys are
  // staged), after the values (the next values may replace this tile's).
  for (std::uint32_t key0 = block_lo; key0 < block_hi; key0 += kKeys) {
    const bool more = key0 + kKeys < block_hi;
    if (more) {
      fetch(key0 + kKeys);
    }

    // Partial scores over this wave's dim slice.
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h k_frag = LoadFrag(&k_lds[sub * kKStride + dim0 + ks * 16]);
        s_acc = Wmma(q_fragment(ks), k_frag, s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][part][2 * i + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    {
      const std::uint32_t position_row = sm_query;
      float vals[kSegKeys];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const std::uint32_t col = sm_seg * kSegKeys + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
#pragma unroll
        for (std::uint32_t p = 0; p < kWavesPerRow; ++p) {
          s += s_lds[rb][p][sm_row][col];
        }
        const bool valid =
            position_row < a.rows && key >= sm_keys.lo && key < sm_keys.hi;
        vals[m] = valid ? s : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t offset = 1; offset < kSegLanes; offset <<= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset, 32));
      }
      const float next_max = fmaxf(running_max, tile_max);
      const float scale =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const float w = vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += w;
        p_lds[rb][sm_row][sm_seg * kSegKeys + m] = __float2half(w);
      }
#pragma unroll
      for (std::uint32_t offset = 1; offset < kSegLanes; offset <<= 1) {
        tile_sum += __shfl_xor(tile_sum, offset, 32);
      }
      running_max = next_max;
      running_sum = running_sum * scale + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = scale;
      }
    }
    // The scores are done with this tile's keys: stage the next ones.
    if (more) {
      store_keys();
    }
    __syncthreads();

    float scale[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      scale[i] = row_scale[rb][2 * i + half_id];
    }
    const v16h p_frag = LoadFrag(&p_lds[rb][sub][0]);
    // Rescale the accumulators before the matrix products, and only when a
    // row's running maximum moved (a factor of 1 is exact): after the first
    // tiles almost none do, and the products then issue back to back.
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && scale[i] == 1.0F;
    }
    if (!__all(unchanged)) {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= scale[i];
        }
      }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      const v16h v_frag = LoadFrag(&vt_lds[(dim0 + t * 16 + sub) * kVtStride]);
      o_acc[t] = Wmma(p_frag, v_frag, o_acc[t]);
    }
    // Every wave is done with this tile's values and probabilities before the
    // next values replace them.
    __syncthreads();
    if (more) {
      store_values();
    }
  }

  if (sm_seg == 0) {
    row_sum[rb][sm_row] = running_sum;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t r = 2 * i + half_id;
    const std::uint32_t row = row0 + r;
    if (row >= a.rows) {
      continue;
    }
    const float l = row_sum[rb][r];
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    const std::size_t at =
        (static_cast<std::size_t>(row) * a.heads + head) * D + dim0 + sub;
    auto* out_half = static_cast<__half*>(a.out_half);
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      const float v = o_acc[t][i] * inv;
      if (a.out != nullptr) {
        a.out[at + t * 16] = v;
      }
      if (out_half != nullptr) {
        out_half[at + t * 16] = HalfOf(v);
      }
    }
  }
}

template<std::uint32_t D, std::uint32_t kHeads, std::uint32_t kQueryBlocks>
bool Launch(const AttentionArgs& a, hipStream_t stream) {
  if ((a.heads / a.kv_heads) % kHeads != 0) {
    return false;
  }
  constexpr std::uint32_t kQueries = 16 * kQueryBlocks;
  const dim3 grid((a.rows + kQueries - 1) / kQueries, a.heads / kHeads);
  constexpr std::uint32_t kThreads = kPrefillThreads<D, kHeads, kQueryBlocks>;
  if constexpr (D == 512) {
    if (a.rope_pairs == kGlobalPairs && a.window == 0 && a.ring == 0 &&
        a.key_ends == nullptr) {
      WmmaPrefillAttentionKernel<D, kHeads, kQueryBlocks, true>
          <<<grid, kThreads, 0, stream>>>(a);
      return true;
    }
  }
  WmmaPrefillAttentionKernel<D, kHeads, kQueryBlocks, false>
      <<<grid, kThreads, 0, stream>>>(a);
  return true;
}

}  // namespace

bool LaunchWmmaPrefillAttention(const AttentionArgs& a, hipStream_t stream) {
  if (a.kv_heads == 0 || a.heads % a.kv_heads != 0) {
    return false;
  }
  if (a.head_dim == 256) {
    // Two heads x four 16-query blocks, two waves per row block: the
    // staged window serves 128 rows (the kernel is bound by its K/V
    // traffic), and 192 VGPRs keep two blocks per WGP.
    return Launch<256, 2, 4>(a, stream);
  }
  if (a.head_dim == 512) {
    // Four heads x one 16-query block, four waves per row block.
    return Launch<512, 4, 1>(a, stream);
  }
  return false;
}

}  // namespace gufo::models::gemma4::rocm
