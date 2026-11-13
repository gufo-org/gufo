#include "staging/models/glm5_next/kernels/rocm/dense_wmma.hpp"

#include <cassert>
#include <cstddef>

namespace gufo::models::glm5_next::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kRowsPerBlock = 128;
constexpr int kStageK = 256;  // one super-block
constexpr int kTileTokens = 32;
constexpr int kActStride = kStageK + 8;

// Q5_K super-block: 44 dwords exactly, staged at an odd dword stride so the
// 16 rows a fragment read touches sit on distinct bank groups.
constexpr int kQ5KBytes = 176;
constexpr int kQ5KStride = 180;

// One stage of Q8_0: eight 34-byte blocks (fp16 scale + 32 signed bytes),
// staged contiguously with the same odd-dword stride trick.
constexpr int kQ8Bytes = 272;
constexpr int kQ8Stride = 276;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

struct Q5KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qh[32];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q5KBlock) == 176, "block_q5_K must be 176 bytes");

__device__ __forceinline__ std::uint32_t U32At(const std::uint8_t* p) {
  std::uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

__device__ __forceinline__ float F16At(const std::uint8_t* p) {
  __half_raw raw;
  raw.x = static_cast<std::uint16_t>(p[0] |
                                     (static_cast<std::uint16_t>(p[1]) << 8));
  return __half2float(*reinterpret_cast<const __half*>(&raw));
}

// 6-bit scale/min field j of the 12-byte scales array (core Q4Value walk).
__device__ __forceinline__ void Q4ScaleMinFromWords(std::uint32_t lo,
                                                    std::uint32_t mid,
                                                    std::uint32_t hi, int index,
                                                    int& scale, int& minimum) {
  if (index < 4) {
    scale = static_cast<int>((lo >> (8 * index)) & 0x3FU);
    minimum = static_cast<int>((mid >> (8 * index)) & 0x3FU);
  } else {
    const std::uint32_t pj4 = (hi >> (8 * (index - 4))) & 0xFFU;
    scale = static_cast<int>(pj4 & 0xFU) |
            (static_cast<int>(((lo >> (8 * (index - 4))) & 0xFFU) >> 6) << 4);
    minimum =
        static_cast<int>(pj4 >> 4) |
        (static_cast<int>(((mid >> (8 * (index - 4))) & 0xFFU) >> 6) << 4);
  }
}

/// The 16 elements [k0, k0 + 16) of the staged Q5_K super-block `blk` as one
/// f16 wmma fragment operand. All 16 share the sub-block (j = k0 >> 5 spans
/// 32 elements), so one (scale, min) pair and one qh bit position serve the
/// fragment; byte and nibble selection mirror the scalar walk element-wise.
__device__ __forceinline__ v16h
DecodeFragmentHalfQ5K(const std::uint8_t* __restrict__ blk, int k0) {
  const float d = F16At(blk);
  const float dmin = F16At(blk + 2);
  const int j = k0 >> 5;
  int scv = 0;
  int m0 = 0;
  Q4ScaleMinFromWords(U32At(blk + 4), U32At(blk + 8), U32At(blk + 12), j, scv,
                      m0);
  const float dmn = dmin * static_cast<float>(m0);
  const std::uint8_t* qh = blk + offsetof(Q5KBlock, qh);
  const std::uint8_t* qs = blk + offsetof(Q5KBlock, qs);
  const int l0 = k0 & 31;
  const int byte = (j >> 1) * 32 + l0;
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const std::uint8_t qb = qs[byte + i];
    const int quant4 = (j & 1) ? (qb >> 4) : (qb & 0xF);
    const int quant = quant4 + ((((qh[l0 + i] >> j) & 1) != 0) ? 16 : 0);
    halves[i] = __float2half((d * static_cast<float>(scv * quant)) - dmn);
  }
  return frag;
}

/// The 16 elements [k0, k0 + 16) of the staged Q8_0 super-block `blk` as one
/// f16 wmma fragment operand. All 16 share one 34-byte block, so a single
/// scale and a contiguous signed-byte run decode the whole fragment.
__device__ __forceinline__ v16h
DecodeFragmentHalfQ8(const std::uint8_t* __restrict__ blk, int k0) {
  const int base = (k0 >> 5) * 34;
  const float d = F16At(blk + base);
  const std::int8_t* q =
      reinterpret_cast<const std::int8_t*>(blk + base + 2) + (k0 & 31);
  v16h frag;
  auto* halves = reinterpret_cast<__half*>(&frag);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    halves[i] = __float2half(d * static_cast<float>(q[i]));
  }
  return frag;
}

/// Dense GEMM over one 32-token tile of a contiguous [rows x cols] Q5_K or
/// Q8_0 matrix: out[t * out_stride + r] = sum_k W[r][k] * x[t][k]. Rows are
/// the wmma A operand staged once per block, tokens the F16-converted B
/// operand; see the routed kernel for the shared staging geometry.
template<int kSuper, int kStride, bool kQ8>
__global__ void __launch_bounds__(kThreads)
    DenseWmmaKernel(const std::uint8_t* __restrict__ w,
                    const float* __restrict__ x, std::uint32_t x_stride,
                    std::uint32_t rows, std::uint32_t tokens,
                    std::uint32_t cols, float* __restrict__ out,
                    std::uint32_t out_stride) {
  constexpr int kTokTiles = kTileTokens / 16;
  constexpr int kChunksPerTok = kStageK / 8;
  constexpr int kActChunks = kTileTokens * kChunksPerTok;
  constexpr int kActPer = (kActChunks + kThreads - 1) / kThreads;
  constexpr int kWords = (kSuper + 3) / 4;
  constexpr int kWords0 = (kWords + 1) / 2;
  constexpr int kWords1 = kWords - kWords0;
  __shared__
      __attribute__((aligned(16))) std::uint8_t s_w[kRowsPerBlock * kStride];
  __shared__
      __attribute__((aligned(16))) __half s_act[kTileTokens * kActStride];

  const int t0 = blockIdx.y * kTileTokens;
  const int live_tiles =
      min(kTokTiles, (static_cast<int>(tokens) - t0 + 15) / 16);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % kWave;
  const int wave = tid / kWave;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{cols} / 256 * kSuper;

  // Weight fetch: two threads per staged row move the kWords dwords of the
  // stage's super-block (p0 the first kWords0, p1 the rest; kSuper is a
  // whole number of dwords, so every read stays inside the block).
  const int f_row = tid >> 1;
  const int f_part = tid & 1;
  const std::uint32_t f_global = blockIdx.x * kRowsPerBlock + f_row;
  const bool f_live = f_global < rows;
  const std::size_t f_off = (f_live ? f_global : rows - 1) * row_bytes;
  std::uint32_t f_word[kWords0];
  const auto fetch_weights = [&](int stage) {
    const std::size_t base = f_part == 0 ? 0U : 4U * kWords0;
    const std::uint8_t* block = w + f_off + stage * kSuper + base;
#pragma unroll
    for (int i = 0; i < kWords0; ++i) {
      if (i < kWords1 || f_part == 0) {
        f_word[i] = U32At(block + i * 4);
      }
    }
  };
  const auto commit_weights = [&] {
    const std::uint32_t d0 = f_part == 0 ? 0U : kWords0;
    auto* dst = reinterpret_cast<std::uint32_t*>(s_w + f_row * kStride);
#pragma unroll
    for (int i = 0; i < kWords1 || (i < kWords0 && f_part == 0); ++i) {
      dst[d0 + i] = f_word[i];
    }
  };

  // Activation fetch: chunk c is token c / 32 at K offset 8 * (c % 32); the
  // staged read converts the 8 F32 values to the F16 fragment layout.
  std::int32_t a_src[kActPer];
#pragma unroll
  for (int i = 0; i < kActPer; ++i) {
    const int chunk = tid + i * kThreads;
    const int t = chunk / kChunksPerTok;
    a_src[i] = -1;
    if (chunk < kActChunks && t0 + t < static_cast<int>(tokens)) {
      a_src[i] = static_cast<std::int32_t>(static_cast<std::size_t>(t0 + t) *
                                               x_stride +
                                           (chunk % kChunksPerTok) * 8);
    }
  }
  float4 a_data[kActPer][2];
  const auto fetch_act = [&](int stage) {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      if (a_src[i] >= 0) {
        const float* p = x + a_src[i] + stage * kStageK;
        a_data[i][0] = *reinterpret_cast<const float4*>(p);
        a_data[i][1] = *reinterpret_cast<const float4*>(p + 4);
      } else {
        a_data[i][0] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        a_data[i][1] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      }
    }
  };
  const auto commit_act = [&] {
#pragma unroll
    for (int i = 0; i < kActPer; ++i) {
      const int chunk = tid + i * kThreads;
      if (chunk < kActChunks) {
        const float* f = reinterpret_cast<const float*>(&a_data[i][0]);
        v16h cvt;
        auto* h = reinterpret_cast<__half*>(&cvt);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          *reinterpret_cast<__half2*>(h + 2 * q) =
              __float22half2_rn(*reinterpret_cast<const float2*>(f + 2 * q));
        }
        *reinterpret_cast<uint4*>(s_act + (chunk / kChunksPerTok) * kActStride +
                                  (chunk % kChunksPerTok) * 8) =
            *reinterpret_cast<const uint4*>(&cvt);
      }
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  }
  const int stages = static_cast<int>(cols / kStageK);
  fetch_weights(0);
  fetch_act(0);
  for (int stage = 0; stage < stages; ++stage) {
    commit_weights();
    commit_act();
    __syncthreads();
    if (stage + 1 < stages) {
      fetch_weights(stage + 1);
      fetch_act(stage + 1);
    }
#pragma unroll
    for (int s = 0; s < kStageK / 32; ++s) {
      const std::uint8_t* row = s_w + (wave * 16 + sub) * kStride;
      v16h w_lo;
      v16h w_hi;
      if constexpr (kQ8) {
        w_lo = DecodeFragmentHalfQ8(row, 32 * s);
        w_hi = DecodeFragmentHalfQ8(row, 32 * s + 16);
      } else {
        w_lo = DecodeFragmentHalfQ5K(row, 32 * s);
        w_hi = DecodeFragmentHalfQ5K(row, 32 * s + 16);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if (j < live_tiles) {
          const __half* b = s_act + (j * 16 + sub) * kActStride + s * 32;
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, b, 32);
          __builtin_memcpy(&b_hi, b + 16, 32);
          acc[j] = Wmma(w_lo, b_lo, acc[j]);
          acc[j] = Wmma(w_hi, b_hi, acc[j]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    const std::size_t t = static_cast<std::size_t>(t0) + j * 16 + sub;
    if (t >= tokens) {
      continue;
    }
    float* out_row = out + t * out_stride;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t r =
          blockIdx.x * kRowsPerBlock + wave * 16 + 2 * i + half;
      if (r < rows) {
        out_row[r] = acc[j][i];
      }
    }
  }
}

}  // namespace

bool DenseWmmaSupports(GemvType type) {
  return type == GemvType::kQ5_K || type == GemvType::kQ8_0;
}

void DenseWmma(const void* base, GemvType type, std::uint32_t rows,
               std::uint32_t cols, const float* x, std::uint32_t x_stride,
               float* out, std::uint32_t out_stride, std::uint32_t tokens,
               hipStream_t stream) {
  assert(DenseWmmaSupports(type));
  assert(rows > 0 && cols % 256 == 0 && tokens >= 16);
  const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock,
                  (tokens + kTileTokens - 1) / kTileTokens);
  if (type == GemvType::kQ8_0) {
    DenseWmmaKernel<kQ8Bytes, kQ8Stride, true><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(base), x, x_stride, rows, tokens, cols,
        out, out_stride);
    return;
  }
  DenseWmmaKernel<kQ5KBytes, kQ5KStride, false><<<grid, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(base), x, x_stride, rows, tokens, cols,
      out, out_stride);
}

}  // namespace gufo::models::glm5_next::rocm
