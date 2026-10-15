// Width-invariant WMMA projections (wmma_gemv.hpp states the arithmetic).
//
// A block computes 128 weight rows (eight waves of 16) over a range of
// super-blocks. The weight rows are the WMMA B operand: lane L holds row
// L % 16 (both half-waves alike) and loads its own super-block straight
// from memory one super-block ahead, so each lane applies its own row's scale
// (and minimum). The activation tiles are the A operand, staged per super-block
// in LDS from the pack; C lane L then holds row L % 16 for tile rows
// 2 i + L / 16.
#include "src/models/gemma4/kernels/rocm/wmma_gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#ifndef ROTMUL
#define ROTMUL 5
#endif

namespace gufo::models::gemma4::rocm {
namespace {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

constexpr int kThreads = 256;
constexpr int kRowsPerBlock = 128;
constexpr int kSuper = 256;
/// Halves per staged activation row: a super-block plus padding against
/// bank conflicts on the fragment reads.
constexpr int kActStride = kSuper + 8;
constexpr std::uint32_t kMaxSplits = 6;
/// Outputs below which a projection may split its reduction (partials).
constexpr std::uint32_t kSplitOutputs = 8192;

/// Weight formats of the WMMA projections.
enum class WmmaFormat { kQ4_K, kQ5_K, kQ6_K, kQ4_0 };

/// Pack layout for `k`: hi and lo halves [16][k], then per sub-block b and
/// row t the inverse scale and the sum of x, each [k / 32][16].
struct Pack {
  __half* hi;
  __half* lo;
  float* inv;
  float* sums;
  float* partials;
};

Pack PackOf(void* scratch, std::uint32_t k) {
  auto* bytes = static_cast<std::uint8_t*>(scratch);
  const std::size_t halves = std::size_t{kWmmaGemvRows} * k;
  const std::size_t blocks = std::size_t{kWmmaGemvRows} * (k / 32);
  Pack p{};
  p.hi = reinterpret_cast<__half*>(bytes);
  p.lo = p.hi + halves;
  p.inv = reinterpret_cast<float*>(p.lo + halves);
  p.sums = p.inv + blocks;
  p.partials = p.sums + blocks;
  return p;
}

/// One thread per (row, 32-value sub-block): the scale 2^e putting the
/// sub-block's maximum in [2^13, 2^14), x 2^e as binary16 hi + lo, 2^-e, and
/// the sequential FP32 sum of x.
__global__ void PackKernel(const float* __restrict__ x, std::uint32_t rows,
                           std::uint32_t k, __half* __restrict__ hi,
                           __half* __restrict__ lo, float* __restrict__ inv,
                           float* __restrict__ sums) {
  const std::uint32_t blocks = k / 32;
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * blocks) {
    return;
  }
  const std::uint32_t t = i / blocks;
  const std::uint32_t b = i % blocks;
  const auto* src =
      reinterpret_cast<const float4*>(x + std::size_t{t} * k + b * 32);
  float v[32];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float4 f = src[j];
    v[4 * j] = f.x;
    v[4 * j + 1] = f.y;
    v[4 * j + 2] = f.z;
    v[4 * j + 3] = f.w;
  }
  float peak = 0.0F;
  float sum = 0.0F;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    peak = fmaxf(peak, fabsf(v[j]));
    sum += v[j];
  }
  int e = 0;
  if (peak > 0.0F) {
    int exponent = 0;
    (void)frexpf(peak, &exponent);
    e = 14 - exponent;
  }
  const float scale = ldexpf(1.0F, e);
  __half h[32];
  __half l[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const float s = v[j] * scale;
    h[j] = __float2half_rn(s);
    l[j] = __float2half_rn(s - __half2float(h[j]));
  }
  auto* dh = reinterpret_cast<uint4*>(hi + std::size_t{t} * k + b * 32);
  auto* dl = reinterpret_cast<uint4*>(lo + std::size_t{t} * k + b * 32);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint4 a;
    uint4 c;
    __builtin_memcpy(&a, &h[8 * j], 16);
    __builtin_memcpy(&c, &l[8 * j], 16);
    dh[j] = a;
    dl[j] = c;
  }
  inv[b * kWmmaGemvRows + t] = ldexpf(1.0F, -e);
  sums[b * kWmmaGemvRows + t] = sum;
}

__device__ __forceinline__ std::uint32_t ByteOf(const uint4& h, int i) {
  const std::uint32_t w[4] = {h.x, h.y, h.z, h.w};
  return (w[i / 4] >> (8 * (i % 4))) & 0xFFU;
}

/// ggml's get_scale_min_k4 for sub-block j of a Q4_K / Q5_K super-block.
__device__ __forceinline__ void ScaleMin(const uint4& h, int j,
                                         std::uint32_t* sc, std::uint32_t* mn) {
  if (j < 4) {
    *sc = ByteOf(h, 4 + j) & 63U;
    *mn = ByteOf(h, 8 + j) & 63U;
  } else {
    *sc = (ByteOf(h, 8 + j) & 15U) | ((ByteOf(h, j) >> 6) << 4);
    *mn = (ByteOf(h, 8 + j) >> 4) | ((ByteOf(h, 4 + j) >> 6) << 4);
  }
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  return __half2float(
      __builtin_bit_cast(__half, static_cast<std::uint16_t>(bits & 0xFFFFU)));
}

/// Four code bytes (each < 1024 - bias) as four exact binary16 values
/// c - bias, through the 0x64 exponent byte (1024 + c).
__device__ __forceinline__ void CodeHalves(std::uint32_t c, __half2 bias,
                                           __half2* out) {
  out[0] =
      __hsub2(__builtin_bit_cast(
                  __half2, __builtin_amdgcn_perm(c, 0x64646464U, 0x01050004U)),
              bias);
  out[1] =
      __hsub2(__builtin_bit_cast(
                  __half2, __builtin_amdgcn_perm(c, 0x64646464U, 0x03070206U)),
              bias);
}

/// 16 code values from four words of byte codes.
__device__ __forceinline__ v16h Fragment(const std::uint32_t (&c)[4],
                                         float bias) {
  const __half2 b = __float2half2_rn(1024.0F + bias);
  __half2 h[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    CodeHalves(c[i], b, &h[2 * i]);
  }
  v16h r;
  __builtin_memcpy(&r, h, 32);
  return r;
}

__device__ __forceinline__ void Words(const uint4& v, std::uint32_t (&w)[4]) {
  w[0] = v.x;
  w[1] = v.y;
  w[2] = v.z;
  w[3] = v.w;
}

/// One weight group of a half super-block: its code fragments (two for 32
/// values, one for 16), the FP32 scale d and the minimum dm (Q4_K, Q5_K).
struct Group {
  v16h frag[2];
  float d;
  float dm;
};

/// Format traits. A super-block holds 256 values (Q4_0: eight blocks); a
/// lane loads its row's super-block as Raw and decodes group g of half n of
/// it exactly as ggml dequantizes, minus the scale and minimum.
template<WmmaFormat F>
struct Fmt;

template<>
struct Fmt<WmmaFormat::kQ4_K> {
  static constexpr int kSuperBytes = 144;
  static constexpr int kGroups = 4;
  static constexpr int kGroupK = 32;
  static constexpr bool kMin = true;
  struct Raw {
    uint4 h;
    uint4 q[8];
  };
  __device__ static Raw Load(const std::uint8_t* sb) {
    const auto* p = reinterpret_cast<const uint4*>(sb);
    Raw r;
    r.h = p[0];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      r.q[i] = p[1 + i];
    }
    return r;
  }
  template<int kN, int kG>
  __device__ static Group Decode(const Raw& r) {
    Group g;
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    ScaleMin(r.h, 4 * kN + kG, &sc, &mn);
    g.d = HalfBits(r.h.x) * static_cast<float>(sc);
    g.dm = HalfBits(r.h.x >> 16) * static_cast<float>(mn);
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      std::uint32_t w[4];
      Words(r.q[4 * kN + 2 * (kG >> 1) + f], w);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        w[i] = (w[i] >> (4 * (kG & 1))) & 0x0F0F0F0FU;
      }
      g.frag[f] = Fragment(w, 0.0F);
    }
    return g;
  }
};

template<>
struct Fmt<WmmaFormat::kQ5_K> {
  static constexpr int kSuperBytes = 176;
  static constexpr int kGroups = 4;
  static constexpr int kGroupK = 32;
  static constexpr bool kMin = true;
  struct Raw {
    uint4 h;
    uint4 qh[2];
    uint4 q[8];
  };
  __device__ static Raw Load(const std::uint8_t* sb) {
    const auto* p = reinterpret_cast<const uint4*>(sb);
    Raw r;
    r.h = p[0];
    r.qh[0] = p[1];
    r.qh[1] = p[2];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      r.q[i] = p[3 + i];
    }
    return r;
  }
  template<int kN, int kG>
  __device__ static Group Decode(const Raw& r) {
    Group g;
    std::uint32_t sc = 0;
    std::uint32_t mn = 0;
    constexpr int kJ = 4 * kN + kG;
    ScaleMin(r.h, kJ, &sc, &mn);
    g.d = HalfBits(r.h.x) * static_cast<float>(sc);
    g.dm = HalfBits(r.h.x >> 16) * static_cast<float>(mn);
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      std::uint32_t w[4];
      std::uint32_t hb[4];
      Words(r.q[4 * kN + 2 * (kG >> 1) + f], w);
      Words(r.qh[f], hb);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        w[i] = ((w[i] >> (4 * (kG & 1))) & 0x0F0F0F0FU) |
               (((hb[i] >> kJ) & 0x01010101U) << 4);
      }
      g.frag[f] = Fragment(w, 0.0F);
    }
    return g;
  }
};

template<>
struct Fmt<WmmaFormat::kQ6_K> {
  static constexpr int kSuperBytes = 210;
  static constexpr int kGroups = 8;
  static constexpr int kGroupK = 16;
  static constexpr bool kMin = false;
  struct Raw {
    uint4 ql[8];
    uint4 qh[4];
    uint4 sc;
    std::uint32_t d;
  };
  // Super-blocks are only 2-byte aligned; gfx1151 serves unaligned loads.
  __device__ static Raw Load(const std::uint8_t* sb) {
    const auto* p = reinterpret_cast<const uint4*>(sb);
    Raw r;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      r.ql[i] = p[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r.qh[i] = p[8 + i];
    }
    r.sc = p[12];
    r.d = *reinterpret_cast<const std::uint16_t*>(sb + 208);
    return r;
  }
  /// Group g of half n covers values 128 n + 16 g..+15: quarter q = g / 2
  /// of ggml's y[l + 32 q], l from 16 (g % 2); its scale is sc[8 n + g].
  template<int kN, int kG>
  __device__ static Group Decode(const Raw& r) {
    constexpr int kQ = kG >> 1;
    constexpr int kH = kG & 1;
    constexpr int kS = 8 * kN + kG;
    Group g;
    std::uint32_t sw[4];
    Words(r.sc, sw);
    const auto scale = static_cast<std::int8_t>(sw[kS / 4] >> (8 * (kS % 4)));
    g.d = HalfBits(r.d) * static_cast<float>(scale);
    g.dm = 0.0F;
    std::uint32_t w[4];
    std::uint32_t hb[4];
    Words(r.ql[4 * kN + 2 * (kQ & 1) + kH], w);
    Words(r.qh[2 * kN + kH], hb);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      w[i] = ((w[i] >> (4 * (kQ >> 1))) & 0x0F0F0F0FU) |
             (((hb[i] >> (2 * kQ)) & 0x03030303U) << 4);
    }
    g.frag[0] = Fragment(w, 32.0F);
    g.frag[1] = g.frag[0];
    return g;
  }
};

template<>
struct Fmt<WmmaFormat::kQ4_0> {
  static constexpr int kSuperBytes = 144;
  static constexpr int kGroups = 4;
  static constexpr int kGroupK = 32;
  static constexpr bool kMin = false;
  /// The super-block's eight 18-byte blocks as words.
  struct Raw {
    std::uint32_t w[37];
  };
  /// Super-blocks are 16-byte aligned (rows are whole super-blocks).
  __device__ static Raw Load(const std::uint8_t* sb) {
    const auto* p = reinterpret_cast<const uint4*>(sb);
    Raw r;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      const uint4 v = p[i];
      r.w[4 * i] = v.x;
      r.w[4 * i + 1] = v.y;
      r.w[4 * i + 2] = v.z;
      r.w[4 * i + 3] = v.w;
    }
    r.w[36] = 0;
    return r;
  }
  /// Block 4 n + g: d at its first two bytes, codes after them (low nibbles
  /// values 0-15, high nibbles 16-31); w = d (q - 8).
  template<int kN, int kG>
  __device__ static Group Decode(const Raw& r) {
    constexpr int kByte = 18 * (4 * kN + kG);
    constexpr int kWord = kByte / 4;
    Group g;
    g.d = HalfBits(kByte % 4 == 0 ? r.w[kWord] : r.w[kWord] >> 16);
    g.dm = 0.0F;
    std::uint32_t c[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c[i] = (kByte + 2) % 4 == 0
                 ? r.w[(kByte + 2) / 4 + i]
                 : __builtin_amdgcn_alignbyte(r.w[(kByte + 2) / 4 + i + 1],
                                              r.w[(kByte + 2) / 4 + i], 2);
    }
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      std::uint32_t w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        w[i] = (c[i] >> (4 * f)) & 0x0F0F0F0FU;
      }
      g.frag[f] = Fragment(w, 8.0F);
    }
    return g;
  }
};

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

/// Packed (rows <= 8): one tile whose rows 0-7 are the hi parts of rows
/// 0-7 and rows 8-15 their lo parts; otherwise a hi tile and a lo tile.
/// Grid: (row blocks, splits); each split covers `per_split` super-blocks
/// and writes partials unless it is the only one.
template<WmmaFormat F, bool kPacked>
__global__ void __launch_bounds__(kThreads)
    GemvKernel(const std::uint8_t* __restrict__ w, Pack pack,
               float* __restrict__ y, std::uint32_t rows, std::uint32_t m,
               std::uint32_t k, std::uint32_t per_split) {
  using Format = Fmt<F>;
  __shared__ __attribute__((aligned(16))) __half s_x[2][16 * kActStride];
  // [32-value block][half][i]: row 2 i + half.
  __shared__ __attribute__((aligned(16))) float s_inv[8][2][8];
  __shared__ __attribute__((aligned(16))) float s_sum[8][2][8];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int sub = lane & 15;
  const int half = lane >> 4;
  const std::size_t row_bytes = std::size_t{k} / kSuper * Format::kSuperBytes;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock + wave * 16 + sub;
  const bool live = row < m;
  const std::uint8_t* w_row = w + std::size_t{live ? row : m - 1} * row_bytes;
  const int sb0 = static_cast<int>(blockIdx.y * per_split);
  const int sb1 =
      min(sb0 + static_cast<int>(per_split), static_cast<int>(k / kSuper));
  const int last = static_cast<int>(rows) - 1;
  const auto super = [&](int sb) {
    return w_row + std::size_t{static_cast<unsigned>(sb)} * Format::kSuperBytes;
  };
  // Activation chunk c = tid + 256 j: tile row c / 32, eight values at
  // 8 (c % 32). Rows past the input repeat its last row (their outputs are
  // never stored).
  const auto act = [&](int j, int sb, bool lo_tile) {
    const int c = tid + j * kThreads;
    const int r = c / 32;
    const int t = min(kPacked ? (r & 7) : r, last);
    const __half* base =
        kPacked ? (r < 8 ? pack.hi : pack.lo) : (lo_tile ? pack.lo : pack.hi);
    return *reinterpret_cast<const uint4*>(
        base + std::size_t{static_cast<unsigned>(t)} * k + sb * kSuper +
        (c % 32) * 8);
  };
  uint4 a00;
  uint4 a01;
  uint4 a10;
  uint4 a11;
  float side = 0.0F;
  const auto fetch_x = [&](int sb) {
    a00 = act(0, sb, false);
    a01 = act(1, sb, false);
    if constexpr (!kPacked) {
      a10 = act(0, sb, true);
      a11 = act(1, sb, true);
    }
    // Threads 0-127 take the inverse scales, 128-255 the sums.
    const int s = (tid & 127) >> 4;
    const int t = min(tid & 15, last);
    const float* src = tid < 128 ? pack.inv : pack.sums;
    side = src[(sb * 8 + s) * kWmmaGemvRows + t];
  };
  const auto commit_x = [&] {
    const int c0 = tid;
    const int c1 = tid + kThreads;
    *reinterpret_cast<uint4*>(&s_x[0][(c0 / 32) * kActStride + (c0 % 32) * 8]) =
        a00;
    *reinterpret_cast<uint4*>(&s_x[0][(c1 / 32) * kActStride + (c1 % 32) * 8]) =
        a01;
    if constexpr (!kPacked) {
      *reinterpret_cast<uint4*>(
          &s_x[1][(c0 / 32) * kActStride + (c0 % 32) * 8]) = a10;
      *reinterpret_cast<uint4*>(
          &s_x[1][(c1 / 32) * kActStride + (c1 % 32) * 8]) = a11;
    }
    const int s = (tid & 127) >> 4;
    const int t = tid & 15;
    (tid < 128 ? s_inv : s_sum)[s][t & 1][t >> 1] = side;
  };

  constexpr int kOut = kPacked ? 4 : 8;
  float total[kOut];
#pragma unroll
  for (int i = 0; i < kOut; ++i) {
    total[i] = 0.0F;
  }
  // Group kG of half kN: S = codes . (hi + lo) over its WMMAs, then the
  // row's scale (and minimum) in FP32.
  const auto group = [&]<int kN, int kG>(const typename Format::Raw& raw) {
    const Group g = Format::template Decode<kN, kG>(raw);
    const float d = live ? g.d : 0.0F;
    const float dm = live ? g.dm : 0.0F;
    constexpr int kOffset = 128 * kN + Format::kGroupK * kG;
    constexpr int kBlock = kOffset / 32;
    constexpr int kFrags = Format::kGroupK / 16;
    v8f hi_chain = {};
    v8f lo_chain = {};
#pragma unroll
    for (int f = 0; f < kFrags; ++f) {
      v16h a;
      __builtin_memcpy(&a, &s_x[0][sub * kActStride + kOffset + 16 * f], 32);
      hi_chain = Wmma(a, g.frag[f], hi_chain);
      if constexpr (!kPacked) {
        v16h l;
        __builtin_memcpy(&l, &s_x[1][sub * kActStride + kOffset + 16 * f], 32);
        lo_chain = Wmma(l, g.frag[f], lo_chain);
      }
    }
    const float4 u0 = *reinterpret_cast<const float4*>(&s_inv[kBlock][half][0]);
    float u[8] = {u0.x, u0.y, u0.z, u0.w, 0.0F, 0.0F, 0.0F, 0.0F};
    float sx[8] = {};
    if constexpr (!kPacked) {
      const float4 u1 =
          *reinterpret_cast<const float4*>(&s_inv[kBlock][half][4]);
      u[4] = u1.x;
      u[5] = u1.y;
      u[6] = u1.z;
      u[7] = u1.w;
    }
    if constexpr (Format::kMin) {
      const float4 x0 =
          *reinterpret_cast<const float4*>(&s_sum[kBlock][half][0]);
      sx[0] = x0.x;
      sx[1] = x0.y;
      sx[2] = x0.z;
      sx[3] = x0.w;
      if constexpr (!kPacked) {
        const float4 x1 =
            *reinterpret_cast<const float4*>(&s_sum[kBlock][half][4]);
        sx[4] = x1.x;
        sx[5] = x1.y;
        sx[6] = x1.z;
        sx[7] = x1.w;
      }
    }
#pragma unroll
    for (int i = 0; i < kOut; ++i) {
      const float hi = hi_chain[i];
      const float lo = kPacked ? hi_chain[i + 4] : lo_chain[i];
      const float v = (hi + lo) * u[i];
      total[i] = __builtin_fmaf(d, v, total[i]);
      if constexpr (Format::kMin) {
        total[i] = __builtin_fmaf(-dm, sx[i], total[i]);
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };
  const auto groups = [&]<int kN>(const typename Format::Raw& raw) {
    [&]<int... kG>(std::integer_sequence<int, kG...>) {
      (group.template operator()<kN, kG>(raw), ...);
    }(std::make_integer_sequence<int, Format::kGroups>{});
  };

  // Blocks walk their super-blocks from a rotation of the block index so
  // that rows a power-of-two-like stride apart do not hit the same memory
  // channels together; each row's order stays fixed. The next super-block
  // loads while this one computes.
  const int span = sb1 - sb0;
  const int rot = span > 0 ? static_cast<int>(blockIdx.x) % span : 0;
  const auto at = [&](int i) { return sb0 + (i + rot) % span; };
  typename Format::Raw next{};
  if (span > 0) {
    next = Format::Load(super(at(0)));
    fetch_x(at(0));
  }
  for (int it = 0; it < span; ++it) {
    const typename Format::Raw cur = next;
    commit_x();
    __syncthreads();
    if (it + 1 < span) {
      next = Format::Load(super(at(it + 1)));
      fetch_x(at(it + 1));
    }
    groups.template operator()<0>(cur);
    groups.template operator()<1>(cur);
    __syncthreads();
  }
  if (!live) {
    return;
  }
  float* out = gridDim.y == 1 ? y : pack.partials;
  const std::size_t base =
      gridDim.y == 1 ? 0 : std::size_t{blockIdx.y} * kWmmaGemvRows * m;
#pragma unroll
  for (int i = 0; i < kOut; ++i) {
    const int t = 2 * i + half;
    if (t < static_cast<int>(rows)) {
      out[base + std::size_t{static_cast<unsigned>(t)} * m + row] = total[i];
    }
  }
}

/// y = the splits' partials added in split order.
__global__ void ReduceKernel(const float* __restrict__ partials,
                             float* __restrict__ y, std::uint32_t rows,
                             std::uint32_t m, std::uint32_t splits) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{rows} * m) {
    return;
  }
  float sum = 0.0F;
  for (std::uint32_t p = 0; p < splits; ++p) {
    sum += partials[std::size_t{p} * kWmmaGemvRows * m + i];
  }
  y[i] = sum;
}

/// Super-blocks per split for the shape alone (never the width): short
/// outputs with long rows spread their reduction over six blocks (cold
/// 5376 x 21504 Q4_K: 360 -> 334 us at one row; shorter rows lose).
std::uint32_t PerSplit(std::uint32_t m, std::uint32_t k) {
  const std::uint32_t supers = k / kSuper;
  const std::uint32_t want = m < kSplitOutputs && k >= 20480 ? kMaxSplits : 1;
  return (supers + want - 1) / want;
}

}  // namespace

namespace {

std::optional<WmmaFormat> FormatOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return WmmaFormat::kQ4_K;
    case core::GgmlType::kQ5_K:
      return WmmaFormat::kQ5_K;
    case core::GgmlType::kQ6_K:
      return WmmaFormat::kQ6_K;
    case core::GgmlType::kQ4_0:
      return WmmaFormat::kQ4_0;
    default:
      return std::nullopt;
  }
}

}  // namespace

bool WmmaGemvSupports(core::GgmlType type, std::uint32_t m,
                      std::uint32_t k) noexcept {
  return FormatOf(type).has_value() && m > 0 && k > 0 && k % kSuper == 0;
}

std::size_t WmmaGemvScratchBytes(std::uint32_t max_k) noexcept {
  const std::size_t halves = std::size_t{kWmmaGemvRows} * max_k;
  const std::size_t blocks = std::size_t{kWmmaGemvRows} * (max_k / 32);
  return 2 * halves * sizeof(__half) + 2 * blocks * sizeof(float) +
         std::size_t{kMaxSplits} * kWmmaGemvRows * kSplitOutputs *
             sizeof(float);
}

bool LaunchWmmaGemv(core::GgmlType type, const void* w, const float* x,
                    float* y, std::uint32_t rows, std::uint32_t m,
                    std::uint32_t k, void* scratch, hipStream_t stream) {
  if (!WmmaGemvSupports(type, m, k) || rows == 0 || rows > kWmmaGemvRows) {
    return false;
  }
  const Pack pack = PackOf(scratch, k);
  const std::uint32_t items = rows * (k / 32);
  PackKernel<<<(items + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      x, rows, k, pack.hi, pack.lo, pack.inv, pack.sums);
  const std::uint32_t per_split = PerSplit(m, k);
  const std::uint32_t splits = (k / kSuper + per_split - 1) / per_split;
  const dim3 grid((m + kRowsPerBlock - 1) / kRowsPerBlock, splits);
  const auto* wb = static_cast<const std::uint8_t*>(w);
  const auto launch = [&]<WmmaFormat F>() {
    if (rows <= 8) {
      GemvKernel<F, true>
          <<<grid, kThreads, 0, stream>>>(wb, pack, y, rows, m, k, per_split);
    } else {
      GemvKernel<F, false>
          <<<grid, kThreads, 0, stream>>>(wb, pack, y, rows, m, k, per_split);
    }
  };
  switch (*FormatOf(type)) {
    case WmmaFormat::kQ4_K:
      launch.template operator()<WmmaFormat::kQ4_K>();
      break;
    case WmmaFormat::kQ5_K:
      launch.template operator()<WmmaFormat::kQ5_K>();
      break;
    case WmmaFormat::kQ6_K:
      launch.template operator()<WmmaFormat::kQ6_K>();
      break;
    case WmmaFormat::kQ4_0:
      launch.template operator()<WmmaFormat::kQ4_0>();
      break;
  }
  if (splits > 1) {
    const std::size_t count = std::size_t{rows} * m;
    ReduceKernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        pack.partials, y, rows, m, splits);
  }
  return true;
}

}  // namespace gufo::models::gemma4::rocm
