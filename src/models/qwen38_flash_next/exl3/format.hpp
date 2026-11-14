// EXL3 trellis format primitives shared by the host reference and the HIP
// kernels: the mul1 codebook, the per-lane tile window and the n-gram row
// codec. Adapted from Crucible (engine/exl3/kernels/lane_window.h,
// engine/exl3/kernels/exl3_gemv.h, engine/layers/ple_row_codec.h),
// Copyright 2026 jaedync, Apache-2.0. The format is ExLlamaV3's,
// Copyright (c) 2025 Turboderp, MIT. See THIRD_PARTY_NOTICES.md.
#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_FORMAT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_FORMAT_HPP_

#include <cstdint>

#if defined(__HIPCC__)
#define GUFO_EXL3_HD __host__ __device__ __attribute__((always_inline))
#define GUFO_EXL3_UNROLL _Pragma("unroll")
#else
#define GUFO_EXL3_HD __attribute__((always_inline))
#define GUFO_EXL3_UNROLL
#endif

namespace gufo::models::qwen38_flash_next::exl3 {

/// Presence marker of the mul1 codebook (`<name>.mul1`, I32 scalar).
inline constexpr std::uint32_t kMul1 = 0x83DCD12DU;
/// Output and input transforms are block-diagonal 128-point Hadamards.
inline constexpr std::uint32_t kHadamardSize = 128;
/// 1 / sqrt(128), the orthonormal Hadamard scale.
inline constexpr float kHadamardNorm = 0.088388347648F;
/// Supported integer bit widths of linear tiles.
inline constexpr int kMinTileK = 2;
inline constexpr int kMaxTileK = 6;
/// The n-gram table row width.
inline constexpr std::uint32_t kNgramRowDim = 160;

/// Dwords of one 16x16 tile: 256 weights of K bits.
[[nodiscard]] constexpr int TileDwords(int k) noexcept {
  return 8 * k;
}

/// uint16 words of one n-gram row: an fp16 scale and 160 K-bit symbols.
[[nodiscard]] constexpr std::uint32_t NgramRowWords(int k) noexcept {
  return 1 + kNgramRowDim * static_cast<std::uint32_t>(k) / 16;
}

/// Bits of the fp16 codebook value of a 16-bit trellis state:
/// fma(1024 + bytesum(state * kMul1), 0x1eee, 0xc931) with one fp16 rounding.
/// The returned value is the fp16 pattern; `Mul1Float` widens it.
GUFO_EXL3_HD inline std::uint32_t Mul1ByteSum(std::uint32_t state) {
  const std::uint32_t p = (state & 0xFFFFU) * kMul1;
  return (p & 255U) + ((p >> 8) & 255U) + ((p >> 16) & 255U) + (p >> 24);
}

/// Host-side fp16 helpers (the device uses native half arithmetic).
[[nodiscard]] inline float HalfToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
  std::uint32_t exp = (h >> 10) & 0x1FU;
  std::uint32_t man = h & 0x3FFU;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400U) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFU;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f = 0.0F;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

/// Round-to-nearest-even float to fp16 bits.
[[nodiscard]] inline std::uint16_t FloatToHalf(float f) noexcept {
  std::uint32_t x = 0;
  __builtin_memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000U;
  const std::uint32_t abs = x & 0x7FFFFFFFU;
  if (abs >= 0x7F800000U) {
    return static_cast<std::uint16_t>(sign | 0x7C00U |
                                      (abs > 0x7F800000U ? 0x200U : 0U));
  }
  if (abs >= 0x477FF000U) {  // rounds to infinity
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (abs < 0x38800000U) {  // subnormal or zero
    if (abs < 0x33000000U) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t e = abs >> 23;
    const std::uint32_t m = (abs & 0x7FFFFFU) | 0x800000U;
    const std::uint32_t shift = 126 - e;  // 14 + (113 - e)
    std::uint32_t h = m >> shift;
    const std::uint32_t rem = m & ((1U << shift) - 1U);
    const std::uint32_t half = 1U << (shift - 1);
    if (rem > half || (rem == half && (h & 1U))) {
      ++h;
    }
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint32_t h = ((abs >> 13) - (112U << 10));
  const std::uint32_t rem = abs & 0x1FFFU;
  if (rem > 0x1000U || (rem == 0x1000U && (h & 1U))) {
    ++h;
  }
  return static_cast<std::uint16_t>(sign | h);
}

/// Exact codebook value of a state as float (host reference). The fp16 fma
/// of two fp16 operands and an fp16 addend is exact in double, then rounded
/// once to fp16.
[[nodiscard]] inline float Mul1Value(std::uint32_t state) noexcept {
  const double h = 1024.0 + static_cast<double>(Mul1ByteSum(state));
  const double inv = static_cast<double>(HalfToFloat(0x1EEE));
  const double bias = static_cast<double>(HalfToFloat(0xC931));
  return HalfToFloat(FloatToHalf(static_cast<float>(h * inv + bias)));
}

/// v_alignbit_b32: the low 32 bits of (hi:lo) >> (s mod 32).
GUFO_EXL3_HD inline std::uint32_t AlignBit(std::uint32_t hi, std::uint32_t lo,
                                           std::uint32_t s) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_alignbit(hi, lo, s);
#else
  return static_cast<std::uint32_t>(
      ((static_cast<std::uint64_t>(hi) << 32) | lo) >> (s & 31U));
#endif
}

/// Lane-invariant geometry of a lane's 96-bit window inside a tile ring.
/// Lane t owns ring positions 8t..8t+7; position i's 16-bit state ends at
/// the end of symbol i, so the eight states are one 7K+16-bit span.
template<int K>
struct LaneWindow {
  static_assert(K >= 1 && K <= 6);
  int d0;     ///< first dword (ring index) of the 3-dword window
  int shift;  ///< 1..32: the lane's first state bit is bit `shift` of w0:w1
  GUFO_EXL3_HD explicit LaneWindow(int t) {
    const int s = (8 * t + 1) * K - 16;
    const int dw = (s - 1 >= 0) ? (s - 1) / 32 : -((32 - s) / 32);
    d0 = ((dw % (8 * K)) + 8 * K) % (8 * K);
    shift = s - dw * 32;
  }
};

/// The lane's 8 trellis states (low 16 bits of st[j]; higher bits unmasked).
/// The stream is big-endian within each dword: bit 0 is the dword's MSB.
template<int K>
GUFO_EXL3_HD inline void LaneStates(std::uint32_t w0, std::uint32_t w1,
                                    std::uint32_t w2, int shift,
                                    std::uint32_t st[8]) {
  const std::uint32_t a0 = AlignBit(w0, w1, 32 - shift);
  const std::uint32_t a1 = AlignBit(w1, w2, 32 - shift);
  GUFO_EXL3_UNROLL
  for (int j = 0; j < 8; ++j) {
    const int end = j * K + 16;
    st[j] = end <= 32 ? a0 >> (32 - end) : AlignBit(a0, a1, 64 - end);
  }
}

/// Element (row, col) of a 16x16 tile that ring position i reconstructs.
/// Lane t = i / 8 owns rows {r0, r0+1, r0+8, r0+9} of columns {c0, c0+8}
/// with r0 = 2 (t % 4) and c0 = t / 4 (NVIDIA tensor-core fragment order).
GUFO_EXL3_HD inline void TilePosition(int i, int* row, int* col) {
  const int t = i >> 3;
  const int j = i & 7;
  const int r0 = 2 * (t % 4);
  const int c0 = t / 4;
  *row = r0 + (j & 1) + ((j & 2) != 0 ? 8 : 0);
  *col = c0 + ((j & 4) != 0 ? 8 : 0);
}

/// Symbol jj (mod 160) of an n-gram row, LSB-first after the scale word.
template<int K>
GUFO_EXL3_HD inline std::uint32_t NgramSymbol(const std::uint16_t* row,
                                              int jj) {
  constexpr int kWords = 10 * K;
  const int j =
      (jj + static_cast<int>(kNgramRowDim)) % static_cast<int>(kNgramRowDim);
  const int bit = j * K;
  const int w = bit >> 4;
  const int sh = bit & 15;
  std::uint32_t v = row[1 + w];
  if (w + 1 < kWords) {
    v |= static_cast<std::uint32_t>(row[2 + w]) << 16;
  }
  return (v >> sh) & ((1U << K) - 1U);
}

/// The 16-bit state of n-gram position i: its symbol, then the preceding
/// symbols (mod 160), K bits each, truncated to 16 bits.
template<int K>
GUFO_EXL3_HD inline std::uint32_t NgramState(const std::uint16_t* row, int i) {
  std::uint32_t state = 0;
  int filled = 0;
  for (int back = 0; filled < 16; ++back) {
    state |= NgramSymbol<K>(row, i - back) << filled;
    filled += K;
  }
  return state & 0xFFFFU;
}

}  // namespace gufo::models::qwen38_flash_next::exl3

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_EXL3_FORMAT_HPP_
