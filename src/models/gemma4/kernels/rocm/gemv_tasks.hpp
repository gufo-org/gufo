// Device helpers shared by the Gemma 4 decode GEMVs (gemv.hip.cpp) and the
// routed expert GEMVs (moe.hip.cpp): one lane's unit of quantized weights
// dequantized once and contracted with FP32 activations in a fixed FMA
// order. Include only from translation units compiled with
// -fno-fast-math -ffp-contract=off.
#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_TASKS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_TASKS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::models::gemma4::rocm::gemv_tasks {

__device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  const auto h = static_cast<unsigned short>(bits & 0xFFFFU);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

/// ggml get_scale_min_k4 over the 12 scale bytes held as three words
/// (s[0..3], s[4..7], s[8..11]): 6-bit scale and min of 32-value sub-block
/// j, extracted with shifts instead of byte indexing.
__device__ __forceinline__ void ScaleMin(std::uint32_t w0, std::uint32_t w1,
                                         std::uint32_t w2, int j, float* sc,
                                         float* m) {
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w0 >> shift) & 0xFFU;
  const std::uint32_t b = (w1 >> shift) & 0xFFU;
  const std::uint32_t c = (w2 >> shift) & 0xFFU;
  const bool low = j < 4;
  *sc = static_cast<float>(low ? (a & 63U) : ((c & 0xFU) | ((a >> 6) << 4)));
  *m = static_cast<float>(low ? (b & 63U) : ((c >> 4) | ((b >> 6) << 4)));
}

/// Dequantizes 16 values packed per byte (optional high bits) as
/// scale * v + offset.
__device__ __forceinline__ void Dequant16(const std::uint8_t* q, int shift,
                                          const std::uint8_t* high, int hshift,
                                          int hmask, float scale, float offset,
                                          float* out) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int v = (q[i] >> shift) & 0xF;
    if (high != nullptr) {
      v |= ((high[i] >> hshift) & hmask) << 4;
    }
    out[i] = __builtin_fmaf(scale, static_cast<float>(v), offset);
  }
}

/// acc += sum_i w[i] * x[i] as one FMA chain over n 64-byte-aligned values.
template<int N>
__device__ __forceinline__ float Accumulate(float acc, const float* w,
                                            const float* x) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 f = *reinterpret_cast<const float4*>(x + i);
    acc = __builtin_fmaf(w[i], f.x, acc);
    acc = __builtin_fmaf(w[i + 1], f.y, acc);
    acc = __builtin_fmaf(w[i + 2], f.z, acc);
    acc = __builtin_fmaf(w[i + 3], f.w, acc);
  }
  return acc;
}

// Q4_K: 8 tasks per 256-value super-block. Task t covers qs bytes
// [32 p + 16 h, +16) with p = t / 2, h = t % 2: low nibbles are values
// 64 p + 16 h + i of sub-block 2 p, high nibbles 64 p + 32 + 16 h + i of
// sub-block 2 p + 1. Q5_K adds the fifth bit of value 16 h + i of pair p
// from bit 2 p (low) or 2 p + 1 (high) of qh[16 h + i]. The 16-byte header
// (d, dmin, 12 scale bytes) is one aligned load. w[16 r + i] belongs to
// activation 64 p + 16 h + 32 r + i.
template<bool kFiveBit>
__device__ __forceinline__ void DecodeQ45K(const std::uint8_t* block, int t,
                                           float (&w)[32]) {
  const int p = t >> 1;
  const int h = t & 1;
  const uint4 header = *reinterpret_cast<const uint4*>(block);
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * p, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * p + 1, &sc1, &m1);
  const int qs = kFiveBit ? 48 : 16;
  const uint4 raw_q =
      *reinterpret_cast<const uint4*>(block + qs + 32 * p + 16 * h);
  const auto* q = reinterpret_cast<const std::uint8_t*>(&raw_q);
  uint4 raw_h{};
  const std::uint8_t* qh = nullptr;
  if constexpr (kFiveBit) {
    raw_h = *reinterpret_cast<const uint4*>(block + 16 + 16 * h);
    qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
  }
  Dequant16(q, 0, qh, 2 * p, 1, d * sc0, -(dmin * m0), w);
  Dequant16(q, 4, qh, 2 * p + 1, 1, d * sc1, -(dmin * m1), w + 16);
}

/// Q4_K/Q5_K sub-block pair p (64 values, one header load): w[i] belongs to
/// activation 64 p + i -- the low nibbles of qs bytes [32 p, +32) are
/// sub-block 2 p, the high nibbles sub-block 2 p + 1.
template<bool kFiveBit>
__device__ __forceinline__ void DecodeQ45KPair(const std::uint8_t* block, int p,
                                               float (&w)[64]) {
  const uint4 header = *reinterpret_cast<const uint4*>(block);
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * p, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * p + 1, &sc1, &m1);
  const int qs = kFiveBit ? 48 : 16;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint4 raw_q =
        *reinterpret_cast<const uint4*>(block + qs + 32 * p + 16 * h);
    const auto* q = reinterpret_cast<const std::uint8_t*>(&raw_q);
    uint4 raw_h{};
    const std::uint8_t* qh = nullptr;
    if constexpr (kFiveBit) {
      raw_h = *reinterpret_cast<const uint4*>(block + 16 + 16 * h);
      qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
    }
    Dequant16(q, 0, qh, 2 * p, 1, d * sc0, -(dmin * m0), w + 16 * h);
    Dequant16(q, 4, qh, 2 * p + 1, 1, d * sc1, -(dmin * m1), w + 32 + 16 * h);
  }
}

/// Activation offset of task t's first 16 values in a Q4_K/Q5_K
/// super-block; the other 16 follow 32 values later.
__device__ __forceinline__ int OffsetQ45K(int t) {
  return 64 * (t >> 1) + 16 * (t & 1);
}

template<bool kFiveBit>
__device__ __forceinline__ float TaskQ45K(const std::uint8_t* block, int t,
                                          const float* x, float acc) {
  float w[32];
  DecodeQ45K<kFiveBit>(block, t, w);
  const float* xv = x + OffsetQ45K(t);
  acc = Accumulate<16>(acc, w, xv);
  return Accumulate<16>(acc, w + 16, xv + 32);
}

// Q6_K: 4 tasks per super-block. Task t = 2 n + h covers ql bytes
// [64 n + 16 h, +16) and [64 n + 32 + 16 h, +16) and qh bytes
// [32 n + 16 h, +16): values 128 n + 16 h + i + {0, 32, 64, 96} with signed
// scales sc[8 n + h + {0, 2, 4, 6}] and an offset of 32. w[16 r + i]
// belongs to activation 128 n + 16 h + 32 r + i.
__device__ __forceinline__ void DecodeQ6K(const std::uint8_t* block, int t,
                                          float (&w)[64]) {
  const int n = t >> 1;
  const int h = t & 1;
  const float d = Half(block + 208);
  const auto* sc =
      reinterpret_cast<const std::int8_t*>(block + 192) + 8 * n + h;
  const uint4 raw_a = *reinterpret_cast<const uint4*>(block + 64 * n + 16 * h);
  const uint4 raw_b =
      *reinterpret_cast<const uint4*>(block + 64 * n + 32 + 16 * h);
  const uint4 raw_h =
      *reinterpret_cast<const uint4*>(block + 128 + 32 * n + 16 * h);
  const auto* ql_a = reinterpret_cast<const std::uint8_t*>(&raw_a);
  const auto* ql_b = reinterpret_cast<const std::uint8_t*>(&raw_b);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(&raw_h);
  const float s0 = d * static_cast<float>(sc[0]);
  const float s1 = d * static_cast<float>(sc[2]);
  const float s2 = d * static_cast<float>(sc[4]);
  const float s3 = d * static_cast<float>(sc[6]);
  // (q - 32) * s = s * q - 32 s.
  Dequant16(ql_a, 0, qh, 0, 3, s0, -32.0F * s0, w);
  Dequant16(ql_b, 0, qh, 2, 3, s1, -32.0F * s1, w + 16);
  Dequant16(ql_a, 4, qh, 4, 3, s2, -32.0F * s2, w + 32);
  Dequant16(ql_b, 4, qh, 6, 3, s3, -32.0F * s3, w + 48);
}

__device__ __forceinline__ int OffsetQ6K(int t) {
  return 128 * (t >> 1) + 16 * (t & 1);
}

__device__ __forceinline__ float TaskQ6K(const std::uint8_t* block, int t,
                                         const float* x, float acc) {
  float w[64];
  DecodeQ6K(block, t, w);
  const float* xv = x + OffsetQ6K(t);
  acc = Accumulate<16>(acc, w, xv);
  acc = Accumulate<16>(acc, w + 16, xv + 32);
  acc = Accumulate<16>(acc, w + 32, xv + 64);
  return Accumulate<16>(acc, w + 48, xv + 96);
}

/// Q8_0 block (34 bytes, 2-byte aligned inside word-aligned rows): values
/// d * q. `odd` says the block starts two bytes past a word boundary; the
/// nine enclosing words are loaded and the payload selected with byte
/// alignment.
__device__ __forceinline__ void DecodeQ8_0(const std::uint8_t* block, bool odd,
                                           float (&w)[32]) {
  const auto* words =
      reinterpret_cast<const std::uint32_t*>(block - (odd ? 2 : 0));
  std::uint32_t r[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    r[i] = words[i];
  }
  const float d = HalfBits(odd ? r[0] >> 16 : r[0]);
  std::uint32_t qs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    qs[i] = odd ? r[i + 1] : __builtin_amdgcn_alignbyte(r[i + 1], r[i], 2);
  }
  const auto* q = reinterpret_cast<const std::int8_t*>(qs);
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    w[i] = d * static_cast<float>(q[i]);
  }
}

/// Q5_1 block (24 bytes, word aligned): d, m, 32 fifth bits, 16 nibble
/// bytes; value i is d * q + m, low nibbles first (fifth bits 0-15), high
/// nibbles second (bits 16-31).
__device__ __forceinline__ void DecodeQ5_1(const std::uint8_t* block,
                                           float (&w)[32]) {
  const auto* words = reinterpret_cast<const std::uint32_t*>(block);
  const std::uint32_t dm = words[0];
  const std::uint32_t qh = words[1];
  std::uint32_t qs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qs[i] = words[2 + i];
  }
  const float d = HalfBits(dm);
  const float m = HalfBits(dm >> 16);
  const auto* q = reinterpret_cast<const std::uint8_t*>(qs);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int lo = (q[i] & 0xF) | static_cast<int>(((qh >> i) & 1U) << 4);
    const int hi = (q[i] >> 4) | static_cast<int>(((qh >> (i + 16)) & 1U) << 4);
    w[i] = __builtin_fmaf(d, static_cast<float>(lo), m);
    w[i + 16] = __builtin_fmaf(d, static_cast<float>(hi), m);
  }
}

// Q4_0: eight 32-value blocks form a 256-value, 144-byte group, and task t
// is block t: values 32 t + i (low nibbles) and 32 t + 16 + i (high) with
// (q - 8) d = d q - 8 d. Blocks are 18 bytes, so only every other block is
// word aligned; each lane loads the five words enclosing its block and
// selects the payload with byte alignment.
/// Q8_0 task t of a 256-value group (eight 34-byte blocks; the group is word
/// aligned, odd blocks start mid-word): block t, ggml's d * q, one FMA chain.
__device__ __forceinline__ float TaskQ80(const std::uint8_t* group, int t,
                                         const float* x, float acc) {
  float w[32];
  DecodeQ8_0(group + 34 * t, (t & 1) != 0, w);
  return Accumulate<32>(acc, w, x + 32 * t);
}

__device__ __forceinline__ float TaskQ40(const std::uint8_t* group, int t,
                                         const float* x, float acc) {
  const std::uint8_t* block = group + 18 * t;
  const bool odd = (t & 1) != 0;
  const auto* words =
      reinterpret_cast<const std::uint32_t*>(block - (odd ? 2 : 0));
  std::uint32_t r[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    r[i] = words[i];
  }
  const float d = HalfBits(odd ? r[0] >> 16 : r[0]);
  std::uint32_t qs[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qs[i] = odd ? r[i + 1] : __builtin_amdgcn_alignbyte(r[i + 1], r[i], 2);
  }
  const auto* q = reinterpret_cast<const std::uint8_t*>(qs);
  float w[32];
  Dequant16(q, 0, nullptr, 0, 0, d, -8.0F * d, w);
  Dequant16(q, 4, nullptr, 0, 0, d, -8.0F * d, w + 16);
  const float* xv = x + 32 * t;
  acc = Accumulate<16>(acc, w, xv);
  return Accumulate<16>(acc, w + 16, xv + 16);
}

}  // namespace gufo::models::gemma4::rocm::gemv_tasks

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_GEMV_TASKS_HPP_
