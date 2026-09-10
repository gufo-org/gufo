#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#define ROCM_QK_K 256

static __device__ __forceinline__ int32_t __vcmpne4(uint32_t a, uint32_t b) {
  // For each byte: 0xFF if a != b, 0x00 if a == b
  uint32_t diff = a ^ b;
  // Spread any set bit in each byte to fill the whole byte
  diff |= (diff >> 1);
  diff |= (diff >> 2);
  diff |= (diff >> 4);
  diff &= 0x01010101u;
  diff *= 0xFFu;  // 0x01 -> 0xFF per byte
  return (int32_t)diff;
}

static __device__ __forceinline__ int32_t __vsub4(int32_t a, int32_t b) {
  // Per-byte subtraction (wrapping, not saturating)
  uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
  // Trick: subtract bytes in parallel avoiding cross-byte borrows
  uint32_t diff =
      ((ua | 0x80808080u) - (ub & 0x7F7F7F7Fu)) ^ ((ua ^ ~ub) & 0x80808080u);
  return (int32_t)diff;
}

// __dp4a: dot product of 4 signed int8s packed in an int32.
// gfx11-class AMD GPUs expose this as a single v_dot4_i32_i8 instruction;
// using the clang builtin avoids expanding every Q8/Q8_K dot into scalar byte
// multiplies in the resident ROCm implementation.
static __device__ __forceinline__ int32_t __dp4a(int32_t a, int32_t b,
                                                 int32_t c) {
  union ds4_i8x4_bits {
    int32_t i;
    char4 v;
  } av, bv;
  av.i = a;
  bv.i = b;
  return amd_mixed_dot(av.v, bv.v, c, false);
}

typedef struct {
  uint8_t scales[ROCM_QK_K / 16];
  uint8_t qs[ROCM_QK_K / 4];
  uint16_t d;
  uint16_t dmin;
} hip_block_q2_K;

typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[ROCM_QK_K / 2];
} hip_block_q4_K;

typedef struct {
  float d;
  int8_t qs[ROCM_QK_K];
  int16_t bsums[ROCM_QK_K / 16];
} hip_block_q8_K;

typedef struct {
  uint16_t d;
  uint16_t qs[ROCM_QK_K / 8];
} hip_block_iq2_xxs;

__device__ static float warp_sum_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v += __shfl_down(v, offset, 32);
#else
    v += __shfl_down_sync(FULL_WARP_MASK, v, offset, 32);
#endif
  }
  return v;
}

__device__ static float warp_max_f32(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    v = fmaxf(v, __shfl_down(v, offset, 32));
#else
    v = fmaxf(v, __shfl_down_sync(FULL_WARP_MASK, v, offset, 32));
#endif
  }
  return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
  uint32_t mask = 0xffu << (threadIdx.x & 24u);
  for (int offset = 4; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(static_cast<uint64_t>(mask), v, offset, 8);
  }
  (void)lane8;
  return v;
}
