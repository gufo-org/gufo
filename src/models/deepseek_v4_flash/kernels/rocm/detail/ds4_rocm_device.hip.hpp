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

/* Q8_0 activation scale from a block's largest magnitude, and one value's
 * code. Both take the maximum itself: the compiler derives the inverse scale
 * as 127 * rcp(amax), and an inverse of the rounded scale differs. */
__device__ __forceinline__ static float q8_0_scale(float amax) {
  return amax / 127.0f;
}

__device__ __forceinline__ static int8_t q8_0_code(float v, float amax) {
  const float d = q8_0_scale(amax);
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  int q = (int)lrintf(v * id);
  q = q > 127 ? 127 : (q < -128 ? -128 : q);
  return (int8_t)q;
}

/* Quantize one Q8_0 activation block held one value per lane of a 32-lane
 * group; lanes at or past `count` contribute zero and store zero. */
__device__ __forceinline__ static void quantize_q8_0_block32(
    float v, uint32_t lane, uint32_t count, int8_t* dst, float* scale) {
  float a = 0.0f;
  if (lane < count)
    a = fabsf(v);
  a = warp_max_f32(a);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  const float amax = __shfl(a, 0, 32);
#else
  const float amax = __shfl_sync(FULL_WARP_MASK, a, 0, 32);
#endif
  if (lane == 0)
    *scale = q8_0_scale(amax);
  dst[lane] = lane < count ? q8_0_code(v, amax) : (int8_t)0;
}

/* The same for a 256-thread block holding a row of N * 256 values, y[k] at
 * k * 256 + tid. One thread takes each block's maximum from shared memory;
 * max is order-independent, so the scales match the lane reduction. */
template <uint32_t N>
__device__ static void quantize_q8_0_cols256(const float (&y)[N], uint32_t tid,
                                             int8_t* dst, float* scale) {
  __shared__ float row[N * 256u];
  __shared__ float block_amax[N * 8u];
#pragma unroll
  for (uint32_t k = 0; k < N; k++) row[k * 256u + tid] = y[k];
  __syncthreads();
  if (tid < N * 8u) {
    float a = fabsf(row[tid * 32u]);
    for (uint32_t j = 1; j < 32u; j++) a = fmaxf(a, fabsf(row[tid * 32u + j]));
    block_amax[tid] = a;
    scale[tid] = q8_0_scale(a);
  }
  __syncthreads();
#pragma unroll
  for (uint32_t k = 0; k < N; k++) {
    const uint32_t i = k * 256u + tid;
    dst[i] = q8_0_code(y[k], block_amax[i / 32u]);
  }
}

/* Q8_K scale from the block's largest-magnitude value, one value's code, and
 * the stored scale. */
__device__ __forceinline__ static float q8_K_iscale(float maxv) {
  return -127.0f / maxv;
}

__device__ __forceinline__ static int8_t q8_K_code(float v, float iscale) {
  int qv = (int)lrintf(iscale * v);
  if (qv > 127) qv = 127;
  if (qv < -128) qv = -128;
  return (int8_t)qv;
}

/* Quantize one Q8_K block held one value per thread of a 256-thread block.
 * The tree's tie order picks the sign of the scale. Every thread of the block
 * calls it; the scratch arrays are shared. */
__device__ static void quantize_q8_K_block256(float v, uint32_t tid,
                                              hip_block_q8_K* yb,
                                              float* abs_part, float* val_part,
                                              float* iscale_s) {
  abs_part[tid] = tid < ROCM_QK_K ? fabsf(v) : 0.0f;
  val_part[tid] = v;
  __syncthreads();
  for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
      abs_part[tid] = abs_part[tid + stride];
      val_part[tid] = val_part[tid + stride];
    }
    __syncthreads();
  }
  const float amax = abs_part[0];
  if (amax == 0.0f) {
    if (tid == 0) yb->d = 0.0f;
    if (tid < ROCM_QK_K) yb->qs[tid] = 0;
    if (tid < ROCM_QK_K / 16) yb->bsums[tid] = 0;
    return;
  }
  if (tid == 0) *iscale_s = q8_K_iscale(val_part[0]);
  __syncthreads();
  if (tid < ROCM_QK_K) yb->qs[tid] = q8_K_code(v, *iscale_s);
  __syncthreads();
  if (tid < ROCM_QK_K / 16) {
    int sum = 0;
    for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
    yb->bsums[tid] = (int16_t)sum;
  }
  if (tid == 0) yb->d = 1.0f / *iscale_s;
}

/* N Q8_K blocks of a row held as in quantize_q8_0_cols256 (block k is y[k] of
 * all 256 threads), each through the per-block tree above with every step's
 * (block, slot) pairs spread over the whole thread block. */
template <uint32_t N>
__device__ static void quantize_q8_K_cols256(const float (&y)[N], uint32_t tid, hip_block_q8_K* yb) {
  __shared__ float abs_part[N * 256u];
  __shared__ float val_part[N * 256u];
  __shared__ float iscale_s[N];
  __shared__ int8_t codes[N * 256u];
#pragma unroll
  for (uint32_t k = 0; k < N; k++) {
    abs_part[k * 256u + tid] = fabsf(y[k]);
    val_part[k * 256u + tid] = y[k];
  }
  __syncthreads();
  for (uint32_t stride = 128u; stride > 0; stride >>= 1) {
    for (uint32_t j = tid; j < N * stride; j += blockDim.x) {
      const uint32_t a = (j / stride) * 256u + j % stride;
      if (abs_part[a + stride] > abs_part[a]) {
        abs_part[a] = abs_part[a + stride];
        val_part[a] = val_part[a + stride];
      }
    }
    __syncthreads();
  }
  if (tid < N) iscale_s[tid] = q8_K_iscale(val_part[tid * 256u]);
  __syncthreads();
#pragma unroll
  for (uint32_t k = 0; k < N; k++) {
    const int8_t q = abs_part[k * 256u] == 0.0f ? (int8_t)0 : q8_K_code(y[k], iscale_s[k]);
    codes[k * 256u + tid] = q;
    yb[k].qs[tid] = q;
  }
  __syncthreads();
  for (uint32_t j = tid; j < N * 16u; j += blockDim.x) {
    int sum = 0;
    for (int i = 0; i < 16; i++) sum += codes[(j / 16u) * 256u + (j % 16u) * 16u + i];
    yb[j / 16u].bsums[j % 16u] = (int16_t)sum;
  }
  if (tid < N) yb[tid].d = abs_part[tid * 256u] == 0.0f ? 0.0f : 1.0f / iscale_s[tid];
}
