#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_DEVICE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_DEVICE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/trellis.hpp"

/// Device helpers for the TQ2_T kernels (format notes in trellis.hpp). HIP
/// sources only.
namespace gufo::models::qwen38_flash_next::rocm {

namespace trellis {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));

/// Copies the codebook (2048 packed F16 pairs) into LDS; callers sync.
__device__ __forceinline__ void LoadCodebook(std::uint32_t* lds,
                                             const std::uint32_t* codebook,
                                             int tid, int threads) {
  for (int i = tid; i < kTrellisCodebookPoints / 4; i += threads) {
    reinterpret_cast<uint4*>(lds)[i] =
        reinterpret_cast<const uint4*>(codebook)[i];
  }
}

/// States of the four steps whose code bytes are `w` (byte j = step j), with
/// `prev` the code byte before the first of them.
__device__ __forceinline__ void LaneStates(std::uint32_t w, std::uint32_t prev,
                                           std::uint32_t s[4]) {
  // Byte j of `lead` is the byte before step j: the high byte of its state.
  const std::uint32_t lead = (w << 8) | (prev & 0xFFU);
  // v_perm interleaves {lead byte j, w byte j} into each 16-bit state.
  const std::uint32_t lo = __builtin_amdgcn_perm(lead, w, 0x05010400U);
  const std::uint32_t hi = __builtin_amdgcn_perm(lead, w, 0x07030602U);
  s[0] = lo & 0xFFFFU;
  s[1] = lo >> 16;
  s[2] = hi & 0xFFFFU;
  s[3] = hi >> 16;
}

/// The two codebook indices of one step.
__device__ __forceinline__ void StepIndices(std::uint32_t s, std::uint32_t& a,
                                            std::uint32_t& b) {
  const std::uint32_t x = (s & 0x7FFFU) * 0x9E3779B1U;
  a = x >> 21;
  b = (x >> 10) & 2047U;
}

/// The code byte before this lane's four (lane l of an 8-lane group holds
/// bytes 4l..4l+3 of one block): byte 3 of lane (l - 1) mod 8's word.
__device__ __forceinline__ std::uint32_t PrevCodeByte(std::uint32_t w) {
  // DPP8 selector: lane i reads lane (i + 7) % 8 of its group.
  constexpr unsigned kRotate = 7U | (0U << 3) | (1U << 6) | (2U << 9) |
                               (3U << 12) | (4U << 15) | (5U << 18) |
                               (6U << 21);
  return static_cast<std::uint32_t>(
             __builtin_amdgcn_mov_dpp8(static_cast<int>(w), kRotate)) >>
         24;
}

/// T·v over one 128-element block held as four consecutive elements per
/// lane of a wave32 (lane l holds 4l..4l+3), signs already applied: the
/// normalized Walsh-Hadamard butterflies of tq_ref.c's fwht_blocks.
__device__ __forceinline__ void Fwht128(float v[4]) {
  float a = v[0] + v[1];
  float b = v[0] - v[1];
  float c = v[2] + v[3];
  float d = v[2] - v[3];
  v[0] = a + c;
  v[1] = b + d;
  v[2] = a - c;
  v[3] = b - d;
  const int lane = static_cast<int>(threadIdx.x & 31U);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const bool upper = (lane & offset) != 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float other = __shfl_xor(v[j], offset, 32);
      v[j] = upper ? other - v[j] : v[j] + other;
    }
  }
  constexpr float kScale = 0.088388347648318440F;  // 1 / sqrt(128)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] *= kScale;
  }
}

}  // namespace trellis
}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_DEVICE_HPP_
