#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_WMMA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_WMMA_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"

/// WMMA tensor-core routed tier for the low-bit IQ expert stacks
/// (glm5-next wide-chunk MoE). One launch dots the 32-slot expert tiles of
/// the stacked weights against an F16 activation copy: gate/up for IQ1_M or
/// IQ2_XXS computes both stacks and fuses the swiglu-clamp epilogue, down
/// for IQ3_XXS stores the raw dot per slot. Each launch replaces two Gemv +
/// SwigluClamp (or one Gemv) round-trips of the scalar bucketed tier.
/// Accumulation runs in f16 fragments over wmma 16x16x16, so outputs follow
/// the tensor-core summation order, not the scalar per-slot dot; the quality
/// contract is the NMSE and bitwise-determinism test, not bit-equality with
/// the scalar tier.
namespace gufo::models::glm5_next::rocm {

/// Fused gate/up + swiglu-clamp over a routed bucket tile map of 32-row
/// tiles: for tile entry `expert | tile << 16` (0x7FFF dead entries are
/// skipped) and bucket rows [tile*32, tile*32 + 32) of expert `expert`,
/// slot = rows_slot[start + t] (slot >= 0), token = slot / slots_per_row,
///   g = min(dot(gate[e][r], x[token]), clamp)
///   u = clamp(dot(up[e][r], x[token]), -clamp, clamp)
///   out[slot * rows + r] = g * sigmoid(g) * u
/// `x` holds the F16 activations (ConvertF32ToHalf) with `x_stride` halves
/// per token; `type` is kIQ1_M or kIQ2_XXS and both weight stacks use that
/// super-block layout, with `expert_stride` = row_bytes * rows bytes between
/// stacked experts and `stack_bytes` the total size of one stack (bounds
/// the ceil-dword block read of the final block).
/// grid = (rows / 128, capacity), asynchronous on `stream`.
void RoutedGateUpWmma(GemvType type, const void* gate, const void* up,
                      std::size_t expert_stride, std::size_t stack_bytes,
                      const __half* x, std::uint32_t x_stride,
                      const std::int32_t* tiles, std::uint32_t capacity,
                      const std::int32_t* pad_bounds,
                      const std::int32_t* rows_slot,
                      std::uint32_t slots_per_row, std::uint32_t rows,
                      std::uint32_t cols, float clamp, float* out,
                      hipStream_t stream);

/// Down projection over the same 32-row tile map with one activation row
/// per slot (slot = rows_slot[start + t], x row = slot):
///   out[slot * rows + r] = dot(down[e][r], x[slot])
/// `type` is kIQ3_XXS; `kBucketMinRows` applies. Same grid and stream
/// semantics as RoutedGateUpWmma.
void RoutedDownWmma(GemvType type, const void* down, std::size_t expert_stride,
                    std::size_t stack_bytes, const __half* x,
                    std::uint32_t x_stride, const std::int32_t* tiles,
                    std::uint32_t capacity, const std::int32_t* pad_bounds,
                    const std::int32_t* rows_slot, std::uint32_t rows,
                    std::uint32_t cols, float* out, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_WMMA_HPP_
