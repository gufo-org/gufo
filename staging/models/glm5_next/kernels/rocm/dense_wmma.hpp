#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DENSE_WMMA_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DENSE_WMMA_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "staging/models/glm5_next/kernels/rocm/gemv.hpp"

namespace gufo::models::glm5_next::rocm {

/// True when `type` has a WMMA dense tier (Q5_K projections today).
bool DenseWmmaSupports(GemvType type);

/// out[t*out_stride + r] = sum_k W[r][k] * W float-conversion tier of the
/// wide-chunk dispatch: decodes super-block rows into F16 wmma fragments
/// once per 32-token tile and runs `wmma_f32_16x16x16_f16_w32`, mirroring
/// RoutedWmma's staging geometry for a contiguous (non-stacked) row-major
/// [rows x cols] matrix. Activations are F32 and converted inside the
/// kernel's staged read. Equivalent to GemvRows up to tensor-core
/// accumulation order (contract: rel. NMSE <= 1e-4, finite, deterministic).
/// Requires cols % 256 == 0 and 16 <= tokens <= Executor::kChunk.
void DenseWmma(const void* base, GemvType type, std::uint32_t rows,
               std::uint32_t cols, const float* x, std::uint32_t x_stride,
               float* out, std::uint32_t out_stride, std::uint32_t tokens,
               hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_DENSE_WMMA_HPP_
