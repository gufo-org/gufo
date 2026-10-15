#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::rocm {

/// Width-invariant decode and verification projections on the matrix cores:
/// Y[t][m] = W X[t] for 1 to kWmmaGemvRows FP32 activation rows and Q4_K,
/// Q5_K, Q6_K or Q4_0 weights.
///
/// Each 32-value block of an activation row is scaled by a power of two (its
/// maximum lands in [2^13, 2^14)) and split into binary16 hi + lo parts
/// (x = hi + lo to about 2^-22). The weight codes (Q6_K: q - 32, Q4_0: q - 8)
/// are exact binary16 values, so per weight group (32 values; Q6_K 16)
/// S = codes . hi + codes . lo accumulates in FP32 WMMAs, each output
/// independent of the other tile rows and columns, and the row's FP32 scale
/// (and Q4_K / Q5_K minimum) applies on the vector units:
///   total = fma(d, (S_hi + S_lo) * 2^-e, total);
///   total = fma(-dmin m, sum(x), total).
/// Up to eight rows pack their hi and lo parts into one WMMA tile; past that
/// a hi tile and a lo tile add the same two chains, so a row rounds
/// identically at every width.
inline constexpr std::uint32_t kWmmaGemvRows = 16;

/// Whether the WMMA projections take W of this type and shape.
[[nodiscard]] bool WmmaGemvSupports(core::GgmlType type, std::uint32_t m,
                                    std::uint32_t k) noexcept;

/// Bytes of the activation pack and partial sums LaunchWmmaGemv needs for
/// reductions up to `max_k`.
[[nodiscard]] std::size_t WmmaGemvScratchBytes(std::uint32_t max_k) noexcept;

/// Y [rows][m] = W X^T for X [rows][k] (1 <= rows <= kWmmaGemvRows). `scratch`
/// holds WmmaGemvScratchBytes. Returns false without launching for
/// unsupported types, shapes or widths.
[[nodiscard]] bool LaunchWmmaGemv(core::GgmlType type, const void* w,
                                  const float* x, float* y, std::uint32_t rows,
                                  std::uint32_t m, std::uint32_t k,
                                  void* scratch, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_
