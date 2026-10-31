#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::gemma4::rocm {

/// Width-invariant decode and verification projections on the matrix cores:
/// Y[t][m] = W X[t] for 1 to kWmmaGemvRows FP32 activation rows and Q4_K,
/// Q5_K, Q6_K, Q4_0 or Q8_0 weights.
///
/// Each 32-value block of an activation row is scaled by a power of two (its
/// maximum lands in [2^13, 2^14)) and split into binary16 hi + lo parts
/// (x = hi + lo to about 2^-22). The weight codes (Q6_K: q - 32, Q4_0: q - 8,
/// Q8_0: q) are exact binary16 values, so per weight group (32 values; Q6_K 16)
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

/// Where a width-k activation pack lives in a LaunchWmmaGemv scratch: a
/// kernel producing a projection input can fill rows [0, rows) of it
/// (PackStaged, wmma_pack.hpp) and the projection then runs with
/// `packed = true`. A null `hi` means no pack.
struct WmmaPackOut {
  void* hi{nullptr};
  void* lo{nullptr};
  float* inv{nullptr};
  float* sums{nullptr};
  std::uint32_t k{0};
};
[[nodiscard]] WmmaPackOut WmmaPackOf(void* scratch, std::uint32_t k) noexcept;

/// Bytes of the activation pack and partial sums LaunchWmmaGemv needs for
/// reductions up to `max_k`.
[[nodiscard]] std::size_t WmmaGemvScratchBytes(std::uint32_t max_k) noexcept;

/// Grouped weight layout: the same 256-value super-blocks as GGUF (Q4_0 and
/// Q8_0: eight blocks), reordered so a 128-row group stores super-block s of
/// all its rows back to back, then s + 1; group g starts at row 128 g's GGUF
/// offset, and the last group is as tall as its rows. The verification walk
/// (eight waves of 16 rows in lockstep over the super-blocks) then reads
/// one contiguous stream per block instead of one per row: 210 -> 220-230
/// GB/s, while the prefill GEMMs read it as before. Same bytes, same
/// results.
[[nodiscard]] bool GroupedLayoutSupports(core::GgmlType type, std::uint32_t m,
                                         std::uint32_t k) noexcept;
/// Reorders GGUF rows `src` [m][k] into the grouped layout at `dst` (another
/// allocation of the same size).
[[nodiscard]] bool GroupWeights(core::GgmlType type, const void* src, void* dst,
                                std::uint32_t m, std::uint32_t k,
                                hipStream_t stream);

/// Y [rows][m] = W X^T for X [rows][k] (1 <= rows <= kWmmaGemvRows). `scratch`
/// holds WmmaGemvScratchBytes. Returns false without launching for
/// unsupported types, shapes or widths. `grouped`: W is in the grouped
/// layout (same results).
[[nodiscard]] bool LaunchWmmaGemv(core::GgmlType type, const void* w,
                                  const float* x, float* y, std::uint32_t rows,
                                  std::uint32_t m, std::uint32_t k,
                                  void* scratch, hipStream_t stream,
                                  bool packed = false, bool grouped = false);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_WMMA_GEMV_HPP_
