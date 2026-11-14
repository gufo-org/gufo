#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// HIP launchers of EXL3 trellis linears:
///   y = H(W^T H(x * suh)) * svh,
/// with W the decoded [in, out] mul1 trellis matrix and H the orthonormal
/// 128-point Hadamard of each block. Tiles are stored column-block-major:
/// tile (a, b) of an expert lives at (b * in/16 + a) * 8K dwords, so a wave
/// streaming one 16-column block reads contiguous memory.
namespace gufo::models::qwen38_flash_next::rocm {

struct Exl3Weights {
  const std::uint32_t* tiles{nullptr};  ///< [experts][out/16][in/16][8K]
  const __half* suh{nullptr};           ///< [experts][in]
  const __half* svh{nullptr};           ///< [experts][out]
  std::uint32_t in{0};
  std::uint32_t out{0};
  std::uint32_t experts{1};
  int k{0};

  [[nodiscard]] __host__ __device__ std::size_t ExpertWords() const noexcept {
    return static_cast<std::size_t>(in) * out * static_cast<unsigned>(k) / 32;
  }
};

/// Device scratch the linears share; sized by Exl3ScratchBytes.
struct Exl3Scratch {
  __half* xh{nullptr};  ///< transformed inputs
  std::size_t xh_elems{0};
  float* partial{nullptr};  ///< split-K partial sums
  std::size_t partial_elems{0};
  /// Split arrival counters, zero between launches (each launch resets
  /// the ones it used).
  unsigned* done{nullptr};
  std::size_t done_elems{0};
};

/// Rows of activation one GEMV launch shares its decoded weights across.
inline constexpr std::uint32_t kExl3GemvRows = 8;
/// Rows of the wide GEMV variant (wide verification passes).
inline constexpr std::uint32_t kExl3GemvWideRows = 64;

/// y[r][0, out) for `rows` rows. Row r reads x + (r / x_div) * ldx and,
/// with `ids`, multiplies expert ids[r]'s matrix (else expert 0). Rows are
/// split into launches that fit `scratch`; every row's result is the same
/// whatever the batch: one fixed accumulation order per output.
[[nodiscard]] bool Exl3Linear(const Exl3Weights& w, const float* x,
                              std::uint32_t ldx, std::uint32_t x_div,
                              const std::int32_t* ids, float* y,
                              std::uint32_t ldy, std::uint32_t rows,
                              const Exl3Scratch& scratch, hipStream_t stream);

/// Rows of one fused GEMM tile, and the routed tile map's tile height.
inline constexpr std::uint32_t kExl3GemmRows = 128;
/// Row tile of GEMMs whose inputs are already transformed (`x_transformed`)
/// or go through the scratch pass; their route maps use this tile.
inline constexpr std::uint32_t kExl3TransformedGemmRows = 256;

/// The matrix-core route of a linear over many rows (prompt passes): input
/// transform, F16 WMMA products with the trellis decoded on the fly, and the
/// output transform in one kernel; no scratch. Every output element sums its
/// products in one fixed order, so a row's result does not depend on the
/// other rows of the pass.
///
/// Dense: rows [0, rows) of x map to rows of y. Routed (`tiles` set): tile i
/// is expert tiles[i] & 0xFFFF, rows [bounds[e] + 128 (tiles[i] >> 16),
/// bounds[e + 1]) of the compacted order; compacted position p reads
/// transformed row x_rows[p] and writes y row y_rows[p] (-1 skips it).
struct Exl3GemmRoute {
  const std::int32_t* tiles{nullptr};
  std::uint32_t n_tiles{0};
  const std::int32_t* bounds{nullptr};
  const std::int32_t* x_rows{nullptr};
  const std::int32_t* y_rows{nullptr};
};
/// Optional forms of a GEMM's input and output: F16 inputs (`x_half`, read
/// instead of x), F16 outputs (`y_half`), or, with `gate` (F32 rows of the
/// output's shape), SwiGLU silu(gate) * y stored as F16 in `y_half`.
struct Exl3GemmOut {
  const __half* x_half{nullptr};
  __half* y_half{nullptr};
  const float* gate{nullptr};
  /// Routed only: the GEMM reads `x_half` rows (stride `in`) by the route's
  /// x_rows and transforms them itself; no scratch input pass.
  bool fused_input{false};
  /// `x_half` rows are already transformed (an earlier GEMM's `next_suh`).
  bool x_transformed{false};
  /// With `y_half`: store the next linear's transformed inputs instead,
  /// H(half(y) * next_suh) / sqrt(128), with the row's expert of next_suh
  /// ([experts][out]); out must be that linear's in.
  const __half* next_suh{nullptr};
};

/// The input transform runs first over `x_rows` rows (row r reads x row
/// r / x_div with expert ids[r], as Exl3Linear) into scratch.xh, which the
/// GEMM then reads by x_rows indices.
[[nodiscard]] bool Exl3LinearGemm(
    const Exl3Weights& w, const float* x, std::uint32_t ldx,
    std::uint32_t x_div, const std::int32_t* ids, std::uint32_t x_rows,
    float* y, std::uint32_t ldy, std::uint32_t rows, const Exl3GemmRoute& route,
    const Exl3Scratch& scratch, hipStream_t stream,
    const Exl3GemmOut& out = {});

/// Bytes of the Q8_0 copy Exl3DequantizeQ8 writes for one matrix.
[[nodiscard]] inline std::size_t Exl3Q8Bytes(const Exl3Weights& w) {
  return static_cast<std::size_t>(w.out) * (w.in / 32) * 34;
}

/// Writes expert `expert` of `w` as the equivalent dense matrix
/// M = diag(svh) H W^T H diag(suh) in Q8_0 rows ([out][in], ggml layout), so
/// prompt passes can take the int8 matrix-core GEMM of Q8_0 weights.
[[nodiscard]] bool Exl3DequantizeQ8(const Exl3Weights& w, std::uint32_t expert,
                                    void* dst, hipStream_t stream);

/// Exl3Linear of two expert stacks of one shape over the same rows (gate
/// and up) in one launch; falls back to two launches when the scratch is
/// too small. Same per-row arithmetic as Exl3Linear.
[[nodiscard]] bool Exl3LinearPair(const Exl3Weights& wa, const Exl3Weights& wb,
                                  const float* x, std::uint32_t ldx,
                                  std::uint32_t x_div, const std::int32_t* ids,
                                  float* ya, float* yb, std::uint32_t ldy,
                                  std::uint32_t rows,
                                  const Exl3Scratch& scratch,
                                  hipStream_t stream);

/// Repacks `experts` stored trellis matrices (row-major tiles, as the
/// checkpoint holds them, at byte offsets `src_offsets[e]` of `base`) into
/// the device's column-block-major order at dst + e * dst_stride dwords.
/// `in_blocks`/`out_blocks` (device arrays of 128-block indices, or null for
/// identity) select stored input/output blocks for each device block.
void Exl3Repack(const std::uint8_t* base, const std::uint64_t* src_offsets,
                std::uint32_t* dst, std::uint32_t src_in, std::uint32_t src_out,
                std::uint32_t in, std::uint32_t out, int k,
                std::uint32_t experts, std::size_t dst_stride,
                const std::uint32_t* in_blocks, const std::uint32_t* out_blocks,
                hipStream_t stream);

/// Gathers `count` fp16 scale vectors of `src_len` elements, selecting 128
/// blocks through `blocks` (null keeps all), from byte offsets
/// `offsets[i]` of `base` into dst[i * dst_len].
void Exl3GatherScales(const std::uint8_t* base, const std::uint64_t* offsets,
                      std::uint32_t count, std::uint32_t src_len,
                      const std::uint32_t* blocks, std::uint32_t dst_len,
                      __half* dst, hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_EXL3_HPP_
