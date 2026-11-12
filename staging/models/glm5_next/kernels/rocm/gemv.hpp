#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_GEMV_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_GEMV_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Model-private quantized matrix-vector product for the GLM 5.3 Flash
/// (glm5-next) trunk. The artifact stores its projections in a mixed map:
/// Q4_K embedding, Q5_K/Q8_0 trunk projections, Q8_0 indexer weights,
/// IQ2_XXS/IQ1_M routed gate/up stacks, IQ3_XXS/IQ4_XS routed down stacks,
/// Q5_K/Q6_K dense and shared-expert trios, and a Q2_K/Q3_K nextn trio
/// (never dequantized at load), so the GEMV tier decodes on the fly. The IQ
/// decode paths replicate src/core/quant/ggml_dequant.cpp element for
/// element. Nothing here is shared with another model.
namespace gufo::models::glm5_next::rocm {

/// The quantized row formats the GEMV tier decodes. The executor maps the
/// artifact's GGUF type onto this.
enum class GemvType {
  kQ8_0,
  kQ5_K,
  kQ6_K,
  kQ4_K,
  kQ2_K,
  kQ3_K,
  kIQ2_XXS,
  kIQ1_M,
  kIQ3_XXS,
  kIQ4_XS,
  kF32,
  kF16,
};

/// Out-of-place F32 -> binary16 retype (round to nearest even) for the
/// resident router matrices; 4 bytes shrink to 2, so it cannot be in place.
/// `overflow` receives a device-side count of values outside the binary16
/// range; the caller fails the load when it is non-zero. Asynchronous on
/// `stream` (null uses the default stream).
void ConvertF32ToHalf(const float* src, __half* dst, std::size_t count,
                      std::uint32_t* overflow, hipStream_t stream);

/// out[r] = sum_k W[r][k] * x[k] for r in [0, rows). `base` points at the first
/// row of a row-major [rows x cols] matrix stored in `type`; `row_bytes` is the
/// encoded size of one row and must match TensorRef::RowBytes for that type and
/// `cols`. For a stacked expert matrix the caller passes the expert's base
/// (TensorRef::Expert). One warp per output row; the launch is asynchronous on
/// `stream` (null uses the default stream).
void Gemv(const void* base, GemvType type, std::uint32_t rows,
          std::uint32_t cols, std::size_t row_bytes, const float* x, float* out,
          hipStream_t stream);

/// Multi-row variant of Gemv for the speculative verify pass:
/// out[o*out_stride + r] = sum_k W[r][k] * x[o*x_stride + k] for o in
/// [0, tokens). `tokens` must be in [2, 16]; a width the tier has no
/// instantiation for runs as descending power-of-two slices. One wave owns
/// an output row and dots the weight row against all activation rows in a
/// single walk, so the matrix is read once for the whole token block. Each
/// per-row dot is bit-identical to the matching Gemv call.
void GemvRows(const void* base, GemvType type, std::uint32_t tokens,
              std::uint32_t rows, std::uint32_t cols, std::size_t row_bytes,
              const float* x, std::uint32_t x_stride, float* out,
              std::uint32_t out_stride, hipStream_t stream);

/// Grouped multi-row variant: the weight rows are `groups` stacked sub-matrices
/// of `group_rows` rows each (a per-head loop over a stacked expert tensor run
/// as one launch). For row r the activation base advances by
/// `(r / group_rows) * group_advance` floats, reproducing the per-head call
/// that passed `x + head * group_advance` as the base; every per-row dot is
/// bit-identical to that loop. out and strides follow GemvRows with
/// `groups * group_rows` rows.
void GemvRowsGrouped(const void* base, GemvType type, std::uint32_t tokens,
                     std::uint32_t groups, std::uint32_t group_rows,
                     std::uint32_t cols, std::size_t row_bytes, const float* x,
                     std::uint32_t x_stride, std::uint32_t group_advance,
                     float* out, std::uint32_t out_stride,
                     hipStream_t stream);

/// Grouped decode GEMV over a stacked expert matrix. For slot s in [0, used)
/// with expert id ids[s]: out[s*rows + r] = sum_k W[e][r][k] *
/// x[(s/slots_per_row)*x_stride + k]. `slots_per_row` is the number of
/// consecutive slots fed by one activation row (ids stay in slot order,
/// [rows][used] when rows > 1): pass the slot count of one row to keep every
/// slot reading the same x row. `expert_stride` is the byte distance between
/// stacked experts (row_bytes * rows). One launch covers all selected experts,
/// so the caller never reads ids back to the host.
void GemvGrouped(const void* base, GemvType type, std::size_t expert_stride,
                 const std::int32_t* ids, std::uint32_t used,
                 std::uint32_t slots_per_row, std::uint32_t rows,
                 std::uint32_t cols, const float* x, std::uint32_t x_stride,
                 float* out, hipStream_t stream);

/// Bucketed tile GEMV over a stacked expert matrix: the wide-chunk form of
/// GemvGrouped. `tiles` (BuildRoutedTiles) lists `capacity` tile entries
/// `expert | tile << 16` (dead beyond the last, skipped), `pad_bounds` /
/// `rows_slot` the 16-row padded bucket layout (RoutedCompact). One wave
/// decodes its (expert, row) once and dots it against all up to 16 bucket
/// slots, writing out[slot*rows + r] for each; every per-slot dot is
/// bit-identical to GemvGrouped's for the same routing. Covers the
/// 256-block super-codec types (IQ2_XXS/IQ1_M/IQ3_XXS/IQ4_XS/Q2_K/Q3_K).
void GemvRoutedTiles(const void* base, GemvType type, std::size_t expert_stride,
                     const std::int32_t* tiles, std::uint32_t capacity,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_slot, std::uint32_t slots_per_row,
                     std::uint32_t rows, std::uint32_t cols, const float* x,
                     std::uint32_t x_stride, float* out, hipStream_t stream);

/// out[i] = W[row][i] for i in [0, cols): dequantizes a single row of a
/// row-major [rows x cols] matrix stored in `type` (the token-embedding
/// lookup). `base` points at the matrix start; the row offset is derived from
/// `cols` and `type`. Asynchronous on `stream` (null uses the default stream).
/// Every GemvType is supported, which doubles as the per-type decode contract
/// checked against gufo::quant::DequantizeRow by the parity test.
void EmbedRow(const void* base, GemvType type, std::uint32_t row,
              std::uint32_t cols, float* out, hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_GEMV_HPP_
