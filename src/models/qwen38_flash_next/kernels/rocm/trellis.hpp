#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Gyro trellis expert weights (TQ2_T, GGUF type 144) and their block-Hadamard
/// rotation; src/core/quant/gyro/tq_ref.c is the normative decoder.
///
/// A 34-byte block holds an F16 scale d and 32 code bytes qs for 128 weights.
/// Step t (0..31) yields four weights from the 16-bit state
/// s = qs[(t + 31) % 32] << 8 | qs[t]: with x = (s & 0x7FFF) * 0x9E3779B1
/// they are d * cb[x >> 21] and d * cb[(x >> 10) & 2047], where cb is the
/// 2048-point F16-pair codebook.
///
/// Rotated tensors store each row as T·w with T = H_128·diag(signs) over the
/// reduction dimension (H_128 the normalized Sylvester-Walsh-Hadamard matrix).
/// T is orthonormal, so the kernels rotate their input once per row instead:
/// (T·w)·(T·x) = w·x.
namespace gufo::models::qwen38_flash_next::rocm {

inline constexpr int kTrellisBlock = 128;
inline constexpr int kTrellisBlockBytes = 34;
inline constexpr int kTrellisCodebookPoints = 2048;

/// Routed TQ2_T gate/up projections of 1..N token rows with SwiGLU:
/// out[t][u][r] = silu(gate_e·x_t) * (up_e·x_t), e = ids[t * used + u], for
/// m = expert width rows of k = hidden columns. Inactive ids (< 0) write
/// zeros. `signs` rotates x when the weights are stored rotated (else null).
/// Tokens routed to the same expert share each weight decode; a token's
/// result does not depend on the others. Requires k == 2560, m % 64 == 0,
/// used <= 32 and tokens * used <= 512.
bool TrellisGatedVec(const void* gate, const void* up, const float* x,
                     const float* signs, const std::uint32_t* codebook,
                     const std::int32_t* ids, float* out, std::uint32_t m,
                     std::uint32_t k, std::uint32_t tokens, std::uint32_t used,
                     hipStream_t stream);

/// Routed TQ2_T down projection of the tokens * used independent slot
/// rows: out[s][r] = w_e·x_s, e = ids[s], for m = hidden rows of k = expert
/// width columns. Slots routed to the same expert share each weight decode,
/// as in TrellisGatedVec. Requires k == 640, m % 128 == 0 and
/// tokens * used <= 512.
bool TrellisDownVec(const void* w, const float* x, const float* signs,
                    const std::uint32_t* codebook, const std::int32_t* ids,
                    float* out, std::uint32_t m, std::uint32_t k,
                    std::uint32_t tokens, std::uint32_t used,
                    hipStream_t stream);

/// out[r] = T·x[r] as F16 for `rows` rows of k (k % 128 == 0), the input of
/// the routed F16 WMMA GEMM over rotated weights. `in_f32` or `in_f16` is
/// the source; with in_f16 == out the rows rotate in place.
void TrellisRotateRows(const float* in_f32, const __half* in_f16, __half* out,
                       const float* signs, std::uint32_t rows, std::uint32_t k,
                       hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_TRELLIS_HPP_
