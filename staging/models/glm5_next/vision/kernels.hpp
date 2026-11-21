#ifndef GUFO_MODELS_GLM5_NEXT_VISION_KERNELS_HPP_
#define GUFO_MODELS_GLM5_NEXT_VISION_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Device operators for the `glm5v` vision tower that are not covered by the
/// trunk's shared ops tier (`rocm::RmsNorm`, `rocm::SwigluClamp`,
/// `rocm::AddBias`, `rocm::Add`). Every launcher mirrors the scalar oracle
/// in `vision/reference.cpp` exactly: the folded temporal-2 Conv3d im2col in
/// merger-block order, the private hipBLAS FP32 GEMM, the vision RoPE over
/// the first half of each head (pairs (d, d+32), y for d < 16 and x after),
/// the gelu_erf projector activation, the per-row LayerNorm, the 2x2 conv
/// merger's im2col and the non-causal attention with 1/sqrt(head_dim)
/// scaling. Buffers are device pointers; launches are asynchronous on
/// `stream`.
namespace gufo::models::glm5_next::vision {

/// Builds the im2col matrix [n_patches x 588] from CHW pixels that are
/// already normalized with the reference mean/std. Token t enumerates the
/// 2x2 merger blocks in y-major order (block = (t/4), gx blocks per row)
/// with dy/dx fastest: patch column px = 2*(block % gw) + (t%4 & 1), row
/// py = 2*(block / gw) + (t%4 >> 1). Column order matches the folded
/// conv weight: col = kx + 14*ky + 196*ci (kx fastest, then ky, then ci).
void PatchFoldIm2Col(const float* pixels, float* patchmat, std::uint32_t nx,
                     std::uint32_t ny, std::uint32_t gw,
                     std::uint32_t n_patches, hipStream_t stream);

/// out[b][r] = sum_c W[r][c] * x[b][c] over row-major FP32 buffers:
/// W is [rows x cols], x and out are [batch x rows]. hipBLAS SGEMM on the
/// tower's private handle, asynchronous on `stream`.
void VisionGemm(const float* weight, const float* x, float* out,
                std::uint32_t rows, std::uint32_t cols, std::uint32_t batch,
                hipStream_t stream);

/// In-place vision RoPE over a [n_patches x stride] tensor whose head rows
/// start at `qk` (head h owns dims h*hd..). head_dim = 4*quarter; pairs
/// (d, d + 2*quarter) for d in [0, 2*quarter): dims below 2*quarter rotate
/// with the y position, the rest with x. The position of token t is
/// 2*(block % gw) + low bit of (t%4) on x and 2*(block / gw) + high bit on
/// y; `inv_freq` holds `quarter` entries (16 for hd=64).
void RopeVision(float* qk, const float* inv_freq, std::uint32_t gw,
                std::uint32_t n_patches, std::uint32_t heads, std::uint32_t hd,
                std::uint32_t stride, hipStream_t stream);

/// dst[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2))) over `count` elements.
void GeluErf(const float* x, float* dst, std::size_t count,
             hipStream_t stream);

/// LayerNorm over rows rows of n: dst[r] = w * (x[r] - mean) / sqrt(var +
/// eps) + b (the projector's mm.post_norm, eps 1e-5).
void LayerNormRows(const float* x, const float* w, const float* b, float* dst,
                   std::uint32_t rows, std::uint32_t n, float eps,
                   hipStream_t stream);

/// Builds the merger's im2col output [n_tokens x emb*4] from the tower's
/// token-order rows [n_patches x emb]: row t' gathers tokens 4t'..4t'+3 and
/// column e = 4*c + 2*ky + kx reads gathered row (2*ky + kx) feature c, so
/// the folded kernel's col index kx + 2*ky + 4*c matches one SGEMM.
void MergerIm2Col(const float* h, float* y, std::uint32_t n_tokens,
                  std::uint32_t emb, hipStream_t stream);

/// Non-causal single-head-group attention. One block per (query, head);
/// online softmax with `scale` folded into the QK dot. q/k/v rows are
/// `stride` floats apart and the head slice is hd wide; out is
/// [n_patches x heads*hd] contiguous.
void VisionAttention(const float* q, const float* k, const float* v, float* out,
                     std::uint32_t n_patches, std::uint32_t heads,
                     std::uint32_t hd, std::uint32_t stride, float scale,
                     hipStream_t stream);

}  // namespace gufo::models::glm5_next::vision

#endif  // GUFO_MODELS_GLM5_NEXT_VISION_KERNELS_HPP_
