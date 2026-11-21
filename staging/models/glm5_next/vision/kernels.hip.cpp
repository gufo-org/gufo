#include "staging/models/glm5_next/vision/kernels.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm5_next::vision {
namespace {

constexpr std::uint32_t kPatch = 14;
constexpr std::uint32_t kPatchVec = kPatch * kPatch * 3;  // 588

hipblasHandle_t BlasHandle() {
  static hipblasHandle_t handle = [] {
    hipblasHandle_t h = nullptr;
    (void)hipblasCreate(&h);
    return h;
  }();
  return handle;
}

__global__ void PatchFoldIm2ColKernel(const float* __restrict__ pixels,
                                      float* __restrict__ patchmat,
                                      std::uint32_t nx, std::uint32_t ny,
                                      std::uint32_t gw, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t t = static_cast<std::uint32_t>(idx / kPatchVec);
  const std::uint32_t col = static_cast<std::uint32_t>(idx % kPatchVec);
  const std::uint32_t blk = t / 4;
  const std::uint32_t sub = t % 4;
  const std::uint32_t px = 2 * (blk % gw) + (sub & 1U);
  const std::uint32_t py = 2 * (blk / gw) + (sub >> 1);
  const std::uint32_t ci = col / (kPatch * kPatch);
  const std::uint32_t rem = col % (kPatch * kPatch);
  const std::uint32_t ky = rem / kPatch;
  const std::uint32_t kx = rem % kPatch;
  patchmat[idx] =
      pixels[(static_cast<std::size_t>(ci) * ny + py * kPatch + ky) * nx +
             px * kPatch + kx];
}

__global__ void RopeVisionKernel(float* __restrict__ qk,
                                 const float* __restrict__ inv_freq,
                                 std::uint32_t gw, std::uint32_t heads,
                                 std::uint32_t hd, std::uint32_t stride,
                                 std::uint32_t quarter, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t pair = static_cast<std::uint32_t>(idx % (2 * quarter));
  const std::uint32_t rest = static_cast<std::uint32_t>(idx / (2 * quarter));
  const std::uint32_t head = rest % heads;
  const std::uint32_t t = rest / heads;
  const std::uint32_t blk = t / 4;
  const std::uint32_t sub = t % 4;
  const float pos = static_cast<float>(pair < quarter
                                           ? 2 * (blk / gw) + (sub >> 1)
                                           : 2 * (blk % gw) + (sub & 1U));
  const float angle = pos * inv_freq[pair % quarter];
  const float c = cosf(angle);
  const float s = sinf(angle);
  float* row = qk + static_cast<std::size_t>(t) * stride +
               static_cast<std::size_t>(head) * hd;
  const float x0 = row[pair];
  const float x1 = row[pair + 2 * quarter];
  row[pair] = x0 * c - x1 * s;
  row[pair + 2 * quarter] = x0 * s + x1 * c;
}

__global__ void GeluErfKernel(const float* __restrict__ x,
                              float* __restrict__ dst, std::size_t count) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const float v = x[idx];
  dst[idx] = 0.5F * v * (1.0F + erff(v * 0.7071067811865475F));
}

__global__ void LayerNormKernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                const float* __restrict__ b,
                                float* __restrict__ dst, std::uint32_t n,
                                float eps, std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::size_t base = idx - idx % n;
  float mean = 0.0F;
  for (std::uint32_t i = 0; i < n; ++i) {
    mean += x[base + i];
  }
  mean /= static_cast<float>(n);
  float var = 0.0F;
  for (std::uint32_t i = 0; i < n; ++i) {
    const float d = x[base + i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(n);
  const float inv = rsqrtf(var + eps);
  const std::uint32_t i = static_cast<std::uint32_t>(idx % n);
  dst[idx] = w[i] * ((x[base + i] - mean) * inv) + b[i];
}

__global__ void MergerIm2ColKernel(const float* __restrict__ h,
                                   float* __restrict__ y, std::uint32_t emb,
                                   std::size_t total) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const std::uint32_t cols = 4 * emb;
  const std::uint32_t t = static_cast<std::uint32_t>(idx / cols);
  const std::uint32_t e = static_cast<std::uint32_t>(idx % cols);
  const std::uint32_t c = e / 4;
  const std::uint32_t q = e % 4;
  y[idx] = h[(static_cast<std::size_t>(4 * t) + q) * emb + c];
}

// One block per (query, head); hd threads, online softmax with the QK scale
// folded into each score. Each thread owns one output feature.
__global__ void AttentionKernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                float* __restrict__ out, std::uint32_t n_pos,
                                std::uint32_t heads, std::uint32_t hd,
                                std::uint32_t stride, float scale) {
  const std::uint32_t p = blockIdx.x;
  const std::uint32_t hh = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const float* qrow = q + static_cast<std::size_t>(p) * stride + hh * hd;
  const float qv = qrow[tid];
  __shared__ float red[128];
  __shared__ float sh_score;
  float acc = 0.0F;
  float m = -INFINITY;
  float l = 0.0F;
  for (std::uint32_t s = 0; s < n_pos; ++s) {
    const float* krow = k + static_cast<std::size_t>(s) * stride + hh * hd;
    red[tid] = qv * krow[tid];
    __syncthreads();
    if (tid == 0) {
      float d = 0.0F;
      for (std::uint32_t j = 0; j < hd; ++j) {
        d += red[j];
      }
      sh_score = d * scale;
    }
    __syncthreads();
    const float score = sh_score;
    const float m_new = fmaxf(m, score);
    const float sc = expf(m - m_new);
    const float pp = expf(score - m_new);
    const float* vrow = v + static_cast<std::size_t>(s) * stride + hh * hd;
    acc = acc * sc + pp * vrow[tid];
    l = l * sc + pp;
    m = m_new;
    __syncthreads();
  }
  out[static_cast<std::size_t>(p) * (heads * hd) + hh * hd + tid] = acc / l;
}

constexpr int kBlock = 256;

}  // namespace

void PatchFoldIm2Col(const float* pixels, float* patchmat, std::uint32_t nx,
                     std::uint32_t ny, std::uint32_t gw,
                     std::uint32_t n_patches, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_patches) * kPatchVec;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  PatchFoldIm2ColKernel<<<grid, kBlock, 0, stream>>>(pixels, patchmat, nx, ny,
                                                     gw, total);
}

void VisionGemm(const float* weight, const float* x, float* out,
                std::uint32_t rows, std::uint32_t cols, std::uint32_t batch,
                hipStream_t stream) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  (void)hipblasSetStream(BlasHandle(), stream);
  (void)hipblasSgemm(
      BlasHandle(), HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(rows),
      static_cast<int>(batch), static_cast<int>(cols), &alpha, weight,
      static_cast<int>(cols), x, static_cast<int>(cols), &beta, out,
      static_cast<int>(rows));
}

void RopeVision(float* qk, const float* inv_freq, std::uint32_t gw,
                std::uint32_t n_patches, std::uint32_t heads, std::uint32_t hd,
                std::uint32_t stride, hipStream_t stream) {
  const std::uint32_t quarter = hd / 4;
  const std::size_t total =
      static_cast<std::size_t>(n_patches) * heads * (2 * quarter);
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  RopeVisionKernel<<<grid, kBlock, 0, stream>>>(qk, inv_freq, gw, heads, hd,
                                                stride, quarter, total);
}

void GeluErf(const float* x, float* dst, std::size_t count,
             hipStream_t stream) {
  const std::uint32_t grid =
      static_cast<std::uint32_t>((count + kBlock - 1) / kBlock);
  GeluErfKernel<<<grid, kBlock, 0, stream>>>(x, dst, count);
}

void LayerNormRows(const float* x, const float* w, const float* b, float* dst,
                   std::uint32_t rows, std::uint32_t n, float eps,
                   hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(rows) * n;
  const std::uint32_t grid =
      static_cast<std::uint32_t>((total + kBlock - 1) / kBlock);
  LayerNormKernel<<<grid, kBlock, 0, stream>>>(x, w, b, dst, n, eps, total);
}

void MergerIm2Col(const float* h, float* y, std::uint32_t n_tokens,
                  std::uint32_t emb, hipStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(n_tokens) * 4 * emb;
  const std::uint32_t grid =
      static_cast<std::size_t>(total + kBlock - 1) / kBlock;
  MergerIm2ColKernel<<<grid, kBlock, 0, stream>>>(h, y, emb, total);
}

void VisionAttention(const float* q, const float* k, const float* v, float* out,
                     std::uint32_t n_patches, std::uint32_t heads,
                     std::uint32_t hd, std::uint32_t stride, float scale,
                     hipStream_t stream) {
  const dim3 grid(n_patches, heads);
  AttentionKernel<<<grid, hd, 0, stream>>>(q, k, v, out, n_patches, heads, hd,
                                           stride, scale);
}

}  // namespace gufo::models::glm5_next::vision
