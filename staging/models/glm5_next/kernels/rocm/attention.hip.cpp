#include "staging/models/glm5_next/kernels/rocm/attention.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>

// Absorbed MLA decode attention over the gathered latent rows; see
// attention.hpp for the layouts. The dot products accumulate sequentially
// along the latent axis and the combine pass sums the latent rows in
// selection order so the host double oracle in
// staging/tests/models/glm5_next/rocm_attention_test.cpp mirrors them
// exactly. Weights here are the dequantized F32 views the executor holds;
// the quantized read paths are a phase-3 tuning concern.
namespace gufo::models::glm5_next::rocm {
namespace {

__device__ __forceinline__ float WarpReduceMax(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down(v, offset));
  }
  return v;
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// Block-wide max for blockDim threads (blockDim % 32 == 0); thread 0 holds
// it after the final __syncthreads.
__device__ __forceinline__ float BlockReduceMax(float v, float* red) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = static_cast<int>(blockDim.x) >> 5;
  v = WarpReduceMax(v);
  if (lane == 0) {
    red[warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    v = (lane < warps) ? red[lane] : -FLT_MAX;
    v = WarpReduceMax(v);
  }
  return v;
}

__global__ void MlaQabsKernel(const float* __restrict__ q,
                              const float* __restrict__ kb, float scale,
                              float* __restrict__ q_abs) {
  __shared__ float sq[kMlaKvRank / 2];
  const std::size_t head = blockIdx.x;
  for (unsigned i = threadIdx.x; i < kMlaKvRank / 2; i += blockDim.x) {
    sq[i] = q[head * (kMlaKvRank / 2) + i];
  }
  __syncthreads();
  const std::size_t row = head * kMlaKvRank + threadIdx.x;
  const float* w = kb + row * (kMlaKvRank / 2);
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kMlaKvRank / 2; ++i) {
    acc += w[i] * sq[i];
  }
  q_abs[row] = scale * acc;
}

__global__ void MlaGatherScoreKernel(const float* __restrict__ q_abs,
                                     const __half* __restrict__ cache,
                                     const std::int32_t* __restrict__ sel,
                                     std::uint32_t n_sel, std::uint32_t tile,
                                     std::uint32_t heads,
                                     float* __restrict__ scores) {
  const std::uint32_t tile_i = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t j = tile_i * tile + threadIdx.x;
  if (j >= n_sel) {
    return;
  }
  const float* qa = q_abs + static_cast<std::size_t>(head) * kMlaKvRank;
  const __half* row = cache + static_cast<std::size_t>(sel[j]) * kMlaKvRank;
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kMlaKvRank; ++i) {
    acc += qa[i] * __half2float(row[i]);
  }
  scores[(static_cast<std::size_t>(tile_i) * heads + head) * tile +
         threadIdx.x] = acc;
}

__global__ void MlaGatherCombineKernel(const float* __restrict__ scores,
                                       const __half* __restrict__ cache,
                                       const std::int32_t* __restrict__ sel,
                                       std::uint32_t tile, std::uint32_t n_sel,
                                       std::uint32_t heads,
                                       float* __restrict__ latent) {
  extern __shared__ float smem[];  // [n_sel] exponents + blockDim reducers
  float* ex = smem;
  float* red = smem + n_sel;
  const std::uint32_t head = blockIdx.x;
  const std::size_t score_row = static_cast<std::size_t>(head) * tile;

  // Softmax max over all tiles is exact under any reduce order; fold the
  // per-thread strided partials with a block max.
  float pm = -FLT_MAX;
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    pm = fmaxf(pm, scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                          score_row + (j % tile)]);
  }
  __syncthreads();
  pm = BlockReduceMax(pm, red);
  if (threadIdx.x == 0) {
    red[0] = pm;
  }
  __syncthreads();
  const float max_s = red[0];

  float psum = 0.0F;
  for (std::uint32_t j = threadIdx.x; j < n_sel; j += blockDim.x) {
    ex[j] = expf(scores[static_cast<std::size_t>(j / tile) * (heads * tile) +
                        score_row + (j % tile)] -
                 max_s);
    psum += ex[j];
  }
  __syncthreads();
  // Sum of exponentials: strided partials then a block tree sum.
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  psum = WarpReduceSum(psum);
  if (lane == 0) {
    red[threadIdx.x >> 5] = psum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.0F;
    for (unsigned w = 0; w < blockDim.x / 32; ++w) {
      s += red[w];
    }
    red[0] = s;
  }
  __syncthreads();
  const float inv = 1.0F / red[0];

  // Latent columns in selection order; each thread owns one column.
  float acc = 0.0F;
  for (std::uint32_t j = 0; j < n_sel; ++j) {
    const __half* row = cache + static_cast<std::size_t>(sel[j]) * kMlaKvRank;
    acc += ex[j] * __half2float(row[threadIdx.x]);
  }
  latent[static_cast<std::size_t>(head) * kMlaKvRank + threadIdx.x] = acc * inv;
}

__global__ void MlaApplyVbKernel(const float* __restrict__ latent,
                                 const float* __restrict__ vb,
                                 float* __restrict__ out) {
  __shared__ float sl[kMlaKvRank];
  const std::size_t head = blockIdx.x;
  for (unsigned i = threadIdx.x; i < kMlaKvRank; i += blockDim.x) {
    sl[i] = latent[head * kMlaKvRank + i];
  }
  __syncthreads();
  const std::size_t row = head * (kMlaKvRank / 2) + threadIdx.x;
  const float* w = vb + row * kMlaKvRank;
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kMlaKvRank; ++i) {
    acc += w[i] * sl[i];
  }
  out[row] = acc;
}

}  // namespace

void MlaQabs(const float* q, const float* kb, std::uint32_t heads, float scale,
             float* q_abs, hipStream_t stream) {
  MlaQabsKernel<<<heads, kMlaKvRank, 0, stream>>>(q, kb, scale, q_abs);
}

std::uint32_t MlaGatherTiles(std::uint32_t n_sel, std::uint32_t tile) {
  return (n_sel + tile - 1) / tile;
}

void MlaGatherScore(const float* q_abs, const __half* cache,
                    const std::int32_t* sel, std::uint32_t n_sel,
                    std::uint32_t tile, std::uint32_t heads, float* scores,
                    hipStream_t stream) {
  const dim3 grid(MlaGatherTiles(n_sel, tile), heads);
  MlaGatherScoreKernel<<<grid, tile, 0, stream>>>(q_abs, cache, sel, n_sel,
                                                  tile, heads, scores);
}

void MlaGatherCombine(const float* scores, const __half* cache,
                      const std::int32_t* sel, std::uint32_t tile,
                      std::uint32_t n_sel, std::uint32_t heads, float* latent,
                      hipStream_t stream) {
  const std::size_t smem = (n_sel + 32) * sizeof(float);
  MlaGatherCombineKernel<<<heads, kMlaKvRank, smem, stream>>>(
      scores, cache, sel, tile, n_sel, heads, latent);
}

void MlaApplyVb(const float* latent, const float* vb, std::uint32_t heads,
                float* out, hipStream_t stream) {
  MlaApplyVbKernel<<<heads, kMlaKvRank / 2, 0, stream>>>(latent, vb, out);
}

}  // namespace gufo::models::glm5_next::rocm
