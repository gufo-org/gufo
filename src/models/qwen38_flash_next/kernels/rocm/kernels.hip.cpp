#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <hipcub/block/block_radix_sort.hpp>
#include <stdexcept>
#include <type_traits>

#include "src/models/qwen38_flash_next/mtp_sampling.hpp"

// HIP kernels follow the layouts and operator formulas in reference.cpp.

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

__global__ void EmbedKernel(const void* table, WeightType type,
                            const std::int32_t* tokens, float* res,
                            std::uint32_t hidden, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const auto* row = static_cast<const std::uint8_t*>(table) +
                    RowBytes(type, hidden) * tokens[t];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = RowElement(row, type, i);
    for (std::uint32_t s = 0; s < streams; ++s) {
      res[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
    }
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    ss += src[i] * src[i];
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
  // A grouped norm shares one gamma row across the groups of a token.
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0f);
  }
}

/// grid (tokens, hidden chunks of kThreads): the stream loop stays inside
/// the thread. With `inject_w` every block also contributes its slice of the
/// inject dot products, written as partial sums [t][stream][chunk] that
/// HcCombine totals (a fixed-order reduction, so the result is stable).
__global__ void HcMixEpilogueKernel(const float* xn, const float* gate,
                                    const float* inject_w, float* mixed,
                                    float* inject, std::uint32_t hidden,
                                    std::uint32_t streams) {
  constexpr std::uint32_t kMaxStreams = 8;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(streams) * hidden;
  const float* x = xn + static_cast<std::size_t>(t) * hc_dim;
  float v[kMaxStreams];
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < streams; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    v[s] = i < hidden ? x[idx] : 0.0f;
    if (i < hidden) {
      acc += v[s] * SigmoidF(gate[static_cast<std::size_t>(t) * hc_dim + idx]);
    }
  }
  if (i < hidden) {
    mixed[static_cast<std::size_t>(t) * hidden + i] =
        acc / static_cast<float>(streams);
  }
  if (inject_w == nullptr) {
    return;
  }
  // All inject dots reduce through one shared pass: wave sums first, then
  // one thread per logit totals the waves.
  __shared__ float partial[kMaxStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  for (std::uint32_t o = 0; o < streams; ++o) {
    const float* w = inject_w + o * hc_dim;
    float dot = 0.0f;
    if (i < hidden) {
      for (std::uint32_t s = 0; s < streams; ++s) {
        dot += w[static_cast<std::size_t>(s) * hidden + i] * v[s];
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < streams) {
    float total = 0.0f;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * streams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// Four adjacent hidden lanes per thread. This keeps the stream arithmetic in
/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}

template<typename XnT>
__global__ void HcMixEpilogueVec4Kernel(const XnT* xn, const float* gate,
                                        const float* inject_w, float* mixed,
                                        float* inject, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const XnT* x = xn + static_cast<std::size_t>(t) * hc_dim;
  float4 v[kStreams];
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    const std::size_t idx = static_cast<std::size_t>(s) * hidden + i;
    v[s] = i < hidden ? Load4(x + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (i < hidden) {
      const float4 g = *reinterpret_cast<const float4*>(
          gate + static_cast<std::size_t>(t) * hc_dim + idx);
      acc.x += v[s].x * SigmoidF(g.x);
      acc.y += v[s].y * SigmoidF(g.y);
      acc.z += v[s].z * SigmoidF(g.z);
      acc.w += v[s].w * SigmoidF(g.w);
    }
  }
  if (i < hidden) {
    constexpr float kInvStreams = 1.0F / static_cast<float>(kStreams);
    acc.x *= kInvStreams;
    acc.y *= kInvStreams;
    acc.z *= kInvStreams;
    acc.w *= kInvStreams;
    *reinterpret_cast<float4*>(mixed + static_cast<std::size_t>(t) * hidden +
                               i) = acc;
  }
  if (inject_w == nullptr) {
    return;
  }
  __shared__ float partial[kStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    const float* w = inject_w + static_cast<std::size_t>(o) * hc_dim;
    float dot = 0.0F;
    if (i < hidden) {
#pragma unroll
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        const float4 q = *reinterpret_cast<const float4*>(
            w + static_cast<std::size_t>(s) * hidden + i);
        dot += q.x * v[s].x;
        dot += q.y * v[s].y;
        dot += q.z * v[s].z;
        dot += q.w * v[s].w;
      }
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }
  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * gridDim.y +
           blockIdx.y] = total;
  }
}

/// The W8A8 GEMM's activation tiles: 16 tokens x one 32-wide K block,
/// codes in fragment order (two 256-byte halves of 16 tokens x 16 codes)
/// followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

/// One token per block keeps independent rows in flight. The FP32 mixed
/// row, optional F16/Q8 projection inputs and inject partials share one
/// read of the four residual streams. Grid: (inject parts, tokens).
template<bool kMix = true>
__global__ void HcMixEpilogueF16Kernel(const __half* xn, const float* gate,
                                       const float* inject_w, float* mixed,
                                       __half* mixed_half, void* mixed_q8,
                                       float* inject, std::uint32_t hidden) {
  constexpr std::uint32_t kStreams = 4;
  const std::uint32_t part = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = (part * blockDim.x + threadIdx.x) * 4;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  const bool live = i < hidden;
  float4 wq[kStreams][kStreams];
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      wq[o][s] = live && inject_w != nullptr
                     ? *reinterpret_cast<const float4*>(
                           inject_w + (static_cast<std::size_t>(o) * hc_dim) +
                           (static_cast<std::size_t>(s) * hidden) + i)
                     : float4{0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
  __shared__ float partial[kStreams][kThreads / 32];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  constexpr float kInvStreams = 1.0F / static_cast<float>(kStreams);
  float4 v[kStreams];
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    const std::size_t idx = (static_cast<std::size_t>(t) * hc_dim) +
                            (static_cast<std::size_t>(s) * hidden) + i;
    v[s] = live ? Load4(xn + idx) : float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (kMix && live) {
      const float4 g = *reinterpret_cast<const float4*>(gate + idx);
      acc.x += v[s].x * SigmoidF(g.x);
      acc.y += v[s].y * SigmoidF(g.y);
      acc.z += v[s].z * SigmoidF(g.z);
      acc.w += v[s].w * SigmoidF(g.w);
    }
  }
  if (kMix && live) {
    acc.x *= kInvStreams;
    acc.y *= kInvStreams;
    acc.z *= kInvStreams;
    acc.w *= kInvStreams;
    *reinterpret_cast<float4*>(mixed + static_cast<std::size_t>(t) * hidden +
                               i) = acc;
    if (mixed_half != nullptr) {
      __half* out = mixed_half + (static_cast<std::size_t>(t) * hidden) + i;
      *reinterpret_cast<__half2*>(out) = __floats2half2_rn(acc.x, acc.y);
      *reinterpret_cast<__half2*>(out + 2) = __floats2half2_rn(acc.z, acc.w);
    }
  }
  if (kMix && mixed_q8 != nullptr) {
    // Eight lanes hold one 32-wide block (every lane joins the reduction).
    float max_abs = fmaxf(fmaxf(fabsf(acc.x), fabsf(acc.y)),
                          fmaxf(fabsf(acc.z), fabsf(acc.w)));
    for (int off = 4; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    if (live) {
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.x * id))));
      const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.y * id))));
      const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.z * id))));
      const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(acc.w * id))));
      const std::size_t kb = i / 32;
      const std::uint32_t pos = i % 32;  // 0, 4, ..., 28
      std::int8_t* tile =
          Q8ActTile(mixed_q8, hidden / 32, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                        (pos & 15u)) =
          q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
      if (pos == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
  if (inject_w == nullptr) {
    return;
  }
#pragma unroll
  for (std::uint32_t o = 0; o < kStreams; ++o) {
    float dot = 0.0F;
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      dot += wq[o][s].x * v[s].x;
      dot += wq[o][s].y * v[s].y;
      dot += wq[o][s].z * v[s].z;
      dot += wq[o][s].w * v[s].w;
    }
    dot = WaveSum(dot);
    if (lane == 0) {
      partial[o][wave] = dot;
    }
  }

  __syncthreads();
  if (threadIdx.x < kStreams) {
    float total = 0.0F;
    for (std::uint32_t w = 0; w < kThreads / 32; ++w) {
      total += partial[threadIdx.x][w];
    }
    inject[(static_cast<std::size_t>(t) * kStreams + threadIdx.x) * gridDim.x +
           part] = total;
  }
}

/// grid (tokens, streams): one block owns one residual stream, so the
/// grouped norm of the next mixer reduces over exactly its own elements.
/// `XnT` is float for the reference route and __half for the F16 mixer
/// input route; a non-null `xn_q8` also receives the norm quantized into
/// the tiled Q8 layout (hidden % 32 == 0) for the W8A8 down projection.
template<typename XnT>
__global__ void HcCombineKernel(float* res, const float* block_out,
                                const float* inject, std::uint32_t inject_parts,
                                const float* gamma, XnT* xn, void* xn_q8,
                                std::uint32_t hidden, std::uint32_t streams,
                                float eps) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  float logit = 0.0f;
  for (std::uint32_t p = 0; p < inject_parts; ++p) {
    logit +=
        inject[(static_cast<std::size_t>(t) * streams + s) * inject_parts + p];
  }
  const float w = 2.0f * SigmoidF(logit / static_cast<float>(streams));
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float* dst = res + base;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] + src[i] * w;
    dst[i] = v;
    ss += v * v;
  }
  if (gamma == nullptr) {
    return;
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(hidden) + eps);
  const float* g = gamma + static_cast<std::size_t>(s) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = dst[i] * scale * g[i];
    xn[base + i] = static_cast<XnT>(v);
    if (xn_q8 != nullptr) {
      // A wave holds one 32-wide block of the row per iteration: quantize
      // it for the next mixer's W8A8 down projection, K = streams * hidden.
      float max_abs = fabsf(v);
      for (int off = 16; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
      }
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q = static_cast<std::int8_t>(roundf(v * id));
      const std::size_t num_blocks =
          (static_cast<std::size_t>(streams) * hidden) / 32;
      const std::size_t kb = (static_cast<std::size_t>(s) * hidden + i) / 32;
      const std::size_t lane = threadIdx.x & 31u;
      std::int8_t* tile =
          Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
      if (lane == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
}

/// HcCombineKernel over one token's whole hyper-connection row (all streams,
/// float4 lanes): a block owns streams * hidden elements in up to
/// kMaxChunks float4 per thread, keeps the updated residual in registers
/// between the two passes, reduces the four stream norms in one block
/// reduction, and quantizes each 32-wide block over its eight lanes with a
/// four-code store. Rows of four streams up to 2,560 wide.
template<typename XnT>
__global__ void HcCombineVec4Kernel(float* res, const float* block_out,
                                    const float* inject,
                                    std::uint32_t inject_parts,
                                    const float* gamma, XnT* xn, void* xn_q8,
                                    std::uint32_t hidden, float eps) {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kMaxChunks = 10;  // 4 x 2560 / 4 / kThreads
  __shared__ float shared[kStreams][kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float w[kStreams];
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    float logit = 0.0f;
    for (std::uint32_t p = 0; p < inject_parts; ++p) {
      logit +=
          inject[(static_cast<std::size_t>(t) * kStreams + s) * inject_parts +
                 p];
    }
    w[s] = 2.0f * SigmoidF(logit / static_cast<float>(kStreams));
  }
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  const float* src = block_out + static_cast<std::size_t>(t) * hidden;
  const std::uint32_t chunks = static_cast<std::uint32_t>(hc_dim / 4);
  float4 v[kMaxChunks];
  float ss[kStreams] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    v[c] = float4{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const std::uint32_t i = e - (s * hidden);
      const float ws = w[s];
      const float4 r = *reinterpret_cast<const float4*>(dst + e);
      const float4 b = *reinterpret_cast<const float4*>(src + i);
      v[c] = float4{r.x + b.x * ws, r.y + b.y * ws, r.z + b.z * ws,
                    r.w + b.w * ws};
      *reinterpret_cast<float4*>(dst + e) = v[c];
      const float sq =
          v[c].x * v[c].x + v[c].y * v[c].y + v[c].z * v[c].z + v[c].w * v[c].w;
#pragma unroll
      for (std::uint32_t k = 0; k < kStreams; ++k) {
        ss[k] += (k == s) ? sq : 0.0F;
      }
    }
  }
  if (gamma == nullptr) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    ss[k] = WaveSum(ss[k]);
    if (lane == 0) {
      shared[k][wave] = ss[k];
    }
  }
  __syncthreads();
  float scale[kStreams];
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total += shared[k][q];
    }
    scale[k] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  const std::size_t num_blocks = hc_dim / 32;
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t e = (c * kThreads + threadIdx.x) * 4;
    // Every lane of the wave takes part in the block reductions below, so
    // lanes past the row carry zeros instead of leaving.
    float4 n{0.0F, 0.0F, 0.0F, 0.0F};
    if (e < hc_dim) {
      const std::uint32_t s = e / hidden;
      const float sc = scale[s];
      const float4 gm = *reinterpret_cast<const float4*>(gamma + e);
      n = float4{v[c].x * sc * gm.x, v[c].y * sc * gm.y, v[c].z * sc * gm.z,
                 v[c].w * sc * gm.w};
      XnT* out = xn + (static_cast<std::size_t>(t) * hc_dim) + e;
      if constexpr (std::is_same_v<XnT, float>) {
        *reinterpret_cast<float4*>(out) = n;
      } else {
        *reinterpret_cast<__half2*>(out) = __floats2half2_rn(n.x, n.y);
        *reinterpret_cast<__half2*>(out + 2) = __floats2half2_rn(n.z, n.w);
      }
    }
    if (xn_q8 == nullptr) {
      continue;
    }
    // Eight lanes hold one 32-wide block: absmax over them, then each lane
    // stores its four codes as one word.
    float max_abs =
        fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
    for (int off = 4; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    if (e < hc_dim) {
      const float d = max_abs / 127.0F;
      const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
      const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.x * id))));
      const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.y * id))));
      const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.z * id))));
      const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(roundf(n.w * id))));
      const std::size_t kb = e / 32;
      const std::uint32_t pos = e % 32;  // 0, 4, ..., 28
      std::int8_t* tile =
          Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
      const std::size_t tl = t % kQ8ActTileTokens;
      *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                        (pos & 15u)) =
          q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
      if (pos == 0) {
        *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                  (tl * sizeof(float))) = d;
      }
    }
  }
}

/// HcCombineVec4Kernel<__half> with the MoE epilogue fused in: the block
/// output row is formed in registers from the routed experts' F16 rows,
/// their weights and the gated shared expert (MoeEpilogueVec4Kernel), so the
/// block output never round-trips memory. Threads own hidden lanes (up to
/// three float4 each) across the four streams.
__global__ void HcCombineMoeF16Kernel(
    float* res, const __half* expert_out, const float* weights,
    const float* shared_out, const float* gate, std::uint32_t gate_stride,
    std::uint32_t used, const float* inject, std::uint32_t inject_parts,
    const float* gamma, __half* xn, void* xn_q8, std::uint32_t hidden,
    float eps) {
  constexpr std::uint32_t kStreams = 4;
  constexpr std::uint32_t kMaxChunks = 3;  // 2560 / 4 / kThreads, rounded up
  __shared__ float shared[kStreams][kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  const std::size_t hc_dim = static_cast<std::size_t>(kStreams) * hidden;
  float w[kStreams];
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
    float logit = 0.0f;
    for (std::uint32_t p = 0; p < inject_parts; ++p) {
      logit +=
          inject[(static_cast<std::size_t>(t) * kStreams + s) * inject_parts +
                 p];
    }
    w[s] = 2.0f * SigmoidF(logit / static_cast<float>(kStreams));
  }
  float* dst = res + static_cast<std::size_t>(t) * hc_dim;
  // The block output: sum of the weighted expert rows plus the gated shared
  // expert, per hidden chunk.
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const __half* rows = expert_out + static_cast<std::size_t>(t) * used * hidden;
  float4 v[kStreams][kMaxChunks];
  float ss[kStreams] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
  for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
    const std::uint32_t i = (c * kThreads + threadIdx.x) * 4;
    float4 b{0.0F, 0.0F, 0.0F, 0.0F};
    if (i < hidden) {
      for (std::uint32_t k = 0; k < used; ++k) {
        const float wk = weights[t * used + k];
        const float4 e = Load4(rows + static_cast<std::size_t>(k) * hidden + i);
        b.x += wk * e.x;
        b.y += wk * e.y;
        b.z += wk * e.z;
        b.w += wk * e.w;
      }
      const float4 sh = *reinterpret_cast<const float4*>(
          shared_out + static_cast<std::size_t>(t) * hidden + i);
      b.x += g * sh.x;
      b.y += g * sh.y;
      b.z += g * sh.z;
      b.w += g * sh.w;
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kStreams; ++s) {
      v[s][c] = float4{0.0F, 0.0F, 0.0F, 0.0F};
      if (i < hidden) {
        const std::size_t e = (static_cast<std::size_t>(s) * hidden) + i;
        const float4 r = *reinterpret_cast<const float4*>(dst + e);
        v[s][c] = float4{r.x + b.x * w[s], r.y + b.y * w[s], r.z + b.z * w[s],
                         r.w + b.w * w[s]};
        *reinterpret_cast<float4*>(dst + e) = v[s][c];
        ss[s] += v[s][c].x * v[s][c].x + v[s][c].y * v[s][c].y +
                 v[s][c].z * v[s][c].z + v[s][c].w * v[s][c].w;
      }
    }
  }
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5u;
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    ss[k] = WaveSum(ss[k]);
    if (lane == 0) {
      shared[k][wave] = ss[k];
    }
  }
  __syncthreads();
  float scale[kStreams];
#pragma unroll
  for (std::uint32_t k = 0; k < kStreams; ++k) {
    float total = 0.0F;
    for (std::uint32_t q = 0; q < kThreads / 32; ++q) {
      total += shared[k][q];
    }
    scale[k] = rsqrtf(total / static_cast<float>(hidden) + eps);
  }
  const std::size_t num_blocks = hc_dim / 32;
  const std::size_t tl = t % kQ8ActTileTokens;
#pragma unroll
  for (std::uint32_t s = 0; s < kStreams; ++s) {
#pragma unroll
    for (std::uint32_t c = 0; c < kMaxChunks; ++c) {
      const std::uint32_t i = (c * kThreads + threadIdx.x) * 4;
      const std::size_t e = (static_cast<std::size_t>(s) * hidden) + i;
      // Every lane of the wave takes part in the block reductions below,
      // so lanes past the row carry zeros instead of leaving.
      float4 n{0.0F, 0.0F, 0.0F, 0.0F};
      if (i < hidden) {
        const float4 gm = *reinterpret_cast<const float4*>(gamma + e);
        n = float4{v[s][c].x * scale[s] * gm.x, v[s][c].y * scale[s] * gm.y,
                   v[s][c].z * scale[s] * gm.z, v[s][c].w * scale[s] * gm.w};
        __half* out = xn + (static_cast<std::size_t>(t) * hc_dim) + e;
        *reinterpret_cast<__half2*>(out) = __floats2half2_rn(n.x, n.y);
        *reinterpret_cast<__half2*>(out + 2) = __floats2half2_rn(n.z, n.w);
      }
      // Eight lanes hold one 32-wide block: absmax over them, then each
      // lane stores its four codes as one word.
      float max_abs =
          fmaxf(fmaxf(fabsf(n.x), fabsf(n.y)), fmaxf(fabsf(n.z), fabsf(n.w)));
      for (int off = 4; off > 0; off >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
      }
      if (i < hidden) {
        const float d = max_abs / 127.0F;
        const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
        const auto q0 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.x * id))));
        const auto q1 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.y * id))));
        const auto q2 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.z * id))));
        const auto q3 = static_cast<std::uint32_t>(static_cast<std::uint8_t>(
            static_cast<std::int8_t>(roundf(n.w * id))));
        const std::size_t kb = e / 32;
        const std::uint32_t pos = e % 32;  // 0, 4, ..., 28
        std::int8_t* tile =
            Q8ActTile(xn_q8, num_blocks, t / kQ8ActTileTokens, kb);
        *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) +
                                          (tl * 16) + (pos & 15u)) =
            q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        if (pos == 0) {
          *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                    (tl * sizeof(float))) = d;
        }
      }
    }
  }
}

__global__ void SiluScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] = SiluF(x[i] * scale);
  }
}

/// out = silu(gate) * up as F16, four elements per thread.
__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 4;
  if (i + 4 > count) {
    for (std::size_t j = i; j < count; ++j) {
      out[j] = __float2half(SiluF(gate[j]) * up[j]);
    }
    return;
  }
  const float4 g = *reinterpret_cast<const float4*>(gate + i);
  const float4 u = *reinterpret_cast<const float4*>(up + i);
  const __half2 lo = __floats2half2_rn(SiluF(g.x) * u.x, SiluF(g.y) * u.y);
  const __half2 hi = __floats2half2_rn(SiluF(g.z) * u.z, SiluF(g.w) * u.w);
  *reinterpret_cast<__half2*>(out + i) = lo;
  *reinterpret_cast<__half2*>(out + i + 2) = hi;
}

/// silu(gate) * up over rows of `k` elements, written only as the W8A8
/// tiled Q8 layout: four elements per lane, eight lanes per K block.
__global__ void SwigluQ8Kernel(const float* gate, const float* up, void* out_q8,
                               std::size_t n_rows, std::size_t k) {
  const std::size_t chunk =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t chunks_per_row = k / 4;
  const std::size_t row = chunk / chunks_per_row;
  const std::size_t i = (chunk % chunks_per_row) * 4;
  // Rows past the batch still join the wave reduction below.
  const bool live = row < n_rows;
  float4 v{0.0F, 0.0F, 0.0F, 0.0F};
  if (live) {
    const std::size_t idx = (row * k) + i;
    const float4 g = *reinterpret_cast<const float4*>(gate + idx);
    const float4 u = *reinterpret_cast<const float4*>(up + idx);
    v = float4{SiluF(g.x) * u.x, SiluF(g.y) * u.y, SiluF(g.z) * u.z,
               SiluF(g.w) * u.w};
  }
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  if (!live) {
    return;
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tl = row % kQ8ActTileTokens;
  const std::uint32_t pos = static_cast<std::uint32_t>(i % 32);
  std::int8_t* tile = Q8ActTile(out_q8, k / 32, row / kQ8ActTileTokens, i / 32);
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (pos == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

/// One wave per weight row, kSmallGemmRows rows per block: the row is read
/// once (four consecutive elements per lane per step) while up to eight
/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmRows = 4;
template<WeightType type, unsigned tokens, bool grouped = false>
__global__ void SmallGemmKernel(const void* w, const float* x, float* out,
                                std::uint32_t m, std::uint32_t k) {
  if constexpr (grouped) {
    x += std::size_t{blockIdx.y} * tokens * k;
    out += std::size_t{blockIdx.y} * tokens * m;
  }
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  float acc[tokens] = {};
  for (std::uint32_t i0 = lane * 4; i0 < k; i0 += warpSize * 4) {
    float wv[4];
#pragma unroll
    for (unsigned r = 0; r < 4; ++r) {
      wv[r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
    }
#pragma unroll
    for (unsigned j = 0; j < tokens; ++j) {
      const float* xr = x + static_cast<std::size_t>(j) * k + i0;
      float dot = 0.0f;
#pragma unroll
      for (unsigned r = 0; r < 4; ++r) {
        dot += wv[r] * (i0 + r < k ? xr[r] : 0.0f);
      }
      acc[j] += dot;
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0) {
      out[static_cast<std::size_t>(j) * m + row] = total;
    }
  }
}

__global__ void PleGateKernel(const float* key_n, const float* query_n,
                              const float* value, float* gated,
                              std::uint32_t hidden, std::uint32_t streams) {
  __shared__ float shared[32];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t s = blockIdx.y;
  const std::size_t base = (static_cast<std::size_t>(t) * streams + s) * hidden;
  float dot = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dot += key_n[base + i] * query_n[base + i];
  }
  dot = BlockSum(dot, shared);
  const float sc = dot * rsqrtf(static_cast<float>(hidden));
  const float mag = sqrtf(fmaxf(fabsf(sc), 1e-6f));
  const float sign = sc < 0.0f ? -1.0f : (sc > 0.0f ? 1.0f : 0.0f);
  const float gate = SigmoidF(sign * mag);
  const float* v = value + static_cast<std::size_t>(t) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    gated[base + i] = v[i] * gate;
  }
}

__global__ void PleConvKernel(const float* in, const float* w,
                              const float* history, float* out,
                              std::uint32_t n_tokens, std::uint32_t channels,
                              std::uint32_t kernel, std::uint32_t dilation,
                              std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t) -
        static_cast<std::int32_t>((kernel - 1 - k) * dilation);
    const float v =
        src_t >= 0
            ? in[static_cast<std::size_t>(src_t) * channels + c]
            : history[static_cast<std::size_t>(hist + src_t) * channels + c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// New history row j is row (n_tokens + j) of [history ; in].
__global__ void HistoryShiftKernel(const float* in, std::uint32_t in_stride,
                                   const float* history, float* scratch,
                                   std::uint32_t n_tokens,
                                   std::uint32_t channels, std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(hist) * channels) {
    return;
  }
  const std::uint32_t j = idx / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = n_tokens + j;
  scratch[idx] = src < hist
                     ? history[static_cast<std::size_t>(src) * channels + c]
                     : in[static_cast<std::size_t>(src - hist) * in_stride + c];
}

__global__ void PleInjectKernel(float* res, const float* gated,
                                const float* conv, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    res[i] += gated[i] + conv[i];
  }
}

/// L2-normalizes the convolved q and k of one (token, key head) into the
/// packed [t][kh][d] buffers the recurrence streams from, so the serial loop
/// carries no block-wide reductions. One wave per (token, head) row.
constexpr unsigned kGdnDim = 128;
template<bool kBatch>
__global__ void GdnPrepKernel(const float* conv_out, float* qn, float* kn,
                              std::uint32_t n_rows, std::uint32_t k_heads,
                              std::uint32_t channels, float eps,
                              const GdnBatchItem* batch, std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    n_rows = item.n_tokens * k_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  const std::uint32_t t = row / k_heads;
  const std::uint32_t kh = row % k_heads;
  const float* conv = conv_out + static_cast<std::size_t>(t) * channels;
  const std::uint32_t per_lane = d / warpSize;
  float qv[d / 32];
  float kv[d / 32];
  float qs = 0.0f;
  float ks = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    qv[r] = conv[kh * d + i];
    kv[r] = conv[k_heads * d + kh * d + i];
    qs += qv[r] * qv[r];
    ks += kv[r] * kv[r];
  }
  qs = rsqrtf(WaveSum(qs) + eps);
  ks = rsqrtf(WaveSum(ks) + eps);
  float* q_out = qn + static_cast<std::size_t>(row) * d;
  float* k_out = kn + static_cast<std::size_t>(row) * d;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    q_out[i] = qv[r] * qs;
    k_out[i] = kv[r] * ks;
  }
}

__global__ void GdnPrepKqKernel(const float* conv_out, float* scales,
                                std::uint32_t n_tokens, std::uint32_t k_heads,
                                std::uint32_t channels, float eps) {
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t kh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (t >= n_tokens || kh >= k_heads) {
    return;
  }
  const float* row = conv_out + static_cast<std::size_t>(t) * channels;
  const auto* q = reinterpret_cast<const float4*>(row + kh * d);
  const auto* k = reinterpret_cast<const float4*>(row + (k_heads + kh) * d);
  const float4 q4 = q[lane];
  const float4 k4 = k[lane];
  float qs = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
  float ks = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
  float kq = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
  for (unsigned offset = 16; offset > 0; offset >>= 1) {
    qs += __shfl_xor(qs, offset);
    ks += __shfl_xor(ks, offset);
    kq += __shfl_xor(kq, offset);
  }
  if (lane == 0) {
    float* dst = scales + (static_cast<std::size_t>(t) * k_heads + kh) * 3;
    dst[0] = rsqrtf(ks + eps);
    dst[1] = rsqrtf(qs + eps) * rsqrtf(static_cast<float>(d));
    dst[2] = kq;
  }
}

__global__ void GdnPrepAbKernel(const float* alpha_beta, const float* a,
                                const float* dt, float* ab, std::size_t count,
                                std::uint32_t v_heads) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::uint32_t h = i % v_heads;
  const std::size_t t = i / v_heads;
  const float alpha = alpha_beta[t * 2 * v_heads + h];
  const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
  ab[i * 2] = __expf(a[h] * SoftplusF(alpha + dt[h]));
  ab[i * 2 + 1] = SigmoidF(beta);
}

template<int kMask>
__device__ __forceinline__ float GdnXorAddDpp(float x) {
  const int y = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, x),
                                            0x160 | kMask, 0xF, 0xF, false);
  return x + __builtin_bit_cast(float, y);
}

__device__ __forceinline__ void GdnRowReduce(float& u, float& p) {
  u = GdnXorAddDpp<1>(u);
  p = GdnXorAddDpp<1>(p);
  u = GdnXorAddDpp<2>(u);
  p = GdnXorAddDpp<2>(p);
}

/// Qwen row-split recurrence specialized to the model's 128x128 state. Four
/// lanes own one row, eight rows share a wave, and two blocks cover a head.
/// The token loop is a short dependency chain that consumed its operands
/// straight from global memory (the conv output is far larger than the
/// caches), so every step paid a DRAM latency; the block now stages four
/// tokens at a time through LDS and keeps three such windows of loads in
/// flight, one window ahead of the one being committed.
constexpr int kGdnWindow = 4;
constexpr int kGdnWindowsAhead = 3;
__launch_bounds__(256) __global__
    void GdnRowSplitKernel(const float* conv_out, const float* scales,
                           const float* ab, float* state, float* raw,
                           std::uint32_t n_tokens, std::uint32_t k_heads,
                           std::uint32_t v_heads) {
  constexpr int d = kGdnDim;
  constexpr int kKeysPerLane = 32;
  constexpr int kVec = kKeysPerLane / 4;
  constexpr int kLanesPerRow = d / kKeysPerLane;
  constexpr int kRowsPerWave = 32 / kLanesPerRow;
  constexpr int kRowsPerBlock = kRowsPerWave * 8;
  constexpr int kW = kGdnWindow;
  static_assert(kW * (d / 4) * 2 == 256, "one q or k float4 per thread");
  static_assert(kW * kRowsPerBlock == 256, "one v element per thread");
  static_assert(kW * 5 <= 256, "scales and decays per window");
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int segment = lane % kLanesPerRow;
  const int row_group = lane / kLanesPerRow;
  const int row_local = (tid >> 5) * kRowsPerWave + row_group;
  const int row = blockIdx.x * kRowsPerBlock + row_local;
  const int kh = h % k_heads;
  const int vec0 = segment * kVec;
  const int channels = 2 * k_heads * d + v_heads * d;

  // Two window slots: q and k rows, the block's v rows, the three scales
  // and two decays per token.
  __shared__ float4 s_qk[2][2][kW][d / 4];
  __shared__ float s_v[2][kW][kRowsPerBlock];
  __shared__ float s_misc[2][kW][5];

  float* state_row = state + (static_cast<std::size_t>(h) * d + row) * d;
  auto* state4 = reinterpret_cast<float4*>(state_row);
  float4 s[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    s[i] = state4[vec0 + i];
  }

  // Window loads: thread tid takes q (tid < 128) or k float4 (tid / 32 of
  // token (tid / 32) % kW ... laid out so a token's 128 floats are 32
  // consecutive threads), one v element and, for tid < 5 kW, one scale.
  const int qk_which = tid >> 7;      // 0: q, 1: k
  const int qk_tok = (tid >> 5) & 3;  // token in the window
  const int qk_vec = tid & 31;
  const int v_tok = tid >> 6;
  const int v_row = tid & 63;
  const int m_tok = tid / 5;
  const int m_idx = tid % 5;
  const float* qk_src =
      conv_out + (qk_which == 0 ? kh : k_heads + kh) * d + (qk_vec * 4);
  const float* v_src =
      conv_out + 2 * k_heads * d + h * d + blockIdx.x * kRowsPerBlock + v_row;
  const float* m_src =
      m_idx < 3 ? scales + kh * 3 + m_idx : ab + h * 2 + (m_idx - 3);
  const std::size_t m_stride = m_idx < 3 ? k_heads * 3 : v_heads * 2;

  // Three windows of loads in flight, as named registers (a ring array
  // lands in scratch).
  struct Window {
    float4 qk;
    float v;
    float m;
  };
  Window r0;
  Window r1;
  Window r2;
  const auto load_window = [&](int w, Window& r) {
    const int last = static_cast<int>(n_tokens) - 1;
    const int t_qk = min((w * kW) + qk_tok, last);
    const int t_v = min((w * kW) + v_tok, last);
    r.qk = *reinterpret_cast<const float4*>(
        qk_src + static_cast<std::size_t>(t_qk) * channels);
    r.v = v_src[static_cast<std::size_t>(t_v) * channels];
    if (tid < 5 * kW) {
      const int t_m = min((w * kW) + m_tok, last);
      r.m = m_src[static_cast<std::size_t>(t_m) * m_stride];
    }
  };
  // A token's 32 float4 are stored [i][segment] so the four segments of
  // a row's lanes read adjacent 16-byte chunks (row-major, they were 128
  // bytes apart: a four-way bank conflict on every fragment).
  const int qk_slot = ((qk_vec % kVec) * kLanesPerRow) + (qk_vec / kVec);
  const auto commit_window = [&](const Window& r, int lds) {
    s_qk[lds][qk_which][qk_tok][qk_slot] = r.qk;
    s_v[lds][v_tok][v_row] = r.v;
    if (tid < 5 * kW) {
      s_misc[lds][m_tok][m_idx] = r.m;
    }
  };

  const int n_windows = (static_cast<int>(n_tokens) + kW - 1) / kW;
  load_window(0, r0);
  if (n_windows > 1) {
    load_window(1, r1);
  }
  if (n_windows > 2) {
    load_window(2, r2);
  }
  commit_window(r0, 0);
  __syncthreads();
  float* out_base = raw + h * d;
  // `cur` held window w (committed already) and takes window w + 3;
  // `next` holds window w + 1, committed after this window's tokens.
  const auto window = [&](int w, Window& cur, const Window& next) {
    if (w >= n_windows) {
      return;
    }
    const int lds = w & 1;
    if (w + kGdnWindowsAhead < n_windows) {
      load_window(w + kGdnWindowsAhead, cur);
    }
    const int t_end = min(kW, static_cast<int>(n_tokens) - (w * kW));
    for (int tl = 0; tl < t_end; ++tl) {
      const float4* q4 = s_qk[lds][0][tl];
      const float4* k4 = s_qk[lds][1][tl];
      const float decay = s_misc[lds][tl][3];
      float u = 0.0F;
      float p = 0.0F;
      float4 kc[kVec];
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        const float4 qv = q4[(i * kLanesPerRow) + segment];
        kc[i] = k4[(i * kLanesPerRow) + segment];
        s[i].x *= decay;
        s[i].y *= decay;
        s[i].z *= decay;
        s[i].w *= decay;
        u += s[i].x * kc[i].x + s[i].y * kc[i].y + s[i].z * kc[i].z +
             s[i].w * kc[i].w;
        p += s[i].x * qv.x + s[i].y * qv.y + s[i].z * qv.z + s[i].w * qv.w;
      }
      GdnRowReduce(u, p);
      const float inv_k = s_misc[lds][tl][0];
      const float q_scale = s_misc[lds][tl][1];
      const float delta =
          (s_v[lds][tl][row_local] - u * inv_k) * s_misc[lds][tl][4];
      if (segment == 0) {
        out_base[row] =
            p * q_scale + delta * inv_k * q_scale * s_misc[lds][tl][2];
      }
      const float update = delta * inv_k;
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        s[i].x += update * kc[i].x;
        s[i].y += update * kc[i].y;
        s[i].z += update * kc[i].z;
        s[i].w += update * kc[i].w;
      }
      out_base += v_heads * d;
    }
    // The other slot was last read one window ago (before the previous
    // barrier), so window w + 1 goes in without a second barrier.
    if (w + 1 < n_windows) {
      commit_window(next, lds ^ 1);
    }
    __syncthreads();
  };
  for (int w = 0; w < n_windows; w += kGdnWindowsAhead) {
    window(w, r0, r1);
    window(w + 1, r1, r2);
    window(w + 2, r2, r0);
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    state4[vec0 + i] = s[i];
  }
}

/// grid (value heads, row groups): each block owns kGdnRowsPerBlock state
/// rows of one head, kGdnLanes threads per row, each lane a contiguous
/// slice of the key dimension. Rows of the delta rule are independent, so
/// splitting a head over blocks only re-reads its q/k. The token loop is
/// serial; every reduction stays inside a lane group, so the loop runs
/// barrier-free. The raw attention rows go out unnormalized;
/// GdnEpilogueKernel finishes them.
constexpr unsigned kGdnLanes = 4;
constexpr unsigned kGdnRowsPerBlock = 32;

// The original vector kernel rounds error*key and then beta*correction for
// the last element of each lane, fusing the state decay into that result.
// HIP's FP intrinsics alone still permit reassociation of this three-product
// expression under fast-math. Keep its actual instruction sequence explicit.
__device__ __forceinline__ float GdnLastUpdate(float state, float decay,
                                               float error, float key,
                                               float beta) {
  float value;
  asm volatile(
      "v_mul_f32 %0, %1, %2\n\t"
      "v_mul_f32 %0, %0, %3\n\t"
      "v_fma_f32 %0, %4, %5, %0"
      : "=&v"(value)
      : "v"(error), "v"(key), "v"(beta), "v"(state), "v"(decay));
  return value;
}

template<bool kBatch>
__global__ void GdnKernel(const float* conv_out, const float* qn,
                          const float* kn, const float* alpha_beta,
                          const float* a, const float* dt, float* state,
                          float* raw, RollbackRows snapshots,
                          std::uint32_t n_tokens, std::uint32_t k_heads,
                          std::uint32_t v_heads, const GdnBatchItem* batch,
                          std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    conv_out = item.conv_scratch;
    qn = item.qn;
    kn = item.kn;
    alpha_beta = item.alpha_beta;
    state = item.state;
    raw = item.raw;
    n_tokens = item.n_tokens;
  }
  constexpr std::uint32_t d = kGdnDim;
  constexpr std::uint32_t slice = d / kGdnLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t kh = h % k_heads;
  const std::uint32_t j =
      blockIdx.y * kGdnRowsPerBlock + threadIdx.x / kGdnLanes;
  const std::uint32_t lane = threadIdx.x % kGdnLanes;
  const std::uint32_t i0 = lane * slice;
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const float a_h = a[h];
  const float dt_h = dt[h];
  float* S = state + static_cast<std::size_t>(h) * d * d + j * d + i0;
  float row[slice];
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    row[i] = S[i];
  }
  const float q_scale = rsqrtf(static_cast<float>(d));
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* q = qn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float* k = kn + (static_cast<std::size_t>(t) * k_heads + kh) * d + i0;
    const float vv = conv_out[static_cast<std::size_t>(t) * channels +
                              2 * k_heads * d + h * d + j];
    const float alpha = alpha_beta[t * 2 * v_heads + h];
    const float beta = alpha_beta[t * 2 * v_heads + v_heads + h];
    const float decay = __expf(a_h * SoftplusF(alpha + dt_h));
    const float b = SigmoidF(beta);
    float kr[slice];
    float u = 0.0f;
    const float last = row[slice - 1];
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      kr[i] = k[i];
      row[i] *= decay;
      u += row[i] * kr[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      u += __shfl_xor(u, off, kGdnLanes);
    }
    const float error = vv - u;
    float acc = 0.0f;
#pragma unroll
    for (std::uint32_t i = 0; i < slice; ++i) {
      // Preserve the original kernel's contraction: round error*key, then
      // fuse beta*correction into the decayed state. Materializing error*beta
      // for rollback would change this order under fast-math.
      const float correction = __fmul_rn(error, kr[i]);
      row[i] = i + 1 == slice ? GdnLastUpdate(last, decay, error, kr[i], b)
                              : __fmaf_rn(b, correction, row[i]);
      acc += row[i] * q[i];
    }
#pragma unroll
    for (unsigned off = kGdnLanes / 2; off > 0; off >>= 1) {
      acc += __shfl_xor(acc, off, kGdnLanes);
    }
    if (lane == 0) {
      raw[static_cast<std::size_t>(t) * v_heads * d + h * d + j] =
          acc * q_scale;
    }
    if ((kBatch ? batch[blockIdx.z].state_snapshots.rows[0]
                : snapshots.rows[0]) != nullptr &&
        t + 1 < n_tokens) {
      float* snap = kBatch ? batch[blockIdx.z].state_snapshots.rows[t]
                           : snapshots.rows[t];
      if (t == 0) {
        snap += static_cast<std::size_t>(h) * d * d + j * d + i0;
#pragma unroll
        for (std::uint32_t i = 0; i < slice; ++i)
          snap[i] = row[i];
      } else {
        if (j == 0 && h < k_heads) {
#pragma unroll
          for (std::uint32_t i = 0; i < slice; ++i)
            snap[h * d + i0 + i] = kr[i];
        }
        if (lane == 0) {
          if (j == 0) {
            snap[k_heads * d + h] = decay;
            snap[k_heads * d + v_heads + h] = b;
          }
          snap[k_heads * d + 2 * v_heads + h * d + j] = error;
        }
      }
    }
  }
#pragma unroll
  for (std::uint32_t i = 0; i < slice; ++i) {
    S[i] = row[i];
  }
}

__global__ void RestoreGdnStateKernel(float* state, RollbackRows snapshots,
                                      std::uint32_t keep, std::uint32_t k_heads,
                                      std::uint32_t v_heads) {
  constexpr std::uint32_t d = kGdnDim;
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i >= std::size_t{v_heads} * d * d)
    return;
  const auto h = i / (d * d);
  const auto j = (i / d) % d;
  const auto k = i % d;
  float value = snapshots.rows[0][i];
  for (std::uint32_t t = 1; t < keep; ++t) {
    const auto* update = snapshots.rows[t];
    const float key = update[(h % k_heads) * d + k];
    const float decay = update[k_heads * d + h];
    const float beta = update[k_heads * d + v_heads + h];
    const float error = update[k_heads * d + 2 * v_heads + h * d + j];
    const float correction = __fmul_rn(error, key);
    value = (k + 1) % (d / kGdnLanes) == 0
                ? GdnLastUpdate(value, decay, error, key, beta)
                : __fmaf_rn(beta, correction, __fmul_rn(value, decay));
  }
  state[i] = value;
}

/// Per-head RMSNorm of the raw attention rows and the sigmoid output gate,
/// one wave per (token, head) row.
/// A non-null `out_q8` receives the row quantized into the tiled Q8 layout
/// of the ssm_out projection (K = v_heads * d) in place of the F32 row:
/// each wave-wide slice of 32 lanes is one K block.
template<bool kBatch>
__global__ void GdnEpilogueKernel(const float* raw, const float* z,
                                  std::uint32_t z_stride, const float* norm_w,
                                  float* out, void* out_q8, __half* out_half,
                                  std::uint32_t n_rows, std::uint32_t v_heads,
                                  float eps, const GdnBatchItem* batch,
                                  std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    raw = item.raw;
    z = item.z;
    out = item.out;
    n_rows = item.n_tokens * v_heads;
  }
  constexpr std::uint32_t d = kGdnDim;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::size_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= n_rows) {
    return;
  }
  // z rows are [t][v_heads*d] with a caller-side row stride.
  const float* zrow = z + (row / v_heads) * z_stride + (row % v_heads) * d;
  const std::uint32_t per_lane = d / warpSize;
  const float* src = raw + row * d;
  float v[d / 32];
  float ss = 0.0f;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    v[r] = src[r * warpSize + lane];
    ss += v[r] * v[r];
  }
  const float scale = rsqrtf(WaveSum(ss) / static_cast<float>(d) + eps);
  if (out_q8 == nullptr) {
#pragma unroll
    for (std::uint32_t r = 0; r < per_lane; ++r) {
      const std::uint32_t i = r * warpSize + lane;
      const float value = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
      if (out_half != nullptr)
        out_half[row * d + i] = __float2half_rn(value);
      else
        out[row * d + i] = value;
    }
    return;
  }
  const std::size_t tok = row / v_heads;
  const std::size_t head = row % v_heads;
  const std::size_t num_blocks = (static_cast<std::size_t>(v_heads) * d) / 32;
  const std::size_t tl = tok % kQ8ActTileTokens;
#pragma unroll
  for (std::uint32_t r = 0; r < per_lane; ++r) {
    const std::uint32_t i = r * warpSize + lane;
    const float n = v[r] * scale * norm_w[i] * SigmoidF(zrow[i]);
    float max_abs = fabsf(n);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float dq = max_abs / 127.0F;
    const float id = (dq != 0.0F) ? (1.0F / dq) : 0.0F;
    const auto q = static_cast<std::int8_t>(roundf(n * id));
    const std::size_t kb = ((head * d) + (r * warpSize)) / 32;
    std::int8_t* tile =
        Q8ActTile(out_q8, num_blocks, tok / kQ8ActTileTokens, kb);
    tile[((lane >> 4u) * 256) + (tl * 16) + (lane & 15u)] = q;
    if (lane == 0) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                (tl * sizeof(float))) = dq;
    }
  }
}

/// Row j of the rolling state after token t is row (t + 1 + j) of the
/// concatenation [history ; rows], for any history depth `hist`.
__global__ void RollingSnapshotKernel(
    const float* rows, std::uint32_t row_stride, const float* history,
    RollbackRows snapshots, std::uint32_t n_tokens, std::uint32_t channels,
    std::uint32_t hist) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t per_token = static_cast<std::size_t>(hist) * channels;
  if (idx >= per_token * n_tokens) {
    return;
  }
  const std::uint32_t t = idx / per_token;
  const std::uint32_t j = (idx % per_token) / channels;
  const std::uint32_t c = idx % channels;
  const std::uint32_t src = t + 1 + j;
  snapshots.rows[t][idx % per_token] =
      src < hist ? history[static_cast<std::size_t>(src) * channels + c]
                 : rows[static_cast<std::size_t>(src - hist) * row_stride + c];
}

/// Causal conv over the chunk with the rolling state, SiLU applied.
__global__ void SsmConvKernel(const float* qkv, std::uint32_t qkv_stride,
                              const float* w, const float* conv_state,
                              float* out, std::uint32_t n_tokens,
                              std::uint32_t channels, std::uint32_t kernel) {
  const std::size_t idx =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_tokens) * channels) {
    return;
  }
  const std::uint32_t t = idx / channels;
  const std::uint32_t c = idx % channels;
  float acc = 0.0f;
  for (std::uint32_t k = 0; k < kernel; ++k) {
    const std::int32_t src_t = static_cast<std::int32_t>(t) -
                               static_cast<std::int32_t>(kernel - 1 - k);
    const float v =
        src_t >= 0 ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
                   : conv_state[static_cast<std::size_t>(kernel - 1 + src_t) *
                                    channels +
                                c];
    acc += w[static_cast<std::size_t>(c) * kernel + k] * v;
  }
  out[idx] = SiluF(acc);
}

/// SsmConvKernel with one thread per (channel, kTokensPerThread tokens) for
/// the 4-tap kernel: the taps stay in one float4 and the window slides in
/// registers, so a token costs one load and one store instead of eight
/// loads. Lanes run along channels, so every access is a contiguous row.
constexpr std::uint32_t kSsmConvTaps = 4;
constexpr std::uint32_t kSsmConvTokensPerThread = 8;
template<bool kSaveHistory, bool kBatch>
__global__ void SsmConv4Kernel(const float* qkv, std::uint32_t qkv_stride,
                               const float* w, float* conv_state, float* out,
                               std::uint32_t n_tokens, std::uint32_t channels,
                               RollbackRows snapshots,
                               const GdnBatchItem* batch,
                               std::uint32_t active) {
  if constexpr (kBatch) {
    if ((active & (1U << blockIdx.z)) == 0)
      return;
    const auto& item = batch[blockIdx.z];
    qkv = item.qkv;
    conv_state = item.conv_state;
    out = item.conv_scratch;
    n_tokens = item.n_tokens;
  }
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  const std::uint32_t t0 = blockIdx.y * kSsmConvTokensPerThread;
  const float4 taps = *reinterpret_cast<const float4*>(w + c * kSsmConvTaps);
  // window[i] holds token t0 - 3 + i.
  float window[kSsmConvTaps - 1];
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTaps - 1; ++i) {
    const std::int32_t src_t =
        static_cast<std::int32_t>(t0) - 3 + static_cast<std::int32_t>(i);
    window[i] =
        src_t >= 0
            ? qkv[static_cast<std::size_t>(src_t) * qkv_stride + c]
            : conv_state[static_cast<std::size_t>(kSsmConvTaps - 1 + src_t) *
                             channels +
                         c];
  }
#pragma unroll
  for (std::uint32_t i = 0; i < kSsmConvTokensPerThread; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= n_tokens) {
      break;
    }
    const float v = qkv[static_cast<std::size_t>(t) * qkv_stride + c];
    const float acc = taps.x * window[0] + taps.y * window[1] +
                      taps.z * window[2] + taps.w * v;
    out[static_cast<std::size_t>(t) * channels + c] = SiluF(acc);
    window[0] = window[1];
    window[1] = window[2];
    window[2] = v;
    if constexpr (kSaveHistory) {
      if ((kBatch ? batch[blockIdx.z].conv_snapshots.rows[0]
                  : snapshots.rows[0]) != nullptr &&
          t + 1 < n_tokens) {
        float* snapshot = kBatch ? batch[blockIdx.z].conv_snapshots.rows[t]
                                 : snapshots.rows[t];
#pragma unroll
        for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
          snapshot[std::size_t{j} * channels + c] = window[j];
      }
    }
  }
  if constexpr (kSaveHistory) {
    // Only one token tile may update history: each channel then has one
    // owner, which has already consumed all three original history values.
#pragma unroll
    for (std::uint32_t j = 0; j < kSsmConvTaps - 1; ++j)
      conv_state[std::size_t{j} * channels + c] = window[j];
  }
}

__global__ void UnpackQGateKernel(const float* qg, std::uint32_t qg_stride,
                                  float* q, float* gate, float* k, float* v,
                                  std::uint32_t heads, std::uint32_t d,
                                  std::uint32_t kv_width) {
  const std::uint32_t t = blockIdx.x;
  const std::size_t width = static_cast<std::size_t>(heads) * d;
  const float* row = qg + static_cast<std::size_t>(t) * qg_stride;
  for (std::size_t i = threadIdx.x; i < width; i += blockDim.x) {
    const std::uint32_t h = i / d;
    const std::uint32_t j = i % d;
    q[t * width + i] = row[h * 2 * d + j];
    gate[t * width + i] = row[h * 2 * d + d + j];
  }
  // A stacked [q|gate ; k ; v] projection carries k and v after the heads.
  if (k != nullptr) {
    for (std::size_t i = threadIdx.x; i < kv_width; i += blockDim.x) {
      k[t * kv_width + i] = row[2 * width + i];
      v[t * kv_width + i] = row[2 * width + kv_width + i];
    }
  }
}

// Keep each head's original eight-wave reduction. Heads in a block share
// rotary angles; normalization statistics remain independent.
template<std::uint32_t kHeadsPerBlock>
__global__ void PrepareAttentionKernel(
    const float* __restrict__ packed, std::uint32_t stride,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float* __restrict__ q, float* __restrict__ gate,
    __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
    std::uint32_t rotary, const std::uint32_t* start_pos, float theta,
    float eps, const qwen::vision::DeviceRope* rope) {
  __shared__ float partial[kHeadsPerBlock][8];
  __shared__ float norm[kHeadsPerBlock][256];
  const std::uint32_t t = blockIdx.x, first = blockIdx.y * kHeadsPerBlock,
                      tid = threadIdx.x;
  const std::uint32_t lane = tid % 32, wave = tid / 32;
  const std::uint32_t live = min(kHeadsPerBlock, heads + kv_heads - first);
  const std::size_t width = std::size_t(heads) * d,
                    kv_width = std::size_t(kv_heads) * d;
  const float* row = packed + std::size_t(t) * stride;
  float values[kHeadsPerBlock];
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    values[j] = j < live && tid < d
                    ? row[(query ? head * 2 * d : 2 * width + head * d) + tid]
                    : 0.0f;
    float ss = values[j] * values[j];
    // Match the separate RMS kernel's rounded square before its reduction.
    // Otherwise fast-math can contract it with the first shuffle addition.
    asm volatile("" : "+v"(ss));
    ss = WaveSum(ss);
    if (lane == 0)
      partial[j][wave] = ss;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    float total = 0.0f;
    for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w)
      total += partial[j][w];
    const float scale = rsqrtf(total / static_cast<float>(d) + eps);
    const float* gamma = query ? q_gamma : k_gamma;
    if (tid < d) {
      norm[j][tid] = values[j] * scale * (gamma != nullptr ? gamma[tid] : 1.0f);
      if (query)
        gate[std::size_t(t) * width + head * d + tid] =
            row[head * 2 * d + d + tid];
      else
        v_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(row[2 * width + kv_width + head * d + tid]);
    }
  }
  __syncthreads();
  const std::uint32_t half = rotary / 2;
  float sine = 0.0f, cosine = 0.0f;
  if (tid < half) {
    const float freq = powf(
        theta, -2.0f * static_cast<float>(tid) / static_cast<float>(rotary));
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, tid) * freq, &sine,
            &cosine);
  }
#pragma unroll
  for (std::uint32_t j = 0; j < kHeadsPerBlock; ++j) {
    if (j >= live)
      continue;
    const std::uint32_t h = first + j;
    const bool query = h < heads;
    const std::uint32_t head = query ? h : h - heads;
    if (tid < half) {
      const float a = norm[j][tid], b = norm[j][tid + half];
      const float lo = __fmaf_rn(a, cosine, -__fmul_rn(b, sine)),
                  hi = __fmaf_rn(a, sine, __fmul_rn(b, cosine));
      if (query) {
        q[std::size_t(t) * width + head * d + tid] = lo;
        q[std::size_t(t) * width + head * d + tid + half] = hi;
      } else {
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(lo);
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid +
                half] = __float2half(hi);
      }
    }
    if (tid >= rotary && tid < d) {
      if (query)
        q[std::size_t(t) * width + head * d + tid] = norm[j][tid];
      else
        k_cache[std::size_t(*start_pos + t) * kv_width + head * d + tid] =
            __float2half(norm[j][tid]);
    }
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim,
                           const std::uint32_t* start_pos, float theta,
                           const qwen::vision::DeviceRope* rope) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, *start_pos + t, i) * freq, &s, &c);
    const float a = v[i];
    const float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::uint32_t row_dim,
                              const std::uint32_t* start_pos) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(*start_pos + t) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
}

__global__ void StoreRowsKernel(const float* src, float* dst,
                                std::uint32_t row_dim,
                                const std::uint32_t* start_pos,
                                std::uint32_t capacity) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t row = (*start_pos + t) & (capacity - 1);
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    dst[static_cast<std::size_t>(row) * row_dim + i] =
        src[static_cast<std::size_t>(t) * row_dim + i];
  }
}

/// grid: the most blocks a batch can complete. Block b pools raw keys
/// [b*ratio, (b+1)*ratio) once every one of them is stored, i.e. for
/// b in [*first_block, (*start_pos + n_tokens) / ratio).
__global__ void PoolBlocksKernel(const float* raw, const float* gamma,
                                 __half* blocks,
                                 const std::uint32_t* first_block,
                                 const std::uint32_t* start_pos,
                                 std::uint32_t n_tokens, std::uint32_t ratio,
                                 std::uint32_t dim, std::uint32_t rotary_dim,
                                 float theta, float eps, std::uint32_t capacity,
                                 const qwen::vision::DeviceRope* rope) {
  __shared__ float v[256];
  __shared__ float shared[32];
  const std::uint32_t b = *first_block + blockIdx.x;
  if (b >= (*start_pos + n_tokens) / ratio) {
    return;
  }
  const std::uint32_t i = threadIdx.x;
  float mean = 0.0f;
  if (i < dim) {
    for (std::uint32_t r = 0; r < ratio; ++r) {
      const std::uint32_t row = (b * ratio + r) & (capacity - 1);
      mean += raw[static_cast<std::size_t>(row) * dim + i];
    }
    mean /= static_cast<float>(ratio);
  }
  const float ss = BlockSum(i < dim ? mean * mean : 0.0f, shared);
  const float scale = rsqrtf(ss / static_cast<float>(dim) + eps);
  if (i < dim) {
    v[i] = mean * scale * gamma[i];
  }
  __syncthreads();
  const std::uint32_t half = rotary_dim / 2;
  float outv = i < dim ? v[i] : 0.0f;
  if (i < rotary_dim) {
    const std::uint32_t p = i < half ? i : i - half;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(p) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, b * ratio, p) * freq, &s, &c);
    const float a = v[p];
    const float bb = v[p + half];
    outv = i < half ? a * c - bb * s : a * s + bb * c;
  }
  if (i < dim) {
    blocks[static_cast<std::size_t>(b) * dim + i] = __float2half_rn(outv);
  }
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all
// four heads. Preserve the original wave32 F32 accumulation and reduction
// tree: even tiny changes can exchange blocks at the top-k boundary.
constexpr std::uint32_t kSelectHeads = 4;
constexpr std::uint32_t kSelectDim = 128;

// F16 fragments are used by attention and projection kernels below.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

__global__ void SelectScoreKernel(const float* q, const __half* blocks,
                                  float* scores, std::uint32_t n_tokens,
                                  const std::uint32_t* start_pos,
                                  std::uint32_t first_token,
                                  std::uint32_t ratio, std::uint32_t budget,
                                  std::uint32_t max_blocks) {
#pragma clang fp reassociate(off)
  const auto t0 = blockIdx.x;
  const auto complete = (*start_pos + first_token + t0 + 1) / ratio;
  const auto b = blockIdx.y * blockDim.x + threadIdx.x;
  if (t0 >= n_tokens || complete <= budget || b >= complete)
    return;
  float key[kSelectDim];
#pragma unroll
  for (unsigned i = 0; i < kSelectDim; ++i)
    key[i] = __half2float(blocks[std::size_t{b} * kSelectDim + i]);
  float total = 0.0F;
#pragma unroll
  for (unsigned h = 0; h < kSelectHeads; ++h) {
    const auto* query = q + (std::size_t{t0} * kSelectHeads + h) * kSelectDim;
    float partial[32];
#pragma unroll
    for (unsigned lane = 0; lane < 32; ++lane) {
      float dot = 0.0F;
#pragma unroll
      for (unsigned i = 0; i < 4; ++i)
        dot = fmaf(query[lane + i * 32], key[lane + i * 32], dot);
      partial[lane] = dot;
      // Bound query-load hoisting: keeping all 128 scalar values live spills
      // registers on gfx1151. Scheduling groups retain every FMA and sum.
      if ((lane + 1) % 16 == 0)
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (unsigned delta = 16; delta; delta >>= 1) {
#pragma unroll
      for (unsigned lane = 0; lane < delta; ++lane)
        partial[lane] += partial[lane + delta];
    }
    total += fmaxf(partial[0], 0.0F);
  }
  scores[std::size_t{t0} * max_blocks + b] = total;
}

/// Locate a descending histogram rank cooperatively. Each thread owns
/// consecutive bins; state receives the bin, its remaining rank and count.
template<std::uint32_t kBinsPerThread>
__device__ void FindHistogramThreshold(const std::uint32_t* histogram,
                                       std::uint32_t wanted,
                                       std::uint32_t* wave_sums,
                                       std::uint32_t* state) {
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5;
  const std::uint32_t first =
      kThreads * kBinsPerThread - 1 - threadIdx.x * kBinsPerThread;
  std::uint32_t count = 0;
#pragma unroll
  for (std::uint32_t j = 0; j < kBinsPerThread; ++j)
    count += histogram[first - j];
  std::uint32_t inclusive = count;
#pragma unroll
  for (std::uint32_t distance = 1; distance < 32; distance *= 2) {
    const std::uint32_t other = __shfl_up(inclusive, distance);
    if (lane >= distance)
      inclusive += other;
  }
  if (lane == 31)
    wave_sums[wave] = inclusive;
  __syncthreads();
  std::uint32_t before = inclusive - count;
  for (std::uint32_t w = 0; w < wave; ++w)
    before += wave_sums[w];
  if (before < wanted && before + count >= wanted) {
    for (std::uint32_t j = 0; j < kBinsPerThread; ++j) {
      const std::uint32_t bin = first - j;
      const std::uint32_t bin_count = histogram[bin];
      if (before + bin_count >= wanted) {
        state[0] = bin;
        state[1] = wanted - before;
        state[2] = bin_count;
        break;
      }
      before += bin_count;
    }
  }
  __syncthreads();
}

/// Exact top-k over non-negative score bits. A sample chooses a histogram
/// window; a clipped threshold falls back to the full key range. At most
/// 4096 candidates are refined in LDS; larger bins use the original scores.
/// Every score participates, and the final mask keeps lowest-index ties.
__global__ void SelectMarkKernel(std::uint32_t* mask, const float* scores,
                                 const std::uint32_t* start_pos,
                                 std::uint32_t first_token, std::uint32_t ratio,
                                 std::uint32_t budget, std::uint32_t mask_words,
                                 std::uint32_t max_blocks) {
  __shared__ std::uint32_t candidates[4096];
  __shared__ std::uint32_t hist[256];
  __shared__ std::uint32_t candidate_count;
  __shared__ std::uint32_t wave_ties[kThreads / 32];
  __shared__ std::uint32_t state[3];  // threshold, remaining, bin count
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t pos = *start_pos + first_token + t;
  const std::uint32_t complete = (pos + 1) / ratio;
  std::uint32_t* words = mask + static_cast<std::size_t>(t) * mask_words;
  for (std::uint32_t w = threadIdx.x; w < mask_words; w += blockDim.x) {
    words[w] = complete <= budget ? 0xFFFFFFFFu : 0u;
  }
  if (complete <= budget) {
    return;
  }
  const auto* sc = reinterpret_cast<const std::uint32_t*>(
      scores + static_cast<std::size_t>(t) * max_blocks);
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t wave = threadIdx.x >> 5;
  // Short rows already fit the original histogram cheaply.
  const bool windowed = complete >= 4096;
  std::uint32_t maximum = 0;
  if (windowed) {
    maximum = threadIdx.x < min(complete, kThreads) ? sc[threadIdx.x] : 0u;
#pragma unroll
    for (unsigned offset = 16; offset > 0; offset /= 2)
      maximum = max(maximum, __shfl_xor(maximum, offset));
    if (lane == 0)
      wave_ties[wave] = maximum;
    __syncthreads();
    for (unsigned w = 0; w < kThreads / 32; ++w)
      maximum = max(maximum, wave_ties[w]);
  }
  // Finer bins reduce contention among scores with similar exponents.
  // The sample only places the window: clipped tails are still counted.
  unsigned prefix_bits = windowed ? 14 : 20;
  unsigned base =
      (maximum >> prefix_bits) > 2047 ? (maximum >> prefix_bits) - 2047 : 0;
  if (threadIdx.x == 0)
    candidate_count = 0;
  for (unsigned attempt = 0; attempt < 2; ++attempt) {
    for (unsigned i = threadIdx.x; i < 4096; i += blockDim.x)
      candidates[i] = 0;
    __syncthreads();
    const auto bin = [&](std::uint32_t value) {
      const auto key = value >> prefix_bits;
      return key > base ? min(key - base, 4095u) : 0u;
    };
    // Aligned production rows use four adjacent scores per lane.
    for (unsigned b = threadIdx.x * 4; b < complete; b += blockDim.x * 4) {
      if (b + 3 < complete && max_blocks % 4 == 0) {
        const uint4 v = *reinterpret_cast<const uint4*>(sc + b);
        atomicAdd(&candidates[bin(v.x)], 1u);
        atomicAdd(&candidates[bin(v.y)], 1u);
        atomicAdd(&candidates[bin(v.z)], 1u);
        atomicAdd(&candidates[bin(v.w)], 1u);
      } else {
        for (unsigned j = 0; j < 4 && b + j < complete; ++j)
          atomicAdd(&candidates[bin(sc[b + j])], 1u);
      }
    }
    __syncthreads();
    FindHistogramThreshold<16>(candidates, budget, wave_ties, state);
    if (!windowed || attempt != 0 ||
        (state[0] != 4095 && (state[0] != 0 || base == 0)))
      break;
    // A clipped threshold needs a histogram over the full 32-bit keys.
    prefix_bits = 20;
    base = 0;
  }
  std::uint32_t prefix = (base + state[0]) << prefix_bits;
  std::uint32_t remaining = state[1];
  // Reuse the first histogram as a bounded candidate list. If its bin is
  // larger, later radix passes read the original scores instead.
  const bool compact = state[2] <= 4096 && remaining != state[2];
  const std::uint32_t input_count = compact ? state[2] : complete;
  if (compact) {
    for (std::uint32_t b = threadIdx.x; b < complete; b += blockDim.x) {
      std::uint32_t v = sc[b];
      if ((v & (~0u << prefix_bits)) == prefix)
        candidates[atomicAdd(&candidate_count, 1u)] = v;
    }
    __syncthreads();
  }
  // A whole bin can be accepted as soon as its count fills the remaining
  // budget. Otherwise refine the remaining bits, at most eight per pass.
  for (unsigned bits = prefix_bits; bits > 0 && remaining != state[2];) {
    const unsigned digit_bits = min(bits, 8u);
    const std::uint32_t prefix_mask = ~0u << bits;
    bits -= digit_bits;
    const std::uint32_t radix_mask = (1u << digit_bits) - 1;
    hist[threadIdx.x] = 0;
    __syncthreads();
    for (std::uint32_t b = threadIdx.x; b < input_count; b += blockDim.x) {
      const std::uint32_t v = compact ? candidates[b] : sc[b];
      if ((v & prefix_mask) == prefix)
        atomicAdd(&hist[(v >> bits) & radix_mask], 1u);
    }
    __syncthreads();
    FindHistogramThreshold<1>(hist, remaining, wave_ties, state);
    prefix |= state[0] << bits;
    remaining = state[1];
    __syncthreads();
  }
  const std::uint32_t threshold = prefix;
  // Blocks strictly above the threshold are in; the first `remaining` ties
  // in index order fill the budget. Ties are ranked with a wave ballot and
  // a per-chunk scan over the eight waves.
  if (remaining == state[2]) {
    // Every threshold tie fits (in particular, a unique threshold). No
    // prefix scan is needed: each wave writes one complete mask word.
    for (std::uint32_t chunk = 0; chunk < complete; chunk += blockDim.x) {
      const std::uint32_t b = chunk + threadIdx.x;
      const bool selected = b < complete && sc[b] >= threshold;
      const auto bits = static_cast<std::uint32_t>(__ballot(selected));
      if (lane == 0 && b < complete) {
        words[b / 32] = bits;
      }
    }
    return;
  }
  std::uint32_t ties_before = 0;
  for (std::uint32_t chunk = 0; chunk < complete; chunk += blockDim.x) {
    const std::uint32_t b = chunk + threadIdx.x;
    const std::uint32_t v = b < complete ? sc[b] : 0u;
    const bool tie = b < complete && v == threshold;
    if (b < complete && v > threshold) {
      atomicOr(&words[b / 32], 1u << (b % 32));
    }
    const std::uint64_t ballot = __ballot(tie);
    if (lane == 0) {
      wave_ties[wave] = static_cast<std::uint32_t>(__popcll(ballot));
    }
    __syncthreads();
    std::uint32_t before = ties_before;
    for (std::uint32_t w = 0; w < wave; ++w) {
      before += wave_ties[w];
    }
    before += static_cast<std::uint32_t>(
        __popcll(ballot & ((std::uint64_t{1} << lane) - 1)));
    if (tie && before < remaining) {
      atomicOr(&words[b / 32], 1u << (b % 32));
    }
    std::uint32_t chunk_ties = 0;
    for (std::uint32_t w = 0; w < kThreads / 32; ++w) {
      chunk_ties += wave_ties[w];
    }
    ties_before += chunk_ties;
    __syncthreads();
  }
}

/// grid (heads, queries), block 256 = head dim. Keys stream in tiles of 256
/// positions: lane j scores key j of the tile, then lane d accumulates
/// value column d with online softmax rescaling. In the sparse window the
/// tiles are gathered from the query's selected blocks (compacted through
/// LDS in windows of 1024 blocks), so the sweep costs the budget rather than
/// the context.
/// grid.z splits the key tiles round-robin across `gridDim.z` blocks; with
/// more than one split each block writes (max, sum, unnormalized acc) to
/// `partials[(t * heads + h) * splits + z]` and AttentionMergeKernel
/// combines them, otherwise the normalized row goes straight to `out`.
__global__ void AttentionKernel(const float* q, const __half* k_cache,
                                const __half* v_cache,
                                const std::uint32_t* mask,
                                std::uint32_t mask_words, float* out,
                                float* partials, const std::uint32_t* start_pos,
                                std::uint32_t heads, std::uint32_t kv_heads,
                                std::uint32_t d, std::uint32_t ratio) {
  constexpr std::uint32_t kTile = 256;
  constexpr std::uint32_t kWindow = 1024;  // blocks compacted per pass
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t splits = gridDim.z;
  __shared__ float qs[256];
  __shared__ float p[kTile];
  __shared__ float shared[32];
  __shared__ std::uint32_t keys[kTile];
  __shared__ std::uint32_t list[kWindow];
  __shared__ std::uint32_t wave_total[8];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t pos = *start_pos + t;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t n_kv = pos + 1;
  const std::uint32_t tail_start = (n_kv / ratio) * ratio;
  const std::uint32_t* words =
      mask == nullptr ? nullptr
                      : mask + static_cast<std::size_t>(t) * mask_words;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5u;
  qs[i] = i < d ? q[(static_cast<std::size_t>(t) * heads + h) * d + i] : 0.0f;
  __syncthreads();
  float m = -INFINITY;
  float l = 0.0f;
  // Every wave accumulates its own 32 keys of each tile into a partial
  // context row (eight dims per lane); the partials are summed at the end.
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const std::size_t kv_stride = static_cast<std::size_t>(kv_heads) * d;
  const __half* k_head = k_cache + kvh * d;
  const __half* v_head = v_cache + kvh * d;

  // Loads eight contiguous halves as floats: one 16-byte load per lane, so a
  // wave reads a whole 256-wide row at once.
  const auto load8 = [&](const __half* row, float* out8) {
    const uint4 packed = *reinterpret_cast<const uint4*>(row + (lane * 8));
    const auto* h2 = reinterpret_cast<const __half2*>(&packed);
#pragma unroll
    for (std::uint32_t j = 0; j < 4; ++j) {
      const float2 f = __half22float2(h2[j]);
      out8[2 * j] = f.x;
      out8[2 * j + 1] = f.y;
    }
  };

  // One tile: lane i holds key position keys[i] (n_kv marks an empty lane).
  // Scores: wave w takes keys 32w..32w+31, a wave-wide dot product per key.
  // Values: the same split, eight dims per lane.
  const auto tile_step = [&]() {
    float qv[8];
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      qv[j] = qs[(lane * 8) + j];
    }
    float s_mine = -INFINITY;
    for (std::uint32_t kk = 0; kk < 32; ++kk) {
      const std::uint32_t slot = (wave * 32) + kk;
      const std::uint32_t j = keys[slot];
      float dot = 0.0f;
      if (j < n_kv) {
        float kv[8];
        load8(k_head + (static_cast<std::size_t>(j) * kv_stride), kv);
#pragma unroll
        for (std::uint32_t x = 0; x < 8; ++x) {
          dot += qv[x] * kv[x];
        }
      }
      dot = WaveSum(dot);
      if (lane == kk) {
        s_mine = j < n_kv ? dot * scale : -INFINITY;
      }
    }
    const float tile_max = BlockMax(s_mine, shared);
    const float m_new = fmaxf(m, tile_max);
    const float rescale = m == -INFINITY ? 0.0f : __expf(m - m_new);
    const float pj = s_mine == -INFINITY ? 0.0f : __expf(s_mine - m_new);
    p[i] = pj;
    const float tile_sum = BlockSum(pj, shared);
    l = l * rescale + tile_sum;
#pragma unroll
    for (std::uint32_t x = 0; x < 8; ++x) {
      acc[x] *= rescale;
    }
    m = m_new;
    __syncthreads();
    for (std::uint32_t kk = 0; kk < 32; ++kk) {
      const std::uint32_t slot = (wave * 32) + kk;
      const float w = p[slot];
      if (w != 0.0f) {
        float vv[8];
        load8(v_head + (static_cast<std::size_t>(keys[slot]) * kv_stride), vv);
#pragma unroll
        for (std::uint32_t x = 0; x < 8; ++x) {
          acc[x] += w * vv[x];
        }
      }
    }
    __syncthreads();
  };

  if (words == nullptr) {
    for (std::uint32_t tile = split * kTile; tile < n_kv;
         tile += kTile * splits) {
      keys[i] = tile + i < n_kv ? tile + i : n_kv;
      __syncthreads();
      tile_step();
    }
  } else {
    const std::uint32_t n_complete = tail_start / ratio;
    const std::uint32_t lane = i & 31u;
    const std::uint32_t wave = i >> 5u;
    for (std::uint32_t w0 = 0; w0 < n_complete; w0 += kWindow) {
      // Thread i owns blocks w0 + 4i .. +3 of the window: flag the selected
      // ones and compact their indices with a block-wide exclusive scan.
      std::uint32_t flags = 0;
      std::uint32_t count = 0;
      for (std::uint32_t k = 0; k < 4; ++k) {
        const std::uint32_t b = w0 + (i * 4) + k;
        const bool set = b < n_complete && ((words[b / 32] >> (b % 32)) & 1u);
        flags |= (set ? 1u : 0u) << k;
        count += set ? 1u : 0u;
      }
      std::uint32_t incl = count;
      for (std::uint32_t off = 1; off < 32; off <<= 1) {
        const std::uint32_t v = __shfl_up(incl, off);
        incl += lane >= off ? v : 0u;
      }
      if (lane == 31) {
        wave_total[wave] = incl;
      }
      __syncthreads();
      std::uint32_t base = 0;
      for (std::uint32_t w = 0; w < wave; ++w) {
        base += wave_total[w];
      }
      std::uint32_t total = 0;
      for (std::uint32_t w = 0; w < 8; ++w) {
        total += wave_total[w];
      }
      std::uint32_t slot = base + incl - count;
      for (std::uint32_t k = 0; k < 4; ++k) {
        if ((flags >> k) & 1u) {
          list[slot++] = w0 + (i * 4) + k;
        }
      }
      __syncthreads();
      for (std::uint32_t t0 = split * (kTile / ratio); t0 < total;
           t0 += (kTile / ratio) * splits) {
        const std::uint32_t entry = t0 + (i / ratio);
        keys[i] = entry < total ? (list[entry] * ratio) + (i % ratio) : n_kv;
        __syncthreads();
        tile_step();
      }
      __syncthreads();
    }
    // The incomplete tail block is always visible.
    if (tail_start < n_kv && split == 0) {
      keys[i] = tail_start + i < n_kv ? tail_start + i : n_kv;
      __syncthreads();
      tile_step();
    }
  }
  // Sum the eight wave partials: wave w's lane holds dims 8*lane..+7.
  __shared__ float red[8][256];
#pragma unroll
  for (std::uint32_t x = 0; x < 8; ++x) {
    red[wave][(lane * 8) + x] = acc[x];
  }
  __syncthreads();
  float total = 0.0f;
  for (std::uint32_t w = 0; w < 8; ++w) {
    total += red[w][i];
  }
  if (splits == 1) {
    if (i < d) {
      out[(static_cast<std::size_t>(t) * heads + h) * d + i] = total / l;
    }
    return;
  }
  float* part =
      partials +
      ((static_cast<std::size_t>(t) * heads + h) * splits + split) * (d + 2);
  if (i < d) {
    part[2 + i] = total;
  }
  if (i == 0) {
    part[0] = m;
    part[1] = l;
  }
}

/// grid (heads, queries): merges the split partials of one row with the
/// usual log-sum-exp rescaling.
__global__ void AttentionMergeKernel(const float* partials, float* out,
                                     std::uint32_t heads, std::uint32_t d,
                                     std::uint32_t splits) {
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const float* base =
      partials + (static_cast<std::size_t>(t) * heads + h) * splits * (d + 2);
  float m = -INFINITY;
  for (std::uint32_t s = 0; s < splits; ++s) {
    m = fmaxf(m, base[s * (d + 2)]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float* part = base + s * (d + 2);
    const float ms = part[0];
    const float scale = ms == -INFINITY ? 0.0f : __expf(ms - m);
    l += part[1] * scale;
    if (i < d) {
      acc += part[2 + i] * scale;
    }
  }
  if (i < d) {
    out[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / l;
  }
}

// Keep the original block-wide softmax reduction. Selection then stays in
// one wave's registers, retaining probability ordering and lowest-index ties.
template<unsigned MaxExperts>
__global__ void RouterTopKKernel(const float* logits, std::uint32_t stride,
                                 std::int32_t* ids, float* weights,
                                 std::uint32_t n_experts, std::uint32_t k) {
  __shared__ float probs[1024];
  __shared__ float shared[32];
  __shared__ std::uint32_t chosen[32];
  __shared__ float chosen_p[32];
  const std::uint32_t t = blockIdx.x;
  const float* src = logits + static_cast<std::size_t>(t) * stride;
  float local_max = -INFINITY;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, src[e]);
  }
  const float max_logit = BlockMax(local_max, shared);
  float local_sum = 0.0f;
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] = __expf(src[e] - max_logit);
    local_sum += probs[e];
  }
  const float denom = BlockSum(local_sum, shared);
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    probs[e] /= denom;
  }
  __syncthreads();

  if (threadIdx.x >= 32)
    return;
  constexpr unsigned Items = MaxExperts / 32;
  float values[Items];
#pragma unroll
  for (unsigned j = 0; j < Items; ++j) {
    unsigned e = threadIdx.x + j * 32;
    values[j] = e < n_experts ? probs[e] : -1.0f;
  }
  for (unsigned slot = 0; slot < k; ++slot) {
    float best = -1.0f;
    unsigned index = 0xffffffffu;
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      unsigned e = threadIdx.x + j * 32;
      if (values[j] > best) {
        best = values[j];
        index = e;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = __shfl_xor(best, offset);
      unsigned oi = __shfl_xor(index, offset);
      if (other > best || (other == best && oi < index)) {
        best = other;
        index = oi;
      }
    }
    if (threadIdx.x == 0) {
      chosen[slot] = index;
      chosen_p[slot] = best;
    }
#pragma unroll
    for (unsigned j = 0; j < Items; ++j) {
      if (threadIdx.x + j * 32 == index)
        values[j] = -1.0f;
    }
  }
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      sum += chosen_p[slot];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (std::uint32_t slot = 0; slot < k; ++slot) {
      ids[t * k + slot] = static_cast<std::int32_t>(chosen[slot]);
      weights[t * k + slot] = chosen_p[slot] / sum;
    }
  }
}

/// grid (tokens, dim chunks of kThreads).
__global__ void MoeEpilogueKernel(const float* expert_out, const float* weights,
                                  const float* shared_out, const float* gate,
                                  std::uint32_t gate_stride, float* out,
                                  std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    acc += weights[t * k + s] *
           expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  out[idx] = acc + SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]) *
                       shared_out[idx];
}

/// Four adjacent output lanes per thread: the same reduction over the top-k
/// slots with a quarter of the waves, so the per-wave issue overhead no
/// longer hides the streaming reads.
/// `ExpertT` is float, or __half when the routed down projection wrote its
/// rows as F16.
template<typename ExpertT>
__global__ void MoeEpilogueVec4Kernel(const ExpertT* expert_out,
                                      const float* weights,
                                      const float* shared_out,
                                      const float* gate,
                                      std::uint32_t gate_stride, float* out,
                                      std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (i >= dim) {
    return;
  }
  float4 acc{0.0F, 0.0F, 0.0F, 0.0F};
  const ExpertT* rows = expert_out + static_cast<std::size_t>(t) * k * dim + i;
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const float4 v = Load4(rows + static_cast<std::size_t>(s) * dim);
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  const std::size_t idx = static_cast<std::size_t>(t) * dim + i;
  const float g = SigmoidF(gate[static_cast<std::size_t>(t) * gate_stride]);
  const float4 sh = *reinterpret_cast<const float4*>(shared_out + idx);
  acc.x += g * sh.x;
  acc.y += g * sh.y;
  acc.z += g * sh.z;
  acc.w += g * sh.w;
  *reinterpret_cast<float4*>(out + idx) = acc;
}

/// dst[t] = row < 0 ? alt[t] : base[(row + t)]: the draft block's hidden
/// input, a kept trunk row or its own carried residual.
__global__ void MtpHiddenKernel(const float* base, const float* alt,
                                const std::int32_t* row, float* dst,
                                std::uint32_t width) {
  const std::uint32_t t = blockIdx.x;
  const float* src = *row < 0
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(*row) + t) * width;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i];
  }
}

__global__ void MtpAddEmbeddingKernel(const float* embedding, float* residual,
                                      std::uint32_t hidden,
                                      std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    residual[static_cast<std::size_t>(t) * streams * hidden + i] +=
        embedding[static_cast<std::size_t>(t) * hidden + i % hidden];
  }
}

__device__ __forceinline__ ArgmaxCandidate BetterCandidate(ArgmaxCandidate a,
                                                           ArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ ArgmaxCandidate ArgmaxBlock(ArgmaxCandidate best,
                                       ArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1) {
    best = BetterCandidate(
        best, {__shfl_xor(best.value, offset), __shfl_xor(best.index, offset)});
  }
  if (lane == 0) {
    shared[wave] = best;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w) {
      best = BetterCandidate(best, shared[w]);
    }
  }
  return best;
}

__global__ void ArgmaxPartialKernel(const float* logits,
                                    ArgmaxCandidate* partial,
                                    std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.y;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {src[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    partial[t * kArgmaxParts + blockIdx.x] = best;
  }
}

__global__ void ArgmaxFinishKernel(const float* logits,
                                   const ArgmaxCandidate* partial,
                                   std::int32_t* out, std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts) {
    best = partial[t * kArgmaxParts + threadIdx.x];
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    // std::max_element keeps element zero when it is NaN; later NaNs
    // never replace a finite candidate.
    out[t] =
        isnan(logits[static_cast<std::size_t>(t) * vocab]) ? 0 : best.index;
  }
}

constexpr unsigned kMtpCandidateTile = 1024;

__global__ void GatherArgmaxCandidatesKernel(const float* logits,
                                             const std::uint32_t* ids,
                                             ArgmaxCandidate* out,
                                             std::uint32_t rows,
                                             std::uint32_t vocab) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const auto id = ids[row];
    out[row] = id < vocab
                   ? ArgmaxCandidate{logits[std::size_t(row) * vocab + id],
                                     static_cast<std::int32_t>(id)}
                   : ArgmaxCandidate{NAN, -1};
  }
}

// A token excluded from its tile's top Keep cannot enter the global top Keep.
// Reduce tiles repeatedly, retaining the original order of equal scores.
// Only token IDs need to survive between passes; the original logit buffer
// supplies scores and preserves the sign of zero in the final output.
template<unsigned Keep>
__global__ void MtpCandidateTileKernel(const float* logits,
                                       const std::uint32_t* input,
                                       std::uint32_t* output, float* scores,
                                       std::uint32_t size,
                                       std::uint32_t vocab) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t i =
        std::size_t(blockIdx.x) * kMtpCandidateTile + threadIdx.x * 4 + j;
    const auto id = i < size ? (input != nullptr ? input[i] : i) : UINT32_MAX;
    ids[j] = static_cast<std::uint32_t>(id);
    const float value = id < vocab ? logits[id] : -INFINITY;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned rank = threadIdx.x * 4 + j;
    if (rank < min(Keep, size)) {
      output[blockIdx.x * Keep + rank] = ids[j];
      if (scores != nullptr) {
        const float value = ids[j] < vocab ? logits[ids[j]] : -INFINITY;
        scores[rank] = isfinite(value) ? value : -INFINITY;
      }
    }
  }
}

__global__ void CopyKernel(const float* src, float* dst, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// Masked prefill attention on the WMMA matrix cores, ported from the Qwen
// 27B route (src/models/qwen/hip/kernels/attention_wmma.hip) to this
// model's 24 x 256 query heads over two KV heads. Wave32 fragment layout: A
// holds row L%16 and 16 contiguous k, B holds column L%16 and 16 contiguous
// k, and C element i is row 2i + L/16, column L%16. Every block owns 32
// queries x 2 heads = 64 rows against one 16-key tile at a time; eight waves
// split as four S tiles x two halves of the head dimension, and the next
// tile's K/V is prefetched into registers behind the current tile's S,
// softmax and PV phases. V reaches LDS transposed with one key per lane so
// the writes stay conflict-free. The optional block mask follows the model's
// sparse selection: keys at or past the query's incomplete tail block are
// always visible, earlier blocks only when their bit is set.
constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 24;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
// Sixteen queries keep the causal-tail accumulator in registers.
constexpr std::uint32_t kWmmaQueryRows = 16;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaRatio = 4;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;
// Union-mask capacity: one bit per 4-token block, 262,144 tokens of context.
constexpr std::uint32_t kWmmaMaxMaskWords = 2048;

/// Two row layouts share the kernel. The dense window packs 16 queries x 2
/// heads (a row block is 16 queries of one head, grid.y over head pairs).
/// The sparse window packs four queries x twelve heads into three row
/// blocks without padding, with adjacent blocks covering both KV heads: the key
/// tiles are gathered from the union of the block's query selections and
/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads,
         bool kLateV = false>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group = 0) {
  // The launcher accepts only four-token selection blocks. Keep that
  // geometry constant throughout the key sweep.
  constexpr std::uint32_t ratio = kWmmaRatio;
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = kPackHeads
                                           ? (kQueryRows * kWmmaGqa + 15) / 16
                                           : (kQueryRows / 16) * kWmmaHeads;
  static_assert(!kPackHeads || kQueryRows * kWmmaGqa == 48,
                "four queries pack twelve heads into three tiles");
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / 2;
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 4;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == (kPackHeads ? 3 : 2), "two waves per S tile");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && (kPackHeads || kQueryRows % 16 == 0),
                "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 2 == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Keep the two KV heads of nearby queries together in the launch order.
  // This improves cache reuse without changing any query's key sweep.
  const std::uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
  const std::uint32_t query_group =
      first_query_group + (kPackHeads ? linear / kWmmaKvHeads : blockIdx.x);
  const std::uint32_t query_start = query_group * kQueryRows;
  const std::uint32_t kv_head =
      kPackHeads ? linear % kWmmaKvHeads : blockIdx.y / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      kPackHeads ? kv_head * kWmmaGqa
                 : (kv_head * kWmmaGqa) +
                       ((blockIdx.y % (kWmmaGqa / kWmmaHeads)) * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row (row block rb, row r) -> (local query, query head, live).
  const auto row_query = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? query_start + (rb * 16 + r) / kWmmaGqa
                      : query_start + ((rb / kWmmaHeads) * 16) + r;
  };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? first_query_head + (rb * 16 + r) % kWmmaGqa
                      : first_query_head + (rb % kWmmaHeads);
  };
  const auto row_live = [&](std::uint32_t rb, std::uint32_t r) {
    return row_query(rb, r) < n_tokens &&
           (!kPackHeads || rb * 16 + r < kQueryRows * kWmmaGqa);
  };

  constexpr std::uint32_t kKvLdsHalves =
      (kKeys * kWmmaKStride > kHeadDim * kVtStride) ? (kKeys * kWmmaKStride)
                                                    : (kHeadDim * kVtStride);
  __shared__ __half kv_lds[kKvLdsHalves];
  // Pad score rows for the four-lane softmax reads and probability rows
  // for the WMMA fragments. Both transposes otherwise repeat LDS banks.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  if (wave < 2 * kSTiles) {
    const std::uint32_t local_query = row_query(s_rb, sub);
    const bool live = row_live(s_rb, sub);
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(live ? row_head(s_rb, sub) : 0) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][2] = {};
  // The same four threads own each softmax row throughout the key sweep.
  // Keep its running statistics in registers; only the final denominator
  // and each tile's rescale factor need to cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Key tiles are gathered from `kBlocksPerTile` selection blocks (ratio
  // keys each). In the sparse window the blocks come from the union of this
  // block's query masks (every row still applies its own mask below), then
  // every block from the earliest row's tail on; the dense window walks the
  // blocks in order. The sweep therefore costs the union of the selected
  // windows, not the context.
  constexpr std::uint32_t kBlocksPerTile = kKeys / 4;
  static_assert(kBlocksPerTile == 4, "a tile is four selection blocks");
  const std::uint32_t n_blocks = (max_visible + ratio - 1) / ratio;
  const std::uint32_t tail_block =
      ((start_pos + query_start + 1) / ratio);  // first always-visible block
  constexpr bool sparse = kPackHeads;
  // Four 512-block selections plus their incomplete tail. The same LDS
  // stores a mask for arbitrary wider selections used by operator callers.
  constexpr unsigned kListCapacity = 4 * 512 + 4;
  __shared__ unsigned union_words[kListCapacity];
  __shared__ unsigned wave_counts[8];
  bool compact = false;
  unsigned selected = 0;
  if (sparse) {
    // Consecutive word ranges per thread make the prefix scan preserve
    // increasing block order, including ties and the always-visible tail.
    const unsigned live_rows = min(kQueryRows, n_tokens - query_start);
    const unsigned word_count = (n_blocks + 31) / 32;
    const unsigned words_per_thread = (word_count + 255) / 256;
    unsigned local_words[8];
    unsigned count = 0;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned w = tid * words_per_thread + j;
      unsigned bits = 0;
      if (j < words_per_thread && w < word_count) {
        for (unsigned r = 0; r < live_rows; ++r)
          bits |= mask[size_t(query_start + r) * mask_words + w];
        if (w * 32 >= tail_block)
          bits = ~0u;
        else if ((w + 1) * 32 > tail_block)
          bits |= ~0u << (tail_block % 32);
        if ((w + 1) * 32 > n_blocks)
          bits &= (1u << (n_blocks % 32)) - 1u;
      }
      local_words[j] = bits;
      count += __popc(bits);
    }
    unsigned scan = count;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset *= 2) {
      unsigned before = __shfl_up(scan, offset, 32);
      if (lane >= offset)
        scan += before;
    }
    if (lane == 31)
      wave_counts[wave] = scan;
    __syncthreads();
    unsigned prefix = scan - count;
#pragma unroll
    for (unsigned w = 0; w < 8; ++w) {
      if (w < wave)
        prefix += wave_counts[w];
      selected += wave_counts[w];
    }
    compact = selected <= kListCapacity;
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
      const unsigned word = tid * words_per_thread + j;
      unsigned bits = local_words[j];
      if (compact) {
        while (bits) {
          unsigned bit = __builtin_ctz(bits);
          union_words[prefix++] = word * 32 + bit;
          bits &= bits - 1;
        }
      } else if (j < words_per_thread && word < word_count) {
        union_words[word] = bits;
      }
    }
    __syncthreads();
  }
  // First visible block at or after `b` (n_blocks when none). Uniform across
  // the block: every thread walks the same words.
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if (!sparse) {
      return b;
    }
    while (b < tail_block && b < n_blocks) {
      const std::uint32_t bits = union_words[b / 32] >> (b % 32);
      // Never jump past the tail: from tail_block on every block is visible.
      if (bits != 0) {
        return min(b + static_cast<std::uint32_t>(__builtin_ctz(bits)),
                   tail_block);
      }
      b = min(((b / 32) + 1) * 32u, tail_block);
    }
    return b;
  };
  struct Tile {
    std::uint32_t block[kBlocksPerTile];
    std::uint32_t count;
  };
  // Gathers the next tile starting the search at block `b`; returns the
  // block to continue from.
  const auto gather_tile = [&](std::uint32_t b, Tile& tile) {
    if (compact) {
      tile.count = min(kBlocksPerTile, selected - min(b, selected));
#pragma unroll
      for (unsigned i = 0; i < kBlocksPerTile; ++i)
        tile.block[i] = b + i < selected ? union_words[b + i] : n_blocks;
      return b + tile.count;
    }
    tile.count = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kBlocksPerTile; ++i) {
      b = next_block(b);
      const bool have = b < n_blocks;
      tile.block[i] = have ? b : n_blocks;
      tile.count += have ? 1u : 0u;
      b = have ? b + 1 : b;
    }
    return b;
  };
  // Key position of tile row `r` (0..15), or context_end past the tile.
  const auto tile_key = [&](const Tile& tile, std::uint32_t r) {
    const std::uint32_t i = r / ratio;
    std::uint32_t blk = n_blocks;
#pragma unroll
    for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
      if (i == j) {
        blk = tile.block[j];
      }
    }
    return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
  };

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * kHeadDim);

  const auto load_v = [&](const Tile& tile, uint4* dst) {
    const std::uint32_t key_position = tile_key(tile, v_key);
    const bool live = key_position < context_end;
    const auto* src =
        v_base +
        (static_cast<std::size_t>(live ? key_position : 0) * kWmmaKvWidth);
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](const Tile& tile, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = tile_key(tile, idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      dst[n] =
          live ? *reinterpret_cast<const uint4*>(
                     k_base +
                     (static_cast<std::size_t>(key_position) * kWmmaKvWidth) +
                     d8)
               : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  uint4 v_cur[kVRegs];
  uint4 k_pre[kKRegs];
  uint4 v_pre[kVRegs];
  Tile cur;
  Tile pre;
  std::uint32_t cursor = gather_tile(0, cur);
  if (cur.count != 0) {
    load_k(cur, k_cur);
    load_v(cur, v_cur);
  }

  while (cur.count != 0) {
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next tile. Everything below covers its latency.
    cursor = gather_tile(cursor, pre);
    if (pre.count != 0) {
      load_k(pre, k_pre);
      if constexpr (!kLateV)
        load_v(pre, v_pre);
    }

    // --- S = Q K^T ---
    if (wave < 2 * kSTiles) {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot; the reader sums them.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // QK has finished reading K. V and softmax P use separate LDS, so
    // their writes can share the barrier at the end of softmax.
    {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
    }
    if constexpr (kLateV) {
      if (pre.count != 0)
        load_v(pre, v_pre);
    }

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      if (rg < kRows) {
        const std::uint32_t seg = tid % kSoftmaxLanes;
        constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
        const std::uint32_t rb = rg / 16;
        const std::uint32_t row = rg % 16;
        const std::uint32_t local_query = row_query(rb, row);
        const bool live_row = row_live(rb, row);
        const std::uint32_t absolute_query = start_pos + local_query;
        // Keys in the query's own incomplete block are always visible; earlier
        // blocks follow the selection mask.
        const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
        const std::uint32_t* words =
            mask == nullptr || !live_row
                ? nullptr
                : mask + static_cast<std::size_t>(local_query) * mask_words;
        float part_max = -INFINITY;
        float vals[kPerLane];
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const std::uint32_t col = (seg * kPerLane) + m;
          const std::uint32_t key_position = tile_key(cur, col);
          bool valid = live_row && key_position <= absolute_query &&
                       key_position < context_end;
          if (valid && words != nullptr && key_position < tail_start) {
            const std::uint32_t b = key_position / ratio;
            valid = ((words[b / 32] >> (b % 32)) & 1u) != 0u;
          }
          const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
          vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                             s_lds[1][tile][row][col % 16])
                          : -INFINITY;
          part_max = fmaxf(part_max, vals[m]);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_max = fmaxf(part_max, __shfl_xor(part_max, off));
        }
        const float prev_max = running_max;
        const float next_max = fmaxf(prev_max, part_max);
        const float prior_scale =
            isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
        float part_sum = 0.0F;
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m) {
          const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
          part_sum += w;
          p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
        }
#pragma unroll
        for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
          part_sum += __shfl_xor(part_sum, off);
        }
        running_max = next_max;
        running_sum = (running_sum * prior_scale) + part_sum;
        if (seg == 0) {
          row_scale[rg] = prior_scale;
        }
      }
    }
    __syncthreads();

    // --- rescale the running O by the new maximum. A lane touches only rows
    // 2i + half_id, so the factors are read once per row block.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }

    // --- O += P V ---
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] = LoadFrag(
            &kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          if constexpr (!kPackHeads) {
            const std::uint32_t first_key = cur.block[0] * ratio;
            // WMMA's dot accumulation can change its last bits when zero
            // probabilities multiply populated future values instead of the
            // zero padding at a chunk boundary. Accumulate each query's final
            // partial tile over its visible keys only. Earlier complete tiles
            // retain the matrix-core path.
            if (first_key + kKeys - 1 > start_pos + query_start) {
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const std::uint32_t row = (2 * i) + half_id;
                const std::uint32_t absolute_query =
                    start_pos + row_query(rb, row);
                if (absolute_query < first_key + kKeys - 1) {
                  float acc = o_acc[rb][t][i];
                  for (std::uint32_t key = 0;
                       key < kKeys && first_key + key <= absolute_query;
                       ++key) {
                    acc = fmaf(
                        __half2float(p_lds[(rb * 16) + row][key]),
                        __half2float(
                            kv_lds[((dim_tile * 16) + sub) * kVtStride + key]),
                        acc);
                  }
                  next[i] = acc;
                }
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
#pragma unroll
    for (std::uint32_t n = 0; n < kVRegs; ++n) {
      v_cur[n] = v_pre[n];
    }
    cur = pre;
  }
  if (tid < kRows * kSoftmaxLanes && tid % kSoftmaxLanes == 0)
    row_sum[tid / kSoftmaxLanes] = running_sum;
  __syncthreads();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        if (!row_live(rb, row)) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(row_query(rb, row)) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(row_head(rb, row)) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= SigmoidF(gate[offset]);
        }
        out[offset] = value;
      }
    }
  }
}

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

/// Four elements per lane, eight lanes per 32-wide K block, four blocks per
/// wave: per-block absmax scale, four codes stored as one word straight
/// into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledVec4Kernel(const float* __restrict__ x,
                                          void* __restrict__ y,
                                          std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 3u;
  const std::uint32_t lane8 = threadIdx.x & 7u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float4 v = *reinterpret_cast<const float4*>(x + (tok * k) + (blk * 32) +
                                                    (lane8 * 4));
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::uint32_t pos = lane8 * 4;  // 0, 4, ..., 28
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane8 == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// One wave per (token, 32-wide K block): per-block absmax scale, codes
/// written straight into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledKernel(const float* __restrict__ x,
                                      void* __restrict__ y, std::size_t batch,
                                      std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float val = x[(tok * k) + (blk * 32) + lane_id];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(val * id));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

// Preserve the separate scale/SiLU kernel's F32 intermediates before narrowing.
// Without these compiler boundaries, fused F16 output can round differently.
__device__ __forceinline__ float HcScaledSilu(float value) {
  float scaled = value * 0.25F;
  asm volatile("" : "+v"(scaled));
  float result = SiluF(scaled);
  asm volatile("" : "+v"(result));
  return result;
}

/// Two-dimensionally blocked W8A8 WMMA GEMM: block = BM rows x BN tokens, BK
/// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
/// Out-of-range rows and K blocks are clamped and their scale zeroed, so
/// they contribute exactly zero without divergence. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kHcDown = false>
__launch_bounds__(256) __global__ void W8A8BlockedWmmaGEMMKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    std::conditional_t<kHcDown, __half, float>* __restrict__ y,
    std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  // One LDS plane: codes and scales of both operands; the epilogue reuses
  // the first 16 KB to transpose 16 x 32 result tiles per wave.
  constexpr int kABytes = BK * kRowTiles * 32 * 16;
  constexpr int kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16;
  constexpr int kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  static_assert(kLdsBytes >= 8 * 512 * 4, "epilogue transposes 16 KB");
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<int32x4_t(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b =
      reinterpret_cast<int32x4_t(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes +
                                                          kDwBytes + kBBytes);

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
#if __clang_major__ >= 23
      // LLVM 23 hoists the next K block's operands above this one's WMMAs,
      // which spills registers on gfx1151. ROCm 7.2.3 (LLVM 22) does not, and
      // the fence costs it about 3%.
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  __syncthreads();
  constexpr unsigned kOutputStride = kHcDown ? 36 : 32;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(lds));
  float* tile_scratch =
      reinterpret_cast<float*>(lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row] over 16 tokens x 32 rows.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        if constexpr (kHcDown) {
          auto* dst = y + tok * m + r0 + static_cast<std::size_t>(row_l);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 v = src[q];
            *reinterpret_cast<__half2*>(dst + 4 * q) =
                __floats2half2_rn(HcScaledSilu(v.x), HcScaledSilu(v.y));
            *reinterpret_cast<__half2*>(dst + 4 * q + 2) =
                __floats2half2_rn(HcScaledSilu(v.z), HcScaledSilu(v.w));
          }
        } else {
          auto* dst = reinterpret_cast<float4*>(
              y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dst[q] = src[q];
          }
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            const float value =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
            if constexpr (kHcDown)
              y[(tok * m) + r] = __float2half_rn(HcScaledSilu(value));
            else
              y[(tok * m) + r] = value;
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k, const void* __restrict__ w_up) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          f_codes[u] = group == 0   ? code_cache[u][0]
                       : group == 1 ? code_cache[u][1]
                       : group == 2 ? code_cache[u][2]
                                    : code_cache[u][3];
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
          mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
               ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
          mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
               ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const __half2 sb =
            __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
          // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
          // of chunk 1; Q5_K adds bit j of the staged high-bit word.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8) && BN >= 48)) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            __half values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float product = base[idx + v + 4 * plane] * base[idx + v];
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(base[idx + v]);
              asm volatile("" : "+v"(value));
              values[v] = __float2half(value);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<__half2*>(out_half + offset) =
                  __halves2half2(values[0], values[1]);
            } else {
              out_half[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_half[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

}  // namespace

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     table, type, tokens, res, hidden, streams);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  // Every (row, group) pair is one block; the kernel recovers the group
  // from the block index to pick its gamma slice.
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows * groups), dim3(kThreads), 0,
                     stream, x, gamma, out, dim / groups, groups, eps);
}

std::uint32_t HcInjectParts(std::uint32_t hidden) {
  return Blocks(hidden);
}

std::uint32_t HcInjectPartsVec4(std::uint32_t hidden) {
  return Blocks((hidden + 3) / 4);
}

void HcMixEpilogue(const float* xn, const float* gate, const float* inject_w,
                   float* mixed, float* inject, std::uint32_t n_tokens,
                   std::uint32_t hidden, std::uint32_t streams,
                   hipStream_t stream) {
  hipLaunchKernelGGL(HcMixEpilogueKernel, dim3(n_tokens, Blocks(hidden)),
                     dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                     inject, hidden, streams);
}

void HcMixEpilogueVec4(const float* xn, const float* gate,
                       const float* inject_w, float* mixed, float* inject,
                       std::uint32_t n_tokens, std::uint32_t hidden,
                       std::uint32_t streams, hipStream_t stream) {
  if (streams != 4 || hidden % 4 != 0) {
    HcMixEpilogue(xn, gate, inject_w, mixed, inject, n_tokens, hidden, streams,
                  stream);
    return;
  }
  hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<float>,
                     dim3(n_tokens, HcInjectPartsVec4(hidden)), dim3(kThreads),
                     0, stream, xn, gate, inject_w, mixed, inject, hidden);
}

void HcMixEpilogueVec4F16(const __half* xn, const float* gate,
                          const float* inject_w, float* mixed,
                          __half* mixed_half, void* mixed_q8, float* inject,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          hipStream_t stream) {
  if (hidden % 32 == 0) {
    hipLaunchKernelGGL((HcMixEpilogueF16Kernel<>),
                       dim3(HcInjectPartsVec4(hidden), n_tokens),
                       dim3(kThreads), 0, stream, xn, gate, inject_w, mixed,
                       mixed_half, mixed_q8, inject, hidden);
    return;
  }
  hipLaunchKernelGGL(HcMixEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, HcInjectPartsVec4(hidden)), dim3(kThreads),
                     0, stream, xn, gate, inject_w, mixed, inject, hidden);
  if (mixed_half != nullptr) {
    NarrowActivations(mixed, mixed_half, false,
                      static_cast<std::size_t>(n_tokens) * hidden, stream);
  }
  if (mixed_q8 != nullptr) {
    QuantizeQ8Tiled(mixed, mixed_q8, n_tokens, hidden, stream);
  }
}

void HcCombine(float* res, const float* block_out, const float* inject,
               std::uint32_t inject_parts, const float* gamma, float* xn,
               std::uint32_t n_tokens, std::uint32_t hidden,
               std::uint32_t streams, float eps, hipStream_t stream,
               bool decode) {
  // One block per token wants a batch: a decode step keeps the
  // one-block-per-stream kernel's parallelism.
  if (!decode && n_tokens >= 16 && streams == 4 && hidden % 128 == 0 &&
      hidden <= 2560) {
    hipLaunchKernelGGL(HcCombineVec4Kernel<float>, dim3(n_tokens),
                       dim3(kThreads), 0, stream, res, block_out, inject,
                       inject_parts, gamma, xn, nullptr, hidden, eps);
    return;
  }
  hipLaunchKernelGGL(HcCombineKernel<float>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, nullptr, hidden, streams, eps);
}

bool HcCombineMoeF16(float* res, const __half* expert_out, const float* weights,
                     const float* shared_out, const float* gate,
                     std::uint32_t gate_stride, std::uint32_t used,
                     const float* inject, std::uint32_t inject_parts,
                     const float* gamma, __half* xn, void* xn_q8,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, float eps, hipStream_t stream) {
  if (streams != 4 || hidden % 128 != 0 || hidden > 3 * 4 * kThreads ||
      gamma == nullptr || xn_q8 == nullptr) {
    return false;
  }
  hipLaunchKernelGGL(HcCombineMoeF16Kernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, res, expert_out, weights, shared_out, gate,
                     gate_stride, used, inject, inject_parts, gamma, xn, xn_q8,
                     hidden, eps);
  return true;
}

void HcCombineF16(float* res, const float* block_out, const float* inject,
                  std::uint32_t inject_parts, const float* gamma, __half* xn,
                  void* xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                  std::uint32_t streams, float eps, hipStream_t stream) {
  // F16 prefill keeps the same reduction even for a one-token tail.
  if (streams == 4 && hidden % 128 == 0 && hidden <= 2560) {
    hipLaunchKernelGGL(HcCombineVec4Kernel<__half>, dim3(n_tokens),
                       dim3(kThreads), 0, stream, res, block_out, inject,
                       inject_parts, gamma, xn, xn_q8, hidden, eps);
    return;
  }
  hipLaunchKernelGGL(HcCombineKernel<__half>, dim3(n_tokens, streams),
                     dim3(kThreads), 0, stream, res, block_out, inject,
                     inject_parts, gamma, xn, xn_q8, hidden, streams, eps);
}

void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SiluScaleKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, scale, count);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, count);
}

void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluHalfKernel, dim3(Blocks((count + 3) / 4)),
                     dim3(kThreads), 0, stream, gate, up, out, count);
}

bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream) {
  if (k % 32 != 0) {
    return false;
  }
  const std::size_t chunks = n_rows * (k / 4);
  hipLaunchKernelGGL(SwigluQ8Kernel, dim3(Blocks(chunks)), dim3(kThreads), 0,
                     stream, gate, up, out_q8, n_rows, k);
  return true;
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(SigmoidMulKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count);
}

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

std::size_t Q8TiledBytes(std::size_t batch, std::size_t k) {
  // The GEMM stages whole 128-token macro tiles, so the buffer is sized to
  // the batch rounded up to one (the padding tiles are never read as
  // results, only staged).
  constexpr std::size_t kMacroTokens = 128;
  const std::size_t padded =
      (batch + kMacroTokens - 1) / kMacroTokens * kMacroTokens;
  return (padded / kQ8ActTileTokens) * (k / 32) * kQ8ActTileBytes;
}

void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream) {
  const std::size_t blocks = batch * (k / 32);
  if (batch * k >= 4096) {
    const std::size_t per_block = kThreads / 8;
    hipLaunchKernelGGL(QuantizeQ8TiledVec4Kernel,
                       dim3((blocks + per_block - 1) / per_block),
                       dim3(kThreads), 0, stream, x, out, batch, k);
    return;
  }
  const std::size_t waves = kThreads / 32;
  hipLaunchKernelGGL(QuantizeQ8TiledKernel, dim3((blocks + waves - 1) / waves),
                     dim3(kThreads), 0, stream, x, out, batch, k);
}

bool HcDownF16Gemm(const void* w, const void* x_tiled, __half* out,
                   std::uint32_t n_tokens, hipStream_t stream) {
  if (n_tokens < 96 || w == nullptr || x_tiled == nullptr || out == nullptr)
    return false;
  hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<64, 128, 4, 2, 4, true>),
                     dim3((n_tokens + 127) / 128, 5), dim3(kThreads), 0, stream,
                     w, x_tiled, out, n_tokens, 320, 10240);
  return true;
}

bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // Qwen's wave64 matrix kernel with four row groups improves the model's
  // large output projections while preserving every K32 accumulator update.
  if (batch >= 1024 && m == 2560 && k == 6144) {
    W8A8GemmWave64(w, x_tiled, out, batch, m, k, stream);
    return true;
  }
  // A 128-token macro tile is the throughput configuration; short chunks
  // would leave most of it idle and take the 64-token variant. A narrow
  // projection (the 320-row mixer down over K = 10240) gets 64-row tiles so
  // it still fills the device.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kNarrowBM, kBN, 4, 2, 4>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, batch,
                       m, k);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  }
  return true;
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  const std::size_t rows = RoutedCompactRows(slots, n_experts);
  (void)hipMemsetAsync(rows_token, 0xFF, rows * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(rows_slot, 0xFF, rows * sizeof(std::int32_t), stream);
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr);
        return true;
      }
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K) ? 256 : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    default:
      return false;
  }
}

template<int BN>
bool LaunchRoutedGatedF16(const void* gate, const void* up, WeightType type,
                          const __half* x, const std::int32_t* tiles,
                          std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, __half* out,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || k % 256 != 0 || n_tiles == 0 || out == nullptr) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ4_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ5_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up);
      return true;
    default:
      return false;
  }
}

bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  if (tile_rows == 128) {
    return LaunchRoutedGatedF16<128>(gate, up, type, x, tiles, n_tiles,
                                     pad_bounds, rows_in, rows_out, out, m, k,
                                     stream);
  }
  return tile_rows == 64 &&
         LaunchRoutedGatedF16<64>(gate, up, type, x, tiles, n_tiles, pad_bounds,
                                  rows_in, rows_out, out, m, k, stream);
}

// Keep the separate four-tap convolution's F32 rounding order when its
// inputs come from the projection's LDS tile.
__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}

constexpr unsigned kSsmProjectionTileTokens = 32;

// The fused projection leaves each 32-token tile's first and last three
// raw rows in qkv. Only the first three convolutions need another tile or
// the previous chunk's history; all other rows are produced in LDS.
__global__ void SsmConvBoundaryKernel(const float* qkv, const float* w,
                                      const float* history, float* out,
                                      std::uint32_t n_tokens,
                                      std::uint32_t channels,
                                      std::uint32_t stride) {
  const std::uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t t = blockIdx.y * kSsmProjectionTileTokens + blockIdx.z;
  if (c >= channels || t >= n_tokens) {
    return;
  }
  const float4 taps = *reinterpret_cast<const float4*>(w + c * 4);
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int src = static_cast<int>(t) - 3 + j;
    v[j] = src < 0 ? history[static_cast<std::size_t>(src + 3) * channels + c]
                   : qkv[static_cast<std::size_t>(src) * stride + c];
  }
  out[static_cast<std::size_t>(t) * channels + c] =
      SsmConv4Value(taps, v[0], v[1], v[2], v[3]);
}

// Optional output layout for the stacked QKV projection. The projection's
// 256-row tile covers one query, gate, key or value head.
struct AttentionProjectionOutput {
  const float* q_gamma;
  const float* k_gamma;
  float* query;
  float* gate;
  __half* keys;
  __half* values;
  const std::uint32_t* position;
  float theta;
  float eps;
  const qwen::vision::DeviceRope* rope;
};

/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. F16 weights skip decoding
/// and use the same ordered K16 products. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHcMix = false, bool kSsmConv = false, bool kAttention = false,
         bool kHalfWeights = false>
__launch_bounds__(256) __global__ void DenseF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const __half* xn = nullptr, __half* mixed_half = nullptr,
    void* mixed_q8 = nullptr, const float* conv_w = nullptr,
    float* conv_out = nullptr, AttentionProjectionOutput attention = {}) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * (kHalfWeights ? 64 : 34);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kHalfWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kHalfWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kHalfWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint4 v;
          __builtin_memcpy(&v, &h[4 * c], 16);
          s_a[kk][row][swizzle(row, c)] = v;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          if constexpr (kHalfWeights)
            acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
          else
            acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  if constexpr (kAttention) {
    static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 && WN == 1);
    static_assert(!kHcMix && !kSsmConv && kRowGroup == 1);
    constexpr unsigned stride = 36, dim = 256, width = 6144, kvwidth = 512;
    float* scratch = reinterpret_cast<float*>(s_lds);
    float* tile = scratch + wave_id * 16 * stride;
    const unsigned projection_head = r_block / 256;
    const bool query = projection_head < 48 && (projection_head % 2) == 0;
    const bool key = projection_head >= 48 && projection_head < 50;
    const bool gate = projection_head < 48 && (projection_head % 2) == 1;
    const unsigned head =
        projection_head < 48 ? projection_head / 2 : projection_head % 2;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
        tile[sub_lane * stride + 16 + 2 * l + half_id] = acc[1][j][l];
      }
      __syncthreads();
#pragma unroll
      for (unsigned phase = 0; phase < 2; ++phase) {
        // One wave handles a token, emulating its original eight-wave norm.
        const unsigned tok_local = wave_id * 2 + phase;
        const std::size_t tok = t_block + j * 16 + tok_local;
        if (tok < batch) {
          float v[8];
#pragma unroll
          for (unsigned c = 0; c < 8; ++c)
            v[c] = scratch[(c * 16 + tok_local) * stride + lane_id];
          if (query || key) {
            float total = 0.0F;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              // Match the separate norm: round each square and each wave sum
              // before accumulating the eight partials in their original order.
              float sq = v[c] * v[c];
              asm volatile("" : "+v"(sq));
              float ss = WaveSum(sq);
              asm volatile("" : "+v"(ss));
              total += ss;
              asm volatile("" : "+v"(total));
            }
            const float scale = rsqrtf(total / 256.0F + attention.eps);
            const float* gamma = query ? attention.q_gamma : attention.k_gamma;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              v[c] = v[c] * scale;
              asm volatile("" : "+v"(v[c]));
              v[c] = v[c] * gamma[c * 32 + lane_id];
              asm volatile("" : "+v"(v[c]));
            }
            const float freq = powf(
                attention.theta, -2.0F * static_cast<float>(lane_id) / 64.0F);
            float sn = 0.0F, cs = 0.0F;
            sincosf(qwen::vision::RopePosition(
                        attention.rope, *attention.position + tok, lane_id) *
                        freq,
                    &sn, &cs);
            const float lo = __fmaf_rn(v[0], cs, -__fmul_rn(v[1], sn));
            const float hi = __fmaf_rn(v[0], sn, __fmul_rn(v[1], cs));
            v[0] = lo;
            v[1] = hi;
          }
#pragma unroll
          for (unsigned c = 0; c < 8; ++c) {
            const unsigned col = c * 32 + lane_id;
            if (query)
              attention.query[tok * width + head * dim + col] = v[c];
            else if (gate)
              attention.gate[tok * width + head * dim + col] = v[c];
            else if (key)
              attention.keys[std::size_t(*attention.position + tok) * kvwidth +
                             head * dim + col] = __float2half_rn(v[c]);
            else
              attention
                  .values[std::size_t(*attention.position + tok) * kvwidth +
                          head * dim + col] = __float2half_rn(v[c]);
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (kHcMix) {
    static_assert(BM == 256 && BN == 128 && BK == 1 && WM == 4 && WN == 2);
    constexpr unsigned kHiddenTile = BM / 4;
    constexpr unsigned kStreamStride = kHiddenTile + 1;
    constexpr unsigned kPlane = 16 * kStreamStride;
    static_assert(4 * kPlane * sizeof(float) <= sizeof(s_lds));
    float* gates = reinterpret_cast<float*>(s_lds);
    const std::size_t hidden = m / 4;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int token_group = 0; token_group < WN; ++token_group) {
        if (wave_tok == token_group) {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              const unsigned row =
                  (wave_row * kWaveRowTiles + i) * 16 + 2 * l + half_id;
              gates[(row % 4) * kPlane + sub_lane * kStreamStride + row / 4] =
                  acc[i][j][l];
            }
          }
        }
        __syncthreads();
        // A half-wave writes all 64 hidden values of one token. The stream
        // planes are padded to avoid the gate transpose's LDS bank conflicts.
        const unsigned token_in_tile = tid / (kHiddenTile / 4);
        const unsigned h_local = (tid % (kHiddenTile / 4)) * 4;
        const std::size_t token =
            t_block + (token_group * kWaveTokTiles + j) * 16 + token_in_tile;
        const std::size_t h = r_block / 4 + h_local;
        if (token < batch) {
          float4 value{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
          for (unsigned stream = 0; stream < 4; ++stream) {
            const float* g = gates + stream * kPlane +
                             token_in_tile * kStreamStride + h_local;
            const float4 v = Load4(xn + token * m + stream * hidden + h);
            // Preserve the separate mixer's F32 FMA rounding.
            value.x = __fmaf_rn(v.x, SigmoidF(g[0]), value.x);
            value.y = __fmaf_rn(v.y, SigmoidF(g[1]), value.y);
            value.z = __fmaf_rn(v.z, SigmoidF(g[2]), value.z);
            value.w = __fmaf_rn(v.w, SigmoidF(g[3]), value.w);
          }
          value.x *= 0.25F;
          value.y *= 0.25F;
          value.z *= 0.25F;
          value.w *= 0.25F;
          *reinterpret_cast<float4*>(y + token * hidden + h) = value;
          if (mixed_half != nullptr) {
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h) =
                __floats2half2_rn(value.x, value.y);
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h + 2) =
                __floats2half2_rn(value.z, value.w);
          }
          if (mixed_q8 != nullptr) {
            float max_abs = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                  fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
              max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
            }
            const float d = max_abs / 127.0F;
            const float id = d != 0.0F ? 1.0F / d : 0.0F;
            const auto q0 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.x * id))));
            const auto q1 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.y * id))));
            const auto q2 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.z * id))));
            const auto q3 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.w * id))));
            const unsigned pos = h % 32;
            auto* tile = Q8ActTile(mixed_q8, hidden / 32,
                                   token / kQ8ActTileTokens, h / 32);
            const std::size_t tl = token % kQ8ActTileTokens;
            *reinterpret_cast<unsigned*>(tile + (pos >> 4) * 256 + tl * 16 +
                                         (pos & 15)) =
                q0 | q1 << 8 | q2 << 16 | q3 << 24;
            if (pos == 0) {
              *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                        tl * sizeof(float)) = d;
            }
          }
        }
        __syncthreads();
      }
    }
    return;
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        if constexpr (kSsmConv) {
          static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 &&
                        WN == 1);
          static_assert(!kHcMix && kRowGroup == 1);
          constexpr std::uint32_t channels = 10240;
          if (tok < batch) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const unsigned row = r0 + row_l + v * 4;
              if (row >= m)
                continue;
              const float4 current = src[v];
              if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                  tok + 3 >= batch) {
                *reinterpret_cast<float4*>(y + tok * m + row) = current;
              }
              if (row < channels && tok_l >= 3) {
                const float4 x0 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
                const float4 x1 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
                const float4 x2 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
                const float4 w0 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
                const float4 w1 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
                const float4 w2 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
                const float4 w3 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
                const float4 value{
                    SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                    SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                    SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                    SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
                *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                    value;
              }
            }
          }
        } else {
          if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

bool AttentionF16Gemm(const void* weights, const __half* input,
                      const float* q_gamma, const float* k_gamma, float* query,
                      float* gate, __half* keys, __half* values,
                      std::uint32_t n_tokens, const std::uint32_t* position,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope) {
  if (n_tokens < 1024 || weights == nullptr || input == nullptr ||
      q_gamma == nullptr || k_gamma == nullptr || query == nullptr ||
      gate == nullptr || keys == nullptr || values == nullptr ||
      position == nullptr)
    return false;
  const AttentionProjectionOutput output{q_gamma, k_gamma,  query, gate, keys,
                                         values,  position, theta, eps,  rope};
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<256, 128, 2, 8, 1, 1, false, false, true>),
      dim3((n_tokens + 127) / 128, 52), dim3(kThreads), 0, stream, weights,
      input, nullptr, n_tokens, 13312, 2560, nullptr, nullptr, nullptr, nullptr,
      nullptr, output);
  return true;
}

bool HcMixF16Gemm(const void* up, const __half* low_rank, const __half* xn,
                  const float* inject_w, float* mixed, __half* mixed_half,
                  void* mixed_q8, float* inject, std::uint32_t n_tokens,
                  std::uint32_t hidden, std::uint32_t rank,
                  hipStream_t stream) {
  if (n_tokens < 96 || hidden != 2560 || rank != 320 || up == nullptr ||
      low_rank == nullptr || xn == nullptr || mixed == nullptr ||
      (inject_w != nullptr && inject == nullptr)) {
    return false;
  }
  const dim3 grid((n_tokens + 127) / 128, 4 * hidden / 256);
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 1, 4, 2, 8, true>), grid,
                     dim3(kThreads), 0, stream, up, low_rank, mixed, n_tokens,
                     4 * hidden, rank, xn, mixed_half, mixed_q8);
  if (inject_w != nullptr) {
    const dim3 inject_grid(HcInjectPartsVec4(hidden), n_tokens);
    hipLaunchKernelGGL((HcMixEpilogueF16Kernel<false>), inject_grid,
                       dim3(kThreads), 0, stream, xn, nullptr, inject_w,
                       nullptr, nullptr, nullptr, inject, hidden);
  }
  return true;
}

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<64, 64, 2, 2, 4, 1, false, false, false, true>),
      dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0, stream, w, x,
      out, batch, m, k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // The same tile plan as the W8A8 route: a 128-token macro tile for wide
  // batches, 64-row tiles for narrow projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    if (m == 10240 && k == 320 && batch >= 1024) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 8>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && (((m == 16384 || m == 13312) && k == 2560) ||
                                 (m == 2560 && k == 6144))) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
}

bool DenseF16SsmGemm(const void* w, const __half* x, const float* conv_w,
                     const float* history, float* qkvz, float* convolved,
                     std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                     std::uint32_t channels, std::uint32_t kernel,
                     hipStream_t stream) {
  if (n_tokens < 1024 || m != 16384 || k != 2560 || channels != 10240 ||
      kernel != kSsmConvTaps) {
    return false;
  }
  hipLaunchKernelGGL((DenseF16GEMMKernel<256, 128, 2, 8, 1, 1, false, true>),
                     dim3((n_tokens + 127) / 128, m / 256), dim3(kThreads), 0,
                     stream, w, x, qkvz, n_tokens, m, k, nullptr, nullptr,
                     nullptr, conv_w, convolved);
  hipLaunchKernelGGL(
      SsmConvBoundaryKernel,
      dim3(Blocks(channels),
           (n_tokens + kSsmProjectionTileTokens - 1) / kSsmProjectionTileTokens,
           3),
      dim3(kThreads), 0, stream, qkvz, conv_w, history, convolved, n_tokens,
      channels, m);
  return true;
}

template<WeightType type, unsigned tokens>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL((SmallGemmKernel<type, tokens>),
                     dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                     dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<WeightType type>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    hipLaunchKernelGGL((SmallGemmKernel<type, 8, true>),
                       dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
                       dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    const auto consumed = groups * 8;
    x += std::size_t{consumed} * k;
    out += std::size_t{consumed} * m;
    tokens -= consumed;
    if (tokens == 0)
      return;
  }
  switch (tokens) {
    case 1:
      return LaunchSmallGemm<type, 1>(w, x, out, m, k, stream);
    case 2:
      return LaunchSmallGemm<type, 2>(w, x, out, m, k, stream);
    case 3:
      return LaunchSmallGemm<type, 3>(w, x, out, m, k, stream);
    case 4:
      return LaunchSmallGemm<type, 4>(w, x, out, m, k, stream);
    case 5:
      return LaunchSmallGemm<type, 5>(w, x, out, m, k, stream);
    case 6:
      return LaunchSmallGemm<type, 6>(w, x, out, m, k, stream);
    case 7:
      return LaunchSmallGemm<type, 7>(w, x, out, m, k, stream);
    case 8:
      return LaunchSmallGemm<type, 8>(w, x, out, m, k, stream);
    default:
      throw std::logic_error("small projection requires 1-8 token rows");
  }
}

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  switch (type) {
    case WeightType::kF32:
      return SmallGemmForType<WeightType::kF32>(w, x, out, n_tokens, m, k,
                                                stream);
    case WeightType::kBF16:
      return SmallGemmForType<WeightType::kBF16>(w, x, out, n_tokens, m, k,
                                                 stream);
    case WeightType::kF16:
      return SmallGemmForType<WeightType::kF16>(w, x, out, n_tokens, m, k,
                                                stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
}

void PleGate(const float* key_n, const float* query_n, const float* value,
             float* gated, std::uint32_t n_tokens, std::uint32_t hidden,
             std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(PleGateKernel, dim3(n_tokens, streams), dim3(kThreads), 0,
                     stream, key_n, query_n, value, gated, hidden, streams);
}

void PleConv(const float* in, const float* w, float* history,
             float* history_scratch, float* out, RollbackRows snapshots,
             std::uint32_t n_tokens, std::uint32_t channels,
             std::uint32_t kernel, std::uint32_t dilation, hipStream_t stream) {
  const std::uint32_t hist = (kernel - 1) * dilation;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  hipLaunchKernelGGL(PleConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, in, w, history, out, n_tokens, channels, kernel,
                     dilation, hist);
  if (snapshots.rows[0] != nullptr && n_tokens > 1) {
    const std::size_t saved = static_cast<std::size_t>(n_tokens - 1) * channels;
    hipLaunchKernelGGL(RollingSnapshotKernel, dim3(Blocks(saved * hist)),
                       dim3(kThreads), 0, stream, in, channels, history,
                       snapshots, n_tokens - 1, channels, hist);
  }
  // Device copies stay kernels: a copy engine transfer is not reliably
  // ordered behind the kernels on this stream.
  const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
  hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                     dim3(kThreads), 0, stream, in, channels, history,
                     history_scratch, n_tokens, channels, hist);
  hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                     stream, history_scratch, history, hist_count);
}

void PleInject(float* res, const float* gated, const float* conv,
               std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(PleInjectKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, res, gated, conv, count);
}

void RestoreGdnState(float* state, RollbackRows snapshots, std::uint32_t keep,
                     std::uint32_t k_heads, std::uint32_t v_heads,
                     hipStream_t stream) {
  const auto count = std::size_t{v_heads} * kGdnDim * kGdnDim;
  RestoreGdnStateKernel<<<(count + 255) / 256, 256, 0, stream>>>(
      state, snapshots, keep, k_heads, v_heads);
}

void GatedDeltaNet(const float* qkv, std::uint32_t qkv_stride, const float* z,
                   std::uint32_t z_stride, const float* alpha_beta,
                   const float* conv_w, const float* a, const float* dt,
                   const float* norm_w, float* conv_state, float* conv_scratch,
                   float* qn, float* kn, float* raw, float* state, float* out,
                   void* out_q8, RollbackRows state_snapshots,
                   RollbackRows conv_snapshots, std::uint32_t n_tokens,
                   std::uint32_t k_heads, std::uint32_t v_heads,
                   std::uint32_t d, std::uint32_t kernel, bool row_split,
                   bool convolved, float eps, hipStream_t stream,
                   __half* out_half) {
  const std::uint32_t channels = 2 * k_heads * d + v_heads * d;
  const std::size_t count = static_cast<std::size_t>(n_tokens) * channels;
  const bool saved_history = !convolved && kernel == kSsmConvTaps &&
                             n_tokens <= kSsmConvTokensPerThread;
  if (!convolved) {
    if (saved_history) {
      hipLaunchKernelGGL((SsmConv4Kernel<true, false>), dim3(Blocks(channels)),
                         dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w,
                         conv_state, conv_scratch, n_tokens, channels,
                         conv_snapshots, nullptr, 0);
    } else if (kernel == kSsmConvTaps) {
      hipLaunchKernelGGL(
          (SsmConv4Kernel<false, false>),
          dim3(Blocks(channels), (n_tokens + kSsmConvTokensPerThread - 1) /
                                     kSsmConvTokensPerThread),
          dim3(kThreads), 0, stream, qkv, qkv_stride, conv_w, conv_state,
          conv_scratch, n_tokens, channels, RollbackRows{}, nullptr, 0);
    } else {
      hipLaunchKernelGGL(SsmConvKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_w, conv_state,
                         conv_scratch, n_tokens, channels, kernel);
    }
  }
  if (!saved_history) {
    if (conv_snapshots.rows[0] != nullptr && n_tokens > 1) {
      const std::size_t saved =
          static_cast<std::size_t>(n_tokens - 1) * channels;
      hipLaunchKernelGGL(RollingSnapshotKernel,
                         dim3(Blocks(saved * (kernel - 1))), dim3(kThreads), 0,
                         stream, qkv, qkv_stride, conv_state, conv_snapshots,
                         n_tokens - 1, channels, kernel - 1);
    }
    // The rolling state is the last kernel-1 projections: [history ; qkv].
    const std::uint32_t hist = kernel - 1;
    const std::size_t hist_count = static_cast<std::size_t>(hist) * channels;
    hipLaunchKernelGGL(HistoryShiftKernel, dim3(Blocks(hist_count)),
                       dim3(kThreads), 0, stream, qkv, qkv_stride, conv_state,
                       conv_scratch + count, n_tokens, channels, hist);
    hipLaunchKernelGGL(CopyKernel, dim3(Blocks(hist_count)), dim3(kThreads), 0,
                       stream, conv_scratch + count, conv_state, hist_count);
  }
  const unsigned waves = kThreads / 32;
  if (row_split && d == kGdnDim && state_snapshots.rows[0] == nullptr) {
    hipLaunchKernelGGL(GdnPrepKqKernel, dim3(k_heads, n_tokens), dim3(32), 0,
                       stream, conv_scratch, qn, n_tokens, k_heads, channels,
                       eps);
    hipLaunchKernelGGL(
        GdnPrepAbKernel,
        dim3(Blocks(static_cast<std::size_t>(n_tokens) * v_heads)),
        dim3(kThreads), 0, stream, alpha_beta, a, dt, kn,
        static_cast<std::size_t>(n_tokens) * v_heads, v_heads);
    hipLaunchKernelGGL(GdnRowSplitKernel, dim3(kGdnDim / 64, v_heads),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn, state,
                       raw, n_tokens, k_heads, v_heads);
  } else {
    hipLaunchKernelGGL(GdnPrepKernel<false>,
                       dim3((n_tokens * k_heads + waves - 1) / waves),
                       dim3(kThreads), 0, stream, conv_scratch, qn, kn,
                       n_tokens * k_heads, k_heads, channels, eps, nullptr, 0);
    hipLaunchKernelGGL(GdnKernel<false>,
                       dim3(v_heads, kGdnDim / kGdnRowsPerBlock),
                       dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream,
                       conv_scratch, qn, kn, alpha_beta, a, dt, state, raw,
                       state_snapshots, n_tokens, k_heads, v_heads, nullptr, 0);
  }
  hipLaunchKernelGGL(
      GdnEpilogueKernel<false>, dim3((n_tokens * v_heads + waves - 1) / waves),
      dim3(kThreads), 0, stream, raw, z, z_stride, norm_w, out, out_q8,
      out_half, n_tokens * v_heads, v_heads, eps, nullptr, 0);
}

bool GatedDeltaNetBatch(const GdnBatchItem* items, std::uint32_t count,
                        std::uint32_t max_tokens, std::uint32_t active,
                        std::uint32_t qkv_stride, std::uint32_t z_stride,
                        const float* conv_w, const float* a, const float* dt,
                        const float* norm_w, std::uint32_t k_heads,
                        std::uint32_t v_heads, float eps, hipStream_t stream) {
  if (items == nullptr || count == 0 || count > 8 || max_tokens == 0 ||
      max_tokens > kSsmConvTokensPerThread)
    return false;
  const auto channels = (2 * k_heads + v_heads) * kGdnDim;
  constexpr auto waves = kThreads / 32;
  hipLaunchKernelGGL((SsmConv4Kernel<true, true>),
                     dim3(Blocks(channels), 1, count), dim3(kThreads), 0,
                     stream, nullptr, qkv_stride, conv_w, nullptr, nullptr,
                     max_tokens, channels, RollbackRows{}, items, active);
  hipLaunchKernelGGL(GdnPrepKernel<true>,
                     dim3((max_tokens * k_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, nullptr,
                     max_tokens * k_heads, k_heads, channels, eps, items,
                     active);
  hipLaunchKernelGGL(
      GdnKernel<true>, dim3(v_heads, kGdnDim / kGdnRowsPerBlock, count),
      dim3(kGdnRowsPerBlock * kGdnLanes), 0, stream, nullptr, nullptr, nullptr,
      nullptr, a, dt, nullptr, nullptr, RollbackRows{}, max_tokens, k_heads,
      v_heads, items, active);
  hipLaunchKernelGGL(GdnEpilogueKernel<true>,
                     dim3((max_tokens * v_heads + waves - 1) / waves, 1, count),
                     dim3(kThreads), 0, stream, nullptr, nullptr, z_stride,
                     norm_w, nullptr, nullptr, nullptr, max_tokens * v_heads,
                     v_heads, eps, items, active);
  return hipGetLastError() == hipSuccess;
}

void UnpackQGate(const float* qg, std::uint32_t qg_stride, float* q,
                 float* gate, float* k, float* v, std::uint32_t n_tokens,
                 std::uint32_t heads, std::uint32_t d, std::uint32_t kv_width,
                 hipStream_t stream) {
  hipLaunchKernelGGL(UnpackQGateKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, qg, qg_stride, q, gate, k, v, heads, d, kv_width);
}

bool PrepareAttention(const float* packed, std::uint32_t stride,
                      const float* q_gamma, const float* k_gamma, float* q,
                      float* gate, __half* k_cache, __half* v_cache,
                      std::uint32_t n_tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, std::uint32_t d,
                      std::uint32_t rotary_dim, const std::uint32_t* start_pos,
                      float theta, float eps, hipStream_t stream,
                      const qwen::vision::DeviceRope* rope, bool prefill) {
  if (d == 0 || d > 256 || rotary_dim == 0 || rotary_dim > d ||
      rotary_dim % 2 != 0 || heads == 0 || kv_heads == 0 ||
      stride < static_cast<std::size_t>(2) * (heads + kv_heads) * d) {
    return false;
  }
  if (n_tokens == 0)
    return true;
  if (!prefill && n_tokens < 32) {
    hipLaunchKernelGGL((PrepareAttentionKernel<1>),
                       dim3(n_tokens, heads + kv_heads), dim3(kThreads), 0,
                       stream, packed, stride, q_gamma, k_gamma, q, gate,
                       k_cache, v_cache, heads, kv_heads, d, rotary_dim,
                       start_pos, theta, eps, rope);
  } else {
    hipLaunchKernelGGL((PrepareAttentionKernel<4>),
                       dim3(n_tokens, (heads + kv_heads + 3) / 4),
                       dim3(kThreads), 0, stream, packed, stride, q_gamma,
                       k_gamma, q, gate, k_cache, v_cache, heads, kv_heads, d,
                       rotary_dim, start_pos, theta, eps, rope);
  }
  return true;
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const qwen::vision::DeviceRope* rope) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta, rope);
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, const std::uint32_t* start_pos,
             hipStream_t stream) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, row_dim, start_pos);
}

void StoreRows(const float* src, float* dst, std::uint32_t n_tokens,
               std::uint32_t row_dim, const std::uint32_t* start_pos,
               std::uint32_t capacity, hipStream_t stream) {
  hipLaunchKernelGGL(StoreRowsKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, dst, row_dim, start_pos, capacity);
}

void PoolIndexerBlocks(const float* raw_keys, const float* gamma,
                       __half* blocks, const std::uint32_t* first_block,
                       const std::uint32_t* start_pos, std::uint32_t n_tokens,
                       std::uint32_t grid_blocks, std::uint32_t ratio,
                       std::uint32_t dim, std::uint32_t rotary_dim, float theta,
                       float eps, std::uint32_t capacity, hipStream_t stream,
                       const qwen::vision::DeviceRope* rope) {
  if (grid_blocks == 0) {
    return;
  }
  hipLaunchKernelGGL(PoolBlocksKernel, dim3(grid_blocks), dim3(kThreads), 0,
                     stream, raw_keys, gamma, blocks, first_block, start_pos,
                     n_tokens, ratio, dim, rotary_dim, theta, eps, capacity,
                     rope);
}

void SelectBlocks(const float* q, const __half* blocks, std::uint32_t* mask,
                  float* scores, std::uint32_t n_tokens,
                  const std::uint32_t* start_pos, std::uint32_t first_token,
                  std::uint32_t heads, std::uint32_t dim, std::uint32_t ratio,
                  std::uint32_t budget, std::uint32_t mask_words,
                  std::uint32_t max_blocks, hipStream_t stream) {
  if (heads != kSelectHeads || dim != kSelectDim) {
    return;
  }
  // Grids are sized by max_blocks so a captured decode graph replays at any
  // position; blocks past the live range return at once.
  const dim3 grid(n_tokens, (max_blocks + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(SelectScoreKernel, grid, dim3(kThreads), 0, stream, q,
                     blocks, scores, n_tokens, start_pos, first_token, ratio,
                     budget, max_blocks);
  hipLaunchKernelGGL(SelectMarkKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, mask, scores, start_pos, first_token, ratio,
                     budget, mask_words, max_blocks);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const std::uint32_t* mask, std::uint32_t mask_words, float* out,
               float* partials, std::uint32_t splits, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d, std::uint32_t ratio,
               hipStream_t stream) {
  if (d != 256) {
    return;  // the kernel is written for the model's 256-wide heads
  }
  const std::uint32_t z = partials != nullptr ? std::max(splits, 1u) : 1u;
  hipLaunchKernelGGL(AttentionKernel, dim3(heads, n_tokens, z), dim3(kThreads),
                     0, stream, q, k_cache, v_cache, mask, mask_words, out,
                     partials, start_pos, heads, kv_heads, d, ratio);
  if (z > 1) {
    hipLaunchKernelGGL(AttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, heads, d, z);
  }
}

bool WmmaCausalAttention(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         const std::uint32_t* mask, std::uint32_t mask_words,
                         float* out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream,
                         bool last_only) {
  if (heads != kWmmaQueryHeads || kv_heads != kWmmaKvHeads ||
      d != kWmmaHeadDim || ratio != kWmmaRatio || n_tokens == 0 ||
      (mask != nullptr && mask_words > kWmmaMaxMaskWords)) {
    return false;
  }
  if (mask != nullptr) {
    constexpr std::uint32_t kPackedQueries = 4;
    const std::uint32_t first_group =
        last_only ? (n_tokens - 1) / kPackedQueries : 0;
    const dim3 grid(
        (n_tokens + kPackedQueries - 1) / kPackedQueries - first_group,
        kWmmaKvHeads);
    // At deep sparse windows, staging the current V before fetching the
    // next one shortens their overlapping register lifetimes.
    if (start_pos >= 65536) {
      hipLaunchKernelGGL(
          (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true, true>),
          grid, dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
          mask_words, out, start_pos, n_tokens, first_group);
    } else {
      hipLaunchKernelGGL(
          (WmmaCausalAttentionKernel<kPackedQueries, kWmmaKeys, true>), grid,
          dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask,
          mask_words, out, start_pos, n_tokens, first_group);
    }
    return true;
  }
  const std::uint32_t first_group =
      last_only ? (n_tokens - 1) / kWmmaQueryRows : 0;
  const dim3 grid(
      (n_tokens + kWmmaQueryRows - 1) / kWmmaQueryRows - first_group,
      kWmmaKvHeads * (kWmmaGqa / kWmmaHeads));
  hipLaunchKernelGGL(
      (WmmaCausalAttentionKernel<kWmmaQueryRows, kWmmaKeys, false>), grid,
      dim3(kThreads), 0, stream, q, gate, k_cache, v_cache, mask, mask_words,
      out, start_pos, n_tokens, first_group);
  return true;
}

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  if (n_experts <= 512) {
    hipLaunchKernelGGL((RouterTopKKernel<512>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  } else {
    hipLaunchKernelGGL((RouterTopKKernel<1024>), dim3(n_tokens), dim3(kThreads),
                       0, stream, logits, stride, ids, weights, n_experts, k);
  }
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
}

void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueKernel, dim3(n_tokens, Blocks(dim)),
                     dim3(kThreads), 0, stream, expert_out, weights, shared,
                     gate, gate_stride, out, k, dim);
}

void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream) {
  if (dim % 4 != 0) {
    MoeEpilogue(expert_out, weights, shared, gate, gate_stride, out, n_tokens,
                k, dim, stream);
    return;
  }
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<float>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream) {
  hipLaunchKernelGGL(MoeEpilogueVec4Kernel<__half>,
                     dim3(n_tokens, Blocks(dim / 4)), dim3(kThreads), 0, stream,
                     expert_out, weights, shared, gate, gate_stride, out, k,
                     dim);
}

void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream) {
  hipLaunchKernelGGL(MtpHiddenKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     base, alt, row, dst, width);
}

void MtpAddEmbedding(const float* embedding, float* residual,
                     std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(MtpAddEmbeddingKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, embedding, residual, hidden, streams);
}

void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxPartialKernel, dim3(kArgmaxParts, n_tokens),
                     dim3(kThreads), 0, stream, logits, scratch, vocab);
  hipLaunchKernelGGL(ArgmaxFinishKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, scratch, out, vocab);
}

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(GatherArgmaxCandidatesKernel, dim3(Blocks(rows)),
                     dim3(kThreads), 0, stream, logits, ids, out, rows, vocab);
}

template<unsigned Keep>
void SelectMtpCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || scratch_ids == nullptr ||
      vocab == 0) {
    throw std::invalid_argument("invalid MTP candidate selection");
  }
  const auto tiles = [](std::uint32_t n) {
    return 1U + (n - 1U) / kMtpCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = vocab; size > kMtpCandidateTile; size = tiles(size) * Keep) {
    ++passes;
  }
  // Choose the first buffer so the final pass always lands in `ids`.
  auto* destination = passes % 2 != 0 ? ids : scratch_ids;
  const std::uint32_t* source = nullptr;
  auto size = vocab;
  for (;;) {
    const auto blocks = tiles(size);
    hipLaunchKernelGGL((MtpCandidateTileKernel<Keep>), dim3(blocks),
                       dim3(kThreads), 0, stream, logits, source, destination,
                       blocks == 1 ? scores : nullptr, size, vocab);
    if (blocks == 1) {
      break;
    }
    size = blocks * Keep;
    source = destination;
    destination = destination == ids ? scratch_ids : ids;
  }
}

std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab) {
  const auto tiles = (vocab + 1023) / 1024;
  return (tiles > 2 ? tiles : 2) * kMtpCandidates;
}

void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream) {
  if (scores == nullptr)
    throw std::invalid_argument("invalid MTP candidate scores");
  SelectMtpCandidates<kMtpCandidates>(logits, ids, scratch_ids, scores, vocab,
                                      stream);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
