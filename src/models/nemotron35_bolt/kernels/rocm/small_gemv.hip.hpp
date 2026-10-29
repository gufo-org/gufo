#ifndef GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SMALL_GEMV_HIP_HPP_
#define GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SMALL_GEMV_HIP_HPP_

// Weight-streaming products for at most kTokens activation rows, shared by
// the trunk router (F32 weights) and the drafter (BF16 weights). The block
// stages the activations through LDS in column chunks; each wave keeps one
// activation vector per token in registers while it streams kRows weight
// rows, and narrow matrices split every chunk across kSplit waves per row so
// enough workgroups stream at once.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::nemotron35_bolt::rocm::small_gemv {

inline constexpr int kTokens = 8;
inline constexpr int kLanes = 32;
inline constexpr int kWaves = 8;
inline constexpr int kChunk = 1024;  ///< Activation columns per LDS stage.

/// Loads 8 consecutive weights as floats.
struct Bf16Weights {
  static constexpr int kVector = 8;
  __device__ static void Load(const void* w, std::size_t index, float* a) {
    const uint4 p = reinterpret_cast<const uint4*>(w)[index];
    const unsigned words[4] = {p.x, p.y, p.z, p.w};
    for (int j = 0; j < 4; ++j) {
      a[2 * j] = __uint_as_float(words[j] << 16);
      a[2 * j + 1] = __uint_as_float(words[j] & 0xFFFF0000U);
    }
  }
};

/// Loads 4 consecutive weights.
struct F32Weights {
  static constexpr int kVector = 4;
  __device__ static void Load(const void* w, std::size_t index, float* a) {
    const float4 p = reinterpret_cast<const float4*>(w)[index];
    a[0] = p.x;
    a[1] = p.y;
    a[2] = p.z;
    a[3] = p.w;
  }
};

/// out[t][r] = sum_c w[r][c] * x[t][c]; cols must be a multiple of kChunk.
/// A block covers kWaves / kSplit row groups of kRows rows.
template<class Weights, int kRows, int kSplit>
__global__ __launch_bounds__(kLanes* kWaves) void Kernel(const void* w,
                                                         const float* x,
                                                         float* out, int rows,
                                                         int cols,
                                                         int n_tokens) {
  constexpr int V = Weights::kVector;
  constexpr int kPerWave = kChunk / V / kSplit;  ///< Vectors per chunk.
  static_assert(kPerWave % kLanes == 0);
  __shared__ float xs[kTokens][kChunk];
  __shared__ float partial[kWaves][kRows][kTokens];
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / kLanes;
  const int lane = tid % kLanes;
  const int group = wave / kSplit;
  const int segment = wave % kSplit;
  const int row0 =
      (static_cast<int>(blockIdx.x) * (kWaves / kSplit) + group) * kRows;
  float acc[kRows][kTokens] = {};
  const std::size_t row_vectors = static_cast<std::size_t>(cols / V);
  for (int c0 = 0; c0 < cols; c0 += kChunk) {
    __syncthreads();
    // Absent tokens read zeros, so every loop below has a fixed trip count
    // and the per-token arrays stay in registers.
    for (int i = tid; i < kTokens * kChunk / 4; i += kLanes * kWaves) {
      const int t = i / (kChunk / 4);
      const int c = (i % (kChunk / 4)) * 4;
      *reinterpret_cast<float4*>(&xs[t][c]) =
          t < n_tokens ? *reinterpret_cast<const float4*>(
                             x + static_cast<std::size_t>(t) * cols + c0 + c)
                       : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    __syncthreads();
    for (int k = 0; k < kPerWave / kLanes; ++k) {
      const int v = segment * kPerWave + k * kLanes + lane;
      float xv[kTokens][V];
#pragma unroll
      for (int t = 0; t < kTokens; ++t) {
#pragma unroll
        for (int j = 0; j < V; j += 4) {
          const float4 b = *reinterpret_cast<const float4*>(&xs[t][v * V + j]);
          xv[t][j] = b.x;
          xv[t][j + 1] = b.y;
          xv[t][j + 2] = b.z;
          xv[t][j + 3] = b.w;
        }
      }
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        // Rows past the end reread the last row; their sums are dropped.
        const int row = min(row0 + r, rows - 1);
        float a[V];
        Weights::Load(w,
                      static_cast<std::size_t>(row) * row_vectors +
                          static_cast<std::size_t>(c0 / V + v),
                      a);
#pragma unroll
        for (int t = 0; t < kTokens; ++t) {
          float sum = 0.0F;
#pragma unroll
          for (int j = 0; j < V; ++j)
            sum += a[j] * xv[t][j];
          acc[r][t] += sum;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int t = 0; t < kTokens; ++t) {
      float v = acc[r][t];
      for (int offset = kLanes / 2; offset > 0; offset /= 2)
        v += __shfl_xor(v, offset, kLanes);
      if (lane == 0)
        partial[wave][r][t] = v;
    }
  }
  __syncthreads();
  // Segments of a row group add in a fixed order.
  for (int i = tid; i < (kWaves / kSplit) * kRows * kTokens;
       i += kLanes * kWaves) {
    const int g = i / (kRows * kTokens);
    const int r = (i / kTokens) % kRows;
    const int t = i % kTokens;
    const int row =
        (static_cast<int>(blockIdx.x) * (kWaves / kSplit) + g) * kRows + r;
    if (t >= n_tokens || row >= rows)
      continue;
    float sum = 0.0F;
    for (int s = 0; s < kSplit; ++s)
      sum += partial[g * kSplit + s][r][t];
    out[static_cast<std::size_t>(t) * rows + row] = sum;
  }
}

template<class Weights, int kRows, int kSplit>
void Run(const void* w, const float* x, float* out, int rows, int cols,
         int n_tokens, hipStream_t stream) {
  constexpr int kRowsPerBlock = kWaves / kSplit * kRows;
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  Kernel<Weights, kRows, kSplit>
      <<<blocks, kLanes * kWaves, 0, stream>>>(w, x, out, rows, cols, n_tokens);
}

/// Picks rows per wave and waves per row so that wide matrices reuse each
/// staged activation over several rows and narrow ones still fill about
/// 128 workgroups.
template<class Weights>
hipError_t Launch(const void* w, const float* x, float* out, int rows, int cols,
                  int n_tokens, hipStream_t stream) {
  if (n_tokens < 1 || n_tokens > kTokens || cols % kChunk != 0)
    return hipErrorInvalidValue;
  if (rows >= 4096)
    Run<Weights, 2, 4>(w, x, out, rows, cols, n_tokens, stream);
  else if (rows >= 1024)
    Run<Weights, 2, 2>(w, x, out, rows, cols, n_tokens, stream);
  else
    Run<Weights, 1, 4>(w, x, out, rows, cols, n_tokens, stream);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm::small_gemv

#endif  // GUFO_MODELS_NEMOTRON35_BOLT_KERNELS_ROCM_SMALL_GEMV_HIP_HPP_
