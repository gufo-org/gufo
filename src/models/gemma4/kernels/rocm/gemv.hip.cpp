// Autoregressive decode projections for Gemma 4's K-quant, Q4_0 and Q8_0
// weights, and the batch-invariant binary16 small-batch projection.
//
// y[m] = sum_k x[k] * W[m][k]; the weights stay in their GGUF blocks and the
// activations stay FP32. A lane's unit of work is one 16-byte vector of
// quantized values -- 32 values of a Q4_K/Q5_K super-block (with their Q5_K
// high bits) or 64 values of a Q6_K super-block -- dequantized once and
// contracted in a fixed FMA order. Lanes holding the same task of different
// rows read the same activations, which the hardware broadcasts; each row
// reduces over its task lanes with a fixed xor tree. The eight waves of a
// workgroup split one row group's reduction into slices, summed in wave
// order, which keeps enough loads in flight for short outputs.
#include "src/models/gemma4/kernels/rocm/gemv.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "src/models/gemma4/kernels/rocm/gemv_tasks.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

using namespace gemv_tasks;

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;

constexpr int kSlices = kWavesPerBlock;

template<GemvFormat F>
struct FormatTraits;
template<>
struct FormatTraits<GemvFormat::kQ4_0> {
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
};
template<>
struct FormatTraits<GemvFormat::kQ4_K> {
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
};
template<>
struct FormatTraits<GemvFormat::kQ5_K> {
  static constexpr int kBlockBytes = 176;
  static constexpr int kTasks = 8;
};
template<>
struct FormatTraits<GemvFormat::kQ6_K> {
  static constexpr int kBlockBytes = 210;
  static constexpr int kTasks = 4;
};
/// Eight 34-byte blocks as one 272-byte group per 256 values.
template<>
struct FormatTraits<GemvFormat::kQ8_0> {
  static constexpr int kBlockBytes = 272;
  static constexpr int kTasks = 8;
};

template<GemvFormat F>
__device__ __forceinline__ float Task(const std::uint8_t* block, int t,
                                      const float* x, float acc) {
  if constexpr (F == GemvFormat::kQ4_0) {
    return TaskQ40(block, t, x, acc);
  } else if constexpr (F == GemvFormat::kQ4_K) {
    return TaskQ45K<false>(block, t, x, acc);
  } else if constexpr (F == GemvFormat::kQ5_K) {
    return TaskQ45K<true>(block, t, x, acc);
  } else if constexpr (F == GemvFormat::kQ8_0) {
    return TaskQ80(block, t, x, acc);
  } else {
    return TaskQ6K(block, t, x, acc);
  }
}

/// Lanes sharing a task position read the same activations, so the wave
/// covers 32 / kTasks rows at once: lane = row_in_wave * kTasks + task. The
/// workgroup's waves take the super-blocks of those rows in kSlices slices:
/// interleaved (wave-adjacent blocks) for short rows, contiguous runs for
/// long ones.
template<GemvFormat F, bool kInterleave>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    KQuantGemvKernel(const std::uint8_t* __restrict__ w,
                     const float* __restrict__ x, float* __restrict__ y,
                     std::uint32_t m, std::uint32_t k) {
  using T = FormatTraits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  __shared__ float partial[kSlices][kWave];
  const int lane = threadIdx.x % kWave;
  const int slice = threadIdx.x / kWave;
  const int t = lane % T::kTasks;
  const std::uint32_t row = blockIdx.x * kRowsPerWave + lane / T::kTasks;
  const std::uint32_t blocks_per_row = k / 256;
  float acc = 0.0F;
  if (row < m) {
    const std::uint8_t* base =
        w + static_cast<std::size_t>(row) * blocks_per_row * T::kBlockBytes;
    if constexpr (kInterleave) {
#pragma unroll 2
      for (std::uint32_t blk = slice; blk < blocks_per_row; blk += kSlices) {
        acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
      }
    } else {
      const std::uint32_t per = (blocks_per_row + kSlices - 1) / kSlices;
      const std::uint32_t end = min(blocks_per_row, (slice + 1) * per);
#pragma unroll 2
      for (std::uint32_t blk = slice * per; blk < end; ++blk) {
        acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
      }
    }
  }
#pragma unroll
  for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  partial[slice][lane] = acc;
  __syncthreads();
  if (slice == 0 && t == 0 && row < m) {
    float sum = partial[0][lane];
#pragma unroll
    for (int s = 1; s < kSlices; ++s) {
      sum += partial[s][lane];
    }
    y[row] = sum;
  }
}

/// Short rows (a few super-blocks, like the drafter's 1024-wide vocabulary
/// head) leave most slices idle; each wave takes whole rows instead.
template<GemvFormat F>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    KQuantGemvRowsKernel(const std::uint8_t* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ y,
                         std::uint32_t m, std::uint32_t k) {
  using T = FormatTraits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  const int lane = threadIdx.x % kWave;
  const int t = lane % T::kTasks;
  const std::uint32_t row =
      (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * kRowsPerWave +
      lane / T::kTasks;
  const std::uint32_t blocks_per_row = k / 256;
  float acc = 0.0F;
  if (row < m) {
    const std::uint8_t* base =
        w + static_cast<std::size_t>(row) * blocks_per_row * T::kBlockBytes;
#pragma unroll 4
    for (std::uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      acc = Task<F>(base + blk * T::kBlockBytes, t, x + blk * 256, acc);
    }
  }
#pragma unroll
  for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
    acc += __shfl_xor(acc, offset, kWave);
  }
  if (t == 0 && row < m) {
    y[row] = acc;
  }
}

template<GemvFormat F>
void Launch(const void* w, const float* x, float* y, std::uint32_t m,
            std::uint32_t k, hipStream_t stream) {
  constexpr int kRowsPerBlock = kWave / FormatTraits<F>::kTasks;
  const auto* weights = static_cast<const std::uint8_t*>(w);
  if (k <= 1024) {
    constexpr int kRows = kRowsPerBlock * kWavesPerBlock;
    KQuantGemvRowsKernel<F>
        <<<(m + kRows - 1) / kRows, kWave * kWavesPerBlock, 0, stream>>>(
            weights, x, y, m, k);
    return;
  }
  const unsigned blocks = (m + kRowsPerBlock - 1) / kRowsPerBlock;
  // Measured on gfx1151: interleaving wins up to K = 8192, contiguous
  // slices on the 21504-long FFN down reduction.
  if (k <= 8192) {
    KQuantGemvKernel<F, true>
        <<<blocks, kWave * kWavesPerBlock, 0, stream>>>(weights, x, y, m, k);
  } else {
    KQuantGemvKernel<F, false>
        <<<blocks, kWave * kWavesPerBlock, 0, stream>>>(weights, x, y, m, k);
  }
}

/// Binary16 weights, up to kMaxHalfGemvRows activation rows per weight
/// pass. A wave takes one output row in 256-value units, eight values (one
/// 16-byte vector) per lane, contracted per activation row in one FMA chain;
/// the workgroup's waves take interleaved units and are summed in wave order
/// after a fixed xor tree. A row's arithmetic never depends on kRows.
template<int kRows>
__global__ void __launch_bounds__(kWave* kWavesPerBlock)
    HalfGemvKernel(const __half* __restrict__ w, const float* __restrict__ x,
                   float* __restrict__ y, std::uint32_t m, std::uint32_t k) {
  __shared__ float partial[kSlices][kRows];
  const int lane = threadIdx.x % kWave;
  const int slice = threadIdx.x / kWave;
  const std::uint32_t row = blockIdx.x;
  const __half* wrow = w + static_cast<std::size_t>(row) * k;
  const std::uint32_t units = (k + 255) / 256;
  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    acc[r] = 0.0F;
  }
#pragma unroll 2
  for (std::uint32_t unit = slice; unit < units; unit += kSlices) {
    const std::uint32_t at = unit * 256 + static_cast<std::uint32_t>(lane) * 8;
    if (at >= k) {
      continue;
    }
    const uint4 packed = *reinterpret_cast<const uint4*>(wrow + at);
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    float wv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __half2 pair = __builtin_bit_cast(__half2, words[i]);
      wv[2 * i] = __low2float(pair);
      wv[2 * i + 1] = __high2float(pair);
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const float* xr = x + static_cast<std::size_t>(r) * k + at;
      const float4 x0 = *reinterpret_cast<const float4*>(xr);
      const float4 x1 = *reinterpret_cast<const float4*>(xr + 4);
      const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[r] = __builtin_fmaf(wv[i], xv[i], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1) {
      acc[r] += __shfl_xor(acc[r], offset, kWave);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      partial[slice][r] = acc[r];
    }
  }
  __syncthreads();
  if (threadIdx.x < kRows) {
    const int r = static_cast<int>(threadIdx.x);
    float sum = partial[0][r];
#pragma unroll
    for (int s = 1; s < kSlices; ++s) {
      sum += partial[s][r];
    }
    y[static_cast<std::size_t>(r) * m + row] = sum;
  }
}

template<int kRows>
void LaunchHalf(const void* w, const float* x, float* y, std::uint32_t m,
                std::uint32_t k, hipStream_t stream) {
  HalfGemvKernel<kRows><<<m, kWave * kWavesPerBlock, 0, stream>>>(
      static_cast<const __half*>(w), x, y, m, k);
}

/// Calls launch.template operator()<rows>() for rows 1 to kMaxHalfGemvRows.
template<class Launch>
bool ForRows(std::uint32_t rows, Launch&& launch) {
  return [&]<int... N>(std::integer_sequence<int, N...>) {
    return ((rows == N + 1 && (launch.template operator()<N + 1>(), true)) ||
            ...);
  }(std::make_integer_sequence<int, kMaxHalfGemvRows>{});
}

__global__ void Bf16ToHalfKernel(std::uint16_t* __restrict__ data,
                                 std::size_t count,
                                 std::uint32_t* __restrict__ overflow) {
  std::uint32_t outside = 0;
  for (std::size_t i =
           blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const float v = __uint_as_float(static_cast<std::uint32_t>(data[i]) << 16U);
    const __half h = __float2half_rn(v);
    outside += __hisinf(h) != 0 || __hisnan(h) ? 1U : 0U;
    data[i] = __builtin_bit_cast(std::uint16_t, h);
  }
  if (outside != 0) {
    atomicAdd(overflow, outside);
  }
}

constexpr int kRepackThreads = 256;

/// ggml's make_qkx2_quants(32, 15, x, weights, L, &min, Laux, -1, 0.1, 20):
/// the weighted least-squares (scale, min) over 21 candidate grids.
__device__ float FitScaleMin(const float* x, const float* weights,
                             std::uint8_t* levels, float* the_min) {
  constexpr int kN = 32;
  constexpr float kMax = 15.0F;
  float lo = x[0];
  float hi = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < kN; ++i) {
    lo = fminf(lo, x[i]);
    hi = fmaxf(hi, x[i]);
    sum_w += weights[i];
    sum_x += weights[i] * x[i];
  }
  lo = fminf(lo, 0.0F);
  if (hi == lo) {
    for (int i = 0; i < kN; ++i) {
      levels[i] = 0;
    }
    *the_min = -lo;
    return 0.0F;
  }
  float iscale = kMax / (hi - lo);
  float scale = 1.0F / iscale;
  float best = 0.0F;
  for (int i = 0; i < kN; ++i) {
    const int l = min(15, max(0, __float2int_rn(iscale * (x[i] - lo))));
    levels[i] = static_cast<std::uint8_t>(l);
    const float diff = scale * static_cast<float>(l) + lo - x[i];
    best += weights[i] * diff * diff;
  }
  std::uint8_t trial[kN];
  for (int step = 0; step <= 20; ++step) {
    iscale = (-1.0F + 0.1F * static_cast<float>(step) + kMax) / (hi - lo);
    float sum_l = 0.0F;
    float sum_l2 = 0.0F;
    float sum_xl = 0.0F;
    for (int i = 0; i < kN; ++i) {
      const int l = min(15, max(0, __float2int_rn(iscale * (x[i] - lo))));
      trial[i] = static_cast<std::uint8_t>(l);
      const float w = weights[i];
      sum_l += w * static_cast<float>(l);
      sum_l2 += w * static_cast<float>(l * l);
      sum_xl += w * static_cast<float>(l) * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det > 0.0F) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
      if (this_min > 0.0F) {
        this_min = 0.0F;
        this_scale = sum_xl / sum_l2;
      }
      float error = 0.0F;
      for (int i = 0; i < kN; ++i) {
        const float diff =
            this_scale * static_cast<float>(trial[i]) + this_min - x[i];
        error += weights[i] * diff * diff;
      }
      if (error < best) {
        for (int i = 0; i < kN; ++i) {
          levels[i] = trial[i];
        }
        best = error;
        scale = this_scale;
        lo = this_min;
      }
    }
  }
  *the_min = -lo;
  return scale;
}

/// One thread per 32-value sub-block (one Q8_0 block); the eight threads of
/// a super-block share their fits through shared memory, as
/// quantize_row_q4_K_ref does within a block.
__global__ void __launch_bounds__(kRepackThreads)
    RepackQ8_0AsQ4KKernel(const std::uint8_t* __restrict__ src,
                          std::uint8_t* __restrict__ dst, std::uint32_t subs) {
  __shared__ float scales[kRepackThreads];
  __shared__ float mins[kRepackThreads];
  __shared__ std::uint8_t levels[kRepackThreads][32];
  const std::uint32_t sub = blockIdx.x * kRepackThreads + threadIdx.x;
  const bool live = sub < subs;
  float x[32];
  float weights[32];
  float sum_x2 = 0.0F;
  const std::uint8_t* block = src + static_cast<std::size_t>(sub) * 34;
  const float d8 = live ? Half(block) : 0.0F;
  for (int i = 0; i < 32; ++i) {
    x[i] = live
               ? d8 * static_cast<float>(static_cast<std::int8_t>(block[2 + i]))
               : 0.0F;
    sum_x2 += x[i] * x[i];
  }
  const float average = sqrtf(sum_x2 / 32.0F);
  for (int i = 0; i < 32; ++i) {
    weights[i] = average + fabsf(x[i]);
  }
  float the_min = 0.0F;
  scales[threadIdx.x] = FitScaleMin(x, weights, levels[threadIdx.x], &the_min);
  mins[threadIdx.x] = the_min;
  __syncthreads();
  const int first = threadIdx.x & ~7;
  const int j = threadIdx.x & 7;
  float max_scale = 0.0F;
  float max_min = 0.0F;
  for (int s = 0; s < 8; ++s) {
    max_scale = fmaxf(max_scale, scales[first + s]);
    max_min = fmaxf(max_min, mins[first + s]);
  }
  const float inv_scale = max_scale > 0.0F ? 63.0F / max_scale : 0.0F;
  const float inv_min = max_min > 0.0F ? 63.0F / max_min : 0.0F;
  const auto level = [&](float v, float inv) {
    return static_cast<std::uint8_t>(min(63, __float2int_rn(inv * v)));
  };
  const __half d = __float2half(max_scale / 63.0F);
  const __half dmin = __float2half(max_min / 63.0F);
  std::uint8_t* out = dst + static_cast<std::size_t>(sub / 8) * 144;
  if (live && j == 0) {
    std::uint8_t packed[12] = {};
    for (int s = 0; s < 8; ++s) {
      const std::uint8_t ls = level(scales[first + s], inv_scale);
      const std::uint8_t lm = level(mins[first + s], inv_min);
      if (s < 4) {
        packed[s] = ls;
        packed[s + 4] = lm;
      } else {
        packed[s + 4] =
            static_cast<std::uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
        packed[s - 4] |= static_cast<std::uint8_t>((ls >> 4) << 6);
        packed[s] |= static_cast<std::uint8_t>((lm >> 4) << 6);
      }
    }
    *reinterpret_cast<__half*>(out) = d;
    *reinterpret_cast<__half*>(out + 2) = dmin;
    for (int i = 0; i < 12; ++i) {
      out[4 + i] = packed[i];
    }
  }
  // Requantize against the rounded 6-bit scale and min.
  const float step = __half2float(d) *
                     static_cast<float>(level(scales[threadIdx.x], inv_scale));
  const float offset = __half2float(dmin) *
                       static_cast<float>(level(mins[threadIdx.x], inv_min));
  if (step != 0.0F) {
    for (int i = 0; i < 32; ++i) {
      levels[threadIdx.x][i] = static_cast<std::uint8_t>(
          min(15, max(0, __float2int_rn((x[i] + offset) / step))));
    }
  }
  __syncthreads();
  // Sub-blocks 2c and 2c+1 share 32 quant bytes (low and high nibbles).
  if (live && (j & 1) == 0) {
    std::uint8_t* qs = out + 16 + (j / 2) * 32;
    for (int i = 0; i < 32; ++i) {
      qs[i] = static_cast<std::uint8_t>(levels[threadIdx.x][i] |
                                        (levels[threadIdx.x + 1][i] << 4));
    }
  }
}

}  // namespace

bool LaunchKQuantGemv(GemvFormat format, const void* w, const float* x,
                      float* y, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (k % 256 != 0) {
    return false;
  }
  switch (format) {
    case GemvFormat::kQ4_0:
      Launch<GemvFormat::kQ4_0>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ4_K:
      Launch<GemvFormat::kQ4_K>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ5_K:
      Launch<GemvFormat::kQ5_K>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ6_K:
      Launch<GemvFormat::kQ6_K>(w, x, y, m, k, stream);
      return true;
    case GemvFormat::kQ8_0:
      Launch<GemvFormat::kQ8_0>(w, x, y, m, k, stream);
      return true;
  }
  return false;
}

bool LaunchHalfGemv(const void* w, const float* x, float* y, std::uint32_t rows,
                    std::uint32_t m, std::uint32_t k, hipStream_t stream) {
  if (k % 8 != 0 || m == 0) {
    return false;
  }
  return ForRows(
      rows, [&]<int kRows>() { LaunchHalf<kRows>(w, x, y, m, k, stream); });
}

void ConvertBf16ToHalf(void* data, std::size_t count, std::uint32_t* overflow,
                       hipStream_t stream) {
  constexpr int kThreads = 256;
  if (count == 0) {
    return;
  }
  const std::size_t blocks =
      std::min<std::size_t>((count + kThreads - 1) / kThreads, 4096);
  Bf16ToHalfKernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      static_cast<std::uint16_t*>(data), count, overflow);
}

void RepackQ8_0AsQ4K(const void* src, void* dst, std::uint32_t rows,
                     std::uint32_t cols, hipStream_t stream) {
  const std::uint32_t subs = rows * (cols / 32);
  RepackQ8_0AsQ4KKernel<<<(subs + kRepackThreads - 1) / kRepackThreads,
                          kRepackThreads, 0, stream>>>(
      static_cast<const std::uint8_t*>(src), static_cast<std::uint8_t*>(dst),
      subs);
}

}  // namespace gufo::models::gemma4::rocm
