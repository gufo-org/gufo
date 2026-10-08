// Gemma 4 26B-A4B routed experts for decode and verification: routing
// (router logits, top-k softmax weights, expert groups), the grouped routed
// projection over FP32 activations, and the mixture combine. Compiled with
// strict floating point: a row's result never depends on the batch it runs
// in, so verification rows equal single-token decode bit for bit.
#include "src/models/gemma4/kernels/rocm/moe.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "src/models/gemma4/kernels/rocm/gemv_tasks.hpp"
#include "src/models/gemma4/kernels/rocm/half_store.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

using namespace gemv_tasks;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWave);
  }
  return v;
}

/// Sum over the 256-thread block in wave order; every thread receives it.
__device__ inline float BlockSum(float v, float* scratch) {
  v = WaveSum(v);
  __syncthreads();
  if (threadIdx.x % kWave == 0) {
    scratch[threadIdx.x / kWave] = v;
  }
  __syncthreads();
  float total = 0.0F;
#pragma unroll
  for (int w = 0; w < kWaves; ++w) {
    total += scratch[w];
  }
  return total;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

constexpr int kRouterRows = 16;  // rows per router block
constexpr int kRouterVec = kMaxRouterHidden / (4 * kWave);

__global__ void ScaleRouterKernel(float* router, const float* scale,
                                  std::uint32_t hidden, std::size_t count) {
  const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (i < count) {
    router[i] = router[i] * scale[i % hidden];
  }
}

__device__ __forceinline__ void LoadRouterRow(const float* router,
                                              std::uint32_t e,
                                              std::uint32_t hidden,
                                              std::uint32_t vecs, int lane,
                                              float4 (&w)[kRouterVec]) {
  const auto* wr =
      reinterpret_cast<const float4*>(router + std::size_t{e} * hidden);
#pragma unroll
  for (std::uint32_t i = 0; i < kRouterVec; ++i) {
    if (i < vecs) {
      w[i] = wr[lane + kWave * i];
    }
  }
}

/// Row xr's logit for this wave's expert: each lane's FMA chain over its
/// float4 slots, wave sums of the dot and of the squares, scaled by
/// rms(x)^-1 / sqrt(hidden).
__device__ __forceinline__ float RouterLogit(const float4 (&w)[kRouterVec],
                                             const float4* xr,
                                             std::uint32_t vecs, int lane,
                                             std::uint32_t hidden, float eps) {
  float dot = 0.0F;
  float ss = 0.0F;
#pragma unroll
  for (std::uint32_t i = 0; i < kRouterVec; ++i) {
    if (i < vecs) {
      const float4 v = xr[lane + kWave * i];
      dot = __builtin_fmaf(w[i].x, v.x, dot);
      dot = __builtin_fmaf(w[i].y, v.y, dot);
      dot = __builtin_fmaf(w[i].z, v.z, dot);
      dot = __builtin_fmaf(w[i].w, v.w, dot);
      ss = __builtin_fmaf(v.x, v.x, ss);
      ss = __builtin_fmaf(v.y, v.y, ss);
      ss = __builtin_fmaf(v.z, v.z, ss);
      ss = __builtin_fmaf(v.w, v.w, ss);
    }
  }
  dot = WaveSum(dot);
  ss = WaveSum(ss);
  const float inv_root = 1.0F / sqrtf(static_cast<float>(hidden));
  const float r_scale =
      1.0F / sqrtf(ss / static_cast<float>(hidden) + eps) * inv_root;
  return dot * r_scale;
}

/// One wave per expert holds router[e] * router_scale in registers (float4
/// i of lane l covers elements 4 (l + 32 i) ..) and dots it with each row of
/// the block's row group.
__global__ void __launch_bounds__(kThreads)
    RouterKernel(const float* __restrict__ x, const float* __restrict__ router,
                 float* __restrict__ logits, std::uint32_t rows,
                 std::uint32_t hidden, std::uint32_t experts, float eps) {
  const int lane = threadIdx.x % kWave;
  const std::uint32_t e = blockIdx.y * kWaves + threadIdx.x / kWave;
  if (e >= experts) {
    return;
  }
  const std::uint32_t vecs = hidden / (4 * kWave);
  float4 w[kRouterVec];
  LoadRouterRow(router, e, hidden, vecs, lane, w);
  const std::uint32_t r0 = blockIdx.x * kRouterRows;
  const std::uint32_t r1 = min(rows, r0 + kRouterRows);
  for (std::uint32_t r = r0; r < r1; ++r) {
    const float logit = RouterLogit(
        w, reinterpret_cast<const float4*>(x + std::size_t{r} * hidden), vecs,
        lane, hidden, eps);
    if (lane == 0) {
      logits[std::size_t{r} * experts + e] = logit;
    }
  }
}

constexpr int kMaxExpertsPerLane = 8;  // 256 experts

/// Row r's top `used` experts (larger logit, then lower expert) and their
/// weights: `used` rounds of a wave argmax, then the softmax over the chosen
/// logits. Every lane of the wave calls it.
__device__ inline void TopKRow(const float* __restrict__ logits,
                               const float* __restrict__ expert_scale,
                               std::int32_t* __restrict__ ids,
                               float* __restrict__ weights,
                               std::uint32_t* __restrict__ counts,
                               std::uint32_t r, std::uint32_t experts,
                               std::uint32_t used, float row_scale, int lane) {
  float v[kMaxExpertsPerLane];
#pragma unroll
  for (int i = 0; i < kMaxExpertsPerLane; ++i) {
    const std::uint32_t e = lane + kWave * i;
    v[i] = e < experts ? logits[std::size_t{r} * experts + e] * row_scale
                       : -INFINITY;
  }
  float chosen[8];
  int chosen_id[8];
  for (std::uint32_t j = 0; j < used; ++j) {
    float best = -INFINITY;
    int best_id = 0x7FFFFFFF;
#pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) {
      const int e = lane + kWave * i;
      if (v[i] > best || (v[i] == best && e < best_id)) {
        best = v[i];
        best_id = e;
      }
    }
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset >>= 1) {
      const float ob = __shfl_xor(best, offset, kWave);
      const int oi = __shfl_xor(best_id, offset, kWave);
      if (ob > best || (ob == best && oi < best_id)) {
        best = ob;
        best_id = oi;
      }
    }
    chosen[j] = best;
    chosen_id[j] = best_id;
#pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) {
      if (lane + kWave * i == best_id) {
        v[i] = -INFINITY;
      }
    }
  }
  if (lane == 0) {
    float total = 0.0F;
    float p[8];
    for (std::uint32_t j = 0; j < used; ++j) {
      p[j] = expf(chosen[j] - chosen[0]);
      total += p[j];
    }
    for (std::uint32_t j = 0; j < used; ++j) {
      const std::size_t slot = std::size_t{r} * used + j;
      ids[slot] = chosen_id[j];
      weights[slot] = p[j] / total * expert_scale[chosen_id[j]];
      if (counts != nullptr) {
        atomicAdd(counts + chosen_id[j], 1U);
      }
    }
  }
}

/// One wave per row: `used` rounds of a wave argmax (larger logit, then
/// lower expert), then the softmax over the chosen logits.
__global__ void __launch_bounds__(kThreads)
    TopKKernel(const float* __restrict__ logits,
               const float* __restrict__ expert_scale,
               std::int32_t* __restrict__ ids, float* __restrict__ weights,
               std::uint32_t* __restrict__ counts, std::uint32_t rows,
               std::uint32_t experts, std::uint32_t used,
               const float* __restrict__ x, std::uint32_t hidden, float eps) {
  const int lane = threadIdx.x % kWave;
  const std::uint32_t r = blockIdx.x * kWaves + threadIdx.x / kWave;
  if (r >= rows) {
    return;
  }
  // With `x` the logits are raw dot products, scaled here by the row's
  // rms(x) / sqrt(hidden) as RouterKernel scales them.
  float row_scale = 1.0F;
  if (x != nullptr) {
    const float* xr = x + std::size_t{r} * hidden;
    float ss = 0.0F;
    for (std::uint32_t i = lane; i < hidden; i += kWave) {
      ss = __builtin_fmaf(xr[i], xr[i], ss);
    }
    ss = WaveSum(ss);
    row_scale = 1.0F / sqrtf(ss / static_cast<float>(hidden) + eps) *
                (1.0F / sqrtf(static_cast<float>(hidden)));
  }
  TopKRow(logits, expert_scale, ids, weights, counts, r, experts, used,
          row_scale, lane);
}

/// The group table of `slots` expert ids, built by one 256-thread block:
/// one thread per expert collects its slots in order; a block scan over the
/// non-empty experts places the groups.
__device__ inline void BuildGroups(const std::int32_t* __restrict__ ids,
                                   std::int32_t* __restrict__ groups,
                                   std::uint32_t slots, std::uint32_t experts) {
  __shared__ std::int32_t s_ids[kMaxGroupSlots * 8];
  __shared__ int scan[kThreads];
  for (std::uint32_t i = threadIdx.x; i < slots; i += kThreads) {
    s_ids[i] = ids[i];
  }
  __syncthreads();
  const auto e = static_cast<std::int32_t>(threadIdx.x);
  int mine[kMaxGroupSlots];
  int n = 0;
  if (static_cast<std::uint32_t>(e) < experts) {
    for (std::uint32_t s = 0; s < slots; ++s) {
      if (s_ids[s] == e && n < static_cast<int>(kMaxGroupSlots)) {
        mine[n++] = static_cast<int>(s);
      }
    }
  }
  scan[threadIdx.x] = n > 0 ? 1 : 0;
  __syncthreads();
  for (int offset = 1; offset < kThreads; offset <<= 1) {
    const int add = threadIdx.x >= static_cast<unsigned>(offset)
                        ? scan[threadIdx.x - offset]
                        : 0;
    __syncthreads();
    scan[threadIdx.x] += add;
    __syncthreads();
  }
  if (n > 0) {
    std::int32_t* g = groups + 1 + (scan[threadIdx.x] - 1) * kGroupInts;
    g[0] = e;
    g[1] = n;
    for (int i = 0; i < n; ++i) {
      g[2 + i] = mine[i];
    }
  }
  if (threadIdx.x == kThreads - 1) {
    groups[0] = scan[kThreads - 1];
  }
}

__global__ void __launch_bounds__(kThreads)
    GroupsKernel(const std::int32_t* __restrict__ ids,
                 std::int32_t* __restrict__ groups, std::uint32_t slots,
                 std::uint32_t experts) {
  BuildGroups(ids, groups, slots, experts);
}

/// Routing at grouped widths in one launch: RouterKernel's blocks (one row
/// group, kWaves experts each), and the last block to finish, counted on
/// `sync` (zero between launches), selects every row's experts and builds
/// the group table. The arithmetic is RouterKernel's, TopKKernel's and
/// GroupsKernel's.
__global__ void __launch_bounds__(kThreads) RouteGroupedKernel(
    const float* __restrict__ x, const float* __restrict__ router,
    float* __restrict__ logits, const float* __restrict__ expert_scale,
    std::int32_t* __restrict__ ids, float* __restrict__ weights,
    std::int32_t* __restrict__ groups, std::uint32_t* __restrict__ sync,
    std::uint32_t rows, std::uint32_t hidden, std::uint32_t experts,
    std::uint32_t used, float eps) {
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const std::uint32_t e = blockIdx.y * kWaves + wave;
  if (e < experts) {
    const std::uint32_t vecs = hidden / (4 * kWave);
    float4 w[kRouterVec];
    LoadRouterRow(router, e, hidden, vecs, lane, w);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const float logit = RouterLogit(
          w, reinterpret_cast<const float4*>(x + std::size_t{r} * hidden), vecs,
          lane, hidden, eps);
      if (lane == 0) {
        logits[std::size_t{r} * experts + e] = logit;
      }
    }
  }
  // Publish this block's logits, then count it.
  __threadfence();
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) {
    last = atomicAdd(sync, 1U) == gridDim.y - 1;
  }
  __syncthreads();
  if (!last) {
    return;
  }
  __threadfence();  // every block's logits are visible past the count
  for (std::uint32_t r = wave; r < rows; r += kWaves) {
    TopKRow(logits, expert_scale, ids, weights, nullptr, r, experts, used, 1.0F,
            lane);
  }
  __syncthreads();
  BuildGroups(ids, groups, rows * used, experts);
  if (threadIdx.x == 0) {
    *sync = 0;
  }
}

// ---------------------------------------------------------------------------
// Grouped routed projection
// ---------------------------------------------------------------------------

/// Storage of each format. K-quants: bytes per 256-value super-block, lanes
/// per row (tasks per super-block) and values per task. Q8_0/Q5_1/binary16:
/// bytes per 32-value block; eight lanes take eight consecutive blocks.
template<ExpertFormat F>
struct Traits;
template<>
struct Traits<ExpertFormat::kQ4_K> {
  static constexpr int kBytes = 144;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
  static constexpr bool kBlockwise = false;
};
template<>
struct Traits<ExpertFormat::kQ5_K> {
  static constexpr int kBytes = 176;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
  static constexpr bool kBlockwise = false;
};
template<>
struct Traits<ExpertFormat::kQ6_K> {
  static constexpr int kBytes = 210;
  static constexpr int kTasks = 4;
  static constexpr int kValues = 64;
  static constexpr bool kBlockwise = false;
};
template<>
struct Traits<ExpertFormat::kQ8_0> {
  static constexpr int kBytes = 34;
  static constexpr int kTasks = 8;
  static constexpr int kValues = 32;
  static constexpr bool kBlockwise = true;
};
template<>
struct Traits<ExpertFormat::kQ5_1> {
  static constexpr int kBytes = 24;
  static constexpr int kTasks = 8;
  static constexpr int kValues = 32;
  static constexpr bool kBlockwise = true;
};
template<>
struct Traits<ExpertFormat::kF16> {
  static constexpr int kBytes = 64;
  static constexpr int kTasks = 8;
  static constexpr int kValues = 32;
  static constexpr bool kBlockwise = true;
};

template<ExpertFormat F>
__host__ __device__ constexpr std::size_t RowBytes(std::uint32_t k) {
  using T = Traits<F>;
  return T::kBlockwise ? std::size_t{k} / 32 * T::kBytes
                       : std::size_t{k} / 256 * T::kBytes;
}

/// Decodes lane task `t` of 256-value unit `unit` of a row; `offset` is the
/// activation index of w[0]. Returns false past the row's end (blockwise
/// formats whose width is not a multiple of 256).
template<ExpertFormat F>
__device__ __forceinline__ bool Decode(const std::uint8_t* row,
                                       std::uint32_t unit, int t,
                                       std::uint32_t k,
                                       float (&w)[Traits<F>::kValues],
                                       std::uint32_t* offset) {
  using T = Traits<F>;
  if constexpr (T::kBlockwise) {
    const std::uint32_t block = unit * 8 + static_cast<std::uint32_t>(t);
    if (block >= k / 32) {
      return false;
    }
    *offset = block * 32;
    const std::uint8_t* p = row + std::size_t{block} * T::kBytes;
    if constexpr (F == ExpertFormat::kQ8_0) {
      // Rows are word aligned (a multiple of eight blocks is 272 bytes and
      // expert widths keep k / 32 even), so odd blocks start mid-word.
      DecodeQ8_0(p, (block & 1U) != 0, w);
    } else if constexpr (F == ExpertFormat::kF16) {
      const auto* v = reinterpret_cast<const uint4*>(p);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 q = v[i];
        const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const __half2 pair = __builtin_bit_cast(__half2, words[j]);
          w[8 * i + 2 * j] = __low2float(pair);
          w[8 * i + 2 * j + 1] = __high2float(pair);
        }
      }
    } else {
      DecodeQ5_1(p, w);
    }
  } else {
    const std::uint8_t* p = row + std::size_t{unit} * T::kBytes;
    if constexpr (F == ExpertFormat::kQ6_K) {
      DecodeQ6K(p, t, w);
      *offset = unit * 256 + OffsetQ6K(t);
    } else {
      DecodeQ45KPair<F == ExpertFormat::kQ5_K>(p, t, w);
      *offset = unit * 256 + 64 * t;
    }
  }
  return true;
}

/// acc += w . x for one decoded task: 16-value runs at x, x + 32, ... for
/// Q6_K (the super-block interleave), contiguous for the others.
template<ExpertFormat F>
__device__ __forceinline__ float Contract(float acc,
                                          const float (&w)[Traits<F>::kValues],
                                          const float* x) {
  constexpr int kRuns = Traits<F>::kValues / 16;
  constexpr int kStride = F == ExpertFormat::kQ6_K ? 32 : 16;
#pragma unroll
  for (int r = 0; r < kRuns; ++r) {
    acc = Accumulate<16>(acc, w + 16 * r, x + kStride * r);
  }
  return acc;
}

constexpr int kSlotsPerPass = 4;

/// Workgroup: kWaves / kSplit wave groups of kRowsPerWave output rows; the
/// kSplit waves of a group take interleaved 256-value units of the
/// reduction and are summed in wave order. blockIdx.y is the expert group.
/// gelu_tanh(x) * u.
__device__ inline float GeGlu(float x, float u) {
  constexpr float kSqrt2OverPi = 0.79788456080286535587989211986876F;
  constexpr float kCoefA = 0.044715F;
  const float g =
      0.5F * x * (1.0F + tanhf(kSqrt2OverPi * x * (1.0F + kCoefA * x * x)));
  return g * u;
}

/// kGeGlu: the first half of the wave groups take gate rows, the second
/// half the matching up rows (m / 2 further), and the block writes
/// GeGlu(gate, up) rows of width m / 2.
template<ExpertFormat F, int kSplit, bool kGeGlu, int kSlots = kSlotsPerPass>
__global__ void __launch_bounds__(kThreads)
    RoutedGemvKernel(const std::uint8_t* __restrict__ w,
                     const std::int32_t* __restrict__ groups,
                     const float* __restrict__ x, std::uint32_t x_div,
                     float* __restrict__ y, std::uint32_t m, std::uint32_t k) {
  using T = Traits<F>;
  constexpr int kRowsPerWave = kWave / T::kTasks;
  constexpr int kGroups = kWaves / kSplit;
  constexpr int kRowsPerBlock = kRowsPerWave * kGroups;
  static_assert(!kGeGlu || kGroups % 2 == 0);
  __shared__ float partial[kSlots][kSplit][kWaves / kSplit][kWave];
  if (static_cast<int>(blockIdx.y) >= groups[0]) {
    return;
  }
  const std::int32_t* group = groups + 1 + blockIdx.y * kGroupInts;
  const int expert = group[0];
  const int count = group[1];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int slice = wave % kSplit;
  const int wave_group = wave / kSplit;
  const int t = lane % T::kTasks;
  const std::uint32_t half_m = m / 2;
  const std::uint32_t row =
      kGeGlu
          ? (wave_group / (kGroups / 2)) * half_m +
                blockIdx.x * (kRowsPerBlock / 2) +
                (wave_group % (kGroups / 2)) * kRowsPerWave + lane / T::kTasks
          : blockIdx.x * kRowsPerBlock + wave_group * kRowsPerWave +
                lane / T::kTasks;
  const std::uint32_t units = (k + 255) / 256;
  const std::uint8_t* base =
      w + (std::size_t{static_cast<std::uint32_t>(expert)} * m + row) *
              RowBytes<F>(k);
  for (int s0 = 0; s0 < count; s0 += kSlots) {
    float acc[kSlots];
    const float* xs[kSlots];
    int slot[kSlots];
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
      acc[j] = 0.0F;
      slot[j] = s0 + j < count ? group[2 + s0 + j] : -1;
      xs[j] =
          x +
          std::size_t{static_cast<std::uint32_t>(max(slot[j], 0)) / x_div} * k;
    }
    if (row < m) {
      for (std::uint32_t unit = slice; unit < units; unit += kSplit) {
        float wv[T::kValues];
        std::uint32_t offset = 0;
        if (!Decode<F>(base, unit, t, k, wv, &offset)) {
          continue;
        }
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            acc[j] = Contract<F>(acc[j], wv, xs[j] + offset);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
#pragma unroll
      for (int offset = T::kTasks / 2; offset > 0; offset >>= 1) {
        acc[j] += __shfl_xor(acc[j], offset, kWave);
      }
    }
    if constexpr (kGeGlu) {
#pragma unroll
      for (int j = 0; j < kSlots; ++j) {
        partial[j][slice][wave_group][lane] = acc[j];
      }
      __syncthreads();
      // Row sums in wave order, as without the fusion.
      if (slice == 0 && t == 0 && wave_group < kGroups / 2 && row < half_m) {
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            float gate = partial[j][0][wave_group][lane];
            float up = partial[j][0][wave_group + kGroups / 2][lane];
#pragma unroll
            for (int s = 1; s < kSplit; ++s) {
              gate += partial[j][s][wave_group][lane];
              up += partial[j][s][wave_group + kGroups / 2][lane];
            }
            y[std::size_t{static_cast<std::uint32_t>(slot[j])} * half_m + row] =
                GeGlu(gate, up);
          }
        }
      }
      __syncthreads();
    } else if constexpr (kSplit == 1) {
      if (t == 0 && row < m) {
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            y[std::size_t{static_cast<std::uint32_t>(slot[j])} * m + row] =
                acc[j];
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < kSlots; ++j) {
        partial[j][slice][wave_group][lane] = acc[j];
      }
      __syncthreads();
      if (slice == 0 && t == 0 && row < m) {
#pragma unroll
        for (int j = 0; j < kSlots; ++j) {
          if (slot[j] >= 0) {
            float sum = partial[j][0][wave_group][lane];
#pragma unroll
            for (int s = 1; s < kSplit; ++s) {
              sum += partial[j][s][wave_group][lane];
            }
            y[std::size_t{static_cast<std::uint32_t>(slot[j])} * m + row] = sum;
          }
        }
      }
      __syncthreads();
    }
  }
}

template<ExpertFormat F, bool kGeGlu>
void LaunchRouted(const void* w, const std::int32_t* groups,
                  std::uint32_t max_groups, const float* x, std::uint32_t x_div,
                  float* y, std::uint32_t m, std::uint32_t k,
                  hipStream_t stream, bool single) {
  constexpr int kRowsPerWave = kWave / Traits<F>::kTasks;
  const auto* weights = static_cast<const std::uint8_t*>(w);
  // A GeGLU launch covers m / 2 gate rows and their up rows.
  const std::uint32_t rows = kGeGlu ? m / 2 : m;
  constexpr int kShare = kGeGlu ? 2 : 1;
  // Measured with cold experts on gfx1151: short reductions (the 704-wide
  // down projection) run whole rows per wave; the 2816-wide gate/up splits
  // its 11 super-blocks over four waves (eight leave most slices one).
  if (k < 2048) {
    constexpr int kRows = kRowsPerWave * kWaves / kShare;
    const dim3 grid((rows + kRows - 1) / kRows, max_groups);
    if (single) {
      RoutedGemvKernel<F, 1, kGeGlu, 1>
          <<<grid, kThreads, 0, stream>>>(weights, groups, x, x_div, y, m, k);
    } else {
      RoutedGemvKernel<F, 1, kGeGlu>
          <<<grid, kThreads, 0, stream>>>(weights, groups, x, x_div, y, m, k);
    }
    return;
  }
  constexpr int kSplit = 4;
  constexpr int kRows = kRowsPerWave * (kWaves / kSplit) / kShare;
  const dim3 grid((rows + kRows - 1) / kRows, max_groups);
  if (single) {
    RoutedGemvKernel<F, kSplit, kGeGlu, 1>
        <<<grid, kThreads, 0, stream>>>(weights, groups, x, x_div, y, m, k);
  } else {
    RoutedGemvKernel<F, kSplit, kGeGlu>
        <<<grid, kThreads, 0, stream>>>(weights, groups, x, x_div, y, m, k);
  }
}

template<ExpertFormat F>
void LaunchRouted(const void* w, const std::int32_t* groups,
                  std::uint32_t max_groups, const float* x, std::uint32_t x_div,
                  float* y, std::uint32_t m, std::uint32_t k,
                  hipStream_t stream, bool geglu, bool single) {
  if (geglu) {
    LaunchRouted<F, true>(w, groups, max_groups, x, x_div, y, m, k, stream,
                          single);
  } else {
    LaunchRouted<F, false>(w, groups, max_groups, x, x_div, y, m, k, stream,
                           single);
  }
}

// ---------------------------------------------------------------------------
// Feed-forward residual
// ---------------------------------------------------------------------------

/// Widest row MoeFinish holds in registers.
constexpr std::uint32_t kFinishElements = 6144;

/// Sum over the block in wave order; every thread receives it.
template<int kBlock>
__device__ inline float WideBlockSum(float v, float* scratch) {
  constexpr int kBlockWaves = kBlock / kWave;
  v = WaveSum(v);
  __syncthreads();
  if (threadIdx.x % kWave == 0) {
    scratch[threadIdx.x / kWave] = v;
  }
  __syncthreads();
  float total = 0.0F;
#pragma unroll
  for (int w = 0; w < kBlockWaves; ++w) {
    total += scratch[w];
  }
  return total;
}

template<int kFinishThreads>
__global__ void __launch_bounds__(kFinishThreads)
    MoeFinishKernel(MoeFinishArgs a) {
  constexpr std::uint32_t kFinishRegisters = kFinishElements / kFinishThreads;
  __shared__ float scratch[kFinishThreads / kWave];
  const std::size_t base = std::size_t{blockIdx.x} * a.hidden;
  const float* e = a.experts + base * a.used;
  const float* wr = a.weights + std::size_t{blockIdx.x} * a.used;
  const float n = static_cast<float>(a.hidden);
  float dv[kFinishRegisters];
  float mv[kFinishRegisters];
  float ss_d = 0.0F;
  float ss_m = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kFinishRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kFinishThreads;
    dv[j] = 0.0F;
    mv[j] = 0.0F;
    if (i < a.hidden) {
      dv[j] = a.dense[base + i];
      float sum = 0.0F;
      for (std::uint32_t s = 0; s < a.used; ++s) {
        sum = __builtin_fmaf(wr[s], e[std::size_t{s} * a.hidden + i], sum);
      }
      mv[j] = sum;
      ss_d = __builtin_fmaf(dv[j], dv[j], ss_d);
      ss_m = __builtin_fmaf(mv[j], mv[j], ss_m);
    }
  }
  const float rd =
      1.0F / sqrtf(WideBlockSum<kFinishThreads>(ss_d, scratch) / n + a.eps);
  const float rm =
      1.0F / sqrtf(WideBlockSum<kFinishThreads>(ss_m, scratch) / n + a.eps);
  float ss_f = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kFinishRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kFinishThreads;
    if (i < a.hidden) {
      dv[j] = dv[j] * rd * a.norm1[i] + mv[j] * rm * a.norm2[i];
      ss_f = __builtin_fmaf(dv[j], dv[j], ss_f);
    }
  }
  const float rf =
      1.0F / sqrtf(WideBlockSum<kFinishThreads>(ss_f, scratch) / n + a.eps);
  float ss_x = 0.0F;
#pragma unroll
  for (std::uint32_t j = 0; j < kFinishRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kFinishThreads;
    if (i < a.hidden) {
      mv[j] = (a.x[base + i] + dv[j] * rf * a.post_norm[i]) * a.scale;
      a.x[base + i] = mv[j];
      ss_x = __builtin_fmaf(mv[j], mv[j], ss_x);
    }
  }
  if (a.next_norm == nullptr) {
    return;
  }
  const float rx =
      1.0F / sqrtf(WideBlockSum<kFinishThreads>(ss_x, scratch) / n + a.eps);
#pragma unroll
  for (std::uint32_t j = 0; j < kFinishRegisters; ++j) {
    const std::uint32_t i = threadIdx.x + j * kFinishThreads;
    if (i < a.hidden) {
      const float v = mv[j] * rx * a.next_norm[i];
      if (a.h != nullptr) {
        a.h[base + i] = v;
      }
      if (a.h_half != nullptr) {
        static_cast<__half*>(a.h_half)[base + i] = HalfOf(v);
      }
    }
  }
}

/// Tile index of a map entry past the last tile: beyond every bucket.
constexpr std::int32_t kDeadTile = 0x7FFF;

/// One block: thread e counts expert e's tiles of both heights, an ordered
/// scan places them, and the remaining entries become dead tiles.
__global__ void __launch_bounds__(256)
    BuildRoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                           std::uint32_t experts, std::uint32_t rows_a,
                           std::uint32_t capacity_a,
                           std::int32_t* __restrict__ tiles_a,
                           std::uint32_t rows_b, std::uint32_t capacity_b,
                           std::int32_t* __restrict__ tiles_b) {
  __shared__ std::uint32_t scan_a[256];
  __shared__ std::uint32_t scan_b[256];
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t padded = e < experts ? (counts[e] + 15U) / 16U * 16U : 0;
  const std::uint32_t n_a = (padded + rows_a - 1) / rows_a;
  const std::uint32_t n_b = (padded + rows_b - 1) / rows_b;
  scan_a[e] = n_a;
  scan_b[e] = n_b;
  __syncthreads();
  // Inclusive Hillis-Steele scan; integer sums, so order-free.
  for (std::uint32_t step = 1; step < 256; step <<= 1) {
    const std::uint32_t add_a = e >= step ? scan_a[e - step] : 0;
    const std::uint32_t add_b = e >= step ? scan_b[e - step] : 0;
    __syncthreads();
    scan_a[e] += add_a;
    scan_b[e] += add_b;
    __syncthreads();
  }
  const std::uint32_t first_a = scan_a[e] - n_a;
  const std::uint32_t first_b = scan_b[e] - n_b;
  for (std::uint32_t j = 0; j < n_a; ++j) {
    tiles_a[first_a + j] = static_cast<std::int32_t>(e | (j << 16));
  }
  for (std::uint32_t j = 0; j < n_b; ++j) {
    tiles_b[first_b + j] = static_cast<std::int32_t>(e | (j << 16));
  }
  constexpr std::int32_t kDead = kDeadTile << 16;
  for (std::uint32_t i = scan_a[255] + e; i < capacity_a; i += 256) {
    tiles_a[i] = kDead;
  }
  for (std::uint32_t i = scan_b[255] + e; i < capacity_b; i += 256) {
    tiles_b[i] = kDead;
  }
}

}  // namespace

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream) {
  if (experts > 256) {
    throw std::invalid_argument("routed tile map supports 256 experts");
  }
  BuildRoutedTilesKernel<<<1, 256, 0, stream>>>(counts, experts, rows_a,
                                                capacity_a, tiles_a, rows_b,
                                                capacity_b, tiles_b);
}

void MoeRoute(const MoeRouteArgs& a, hipStream_t stream) {
  if (a.sync != nullptr && a.groups != nullptr && !a.raw_logits &&
      a.rows <= kRouterRows) {
    RouteGroupedKernel<<<dim3(1, (a.experts + kWaves - 1) / kWaves), kThreads,
                         0, stream>>>(
        a.x, a.router, a.logits, a.expert_scale, a.ids, a.weights, a.groups,
        a.sync, a.rows, a.hidden, a.experts, a.used, a.eps);
    return;
  }
  if (!a.raw_logits) {
    RouterKernel<<<dim3((a.rows + kRouterRows - 1) / kRouterRows,
                        (a.experts + kWaves - 1) / kWaves),
                   kThreads, 0, stream>>>(a.x, a.router, a.logits, a.rows,
                                          a.hidden, a.experts, a.eps);
  }
  TopKKernel<<<(a.rows + kWaves - 1) / kWaves, kThreads, 0, stream>>>(
      a.logits, a.expert_scale, a.ids, a.weights, a.counts, a.rows, a.experts,
      a.used, a.raw_logits ? a.x : nullptr, a.hidden, a.eps);
  if (a.groups != nullptr) {
    GroupsKernel<<<1, kThreads, 0, stream>>>(a.ids, a.groups, a.rows * a.used,
                                             a.experts);
  }
}

bool LaunchRoutedGemv(ExpertFormat format, const void* w,
                      const std::int32_t* groups, std::uint32_t max_groups,
                      const float* x, std::uint32_t x_div, float* y,
                      std::uint32_t m, std::uint32_t k, hipStream_t stream,
                      bool geglu, bool single) {
  if (max_groups == 0 || x_div == 0 || (geglu && m % 2 != 0)) {
    return false;
  }
  switch (format) {
    case ExpertFormat::kQ4_K:
      if (k % 256 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kQ4_K>(w, groups, max_groups, x, x_div, y, m,
                                        k, stream, geglu, single);
      return true;
    case ExpertFormat::kQ5_K:
      if (k % 256 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kQ5_K>(w, groups, max_groups, x, x_div, y, m,
                                        k, stream, geglu, single);
      return true;
    case ExpertFormat::kQ6_K:
      if (k % 256 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kQ6_K>(w, groups, max_groups, x, x_div, y, m,
                                        k, stream, geglu, single);
      return true;
    case ExpertFormat::kQ8_0:
      // Word-aligned rows: an even number of 34-byte blocks.
      if (k % 64 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kQ8_0>(w, groups, max_groups, x, x_div, y, m,
                                        k, stream, geglu, single);
      return true;
    case ExpertFormat::kQ5_1:
      if (k % 32 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kQ5_1>(w, groups, max_groups, x, x_div, y, m,
                                        k, stream, geglu, single);
      return true;
    case ExpertFormat::kF16:
      if (k % 32 != 0) {
        return false;
      }
      LaunchRouted<ExpertFormat::kF16>(w, groups, max_groups, x, x_div, y, m, k,
                                       stream, geglu, single);
      return true;
    case ExpertFormat::kQ4_0:
      return false;
  }
  return false;
}

void MoeFinish(const MoeFinishArgs& args, hipStream_t stream) {
  // One row per 1024-thread block keeps six values per thread in registers
  // (narrower blocks spill the row).
  MoeFinishKernel<1024><<<args.rows, 1024, 0, stream>>>(args);
}

void ScaleRouter(float* router, const float* scale, std::uint32_t experts,
                 std::uint32_t hidden, hipStream_t stream) {
  const std::size_t count = std::size_t{experts} * hidden;
  ScaleRouterKernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      router, scale, hidden, count);
}

}  // namespace gufo::models::gemma4::rocm
