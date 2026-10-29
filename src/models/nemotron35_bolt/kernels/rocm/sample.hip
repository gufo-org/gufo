#include "src/models/nemotron35_bolt/kernels/rocm/sample.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::models::nemotron35_bolt::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kPerThread = 16;
constexpr int kSpan = kThreads * kPerThread;  ///< Logits per block.
/// Bins of 1/16 nat below the maximum at the sampling temperature, 32 nats
/// deep, then one bin for everything further below.
constexpr float kBinsPerNat = 16.0F;
constexpr int kBins = 512;
constexpr int kHistogram = kBins + 1;

struct Scratch {
  float* part_max;   ///< [rows][blocks]
  double* part_sum;  ///< [rows][blocks]
  float* part_mass;  ///< [rows][blocks][kHistogram]
  int* part_count;   ///< [rows][blocks][kHistogram]
  int* cut;          ///< [rows] last bin of the head
  int* fill;         ///< [rows] candidates written
};

std::size_t AlignUp(std::size_t n) { return (n + 255) & ~std::size_t{255}; }

Scratch Layout(void* base, int rows, int blocks, std::size_t* bytes) {
  const auto parts = static_cast<std::size_t>(rows) * blocks;
  std::size_t offset = 0;
  const auto take = [&](std::size_t size) {
    const std::size_t at = offset;
    offset += AlignUp(size);
    return static_cast<char*>(base) + at;
  };
  Scratch s{};
  s.part_max = reinterpret_cast<float*>(take(parts * sizeof(float)));
  s.part_sum = reinterpret_cast<double*>(take(parts * sizeof(double)));
  s.part_mass =
      reinterpret_cast<float*>(take(parts * kHistogram * sizeof(float)));
  s.part_count = reinterpret_cast<int*>(take(parts * kHistogram * sizeof(int)));
  s.cut = reinterpret_cast<int*>(take(rows * sizeof(int)));
  s.fill = reinterpret_cast<int*>(take(rows * sizeof(int)));
  if (bytes != nullptr)
    *bytes = offset;
  return s;
}

/// The bin of a finite logit; monotone in the logit, so a head of whole bins
/// is every token above some value, ties included.
__device__ __forceinline__ int HeadBin(float x, float maximum,
                                       float inv_temperature) {
  const float depth = (maximum - x) * inv_temperature;
  return depth < kBins / kBinsPerNat ? static_cast<int>(depth * kBinsPerNat)
                                     : kBins;
}

__device__ __forceinline__ float RowMaximum(const float* part_max,
                                            int blocks) {
  float m = -INFINITY;
  for (int k = 0; k < blocks; ++k)
    m = fmaxf(m, part_max[k]);
  return m;
}

__global__ __launch_bounds__(kThreads) void HeadMaxKernel(
    const float* __restrict__ logits, int vocab, float* __restrict__ part_max) {
  __shared__ float s_max[kThreads];
  const int tid = static_cast<int>(threadIdx.x);
  const float* x = logits + static_cast<std::size_t>(blockIdx.y) * vocab;
  float m = -INFINITY;
  for (int i = 0; i < kPerThread; ++i) {
    const int index = static_cast<int>(blockIdx.x) * kSpan + i * kThreads + tid;
    if (index < vocab && isfinite(x[index]))
      m = fmaxf(m, x[index]);
  }
  s_max[tid] = m;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
    __syncthreads();
  }
  if (tid == 0)
    part_max[blockIdx.y * gridDim.x + blockIdx.x] = s_max[0];
}

/// Per block: the F64 softmax sum and the F32 mass and count per bin.
__global__ __launch_bounds__(kThreads) void HeadSumKernel(
    const float* __restrict__ logits, int vocab, float temperature,
    float inv_temperature, Scratch s) {
  __shared__ float s_mass[kHistogram];
  __shared__ int s_count[kHistogram];
  __shared__ double s_sum[kThreads];
  const int tid = static_cast<int>(threadIdx.x);
  const int blocks = static_cast<int>(gridDim.x);
  const int row = static_cast<int>(blockIdx.y);
  const int part = row * blocks + static_cast<int>(blockIdx.x);
  for (int b = tid; b < kHistogram; b += kThreads) {
    s_mass[b] = 0.0F;
    s_count[b] = 0;
  }
  const float m = RowMaximum(s.part_max + row * blocks, blocks);
  __syncthreads();
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  double sum = 0.0;
  for (int i = 0; i < kPerThread; ++i) {
    const int index = static_cast<int>(blockIdx.x) * kSpan + i * kThreads + tid;
    if (index >= vocab || !isfinite(x[index]))
      continue;
    const double e = exp((static_cast<double>(x[index]) - m) /
                         static_cast<double>(temperature));
    sum += e;
    const int b = HeadBin(x[index], m, inv_temperature);
    atomicAdd(&s_mass[b], static_cast<float>(e));
    atomicAdd(&s_count[b], 1);
  }
  s_sum[tid] = sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  if (tid == 0)
    s.part_sum[part] = s_sum[0];
  for (int b = tid; b < kHistogram; b += kThreads) {
    s.part_mass[static_cast<std::size_t>(part) * kHistogram + b] = s_mass[b];
    s.part_count[static_cast<std::size_t>(part) * kHistogram + b] = s_count[b];
  }
}

/// Per row: the softmax sum and the first bin whose cumulative count or mass
/// covers the filters; the mass target carries a margin over the bins' F32
/// rounding.
__global__ __launch_bounds__(kThreads) void HeadCutKernel(
    LogitHeadParams p, int blocks, Scratch s,
    LogitHeadRecord* __restrict__ records) {
  __shared__ float s_mass[kHistogram];
  __shared__ int s_count[kHistogram];
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  for (int b = tid; b < kHistogram; b += kThreads) {
    float mass = 0.0F;
    int count = 0;
    for (int k = 0; k < blocks; ++k) {
      const std::size_t at =
          static_cast<std::size_t>(row * blocks + k) * kHistogram + b;
      mass += s.part_mass[at];
      count += s.part_count[at];
    }
    s_mass[b] = mass;
    s_count[b] = count;
  }
  __syncthreads();
  if (tid != 0)
    return;
  const float m = RowMaximum(s.part_max + row * blocks, blocks);
  double total = 0.0;
  for (int k = 0; k < blocks; ++k)
    total += s.part_sum[row * blocks + k];
  const double target =
      fmin(1.0, static_cast<double>(p.top_p) * (1.0 + 1e-3)) * total;
  double mass = 0.0;
  int count = 0;
  int cut = kBins;
  for (int b = 0; b < kHistogram; ++b) {
    mass += s_mass[b];
    count += s_count[b];
    const bool covered = p.keep > 0 ? count >= p.keep
                                    : count >= p.minimum && mass >= target;
    if (covered) {
      cut = b;
      break;
    }
  }
  if (cut == kBins) {
    // Everything finite: the full count, covering or not.
    count = 0;
    for (int b = 0; b < kHistogram; ++b)
      count += s_count[b];
  }
  records[row] = {total, m,
                  m > -INFINITY && count <= kLogitHeadCandidates ? count : 0};
  s.cut[row] = cut;
  s.fill[row] = 0;
}

__global__ __launch_bounds__(kThreads) void HeadGatherKernel(
    const float* __restrict__ logits, int vocab, float inv_temperature,
    Scratch s, const LogitHeadRecord* __restrict__ records,
    LogitHeadCandidate* __restrict__ candidates) {
  const int row = static_cast<int>(blockIdx.y);
  const LogitHeadRecord record = records[row];
  if (record.count == 0)
    return;
  const int cut = s.cut[row];
  const float* x = logits + static_cast<std::size_t>(row) * vocab;
  LogitHeadCandidate* out =
      candidates + static_cast<std::size_t>(row) * kLogitHeadCandidates;
  for (int i = 0; i < kPerThread; ++i) {
    const int index = static_cast<int>(blockIdx.x) * kSpan + i * kThreads +
                      static_cast<int>(threadIdx.x);
    if (index >= vocab || !isfinite(x[index]) ||
        HeadBin(x[index], record.maximum, inv_temperature) > cut)
      continue;
    const int slot = atomicAdd(&s.fill[row], 1);
    out[slot] = {static_cast<std::uint32_t>(index), x[index]};
  }
}

}  // namespace

std::size_t LogitHeadScratchBytes(int rows, int vocab) {
  std::size_t bytes = 0;
  static_cast<void>(
      Layout(nullptr, rows, (vocab + kSpan - 1) / kSpan, &bytes));
  return bytes;
}

hipError_t LogitHeads(const float* logits, int rows, int vocab,
                      const LogitHeadParams& params, void* scratch,
                      LogitHeadRecord* records, LogitHeadCandidate* candidates,
                      hipStream_t stream) {
  if (rows <= 0 || vocab <= 0 || !(params.temperature > 0.0F) ||
      params.keep < 0 || params.keep > kLogitHeadCandidates ||
      params.minimum < 1)
    return hipErrorInvalidValue;
  const int blocks = (vocab + kSpan - 1) / kSpan;
  const Scratch s = Layout(scratch, rows, blocks, nullptr);
  const float inv_temperature = 1.0F / params.temperature;
  const dim3 grid(static_cast<unsigned>(blocks), static_cast<unsigned>(rows));
  HeadMaxKernel<<<grid, kThreads, 0, stream>>>(logits, vocab, s.part_max);
  HeadSumKernel<<<grid, kThreads, 0, stream>>>(logits, vocab,
                                               params.temperature,
                                               inv_temperature, s);
  HeadCutKernel<<<rows, kThreads, 0, stream>>>(params, blocks, s, records);
  HeadGatherKernel<<<grid, kThreads, 0, stream>>>(logits, vocab,
                                                  inv_temperature, s, records,
                                                  candidates);
  return hipGetLastError();
}

}  // namespace gufo::models::nemotron35_bolt::rocm
