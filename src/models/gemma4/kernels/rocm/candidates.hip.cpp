#include "src/models/gemma4/kernels/rocm/candidates.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"

namespace gufo::models::gemma4::rocm {
namespace {

constexpr unsigned kThreads = 256;
/// Threshold histogram over the top 12 bits of the ordered logit key.
constexpr unsigned kBins = 4096;
constexpr unsigned kBinShift = 20;
/// Logits in the threshold bin one row can rank exactly.
constexpr unsigned kBoundary = 2048;
constexpr unsigned kMaxTop = kCandidateTop + kMaxCandidateExtra;
/// Blocks per row of the histogram and filter passes.
constexpr unsigned kRowBlocks = 32;

/// Unsigned key ordered like the logit; 0 for a non-finite one (sampling
/// skips those) and +0 for -0 (they compare equal). Bitwise: fast math may
/// assume no signed zeros or non-finite values.
__device__ inline std::uint32_t Key(float value) {
  std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    return 0;
  }
  if ((bits & 0x7FFFFFFFU) == 0) {
    bits = 0;
  }
  return (bits & 0x80000000U) != 0 ? ~bits : (bits | 0x80000000U);
}

__device__ inline void RowRange(std::uint32_t vocab, unsigned* begin,
                                unsigned* end) {
  const unsigned quads = vocab / 4;
  const unsigned per_block = (quads + kRowBlocks - 1) / kRowBlocks;
  *begin = min(blockIdx.x * per_block, quads);
  *end = min(*begin + per_block, quads);
}

__global__ __launch_bounds__(kThreads) void HistogramKernel(
    const float* logits, std::uint32_t vocab, CandidateWork work) {
  __shared__ std::uint32_t bins[kBins];
  for (unsigned i = threadIdx.x; i < kBins; i += kThreads) {
    bins[i] = 0;
  }
  __syncthreads();
  const unsigned row = blockIdx.y;
  const auto* x =
      reinterpret_cast<const float4*>(logits + std::size_t{row} * vocab);
  unsigned begin;
  unsigned end;
  RowRange(vocab, &begin, &end);
  for (unsigned i = begin + threadIdx.x; i < end; i += kThreads) {
    const float4 v = x[i];
    const std::uint32_t keys[4] = {Key(v.x), Key(v.y), Key(v.z), Key(v.w)};
#pragma unroll
    for (const std::uint32_t key : keys) {
      if (key != 0) {
        atomicAdd(&bins[key >> kBinShift], 1U);
      }
    }
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < kBins; i += kThreads) {
    if (bins[i] != 0) {
      atomicAdd(&work.histogram[row * kBins + i], bins[i]);
    }
  }
}

/// The highest bin b holding at least `top` logits at or above it (0 when
/// the row has fewer). Logits above b are among the row's top ones, and b
/// holds the rest.
__device__ unsigned ThresholdBin(const std::uint32_t* histogram, unsigned top) {
  constexpr unsigned kPer = kBins / kThreads;
  __shared__ std::uint32_t sums[kThreads];
  __shared__ unsigned result;
  // Thread t sums bins [hi - kPer, hi), counting from the highest.
  const unsigned hi = kBins - kPer * threadIdx.x;
  std::uint32_t sum = 0;
  for (unsigned b = hi - kPer; b < hi; ++b) {
    sum += histogram[b];
  }
  sums[threadIdx.x] = sum;
  if (threadIdx.x == 0) {
    result = 0;
  }
  __syncthreads();
  for (unsigned offset = 1; offset < kThreads; offset <<= 1) {
    const std::uint32_t add =
        threadIdx.x >= offset ? sums[threadIdx.x - offset] : 0;
    __syncthreads();
    sums[threadIdx.x] += add;
    __syncthreads();
  }
  const std::uint32_t above = threadIdx.x != 0 ? sums[threadIdx.x - 1] : 0;
  if (above < top && sums[threadIdx.x] >= top) {
    std::uint32_t count = above;
    for (unsigned b = hi; b-- > hi - kPer;) {
      count += histogram[b];
      if (count >= top) {
        result = b;
        break;
      }
    }
  }
  __syncthreads();
  return result;
}

__global__ __launch_bounds__(kThreads) void FilterKernel(const float* logits,
                                                         std::uint32_t vocab,
                                                         std::uint32_t top,
                                                         CandidateWork work) {
  const unsigned row = blockIdx.y;
  const unsigned threshold = ThresholdBin(work.histogram + row * kBins, top);
  const auto* x =
      reinterpret_cast<const float4*>(logits + std::size_t{row} * vocab);
  unsigned begin;
  unsigned end;
  RowRange(vocab, &begin, &end);
  for (unsigned i = begin + threadIdx.x; i < end; i += kThreads) {
    const float4 v = x[i];
    const std::uint32_t keys[4] = {Key(v.x), Key(v.y), Key(v.z), Key(v.w)};
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned bin = keys[j] >> kBinShift;
      if (keys[j] == 0 || bin < threshold) {
        continue;
      }
      const std::uint32_t id = i * 4 + j;
      if (bin > threshold) {
        const auto at = atomicAdd(&work.counts[row * 2], 1U);
        work.sure[row * kMaxTop + at] = id;
      } else {
        const auto at = atomicAdd(&work.counts[row * 2 + 1], 1U);
        if (at < kBoundary) {
          // Descending key, then ascending id: sampling's tie order.
          work.boundary[row * kBoundary + at] =
              (std::uint64_t{keys[j]} << 32) | ~id;
        }
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void FinishKernel(
    const float* logits, std::uint32_t vocab, std::uint32_t top,
    const std::uint32_t* extra, std::uint32_t n_extra, CandidateWork work,
    std::uint32_t* counts, sampling::LogitCandidate* out) {
  __shared__ std::uint64_t keys[kBoundary];
  __shared__ unsigned taken;
  const unsigned row = blockIdx.x;
  const float* x = logits + std::size_t{row} * vocab;
  sampling::LogitCandidate* o = out + std::size_t{row} * (top + n_extra);
  const unsigned sure = work.counts[row * 2];
  const unsigned boundary = work.counts[row * 2 + 1];
  if (threadIdx.x == 0) {
    taken = 0;
  }
  __syncthreads();
  if (boundary > kBoundary) {
    if (threadIdx.x == 0) {
      counts[row] = kCandidateOverflow;
    }
  } else {
    for (unsigned i = threadIdx.x; i < sure; i += kThreads) {
      const std::uint32_t id = work.sure[row * kMaxTop + i];
      o[i] = {id, x[id]};
    }
    const unsigned need = top - sure;
    for (unsigned i = threadIdx.x; i < boundary; i += kThreads) {
      keys[i] = work.boundary[row * kBoundary + i];
    }
    __syncthreads();
    // The bin's best `need` logits by rank (keys are distinct).
    for (unsigned i = threadIdx.x; i < boundary; i += kThreads) {
      const std::uint64_t key = keys[i];
      unsigned rank = i;
      if (boundary > need) {
        rank = 0;
        for (unsigned j = 0; j < boundary; ++j) {
          rank += keys[j] > key ? 1U : 0U;
        }
      }
      if (rank < need) {
        const auto id = static_cast<std::uint32_t>(~key);
        o[sure + rank] = {id, x[id]};
        atomicAdd(&taken, 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      counts[row] = sure + taken;
    }
  }
  for (unsigned i = threadIdx.x; i < n_extra; i += kThreads) {
    o[top + i] = {extra[i], x[extra[i]]};
  }
  // Leave the workspace zeroed for the next launch.
  for (unsigned i = threadIdx.x; i < kBins; i += kThreads) {
    work.histogram[row * kBins + i] = 0;
  }
  if (threadIdx.x < 2) {
    work.counts[row * 2 + threadIdx.x] = 0;
  }
}

std::size_t Align(std::size_t bytes) {
  return (bytes + 255) & ~std::size_t{255};
}

}  // namespace

std::size_t CandidateWorkBytes(std::uint32_t rows) noexcept {
  return Align(std::size_t{rows} * kBins * sizeof(std::uint32_t)) +
         Align(std::size_t{rows} * 2 * sizeof(std::uint32_t)) +
         Align(std::size_t{rows} * kMaxTop * sizeof(std::uint32_t)) +
         Align(std::size_t{rows} * kBoundary * sizeof(std::uint64_t));
}

CandidateWork CandidateWorkAt(void* bytes, std::uint32_t rows,
                              hipStream_t stream) {
  auto* base = static_cast<std::uint8_t*>(bytes);
  CandidateWork work{};
  work.histogram = reinterpret_cast<std::uint32_t*>(base);
  base += Align(std::size_t{rows} * kBins * sizeof(std::uint32_t));
  work.counts = reinterpret_cast<std::uint32_t*>(base);
  base += Align(std::size_t{rows} * 2 * sizeof(std::uint32_t));
  work.sure = reinterpret_cast<std::uint32_t*>(base);
  base += Align(std::size_t{rows} * kMaxTop * sizeof(std::uint32_t));
  work.boundary = reinterpret_cast<std::uint64_t*>(base);
  HIP_CHECK(hipMemsetAsync(bytes, 0, CandidateWorkBytes(rows), stream));
  return work;
}

void LogitCandidates(const float* logits, std::uint32_t rows,
                     std::uint32_t vocab, std::uint32_t top,
                     const std::uint32_t* extra, std::uint32_t n_extra,
                     CandidateWork work, std::uint32_t* counts,
                     sampling::LogitCandidate* out, hipStream_t stream) {
  if (rows == 0 || vocab == 0 || vocab % 4 != 0 || top == 0 || top > kMaxTop ||
      n_extra > kMaxCandidateExtra || (n_extra != 0 && extra == nullptr)) {
    throw std::invalid_argument("invalid gemma4 logit candidate shape");
  }
  HistogramKernel<<<dim3(kRowBlocks, rows), kThreads, 0, stream>>>(logits,
                                                                   vocab, work);
  FilterKernel<<<dim3(kRowBlocks, rows), kThreads, 0, stream>>>(logits, vocab,
                                                                top, work);
  FinishKernel<<<rows, kThreads, 0, stream>>>(logits, vocab, top, extra,
                                              n_extra, work, counts, out);
  HIP_CHECK(hipGetLastError());
}

}  // namespace gufo::models::gemma4::rocm
