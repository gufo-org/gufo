// Bucket-compaction builders for the wide-chunk MoE tier. The kernels are
// the Gemma-4 route as staged through flint `moe/routed_compact` (see
// staging/models/flint/kernels/rocm/moe/VENDOR.md), adapted to 288-expert
// group-limited routing: no token-side map (the GEMV gathers activation
// rows from the slot), and the tile builder scans up to 1024 experts.
#include "staging/models/glm5_next/kernels/rocm/routed_compact.hpp"

#include <cstdint>

namespace gufo::models::glm5_next::rocm {
namespace {

constexpr int kThreads = 256;
constexpr int kTileThreads = 1024;
constexpr std::size_t kRoutedTileTokens = 16;

__global__ void ExpertCountsKernel(const std::int32_t* __restrict__ ids,
                                   std::uint32_t n_ids,
                                   std::uint32_t* __restrict__ counts) {
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ids;
       i += gridDim.x * blockDim.x) {
    const std::int32_t e = ids[i];
    if (e >= 0) {
      atomicAdd(&counts[e], 1U);
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::int32_t* __restrict__ rows_slot,
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
  __syncthreads();
  // Scatter overwrites every occupied row. Only the padding can retain a
  // previous batch's assignments; clear it while the bucket bounds are hot.
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    for (std::uint32_t row = counts[e];
         row < static_cast<std::uint32_t>(padded[e]); ++row) {
      rows_slot[pad_bounds[e] + row] = -1;
    }
  }
}

/// rows_slot[c] for every routed (token, slot); the order inside a bucket
/// is whatever the atomics produce, which changes nothing: every output row
/// is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  static_cast<void>(k);
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

// One block: thread e counts expert e's tiles of both heights, an ordered
// scan places them, and the remaining capacity entries become dead tiles.
__global__ void __launch_bounds__(1024) BuildRoutedTilesKernel(
    const std::uint32_t* __restrict__ counts, std::uint32_t experts,
    std::uint32_t rows_a, std::uint32_t capacity_a,
    std::int32_t* __restrict__ tiles_a, std::uint32_t rows_b,
    std::uint32_t capacity_b, std::int32_t* __restrict__ tiles_b) {
  __shared__ std::uint32_t scan_a[1024];
  __shared__ std::uint32_t scan_b[1024];
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t padded = e < experts ? (counts[e] + 15U) / 16U * 16U : 0;
  const std::uint32_t n_a = (padded + rows_a - 1) / rows_a;
  const std::uint32_t n_b = (padded + rows_b - 1) / rows_b;
  scan_a[e] = n_a;
  scan_b[e] = n_b;
  __syncthreads();
  // Inclusive Hillis-Steele scan; integer sums, so order-free.
  for (std::uint32_t step = 1; step < kTileThreads; step <<= 1) {
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
  constexpr std::int32_t kDead = 0x7FFF << 16;
  for (std::uint32_t i = scan_a[1023] + e; i < capacity_a; i += kTileThreads) {
    tiles_a[i] = kDead;
  }
  for (std::uint32_t i = scan_b[1023] + e; i < capacity_b; i += kTileThreads) {
    tiles_b[i] = kDead;
  }
}

}  // namespace

void ExpertCounts(const std::int32_t* ids, std::uint32_t n_ids,
                  std::uint32_t n_experts, std::uint32_t* counts,
                  hipStream_t stream) {
  static_cast<void>(
      hipMemsetAsync(counts, 0, static_cast<std::size_t>(n_experts) * 4,
                     stream));
  const std::uint32_t blocks = (n_ids + kThreads - 1) / kThreads;
  ExpertCountsKernel<<<blocks, kThreads, 0, stream>>>(ids, n_ids, counts);
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_slot, std::uint32_t n_tokens,
                   std::uint32_t k, std::uint32_t n_experts,
                   hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  RoutedPadBoundsKernel<<<1, kTileThreads, 0, stream>>>(
      counts, pad_bounds, cursors, rows_slot, n_experts);
  RoutedScatterKernel<<<static_cast<unsigned>((slots + kThreads - 1) /
                                              kThreads),
                        kThreads, 0,
                        stream>>>(ids, pad_bounds, cursors, rows_slot,
                                  static_cast<std::uint32_t>(slots), k);
}

void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream) {
  BuildRoutedTilesKernel<<<1, kTileThreads, 0, stream>>>(
      counts, experts, rows_a, capacity_a, tiles_a, rows_b, capacity_b,
      tiles_b);
}

}  // namespace gufo::models::glm5_next::rocm
