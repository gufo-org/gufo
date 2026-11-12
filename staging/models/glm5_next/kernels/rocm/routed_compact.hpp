#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_COMPACT_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_COMPACT_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Bucket-compaction builders for the wide-chunk MoE tiers. Forked from the
/// Gemma-4 route (as staged through flint `moe/routed_compact`): 16-row
/// padded expert buckets and the `expert | tile << 16` tile map, adapted to
/// this model's 288-expert group-limited routing (no token-side map, no
/// F16 narrowing; the tile builder covers up to 1024 experts). All launches
/// are asynchronous on `stream` (null uses the default stream).
namespace gufo::models::glm5_next::rocm {

/// counts[e] = number of slots in ids[0..n_ids) selecting expert e. ids
/// entries below 0 (unrouted slots) are ignored.
void ExpertCounts(const std::int32_t* ids, std::uint32_t n_ids,
                  std::uint32_t n_experts, std::uint32_t* counts,
                  hipStream_t stream);

/// Rows the compacted bucket layout needs for `slots` assignments over
/// `n_experts` experts with 16-row padded buckets.
[[nodiscard]] std::size_t RoutedCompactRows(std::size_t slots,
                                            std::size_t n_experts);

/// Compacts the routed assignments by expert into 16-row padded buckets.
/// `ids` [n_tokens * k] holds the chosen expert per (token, slot) or -1;
/// `counts` [n_experts] is the per-expert histogram produced by
/// ExpertCounts. `pad_bounds` [n_experts + 1] receives the exclusive scan of
/// the padded bucket starts, `cursors` [n_experts] is scratch, and
/// `rows_slot` (RoutedCompactRows long) maps every compacted row back to
/// its slot (slot = token * k + j), with -1 in the bucket padding.
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_slot, std::uint32_t n_tokens,
                   std::uint32_t k, std::uint32_t n_experts,
                   hipStream_t stream);

/// Entries of a routed tile map of `rows`-row tiles that any routing of
/// `slots` assignments over `experts` 16-padded buckets fits in.
[[nodiscard]] constexpr std::uint32_t RoutedTileCapacity(
    std::uint32_t slots, std::uint32_t experts, std::uint32_t rows) noexcept {
  return (slots + 15 * experts) / rows + experts;
}

/// Builds the routed tile maps of two tile heights on the device from the
/// per-expert counts: `expert | tile << 16` in expert order, capacity
/// entries past the last tile hold a tile index beyond every bucket
/// (0x7FFF), which the routed GEMV skips. `experts` must be at most 1024.
void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream);

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_ROUTED_COMPACT_HPP_
