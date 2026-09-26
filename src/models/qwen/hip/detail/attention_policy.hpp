#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};

/// Head geometries the tiled and WMMA prefill kernels are instantiated for.
/// The per-block work (two query heads of one KV group) does not depend on the
/// head counts; they only set strides and the query-to-KV head mapping, so a
/// new geometry needs an entry here and a matching instantiation in both
/// kernels' launchers. The GQA ratio must be even.
struct TiledAttentionHeadShape {
  std::uint32_t query_heads;
  std::uint32_t kv_heads;
};
inline constexpr TiledAttentionHeadShape kTiledAttentionShape27B{24, 4};
inline constexpr TiledAttentionHeadShape kTiledAttentionShape35BA3B{16, 2};
// The launchers pick an instantiation by query-head count alone.
static_assert(kTiledAttentionShape27B.query_heads !=
              kTiledAttentionShape35BA3B.query_heads);

[[nodiscard]] constexpr bool IsTiledAttentionHeadShape(
    std::uint32_t query_heads, std::uint32_t kv_heads) noexcept {
  for (const auto shape : {kTiledAttentionShape27B, kTiledAttentionShape35BA3B}) {
    if (query_heads == shape.query_heads && kv_heads == shape.kv_heads) {
      return true;
    }
  }
  return false;
}
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{128};
inline constexpr std::uint32_t kSplitKDecodeAttentionMaxSplits{32};
inline constexpr std::uint32_t kFusedQkNormMaxHeadDim{256};

struct AttentionSupportParams {
  std::size_t batch_size{0};
  std::uint32_t start_pos{0};
  std::uint32_t max_context{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  bool has_k_cache_f16{false};
  bool has_v_cache_f16{false};
};

[[nodiscard]] constexpr bool ShouldAttemptOptimizedAttention(
    std::size_t visible_context) noexcept {
  return visible_context >= kOptimizedAttentionMinBatch;
}

// opt-c010-qk-rope-kv: fuse per-head Q/K RMSNorm, RoPE, and the KV-cache write
// into a single kernel per token (decode) / per token row (prefill). Flip to
// false to revert to the unfused chain (PerHeadRMSNorm x2 + RoPE +
// WriteKVCache*), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_qk_norm_rope_kv;
}

[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite() noexcept {
  return ShouldFuseQKNormRoPEKvWrite(QwenExecutionPolicy::Production());
}

[[nodiscard]] constexpr std::uint32_t SelectDecodeAttentionSplitCount(
    std::size_t sequence_length) noexcept {
  if (sequence_length < kSplitKDecodeAttentionMinContext) {
    return 1;
  }
  return kSplitKDecodeAttentionMaxSplits;
}

[[nodiscard]] constexpr bool IsFusedQkNormSupported(
    std::uint32_t head_dim) noexcept {
  return head_dim != 0 && head_dim <= kFusedQkNormMaxHeadDim;
}

[[nodiscard]] constexpr bool IsSplitKDecodeAttentionSupported(
    std::size_t sequence_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  return SelectDecodeAttentionSplitCount(sequence_length) > 1 &&
         num_heads != 0 && num_kv_heads != 0 &&
         (num_heads % num_kv_heads) == 0 && head_dim == 256;
}

[[nodiscard]] constexpr std::size_t DecodeAttentionScratchElements(
    std::uint32_t num_heads, std::uint32_t head_dim) noexcept {
  return static_cast<std::size_t>(num_heads) * kSplitKDecodeAttentionMaxSplits *
         (static_cast<std::size_t>(head_dim) + 2);
}

[[nodiscard]] constexpr bool IsTiledAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 &&
         IsTiledAttentionHeadShape(params.num_heads, params.num_kv_heads) &&
         params.head_dim == kTiledAttentionHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         params.has_k_cache_f16 && params.has_v_cache_f16;
}

/// The row-split recurrence needs two scratch planes and a 128 x 128 state
/// tile.
[[nodiscard]] constexpr bool ShouldUseSsmRowSplitRecurrence(
    bool shape_supported, bool scratch_ready) noexcept {
  return shape_supported && scratch_ready;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then the baseline implementation.
template<typename TiledLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     BaselineLauncher&& launch_baseline) {
  if (ShouldAttemptOptimizedAttention(visible_context)) {
    if (std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
