#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionQueryHeads{24};
inline constexpr std::uint32_t kTiledAttentionKvHeads{4};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};
inline constexpr std::uint32_t kCkAttentionKvHeads{4};
inline constexpr std::uint32_t kCkAttentionHeadDim{256};
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{4096};
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
  bool has_scratch_f16{false};
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

// opt-c010-residual-rmsnorm: fuse the post-attention residual add with the
// subsequent FFN RMSNorm into a single kernel per token (decode) / per token
// row (prefill). Flip to false to revert to the unfused chain
// (ResidualAdd + RMSNorm), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseResidualAddRMSNorm(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_residual_rmsnorm;
}

[[nodiscard]] constexpr bool ShouldFuseResidualAddRMSNorm() noexcept {
  return ShouldFuseResidualAddRMSNorm(QwenExecutionPolicy::Production());
}

// opt-c010-ffn-swiglu: fuse the FFN gate/up projections with the SwiGLU
// activation into a single batched kernel for prefill. Rejected on gfx1151:
// the naive per-row fused kernel regresses prefill ~37x (pp1024 9.70 vs 362.87
// tok/s) against the hipBLASLt BF16 gate/up GEMMs plus SwiGLU activation, so
// the unfused chain is the production route. Kept at false for re-evaluation.
[[nodiscard]] constexpr bool ShouldFuseFFNSwiGLU(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_prefill_ffn_swiglu;
}

[[nodiscard]] constexpr bool ShouldFuseFFNSwiGLU() noexcept {
  return ShouldFuseFFNSwiGLU(QwenExecutionPolicy::Production());
}

// opt-c010-ssm-gate-residual: fuse the SSM per-head post-RMSNorm + SiLU gate
// into the DeltaNet recurrence epilogue (prefill, replacing
// BatchedSSMPostNormGateKernel) and fold the post-SSM residual add into the
// ssm_out GEMV (decode). Flip to false to revert to the unfused chain
// (recurrence + post-norm kernel; ssm_out GEMV + residual add), which stays
// wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseDecodeSSMOutputResidual(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_decode_ssm_output_residual;
}

[[nodiscard]] constexpr bool ShouldFusePrefillSSMPostNormGate(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_prefill_ssm_post_norm_gate;
}

[[nodiscard]] constexpr bool ShouldFuseSSMGateResidual() noexcept {
  const auto policy = QwenExecutionPolicy::Production();
  return ShouldFuseDecodeSSMOutputResidual(policy) &&
         ShouldFusePrefillSSMPostNormGate(policy);
}

// opt-c010-rmsnorm-projection: fuse the layer pre-RMSNorm into the fused
// decode projection GEMVs (QKV, SSM input, and FFN SwiGLU), so the projection
// kernel reads the raw hidden row and prepares its own normed input. Flip to
// false to revert to the unfused chain (RMSNormKernel + projection kernel),
// which stays wired as the independent reference. Prefill is unaffected: its
// batched norm kernel already fuses the FP32 + BF16 input preparation.
[[nodiscard]] constexpr bool ShouldFuseRMSNormProjection(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_decode_rmsnorm_projection;
}

[[nodiscard]] constexpr bool ShouldFuseRMSNormProjection() noexcept {
  return ShouldFuseRMSNormProjection(QwenExecutionPolicy::Production());
}

// opt-c014-layer-prefetch: issue an asynchronous GPU touch of the next layer's
// weight pages on a side stream while the current layer executes, so the next
// layer's projection kernels do not stall on first-touch page walks. Flip to
// false to revert to the no-prefetch route, which stays wired as the
// independent reference.
[[nodiscard]] constexpr bool ShouldPrefetchNextLayer(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.prefetch_next_layer;
}

[[nodiscard]] constexpr bool ShouldPrefetchNextLayer() noexcept {
  return ShouldPrefetchNextLayer(QwenExecutionPolicy::Production());
}

// opt-q4kxl-rows3: three rows amortizes the exact small-batch K-quant
// verifier's shared activation tile over 1.5x as many output rows and improves
// Q4 DFlash-2 end to end. Q6_K stays on two rows because its extra per-half
// scale state spills with three; four rows remains a rejected, measurable
// alternative. `2` pins the independent production reference.
enum class KQuantSmallBatchRows : std::uint8_t {
  kTwo = 2,
  kThree = 3,
  kFour = 4,
};

[[nodiscard]] inline KQuantSmallBatchRows ResolveKQuantSmallBatchRows(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantSmallBatchRows::kThree;
  }
  const std::string_view text{value};
  if (text == "3" || text == "three" || text == "rows3") {
    return KQuantSmallBatchRows::kThree;
  }
  return text == "4" || text == "four" || text == "rows4"
             ? KQuantSmallBatchRows::kFour
             : KQuantSmallBatchRows::kTwo;
}

[[nodiscard]] inline KQuantSmallBatchRows KQuantSmallBatchRowsFromEnv() {
  static const KQuantSmallBatchRows rows =
      ResolveKQuantSmallBatchRows(std::getenv("GUFO_KQUANT_SMALL_BATCH_ROWS"));
  return rows;
}

// opt-q4kxl-occ12: how many wave32s per SIMD the exact small-batch K-quant
// verifier asks the register allocator for.
//
// The kernel launches sixteen waves per workgroup, so residency moves in whole
// sixteen-wave steps: ten waves per SIMD is forty per CU and holds two
// workgroups, twelve is forty-eight and holds three. Twelve needs VGPR <= 128,
// and the LDS stage is 2688 bytes per draft token, so three workgroups at the
// widest draft (batch 8, 21,504 bytes) is 64,512 -- just inside the 64 KB.
//
// The evidence that this matters came from a draft-width sweep. At draft width
// five (verify batch six) the allocator happens to land on 120 VGPR for Q5_K
// and the kernel runs 264 us per dispatch, against 324 at batch seven and 339
// at batch eight where it lands on 136-144. That is not a width effect: batch
// three costs 279 us, more than batch six. It is the third workgroup.
//
// `10` pins the previously retained hint and stays selectable.
enum class KQuantSmallBatchOccupancy : std::uint8_t {
  kTenWaves = 10,
  kTwelveWaves = 12,
};

[[nodiscard]] inline KQuantSmallBatchOccupancy ResolveKQuantSmallBatchOccupancy(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantSmallBatchOccupancy::kTwelveWaves;
  }
  const std::string_view text{value};
  return text == "10" || text == "ten"
             ? KQuantSmallBatchOccupancy::kTenWaves
             : KQuantSmallBatchOccupancy::kTwelveWaves;
}

[[nodiscard]] inline KQuantSmallBatchOccupancy
KQuantSmallBatchOccupancyFromEnv() noexcept {
  static const KQuantSmallBatchOccupancy waves =
      ResolveKQuantSmallBatchOccupancy(
          std::getenv("GUFO_KQUANT_SMALL_BATCH_WAVES"));
  return waves;
}

// opt-q4kxl-hoist: where the exact small-batch K-quant verifier issues its
// weight loads relative to the activation stage.
//
// The kernel stages a tile of FP32 activations into LDS, syncs, then decodes
// the weight sub-blocks each lane owns and multiplies. The weight fetch does
// not read anything the stage produces, so waiting behind the stage's barrier
// exposes its latency at the top of every tile iteration. `hoist` issues it
// first, which is the only prefetch this kernel can afford for free: the
// decoded sub-blocks are already live across the whole compute phase, so
// extending them over one barrier adds no registers and cannot cost the third
// resident workgroup that `GUFO_KQUANT_SMALL_BATCH_WAVES=12` buys.
//
// The arithmetic is untouched -- same terms, same order -- so the bit-exact
// contract against the decode GEMV holds by construction.
enum class KQuantSmallBatchFetch : std::uint8_t {
  kInOrder,
  kHoisted,
};

[[nodiscard]] inline KQuantSmallBatchFetch ResolveKQuantSmallBatchFetch(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantSmallBatchFetch::kHoisted;
  }
  const std::string_view text{value};
  return text == "0" || text == "in-order" || text == "inorder"
             ? KQuantSmallBatchFetch::kInOrder
             : KQuantSmallBatchFetch::kHoisted;
}

[[nodiscard]] inline KQuantSmallBatchFetch
KQuantSmallBatchFetchFromEnv() noexcept {
  static const KQuantSmallBatchFetch fetch = ResolveKQuantSmallBatchFetch(
      std::getenv("GUFO_KQUANT_SMALL_BATCH_FETCH"));
  return fetch;
}

// opt-dflash2-verify-marginal: measurement-only route that deletes the
// Q4_K/Q5_K minimum correction from the small-batch verifier. `w = scale*q -
// offset` is not `scale*q`, so the result is WRONG and the correctness gate
// rejects it; it exists to bound what an exact cheaper formulation of the
// correction could be worth in the speculative step, the same way
// `GUFO_KQUANT_PREFILL_TILE=drop-offset-probe` bounds it for prefill. The
// verifier pays the term once per verified row, so its cost scales with draft
// width where prefill's scales with prompt length.
enum class KQuantSmallBatchOffset : std::uint8_t {
  kExact,
  kDropProbe,
};

[[nodiscard]] inline KQuantSmallBatchOffset ResolveKQuantSmallBatchOffset(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantSmallBatchOffset::kExact;
  }
  const std::string_view text{value};
  return text == "drop-probe" || text == "drop-offset-probe"
             ? KQuantSmallBatchOffset::kDropProbe
             : KQuantSmallBatchOffset::kExact;
}

[[nodiscard]] inline KQuantSmallBatchOffset
KQuantSmallBatchOffsetFromEnv() noexcept {
  static const KQuantSmallBatchOffset offset = ResolveKQuantSmallBatchOffset(
      std::getenv("GUFO_KQUANT_SMALL_BATCH_OFFSET"));
  return offset;
}

// opt-q4kxl-actsum: Q4_K/Q5_K blocked WMMA needs the sum of each quantized
// 32-element activation block for its minimum correction. Computing it in the
// kernel costs eight `sudot4` per token tile per K block, and every row tile in
// the block recomputes the same value. Storing it once beside the unchanged
// tiled Q8_1 layout is retained as the default.
//
// Measured on the UD-Q4_K_XL shard, four interleaved single-repetition pairs:
//
//   pp2048 reference 471.09/467.44/465.74/461.76 (median 466.59)
//   pp2048 staged    472.29/469.54/468.55/465.62 (median 469.05, +0.53%)
//   pp2048 direct    459.04/456.61/452.41/453.34 (median 454.98, -2.49%)
//
// Staged wins every pair, direct loses every pair. `direct` re-reads the
// sidecar from global memory inside the innermost token-tile loop, so each row
// tile pays the load again instead of sharing one LDS read -- it is kept
// selectable because it is the natural alternative and the measurement is the
// only thing that separates them. The producers write the sidecar on the decode
// path too, where nothing reads it; that was checked and costs nothing
// (tg128 11.60/11.61/11.61 against 11.60/11.61/11.62, and tg128-dflash2 is
// 3/3 pairs slightly ahead at 18.76/18.76/18.75 against 18.72/18.74/18.72).
//
// `0` pins the independent in-kernel `sudot4` reference route.
enum class KQuantActivationSumMode : std::uint8_t {
  kReference,
  kStaged,
  kDirect,
};

[[nodiscard]] inline KQuantActivationSumMode ResolveKQuantActivationSums(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantActivationSumMode::kStaged;
  }
  const std::string_view text{value};
  if (text == "direct") {
    return KQuantActivationSumMode::kDirect;
  }
  return text == "1" || text == "true" || text == "on" || text == "sidecar" ||
                 text == "staged"
             ? KQuantActivationSumMode::kStaged
             : KQuantActivationSumMode::kReference;
}

[[nodiscard]] inline KQuantActivationSumMode
KQuantActivationSumModeFromEnv() noexcept {
  static const KQuantActivationSumMode mode =
      ResolveKQuantActivationSums(std::getenv("GUFO_KQUANT_ACTIVATION_SUMS"));
  return mode;
}

// opt-q4kxl-wide: the blocked K-quant WMMA prefill tile. Every weight element
// is decoded once per token block, so the decode cost the K-quants pay over
// Q8_0 scales with ceil(batch / BN). BN is pinned by the accumulator array,
// which is BM*BN/threads registers: at 256 threads BM*BN cannot exceed
// 128*128, which fixes the re-decode count at 16 for a 2048-token prefill.
// A 512-thread block keeps the same 64-register accumulator at BM=128/BN=256
// and halves the re-decode count to 8.
//
// `default` pins the retained 128x128 / 256-thread tile; `wide` selects the
// 128x256 / 512-thread tile for the large-batch route only. Both stay
// runnable so the comparison can be re-measured.
enum class KQuantPrefillTile : std::uint8_t {
  kDefault,
  kWide,
  kWideForcedOccupancy,
  kNarrowRows,
  // opt-q4kxl-bk1: one K block per LDS stage instead of two. The stage buffers
  // drop from 19,456 to ~10,240 bytes, which takes LDS out of the residency
  // equation entirely; whether a fourth workgroup actually lands then depends
  // only on whether the allocator stays at or below 192 VGPR.
  kSingleKBlock,
  // opt-q4kxl-fuse: fold the Q4_K/Q5_K minimum correction into the main term
  // before accumulating, halving the dependency chain on `acc`.
  kFusedOffset,
  // Measurement only, and numerically WRONG: deletes the Q4_K/Q5_K minimum
  // correction to bound what an exact cheaper formulation could ever be worth.
  kDropOffsetProbe,
};

[[nodiscard]] inline KQuantPrefillTile ResolveKQuantPrefillTile(
    const char* value) noexcept {
  if (value == nullptr) {
    return KQuantPrefillTile::kDefault;
  }
  const std::string_view text{value};
  if (text == "wide-occ" || text == "wide8") {
    return KQuantPrefillTile::kWideForcedOccupancy;
  }
  if (text == "narrow" || text == "64x256") {
    return KQuantPrefillTile::kNarrowRows;
  }
  if (text == "drop-offset-probe") {
    return KQuantPrefillTile::kDropOffsetProbe;
  }
  if (text == "bk1") {
    return KQuantPrefillTile::kSingleKBlock;
  }
  if (text == "fuse") {
    return KQuantPrefillTile::kFusedOffset;
  }
  return text == "wide" || text == "256" || text == "1"
             ? KQuantPrefillTile::kWide
             : KQuantPrefillTile::kDefault;
}

[[nodiscard]] inline KQuantPrefillTile KQuantPrefillTileFromEnv() noexcept {
  static const KQuantPrefillTile tile =
      ResolveKQuantPrefillTile(std::getenv("GUFO_KQUANT_PREFILL_TILE"));
  return tile;
}

[[nodiscard]] inline bool ShouldStoreKQuantActivationSums() noexcept {
  return KQuantActivationSumModeFromEnv() !=
         KQuantActivationSumMode::kReference;
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
         params.num_heads == kTiledAttentionQueryHeads &&
         params.num_kv_heads == kTiledAttentionKvHeads &&
         params.head_dim == kTiledAttentionHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         params.has_k_cache_f16 && params.has_v_cache_f16;
}

[[nodiscard]] constexpr bool IsCkAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 && params.start_pos == 0 &&
         params.num_kv_heads == kCkAttentionKvHeads &&
         params.num_heads % kCkAttentionKvHeads == 0 &&
         params.head_dim == kCkAttentionHeadDim &&
         params.batch_size <= params.max_context && params.has_k_cache_f16 &&
         params.has_v_cache_f16 && params.has_scratch_f16;
}

/// Prefill DeltaNet recurrence backend preference. The production route is the
/// row-split kernel (opt-c170-deltanet-rowsplit);
/// `GUFO_SSM_RECURRENCE=baseline` pins the previous single-block kernel so it
/// stays measurable in the same binary rather than only being reachable when a
/// shape is unsupported.
enum class SsmRecurrencePreference : std::uint8_t {
  kAuto,
  kRowSplit,
  kBaseline,
};

[[nodiscard]] inline SsmRecurrencePreference ResolveSsmRecurrencePreference(
    const char* value) noexcept {
  if (value == nullptr) {
    return SsmRecurrencePreference::kAuto;
  }
  const std::string_view text{value};
  if (text == "rowsplit" || text == "row_split") {
    return SsmRecurrencePreference::kRowSplit;
  }
  if (text == "baseline") {
    return SsmRecurrencePreference::kBaseline;
  }
  return SsmRecurrencePreference::kAuto;
}

[[nodiscard]] inline SsmRecurrencePreference SsmRecurrencePreferenceFromEnv() {
  static const SsmRecurrencePreference preference =
      ResolveSsmRecurrencePreference(std::getenv("GUFO_SSM_RECURRENCE"));
  return preference;
}

/// The row-split recurrence needs its two prologue scratch planes and the
/// 128 x 128 per-head state tile the kernel's register geometry is built for.
[[nodiscard]] inline bool ShouldUseSsmRowSplitRecurrence(bool shape_supported,
                                                         bool scratch_ready) {
  const SsmRecurrencePreference preference = SsmRecurrencePreferenceFromEnv();
  if (preference == SsmRecurrencePreference::kBaseline) {
    return false;
  }
  return shape_supported && scratch_ready;
}

/// opt-c177-attn-wmma: the masked WMMA kernel covers the whole visible range in
/// one pass and beats the tiled kernel by 3.5x at depth 0 and the tiled
/// diagonal plus AOTriton prefix by 1.3-1.4x at depth, so it is the production
/// route and the split path is no longer used. `GUFO_PREFILL_ATTENTION=split`
/// (or `tile`, `ck`, `baseline`) pins an alternative for comparison.
[[nodiscard]] inline bool ShouldUseWmmaPrefillAttention();

/// Prefill attention backend preference. The fallback order behind the WMMA
/// kernel is tiled, then Composable Kernel, then baseline;
/// `GUFO_PREFILL_ATTENTION` pins one backend so the alternatives stay
/// measurable in the same binary instead of only being reachable when the
/// preferred one rejects a shape.
enum class PrefillAttentionPreference : std::uint8_t {
  kAuto,
  kWmma,
  kTiled,
  kComposableKernel,
  kBaseline,
  kSplit,
};

[[nodiscard]] inline PrefillAttentionPreference
ResolvePrefillAttentionPreference(const char* value) noexcept {
  if (value == nullptr) {
    return PrefillAttentionPreference::kAuto;
  }
  const std::string_view text{value};
  if (text == "wmma") {
    return PrefillAttentionPreference::kWmma;
  }
  if (text == "tile" || text == "tiled") {
    return PrefillAttentionPreference::kTiled;
  }
  if (text == "ck" || text == "composable_kernel") {
    return PrefillAttentionPreference::kComposableKernel;
  }
  if (text == "baseline") {
    return PrefillAttentionPreference::kBaseline;
  }
  if (text == "split") {
    return PrefillAttentionPreference::kSplit;
  }
  return PrefillAttentionPreference::kAuto;
}

[[nodiscard]] inline PrefillAttentionPreference
PrefillAttentionPreferenceFromEnv() {
  static const PrefillAttentionPreference preference =
      ResolvePrefillAttentionPreference(std::getenv("GUFO_PREFILL_ATTENTION"));
  return preference;
}

[[nodiscard]] inline bool ShouldUseWmmaPrefillAttention() {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  return preference == PrefillAttentionPreference::kAuto ||
         preference == PrefillAttentionPreference::kWmma;
}

inline constexpr std::size_t kPrefillAttentionSplitMinPrefix{1024};

/// The split prefill attention (AOTriton non-causal prefix plus the tiled
/// causal diagonal) only pays off once the prefix is a meaningful share of the
/// work: it costs one extra kernel plus an FP16 query conversion and a merge
/// pass. Below the threshold, or with no prefix at all, the unsplit chain wins.
[[nodiscard]] constexpr bool IsPrefillAttentionSplitProfitable(
    std::uint32_t start_pos, std::size_t batch_size) noexcept {
  return start_pos >= kPrefillAttentionSplitMinPrefix && batch_size >= 64;
}

[[nodiscard]] inline bool ShouldUsePrefillAttentionSplit(
    std::uint32_t start_pos, std::size_t batch_size) {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  if (preference == PrefillAttentionPreference::kSplit) {
    return start_pos > 0 && batch_size > 0;
  }
  // Not auto-selected any more: the WMMA kernel covers the same work in one
  // pass and is faster at every depth.
  return false;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then Composable Kernel, then the baseline implementation.
template<typename TiledLauncher, typename CkLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     CkLauncher&& launch_ck,
                                     BaselineLauncher&& launch_baseline) {
  const PrefillAttentionPreference preference =
      PrefillAttentionPreferenceFromEnv();
  if (preference != PrefillAttentionPreference::kBaseline &&
      ShouldAttemptOptimizedAttention(visible_context)) {
    if (preference != PrefillAttentionPreference::kComposableKernel &&
        std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
    if (preference != PrefillAttentionPreference::kTiled &&
        std::forward<CkLauncher>(launch_ck)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
