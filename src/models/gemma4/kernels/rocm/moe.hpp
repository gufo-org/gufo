#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// Routed expert weight formats with decode and verification kernels.
/// 32 bits wide: the value crosses from GCC-built host code into clang-built
/// HIP code, and clang assumes a narrower argument arrives zero-extended where
/// GCC leaves the upper bits undefined.
enum class ExpertFormat : std::uint32_t {
  kQ4_K,
  kQ5_K,
  kQ6_K,
  kQ8_0,
  kQ5_1,
  kF16,
  /// Dense prefill only (the QAT targets): no expert kernels.
  kQ4_0,
};

/// Rows whose assignments one expert group can hold: an expert appears at
/// most once per row, so this bounds the rows of a grouped (decode or
/// verification) forward.
inline constexpr std::uint32_t kMaxGroupSlots = 16;
/// Ints per group in the table: expert, slot count, slots.
inline constexpr std::uint32_t kGroupInts = 2 + kMaxGroupSlots;
/// Table ints for up to `max_groups` groups (a leading group count).
[[nodiscard]] constexpr std::size_t ExpertGroupInts(
    std::uint32_t max_groups) noexcept {
  return 1 + std::size_t{max_groups} * kGroupInts;
}
/// Router widths the routing kernel holds in registers.
inline constexpr std::uint32_t kMaxRouterHidden = 3072;

/// router[e][i] *= scale[i]: the router with ffn_gate_inp.scale folded in,
/// once at load (the product the routing kernel formed per launch).
void ScaleRouter(float* router, const float* scale, std::uint32_t experts,
                 std::uint32_t hidden, hipStream_t stream);

/// Expert routing of `rows` attention residual rows x ([rows][hidden]):
///   logits[r][e] = router[e] . x[r] * rms(x[r]) / sqrt(hidden)
/// with the scale-folded router (ScaleRouter),
/// the `used` largest logits (ties to the lower expert) in descending order
/// in ids[r][j], and weights[r][j] = softmax over the chosen logits times
/// expert_scale[ids[r][j]]. Every row's arithmetic is independent of the
/// batch. Optional outputs: `counts` ([experts], zeroed by the caller)
/// accumulates assignments per expert; `groups` (rows <= kMaxGroupSlots)
/// receives the group table the routed GEMV reads: every selected expert in
/// increasing order with its slots (r * used + j) in increasing order.
/// With `raw_logits` the caller has already written router[e] . x[r] to
/// `logits` (a prefill GEMM); only the per-row scale and the selection run.
struct MoeRouteArgs {
  const float* x;
  const float* router;        ///< F32 [experts][hidden], scale folded in
  const float* expert_scale;  ///< [experts]
  float* logits;              ///< [rows][experts]
  bool raw_logits;
  std::int32_t* ids;  ///< [rows][used]
  float* weights;     ///< [rows][used]
  std::uint32_t* counts;
  std::int32_t* groups;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t experts;
  std::uint32_t used;
  float eps;
  /// Optional: a zero counter that lets grouped routing (with `groups`,
  /// rows <= 16) run as one launch; left at zero afterwards.
  std::uint32_t* sync;
};
void MoeRoute(const MoeRouteArgs& args, hipStream_t stream);

/// Grouped routed projection over FP32 activations: for every slot s of
/// every group in `groups` (at most `max_groups`, the launch bound),
/// y[s][0..m) = W[expert] x[s / x_div], where W is [experts][m][k] in
/// `format`. Each expert's weights are decoded once per pass and applied to
/// up to four of its slots; a slot's FMA order depends only on its row and
/// the shape, so decode and verification rows round identically. With
/// `geglu`, W holds fused [gate | up] rows (m even) and y[s][0..m / 2)
/// receives gelu_tanh(gate) * up instead. `single` (every group holds one
/// slot: one-row decode) takes a one-slot build of the same arithmetic.
/// Returns false without launching for unsupported shapes.
[[nodiscard]] bool LaunchRoutedGemv(ExpertFormat format, const void* w,
                                    const std::int32_t* groups,
                                    std::uint32_t max_groups, const float* x,
                                    std::uint32_t x_div, float* y,
                                    std::uint32_t m, std::uint32_t k,
                                    hipStream_t stream, bool geglu = false,
                                    bool single = false);

/// Routed prefill projection over binary16 activation rows `x` ([rows][k])
/// in Flash-Next's routing layout (RoutedCompact buckets and a tile map of
/// expert | tile << 16 entries built for `tile_rows` bucket rows per tile):
/// row rows_out[i] of `out` (FP32) or `out_half` (binary16, saturated)
/// receives W[expert] x[rows_in[i]]. Covers the formats the Flash-Next
/// routed GEMM lacks or runs slower (Q4_K, Q5_K, Q6_K, and Q5_1 / Q8_0 /
/// binary16 with FP32 outputs; 96-row tiles); returns false for others. With
/// `geglu` (a
/// K-quant, Q8_0 or binary16 fused [gate | up] W, binary16 output) row r of
/// `out_half`
/// holds the m / 2 values GeGluPackedHalf would make of the binary16 pair.
[[nodiscard]] bool LaunchRoutedHalfGemm(
    ExpertFormat format, const void* w, const void* x,
    const std::int32_t* tiles, std::uint32_t n_tiles, std::uint32_t tile_rows,
    const std::int32_t* pad_bounds, const std::int32_t* rows_in,
    const std::int32_t* rows_out, float* out, void* out_half, std::uint32_t m,
    std::uint32_t k, hipStream_t stream, bool geglu = false);

/// Dense prefill projection over binary16 activation rows `x` ([rows][k])
/// for Q4_K, Q5_K, Q6_K (the routed GEMM's arithmetic over an identity
/// routing; k a multiple of 256) or Q4_0 (the routed down kernel's layout
/// with weights (q - 8) d; k a multiple of 64). `out` (FP32) or `out_half`
/// (binary16, saturated; K-quants only) is [rows][m]; `geglu` as in
/// LaunchRoutedHalfGemm. Returns false for other formats or shapes.
[[nodiscard]] bool LaunchDenseHalfGemm(ExpertFormat format, const void* w,
                                       const void* x, float* out,
                                       void* out_half, std::uint32_t rows,
                                       std::uint32_t m, std::uint32_t k,
                                       hipStream_t stream, bool geglu = false);

/// Entries of a routed tile map of `rows`-row tiles that any routing of
/// `slots` assignments over `experts` 16-padded buckets fits in.
[[nodiscard]] constexpr std::uint32_t RoutedTileCapacity(
    std::uint32_t slots, std::uint32_t experts, std::uint32_t rows) noexcept {
  return (slots + 15 * experts) / rows + experts;
}

/// Builds on the device the routed tile maps of two tile heights from the
/// per-expert assignment counts: expert | tile << 16 in expert order, the
/// entries past the last tile holding a tile index beyond every bucket, which
/// the routed GEMMs skip. The GEMMs launch `capacity` tiles, so the host never
/// reads the counts. experts <= 256.
void BuildRoutedTiles(const std::uint32_t* counts, std::uint32_t experts,
                      std::uint32_t rows_a, std::uint32_t capacity_a,
                      std::int32_t* tiles_a, std::uint32_t rows_b,
                      std::uint32_t capacity_b, std::int32_t* tiles_b,
                      hipStream_t stream);

/// The feed-forward residual of an expert layer, per row r:
///   f = rms(dense[r]) * norm1 + rms(sum_j weights[r][j] *
///       experts[r * used + j]) * norm2   (summed over j in order),
///   x[r] = (x[r] + rms(f) * post_norm) * scale,
///   h[r] = rms(x[r]) * next_norm (skipped when next_norm is null).
struct MoeFinishArgs {
  const float* dense;    ///< Dense MLP output, [rows][hidden]
  const float* experts;  ///< Expert outputs, [rows * used][hidden]
  const float* weights;  ///< [rows][used]
  const float* norm1;
  const float* norm2;
  const float* post_norm;
  float scale;
  float* x;
  const float* next_norm;
  float* h;
  std::uint32_t rows;
  std::uint32_t hidden;
  std::uint32_t used;
  float eps;
  /// Optional binary16 copy of h (round to nearest, as NarrowActivations);
  /// h may then be null.
  void* h_half;
};
void MoeFinish(const MoeFinishArgs& args, hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_MOE_HPP_
