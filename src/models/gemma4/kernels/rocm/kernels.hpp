#ifndef GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::gemma4::rocm {

/// x[r] *= scale; h[r] = rms(x[r]) * weight. Used for the scaled token
/// embedding feeding the first attention norm.
void ScaleRmsNorm(float* x, float scale, const float* weight, float* h,
                  std::uint32_t rows, std::uint32_t dim, float eps,
                  hipStream_t stream);

/// Plain row RMSNorm: y[r] = rms(x[r]) * weight (weight may be null).
void RmsNorm(const float* x, const float* weight, float* y, std::uint32_t rows,
             std::uint32_t dim, float eps, hipStream_t stream);

/// Per-head attention input post-processing for `rows` consecutive positions
/// starting at `first_position`:
///   Q = rope(rms(q) * q_norm) in place (FP32);
///   K = rope(rms(k) * k_norm), V = rms(v_source) written as binary16 into the
///   caches at slot (position % ring) — ring = 0 means a linear cache.
/// `v` may alias `k` (global layers use the raw K projection as V).
/// With `rotated_pairs` > 0 (V is the normalized K projection) the K cache
/// keeps only the rotated dims of each head, [0, pairs) then [dim / 2,
/// dim / 2 + pairs), and Q's other dims are multiplied by k_norm (see
/// AttentionArgs).
/// Pair i turns by position * theta_scale^i / freq_factors[i] with
/// theta_scale = theta^(-2/dim) (`freq_factors` null means 1), as ggml's
/// NEOX rope computes it.
/// The last `siblings` rows are sibling drafts beside the chain's drafts:
/// sibling i sits at position first_position + 1 + i and stores its K/V at
/// key spare_key + i.
struct QkvPostArgs {
  float* q;
  const float* k;
  const float* v;
  const float* q_norm;
  const float* k_norm;
  float theta_scale;
  const float* freq_factors;
  std::uint16_t* k_cache;  ///< binary16
  std::uint16_t* v_cache;  ///< binary16
  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
  std::uint32_t first_position;
  std::uint32_t ring;
  float eps;
  std::uint32_t rotated_pairs;
  /// Floats between consecutive rows of q, k and v when they are slices of
  /// one fused projection's rows; 0 means packed rows of their own widths.
  std::uint32_t row_stride;
  /// Packed destination of the processed Q ([rows][heads * head_dim]); null
  /// rewrites q in place.
  float* q_out;
  std::uint32_t siblings;
  std::uint32_t spare_key;
};
void QkvPost(const QkvPostArgs& args, hipStream_t stream);

/// Query-only post-processing (MTP draft layers): Q = rope(rms(q) * q_norm)
/// at one shared position for every row; with `rotated_pairs` > 0 the
/// unrotated dims are multiplied by `key_weight` as QkvPost does.
void QueryPost(float* q, const float* q_norm, float theta_scale,
               const float* freq_factors, std::uint32_t rows,
               std::uint32_t heads, std::uint32_t head_dim,
               std::uint32_t position, bool shared_position, float eps,
               const float* key_weight, std::uint32_t rotated_pairs,
               hipStream_t stream);

/// Scale-1 attention of FP32 queries over binary16 K/V caches.
/// Row r queries position `first_position + r` (or `first_position` for all
/// rows when `shared_position`), attends keys [lo, hi) with
/// hi = min(position + 1, key_limit) and lo = hi-window bound
/// (position + 1 - window, clamped at 0) when `window` > 0. Keys live at slot
/// (key % ring) when ring > 0. Head h reads KV head h / (heads / kv_heads).
/// Up to kSplitRows rows split keys into fixed chunks aligned to absolute
/// key positions and merge them in a fixed order, so a row's result does not
/// depend on the batch it runs in; `partials` must hold
/// AttentionPartialFloats(...) floats. Larger batches (prefill) attend in a
/// single pass per (row, head).
///
/// With `rope_pairs` > 0 (head_dim 512 only) V is the layer's normalized K
/// projection and K = rope(k_norm * V): the K cache holds just the rotated
/// dims, [rope_pairs) and [head_dim / 2, + rope_pairs) of each head
/// (rope_pairs * 2 values), and every unrotated key dim is its value, the
/// queries carrying k_norm on those dims (QkvPost, QueryPost).
///
/// `key_ends` (device, per row; null means causal) raises a row's upper key
/// bound to max(position + 1, key_ends[row]): image rows attend to every key
/// of their image in sliding layers. Those bounds never decrease with the
/// row and never pass the last row's position + 1.
///
/// The last `siblings` rows (QkvPost's) sit at first_position + 1 + i and
/// attend what the chain's row there attends, except that their own key
/// comes from spare_key + i. Split attention adds every row's own (last) key
/// in one step after the others, so a sibling computes what its row alone
/// would.
struct AttentionArgs {
  const float* q;
  const std::uint16_t* k_cache;  ///< binary16; rotated dims when derived
  const std::uint16_t* v_cache;  ///< binary16
  float* out;
  float* partials;
  std::uint32_t rows;
  std::uint32_t heads;
  std::uint32_t kv_heads;
  std::uint32_t head_dim;
  std::uint32_t first_position;
  bool shared_position;
  std::uint32_t key_limit;
  std::uint32_t window;
  std::uint32_t ring;
  std::uint32_t rope_pairs;  ///< rotated pairs of derived keys, or 0
  const std::uint32_t* key_ends;
  /// Optional binary16 copy of `out` (the WMMA prefill path writes it; there
  /// `out` may be null).
  void* out_half;
  std::uint32_t siblings;
  std::uint32_t spare_key;
};
inline constexpr std::uint32_t kSplitRows = 16;
/// Derived keys: rope_pairs must be a multiple of 16 and at most this.
inline constexpr std::uint32_t kMaxDerivedKeyPairs = 64;

void Attention(const AttentionArgs& args, hipStream_t stream);
[[nodiscard]] std::size_t AttentionPartialFloats(std::uint32_t rows,
                                                 std::uint32_t heads,
                                                 std::uint32_t head_dim,
                                                 std::uint32_t max_keys);

/// x[r] += rms(o[r]) * post_norm; h[r] = rms(x[r]) * next_norm.
/// With `q8`, h is written instead as the tiled Q8_1 prefill activation of
/// hip::LaunchQuantizeActivationQ8_1FromFp32 (same layout and rounding; a
/// block scale may differ by one ulp). With `h2`, also
/// h2[r] = rms(x[r]) * second_norm (the expert input of a MoE layer).
/// `h_half` / `h2_half` receive binary16 copies of h / h2 (round to
/// nearest, as NarrowActivations) for the binary16 prefill GEMMs; a null
/// h / h2 then skips the FP32 row.
void PostAttentionNorm(const float* o, const float* post_norm, float* x,
                       const float* next_norm, float* h, std::uint32_t rows,
                       std::uint32_t dim, float eps, hipStream_t stream,
                       void* q8 = nullptr, const float* second_norm = nullptr,
                       float* h2 = nullptr, void* h_half = nullptr,
                       void* h2_half = nullptr);

/// x[r] = (x[r] + rms(f[r]) * post_norm) * scale;
/// h[r] = rms(x[r]) * next_norm (next_norm null skips h); with `q8`, h is
/// written as the Q8_1 prefill activation instead, and `h_half` receives a
/// binary16 copy (a null h then skips the FP32 row; see PostAttentionNorm).
void PostFeedForwardNorm(const float* f, const float* post_norm, float scale,
                         float* x, const float* next_norm, float* h,
                         std::uint32_t rows, std::uint32_t dim, float eps,
                         hipStream_t stream, void* q8 = nullptr,
                         void* h_half = nullptr);

/// out = gelu_tanh(gate) * up, elementwise over `count` values.
void GeGlu(const float* gate, const float* up, float* out, std::size_t count,
           hipStream_t stream);
/// GeGlu over [rows][cols] written as the Q8_1 prefill activation, as GeGlu
/// followed by hip::LaunchQuantizeActivationQ8_1FromFp32 (see
/// PostAttentionNorm).
void GeGluQuantize(const float* gate, const float* up, void* q8,
                   std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// out[s][i] = gelu_tanh(gu[s][i]) * gu[s][width + i] over [slots][2 * width]
/// fused gate/up rows, rounding like GeGlu; `out_half` receives a binary16
/// copy (round to nearest, as NarrowActivations), and a null `out` then
/// skips the FP32 rows.
void GeGluPacked(const float* gu, float* out, std::uint32_t slots,
                 std::uint32_t width, hipStream_t stream,
                 void* out_half = nullptr);
/// GeGluPacked over binary16 rows (the prefill expert route); results
/// saturate at the binary16 range.
void GeGluPackedHalf(const void* gu, void* out, std::uint32_t slots,
                     std::uint32_t width, hipStream_t stream);

/// y = x * scale, elementwise (y may alias x).
void Scale(const float* x, float scale, float* y, std::size_t count,
           hipStream_t stream);

/// logits = cap * tanh(logits / cap), elementwise.
void Softcap(float* logits, std::size_t count, float cap, hipStream_t stream);

/// Copies rows `src[index[i]]` into dst[i] (row width `dim` floats).
void GatherRows(const float* src, const std::uint32_t* index, float* dst,
                std::uint32_t rows, std::uint32_t dim, hipStream_t stream);

/// One session's K/V caches, every layer: copies the binary16 row of key
/// `from` over that of key `to` (keys at slot key % ring[l] where ring[l] >
/// 0), widths in values per row.
struct MoveKeyArgs {
  static constexpr std::uint32_t kMaxLayers = 64;
  std::uint16_t* k[kMaxLayers];
  std::uint16_t* v[kMaxLayers];
  std::uint32_t k_width[kMaxLayers];
  std::uint32_t v_width[kMaxLayers];
  std::uint32_t ring[kMaxLayers];
  std::uint32_t layers;
  std::uint32_t from;
  std::uint32_t to;
};
void MoveKey(const MoveKeyArgs& args, hipStream_t stream);

/// One copy of CopyRuns: `bytes` (a multiple of 16) between 16-byte aligned
/// addresses the device can access (device or registered host memory).
struct CopyRun {
  const void* from;
  void* to;
  std::uint64_t bytes;
};
/// Performs the `count` copies of the array `runs` (device-accessible), none
/// longer than `longest` bytes, in one launch.
void CopyRuns(const CopyRun* runs, std::uint32_t count, std::uint64_t longest,
              hipStream_t stream);

}  // namespace gufo::models::gemma4::rocm

#endif  // GUFO_MODELS_GEMMA4_KERNELS_ROCM_KERNELS_HPP_
