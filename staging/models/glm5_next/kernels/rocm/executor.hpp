#ifndef GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "staging/models/glm5_next/kernels/rocm/device_model.hpp"

namespace gufo::models::glm5_next::rocm {

/// Per-run cache state: the KDA convolution rings and recurrent states, the
/// latent-only MLA cache and the indexer pool cache, and one reuse slot for
/// the selection scratch sized to the session capacity. Positions run
/// contiguously from 0 (single-sequence decode, the parity contract); a new
/// sequence uses a fresh session.
class Session {
  friend class Executor;

 public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Number of tokens already processed (the position of the next token).
  [[nodiscard]] std::int64_t position() const noexcept { return pos_; }
  /// Put the session back to its empty state (next token at 0) reusing the
  /// allocated caches: the KDA conv rings and recurrent states are zeroed;
  /// latent/pool rows are only read below the position, so they need no
  /// clear. Used by serve reuse; a fresh sequence never carries state
  /// across sessions otherwise.
  void Reset();

  [[nodiscard]] std::uint32_t capacity() const noexcept { return cap_; }

 private:
  Session() = default;

  std::int64_t pos_{-1};
  std::uint32_t cap_{0};
  // Write cursor for the draft block's caches: the rows below it were
  // written by the draft chain / prefill for the committed prefix, so it
  // matches pos_ between draft rounds (RollbackVerify rewinds both; rows at
  // or above the cursor are never read before being overwritten).
  std::int64_t mtp_pos_{0};
  std::vector<void*> allocations_;
  // One entry per DSA / KDA layer, in `layers()` order filtered by kind.
  // With a nextn block its own latent/pool caches are appended last, so
  // the draft layer runs through DsaLayer at slot = DSA layer count.
  std::vector<__half*> latent_cache;  // [cap][512] binary16 latent rows
  std::vector<float*> pool_cache;     // [cap][256] F32 [ik | gate] rows
  std::vector<float*> conv_hist;      // [3 * 3 * inner] conv ring per KDA layer
  std::vector<float*> kda_state;      // [heads][d][d] state per KDA layer
  std::size_t conv_bytes{0};          // sizeof an entry of conv_hist
  std::size_t state_bytes{0};         // sizeof an entry of kda_state
  // Selection scratch sized by the capacity. The batched prefill regions
  // hold one per-token slice each (executor chunk kChunk) with these
  // strides; the single-token path reuses slice 0.
  float* pool_scores{nullptr};
  std::size_t pool_scores_stride{0};
  std::int32_t* pool_sel{nullptr};
  std::size_t pool_sel_stride{0};
  std::int32_t* sel{nullptr};
  std::size_t sel_stride{0};
  std::uint64_t* topk_scratch{nullptr};
};

/// One image's embedding rows staged on the device and the prompt-token span
/// they replace. `rows` points at `row_count * hidden_size` f32 device floats
/// produced by the vision encoder; the offsets are absolute indices into the
/// prompt passed to the next `Prefill`.
struct VisionImage {
  const float* rows{nullptr};
  std::uint32_t offset{0};
  std::uint32_t row_count{0};
};

/// Per-token decode executor of the resident device model. `Step` embeds the
/// token, walks the 45 trunk blocks (mHC, KDA or DSA attention, dense or MoE
/// FFN) and leaves the logits of the sampled position in `logits()`; the
/// caller reads them when it needs them. All launches are ordered on the
/// default stream.
///
/// `Prefill` runs a prompt as chunks of `kChunk` tokens: every weight GEMV
/// (projections, absorbed-MLA head stacks, dense/shared FFN, routed experts)
/// reads its weights once for the whole chunk through the multi-row and
/// grouped tiers, while the parts that are inherently sequential per token
/// (KDA conv rings and recurrent states, mHC coefficients, indexer pool
/// selection, gathered attention) run token-by-token inside the chunk. Each
/// per-token dot product is bit-identical to the same computation under
/// `Step`, so chunked prefill leaves exactly the same state and logits as
/// stepping the tokens one by one.
///
/// Vision prompts: `SetVision` stores the staged image embedding rows, the
/// next `Prefill` substitutes them for the token embeddings inside each
/// span and clears the list; `Step` never sees them.
class Executor {
 public:
  /// Validates that every stored tensor type is decodable by the GEMV tier
  /// and allocates the shared activation scratch. Returns nullptr and fills
  /// `error_msg` on mismatch.
  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, std::string* error_msg = nullptr);

  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  /// Device bytes one session needs for `capacity` tokens (caches plus
  /// selection scratch).
  [[nodiscard]] std::size_t SessionBytes(std::uint32_t capacity) const;

  /// Allocates and zeroes one session (cache state, selection scratch).
  /// Returns nullptr and fills `error_msg` on allocation failure.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t capacity, std::string* error_msg = nullptr) const;

/// Chunk width of `Prefill`. The multi-row tier walks any width as
/// power-of-two slices; the bucketed MoE tier covers the wide chunks.
  static constexpr std::uint32_t kChunk = 2048;

  /// Chunks of at least this many tokens take the bucketed routed MoE
  /// tier; narrower ones (and every `Step`) stay on the grouped tier.
  static constexpr std::uint32_t kBucketMinRows = 32;

  /// Runs one decode token at position `session.position() + 1` and returns
  /// the device pointer holding the `vocab_size` logits (valid until the
  /// next Step; the default stream orders the read).
  void Step(Session& session, std::int32_t token);

  /// Runs `count` prompt tokens at consecutive positions starting at
  /// `session.position() + 1` in chunks of `kChunk` (bit-identical to
  /// calling Step once per token; the logits after the call belong to the
  /// last token). A debug dump forces the one-token path so the anchors keep
  /// their per-step semantics.
  void Prefill(Session& session, const std::int32_t* tokens,
               std::size_t count);

  /// Arms image embedding rows for the next `Prefill`. Spans must be
  /// disjoint and inside the prompt; the rows replace the token embeddings
  /// at those absolute positions and the list is cleared when `Prefill`
  /// returns. An empty list is the text-only normal state.
  void SetVision(std::vector<VisionImage> images) {
    vision_ = std::move(images);
  }

  [[nodiscard]] const float* logits() const noexcept { return logits_; }
  [[nodiscard]] const DeviceModel& model() const noexcept { return *model_; }

  // ------------------------------------------------------------- MTP draft
  /// True when the artifact contains the nextn draft block.
  [[nodiscard]] bool HasMtp() const noexcept { return c_.nextn_layers > 0U; }

  /// Maximum trunk rows one `Verify` can consume (one accepted token plus
  /// at most `kMaxVerifyRows - 1` drafts).
  static constexpr std::uint32_t kMaxVerifyRows = 5;

  /// Runs the nextn draft block on `token` (whose trunk position is
  /// `session.position() + 1`) with the hidden state `h_in` (the normalized
  /// output of position `session.position()` after a trunk pass). Writes the
  /// draft cache row, updates the draft chain hidden (`MtpDraft` inputs) and
  /// leaves the draft logits in `mtp_logits()`. The first call of a round
  /// takes the trunk's own hidden row; with the caches aligned this also
  /// serves as `MtpAdvance`/`MtpDraft` given the right hidden pointer.
  void MtpStep(Session& session, std::int32_t token, const float* h_in);

  /// Chains the draft block: consumes `token` at the position right after
  /// the last draft row, conditioning on the previous draft's own normed
  /// output hidden (not a trunk row).
  void MtpDraft(Session& session, std::int32_t token);

  /// Advances the draft cache over the last verified draft: `token` at the
  /// position of the last verify row, with that row's trunk hidden (`row`
  /// of `verify_hidden()`). Only the cache row matters; its logits are
  /// superseded by the next round.
  void MtpAdvance(Session& session, std::int32_t token, const float* h_in);

  /// Runs `k + 1` trunk rows (the sampled token followed by `k` drafts) as
  /// one batched pass, keeping a per-row snapshot of the recurrent KDA
  /// state so a partial accept can rewind, and computes every row's logits
  /// (`verify_logits()`, row-major, k+1 rows). `session.position()` must be
  /// the position of `token`.
  void Verify(Session& session, std::int32_t token,
              const std::int32_t* drafts, std::uint32_t k);

  /// Rewinds the trunk position, the recurrent KDA state (from the row
  /// snapshots of the matching `Verify`) and the draft write cursor so that
  /// exactly `keep` (1..k of a matching `Verify`) rows stay consumed; rows
  /// at or above the rewound cursor are overwritten by the next draft round.
  void RollbackVerify(Session& session, std::uint32_t keep);

  /// When enabled, `Prefill` also runs the draft block over every chunk
  /// (positions `>= 1`, paired with the preceding trunk hidden row) so the
  /// draft caches are pre-filled at decode time. Must be set before the
  /// prefill that should fill them; no effect without a nextn block.
  void SetMtpPrefill(bool enabled) { mtp_prefill_ = enabled; }

  [[nodiscard]] const float* mtp_logits() const noexcept {
    return mtp_logits_;
  }
  /// [k+1][vocab] logits of the last `Verify`, row r = position
  /// `verify_p0 + r`; valid until the next executor call.
  [[nodiscard]] const float* verify_logits() const noexcept {
    return verify_logits_;
  }
  /// [k+1][hidden] normed trunk hidden rows of the last `Verify` (the
  /// `MtpAdvance` input takes its row k-1).
  [[nodiscard]] const float* verify_hidden() const noexcept { return h_; }
  /// The last trunk run's final-row hidden pointer (normalized output of
  /// `session.position()`); valid until the next executor call.
  [[nodiscard]] const float* last_hidden() const noexcept { return h_last_; }

  /// When non-empty, Step() downloads and writes every trunk anchor tensor
  /// (hc mixes, block norms, sub-layer outputs, stream write-backs, final
  /// norm, logits) as raw f32 files `<dir>/<anchor>.f32` for parity
  /// bisection against a reference-engine dump. Empty = disabled (default).
  void SetDebugDump(std::string dir) { dbg_dir_ = std::move(dir); }

 private:
  Executor() = default;

  void Dbg(const char* fmt, std::uint32_t il, const float* dev,
           std::size_t n) const;

  /// Runs `n` (1..kChunk) consecutive tokens; the shared layer helpers
  /// batch their weight GEMVs over the chunk and loop over the tokens for
  /// the sequential state. `Step` is the n == 1 case. `token_base` is the
  /// absolute index of `tokens[0]` in the prompt, used to match the armed
  /// vision spans (0 outside prefill).
  void RunTokens(Session& session, const std::int32_t* tokens,
                 std::uint32_t n, std::uint32_t token_base);

  /// Chunk-friendly GEMV dispatch (see the note above its definition).
  void GemvN(std::uint32_t tokens, const DeviceTensor& t, const float* x,
             std::uint32_t x_stride, float* out, std::uint32_t out_stride);

  void KdaLayer(const DeviceLayer& l, std::uint32_t slot, const float* x,
                std::uint32_t n, std::int64_t p0, Session& s, float* y);
  void DsaLayer(const DeviceLayer& l, std::uint32_t slot, const float* x,
                std::uint32_t n, std::int64_t p0, Session& s, float* y);
  void Ffn(const DeviceLayer& l, std::uint32_t il, const float* x,
           std::uint32_t n, float* y);

  /// One draft-block forward of `rows` consecutive positions `p0..`: row r
  /// consumes token `tok[r]` and the hidden row `h_in[r*H..]` (the trunk or
  /// previous-draft normed hidden of position `p0 + r - 1`).
  /// [enorm(embed) | hnorm(h_in)] -> eh_proj -> attn_norm -> DsaLayer draft
  /// cache rows -> residual -> ffn_norm -> MoE + shared FFN -> residual.
  /// `final_out` (single row) additionally norms the final residual with
  /// shared_head_norm into `mtp_chain_` and writes `mtp_logits_`; prefill
  /// rows skip both (only the caches are needed).
  void DraftForward(Session& session, const std::int32_t* tok,
                    const float* h_in, std::uint32_t rows, std::int64_t p0,
                    bool final_out);
  void MtpPrefillChunk(Session& session, const std::int32_t* tok,
                       std::uint32_t cnt, std::int64_t p0);

  const DeviceModel* model_{nullptr};
  Config c_{};
  std::string dbg_dir_;
  std::uint32_t dbg_il_{0};
  // Image embedding spans armed by `SetVision`, consumed and cleared by
  // the next `Prefill`.
  std::vector<VisionImage> vision_;

  // Shared scratch (default-stream order, reused by every layer). Buffers
  // marked NC hold `kChunk` rows behind the stride of the current use; the
  // n == 1 paths address row 0 and keep the exact Step() bit sequence.
  float* w_{nullptr};      // ones for the unweighted hc flat norm [hc*H]
  float* mean4_{nullptr};  // device constant [1/hc, 1/hc, 1/hc, 1/hc]

  // mHC ping-pong streams and one fused-block path (NC rows).
  float* s0_{nullptr};  // [NC][hc][H]
  float* s1_{nullptr};  // [NC][hc][H]
  float* x_{nullptr};   // [NC][H] block input after the sub-layer norm
  float* y0_{nullptr};  // [NC][H] sub-layer output before write-back
  std::uint32_t* overflow_{nullptr};  // ConvertF32ToHalf sentinel count

  // Per-sub-layer coefficient staging (NC rows, one set per token).
  float* mix_a_{nullptr};  // [NC][24] hc mix
  float* pre_a_{nullptr};  // [NC][hc]
  float* post_a_{nullptr};  // [NC][hc]
  float* comb_a_{nullptr};  // [NC][hc][hc]

  // Large per-layer activation buffers (NC rows unless noted).
  float* q_{nullptr};       // [NC][max(heads*mla_head_dim, inner)] reuse
  float* head_out_{nullptr};   // [NC][MLA v size]
  float* latent_{nullptr};     // [NC][hc*H] flat mHC norm buffer
  float* dsa_latent_{nullptr};  // [NC][MLA v size] gathered context rows
  float* q_abs_{nullptr};    // [NC][heads][kv_lora_rank]
  __half* q_abs_h_{
      nullptr};  // [NC][2][heads][kv_lora_rank] F16 hi/lo score scratch
  float* kv_{nullptr};       // [NC][latent]
  float* qr_{nullptr};       // [NC][q_lora]
  float* iq_{nullptr};       // [NC][indexer heads * key]
  float* iw_{nullptr};       // [NC][indexer heads]
  float* vq_{nullptr};       // [NC][inner] KDA raw q / conv q input
  float* vk_{nullptr};       // [NC][inner] KDA conv v
  float* eg_{nullptr};       // [NC][inner] raw v, then the decay gate
  float* sk_{nullptr};       // [NC][inner] raw k, then fold-sk
  float* ko_{nullptr};       // [NC][inner] conv q / normed / gated out
  float* g2_{nullptr};       // [NC][inner]
  float* g1_{nullptr};       // [NC][inner]
  float* beta_{nullptr};     // [NC][heads]
  float* moe_w_{nullptr};    // [NC][used] routing weights
  std::int32_t* head_ids_{nullptr};  // [NC][heads] 0..heads-1 per token
  float* tile_scores_{nullptr};      // per-token gather scratch (reused)
  float* moe_gate_{nullptr};  // [NC][used * expert_ff]
  float* moe_up_{nullptr};    // [NC][used * expert_ff]
  float* moe_inter_{nullptr};  // [NC][used * expert_ff]
  float* moe_down_{nullptr};   // [NC][used * H]
  float* f_gate_{nullptr};     // [NC][max(dense_ff, shared_ff)]
  float* f_up_{nullptr};       // [NC][max(dense_ff, shared_ff)]
  float* f_inter_{nullptr};    // [NC][max(dense_ff, shared_ff)]
  float* router_{nullptr};     // [NC][experts]
  std::int32_t* ids_{nullptr};  // [NC][used]
  // Bucket-compaction scratch for the wide-chunk routed tier (sized for
  // kChunk slots; the n >= kBucketMinRows path rebuilds it per layer).
  std::uint32_t* expert_counts_{nullptr};  // [experts]
  std::int32_t* pad_bounds_{nullptr};      // [experts + 1]
  std::int32_t* cursors_{nullptr};         // [experts]
  std::int32_t* rows_slot_{nullptr};  // [RoutedCompactRows(kChunk * used, e)]
  // 16-slot bucket tiles (`expert | tile << 16`); the tile list is shared
  // by all three GEMM heights, which vary only in rows/cols.
  std::int32_t* tiles_{nullptr};
  std::uint32_t tiles_cap_{0};
  // 32-row tiles for the WMMA tier (second tile height of the same
  // BuildRoutedTiles call), its F16 copy of the kChunk hidden-state rows and
  // the F16 copy of the routed intermediate for the WMMA down tier.
  std::int32_t* tiles_wide_{nullptr};
  std::uint32_t tiles_wide_cap_{0};
  __half* x_half_{nullptr};  // [kChunk][H]
  __half* moe_inter_half_{nullptr};  // [kChunk * used][expert_ff]
  float* h_{nullptr};           // [NC][H] collapsed stream
  float* logits_{nullptr};      // [vocab]
  const float* h_last_{nullptr};  // points at the last row of h_

  // Blocked W8A8 dense tier (default, resolved in `Create`): tiled Q8_1
  // activation scratch sized for kChunk rows of the widest dense tensor
  // (0 => the tier is off for tensors wider than the scratch). All launches
  // stay on the default stream like every other tier.
  bool w8a8_{false};
  void* w8a8_act_{nullptr};
  std::size_t w8a8_bytes_{0};

  // MTP draft-block state (only allocated when the artifact has a nextn
  // block; the draft caches themselves are the trailing session entries).
  static constexpr std::uint32_t kMtpRows = 256;  // prefill batch rows
  std::uint32_t mtp_slot_{0};    // draft entry index in the session caches
  std::uint32_t kda_layers_{0};  // KDA entry count (snapshot stride math)
  bool mtp_prefill_{false};
  bool kda_snap_{false};         // Verify runs KdaLayer per-row + snapshots
  std::int64_t verify_p0_{-1};   // first position of the last Verify
  float* mtp_cat_{nullptr};      // [kMtpRows][2H] [enorm(e) | hnorm(h)]
  float* mtp_x_{nullptr};        // [kMtpRows][H] eh_proj/residual base
  float* mtp_chain_{nullptr};    // [H] draft chain hidden (normed output)
  float* mtp_prev_{nullptr};     // [H] last h_ row of the previous chunk
  float* mtp_logits_{nullptr};   // [vocab] draft-block logits
  float* verify_logits_{nullptr};  // [kMaxVerifyRows][vocab]
  // Per-KDA-layer, per-row input snapshots taken by Verify's per-row KDA
  // path: index [kda slot][row] with row in 1..kMaxVerifyRows-1 (the state
  // entering that row = the state after accepting exactly `row` drafts).
  float* mtp_snap_conv_{nullptr};   // [kda_layers][kMaxVerifyRows][9*inner]
  float* mtp_snap_state_{nullptr};  // [kda_layers][kMaxVerifyRows][heads*d*d]
  std::size_t snap_conv_elems_{0};
  std::size_t snap_state_elems_{0};
};

}  // namespace gufo::models::glm5_next::rocm

#endif  // GUFO_MODELS_GLM5_NEXT_KERNELS_ROCM_EXECUTOR_HPP_
