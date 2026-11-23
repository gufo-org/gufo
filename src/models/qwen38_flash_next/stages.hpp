#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_STAGES_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_STAGES_HPP_

// The per-layer work of a Flash-Next pass as overridable stages. The
// executor owns the pipeline (passes, KV, recurrent state, rollback,
// snapshots, sampling) and calls one stage per layer step; each default is
// the GGUF implementation, callable from an override through
// StageContext::defaults. Experimental: no compatibility promise yet.
//
// The header includes <hip/hip_runtime.h> (hipStream_t): Flash-Next runs on
// ROCm only.
//
// Rules the executor relies on:
// - A stage queues device work on StageContext::stream and returns; it never
//   synchronizes the stream or allocates device memory (decode passes are
//   recorded into HIP graphs and replayed without host code). Device memory
//   comes from ScratchBytes(), allocated once. The one host wait is
//   DefaultStages::WaitCounts / WaitRoutes, which wait for the expert row
//   counts only on passes wide enough to download them; those passes are
//   never captured.
// - DefaultStages queue on the executor's current stream, which is
//   StageContext::stream when the stage is called. A stage that queues on
//   a stream of its own must not call the defaults from there.
// - Stage results are deterministic for a pass shape: a captured graph
//   replays the launches of its recorded pass.
// - Linear and LinearGroup are called only for tensors that carry a
//   format_state() (see Tensor), and for groups with such a tensor; GGUF
//   projections keep their inline routes and add no host call.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "primitives.hpp"

namespace gufo::ext::qwen::flash_next {

class DefaultStages;
struct DefaultStagesImpl;
// Executor-owned arguments of a stage's default implementation; opaque to
// extensions, which pass them on to StageContext::defaults.
struct AttentionArgs;
struct LinearAttentionArgs;
struct MtpHeadArgs;

/// What every stage call receives.
struct StageContext {
  hipStream_t
      stream;  ///< The stream to queue on (the side stream in overlaps).
  Phase phase;
  /// Token rows of the pass (or pass segment) the call belongs to: route
  /// choices that must not depend on how a prompt is chunked read this.
  std::uint32_t n_tokens;
  /// The layer being run (Attention, LinearAttention, MoeDefersEpilogue,
  /// MoeFfn, PromptLayer). Always invalid for Linear and LinearGroup, which
  /// also serve model-level projections (output head, draft block), and for
  /// MtpHead.
  Layer layer;
  const DefaultStages& defaults;
  void* scratch;  ///< The ScratchBytes() region; null when none was asked.
  std::string* error;
};

/// The activation rows a linear reads. Output row r (of rows * slots) reads
/// input row r / slots, through expert ids[r] when ids is set.
struct LinearInput {
  const float* x{nullptr};  ///< [rows][w.cols()] F32
  std::uint32_t rows{0};
  std::uint32_t slots{1};
  const std::int32_t* ids{nullptr};  ///< [rows * slots]; null for dense weights
};

/// One projection of a group sharing a LinearInput.
struct LinearJob {
  Tensor w;
  float* out;  ///< [rows * slots][w.rows()] F32
};

/// Scratch the MoE FFN writes, sized for the pass: rows are
/// n_tokens * used slots (routed) or n_tokens (shared).
struct MoeScratch {
  float* gate;       ///< routed gate rows; SwiGLU result of the default
  float* up;         ///< routed up rows (F16 on the routed WMMA route)
  float* down;       ///< routed down rows (F16 on the routed WMMA route)
  float* shared_up;  ///< shared-expert first projection
  float* shared_gate;
  float* shared_out;  ///< shared-expert output rows, read by the epilogue
  std::uint32_t* expert_counts;  ///< [num_experts] device row counts
  /// Routed rows compacted by expert (CompactRoutes): 16-row padded bucket
  /// bounds, scatter cursors and the compact row -> token / slot maps.
  std::int32_t* routed_bounds;
  std::int32_t* routed_cursors;
  std::int32_t* rows_token;
  std::int32_t* rows_slot;
  std::int32_t* routed_tiles;  ///< device tile map ((expert | tile << 16))
};

/// The routing a MoE FFN runs after: router logits, top-k ids and weights.
struct MoeRouting {
  const std::int32_t* ids;  ///< [n_tokens][used]
  const float* weights;     ///< [n_tokens][used]
  /// [n_tokens][router_stride] logits; column num_experts gates the shared
  /// expert.
  const float* router;
  std::uint32_t router_stride;
  std::uint32_t num_experts;
  std::uint32_t used;
  /// Pinned [num_experts] rows per expert: valid after WaitCounts() (or
  /// WaitRoutes()) on passes wide enough to download them.
  const std::uint32_t* host_counts;
  /// Pinned staging for a stage's own tile map upload (host_tiles_capacity
  /// entries). WaitRoutes and RoutedExperts upload the default maps from
  /// it, so a stage reusing it calls WaitCounts instead.
  std::int32_t* host_tiles;
  std::size_t host_tiles_capacity;
  bool last_only;  ///< catch-up: only the last row's output is consumed
  bool overlap;    ///< the shared expert may run on the side stream
  MoeScratch scratch;
};

/// The executor's sizes, for ScratchBytes and Attach.
struct Limits {
  std::uint32_t max_batch;     ///< widest pass
  std::uint32_t narrow_batch;  ///< rows every pass may take
  std::uint32_t max_context;
  std::span<const Layer> layers;  ///< trunk layers in order
  Tensor output;                  ///< output head
  /// The MTP draft block (index() == Layer::kMtp); invalid without one.
  Layer mtp;
};

struct Attachment {
  void* scratch;  ///< ScratchBytes() bytes of device memory, or null
  std::size_t scratch_bytes;
  hipStream_t stream;  ///< the executor's main stream
  Limits limits;       ///< its spans stay valid while the executor lives
};

/// Optional per-layer calls a Stages implementation asks for (Hooks()).
enum StageHook : std::uint32_t {
  kNoStageHooks = 0,
  /// PromptLayer() runs before every trunk layer of a prompt pass.
  kPromptLayers = 1U << 0,
};

/// The GGUF implementations, and building blocks for overrides of MoeFfn.
/// Methods report failures through StageContext::error.
class DefaultStages {
public:
  explicit DefaultStages(const DefaultStagesImpl* impl) noexcept
      : impl_(impl) {}

  /// A GGUF projection: dense when in.ids is null (in.slots must be 1),
  /// routed experts otherwise. Fails on a tensor with format_state.
  bool Linear(const StageContext& ctx, const Tensor& w, const LinearInput& in,
              float* out) const;
  bool Attention(const StageContext& ctx, const AttentionArgs& args,
                 const float* x, float* out) const;
  bool LinearAttention(const StageContext& ctx, const LinearAttentionArgs& args,
                       const float* x, float* out) const;
  [[nodiscard]] bool MoeDefersEpilogue(const StageContext& ctx) const;
  bool MoeFfn(const StageContext& ctx, const MoeRouting& routing,
              const float* x, float* out) const;
  bool MtpHead(const StageContext& ctx, const MtpHeadArgs& args) const;

  // MoeFfn building blocks, in the order the default runs them.
  /// Queues the shared expert (into MoeScratch::shared_out), on the side
  /// stream when `side`; JoinSide orders the main stream after it.
  bool SharedExpert(const StageContext& ctx, const MoeRouting& routing,
                    const float* x, bool side) const;
  bool JoinSide(const StageContext& ctx) const;
  /// Waits for the expert row counts (host_counts) on wide passes; queues
  /// nothing.
  bool WaitCounts(const StageContext& ctx) const;
  /// WaitCounts, then the default routed tile maps uploaded through
  /// host_tiles.
  bool WaitRoutes(const StageContext& ctx) const;
  /// Compacts the routed rows by expert into MoeScratch's bucket maps.
  void CompactRoutes(const StageContext& ctx, const MoeRouting& routing) const;
  /// F16 copy of `x` ([n_tokens][cols]), reused while it holds these rows.
  [[nodiscard]] const void* HalfRows(const StageContext& ctx, const float* x,
                                     std::uint32_t cols) const;
  /// The routed experts (WaitRoutes first) and the epilogue, once the
  /// shared expert is queued.
  bool RoutedExperts(const StageContext& ctx, const MoeRouting& routing,
                     const float* x, float* out) const;
  /// out = sum of the routed down rows by weight + gated shared output. With
  /// `half_down` (F16 down rows) and `out` the layer's block output, the
  /// following combine applies it instead (see MoeDefersEpilogue).
  void MoeEpilogue(const StageContext& ctx, const MoeRouting& routing,
                   float* out, bool half_down) const;

private:
  const DefaultStagesImpl* impl_;
};

/// Overridable per-layer work. Every method defaults to DefaultStages.
class Stages {
public:
  virtual ~Stages() = default;

  /// StageHook bits; read once when the executor is created.
  [[nodiscard]] virtual std::uint32_t Hooks() const { return kNoStageHooks; }
  /// Device scratch the stages need, allocated once with the executor.
  [[nodiscard]] virtual std::size_t ScratchBytes(const Limits& limits) const {
    (void)limits;
    return 0;
  }
  /// Called once after the scratch is allocated, before any pass.
  virtual bool Attach(const Attachment& attachment, std::string* error) {
    (void)attachment;
    (void)error;
    return true;
  }

  /// One projection of a tensor with format_state().
  virtual bool Linear(const StageContext& ctx, const Tensor& w,
                      const LinearInput& in, float* out) {
    return ctx.defaults.Linear(ctx, w, in, out);
  }
  /// Projections of one input (GDN qkv + z, attention q + k + v, routed
  /// gate + up); the default runs one Linear per job. The routed pair's
  /// SwiGLU follows in the executor.
  virtual bool LinearGroup(const StageContext& ctx,
                           std::span<const LinearJob> jobs,
                           const LinearInput& in) {
    for (const LinearJob& job : jobs) {
      if (!Linear(ctx, job.w, in, job.out))
        return false;
    }
    return true;
  }
  virtual bool Attention(const StageContext& ctx, const AttentionArgs& args,
                         const float* x, float* out) {
    return ctx.defaults.Attention(ctx, args, x, out);
  }
  /// Gated DeltaNet.
  virtual bool LinearAttention(const StageContext& ctx,
                               const LinearAttentionArgs& args, const float* x,
                               float* out) {
    return ctx.defaults.LinearAttention(ctx, args, x, out);
  }
  /// Asked before the attention-side combine of a layer: whether its MoeFfn
  /// leaves the epilogue to the FFN-side combine (MoeEpilogue with
  /// half_down), so that combine may defer its residual update.
  [[nodiscard]] virtual bool MoeDefersEpilogue(const StageContext& ctx) {
    return ctx.defaults.MoeDefersEpilogue(ctx);
  }
  /// Shared and routed experts after routing, through the epilogue.
  virtual bool MoeFfn(const StageContext& ctx, const MoeRouting& routing,
                      const float* x, float* out) {
    return ctx.defaults.MoeFfn(ctx, routing, x, out);
  }
  /// The draft block's head: mixer, logits and the draft token/candidates.
  virtual bool MtpHead(const StageContext& ctx, const MtpHeadArgs& args) {
    return ctx.defaults.MtpHead(ctx, args);
  }
  /// kPromptLayers: before trunk layer ctx.layer of a prompt pass of
  /// ctx.n_tokens rows. Setting `*use` runs the pass on `edit` (kept across
  /// passes) instead.
  virtual bool PromptLayer(const StageContext& ctx, LayerEdit& edit,
                           bool* use) {
    (void)ctx;
    (void)edit;
    *use = false;
    return true;
  }
};

}  // namespace gufo::ext::qwen::flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_STAGES_HPP_
