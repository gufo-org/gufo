// The executor's side of stages.hpp: Layer / LayerEdit accessors over
// DeviceLayer, and DefaultStages, the GGUF implementation of every stage.

#include "src/models/qwen38_flash_next/stages.hpp"

#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::ext::qwen::flash_next {
namespace {

using models::qwen38_flash_next::rocm::DeviceLayer;
using models::qwen38_flash_next::rocm::DeviceTensor;
using models::qwen38_flash_next::rocm::FromTensor;
using models::qwen38_flash_next::rocm::ToTensor;

template<typename L>
auto& Field(L& l, TensorId id) noexcept {
  switch (id) {
    case TensorId::kSsmIn:
      return l.ssm_in;
    case TensorId::kSsmQkv:
      return l.ssm_qkv;
    case TensorId::kSsmGate:
      return l.ssm_gate;
    case TensorId::kSsmAlphaBeta:
      return l.ssm_alpha_beta;
    case TensorId::kSsmOut:
      return l.ssm_out;
    case TensorId::kAttnQkv:
      return l.attn_qkv;
    case TensorId::kAttnQ:
      return l.attn_q;
    case TensorId::kAttnK:
      return l.attn_k;
    case TensorId::kAttnV:
      return l.attn_v;
    case TensorId::kAttnOut:
      return l.attn_out;
    case TensorId::kIndexerQ:
      return l.indexer_q;
    case TensorId::kIndexerK:
      return l.indexer_k;
    case TensorId::kRouter:
      return l.router;
    case TensorId::kFfnGateExps:
      return l.ffn_gate_exps;
    case TensorId::kFfnUpExps:
      return l.ffn_up_exps;
    case TensorId::kFfnDownExps:
      return l.ffn_down_exps;
    case TensorId::kShexpGate:
      return l.shexp_gate;
    case TensorId::kShexpUp:
      return l.shexp_up;
    case TensorId::kShexpDown:
      return l.shexp_down;
    case TensorId::kNextnFcEmbedding:
      return l.nextn_fc_embedding;
    case TensorId::kNextnFcHidden:
      break;
  }
  return l.nextn_fc_hidden;
}

}  // namespace

bool Layer::linear() const noexcept {
  return LayerAccess::Device(*this).linear;
}

Tensor Layer::tensor(TensorId id) const noexcept {
  return ToTensor(Field(LayerAccess::Device(*this), id));
}

Layer LayerEdit::layer() const noexcept {
  return LayerAccess::Make(&LayerAccess::Device(*this), index_);
}

void LayerEdit::Set(TensorId id, const Tensor& tensor) noexcept {
  Field(LayerAccess::Device(*this), id) = FromTensor(tensor);
}

bool DefaultStages::Linear(const StageContext& ctx, const Tensor& w,
                           const LinearInput& in, float* out) const {
  const DeviceTensor d = FromTensor(w);
  if (d.format_state != nullptr) {
    if (ctx.error != nullptr)
      *ctx.error = "no stage implements this tensor format";
    return false;
  }
  if (in.ids == nullptr && in.slots != 1) {
    if (ctx.error != nullptr)
      *ctx.error = "a dense projection takes one slot per input row";
    return false;
  }
  const auto& e = *impl_->executor;
  if (in.ids == nullptr && in.slots == 1)
    return e.Dense(d, in.x, out, in.rows, ctx.error);
  return e.Experts(d, in.x, in.ids, out, in.rows, in.slots, ctx.n_tokens,
                   ctx.error);
}

bool DefaultStages::Attention(const StageContext& ctx, const AttentionArgs& a,
                              const float* x, float* out) const {
  return impl_->executor->DefaultAttention(
      LayerAccess::Device(ctx.layer), a.state, x, out, a.n_tokens, a.pos,
      a.first_block, a.start_pos, a.pool_grid, a.max_context, a.sparse,
      ctx.error, a.last_only, a.projections_ready, a.project_output,
      a.checkpoints);
}

bool DefaultStages::LinearAttention(const StageContext& ctx,
                                    const LinearAttentionArgs& a,
                                    const float* x, float* out) const {
  return impl_->executor->DefaultLinearAttention(
      LayerAccess::Device(ctx.layer), a.state, x, out, a.n_tokens,
      a.speculative, ctx.error, a.projections_ready, a.project_output,
      a.checkpoint);
}

bool DefaultStages::MoeDefersEpilogue(const StageContext& ctx) const {
  return impl_->executor->WmmaExperts(LayerAccess::Device(ctx.layer),
                                      ctx.n_tokens);
}

bool DefaultStages::MoeFfn(const StageContext& ctx, const MoeRouting& routing,
                           const float* x, float* out) const {
  return impl_->executor->DefaultMoeFfn(LayerAccess::Device(ctx.layer), routing,
                                        x, out, ctx.n_tokens, ctx.error);
}

bool DefaultStages::MtpHead(const StageContext& ctx,
                            const MtpHeadArgs& a) const {
  return impl_->executor->DefaultMtpHead(a.head, a.res, a.token, a.candidates,
                                         a.draft, ctx.error);
}

bool DefaultStages::SharedExpert(const StageContext& ctx,
                                 const MoeRouting& routing, const float* x,
                                 bool side) const {
  (void)routing;
  const auto& e = *impl_->executor;
  const DeviceLayer& l = LayerAccess::Device(ctx.layer);
  if (!side)
    return e.SharedExpertRows(l, x, ctx.n_tokens, ctx.error);
  const hipStream_t main = e.stream_;
  if (hipEventRecord(e.side_start_, main) != hipSuccess ||
      hipStreamWaitEvent(e.side_stream_, e.side_start_, 0) != hipSuccess) {
    if (ctx.error != nullptr)
      *ctx.error = "side stream fork failed";
    return false;
  }
  if (e.capturing_) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(e.side_stream_, &status) != hipSuccess ||
        status != hipStreamCaptureStatusActive) {
      if (ctx.error != nullptr)
        *ctx.error = "shared-expert overlap did not join the graph capture";
      return false;
    }
  }
  // Restores the launch stream and publishes the queued side work even when
  // the shared expert fails or throws, so the next JoinSide (the Moe
  // wrapper's guard at the latest) orders the main stream after it.
  struct SideScope {
    const models::qwen38_flash_next::rocm::Executor& e;
    hipStream_t main;
    bool finished{false};
    hipError_t Finish() noexcept {
      e.stream_ = main;
      if (finished)
        return hipSuccess;
      finished = true;
      const hipError_t status = hipEventRecord(e.side_done_, e.side_stream_);
      if (status == hipSuccess)
        e.side_pending_ = true;
      else if (!e.capturing_)
        (void)hipStreamSynchronize(e.side_stream_);
      return status;
    }
    ~SideScope() { (void)Finish(); }
  } scope{e, main};
  e.stream_ = e.side_stream_;
  const bool ok = e.SharedExpertRows(l, x, ctx.n_tokens, ctx.error);
  const hipError_t joined = scope.Finish();
  if (!ok)
    return false;  // keeps the shared expert's own message
  if (joined != hipSuccess) {
    if (ctx.error != nullptr)
      *ctx.error = "side stream join failed";
    return false;
  }
  return true;
}

bool DefaultStages::JoinSide(const StageContext& ctx) const {
  return impl_->executor->JoinSide(ctx.error);
}

bool DefaultStages::WaitCounts(const StageContext& ctx) const {
  return impl_->executor->WaitCounts(ctx.n_tokens, ctx.error);
}

bool DefaultStages::WaitRoutes(const StageContext& ctx) const {
  return impl_->executor->RouteHints(ctx.n_tokens, ctx.error);
}

void DefaultStages::CompactRoutes(const StageContext& ctx,
                                  const MoeRouting& routing) const {
  const auto& s = routing.scratch;
  models::qwen38_flash_next::rocm::RoutedCompact(
      routing.ids, s.expert_counts, s.routed_bounds, s.routed_cursors,
      s.rows_token, s.rows_slot, ctx.n_tokens, routing.used,
      routing.num_experts, impl_->executor->stream_);
}

const void* DefaultStages::HalfRows(const StageContext& ctx, const float* x,
                                    std::uint32_t cols) const {
  const auto& e = *impl_->executor;
  e.PrepareHalfInput(x, ctx.n_tokens, cols);
  return e.s_.x_half;
}

bool DefaultStages::RoutedExperts(const StageContext& ctx,
                                  const MoeRouting& routing, const float* x,
                                  float* out) const {
  (void)routing;
  return impl_->executor->MoeExperts(LayerAccess::Device(ctx.layer), x, out,
                                     ctx.n_tokens, ctx.error);
}

void DefaultStages::MoeEpilogue(const StageContext& ctx,
                                const MoeRouting& routing, float* out,
                                bool half_down) const {
  (void)routing;
  impl_->executor->MoeEpilogueRows(out, ctx.n_tokens, half_down);
}

}  // namespace gufo::ext::qwen::flash_next
